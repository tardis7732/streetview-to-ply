"""Fixed-population refinement of an existing standard SH2 Gaussian PLY.

Run through the approved Linux run_python.sh. Installed gsplat Python and the
loaded native extension are pinned; historical optimization math is preserved.
Inputs/exports use EDN metres; camera calibration/poses, point order and Gaussian
count remain fixed. SH2 is active from step zero. No MCMC noise or relocation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import cv2
from PIL import Image
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parent
from .brush_refine_runtime import renderer_provenance, assert_loaded_renderer
from gsplat import MCMCStrategy, export_splats, rasterization
import gsplat
from .brush_refine_ply import load_gaussian_ply


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, data):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def log(data):
    print(json.dumps(data,allow_nan=False),flush=True)


def scalar(value):
    return float(value.detach()) if isinstance(value,torch.Tensor) else float(value)


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--renderer-library',type=Path,required=True)
    p.add_argument('--renderer-library-sha256',required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--init-gaussians-ply',type=Path,required=True)
    p.add_argument('--strategy',choices=['fixed'],default='fixed')
    p.add_argument('--regularization-scope',choices=['visible_sum','global'],default='visible_sum')
    p.add_argument('--render-only',action='store_true')
    p.add_argument('--lr-multiplier',type=float,default=.1)
    p.add_argument('--means-lr-multiplier',type=float)
    p.add_argument('--scale-lr-multiplier',type=float)
    p.add_argument('--opacity-lr-multiplier',type=float)
    p.add_argument('--sh-lr-multiplier',type=float)
    p.add_argument('--rotation-lr-multiplier',type=float)
    p.add_argument('--steps',type=int,default=6000)
    p.add_argument('--resolution',type=int,default=768)
    p.add_argument('--max-splats',type=int,default=2000000)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--opacity-reg',type=float,default=0.)
    p.add_argument('--scale-reg',type=float,default=0.)
    p.add_argument('--depth-weight',type=float,default=0.)
    p.add_argument('--depth-dir',type=Path)
    p.add_argument('--eval-every',type=int,default=6000)
    p.add_argument('--export-every',type=int,default=1000)
    p.add_argument('--resume',type=Path)
    p.add_argument('--depth-start',type=int)
    p.add_argument('--depth-ramp',type=int)
    p.add_argument('--depth-alpha-min',type=float,default=.05)
    p.add_argument('--depth-huber-beta',type=float,default=.1)
    p.add_argument('--sh-degree',type=int,default=2,choices=[2])
    p.add_argument('--cpu-workers',type=int,default=4)
    a=p.parse_args()
    a.dataset=a.dataset.resolve();a.output=a.output.resolve()
    a.init_gaussians_ply=a.init_gaussians_ply.resolve()
    if a.render_only:a.steps=0
    if a.depth_dir:a.depth_dir=a.depth_dir.resolve()
    if (a.steps<1 and not a.render_only) or a.resolution<16 or a.max_splats<4:raise ValueError('Invalid training budget')
    if any(value is not None and value<0 for key,value in vars(a).items() if key.endswith('lr_multiplier')):raise ValueError('LR multipliers must be nonnegative')
    if min(a.opacity_reg,a.scale_reg,a.depth_weight)<0:raise ValueError('Loss weights must be nonnegative')
    if a.depth_weight and not a.depth_dir:raise ValueError('Depth weight requires a validated depth directory')
    if a.output==a.dataset or a.dataset in a.output.parents:raise ValueError('Training output cannot modify the dataset')
    if a.run_root.resolve() not in a.output.parents:raise ValueError('Output must stay inside the declared isolated run root')
    return a


class FixedStrategy:
    def initialize_state(self):return {}
    def check_sanity(self,params,optimizers):
        if set(params)!=set(optimizers):raise ValueError('Optimizer parameter keys differ')
    def step_post_backward(self,*args,**kwargs):return None


def prepare_frame(frame, dataset, resolution, center, radius, with_pixels):
    frame=dict(frame)
    factor=min(1.,resolution/max(frame['w'],frame['h']))
    w,h=max(1,round(frame['w']*factor)),max(1,round(frame['h']*factor))
    sx,sy=w/frame['w'],h/frame['h']
    K=np.array([[frame['fl_x']*sx,0,frame['cx']*sx],[0,frame['fl_y']*sy,frame['cy']*sy],[0,0,1]],np.float32)
    c2w=np.array(frame['transform_matrix'],np.float64)
    Rcv=c2w[:3,:3]@np.diag([1.,-1.,-1.])
    camera_center=(c2w[:3,3]-center)/radius
    view=np.eye(4,dtype=np.float32)
    view[:3,:3]=Rcv.T;view[:3,3]=-Rcv.T@camera_center
    result=dict(meta=frame,w=w,h=h,K=K,view=view,name=Path(frame['file_path']).name)
    if with_pixels:
        with Image.open(dataset/frame['file_path']) as image:
            result['rgb']=np.asarray(image.convert('RGB').resize((w,h),Image.Resampling.LANCZOS)).copy()
        mask_path=dataset/frame.get('mask_path',f"masks/{Path(frame['file_path']).stem}.png")
        if not mask_path.is_file():raise FileNotFoundError(f'Required static mask absent: {mask_path}')
        with Image.open(mask_path) as image:
            result['mask']=np.asarray(image.convert('L').resize((w,h),Image.Resampling.NEAREST))==255
        if not result['mask'].any():raise ValueError(f'No supervised RGB pixels: {frame["file_path"]}')
        result['ssim_mask']=cv2.erode(result['mask'].astype(np.uint8),np.ones((11,11),np.uint8),
            borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
    return result


def load_depths(args, frames):
    if not args.depth_weight:return {},None,{}
    candidates=[args.depth_dir/n for n in ['depth_manifest.json','index.json','sparse_index.json','depth_index.json']]
    index=next((p for p in candidates if p.exists()),None)
    if index is None:raise FileNotFoundError('No recognized depth manifest')
    manifest=load_json(index)
    if 'status' in manifest and manifest['status']!='accepted':raise ValueError('Depth manifest has not been accepted')
    if manifest.get('depth_convention','camera_z')!='camera_z':raise ValueError('Depth must be camera-Z, not ray distance')
    entries=manifest.get('entries',manifest.get('images',[]))
    if isinstance(entries,dict):entries=[dict(value,image=key) for key,value in entries.items()]
    lookup={Path(e['image']).name:e for e in entries}
    dataset_manifest=load_json(args.dataset/'dataset_manifest.json')
    world_from_enu=np.array(dataset_manifest['world_from_enu'],np.float64)
    if world_from_enu.shape!=(4,4) or not np.allclose(world_from_enu[:3,:3].T@world_from_enu[:3,:3],np.eye(3),atol=1e-10):
        raise ValueError('Dataset ENU-to-viewer transformation is not rigid')
    result={};stats=dict(images=0,valid_pixels=0,sparse_pixels=0,dense_pixels=0,
        calibration_pose_verified_images=0,max_K_element_error=0.,max_w2c_element_error=0.)
    for frame in frames:
        entry=lookup.get(frame['name'])
        if entry is None:continue
        path=(index.parent/entry['npz']).resolve()
        with np.load(path,allow_pickle=False) as data:
            z=np.asarray(data['depth_z'],np.float32)
            valid=np.asarray(data['valid'],bool)
            confidence=np.asarray(data['confidence'],np.float32)
            support=np.asarray(data['source_count'])
            source=np.asarray(data['source_type']) if 'source_type' in data else np.where(valid,1,0).astype(np.uint8)
            if z.ndim!=2 or any(a.shape!=z.shape for a in [valid,confidence,support,source]):raise ValueError('Depth array shape mismatch')
            dh,dw=z.shape
            meta=frame['meta']
            required=['K','w','h','camera_from_world','unit','depth_convention','world_frame','pixel_center_offset']
            if any(key not in data for key in required):raise ValueError(f'Incomplete depth calibration: {path}')
            if str(data['unit'].item()) not in ['m','metre','metres','meter','meters']:raise ValueError('Depth units are not metres')
            if str(data['depth_convention'].item())!='camera_z':raise ValueError('NPZ depth is not camera-Z')
            if str(data['world_frame'].item())!='ENU':raise ValueError('Depth camera pose must declare the ENU frame')
            if not np.isclose(float(data['pixel_center_offset']),.5,atol=1e-7):raise ValueError('Depth pixel-center convention differs')
            if int(data['w'])!=dw or int(data['h'])!=dh:raise ValueError('Depth size metadata differs')
            expected_K=np.array([[meta['fl_x']/meta['w']*dw,0,meta['cx']/meta['w']*dw],
                [0,meta['fl_y']/meta['h']*dh,meta['cy']/meta['h']*dh],[0,0,1.]])
            k_error=float(np.max(np.abs(np.asarray(data['K'])-expected_K)))
            original_c2w=np.array(meta['transform_matrix'])@np.diag([1.,-1.,-1.,1.])
            expected_w2c=np.linalg.inv(original_c2w)[:3]
            enu_w2c=np.asarray(data['camera_from_world'])
            if enu_w2c.shape!=(3,4):raise ValueError('Invalid depth camera pose shape')
            actual_w2c=enu_w2c@np.linalg.inv(world_from_enu)
            pose_error=float(np.max(np.abs(actual_w2c-expected_w2c)))
            if k_error>1e-5 or pose_error>1e-6:raise ValueError(f'Depth camera/calibration mismatch: {frame["name"]}, K={k_error}, w2c={pose_error}')
            stats['calibration_pose_verified_images']+=1
            stats['max_K_element_error']=max(stats['max_K_element_error'],k_error)
            stats['max_w2c_element_error']=max(stats['max_w2c_element_error'],pose_error)
            for key,reference in [('fx',meta['fl_x']/meta['w']*dw),('fy',meta['fl_y']/meta['h']*dh),('cx',meta['cx']/meta['w']*dw),('cy',meta['cy']/meta['h']*dh)]:
                if key in data and not np.isclose(float(data[key]),reference,rtol=1e-5,atol=1e-4):raise ValueError(f'Depth/image calibration differs: {frame["name"]}/{key}')
            if 'other physical' in manifest.get('source_count_convention','').lower():
                enough_support=np.where(source==2,support>=2,support>=1)
            else:
                enough_support=support>=2
            valid &= np.isfinite(z)&(z>.05)&np.isfinite(confidence)&(confidence>0)&enough_support
            safe_z=np.where(valid,z,1.).astype(np.float32)
            values=torch.from_numpy(np.stack([safe_z,valid.astype(np.float32),confidence,source.astype(np.float32)]))[None]
            resized=F.interpolate(values,size=(frame['h'],frame['w']),mode='nearest-exact')[0].numpy()
            dvalid=(resized[1]>.5)&frame['mask']
            dconfidence=np.where(dvalid,np.clip(resized[2],0,1),0).astype(np.float32)
            if not dvalid.any():continue
            result[frame['name']]=dict(z=resized[0].copy(),valid=dvalid,confidence=dconfidence,source=resized[3].astype(np.uint8))
            stats['images']+=1;stats['valid_pixels']+=int(dvalid.sum())
            stats['sparse_pixels']+=int((dvalid&(resized[3]==1)).sum());stats['dense_pixels']+=int((dvalid&(resized[3]==2)).sum())
    if not result:raise ValueError('No valid static, positive, supported depth pixels remain')
    return result,index,stats


SSIM_KERNELS={}


def masked_losses(prediction,target,mask,ssim_mask):
    l1=((prediction-target).abs()*mask[...,None]).sum()/(mask.sum()*3).clamp_min(1)
    x,y=prediction.permute(2,0,1)[None],target.permute(2,0,1)[None]
    key=(str(x.device),x.dtype)
    if key not in SSIM_KERNELS:
        coords=torch.arange(11,device=x.device,dtype=x.dtype)-5
        kernel=torch.exp(-coords.square()/(2*1.5**2));kernel/=kernel.sum()
        SSIM_KERNELS[key]=(kernel[:,None]*kernel[None,:])[None,None].repeat(15,1,1,1)
    window=SSIM_KERNELS[key]
    stats=F.conv2d(torch.cat([x,y,x*x,y*y,x*y],1),window,padding=5,groups=15)
    mx,my,xx,yy,xy=stats.split(3,dim=1)
    ss=((2*mx*my+.01**2)*(2*(xy-mx*my)+.03**2))/((mx.square()+my.square()+.01**2)*(xx-mx.square()+yy-my.square()+.03**2)).clamp_min(1e-12)
    # Only windows whose complete 11x11 support is static are supervised.
    valid=ssim_mask[None,None]
    dssim=((1-ss)*valid).sum()/(valid.sum()*3).clamp_min(1)
    return l1,dssim


def render(params,frame,degree,depth=False):
    sh=torch.cat([params['sh0'],params['shN']],dim=1)
    return rasterization(params['means'],params['quats'],params['scales'].exp(),params['opacities'].sigmoid(),
        sh,frame['view_gpu'][None],frame['K_gpu'][None],frame['w'],frame['h'],sh_degree=degree,
        render_mode='RGB+ED' if depth else 'RGB',packed=True,
        near_plane=frame['near_normalized'],far_plane=frame['far_normalized'],rasterize_mode='classic')


def export_ply(params,path,center,radius):
    for value in params.values():
        if not torch.isfinite(value).all():raise FloatingPointError('Non-finite Gaussian parameter before export')
    with torch.no_grad():
        export_splats(means=params['means']*radius+torch.tensor(center,device='cuda',dtype=torch.float32),
            scales=params['scales']+math.log(radius),quats=F.normalize(params['quats'],dim=-1),
            opacities=params['opacities'],sh0=params['sh0'],shN=params['shN'],format='ply',save_to=str(path))
    temporary=path.with_suffix('.ply.tmp')
    with path.open('rb') as source,temporary.open('wb') as target:
        first=source.readline()
        if first!=b'ply\n':raise ValueError('Unexpected PLY header')
        target.write(first)
        second=source.readline()
        if second.strip()!=b'format binary_little_endian 1.0':raise ValueError('Unexpected PLY format')
        target.write(second)
        target.write(b'comment coordinates EDN metres; world_up 0 -1 0\n')
        target.write(b'comment quaternion WXYZ; scales log metres; opacity logits\n')
        shutil.copyfileobj(source,target,length=1024*1024)
    temporary.replace(path)


def evaluate(params,frames,output,step,degree):
    directory=output/f'eval_{step}';directory.mkdir(exist_ok=True)
    rows=[]
    with torch.no_grad():
        for frame in frames:
            name=frame['meta']['render_name']
            if Path(name).name!=name:raise ValueError('Unsafe render_name')
            pixels,alpha,_=render(params,frame,degree)
            rgb=(pixels[0].clamp(0,1)*255).round().byte().cpu().numpy()
            a=(alpha[0,:,:,0].clamp(0,1)*255).round().byte().cpu().numpy()
            image_path=directory/f'{name}.png';alpha_path=directory/f'{name}_alpha.png'
            Image.fromarray(rgb).save(image_path);Image.fromarray(a).save(alpha_path)
            rows.append(dict(file_path=frame['meta']['file_path'],render_name=name,
                evaluation_kind=frame['meta'].get('evaluation_kind','visual_QA'),valid_ground_truth=frame['meta'].get('valid_ground_truth',False),
                output_file=image_path.relative_to(output).as_posix(),alpha_file=alpha_path.relative_to(output).as_posix(),
                width=frame['w'],height=frame['h']))
    return rows


def train(args):
    runtime=renderer_provenance(args.renderer_library,args.renderer_library_sha256)
    torch.set_num_threads(args.cpu_workers)
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    torch.cuda.set_device(0)
    args.output.mkdir(parents=True,exist_ok=True)
    train_path=args.dataset/'transforms_train.json';val_path=args.dataset/'transforms_val.json'
    raw_frames=load_json(train_path)['frames'];raw_val=load_json(val_path)['frames']
    fingerprint_data=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ['output','resume']},
        train_sha256=sha(train_path),val_sha256=sha(val_path),seed_sha256=sha(args.init_gaussians_ply),trainer_sha256=sha(__file__),
        loader_sha256=sha(ROOT/'brush_refine_ply.py'))
    depth_manifest_path=next((args.depth_dir/n for n in ['depth_manifest.json','index.json','sparse_index.json','depth_index.json'] if (args.depth_dir/n).exists()),None) if args.depth_dir else None
    fingerprint_data['depth_manifest_sha256']=sha(depth_manifest_path) if depth_manifest_path else None
    signature=hashlib.sha256(json.dumps(fingerprint_data,sort_keys=True).encode()).hexdigest()
    status_path=args.output/'training_run.json'
    if status_path.exists():
        old=load_json(status_path)
        if old.get('fingerprint')!=signature:raise ValueError('Existing run configuration/source differs; select a new output directory')
        if old.get('status')=='process_completed' and (args.output/'final.ply').exists() and all((args.output/e['output_file']).exists() and (args.output/e['alpha_file']).exists() for e in old.get('rendered_frames',[])):
            log(dict(event='already_completed',output=str(args.output),steps=old['completed_steps']));return
    status=dict(status='running',exit_code=None,fingerprint=signature,dataset_path=str(args.dataset),
        transforms_train_sha256=sha(train_path),transforms_val_sha256=sha(val_path),
        dataset_manifest_sha256=sha(args.dataset/'dataset_manifest.json') if (args.dataset/'dataset_manifest.json').exists() else None,
        trainer_sha256=sha(__file__),sh_degree=args.sh_degree,expected_steps=args.steps,completed_steps=0,
        resolution=args.resolution,max_splats=args.max_splats,seed=args.seed,
        source_initial_ply=str(args.init_gaussians_ply),source_initial_ply_sha256=sha(args.init_gaussians_ply),
        initial_ply_sha256=sha(args.init_gaussians_ply),reference_mode=args.render_only,optimization_steps=0,
        fixed_count=True,point_order_preserved=True,regularization_scope=args.regularization_scope,
        renderer_fingerprint=runtime,
        depth_manifest_sha256=fingerprint_data['depth_manifest_sha256'],
        ply_path='final.ply',rendered_frames=[],started_utc=now(),settings=fingerprint_data['arguments'],
        torch_version=torch.__version__,gsplat_version=gsplat.__version__,gpu=torch.cuda.get_device_name(0))
    json_write(status_path,status)
    start=time.monotonic();completed=0
    try:
        from .brush_refine import physical_normalization
        center,radius=physical_normalization(raw_frames)
        if not np.isfinite(radius) or radius<=0:raise ValueError('Invalid camera scene extent')
        with ThreadPoolExecutor(args.cpu_workers) as pool:
            frames=list(pool.map(lambda f:prepare_frame(f,args.dataset,args.resolution,center,radius,True),raw_frames))
        val_frames=[prepare_frame(f,args.dataset,args.resolution,center,radius,False) for f in raw_val]
        if len({f['meta']['render_name'] for f in val_frames})!=len(val_frames):raise ValueError('Duplicate evaluation render name')
        for frame in frames+val_frames:
            frame['view_gpu']=torch.tensor(frame['view'],device='cuda');frame['K_gpu']=torch.tensor(frame['K'],device='cuda')
            frame['near_normalized']=.05/radius;frame['far_normalized']=1e7/radius
        depths,depth_index,depth_stats=load_depths(args,frames)
        loaded,loader_diagnostics=load_gaussian_ply(args.init_gaussians_ply,center,radius)
        count=len(loaded['means'])
        if count>args.max_splats:raise ValueError('Fixed refinement must preserve every Gaussian; increase --max-splats instead of truncating')
        params=torch.nn.ParameterDict({key:torch.nn.Parameter(torch.tensor(value,device='cuda')) for key,value in loaded.items()})
        del loaded
        base_lr=dict(means=1.6e-4,scales=.005,quats=.001,opacities=.05,sh0=.0025,shN=.0025/20)
        multipliers=dict(means=args.means_lr_multiplier,scales=args.scale_lr_multiplier,quats=args.rotation_lr_multiplier,
            opacities=args.opacity_lr_multiplier,sh0=args.sh_lr_multiplier,shN=args.sh_lr_multiplier)
        multipliers={key:args.lr_multiplier if value is None else value for key,value in multipliers.items()}
        lr0={key:value*multipliers[key] for key,value in base_lr.items()}
        optimizer=lambda: {name:torch.optim.Adam([value],lr=lr0[name],eps=1e-15) for name,value in params.items()}
        optimizers=optimizer()
        factor=args.steps/30000
        strategy=FixedStrategy()
        state=strategy.initialize_state();strategy.check_sanity(params,optimizers)
        depth_start=args.depth_start if args.depth_start is not None else round(.05*args.steps)
        depth_ramp=args.depth_ramp if args.depth_ramp is not None else max(1,round(.1*args.steps))
        schedule=np.random.default_rng(args.seed).integers(0,len(frames),size=args.steps)
        depth_gradient_verified=False;depth_supervised_steps=0
        checkpoint=args.resume or args.output/'checkpoint_last.pt'
        if checkpoint.exists():
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if saved['fingerprint']!=signature:raise ValueError('Checkpoint configuration differs')
            params=torch.nn.ParameterDict({k:torch.nn.Parameter(v.to('cuda')) for k,v in saved['params'].items()})
            optimizers=optimizer()
            for name,value in optimizers.items():value.load_state_dict(saved['optimizers'][name])
            state=saved['strategy_state'];completed=saved['completed_steps']
            depth_gradient_verified=saved['depth_gradient_verified'];depth_supervised_steps=saved['depth_supervised_steps']
            torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
            log(dict(event='resumed',checkpoint=str(checkpoint),completed_steps=completed))
        def save_checkpoint(step):
            data=dict(fingerprint=signature,completed_steps=step,params={k:v.detach().cpu() for k,v in params.items()},
                optimizers={k:o.state_dict() for k,o in optimizers.items()},strategy_state=state,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
                depth_gradient_verified=depth_gradient_verified,depth_supervised_steps=depth_supervised_steps)
            tmp=args.output/'checkpoint_last.pt.tmp';last=args.output/'checkpoint_last.pt'
            torch.save(data,tmp)
            if last.exists():last.replace(args.output/'checkpoint_previous.pt')
            tmp.replace(last)
        status.update(scene_radius_m=radius,normalization_center_edn_m=center.tolist(),
            coordinate_policy='Internal (EDN_m-center)/radius; exports restore EDN metres. Original quaternion/SH directions and point order unchanged.',
            means_lr_initial_normalized=lr0['means'],means_lr_initial_m=lr0['means']*radius,
            means_lr_final_normalized=lr0['means']*.01,means_lr_final_m=lr0['means']*.01*radius,
            scale_regularizer='scale_reg * sum(selected scale_metres)/(3*N*scene_radius_metres); selected=visible unique IDs or all',
            opacity_regularizer='opacity_reg * sum(selected sigmoid(opacity_logits))/N; selected=visible unique IDs or all',
            depth_loss='Confidence-weighted smooth-L1 of (1/z_render_m - 1/z_target_m)*scene_radius; static+valid+positive; sparse >=2 physical stations including target, MVS >=2 other physical stations; detached alpha gate',
            depth_index=str(depth_index) if depth_index else None,depth_index_sha256=sha(depth_index) if depth_index else None,
            depth_input=depth_stats,depth_start_step=depth_start,depth_ramp_steps=depth_ramp,
            near_plane_m=.05,far_plane_m=1e7,initial_gaussians=count,
            initial_scales_m=loader_diagnostics['scales_m'],loader_diagnostics=loader_diagnostics,
            strategy=dict(name='fixed',noise=False,relocation=False,growth=False,pruning=False),
            base_learning_rates=base_lr,learning_rate_multipliers=multipliers,initial_learning_rates=lr0,
            training_images=len(frames),
            training_physical_stations=len({str(f['meta']['station_id']) for f in frames}),completed_steps=completed)
        json_write(status_path,status);log(dict(event='initialized',scene_radius_m=radius,gaussians=len(params['means']),depth_input=depth_stats))
        metrics_file=(args.output/'metrics_history.jsonl').open('a',encoding='utf-8')
        loop_start=time.monotonic();start_step=completed
        for step in range(completed,args.steps):
            frame=frames[int(schedule[step])]
            pixels=torch.tensor(frame['rgb'],device='cuda',dtype=torch.float32)/255
            mask=torch.tensor(frame['mask'],device='cuda')
            ssim_mask=torch.tensor(frame['ssim_mask'],device='cuda')
            d=depths.get(frame['name']);use_depth=d is not None
            degree=2
            lr=lr0['means']*.01**(step/args.steps)
            optimizers['means'].param_groups[0]['lr']=lr
            for opt in optimizers.values():opt.zero_grad(set_to_none=True)
            rendered,alpha,info=render(params,frame,degree,use_depth)
            if step==start_step:assert_loaded_renderer(runtime)
            l1,ssim=masked_losses(rendered[0,...,:3],pixels,mask,ssim_mask)
            opacity_values=params['opacities'].sigmoid()
            scale_values=params['scales'].exp()
            opacity_mean=opacity_values.mean();scale_mean_normalized=scale_values.mean()
            visible=info['gaussian_ids'].unique()
            if args.regularization_scope=='visible_sum':
                penalty_opacity=opacity_values[visible].sum()/count
                penalty_scale=scale_values[visible].sum()/(3*count)
            else:
                penalty_opacity=opacity_mean;penalty_scale=scale_mean_normalized
            reg=args.opacity_reg*penalty_opacity+args.scale_reg*penalty_scale
            loss_rgb=.8*l1+.2*ssim
            loss_depth=rendered[0,0,0,0]*0
            valid_count=0;confidence_mean=0.;grad_record={}
            ramp=max(0.,min(1.,(step-depth_start)/depth_ramp))
            weight=args.depth_weight*ramp
            if use_depth and weight>0:
                target_z=torch.tensor(d['z'],device='cuda')
                confidence=torch.tensor(d['confidence'],device='cuda')
                valid=torch.tensor(d['valid'],device='cuda')&mask&(alpha[0,...,0].detach()>args.depth_alpha_min)
                predicted_z=rendered[0,...,3]*radius
                valid=valid&torch.isfinite(predicted_z)&(predicted_z>.05)
                valid_count=int(valid.sum())
                if valid_count:
                    error=(predicted_z[valid].reciprocal()-target_z[valid].reciprocal())*radius
                    robust=F.smooth_l1_loss(error,torch.zeros_like(error),beta=args.depth_huber_beta,reduction='none')
                    confidence=confidence[valid]
                    loss_depth=(robust*confidence).sum()/confidence.sum().clamp_min(1e-6)
                    confidence_mean=float(confidence.mean());depth_supervised_steps+=1
                    if not depth_gradient_verified or (step+1)%100==0:
                        grads=torch.autograd.grad(weight*loss_depth,[params[k] for k in ['means','scales','opacities']],retain_graph=True,allow_unused=True)
                        grad_record={name:float(g.norm()) if g is not None else 0. for name,g in zip(['depth_gradient_means_normalized','depth_gradient_logscales','depth_gradient_opacity_logits'],grads)}
                        if all(np.isfinite(v) for v in grad_record.values()) and grad_record['depth_gradient_means_normalized']>0:depth_gradient_verified=True
            loss=loss_rgb+reg+weight*loss_depth
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at step {step+1}')
            loss.backward()
            for opt in optimizers.values():opt.step()
            strategy.step_post_backward(params,optimizers,state,step,info,lr=lr)
            if len(params['means'])!=count:raise RuntimeError('Fixed Gaussian count invariant violated')
            completed=step+1
            if completed==1 or completed%100==0 or completed==args.steps or grad_record:
                torch.cuda.synchronize()
                means_finite=bool(torch.isfinite(params['means']).all())
                if not means_finite:raise FloatingPointError('Nonfinite Gaussian position')
                elapsed=time.monotonic()-loop_start
                row=dict(event='training',step=completed,gaussians=len(params['means']),image=frame['name'],
                    loss=scalar(loss),rgb_l1=scalar(l1),dssim=scalar(ssim),opacity_mean=scalar(opacity_mean),
                    visible_gaussians=len(visible),visible_fraction=len(visible)/count,regularization_scope=args.regularization_scope,
                    scale_mean_m=scalar(scale_mean_normalized)*radius,scale_mean_normalized=scalar(scale_mean_normalized),
                    regularization_loss=scalar(reg),depth_loss=scalar(loss_depth),depth_effective_weight=weight,
                    depth_valid_pixels=valid_count,depth_confidence_mean=confidence_mean,depth_supervised_steps=depth_supervised_steps,
                    depth_gradient_verified=depth_gradient_verified,mean_lr_normalized=lr,mean_lr_m=lr*radius,
                    steps_per_second=(completed-start_step)/max(elapsed,1e-6),peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30,**grad_record)
                metrics_file.write(json.dumps(row,allow_nan=False)+'\n');metrics_file.flush();log(row)
            if args.export_every>0 and completed%args.export_every==0:
                save_checkpoint(completed);export_ply(params,args.output/f'step_{completed}.ply',center,radius)
                status.update(completed_steps=completed,current_gaussians=len(params['means']),last_checkpoint='checkpoint_last.pt',updated_utc=now())
                json_write(status_path,status)
            if args.eval_every>0 and completed%args.eval_every==0 and completed!=args.steps:
                evaluate(params,val_frames,args.output,completed,degree)
        metrics_file.close()
        if not args.render_only:save_checkpoint(completed)
        if args.depth_weight and not depth_gradient_verified and not args.render_only:raise RuntimeError('No verified nonzero depth geometry gradient; depth experiment is incomplete')
        export_ply(params,args.output/'final.ply',center,radius)
        rendered_frames=evaluate(params,val_frames,args.output,completed,2)
        assert_loaded_renderer(runtime)
        status.update(status='process_completed',exit_code=0,completed_steps=completed,completed_utc=now(),
            final_gaussians=len(params['means']),rendered_frames=rendered_frames,depth_gradient_verified=depth_gradient_verified,
            optimization_steps=completed,point_order_preserved=True,
            depth_supervised_steps=depth_supervised_steps,duration_seconds=time.monotonic()-start,
            peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30,last_checkpoint=None if args.render_only else 'checkpoint_last.pt')
        json_write(status_path,status);log(dict(event='process_completed',steps=completed,gaussians=len(params['means']),output=str(args.output)))
    except BaseException as exc:
        status.update(status='failed',exit_code=1,completed_steps=completed,error_type=type(exc).__name__,error=str(exc),failed_utc=now())
        json_write(status_path,status)
        raise


if __name__=='__main__':
    train(parse_args())
