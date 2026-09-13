"""Optional full-forward UniSHARP depth diagnostics; never changes SfM or RGB.

Only run() imports Torch/UniSHARP or starts inference. Numeric calibration uses
actual training SfM tracks. Every exported learned-valid pixel needs agreement
from other physical stations; unobservable floor/sky stays invalid.
"""
from __future__ import annotations
from dataclasses import asdict,dataclass,fields
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class DepthPriorSettings:
    repo_path:str=''
    checkpoint_path:str=''
    checkpoint_sha256:str=''
    unik3d_snapshot_path:str=''
    unik3d_sha256:str=''
    extra_python_path:str=''
    output_dir:str='depth_prior'
    resume_raw:bool=False
    device:str='cuda:0'
    erp_width:int=1536
    input_projection:str='spherical'
    perspective_side:int=768
    output_side:int=384
    seed:int=0
    minimum_fit_points:int=20
    minimum_test_points:int=8
    fit_fraction:float=.7
    maximum_test_median_relative_error:float=.2
    maximum_test_p90_relative_error:float=.5
    minimum_test_inlier_fraction:float=.65
    test_inlier_relative_error:float=.25
    minimum_anchor_station_support:int=2
    minimum_anchor_angle_degrees:float=2.
    maximum_anchor_reprojection_px:float=2.
    maximum_anchor_distance_fraction:float=1/12
    maximum_local_relative_error:float=.2
    calibration_quantile:float=.02
    maximum_depth_edge_relative:float=.12
    minimum_other_station_support:int=2
    maximum_source_stations:int=12
    maximum_source_faces_per_station:int=3
    maximum_relative_depth_error:float=.05
    maximum_roundtrip_pixels:float=2.
    maximum_rgb_patch_error:float=.1
    minimum_ray_angle_degrees:float=.5

    def __post_init__(self):
        if not isinstance(self.resume_raw,bool) or not isinstance(self.output_dir,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',self.output_dir):raise ValueError('Invalid depth output/resume setting')
        if self.input_projection not in {'spherical','perspective'}:raise ValueError('Unsupported depth input projection')
        for name in ['erp_width','perspective_side','output_side','minimum_fit_points','minimum_test_points','minimum_anchor_station_support','minimum_other_station_support','maximum_source_stations','maximum_source_faces_per_station']:
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or value<1:raise ValueError('Invalid depth setting: '+name)
        if self.erp_width<32 or self.erp_width%2 or not 8<=self.output_side<=4096:raise ValueError('Invalid ERP/output resolution')
        if not 32<=self.perspective_side<=4096:raise ValueError('Invalid perspective input resolution')
        if self.minimum_anchor_station_support<2 or self.minimum_other_station_support<2:raise ValueError('Depth requires independent physical station support')
        if self.maximum_source_stations<self.minimum_other_station_support:raise ValueError('Not enough configured source stations')
        if not isinstance(self.seed,int) or isinstance(self.seed,bool) or self.seed<0:raise ValueError('Invalid seed')
        for field in fields(self):
            value=getattr(self,field.name)
            if isinstance(value,float) and (not math.isfinite(value) or value<=0):raise ValueError('Invalid depth setting: '+field.name)
        for name in ['fit_fraction','minimum_test_inlier_fraction','maximum_anchor_distance_fraction','maximum_local_relative_error','maximum_relative_depth_error','maximum_depth_edge_relative','maximum_rgb_patch_error']:
            if not 0<getattr(self,name)<1:raise ValueError('Invalid fraction: '+name)
        if not 0<self.calibration_quantile<.25 or self.minimum_ray_angle_degrees>=90:raise ValueError('Invalid depth range/parallax setting')
        if not self.device.startswith('cuda'):raise ValueError('UniSHARP inference uses the configured cloud CUDA device')


def _sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def _read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def _save(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')


def _file(root,relative):
    path=Path(relative)
    if path.is_absolute() or '..' in path.parts:raise ValueError('Expected a job-relative path')
    result=(root/path).resolve()
    if not result.is_relative_to(root.resolve()) or not result.is_file():raise ValueError('Missing or escaped input: '+str(relative))
    return result


def _token(value):return hashlib.sha256(str(value).encode()).hexdigest()[:24]


def _rotation(matrix):
    a=np.asarray(matrix,np.float64)
    if a.shape!=(4,4) or not np.isfinite(a).all() or not np.allclose(a[3],[0,0,0,1]) or not np.allclose(a[:3,:3].T@a[:3,:3],np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(a[:3,:3]),1,atol=1e-6):raise ValueError('Invalid calibrated rigid transform')
    return a


def camera(frame,side=None):
    w,h=int(frame['w']),int(frame['h']);ow,oh=(w,h) if side is None else (side,side)
    K=np.array([[frame['fl_x']*ow/w,0,frame['cx']*ow/w],[0,frame['fl_y']*oh/h,frame['cy']*oh/h],[0,0,1]],np.float64)
    if w<1 or h<1 or not np.isfinite(K).all() or K[0,0]<=0 or K[1,1]<=0:raise ValueError('Invalid pinhole calibration')
    c2w=_rotation(frame['transform_matrix'])@np.diag([1.,-1.,-1.,1.])
    return K,np.linalg.inv(c2w),c2w


def erp_rays(width):
    height=width//2;lon=((np.arange(width)+.5)/width-.5)*2*np.pi;lat=((np.arange(height)+.5)/height-.5)*np.pi
    return np.stack([np.cos(lat[:,None])*np.sin(lon[None]),np.broadcast_to(np.sin(lat[:,None]),(height,width)),np.cos(lat[:,None])*np.cos(lon[None])],-1).astype(np.float32)


def sample_erp(array,rays):
    """Bilinear spherical lookup: longitude wraps, latitude clamps at poles."""
    rays=np.asarray(rays,np.float64);rays=rays/np.maximum(np.linalg.norm(rays,axis=-1,keepdims=True),1e-15)
    h,w=array.shape[:2];x=(np.arctan2(rays[...,0],rays[...,2])/(2*np.pi)+.5)*w-.5
    y=(np.arcsin(np.clip(rays[...,1],-1,1))/np.pi+.5)*h-.5
    x0=np.floor(x).astype(np.int64);y=np.clip(y,0,h-1);y0=np.floor(y).astype(np.int64)
    dx=x-x0;dy=y-y0
    if array.ndim==3:dx=dx[...,None];dy=dy[...,None]
    return ((1-dy)*((1-dx)*array[y0,x0%w]+dx*array[y0,(x0+1)%w])+dy*((1-dx)*array[np.minimum(y0+1,h-1),x0%w]+dx*array[np.minimum(y0+1,h-1),(x0+1)%w]))


def perspective_input(frame,dataset,side):
    """Native rectilinear RGB plus known K; never estimate a new focal length."""
    with Image.open(_file(dataset,frame['file_path'])) as image:
        if image.size!=(frame['w'],frame['h']):raise ValueError('Source RGB size disagrees with calibration')
        image=image.convert('RGB')
        # Cube faces are square; reject a silent aspect-ratio change.
        if frame['w']!=frame['h']:raise ValueError('Perspective cube input must be square')
        side=min(side,frame['w'])
        rgb=np.asarray(image.resize((side,side),Image.Resampling.LANCZOS))
    K,_,_=camera(frame,side)
    return rgb,dict(projection='perspective',width=side,height=side,
        intrinsics_pixel_edge=K.tolist(),camera_to_station_cv=frame['camera_to_station_cv'],
        source_frame=frame['file_path'],source_image_sha256=frame['image_sha256'],
        ray_convention='Known cube pinhole; pixel-edge K, sample centers x+0.5/y+0.5; model integer-center principal point offset -0.5')


def sample_raw(raw,key,station_rays):
    """Sample range/confidence in the actual model projection."""
    if raw.get('projection','spherical')=='spherical':return sample_erp(raw[key],station_rays)
    if raw['projection']!='perspective':raise ValueError('Unsupported raw depth projection')
    rays=np.asarray(station_rays,np.float64)@_rotation(raw['camera_to_station_cv'])[:3,:3]
    K=np.asarray(raw['intrinsics_pixel_edge'],np.float64);h,w=raw[key].shape
    projected=rays@K.T
    u=projected[...,0]/np.maximum(projected[...,2],1e-15);v=projected[...,1]/np.maximum(projected[...,2],1e-15)
    inside=(rays[...,2]>0)&(u>=0)&(u<w)&(v>=0)&(v<h)
    x=np.clip(u-.5,0,w-1);y=np.clip(v-.5,0,h-1)
    x0=np.floor(x).astype(np.int64);y0=np.floor(y).astype(np.int64);dx=x-x0;dy=y-y0
    a=raw[key]
    sampled=(1-dy)*((1-dx)*a[y0,x0]+dx*a[y0,np.minimum(x0+1,w-1)])+dy*((1-dx)*a[np.minimum(y0+1,h-1),x0]+dx*a[np.minimum(y0+1,h-1),np.minimum(x0+1,w-1)])
    return np.where(inside,sampled,np.nan)


def assemble_erp(frames,dataset,width):
    """Project immutable six-face RGB through supplied camera rotations/K."""
    import cv2
    if len(frames)!=6:raise ValueError('A complete panorama requires exactly six calibrated cube faces')
    if len({str(f['face']) for f in frames})!=6:raise ValueError('Duplicate cube face')
    rays=erp_rays(width);rgb=np.zeros((*rays.shape[:2],3),np.uint8);owner=np.full(rays.shape[:2],-1,np.int16);score=np.full(rays.shape[:2],-np.inf,np.float32)
    directions=[]
    for index,frame in enumerate(frames):
        pose=_rotation(frame['camera_to_station_cv'])
        if not np.allclose(pose[:3,3],0,atol=1e-8):raise ValueError('Cube faces must share one capture center')
        directions.append(pose[:3,2]);local=rays@pose[:3,:3]
        z=local[...,2];u=frame['fl_x']*local[...,0]/np.maximum(z,1e-12)+frame['cx'];v=frame['fl_y']*local[...,1]/np.maximum(z,1e-12)+frame['cy']
        valid=(z>0)&(u>=0)&(u<frame['w'])&(v>=0)&(v<frame['h'])&(z>score)
        with Image.open(_file(dataset,frame['file_path'])) as image:
            if image.size!=(frame['w'],frame['h']):raise ValueError('Source RGB size disagrees with calibration')
            image=np.asarray(image.convert('RGB'))
        mapped=cv2.remap(image,(u-.5).astype(np.float32),(v-.5).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        rgb[valid]=mapped[valid];owner[valid]=index;score[valid]=z[valid]
    if (owner<0).any() or len(np.unique(owner))!=6:raise ValueError('Calibrated six faces do not cover the full ERP, including poles')
    if np.max(np.asarray(directions)@np.asarray(directions).T-np.eye(6)*2)>.99:raise ValueError('Duplicated optical axes')
    return rgb,dict(width=width,height=width//2,face_pixel_counts={str(f['face']):int((owner==i).sum()) for i,f in enumerate(frames)},ray_convention='OpenCV station-local X right/Y down/Z front; half-pixel spherical rays')


def point_fit_split(point_ids,seed,fit_fraction):
    # Global point-ID split across all images: duplicate views never leak a point.
    threshold=int(fit_fraction*2**64)
    return np.array([int.from_bytes(hashlib.blake2b(f'{seed}:{int(pid)}'.encode(),digest_size=8).digest(),'big')<threshold for pid in point_ids],bool)


def deduplicate_observations(observations):
    """One actual row per (frame, point), lowest error then xy for stable ties.

    Reject a group with inconsistent point depth/support/split metadata. Never
    average its pixels or increase physical-station support from duplicate rows.
    """
    groups={}
    for index,(frame_name,point_id) in enumerate(zip(observations['frame_name'],observations['point_id'])):
        groups.setdefault((str(frame_name),int(point_id)),[]).append(index)
    selected=[];duplicates=[];rejected=[]
    for (frame_name,point_id),indices in sorted(groups.items()):
        if len(indices)>1:
            consistent=(len(set(map(str,observations['station_id'][indices])))==1 and len(set(map(str,observations['split'][indices])))==1
                        and np.allclose(observations['depth_z'][indices],observations['depth_z'][indices[0]],rtol=1e-9,atol=0)
                        and np.all(observations['support_station_count'][indices]==observations['support_station_count'][indices[0]])
                        and np.allclose(observations['triangulation_angle_degrees'][indices],observations['triangulation_angle_degrees'][indices[0]],rtol=1e-6,atol=0))
            if not consistent:
                rejected.append(dict(frame_name=frame_name,point_id=point_id,source_row_indices=indices,reason='Conflicting depth/support/station/split metadata for one frame-point'));continue
        chosen=min(indices,key=lambda i:(float(observations['reprojection_error_px'][i]),float(observations['xy'][i,0]),float(observations['xy'][i,1]),i))
        selected.append(chosen)
        if len(indices)>1:duplicates.append(dict(frame_name=frame_name,point_id=point_id,source_row_indices=indices,selected_source_row_index=chosen,selected_reprojection_error_px=float(observations['reprojection_error_px'][chosen]),selected_xy=observations['xy'][chosen].tolist()))
    selected=np.asarray(selected,np.int64)
    result={key:values[selected] for key,values in observations.items()}
    return result,dict(input_rows=len(observations['point_id']),unique_frame_point_groups=len(groups),output_rows=len(selected),duplicate_groups=len(duplicates)+len(rejected),removed_rows=len(observations['point_id'])-len(selected),conflicting_groups_rejected=len(rejected),policy='One actual observation per frame/point; minimum reprojection error, then lexicographic xy; identical ties choose earliest row. No averaging or extra station vote.',selected_duplicates=duplicates,rejected_groups=rejected,limitation='Distinct point IDs may share aliased visual features; point-ID calibration holdout remains transductive, not independent measured truth.')


def _metrics(pred,target):
    if not len(target):return dict(count=0,median_relative_error=None,p90_relative_error=None)
    errors=np.abs(pred-target)/target
    return dict(count=len(target),median_relative_error=float(np.median(errors)),p90_relative_error=float(np.quantile(errors,.9)))


def calibrate_scale(raw,metric,point_ids,options):
    """One robust multiplicative model; test points never estimate its scale."""
    raw=np.asarray(raw,np.float64);metric=np.asarray(metric,np.float64);point_ids=np.asarray(point_ids,np.int64)
    good=np.isfinite(raw)&np.isfinite(metric)&(raw>0)&(metric>0)
    fit=point_fit_split(point_ids,options.seed,options.fit_fraction)&good;test=(~point_fit_split(point_ids,options.seed,options.fit_fraction))&good
    report=dict(accepted=False,model='multiplicative_radial_scale',fit_count=int(fit.sum()),test_count=int(test.sum()),test_scope='Disjoint point IDs within TRAIN stations, same transductive SfM geometry; not independent truth')
    if len(np.unique(point_ids))!=len(point_ids):raise ValueError('Duplicate calibration point IDs in one face')
    if fit.sum()<options.minimum_fit_points or test.sum()<options.minimum_test_points:
        return report,fit,test
    scale=float(np.exp(np.median(np.log(metric[fit]/raw[fit]))));pred=raw*scale
    errors=np.abs(pred[test]-metric[test])/metric[test];stats=_metrics(pred[test],metric[test]);inliers=float((errors<options.test_inlier_relative_error).mean())
    report.update(scale=scale,validation=stats,test_inlier_fraction=inliers,
                  raw_fit_range=np.quantile(raw[fit],[options.calibration_quantile,1-options.calibration_quantile]).tolist(),
                  metric_fit_range=np.quantile(metric[fit],[options.calibration_quantile,1-options.calibration_quantile]).tolist())
    q=np.quantile(metric[fit],[0,.5,1]);report['test_depth_bins']={name:_metrics(pred[test&subset],metric[test&subset]) for name,subset in [('nearer_than_fit_median',metric<=q[1]),('farther_than_fit_median',metric>q[1])]}
    report['test_depth_bins_median_m']=float(q[1])
    report['accepted']=bool(stats['median_relative_error']<=options.maximum_test_median_relative_error and stats['p90_relative_error']<=options.maximum_test_p90_relative_error and inliers>=options.minimum_test_inlier_fraction)
    return report,fit,test


def calibrated_field(frame,dataset,raw,observations,options):
    """Calibrate one face using its actual, foreground TRAIN image tracks."""
    import cv2
    from scipy.spatial import cKDTree
    side=options.output_side;K,w2c,c2w=camera(frame,side)
    yy,xx=np.indices((side,side),dtype=np.float64);uv=np.stack([xx+.5,yy+.5,np.ones_like(xx)],-1)
    ray=uv@np.linalg.inv(K).T;pose=_rotation(frame['camera_to_station_cv']);station_rays=ray@pose[:3,:3].T
    predicted=sample_raw(raw,'radial_distance_model',station_rays)
    confidence_raw=sample_raw(raw,'confidence',station_rays)
    with Image.open(_file(dataset,frame['file_path'])) as image:rgb=np.asarray(image.convert('RGB').resize((side,side),Image.Resampling.BOX),np.float32)/255
    with Image.open(_file(dataset,frame['sfm_mask_path'])) as image:
        native_mask=np.asarray(image.convert('L'))>0
        # BOX requires every contributing source pixel valid; unknown regions
        # cannot become certified foreground by nearest-neighbor downsampling.
        coverage=np.asarray(image.convert('F').resize((side,side),Image.Resampling.BOX))
    static=cv2.erode((coverage>=254.999).astype(np.uint8),np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    field=dict(frame=frame,K=K,w2c=w2c,c2w=c2w,ray=ray,rgb=rgb,depth_z=np.zeros((side,side),np.float32),valid=np.zeros((side,side),bool),confidence=np.zeros((side,side),np.float32))
    report=dict(frame_name=frame['file_path'],pano_id=str(frame['pano_id']),station_id=str(frame['station_id']),face=str(frame['face']),split=frame['split'],accepted=False,candidate_pixels=0)
    if frame['split']!='train':
        report['reason']='Heldout capture has no training-only per-face calibration; raw inference is diagnostic only';return field,report
    choose=(observations['frame_name']==frame['file_path'])&(observations['split']=='train')&(observations['station_id']==str(frame['station_id']))
    indices=np.flatnonzero(choose)
    if not len(indices):report['reason']='No actual training SfM observations';return field,report
    xy=np.asarray(observations['xy'][indices],np.float64);depth=np.asarray(observations['depth_z'][indices],np.float64)
    knative,_,_=camera(frame);unit=np.column_stack([xy,np.ones(len(xy))])@np.linalg.inv(knative).T
    araw=sample_raw(raw,'radial_distance_model',unit@pose[:3,:3].T);metric=depth*np.linalg.norm(unit,axis=-1)
    valid=np.isfinite(xy).all(axis=1)&np.isfinite(depth)&(depth>0)&(observations['support_station_count'][indices]>=options.minimum_anchor_station_support)&(observations['triangulation_angle_degrees'][indices]>=options.minimum_anchor_angle_degrees)&(observations['reprojection_error_px'][indices]<=options.maximum_anchor_reprojection_px)
    pixel=np.floor(np.nan_to_num(xy,nan=-3,posinf=-3,neginf=-3)).astype(np.int64)
    valid&=(pixel[:,0]>=1)&(pixel[:,0]<frame['w']-1)&(pixel[:,1]>=1)&(pixel[:,1]<frame['h']-1)
    eroded=cv2.erode(native_mask.astype(np.uint8),np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
    valid&=eroded[np.clip(pixel[:,1],0,frame['h']-1),np.clip(pixel[:,0],0,frame['w']-1)]>0
    indices=indices[valid];xy=xy[valid];araw=araw[valid];metric=metric[valid]
    fit_report,fit,test=calibrate_scale(araw,metric,observations['point_id'][indices],options);report.update(fit_report)
    report['fit_point_ids']=[int(i) for i in observations['point_id'][indices][fit]]
    report['test_point_ids']=[int(i) for i in observations['point_id'][indices][test]]
    if not report['accepted']:report['reason']='Insufficient calibration support or failed disjoint-point test';return field,report
    normalized=xy/np.array([frame['w'],frame['h']]);query=uv[...,:2].reshape(-1,2)/side
    nearest,neighbors=cKDTree(normalized[fit]).query(query,k=min(5,int(fit.sum())))
    if nearest.ndim==1:nearest=nearest[:,None];neighbors=neighbors[:,None]
    residual=np.abs(araw[fit]*report['scale']-metric[fit])/metric[fit]
    local_error=np.median(residual[neighbors],axis=1).reshape(side,side);distance=nearest[:,0].reshape(side,side)
    radial=predicted*report['scale'];z=radial/np.linalg.norm(ray,axis=-1)
    rawlo,rawhi=report['raw_fit_range'];lo,hi=report['metric_fit_range']
    valid=static&np.isfinite(z)&(z>0)&np.isfinite(confidence_raw)&(confidence_raw>0)&(predicted>=rawlo)&(predicted<=rawhi)&(radial>=lo)&(radial<=hi)&(distance<=options.maximum_anchor_distance_fraction)&(local_error<=options.maximum_local_relative_error)
    z=np.nan_to_num(z,nan=0,posinf=0,neginf=0).astype(np.float32)
    spread=cv2.dilate(z,np.ones((3,3),np.uint8))-cv2.erode(z,np.ones((3,3),np.uint8))
    valid&=spread<=options.maximum_depth_edge_relative*z
    confidence=(.5*np.clip(1-local_error/options.maximum_local_relative_error,0,1)).astype(np.float32)
    field.update(depth_z=z,valid=valid,confidence=np.where(valid,confidence,0))
    report['candidate_pixels']=int(valid.sum());report['raw_confidence_policy']='Requires finite positive model confidence; numeric magnitude is not treated as a probability'
    return field,report


def _world(field,uv=None,depth=None):
    ray=field['ray'] if uv is None else uv@np.linalg.inv(field['K']).T
    z=field['depth_z'] if depth is None else depth
    return (ray*z[...,None])@field['c2w'][:3,:3].T+field['c2w'][:3,3]


def select_sources(target,all_fields,options):
    """Rank by actual candidate-ray frustum overlap, without facing/metric cuts."""
    side=target['valid'].shape[0];locations=np.argwhere(target['valid'])
    if not len(locations):return []
    locations=locations[np.linspace(0,len(locations)-1,min(128,len(locations))).astype(int)]
    y,x=locations.T;uv=np.column_stack([x+.5,y+.5,np.ones(len(x))]);world=_world(target,uv,target['depth_z'][y,x])
    groups={};station=str(target['frame']['station_id'])
    for source in all_fields:
        source_station=str(source['frame']['station_id'])
        if source_station==station or source['frame']['split']!='train' or not source['valid'].any():continue
        q=world@source['w2c'][:3,:3].T+source['w2c'][:3,3];p=q@source['K'].T
        xy=p[:,:2]/np.maximum(p[:,2:],1e-12)
        inside=(q[:,2]>0)&(xy[:,0]>=1)&(xy[:,0]<side-1)&(xy[:,1]>=1)&(xy[:,1]<side-1)
        if inside.any():groups.setdefault(source_station,[]).append((int(inside.sum()),source))
    ranked=[]
    for station,rows in groups.items():
        rows.sort(key=lambda row:(-row[0],row[1]['frame']['file_path']));ranked.append((max(row[0] for row in rows),station,[row[1] for row in rows[:options.maximum_source_faces_per_station]]))
    ranked.sort(key=lambda row:(-row[0],row[1]))
    return [(station,sources) for _,station,sources in ranked[:options.maximum_source_stations]]


def multiview_support(target,source_groups,options):
    """Other-station depth + roundtrip + RGB agreement. Occlusion is neutral."""
    import cv2
    z=target['depth_z'];side=z.shape[0];world=_world(target);yy,xx=np.indices(z.shape)
    count=np.zeros(z.shape,np.uint16);station_counts={}
    ray1=world-target['c2w'][:3,3];norm1=np.linalg.norm(ray1,axis=-1)
    used=set();target_station=str(target['frame']['station_id'])
    for station,sources in source_groups:
        station=str(station)
        if station in used or station==target_station:continue
        used.add(station);group_good=np.zeros(z.shape,bool)
        for source in sources:
            if str(source['frame']['station_id'])!=station or source['frame']['split']!='train':raise ValueError('Invalid physical-station source grouping')
            q=world@source['w2c'][:3,:3].T+source['w2c'][:3,3];p=q@source['K'].T
            uv=p[...,:2]/np.maximum(p[...,2:],1e-12)
            finite=np.isfinite(uv).all(axis=-1);uv=np.nan_to_num(uv,nan=-2,posinf=-2,neginf=-2)
            inside=finite&(q[...,2]>0)&(uv[...,0]>=1)&(uv[...,0]<side-1)&(uv[...,1]>=1)&(uv[...,1]<side-1)
            sx=np.floor(np.clip(uv[...,0],0,side-1)).astype(int);sy=np.floor(np.clip(uv[...,1],0,side-1)).astype(int)
            source_z=source['depth_z'][sy,sx];source_valid=source['valid'][sy,sx]
            sample_uv=np.stack([sx+.5,sy+.5,np.ones_like(sx)],-1)
            source_world=_world(source,sample_uv,source_z)
            back=source_world@target['w2c'][:3,:3].T+target['w2c'][:3,3];bp=back@target['K'].T
            backuv=bp[...,:2]/np.maximum(bp[...,2:],1e-12)
            reprojection=np.sqrt((backuv[...,0]-(xx+.5))**2+(backuv[...,1]-(yy+.5))**2)
            source_rgb=cv2.remap(source['rgb'],(uv[...,0]-.5).astype(np.float32),(uv[...,1]-.5).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            photo=cv2.boxFilter(np.mean(np.abs(source_rgb-target['rgb']),axis=-1),-1,(3,3),normalize=True,borderType=cv2.BORDER_CONSTANT)
            ray2=world-source['c2w'][:3,3];cosine=np.sum(ray1*ray2,axis=-1)/np.maximum(norm1*np.linalg.norm(ray2,axis=-1),1e-15)
            parallax=np.degrees(np.arccos(np.clip(cosine,-1,1)))
            good=target['valid']&inside&source_valid&(back[...,2]>0)&(np.abs(source_z-q[...,2])<=options.maximum_relative_depth_error*q[...,2])&(reprojection<=options.maximum_roundtrip_pixels)&(photo<=options.maximum_rgb_patch_error)&(parallax>=options.minimum_ray_angle_degrees)
            group_good|=good
        count+=group_good;station_counts[station]=int(group_good.sum())
    accepted=target['valid']&(count>=options.minimum_other_station_support)&(target['confidence']>0)
    return accepted,count,station_counts


def validate_raw(raw,width):
    expected=erp_rays(width);shape=expected.shape[:2]
    radial=np.asarray(raw['radial_distance_model'],np.float32).squeeze()
    confidence=np.asarray(raw['confidence'],np.float32).squeeze()
    rays=np.asarray(raw['geometry_rays'],np.float32).squeeze()
    if rays.shape==(3,*shape):rays=np.moveaxis(rays,0,-1)
    if radial.shape!=shape or confidence.shape!=shape or rays.shape!=expected.shape:raise ValueError('Unexpected full-resolution UniSHARP array shapes')
    if not np.isfinite(radial).all() or not (radial>0).all() or not np.isfinite(rays).all():raise ValueError('Invalid raw UniSHARP range/rays')
    error=np.linalg.norm(rays-expected,axis=-1)
    if float(error.max())>2e-3:raise ValueError('UniSHARP geometry rays disagree with calibrated ERP convention')
    return dict(radial_distance_model=radial,geometry_rays=rays,confidence=confidence),float(error.max())


def validate_perspective_raw(raw,projection):
    h,w=projection['height'],projection['width'];K=np.asarray(projection['intrinsics_pixel_edge'],np.float64)
    yy,xx=np.indices((h,w));expected=np.stack([xx+.5,yy+.5,np.ones_like(xx)],-1)@np.linalg.inv(K).T
    expected/=np.linalg.norm(expected,axis=-1,keepdims=True)
    radial=np.asarray(raw['radial_distance_model'],np.float32).squeeze();confidence=np.asarray(raw['confidence'],np.float32).squeeze()
    rays=np.asarray(raw['geometry_rays'],np.float32).squeeze()
    if rays.shape==(3,h,w):rays=np.moveaxis(rays,0,-1)
    if radial.shape!=(h,w) or confidence.shape!=(h,w) or rays.shape!=expected.shape:raise ValueError('Unexpected perspective UniSHARP array shapes')
    if not np.isfinite(radial).all() or not (radial>0).all() or not np.isfinite(rays).all():raise ValueError('Invalid perspective UniSHARP range/rays')
    error=float(np.linalg.norm(rays-expected,axis=-1).max())
    if error>2e-3:raise ValueError('UniSHARP rays disagree with known pinhole camera')
    return dict(radial_distance_model=radial,geometry_rays=rays,confidence=confidence,projection='perspective',
        intrinsics_pixel_edge=K,camera_to_station_cv=np.asarray(projection['camera_to_station_cv'])),error


def model_assets(options):
    repo=Path(options.repo_path).resolve();checkpoint=Path(options.checkpoint_path).resolve();snapshot=Path(options.unik3d_snapshot_path).resolve()
    for path in [repo/'unisharp/models/unisharp_feature.py',repo/'UniK3D/unik3d/models/unik3d.py',checkpoint,snapshot/'model.safetensors',snapshot/'config.json']:
        if not path.is_file():raise ValueError('Missing configured UniSHARP asset: '+str(path))
    for path,expected in [(checkpoint,options.checkpoint_sha256),(snapshot/'model.safetensors',options.unik3d_sha256)]:
        if len(expected)!=64 or any(c not in '0123456789abcdef' for c in expected):raise ValueError('Exact trusted checkpoint SHA-256 must be configured')
        if _sha(path)!=expected:raise ValueError('Configured checkpoint hash mismatch: '+str(path))
    source_files={path.relative_to(repo).as_posix():_sha(path) for folder in [repo/'unisharp',repo/'UniK3D/unik3d'] for path in sorted(folder.rglob('*.py'))}
    source_hash=hashlib.sha256(json.dumps(source_files,sort_keys=True).encode()).hexdigest()
    return dict(repo_path=str(repo),checkpoint_path=str(checkpoint),checkpoint_sha256=options.checkpoint_sha256,
                unik3d_snapshot_path=str(snapshot),unik3d_sha256=options.unik3d_sha256,
                unik3d_config_sha256=_sha(snapshot/'config.json'),source_tree_sha256=source_hash,source_files=source_files)


class UniSharpInference:
    """Full official model, offline pretrained lookup, no renderer or fine-tuning."""
    def __init__(self,options,assets):
        if sys.platform=='win32':raise RuntimeError('Run UniSHARP inference on the configured Linux cloud host')
        import gc
        import torch
        if not torch.cuda.is_available():raise RuntimeError('Configured CUDA device unavailable')
        self.torch=torch;self.device=torch.device(options.device)
        repo=Path(assets['repo_path']);snapshot=Path(assets['unik3d_snapshot_path']);checkpoint=Path(assets['checkpoint_path'])
        if options.extra_python_path:
            extra=Path(options.extra_python_path).resolve()
            if not extra.is_dir():raise ValueError('Missing isolated extra_python_path')
            sys.path.insert(0,str(extra))
        sys.path[:0]=[str(repo),str(repo/'UniK3D')]
        # A long-running process cannot silently resolve another imported model.
        for name,root in [('unisharp',repo/'unisharp'),('unik3d',repo/'UniK3D/unik3d')]:
            module=sys.modules.get(name)
            if module is not None and (not getattr(module,'__file__',None) or not Path(module.__file__).resolve().is_relative_to(root)):
                raise RuntimeError('Already imported model package comes from another source tree: '+name)
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['WANDB_MODE']='disabled'
        from unik3d.models import UniK3D
        import unisharp.utils.unik3d_adapter as adapter
        original_loader=adapter.load_unik3d_model
        def local_loader(backbone='vitl',pretrained=True,device=None,cache_root=None):
            if backbone!='vitl' or not pretrained:raise ValueError('This verified adapter requires pretrained UniK3D ViT-L')
            model=UniK3D.from_pretrained(str(snapshot),local_files_only=True);model.eval();model.resolution_level=0
            return model.to(device or 'cpu')
        adapter.load_unik3d_model=local_loader
        try:
            from unisharp.models.unisharp_feature import UnisharpFeatureConfig,UnisharpFeatureModel
            # Only a user-configured checkpoint with its verified trusted hash
            # reaches this official pickle-compatible loader.
            payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if not isinstance(payload,dict):raise ValueError('Unexpected UniSHARP checkpoint payload')
            config=UnisharpFeatureConfig();merged=dict(payload.get('config',{}))
            for key in config.__dict__:
                if key in payload:merged[key]=payload[key]
            for key in config.__dict__:
                if key in merged:setattr(config,key,merged[key])
            if config.unik3d_backbone!='vitl' or config.unik3d_resolution_level!=0:raise ValueError('Unexpected backbone/resolution in configured checkpoint')
            step=int(payload.get('step',0));del payload;gc.collect()
            self.model=UnisharpFeatureModel(config).to(self.device)
            missing,unexpected=self.model.load_from_checkpoint(str(checkpoint),strict=False)
            if missing or set(unexpected)-{'payload.depth_alignment'}:raise ValueError('Unexpected UniSHARP checkpoint compatibility keys')
            self.model.eval();self.metadata=dict(config=asdict(config),checkpoint_step=step,missing_keys=list(missing),unexpected_keys=list(unexpected),torch_version=torch.__version__,cuda_version=torch.version.cuda,device=torch.cuda.get_device_name(self.device),forward=f'Official full UniSHARP {options.input_projection} forward; unshifted first surface only',depth_scope='UniK3D first surface inside UniSHARP; Gaussian shifted means/extra layers are not measured surfaces')
        finally:adapter.load_unik3d_model=original_loader

    def infer(self,rgb,intrinsics_pixel_edge=None):
        torch=self.torch;u8=torch.from_numpy(np.array(rgb,copy=True)).permute(2,0,1)[None].contiguous().to(self.device)
        intrinsics=None
        if intrinsics_pixel_edge is not None:
            # Official UniSHARP pinhole rays place centers at integer indices.
            # Our dataset K uses pixel edges, so preserve identical rays here.
            matrix=np.array(intrinsics_pixel_edge,np.float32,copy=True);matrix[0,2]-=.5;matrix[1,2]-=.5
            intrinsics=torch.as_tensor(matrix,device=self.device)[None]
        start=time.monotonic()
        with torch.inference_mode():
            result=self.model(image=u8.float()/255,image_u8=u8,camera_intrinsics=intrinsics,camera_params=None,camera_model='pinhole' if intrinsics is not None else 'spherical',depth_gt=None,distance_init_cap_m=None,return_aux=True)
            radial=result['unik3d_distance'].detach().float().cpu().numpy().squeeze()
            first=result['distance_layers'][:,0].detach().float().cpu().numpy().squeeze()
            if not np.array_equal(radial,first):raise ValueError('Official first-surface semantics changed')
            output=dict(radial_distance_model=radial,geometry_rays=result['geometry_rays'].detach().float().cpu().numpy().squeeze(),confidence=self.model.feature_extractor._unisharp_last_unik3d_output['confidence'].detach().float().cpu().numpy().squeeze())
        self.model.feature_extractor._unisharp_last_unik3d_output=None
        del result,u8
        return output,dict(seconds=time.monotonic()-start,first_layer_equals_unik3d=True,includes_predicted_gaussians=False)

    def close(self):
        import gc
        self.model.feature_extractor._unisharp_last_unik3d_output=None
        del self.model;gc.collect();self.torch.cuda.empty_cache()


def preflight(settings):
    """Read/hash cached weights and import dependencies; no model/CUDA allocation.

    This verifies cached container/header readability, not a successful neural
    forward pass. Root's separately scheduled GPU smoke supplies that evidence.
    """
    import importlib
    import zipfile
    settings=settings.get('depth_prior',settings) if isinstance(settings,dict) else settings
    options=settings if isinstance(settings,DepthPriorSettings) else DepthPriorSettings(**settings)
    assets=model_assets(options);repo=Path(assets['repo_path'])
    if options.extra_python_path:sys.path.insert(0,str(Path(options.extra_python_path).resolve()))
    sys.path[:0]=[str(repo),str(repo/'UniK3D')]
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['WANDB_MODE']='disabled'
    modules={}
    for name in ['numpy','scipy','PIL','cv2','torch','torchvision','huggingface_hub','safetensors','einops','yaml','timm','wandb','unik3d.models','unisharp.models.unisharp_feature']:
        module=importlib.import_module(name);modules[name]=dict(version=str(getattr(module,'__version__','not_exposed')),path=str(getattr(module,'__file__','namespace')))
    from safetensors import safe_open
    with safe_open(str(Path(assets['unik3d_snapshot_path'])/'model.safetensors'),framework='np') as archive:
        tensor_keys=list(archive.keys())
    if not tensor_keys:raise ValueError('Cached UniK3D tensor file is empty')
    with zipfile.ZipFile(assets['checkpoint_path']) as archive:
        members=archive.namelist()
        if not any(name.endswith('/data.pkl') for name in members):raise ValueError('Unexpected official PyTorch checkpoint container')
    return dict(status='imports_and_cached_weight_containers_verified',model_loaded=False,forward_run=False,
                assets=assets,modules=modules,unik3d_tensor_count=len(tensor_keys),checkpoint_zip_members=len(members))


def load_inputs(config,job_dir):
    dataset=job_dir/'sfm/dataset';sfm_path=_file(job_dir,'sfm/manifest.json');sfm=_read(sfm_path)
    if sfm.get('status')!='completed':raise ValueError('Depth requires a completed actual SfM stage')
    manifest_path=_file(dataset,'dataset_manifest.json');manifest=_read(manifest_path)
    if _sha(manifest_path)!=sfm.get('dataset_manifest_sha256'):raise ValueError('SfM dataset manifest binding mismatch')
    if manifest.get('coordinate_frame')!='EDN' or manifest.get('units')!='metres' or manifest.get('camera_convention')!='OpenGL_c2w':raise ValueError('Depth requires calibrated OpenGL cameras in EDN metres')
    if not {'transforms_train.json','transforms_heldout.json','sparse_depth_manifest.json','sparse_depth_observations.npz'}<=set(manifest.get('files',{})):raise ValueError('Dataset must hash-bind sparse observations and both transforms')
    bindings={str(sfm_path):_sha(sfm_path),str(manifest_path):_sha(manifest_path)}
    for name,value in manifest['files'].items():
        path=_file(dataset,name)
        if _sha(path)!=value:raise ValueError('Dataset file hash mismatch: '+name)
        bindings[str(path)]=value
    frames=[];seen=set();groups={}
    for split in ['train','heldout']:
        for source in _read(_file(dataset,'transforms_'+split+'.json'))['frames']:
            frame=dict(source,split=split);name=frame['file_path'];station=str(frame['station_id'])
            if name in seen:raise ValueError('Duplicate transform frame')
            seen.add(name);groups.setdefault(station,set()).add(split);camera(frame)
            for key,hash_key in [('file_path','image_sha256'),('sfm_mask_path','sfm_mask_sha256')]:
                path=_file(dataset,frame[key]);actual=_sha(path)
                if actual!=frame.get(hash_key):raise ValueError('Depth RGB/foreground input hash mismatch')
                bindings[str(path)]=actual
            frames.append(frame)
    if not frames or any(len(splits)!=1 for splits in groups.values()):raise ValueError('Empty dataset or physical station split leakage')
    if {str(frame['pano_id']) for frame in frames}!=set(config.get('panorama_ids',[])):raise ValueError('Depth dataset panorama roster differs from frozen selection')
    train={station for station,splits in groups.items() if 'train' in splits};heldout=set(groups)-train
    if train!=set(map(str,manifest['training_station_ids'])) or heldout!=set(map(str,manifest['heldout_station_ids'])):raise ValueError('Dataset station split binding mismatch')
    by_capture={}
    for frame in frames:by_capture.setdefault(str(frame['pano_id']),[]).append(frame)
    for capture_frames in by_capture.values():
        if len(capture_frames)!=6 or len({str(f['face']) for f in capture_frames})!=6 or len({str(f['station_id']) for f in capture_frames})!=1:raise ValueError('Depth needs complete six-face capture rigs with consistent station groups')
        poses=[camera(frame)[2]@np.linalg.inv(_rotation(frame['camera_to_station_cv'])) for frame in capture_frames]
        if not all(np.allclose(pose,poses[0],atol=1e-6) for pose in poses):raise ValueError('Cube camera poses disagree with their shared capture rig')
    sidecar=_read(_file(dataset,'sparse_depth_manifest.json'))
    required=dict(coordinate_frame='EDN',units='metres',depth_convention='camera_z',pixel_center_offset=.5,observation_kind='actual_sfm_tracks',geometry_scope='transductive_shared_sfm')
    if any(sidecar.get(key)!=value for key,value in required.items()):raise ValueError('Unsupported sparse-depth observation convention')
    if sidecar.get('status') not in {'supported_observations','insufficient_observations'}:raise ValueError('Sparse-depth sidecar status is unassessed')
    for split in ['train','heldout']:
        if sidecar.get('transforms_'+split+'_sha256')!=_sha(dataset/('transforms_'+split+'.json')):raise ValueError('Sparse-depth transform hash mismatch')
    if set(map(str,sidecar['train_station_ids']))!=train or set(map(str,sidecar['heldout_station_ids']))!=heldout:raise ValueError('Sparse-depth station split mismatch')
    path=_file(dataset,sidecar['npz'])
    if _sha(path)!=sidecar.get('sha256'):raise ValueError('Sparse-depth NPZ hash mismatch')
    with np.load(path,allow_pickle=False) as archive:observations={key:archive[key] for key in archive.files}
    count=len(observations['depth_z'])
    if any(len(observations[key])!=count for key in ['frame_name','station_id','split','point_id','xy','support_station_count','triangulation_angle_degrees','reprojection_error_px']) or observations['xy'].shape!=(count,2):raise ValueError('Malformed actual-observation table')
    for key in ['xy','depth_z','support_station_count','triangulation_angle_degrees','reprojection_error_px']:
        if not np.isfinite(observations[key]).all():raise ValueError('Non-finite actual-observation data')
    if (observations['depth_z']<=0).any() or (observations['reprojection_error_px']<0).any():raise ValueError('Invalid observed depth/reprojection error')
    lookup={frame['file_path']:frame for frame in frames}
    for name,station,split in set(zip(observations['frame_name'].tolist(),observations['station_id'].tolist(),observations['split'].tolist())):
        if name not in lookup or station!=str(lookup[name]['station_id']) or split!=lookup[name]['split']:raise ValueError('Sparse-depth row is not bound to its frame/station/split')
    return dataset,frames,by_capture,observations,manifest,bindings


def _signature_fingerprint(signature):
    value=dict(signature);value['settings']={key:item for key,item in signature['settings'].items() if key not in {'output_dir','resume_raw'}}
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def _raw_fingerprint(signature,pid,erp,projection=None):
    value=dict(input_fingerprint=_signature_fingerprint(signature),pano_id=pid,erp_rgb_sha256=hashlib.sha256(np.ascontiguousarray(erp).tobytes()).hexdigest())
    if projection is not None:value['projection']=projection
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def load_cached_raw(output,token,pid,erp,signature,width,projection=None):
    """Reuse only committed per-capture hashes; legacy unbound NPZs are rejected."""
    path=output/'raw'/f'{token}.npz';record_path=output/'raw'/f'{token}.json'
    if not record_path.is_file():raise ValueError('Raw NPZ has no committed per-capture fingerprint; preserve it and choose a new output_dir')
    record=_read(record_path)
    if record.get('pano_id')!=pid or record.get('raw_input_fingerprint')!=_raw_fingerprint(signature,pid,erp,projection):raise ValueError('Raw cache input/model fingerprint mismatch')
    if record.get('sha256')!=_sha(path):raise ValueError('Raw cache NPZ hash mismatch')
    erp_path=_file(output,record['erp'])
    if record.get('erp_sha256')!=_sha(erp_path) or record.get('raw_provenance_sha256')!=_sha(output/'raw/provenance.json'):raise ValueError('Raw cache ERP/provenance hash mismatch')
    with Image.open(erp_path) as old:
        if not np.array_equal(np.asarray(old.convert('RGB')),erp):raise ValueError('Rebuilt six-face ERP differs from cached input')
    with np.load(path,allow_pickle=False) as archive:
        if str(archive['pano_id'])!=pid or str(archive['depth_convention'])!='radial_range' or str(archive['units'])!='uncalibrated_model_metric_prediction':raise ValueError('Raw cache metadata mismatch')
        raw={key:archive[key] for key in ['radial_distance_model','geometry_rays','confidence']}
    raw,error=validate_perspective_raw(raw,projection) if projection is not None else validate_raw(raw,width)
    return raw,dict(record,reused_raw=True,geometry_ray_max_error=error)


def run(config,job_dir,settings=None):
    """Produce diagnostics only. No dataset, image, mask, or PLY mutations."""
    settings=settings or {};settings=settings.get('depth_prior',settings) if isinstance(settings,dict) else settings
    options=settings if isinstance(settings,DepthPriorSettings) else DepthPriorSettings(**settings)
    job_dir=Path(job_dir).resolve();output=job_dir/options.output_dir
    if output.exists() and not options.resume_raw:raise FileExistsError('Depth-prior output exists; configure resume_raw for verified cache or a new output_dir')
    dataset,frames,captures,observations,dataset_manifest,bindings=load_inputs(config,job_dir)
    observations,dedup_report=deduplicate_observations(observations)
    assets=model_assets(options);started=time.monotonic();model=None
    signature=dict(job_config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True,allow_nan=False).encode()).hexdigest(),settings=asdict(options),inputs_sha256=bindings,models=assets)
    if output.exists():
        previous=_read(output/'input_manifest.json')
        if _signature_fingerprint(previous)!=_signature_fingerprint(signature):raise ValueError('Existing depth-prior input/model/settings fingerprint differs; no outputs were overwritten')
    else:output.mkdir()
    _save(output/'input_manifest.json',signature)
    _save(output/'observation_deduplication.json',dedup_report)
    try:
        calibrated=[];calibration_reports=[];raw_entries=[]
        for completed,(pid,capture_frames) in enumerate(sorted(captures.items()),1):
            if options.input_projection=='perspective':
                inputs=[]
                for frame in sorted(capture_frames,key=lambda item:item['file_path']):
                    rgb,projection=perspective_input(frame,dataset,options.perspective_side)
                    inputs.append(([frame],rgb,projection,projection,_token(frame['file_path'])))
            else:
                erp,erp_report=assemble_erp(capture_frames,dataset,options.erp_width)
                inputs=[(capture_frames,erp,erp_report,None,_token(pid))]
            for input_frames,rgb,input_report,projection,token in inputs:
                image_path=output/('perspective' if projection else 'erp')/f'{token}.png';raw_path=output/'raw'/f'{token}.npz'
                if raw_path.exists():
                    if not options.resume_raw:raise ValueError('Raw output unexpectedly exists')
                    raw,raw_entry=load_cached_raw(output,token,pid,rgb,signature,options.erp_width,projection)
                else:
                    if (output/'raw'/f'{token}.json').exists():raise ValueError('Committed raw record exists but its NPZ is missing')
                    if image_path.exists():
                        with Image.open(image_path) as prior:
                            if not np.array_equal(np.asarray(prior.convert('RGB')),rgb):raise ValueError('Existing model image differs from rebuilt calibrated input')
                    else:image_path.parent.mkdir(exist_ok=True);Image.fromarray(rgb).save(image_path)
                    if model is None:
                        model=UniSharpInference(options,assets);provenance=dict(assets,**model.metadata);provenance_path=output/'raw/provenance.json'
                        if provenance_path.exists() and _read(provenance_path)!=provenance:raise ValueError('Current runtime/model provenance differs from cached raw inference')
                        _save(provenance_path,provenance)
                    if projection is not None:
                        raw,timing=model.infer(rgb,intrinsics_pixel_edge=projection['intrinsics_pixel_edge'])
                        raw,ray_error=validate_perspective_raw(raw,projection)
                    else:
                        raw,timing=model.infer(rgb);raw,ray_error=validate_raw(raw,options.erp_width)
                    np.savez_compressed(raw_path,**raw,pano_id=np.asarray(pid),depth_convention=np.asarray('radial_range'),units=np.asarray('uncalibrated_model_metric_prediction'))
                    raw_entry=dict(pano_id=pid,station_id=str(capture_frames[0]['station_id']),erp=image_path.relative_to(output).as_posix(),erp_sha256=_sha(image_path),npz=raw_path.relative_to(output).as_posix(),sha256=_sha(raw_path),geometry_ray_max_error=ray_error,raw_input_fingerprint=_raw_fingerprint(signature,pid,rgb,projection),raw_provenance_sha256=_sha(output/'raw/provenance.json'),reused_raw=False,**input_report,**timing)
                    _save(output/'raw'/f'{token}.json',raw_entry)
                raw_entries.append(raw_entry)
                for frame in input_frames:
                    field,report=calibrated_field(frame,dataset,raw,observations,options);calibrated.append(field);calibration_reports.append(report)
            print(json.dumps(dict(stage='depth_prior',phase='inference',input_projection=options.input_projection,completed_captures=completed,total_captures=len(captures),completed_model_images=len(raw_entries))),flush=True)
        if model is not None:model.close();model=None
        _save(output/'calibration_report.json',dict(model='per_face_multiplicative_scale',fit_policy='One actual TRAIN row per frame/point; globally disjoint point-ID fit/test; no model selection on test',deduplication_report_sha256=_sha(output/'observation_deduplication.json'),faces=calibration_reports))
        entries=[];(output/'validated').mkdir(exist_ok=True)
        for index,field in enumerate(calibrated):
            sources=select_sources(field,calibrated,options);valid,count,station_counts=multiview_support(field,sources,options)
            frame=field['frame'];token=_token(frame['file_path']);path=output/'validated'/f'{token}.npz'
            np.savez_compressed(path,depth_z=np.where(valid,field['depth_z'],0).astype(np.float32),valid=valid,confidence=np.where(valid,field['confidence'],0).astype(np.float32),source_count=np.where(valid,count,0).astype(np.uint16),candidate_depth_z=field['depth_z'],candidate_valid=field['valid'],K=field['K'],camera_from_world=field['w2c'],world_frame=np.asarray('EDN'),units=np.asarray('metres'),depth_convention=np.asarray('camera_z'),pixel_center_offset=np.float32(.5),frame_name=np.asarray(frame['file_path']),station_id=np.asarray(str(frame['station_id'])),pano_id=np.asarray(str(frame['pano_id'])),split=np.asarray(frame['split']),evidence=np.asarray('calibrated_UniSHARP_first_surface_multistation_consistency'))
            entries.append(dict(frame_name=frame['file_path'],station_id=str(frame['station_id']),pano_id=str(frame['pano_id']),face=str(frame['face']),split=frame['split'],npz=path.relative_to(output).as_posix(),sha256=_sha(path),valid_count=int(valid.sum()),candidate_count=int(field['valid'].sum()),support_by_other_station=station_counts,width=options.output_side,height=options.output_side,K=field['K'].tolist(),camera_from_world=field['w2c'].tolist()))
            entries[-1].update(status='accepted' if valid.any() else 'insufficient_support',image_sha256=frame['image_sha256'],sfm_mask_sha256=frame['sfm_mask_sha256'])
            print(json.dumps(dict(stage='depth_prior',phase='multiview_validation',completed_frames=index+1,total_frames=len(calibrated),accepted_pixels=int(valid.sum()))),flush=True)
        if any(_sha(path)!=value for path,value in bindings.items()):raise ValueError('Inputs changed during depth-prior generation')
        accepted=sum(row['valid_count'] for row in entries);face_counts={}
        for row in entries:
            face_counts.setdefault(row['face'],dict(frames=0,accepted_pixels=0,accepted_frames=0));face_counts[row['face']]['frames']+=1;face_counts[row['face']]['accepted_pixels']+=row['valid_count'];face_counts[row['face']]['accepted_frames']+=int(row['valid_count']>0)
        manifest=dict(schema_version=1,status='complete',stage='depth_prior',validation_status='multistation_consistent_subset' if accepted else 'insufficient_supported_depth',training_integration_status='not_attached_to_dataset_or_trainer',coordinate_frame='EDN',units='metres',depth_convention='camera_z',pixel_center_offset=.5,camera_convention='OpenCV_world_to_camera',geometry_scope='transductive_shared_sfm_calibration',minimum_other_station_support=options.minimum_other_station_support,train_station_ids=dataset_manifest['training_station_ids'],heldout_station_ids=dataset_manifest['heldout_station_ids'],metric_alignment=dataset_manifest.get('metric_alignment'),accepted_pixels=accepted,face_counts=face_counts,raw_entries=raw_entries,entries=entries,settings=asdict(options),elapsed_seconds=time.monotonic()-started,input_manifest_sha256=_sha(output/'input_manifest.json'),calibration_report_sha256=_sha(output/'calibration_report.json'),provenance_sha256=_sha(output/'raw/provenance.json'),source_transforms_sha256={split:_sha(dataset/('transforms_'+split+'.json')) for split in ['train','heldout']},limitations=['Model/calibration confidence is not surveyed ground truth.','Only training captures supply calibration anchors or multiview validation sources; heldout raw maps are diagnostic and not calibrated.','No finite sky, floor plane, interpolated sparse anchors, or rejected learned depth is supplied as truth.','Dataset remains unchanged; candidate_depth_z/candidate_valid are diagnostics and must never be used as accepted supervision.'])
        manifest.update(input_projection=options.input_projection,source_dataset_files_sha256={Path(path).relative_to(dataset).as_posix():value for path,value in bindings.items() if Path(path).is_relative_to(dataset)},observation_deduplication_sha256=_sha(output/'observation_deduplication.json'),observation_deduplication_summary={key:dedup_report[key] for key in ['input_rows','output_rows','duplicate_groups','removed_rows','conflicting_groups_rejected']},reused_raw_captures=len({row['pano_id'] for row in raw_entries if row['reused_raw']}),new_inference_captures=len({row['pano_id'] for row in raw_entries if not row['reused_raw']}),reused_raw_images=sum(int(row['reused_raw']) for row in raw_entries),new_inference_images=sum(int(not row['reused_raw']) for row in raw_entries),output_path=output.relative_to(job_dir).as_posix())
        _save(output/'manifest.json',manifest);return manifest
    except Exception as error:
        _save(output/'failure.json',dict(status='failed',reason=str(error),error_type=type(error).__name__,elapsed_seconds=time.monotonic()-started));raise
    finally:
        if model is not None:model.close()
