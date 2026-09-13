"""DA3 image depth used only to abstain from uncertain sky-removal votes.

Two isolated processes preserve the accepted pipeline's two pinned source
versions: nested metric-branch sky scores, then DA3-LARGE-1.1 known-pose depth.
Physical camera baseline alignment is not validated per-pixel metric evidence.
No Gaussian prediction, depth training loss, or point deletion occurs here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from .export import sha256 as sha, write_json


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def validate_assets(settings):
    options = settings.get('da3_raw_depth', settings)
    if not isinstance(options, dict) or set(options)-{'pose', 'metric', 'side', 'gpu_lock', 'extra_python'}:
        raise ValueError('Invalid raw DA3 operator settings')
    side = options.get('side', 504)
    if type(side) is not int or side != 504:
        raise ValueError('The accepted raw-depth abstention recipe uses exactly 504 pixels')
    result = dict(options, side=side)
    for mode in ('metric', 'pose'):
        spec = options.get(mode)
        required = {'repo_path', 'model_path', 'repo_revision', 'model_sha256', 'config_sha256', 'model_name'}
        if not isinstance(spec, dict) or set(spec) != required:
            raise ValueError('Both pinned metric and pose model assets are required')
        repo, model = Path(spec['repo_path']), Path(spec['model_path'])
        if not repo.is_absolute() or not model.is_absolute() or not (repo/'src/depth_anything_3/api.py').is_file():
            raise ValueError('Missing configured DA3 source tree')
        for file, key in [(model/'model.safetensors', 'model_sha256'), (model/'config.json', 'config_sha256')]:
            if not file.is_file() or sha(file) != spec[key]:
                raise ValueError('Missing or changed pinned DA3 '+mode+' asset')
        revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        if revision != spec['repo_revision']:
            raise ValueError('DA3 source revision differs: '+mode)
        changes = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--', 'src/depth_anything_3'], text=True).strip()
        if changes:
            raise ValueError('DA3 tracked source tree has local changes: '+mode)
        config = _read(model/'config.json')
        if config.get('model_name') != spec['model_name'] or ('metric' not in config.get('config', {}) and mode == 'metric'):
            raise ValueError('DA3 branch/configuration differs')
    if options.get('extra_python') and not Path(options['extra_python']).is_dir():
        raise ValueError('Configured DA3 dependency directory is missing')
    return result


def capability(settings):
    try:
        if not sys.platform.startswith('linux'):
            raise ValueError('Raw DA3 inference runs on the configured Linux cloud host')
        options = validate_assets(settings)
        return dict(available=True, side=options['side'], depth_role='heuristic_abstention_only', metric_calibration_accepted=False)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return dict(available=False, reason=str(error))


def _frames(dataset, camera_json):
    doc = _read(camera_json)
    if (doc.get('coordinate_frame') != 'EDN' or doc.get('units') not in ('metres', 'meters', 'm')
            or doc.get('camera_convention') != 'OpenGL_c2w'):
        raise ValueError('Raw-depth cameras require explicit OpenGL c2w in EDN metres')
    frames, seen = [], {}
    for raw in doc.get('frames', []):
        frame = dict({k:doc[k] for k in ('w', 'h', 'fl_x', 'fl_y', 'cx', 'cy') if k in doc}, **raw)
        if frame.get('station_id') is None or isinstance(frame['station_id'], bool) or not str(frame['station_id']):
            raise ValueError('Physical camera station IDs required')
        frame['station_id'] = str(frame['station_id'])
        identity = (frame['station_id'], str(frame.get('face')), frame['file_path'])
        if identity in seen:
            if seen[identity] != frame:
                raise ValueError('Conflicting repeated camera frame')
            continue
        seen[identity] = dict(frame)
        for key, hashkey in [('file_path', 'image_sha256'), ('original_valid_mask_path', 'original_valid_mask_sha256'),
                             ('edit_alpha_path', 'edit_alpha_sha256')]:
            rawpath = frame.get(key)
            if not isinstance(rawpath, str) or Path(rawpath).is_absolute():
                raise ValueError('Required bound relative DA3 input: '+key)
            path = (dataset/rawpath).resolve()
            if not path.is_relative_to(dataset) or not path.is_file() or sha(path) != frame.get(hashkey):
                raise ValueError('Changed or absent raw-depth input: '+key)
        if frame['w'] != frame['h'] or type(frame['w']) is not int or frame['w'] <= 0:
            raise ValueError('Accepted raw-depth workflow expects square perspective cube faces')
        with Image.open(dataset/frame['file_path']) as image:
            if image.size != (frame['w'], frame['h']):
                raise ValueError('Registered cube and image dimensions differ')
        for key in ('fl_x', 'fl_y', 'cx', 'cy'):
            value = frame.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('Finite fixed camera intrinsics required')
        if min(frame['fl_x'], frame['fl_y']) <= 0 or any(abs(float(frame.get(k, 0))) > 1e-8 for k in ('k1', 'k2', 'p1', 'p2')):
            raise ValueError('Undistorted positive-focal pinhole cameras required')
        with Image.open(dataset/frame['original_valid_mask_path']) as mask:
            if mask.mode != 'L' or mask.size != (frame['w'], frame['h']) or not np.isin(np.asarray(mask), [0, 255]).all():
                raise ValueError('Original binary validity mask differs from native image grid')
        alpha = np.load(dataset/frame['edit_alpha_path'], allow_pickle=False)
        if alpha.shape != (frame['h'], frame['w']) or not np.isfinite(alpha).all() or (alpha < 0).any() or (alpha > 1).any():
            raise ValueError('Original edit alpha differs from native image grid/range')
        matrix = np.asarray(frame['transform_matrix'], float)
        if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
                or not np.allclose(matrix[:3, :3].T@matrix[:3, :3], np.eye(3), atol=1e-5, rtol=0)
                or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5, rtol=0)):
            raise ValueError('Invalid rigid camera transform')
        frames.append(frame)
    groups = {}
    for frame in frames:
        groups.setdefault(frame['station_id'], []).append(frame)
    if len(groups) < 3:
        raise ValueError('DA3 depth abstention needs at least three physical stations')
    centers = np.array([np.asarray(fs[0]['transform_matrix'])[:3, 3] for fs in groups.values()])
    radius = float(np.linalg.norm(centers-centers.mean(0), axis=1).max())
    if radius <= 0:
        raise ValueError('Degenerate physical camera baselines')
    for fs in groups.values():
        origin = np.asarray(fs[0]['transform_matrix'])[:3, 3]
        if any(np.linalg.norm(np.asarray(f['transform_matrix'])[:3, 3]-origin) > radius*1e-6 for f in fs):
            raise ValueError('Views grouped as one physical station have different centers')
        if len({str(f['face']) for f in fs}) != len(fs):
            raise ValueError('Duplicate same-face physical camera input')
    return frames


def evidence_mask(depth, metric_sky_score, original_valid, alpha, side=504):
    """Preserve exact accepted 0.3 sky, >0 edit and 0.12 depth-edge guards."""
    import cv2
    from scipy.ndimage import maximum_filter, minimum_filter
    depth, sky = np.asarray(depth), np.asarray(metric_sky_score)
    if depth.shape != (side, side) or sky.shape != depth.shape or np.asarray(alpha).shape != np.asarray(original_valid).shape:
        raise ValueError('Raw-depth evidence grids differ')
    if np.asarray(alpha).ndim != 2 or not np.isfinite(alpha).all() or (np.asarray(alpha) < 0).any() or (np.asarray(alpha) > 1).any():
        raise ValueError('Edit alpha must be finite in [0, 1]')
    generic = cv2.resize((np.asarray(original_valid)>0).astype(np.float32), (side, side), interpolation=cv2.INTER_AREA) >= 1-1e-6
    edited = cv2.resize(np.asarray(alpha, np.float32), (side, side), interpolation=cv2.INTER_AREA) > 0
    finite = np.isfinite(depth)&(depth>0)&np.isfinite(sky)
    valid = finite&(sky<.3)&generic
    edge = (maximum_filter(depth, size=3)-minimum_filter(depth, size=3))/np.maximum(depth, 1e-8)
    return valid&~edited&(edge<=.12), dict(valid=valid, generated_edit_mask=edited, generic_valid=generic,
                                         relative_depth_edge=edge.astype(np.float32))


def verify_depth_field(path, frame, side):
    with np.load(path, allow_pickle=False) as field:
        z, valid = field['depth_z'], field['evidence_valid']
        if z.shape != (side, side) or valid.shape != z.shape or valid.dtype != bool:
            raise ValueError('Raw depth field dimensions or validity differ')
        if not np.isfinite(z[valid]).all() or (z[valid] <= 0).any() or bool(field['metric_calibration_accepted']):
            raise ValueError('Raw depth evidence is invalid or falsely marked metrically calibrated')
        expected = dict(source_image_sha256=frame['image_sha256'], station_id=frame['station_id'],
                        image=frame['file_path'], world_frame='EDN', unit='metres', depth_convention='camera_z')
        if any(str(field[key]) != value for key, value in expected.items()) or float(field['pixel_center_offset']) != .5:
            raise ValueError('Raw depth image identity or coordinate semantics differ')
        K = np.array([[frame['fl_x']*side/frame['w'], 0, frame['cx']*side/frame['w']],
                      [0, frame['fl_y']*side/frame['h'], frame['cy']*side/frame['h']], [0, 0, 1]], float)
        view = np.linalg.inv(np.asarray(frame['transform_matrix'])@np.diag([1., -1., -1., 1.]))
        if not np.allclose(field['K'], K, rtol=0, atol=1e-8) or not np.allclose(field['camera_from_world'], view, rtol=0, atol=1e-8):
            raise ValueError('Raw depth pose or pixel-grid calibration differs')


def baseline_scale(predicted_views,known_views,stations):
    """Scale-only physical baseline fit remains defined for collinear streets."""
    def centers(views):
        v=np.asarray(views,np.float64)
        return -np.einsum('nji,nj->ni',v[:,:3,:3],v[:,:3,3])
    pc=centers(predicted_views);kc=centers(known_views);stations=np.asarray(stations)
    names=sorted(set(stations));p=np.asarray([np.mean(pc[stations==s],axis=0) for s in names]);k=np.asarray([np.mean(kc[stations==s],axis=0) for s in names])
    pairs=[]
    for i in range(len(names)):
        for j in range(i):
            pd=float(np.linalg.norm(p[i]-p[j]));kd=float(np.linalg.norm(k[i]-k[j]))
            if pd>1e-8 and kd>1e-8:pairs.append(dict(a=names[j],b=names[i],known_m=kd,predicted_units=pd,depth_multiplier=kd/pd))
    if len(pairs)<2:raise ValueError('Insufficient distinct physical baselines for camera scale')
    mul=float(np.median([x['depth_multiplier'] for x in pairs]))
    return mul,dict(method='Median ratio of known/predicted inter-station distances, six cube faces collapsed into one physical camera center',
        physical_baseline_pairs=pairs,ratio_min=min(x['depth_multiplier'] for x in pairs),ratio_max=max(x['depth_multiplier'] for x in pairs),depth_multiplier=mul,
        known_camera_rank=int(np.linalg.matrix_rank(k-k.mean(0))),predicted_same_station_max_spread=float(max(np.linalg.norm(pc[stations==s]-p[i],axis=1).max() for i,s in enumerate(names))))

def pose_group(network,processor,roster,station,dataset,side,torch):
    """Official camera-conditioned forward + physical camera-baseline scale fit."""
    from depth_anything_3.api import DepthAnything3
    groups={}
    for f in roster:groups.setdefault(f['station_id'],[]).append(f)
    center={s:np.asarray(fs[0]['transform_matrix'])[:3,3] for s,fs in groups.items()}
    nearest=sorted(groups,key=lambda s:(np.linalg.norm(center[s]-center[station]),s))[:3]
    fs=sorted([f for s in nearest for f in groups[s]],key=lambda f:(nearest.index(f['station_id']),f['face']))
    images=[];Ks=[];views=[]
    for f in fs:
        p=dataset/f['file_path'];assert sha(p)==f['image_sha256']
        with Image.open(p) as im:images.append(np.asarray(im.convert('RGB').resize((side,side),Image.Resampling.LANCZOS)))
        Ks.append([[f['fl_x']*side/f['w'],0,f['cx']*side/f['w']-.5],[0,f['fl_y']*side/f['h'],f['cy']*side/f['h']-.5],[0,0,1]])
        views.append(np.linalg.inv(np.asarray(f['transform_matrix'])@np.diag([1.,-1.,-1.,1.])))
    x,ext,K=processor(images,extrinsics=np.asarray(views,np.float32),intrinsics=np.asarray(Ks,np.float32),process_res=side,sequential=True)
    norm=DepthAnything3._normalize_extrinsics(None,ext[None].cuda().clone())
    with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
        pred=network(x[None].cuda(),norm,K[None].cuda(),infer_gs=False,ref_view_strategy='saddle_balanced')
        if getattr(network,'_unused_auxiliary_keys',None) and not getattr(network,'_aux_probe_done',False):
            old={key:network.get_parameter(key).clone() for key in network._unused_auxiliary_keys}
            for key in old:network.get_parameter(key).fill_(7)
            probe=network(x[None].cuda(),norm,K[None].cuda(),infer_gs=False,ref_view_strategy='saddle_balanced')
            for key in ('depth','depth_conf','extrinsics','intrinsics'):
                assert torch.equal(pred[key],probe[key]),f'Unused auxiliary weights unexpectedly change {key}'
            for key,value in old.items():network.get_parameter(key).copy_(value)
            network._aux_probe_done=True
    depth=pred.depth[0].detach().float().cpu().numpy();conf=pred.depth_conf[0].detach().float().cpu().numpy();extout=pred.extrinsics[0].detach().float().cpu().numpy()
    mul,alignment=baseline_scale(extout,np.asarray(views),[f['station_id'] for f in fs])
    assert np.isfinite(mul) and mul>0
    if depth.ndim==4:depth=depth[...,0]
    assert depth.shape==(len(fs),side,side)
    metadata=dict(target_station=station,other_input_stations=[s for s in nearest if s!=station],frame_names=[f['file_path'] for f in fs],
        input_image_sha256=[f['image_sha256'] for f in fs],baseline_alignment=alignment,unused_auxiliary_weights_invariance_verified=bool(getattr(network,'_aux_probe_done',False)),mode='Known-pose 3 physical stations x 6 cube faces; camera-baseline-aligned DA3 any-view branch only')
    return {f['file_path']:(depth[i]*mul,conf[i]) for i,f in enumerate(fs)},metadata


def _load_model(spec, mode, extra_python):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.path[:0] = [str(Path(spec['repo_path'])/'src')] + ([extra_python] if extra_python else [])
    import torch
    from safetensors import safe_open
    from omegaconf import OmegaConf
    from depth_anything_3.cfg import create_object
    from depth_anything_3.utils.io.input_processor import InputProcessor
    cfg = _read(Path(spec['model_path'])/'config.json')
    nested = 'anyview' in cfg['config']
    if mode == 'metric' and not nested:
        raise ValueError('Metric sky guard requires the pinned nested metric branch')
    network = create_object(OmegaConf.create(cfg['config']['metric' if mode == 'metric' else 'anyview'] if nested else cfg['config']))
    prefix = ('model.da3_metric.' if mode == 'metric' else 'model.da3.') if nested else 'model.'
    with safe_open(Path(spec['model_path'])/'model.safetensors', framework='pt', device='cpu') as archive:
        weights = {k[len(prefix):]:archive.get_tensor(k) for k in archive.keys() if k.startswith(prefix)}
    if mode == 'pose':
        expected = network.state_dict()
        missing = set(expected)-set(weights)
        allowed = {f'head.scratch.output_conv2_aux.{level}.2.{kind}' for level in (1, 2, 3) for kind in ('weight', 'bias')}
        if (missing and missing != allowed) or set(weights)-set(expected):
            raise ValueError('Unexpected DA3 checkpoint tensor mismatch')
        for key in missing:
            weights[key] = torch.ones_like(expected[key]) if key.endswith('weight') and expected[key].ndim == 1 else torch.zeros_like(expected[key])
        network._unused_auxiliary_keys = sorted(missing)
    network.load_state_dict(weights, strict=True)
    del weights
    return network.eval().cuda(), InputProcessor(), torch


def _worker(config_path, mode):
    config = _read(config_path)
    options = config['options']
    dataset, camera = Path(config['dataset']), Path(config['camera_json'])
    frames = _frames(dataset, camera)
    if sha(camera) != config['cameras_sha256']:
        raise ValueError('Camera JSON changed before model inference')
    import fcntl
    lock = Path(options['gpu_lock']).open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        validate_assets(options)
        network, processor, torch = _load_model(options[mode], mode, options.get('extra_python'))
        torch.set_num_threads(8); torch.manual_seed(0); np.random.seed(0)
        output, side = Path(config['output']), options['side']
        records, groups = [], []
        names = {f['file_path']:f'frame_{i:06d}' for i, f in enumerate(frames)}
        if mode == 'metric':
            for frame in frames:
                with Image.open(dataset/frame['file_path']) as im:
                    rgb = np.asarray(im.convert('RGB').resize((side, side), Image.Resampling.LANCZOS))
                K = np.array([[frame['fl_x']*side/frame['w'], 0, frame['cx']*side/frame['w']-.5],
                              [0, frame['fl_y']*side/frame['h'], frame['cy']*side/frame['h']-.5], [0, 0, 1]], np.float32)
                x, _, processed = processor([rgb], intrinsics=K[None], process_res=side, sequential=True)
                if tuple(x.shape) != (1, 3, side, side) or not np.allclose(processed.numpy()[0], K):
                    raise ValueError('Metric sky model unexpectedly changed the pixel grid')
                with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    pred = network(x[None].cuda(), infer_gs=False)
                sky = pred.sky.detach().float().cpu().numpy().reshape(side, side)
                target = output/'metric'/(names[frame['file_path']]+'.npz')
                np.savez_compressed(target, raw_sky_score=sky, source_image_sha256=frame['image_sha256'])
                records.append(dict(image=frame['file_path'], path=str(target), sha256=sha(target)))
                print(json.dumps(dict(mode=mode, completed=len(records), total=len(frames))), flush=True)
                del pred, x
            write_json(output/'metric_manifest.json', dict(status='completed', frames=records, model=options['metric']))
        else:
            metric_manifest = _read(output/'metric_manifest.json')
            metric = {row['image']:row for row in metric_manifest['frames']}
            for station in sorted({f['station_id'] for f in frames}):
                predictions, group = pose_group(network, processor, frames, station, dataset, side, torch)
                groups.append(group)
                for frame in [f for f in frames if f['station_id'] == station]:
                    z, confidence = predictions[frame['file_path']]
                    metric_row = metric[frame['file_path']]
                    if sha(metric_row['path']) != metric_row['sha256']:
                        raise ValueError('Pinned metric sky output changed')
                    with np.load(metric_row['path'], allow_pickle=False) as data:
                        sky = data['raw_sky_score'].copy()
                        if str(data['source_image_sha256']) != frame['image_sha256']:
                            raise ValueError('Metric sky guard was inferred from a different RGB image')
                    with Image.open(dataset/frame['original_valid_mask_path']) as image:
                        valid = np.asarray(image.convert('L'))
                    alpha = np.load(dataset/frame['edit_alpha_path'], allow_pickle=False)
                    evidence, detail = evidence_mask(z, sky, valid, alpha, side)
                    K = np.array([[frame['fl_x']*side/frame['w'], 0, frame['cx']*side/frame['w']],
                                  [0, frame['fl_y']*side/frame['h'], frame['cy']*side/frame['h']], [0, 0, 1]], float)
                    view = np.linalg.inv(np.asarray(frame['transform_matrix'])@np.diag([1., -1., -1., 1.]))
                    target = output/'raw'/(names[frame['file_path']]+'.npz')
                    np.savez_compressed(target, depth_z=np.asarray(z, np.float32), evidence_valid=evidence, K=K,
                        camera_from_world=view, source_image_sha256=frame['image_sha256'], station_id=station,
                        image=frame['file_path'], face=frame['face'], world_frame='EDN', unit='metres',
                        depth_convention='camera_z', pixel_center_offset=.5, metric_calibration_accepted=False,
                        raw_confidence=np.asarray(confidence, np.float32), raw_sky_score=sky, **detail)
                    records.append(dict(image=frame['file_path'], station_id=station, face=frame['face'],
                                        depth_path=str(target), depth_sha256=sha(target), evidence_pixels=int(evidence.sum())))
                print(json.dumps(dict(mode=mode, completed=len(records), total=len(frames))), flush=True)
            write_json(output/'pose_manifest.json', dict(status='completed', frames=records, groups=groups,
                       model=options['pose'], auxiliary_ray_head_invariance_verified=bool(getattr(network, '_aux_probe_done', False))))
        del network
        torch.cuda.empty_cache()
    finally:
        lock.close()


def infer_raw_depth(dataset_dir, camera_json, new_output_dir, settings):
    """Explicit inference; emits only hash-bound heuristic abstention fields."""
    if not sys.platform.startswith('linux'):
        raise ValueError('Run raw DA3 on the configured Linux cloud host')
    options = validate_assets(settings)
    if not isinstance(options.get('gpu_lock'), str) or not Path(options['gpu_lock']).is_absolute():
        raise ValueError('Shared operator-owned GPU lock path is required')
    dataset, cameras, output = Path(dataset_dir).resolve(), Path(camera_json).resolve(), Path(new_output_dir).resolve()
    frames = _frames(dataset, cameras)
    if output.exists() or output.is_relative_to(dataset):
        raise ValueError('Raw-depth output directory must be new and outside its input dataset')
    output.mkdir(parents=True)
    (output/'metric').mkdir(); (output/'raw').mkdir()
    config = dict(dataset=str(dataset), camera_json=str(cameras), cameras_sha256=sha(cameras), output=str(output), options=options)
    config_path = output/'inference_config.json'
    write_json(config_path, config)
    state = dict(status='running', cameras_sha256=config['cameras_sha256'], metric_calibration_accepted=False,
                 role='heuristic_non_sky_abstention_only', frames=[], model_settings=options)
    write_json(output/'manifest.json', state)
    try:
        for mode in ('metric', 'pose'):
            with (output/(mode+'.log')).open('wb') as log:
                environment = dict(os.environ)
                package_root = str(Path(__file__).resolve().parents[2])
                environment['PYTHONPATH'] = package_root + (os.pathsep + environment['PYTHONPATH'] if environment.get('PYTHONPATH') else '')
                process = subprocess.Popen([sys.executable, '-m', 'tools.streetview_engine.da3_raw_depth',
                                            '--worker-config', str(config_path), '--mode', mode], stdout=log, stderr=subprocess.STDOUT, env=environment)
                try:
                    code = process.wait()
                finally:
                    if process.poll() is None:
                        process.terminate(); process.wait(timeout=30)
            if code:
                raise RuntimeError('Raw DA3 '+mode+' inference failed; inspect its log')
        result = _read(output/'pose_manifest.json')
        if result.get('status') != 'completed' or {r['image'] for r in result['frames']} != {f['file_path'] for f in frames}:
            raise ValueError('Raw depth output roster is incomplete')
        if _frames(dataset, cameras) != frames or sha(cameras) != config['cameras_sha256']:
            raise ValueError('Raw-depth input bindings changed during inference')
        lookup = {f['file_path']:f for f in frames}
        for row in result['frames']:
            if sha(row['depth_path']) != row['depth_sha256']:
                raise ValueError('Raw depth output hash mismatch')
            verify_depth_field(row['depth_path'], lookup[row['image']], options['side'])
        state.update(status='completed', frames=result['frames'], groups=result['groups'],
            auxiliary_ray_head_invariance_verified=result['auxiliary_ray_head_invariance_verified'],
            metric_sky_manifest_sha256=sha(output/'metric_manifest.json'), pose_manifest_sha256=sha(output/'pose_manifest.json'),
            interpretation='Known-camera baseline scaled predictions. No SfM metric-calibration fit is applied or claimed; only abstain from non-sky visibility votes.',
            sky_guard='Independent pinned nested metric branch raw ReLU sky score < 0.3',
            depth_evidence_guard='Original valid pixels, no nonzero edit support, finite positive depth, relative 3x3 depth edge <=0.12')
        write_json(output/'manifest.json', state)
        return state
    except BaseException as error:
        state.update(status='failed', error=str(error))
        write_json(output/'manifest.json', state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-config', required=True)
    parser.add_argument('--mode', choices=['metric', 'pose'], required=True)
    args = parser.parse_args()
    _worker(args.worker_config, args.mode)
