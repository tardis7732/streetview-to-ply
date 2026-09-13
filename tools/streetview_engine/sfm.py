"""Calibrated capture-rig SfM, auxiliary GPS adjustment, and dataset packaging.

`run(config, job_dir, settings)` consumes prepared/manifest.json. It never imports
scene experiments or inserts GPS cameras as a substitute for reconstruction.
PyCOLMAP 4.2 is loaded only when this stage is explicitly run.
"""
from __future__ import annotations
from dataclasses import asdict,dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import numpy as np
from PIL import Image
from .sfm_geometry import camera_to_station,candidate_capture_pairs,gps_local_enu,upright_from_rig_rotations,fit_horizontal_similarity,EDN_FROM_ENU


@dataclass(frozen=True)
class SfMSettings:
    threads:int=8
    device:str='auto'
    max_features:int=8192
    neighbors:int=8
    random_seed:int=0
    minimum_registered_fraction:float=1.0
    minimum_physical_stations:int=3
    minimum_visual_points:int=20
    maximum_reprojection_error_px:float=3.
    minimum_seed_stations:int=2
    minimum_seed_angle_degrees:float=2.
    minimum_seed_points:int=4
    maximum_seed_points:int=500000
    holdout_fraction:float=.2
    gps_horizontal_sigma_m:float=3.
    gps_vertical_sigma_m:float=20.
    minimum_gps_extent_sigma_ratio:float=2.
    maximum_gps_rmse_sigma_ratio:float=4.
    maximum_upright_residual_degrees:float=25.
    gps_ba_iterations:int=60
    gps_ba_seconds:float=180.
    maximum_visual_rmse_growth:float=1.15
    mapper_seconds:float=1800.
    recovery_rounds:int=2
    recovery_seconds:float=120.
    recovery_match_neighbors:int=0

    def __post_init__(self):
        if self.device not in {'auto','cpu','cuda'}:raise ValueError('SfM device must be auto/cpu/cuda')
        for key in ['threads','max_features','neighbors','minimum_physical_stations','minimum_visual_points','minimum_seed_stations','minimum_seed_points','maximum_seed_points','gps_ba_iterations']:
            value=getattr(self,key)
            if isinstance(value,bool) or not isinstance(value,int) or value<1:raise ValueError(f'{key} must be a positive integer')
        if self.minimum_physical_stations<3 or self.minimum_seed_stations<2:raise ValueError('SfM needs >=3 physical groups and seeds need >=2 training groups')
        if type(self.recovery_rounds) is not int or not 0<=self.recovery_rounds<=4:raise ValueError('recovery_rounds must be an integer in 0..4')
        if type(self.recovery_match_neighbors) is not int or not 0<=self.recovery_match_neighbors<=500:raise ValueError('recovery_match_neighbors must be an integer in 0..500')
        if not 0<self.minimum_registered_fraction<=1:raise ValueError('Invalid registered coverage fraction')
        if (isinstance(self.holdout_fraction,bool) or not isinstance(self.holdout_fraction,(int,float))
                or not np.isfinite(self.holdout_fraction) or not 0<=self.holdout_fraction<1):
            raise ValueError('holdout_fraction must be finite in [0,1)')
        for key in ['maximum_reprojection_error_px','minimum_seed_angle_degrees','gps_horizontal_sigma_m','gps_vertical_sigma_m','minimum_gps_extent_sigma_ratio','maximum_gps_rmse_sigma_ratio','maximum_upright_residual_degrees','gps_ba_seconds','maximum_visual_rmse_growth','mapper_seconds','recovery_seconds']:
            if not np.isfinite(getattr(self,key)) or getattr(self,key)<=0:raise ValueError(f'{key} must be finite positive')


def validate_split_policy(settings, options=None):
    """All-training is explicit and cannot also claim heldout quality evaluation."""
    if options is None:
        if not isinstance(settings,dict):raise ValueError('Operator split settings must be an object')
        raw=settings.get('sfm',{})
        options=raw if isinstance(raw,SfMSettings) else SfMSettings(**raw)
    if options.holdout_fraction==0:
        quality=settings.get('quality') if isinstance(settings,dict) else None
        if not isinstance(quality,dict) or quality.get('enabled') is not False:
            raise ValueError('sfm.holdout_fraction=0 requires explicit quality.enabled=false; no heldout evaluation is available')
        return 'all_train'
    return 'physical_station_holdout'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')


def owned_file(root,relative):
    path=Path(relative)
    if path.is_absolute() or '..' in path.parts:raise ValueError('Prepared paths must be relative to the job directory')
    result=(root/path).resolve()
    if not result.is_relative_to(root.resolve()) or not result.is_file():raise ValueError('Prepared input missing or outside job directory: '+str(relative))
    return result


def copy_file(source,target):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise FileExistsError('Refusing to replace existing dataset input: '+str(target))
    # Copies stay independently immutable even if a later stage rewrites a mask.
    shutil.copyfile(source,target)
    if sha(source)!=sha(target):raise RuntimeError('Dataset copy hash mismatch')


def validate_prepared_selection(config,prepared):
    """Require a completed prepared artifact for exactly this frozen roster."""
    from .processing_options import read_processing_options
    if prepared.get('status')!='complete':raise ValueError('Prepared stage is not complete')
    if read_processing_options(config)!=read_processing_options(prepared):raise ValueError('Prepared processing options differ from frozen selection')
    ids=config.get('panorama_ids')
    if not isinstance(ids,list) or not ids or any(not isinstance(i,str) or not i for i in ids) or len(set(ids))!=len(ids):
        raise ValueError('Frozen selection must contain unique panorama IDs')
    stations=prepared.get('stations',[]);frames=prepared.get('frames',[])
    station_ids=[station.get('pano_id') for station in stations]
    if len(station_ids)!=len(ids) or set(station_ids)!=set(ids) or {frame.get('pano_id') for frame in frames}!=set(ids):
        raise ValueError('Prepared panorama roster differs from frozen selection')


def prepare_inputs(job_dir,output,prepared):
    frames=prepared.get('frames',[]);station_rows=prepared.get('stations',[])
    if not frames or not station_rows:raise ValueError('Prepared manifest must contain frames and capture stations')
    stations={str(p.get('pano_id',p.get('id'))):p for p in station_rows}
    if len(stations)!=len(station_rows) or 'None' in stations:raise ValueError('Capture metadata requires unique pano_id')
    by_capture={};input_hashes={};face_calibration={}
    for frame in frames:
        pid=str(frame['pano_id']);group=str(frame['station_id']);face=str(frame['face'])
        if pid not in stations or str(stations[pid]['station_id'])!=group:raise ValueError('Frame capture/physical-group metadata mismatch')
        pose=camera_to_station(frame['camera_to_station_cv'])
        intrinsics=np.array([frame[k] for k in ['fl_x','fl_y','cx','cy']],np.float64)
        width,height=int(frame['w']),int(frame['h'])
        if min(width,height)<16 or not np.isfinite(intrinsics).all() or np.any(intrinsics[:2]<=0):raise ValueError('Invalid prepared calibration')
        signature=(width,height,*intrinsics,*pose.reshape(-1))
        if face in face_calibration and face_calibration[face]!=signature:raise ValueError('Captures must share a prepared calibration per face; resample differing cube layouts before SfM')
        face_calibration[face]=signature
        for path_key,hash_key in [('file_path','source_sha256'),('sfm_mask_path','sfm_mask_sha256'),('mask_path','mask_sha256'),('sky_mask_path','sky_mask_sha256'),('ground_mask_path','ground_mask_sha256')]:
            path=owned_file(job_dir,frame[path_key]);actual=sha(path)
            if frame.get(hash_key)!=actual:raise ValueError('Prepared image/mask hash mismatch: '+str(frame[path_key]))
            with Image.open(path) as image:
                if image.size!=(width,height):raise ValueError('Prepared image/mask dimensions disagree with calibration')
            input_hashes[path.relative_to(job_dir).as_posix()]=actual
        capture=by_capture.setdefault(pid,{})
        if face in capture:raise ValueError('Duplicate cube face in one capture')
        capture[face]=dict(frame)
    layouts={tuple(sorted(capture)) for capture in by_capture.values()}
    if len(layouts)!=1:raise ValueError('Every capture must contain the same calibrated face layout')
    if any(len({Path(row['file_path']).suffix.lower() for row in capture.values()})!=1 for capture in by_capture.values()):raise ValueError('Prepared faces in a capture must share one image extension for exact rig frame association')
    if set(by_capture)!=set(stations):raise ValueError('Capture metadata and prepared images differ')
    # A reference face must have identity camera-to-station orientation. Names
    # are derived from calibration, not a location or chosen camera ID.
    faces=sorted(face_calibration)
    references=[face for face in faces if np.allclose(np.array(face_calibration[face][6:]).reshape(4,4),np.eye(4),atol=1e-8,rtol=0)]
    if not references:raise ValueError('Prepared rig requires one identity reference camera')
    reference=references[0];faces=[reference,*[face for face in faces if face!=reference]]
    mapping={};captures=[]
    for pid in sorted(by_capture):
        token=hashlib.sha256(pid.encode()).hexdigest()[:24];captures.append(dict(stations[pid],pano_id=pid))
        for sensor,face in enumerate(faces):
            row=by_capture[pid][face];suffix=Path(row['file_path']).suffix.lower()
            name=f'sensor_{sensor:02d}/{token}{suffix}'
            copy_file(owned_file(job_dir,row['file_path']),output/'images'/name)
            copy_file(owned_file(job_dir,row['sfm_mask_path']),output/'masks'/(name+'.png'))
            mapping[name]=row
    return mapping,captures,faces,input_hashes


def extract_and_configure_rig(pc,output,mapping,faces,settings):
    device=settings.device
    if device=='auto':device='cuda' if bool(pc.has_cuda) else 'cpu'
    if device=='cuda' and not bool(pc.has_cuda):raise RuntimeError('CUDA SfM requested but installed pycolmap has no CUDA; configure cpu or install a CUDA-enabled build')
    extraction=pc.FeatureExtractionOptions(num_threads=settings.threads,use_gpu=device=='cuda')
    extraction.sift.max_num_features=settings.max_features
    for index,face in enumerate(faces):
        names=sorted(name for name,row in mapping.items() if row['face']==face);row=mapping[names[0]]
        reader=pc.ImageReaderOptions(camera_model='PINHOLE',camera_params=','.join(str(row[k]) for k in ['fl_x','fl_y','cx','cy']),mask_path=output/'masks')
        pc.extract_features(output/'database.db',output/'images',image_names=names,camera_mode=pc.CameraMode.SINGLE,reader_options=reader,extraction_options=extraction,device=getattr(pc.Device,device))
    cameras=[]
    for index,face in enumerate(faces):
        row=next(row for row in mapping.values() if row['face']==face);cam_from_rig=np.linalg.inv(camera_to_station(row['camera_to_station_cv']))
        args=dict(image_prefix=f'sensor_{index:02d}/',ref_sensor=index==0)
        if index:args['cam_from_rig']=pc.Rigid3d(pc.Rotation3d(cam_from_rig[:3,:3]),cam_from_rig[:3,3])
        cameras.append(pc.RigConfigCamera(**args))
    database=pc.Database.open(output/'database.db')
    try:
        pc.apply_rig_config([pc.RigConfig(cameras=cameras)],database)
        images=database.read_all_images()
        if {image.name for image in images}!=set(mapping):raise RuntimeError('Feature database does not cover prepared images')
        frame_ids={image.frame_id for image in images};pano_ids={row['pano_id'] for row in mapping.values()}
        if len(frame_ids)!=len(pano_ids):raise RuntimeError('Rig configuration did not create one frame per capture')
        statistics=[dict(image=image.name,keypoints=len(database.read_keypoints(image.image_id)),camera_id=int(image.camera_id),frame_id=int(image.frame_id),pano_id=mapping[image.name]['pano_id'],station_id=mapping[image.name]['station_id']) for image in images]
    finally:database.close()
    return dict(pycolmap_version=pc.__version__,cuda_available=bool(pc.has_cuda),feature_device=device,cpu_fallback=settings.device=='auto' and device=='cpu',capture_rigs=len(frame_ids),sensors=len(faces),images=statistics)


def incremental_options(pc,settings):
    # PyCOLMAP 4.2 binds max_runtime_seconds as an integer. Positive fractional
    # configuration seconds round upward so subsecond limits cannot become 0.
    options=pc.IncrementalPipelineOptions(num_threads=settings.threads,random_seed=settings.random_seed,min_model_size=settings.minimum_physical_stations,extract_colors=False,load_all_images=True,ba_refine_focal_length=False,ba_refine_principal_point=False,ba_refine_extra_params=False,ba_refine_sensor_from_rig=False,max_runtime_seconds=math.ceil(settings.mapper_seconds))
    options.mapper.abs_pose_refine_focal_length=False;options.mapper.abs_pose_refine_extra_params=False
    options.mapper.init_min_num_inliers=40;options.mapper.init_min_tri_angle=4.
    return options


def recover_capture_rigs(pc,output,rec,mapping,settings):
    """Attempt missing rigs directly, retaining native pose/inlier thresholds.

    COLMAP 4.2 FindNextImages gates each individual image at 30 visible points,
    whereas RegisterNextImage can pool all calibrated rig cameras. A rig whose
    faces each fall below that gate can therefore be skipped without a pose
    attempt. This bounded pass changes scheduling only, never pose thresholds.
    https://github.com/colmap/colmap/blob/4.2.0/src/colmap/sfm/incremental_mapper_impl.cc
    """
    expected={row['pano_id'] for row in mapping.values()}
    registered={mapping[im.name]['pano_id'] for im in rec.images.values() if im.has_pose}
    report=dict(status='not_needed',before=len(registered),after=len(registered),attempts=[],
                missing_capture_ids=sorted(expected-registered),pose_policy='Native generalized rig RANSAC; unchanged minimum inliers and pixel threshold; no GPS pose substitution')
    if registered==expected or settings.recovery_rounds==0:
        if registered!=expected:report['status']='disabled'
        return rec,report
    started=time.monotonic();candidate=pc.Reconstruction(rec)
    database=pc.Database.open(output/'database.db')
    try:
        pipeline=incremental_options(pc,settings)
        cache=pc.DatabaseCache.create(database,pc.DatabaseCacheOptions(min_num_matches=pipeline.min_num_matches,
            ignore_watermarks=pipeline.ignore_watermarks,load_all_images=True))
        mapper=pc.IncrementalMapper(cache);mapper.begin_reconstruction(candidate)
        native=pipeline.mapper;native.num_threads=settings.threads;native.random_seed=settings.random_seed
        report.update(minimum_pose_inliers=native.abs_pose_min_num_inliers,maximum_pose_error_px=native.abs_pose_max_error)
        original_frames=set(rec.reg_frame_ids());new_frames=set();refined=False
        try:
            for round_index in range(settings.recovery_rounds):
                missing={}
                for im in candidate.images.values():
                    if not im.has_pose and im.name in mapping:
                        missing.setdefault(im.frame_id,[]).append(im)
                progress=False
                for frame,images in sorted(missing.items(),key=lambda item:min(mapping[im.name]['pano_id'] for im in item[1])):
                    if time.monotonic()-started>=settings.recovery_seconds:break
                    representative=min(images,key=lambda im:im.name)
                    visible=[int(mapper.observation_manager.num_visible_points3D(im.image_id)) for im in images]
                    okay=bool(mapper.register_next_image(native,representative.image_id))
                    report['attempts'].append(dict(round=round_index,pano_id=mapping[representative.name]['pano_id'],
                        frame_id=int(frame),per_face_visible_points=visible,registered=okay))
                    if okay:new_frames.add(frame);progress=True;refined=False
                if not progress:break
                # Refine actual observations without changing fixed intrinsics
                # or calibrated sensor rotations. No extra matching is implied.
                remaining=settings.recovery_seconds-(time.monotonic()-started)
                if remaining<=0:break
                ba=pipeline.get_global_bundle_adjustment()
                ba.refine_focal_length=False;ba.refine_principal_point=False;ba.refine_extra_params=False;ba.refine_sensor_from_rig=False
                ba.ceres.use_gpu=False;ba.ceres.solver_options.num_threads=settings.threads
                ba.ceres.solver_options.max_solver_time_in_seconds=remaining
                if not mapper.adjust_global_bundle(native,ba):
                    report['status']='rejected_unusable_bundle';break
                mapper.filter_points(native);mapper.filter_frames(native);refined=True
                if len(candidate.reg_frame_ids())==len(expected):break
            candidate.update_point_3d_errors()
            after_error=float(candidate.compute_mean_reprojection_error());before_error=float(rec.compute_mean_reprojection_error())
            active=set(candidate.reg_frame_ids());retained=[]
            for frame in sorted(new_frames&active):
                points={int(pt.point3D_id) for im in candidate.images.values() if im.frame_id==frame for pt in im.points2D if pt.has_point3D()}
                retained.append(dict(frame_id=int(frame),unique_observed_points=len(points)))
            report.update(after_reprojection_px=after_error,before_reprojection_px=before_error,recovered_support=retained)
            valid=(refined and bool(retained) and original_frames<=active and all(row['unique_observed_points']>=native.abs_pose_min_num_inliers for row in retained)
                and np.isfinite(after_error) and after_error<=max(before_error*settings.maximum_visual_rmse_growth,before_error+.15)
                and report['status']!='rejected_unusable_bundle')
            if valid:
                report['status']='recovered';rec=candidate
            else:report['status']='insufficient_verified_support'
        finally:mapper.end_reconstruction(False)
    finally:database.close()
    registered={mapping[im.name]['pano_id'] for im in rec.images.values() if im.has_pose}
    report.update(after=len(registered),missing_capture_ids=sorted(expected-registered),elapsed_seconds=time.monotonic()-started)
    return rec,report


def reuse_visual_cache(pc,source_job,job_dir,output,mapping,settings,signature):
    """Copy operator-selected terminal SfM state; never link mutable databases."""
    source=Path(source_job).expanduser().resolve()
    if source==job_dir:raise ValueError('SfM cache source must be another job')
    old_path=owned_file(source,'sfm/input_signature.json');old=json.loads(old_path.read_text(encoding='utf8'))
    if old.get('prepared_sha256')!=signature['prepared_sha256'] or old.get('job_config_sha256')!=signature['job_config_sha256']:
        raise ValueError('SfM cache belongs to a different prepared artifact or frozen config')
    old_options=asdict(SfMSettings(**old['settings']));current=asdict(settings)
    recovery={'recovery_rounds','recovery_seconds','recovery_match_neighbors'}
    if {k:v for k,v in old_options.items() if k not in recovery}!={k:v for k,v in current.items() if k not in recovery}:
        raise ValueError('SfM cache feature/mapping settings differ')
    terminal=None
    for name,status in [('sfm/manifest.json','completed'),('sfm/failure.json','failed')]:
        path=source/name
        if path.is_file():
            data=json.loads(path.read_text(encoding='utf8'))
            if data.get('status')==status and data.get('prepared_sha256')==old['prepared_sha256']:terminal=name;break
    if terminal is None:raise ValueError('SfM cache must come from a terminal stage')
    source_prepared=owned_file(source,'prepared/manifest.json')
    if sha(source_prepared)!=signature['prepared_sha256']:raise ValueError('SfM cache prepared bytes changed')
    # Source hashes must still describe its actual inputs, not just the clone.
    prepared=json.loads(source_prepared.read_text(encoding='utf8'))
    for row in prepared['frames']:
        for path_key,hash_key in [('file_path','source_sha256'),('sfm_mask_path','sfm_mask_sha256')]:
            if sha(owned_file(source,row[path_key]))!=row[hash_key]:raise ValueError('SfM cache source photograph/mask changed')
    feature_path=owned_file(source,'sfm/feature_report.json');features=json.loads(feature_path.read_text(encoding='utf8'))
    if features.get('pycolmap_version')!=pc.__version__:raise ValueError('SfM cache PyCOLMAP version differs')
    if {row['image'] for row in features['images']}!=set(mapping):raise ValueError('SfM cache feature roster differs')
    model_roots=[p for p in (source/'sfm/visual_models').glob('*') if p.is_dir()]
    if (source/'sfm/visual_model').is_dir():model_roots.append(source/'sfm/visual_model')
    models=[]
    for path in model_roots:
        if path.is_symlink() or any(p.is_symlink() for p in path.rglob('*')):raise ValueError('SfM cache contains symlinked model files')
        model=pc.Reconstruction(path)
        if any(im.name not in mapping for im in model.images.values()):raise ValueError('Cached visual model has unrelated images')
        models.append((model,path))
    if not models:raise ValueError('SfM cache contains no visual model')
    rec,model_path=max(models,key=lambda item:(item[0].num_reg_frames(),item[0].num_points3D(),str(item[1])))
    source_db=owned_file(source,'sfm/database.db')
    for suffix in ['-wal','-journal']:
        journal=Path(str(source_db)+suffix)
        if journal.exists() and journal.stat().st_size:raise ValueError('SfM cache database has pending journal state')
    files=[old_path,source_prepared,owned_file(source,terminal),feature_path,source_db,owned_file(source,'sfm/match_pairs.txt')]
    files+=sorted(p for p in model_path.rglob('*') if p.is_file())
    hashes={p.relative_to(source).as_posix():sha(p) for p in files}
    copy_file(source_db,output/'database.db');copy_file(feature_path,output/'feature_report.json')
    copy_file(source/'sfm/match_pairs.txt',output/'match_pairs.txt')
    target=output/'cached_visual_model'
    for path in sorted(p for p in model_path.rglob('*') if p.is_file()):copy_file(path,target/path.relative_to(model_path))
    db=pc.Database.open(output/'database.db')
    try:
        images=db.read_all_images();cameras={cam.camera_id:cam for cam in db.read_all_cameras()}
        frames={frame.frame_id:frame for frame in db.read_all_frames()};rigs={rig.rig_id:rig for rig in db.read_all_rigs()}
        if {im.name for im in images}!=set(mapping):raise ValueError('Cached database image roster differs')
        groups={};id_bindings={}
        for im in images:
            row=mapping[im.name];camera=cameras[im.camera_id];frame=frames[im.frame_id];rig=rigs[frame.rig_id]
            if camera.model_name!='PINHOLE' or (camera.width,camera.height)!=(row['w'],row['h']) or not np.allclose(camera.params,[row[k] for k in ['fl_x','fl_y','cx','cy']],rtol=0,atol=1e-9):raise ValueError('Cached camera calibration differs')
            sensor=im.data_id.sensor_id
            pose=np.eye(4)[:3] if sensor==rig.ref_sensor_id else rig.sensor_from_rig(sensor).matrix()
            if not np.allclose(pose,np.linalg.inv(camera_to_station(row['camera_to_station_cv']))[:3],rtol=0,atol=1e-9):raise ValueError('Cached rig sensor calibration differs')
            groups.setdefault(row['pano_id'],set()).add(im.frame_id);id_bindings[im.image_id]=(im.name,im.camera_id,im.frame_id)
        if any(len(ids)!=1 for ids in groups.values()) or len({next(iter(ids)) for ids in groups.values()})!=len(groups):raise ValueError('Cached rig capture centers are not independently bound')
        for im in rec.images.values():
            if id_bindings.get(im.image_id)!=(im.name,im.camera_id,im.frame_id):raise ValueError('Cached reconstruction image IDs disagree with database')
            camera=rec.cameras[im.camera_id];expected=cameras[im.camera_id]
            if camera.model_name!=expected.model_name or (camera.width,camera.height)!=(expected.width,expected.height) or not np.allclose(camera.params,expected.params,rtol=0,atol=1e-9):raise ValueError('Cached reconstruction calibration differs')
            rig=rec.rigs[rec.frames[im.frame_id].rig_id];sensor=im.data_id.sensor_id
            pose=np.eye(4)[:3] if sensor==rig.ref_sensor_id else rig.sensor_from_rig(sensor).matrix()
            if not np.allclose(pose,np.linalg.inv(camera_to_station(mapping[im.name]['camera_to_station_cv']))[:3],rtol=0,atol=1e-9):raise ValueError('Cached reconstruction rig calibration differs')
    finally:db.close()
    if any(sha(source/name)!=value for name,value in hashes.items()):raise RuntimeError('SfM cache source changed during copy')
    receipt=dict(status='verified_copied',source_job=str(source),source_files_sha256=hashes,
        database_sha256_at_copy=hashes['sfm/database.db'],database_sha256=sha(output/'database.db'),model_directory='cached_visual_model',
        registered_captures=rec.num_reg_frames(),points3D=rec.num_points3D(),mutable_files_independently_copied=True)
    save(output/'cache_receipt.json',receipt)
    return pc.Reconstruction(target),features,receipt


def visual_reconstruction(pc,output,mapping,captures,settings,device,cached_rec=None):
    pairs=candidate_capture_pairs(captures,settings.neighbors);by_capture={p['pano_id']:[] for p in captures}
    for name,row in mapping.items():by_capture[row['pano_id']].append(name)
    names=[]
    for a,b in pairs:
        names.extend(f'{x} {y}' for x in sorted(by_capture[captures[a]['pano_id']]) for y in sorted(by_capture[captures[b]['pano_id']]))
    if not names:raise RuntimeError('Insufficient independent physical station pairs')
    if cached_rec is None:
        (output/'match_pairs.txt').write_text('\n'.join(names)+'\n',encoding='utf8')
        matching=pc.FeatureMatchingOptions(num_threads=settings.threads,use_gpu=device=='cuda',guided_matching=True,skip_image_pairs_in_same_frame=True)
        verification=pc.TwoViewGeometryOptions();verification.ransac.random_seed=settings.random_seed
        pc.match_image_pairs(output/'database.db',matching_options=matching,pairing_options=pc.ImportedPairingOptions(match_list_path=output/'match_pairs.txt',block_size=128),verification_options=verification,device=getattr(pc.Device,device))
        options=incremental_options(pc,settings)
        (output/'visual_models').mkdir()
        models=pc.incremental_mapping(output/'database.db',output/'images',output/'visual_models',options=options)
        if not models:raise RuntimeError('Visual SfM found no connected reconstruction; GPS poses were not substituted')
        rec=max(models.values(),key=lambda model:(model.num_reg_frames(),model.num_points3D()))
    else:rec=cached_rec;models={0:cached_rec}
    rec,recovery=recover_capture_rigs(pc,output,rec,mapping,settings)
    from .sfm_support_recovery import recover_with_additional_matches
    rec,recovery=recover_with_additional_matches(pc,output,rec,mapping,captures,settings,device,recovery)
    save(output/'recovery_report.json',recovery)
    registered={mapping[im.name]['pano_id'] for im in rec.images.values() if im.has_pose and im.name in mapping}
    groups={str(mapping[im.name]['station_id']) for im in rec.images.values() if im.has_pose and im.name in mapping}
    fraction=len(registered)/len(captures)
    visual=dict(models=len(models),registered_captures=len(registered),physical_stations=len(groups),registered_fraction=fraction,points3D=rec.num_points3D(),mean_reprojection_error_px=float(rec.compute_mean_reprojection_error()),unregistered_capture_ids=sorted(set(by_capture)-registered),capture_pairs=len(pairs),image_pairs=len(names),cache_reused=cached_rec is not None,recovery=recovery,pair_policy='GPS nearest captures and provider links propose all cross-face combinations; no same-station baseline or heading cutoff; visual matching estimates poses')
    if recovery.get('additional_matching',{}).get('status')=='matched':
        visual.update(base_capture_pairs=len(pairs),base_image_pairs=len(names),
            capture_pairs=len(pairs)+recovery['additional_matching']['additional_capture_pairs'],
            image_pairs=len(names)+recovery['additional_matching']['additional_image_pairs'])
    save(output/'visual_report.json',visual)
    if len(groups)<settings.minimum_physical_stations or fraction<settings.minimum_registered_fraction or rec.num_points3D()<settings.minimum_visual_points:raise RuntimeError(f'Insufficient visual reconstruction: {len(groups)} physical groups, {fraction:.1%} captures, {rec.num_points3D()} points')
    return rec,visual


def gps_adjust(pc,rec,mapping,captures,settings):
    lookup={p['pano_id']:p for p in captures};representatives={}
    for image in rec.images.values():
        if not image.has_pose or image.name not in mapping or image.num_points3D==0:continue
        row=mapping[image.name];group=str(row['station_id']);candidate=(image.num_points3D,image.name,image.image_id)
        if group not in representatives or candidate>representatives[group]:representatives[group]=candidate
    image_ids=[representatives[group][2] for group in sorted(representatives)]
    metadata=[lookup[mapping[rec.image(i).name]['pano_id']] for i in image_ids]
    if len(image_ids)<settings.minimum_physical_stations:raise RuntimeError('Insufficient visually observed physical stations for GPS scale')
    gps,has_height,origin=gps_local_enu(metadata)
    def align():
        rotations=[camera_to_station(mapping[rec.image(i).name]['camera_to_station_cv'])[:3,:3]@rec.image(i).cam_from_world().rotation.matrix() for i in image_ids]
        headings=[row.get('heading',row.get('heading_degrees')) for row in metadata]
        upright,orientation=upright_from_rig_rotations(rotations,headings)
        scale,rotation,translation,report=fit_horizontal_similarity(np.array([rec.image(i).projection_center() for i in image_ids]),gps,upright=upright,huber_m=settings.gps_horizontal_sigma_m)
        if orientation['orientation_residual_median_degrees']>settings.maximum_upright_residual_degrees:raise RuntimeError('Cube/heading orientation is inconsistent; refusing to invent geographic up')
        if report['gps_horizontal_extent_m']<settings.minimum_gps_extent_sigma_ratio*settings.gps_horizontal_sigma_m:raise RuntimeError('GPS baseline is too short relative to its configured uncertainty to establish metric scale')
        if report['horizontal_rmse_m']>settings.maximum_gps_rmse_sigma_ratio*settings.gps_horizontal_sigma_m:raise RuntimeError('Visual trajectory and GPS disagree beyond configured auxiliary tolerance')
        rec.transform(pc.Sim3d(scale,pc.Rotation3d(rotation),translation))
        return dict(report,**orientation,rotation=rotation.tolist(),translation=translation.tolist())
    initial=align();rec.update_point_3d_errors();before=float(rec.compute_mean_reprojection_error());before_observations=sum(int(im.num_points3D) for im in rec.images.values() if im.has_pose)
    # COLMAP's native pose-prior factory performs a 3D similarity fit before
    # inserting priors. Its estimator rejects planar targets. Unknown height
    # therefore uses the existing visual height as an initialization proxy,
    # with effectively zero vertical weight, never a measured ground height.
    prior_positions=gps.copy();visual_centers=np.array([rec.image(i).projection_center() for i in image_ids])
    prior_positions[~has_height,2]=visual_centers[~has_height,2]
    base_report=dict(status='aligned_with_auxiliary_GPS',origin=origin,initial_similarity=initial,physical_alignment_group_count=len(image_ids),height_prior_count=int(has_height.sum()),unknown_height_policy='Existing visual height is only a factory-initialization proxy with effectively zero vertical weight; no ground or camera height observation is invented',horizontal_sigma_m=settings.gps_horizontal_sigma_m,vertical_sigma_m=settings.gps_vertical_sigma_m,prior_station_ids=sorted(representatives),metric_scale_status='auxiliary_GPS_estimate_with_configured_uncertainty_not_survey_truth',upright_assumption='Prepared cube local -Y is approximately upright; provider headings are orientation evidence; raw capture angles retained for audit')
    ranks=[np.linalg.matrix_rank(values-values.mean(axis=0),tol=max(1.,np.linalg.norm(values-values.mean(axis=0)))*1e-9) for values in [visual_centers,prior_positions]]
    if min(ranks)<3:
        return dict(base_report,gps_ba_status='not_run_rank_deficient_native_3d_alignment',physical_prior_count=0,post_ba_similarity=initial,reprojection_before_px=before,reprojection_after_px=before,residual_components=0,limitation='Native COLMAP 3D prior initializer rejects a planar/linear trajectory. Actual visual bundle adjustment and robust horizontal GPS alignment are retained; no joint GPS BA is claimed.')
    priors=[]
    for index,image_id in enumerate(image_ids):
        vertical=settings.gps_vertical_sigma_m if has_height[index] else max(settings.gps_horizontal_sigma_m,initial['gps_horizontal_extent_m'])*1e6
        covariance=np.diag([settings.gps_horizontal_sigma_m**2,settings.gps_horizontal_sigma_m**2,vertical**2])
        priors.append(pc.PosePrior(corr_data_id=rec.image(image_id).data_id,position=prior_positions[index],position_covariance=covariance,coordinate_system=pc.PosePriorCoordinateSystem.CARTESIAN))
    config=pc.BundleAdjustmentConfig()
    for image_id in rec.reg_image_ids():config.add_image(image_id)
    options=pc.BundleAdjustmentOptions(refine_focal_length=False,refine_principal_point=False,refine_extra_params=False,refine_sensor_from_rig=False,refine_rig_from_world=True,refine_points3D=True)
    options.ceres.use_gpu=False;options.ceres.solver_options.num_threads=settings.threads;options.ceres.solver_options.max_num_iterations=settings.gps_ba_iterations;options.ceres.solver_options.max_solver_time_in_seconds=settings.gps_ba_seconds
    prior_options=pc.PosePriorBundleAdjustmentOptions();prior_options.ceres.prior_position_loss_function_type=pc.LossFunctionType.HUBER;prior_options.ceres.prior_position_loss_scale=2.
    prior_options.alignment_ransac.max_error=settings.gps_horizontal_sigma_m*5;prior_options.alignment_ransac.max_num_trials=10000;prior_options.alignment_ransac.random_seed=settings.random_seed
    result=pc.create_pose_prior_bundle_adjuster(options,prior_options,config,priors,rec).solve()
    if not result.is_solution_usable():raise RuntimeError('Auxiliary GPS bundle adjustment did not produce a usable solution')
    residuals=int(result.ceres_summary.num_residuals)
    if residuals-2*before_observations!=3*len(priors):raise RuntimeError('Pose-prior residual count does not match one observation per physical group')
    restored=align();rec.update_point_3d_errors();after=float(rec.compute_mean_reprojection_error())
    if not np.isfinite(after) or after>max(before*settings.maximum_visual_rmse_growth,before+.15):raise RuntimeError('Auxiliary GPS adjustment materially worsened visual reprojection')
    return dict(base_report,gps_ba_status='completed_with_verified_prior_residuals',post_ba_similarity=restored,physical_prior_count=len(priors),reprojection_before_px=before,reprojection_after_px=after,residual_components=residuals)


def run(config,job_dir,settings=None):
    """Run only in the supplied job directory; raise on unsupported geometry."""
    cache_settings=settings.get('input_cache',{}) if isinstance(settings,dict) else {}
    if not isinstance(cache_settings,dict):raise ValueError('input_cache must be an operator settings object')
    operator_settings=settings or {}
    settings=settings or {};settings=settings.get('sfm',settings) if isinstance(settings,dict) else settings
    options=settings if isinstance(settings,SfMSettings) else SfMSettings(**settings)
    validate_split_policy(operator_settings,options)
    job_dir=Path(job_dir).resolve();prepared_path=owned_file(job_dir,'prepared/manifest.json');prepared=json.loads(prepared_path.read_text(encoding='utf8'))
    validate_prepared_selection(config,prepared)
    collection_path=owned_file(job_dir,prepared.get('collection_manifest_path','collection/manifest.json'))
    if not prepared.get('collection_sha256') or sha(collection_path)!=prepared['collection_sha256']:raise ValueError('Prepared collection binding does not match actual collection artifact')
    output=job_dir/'sfm'
    if output.exists():raise FileExistsError('SfM output already exists; use a fresh job rather than reuse unbound state')
    output.mkdir();started=time.monotonic()
    signature=dict(prepared_sha256=sha(prepared_path),settings=asdict(options),job_config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True,allow_nan=False).encode()).hexdigest())
    save(output/'input_signature.json',signature)
    try:
        import pycolmap as pc
        if not hasattr(pc,'FeatureExtractionOptions') or not hasattr(pc,'create_pose_prior_bundle_adjuster'):raise RuntimeError('SfM requires the pycolmap 4.2 rig and pose-prior APIs')
        mapping,captures,faces,hashes=prepare_inputs(job_dir,output,prepared)
        if len(set(str(p['station_id']) for p in captures))<options.minimum_physical_stations:raise RuntimeError('Not enough independent physical stations for reconstruction and appearance holdout')
        cached_rec=None
        if cache_settings.get('source_job_dir'):
            cached_rec,features,_=reuse_visual_cache(pc,cache_settings['source_job_dir'],job_dir,output,mapping,options,signature)
        else:
            features=extract_and_configure_rig(pc,output,mapping,faces,options);save(output/'feature_report.json',features)
        rec,visual=visual_reconstruction(pc,output,mapping,captures,options,features['feature_device'],cached_rec=cached_rec);save(output/'visual_report.json',visual)
        (output/'visual_model').mkdir();rec.write(output/'visual_model')
        gps=gps_adjust(pc,rec,mapping,captures,options);save(output/'gps_report.json',gps)
        rec.transform(pc.Sim3d(1.,pc.Rotation3d(EDN_FROM_ENU),np.zeros(3)));(output/'model').mkdir();rec.write(output/'model')
        from .sfm_dataset import package_dataset
        dataset=package_dataset(rec,mapping,captures,job_dir,output/'dataset',options,gps,config=config)
        if sha(prepared_path)!=signature['prepared_sha256'] or any(sha(owned_file(job_dir,path))!=value for path,value in hashes.items()):raise RuntimeError('Prepared inputs changed while SfM was running')
        report=dict(schema_version=1,status='completed',**signature,elapsed_seconds=time.monotonic()-started,visual=visual,gps=gps,dataset_path='sfm/dataset',dataset_manifest_sha256=sha(output/'dataset/dataset_manifest.json'),input_images_and_sfm_masks_sha256=hashes,features=dict(pycolmap_version=pc.__version__,device=features['feature_device'],cpu_fallback=features['cpu_fallback']),dataset=dataset,geometry_source='actual calibrated-rig feature matching, triangulation and bundle adjustment; GPS is auxiliary')
        save(output/'manifest.json',report);return report
    except Exception as error:
        save(output/'failure.json',dict(status='failed',reason=str(error),error_type=type(error).__name__,elapsed_seconds=time.monotonic()-started,**signature))
        raise
