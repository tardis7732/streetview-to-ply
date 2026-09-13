"""Package only visually registered cameras and training-only colored seeds."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from PIL import Image
from tools.streetview_geometry.contracts import station_split
from .sfm_geometry import CV_FROM_GL,maximum_ray_angle_degrees,training_observation_color,camera_to_station
from .processing_options import read_processing_options
from .preprocess import mask_conventions


def seed_arrays(rec,mapping,job_dir,train_groups,settings):
    """Use only static foreground observations from training physical groups.

    Points/poses may have been reconstructed using all images. That is recorded
    as appearance holdout with known poses, never novel-scene geometry holdout.
    """
    rows=[];rejected={'reprojection':0,'training_support':0,'triangulation':0}
    candidates=[];by_image={}
    for point_id,point in sorted(rec.points3D.items()):
        xyz=np.asarray(point.xyz,np.float64)
        if not np.isfinite(xyz).all() or not np.isfinite(point.error) or point.error<0 or point.error>settings.maximum_reprojection_error_px:
            rejected['reprojection']+=1;continue
        index=len(candidates);all_groups=set()
        for element in point.track.elements:
            image=rec.image(element.image_id);frame=mapping.get(image.name)
            if frame is None or not image.has_pose:continue
            group=str(frame['station_id']);all_groups.add(group)
            if group in train_groups:by_image.setdefault(image.image_id,[]).append((index,element.point2D_idx))
        candidates.append(dict(point_id=int(point_id),xyz=xyz,error=float(point.error),all_groups=len(all_groups),observations=[],centers={}))
    # Decode each training image once, rather than rereading full photographs
    # in point-track order. Pixel sampling is batched per image.
    delta=np.array([(x,y) for y in [-1,0,1] for x in [-1,0,1]])
    for image_id,records in by_image.items():
        image=rec.image(image_id);frame=mapping[image.name];group=str(frame['station_id'])
        indices=np.array([record[0] for record in records],np.int64)
        xy=np.array([image.points2D[record[1]].xy for record in records],np.float64)
        finite=np.isfinite(xy).all(axis=1);pixels=np.floor(np.where(np.isfinite(xy),xy,-2)).astype(np.int64)
        inside=finite&(pixels[:,0]>=1)&(pixels[:,0]<frame['w']-1)&(pixels[:,1]>=1)&(pixels[:,1]<frame['h']-1)
        pixels=pixels[inside];indices=indices[inside]
        if not len(pixels):continue
        with Image.open(job_dir/frame['sfm_mask_path']) as image_file:mask=np.asarray(image_file.convert('L'))
        x=pixels[:,0,None]+delta[None,:,0];y=pixels[:,1,None]+delta[None,:,1]
        valid=np.all(mask[y,x]>0,axis=1);indices=indices[valid];x=x[valid];y=y[valid]
        if not len(indices):continue
        with Image.open(job_dir/frame['file_path']) as image_file:rgb=np.asarray(image_file.convert('RGB'))
        colors=rgb[y,x].mean(axis=1);center=np.asarray(image.projection_center(),np.float64)
        for index,color in zip(indices,colors):
            candidates[index]['observations'].append((group,color));candidates[index]['centers'].setdefault(group,center)
    for candidate in candidates:
        colored=training_observation_color(candidate['observations'],train_groups,minimum_groups=settings.minimum_seed_stations)
        if colored is None:rejected['training_support']+=1;continue
        angle=maximum_ray_angle_degrees(candidate['xyz'],list(candidate['centers'].values()))
        if angle<settings.minimum_seed_angle_degrees:rejected['triangulation']+=1;continue
        rgb,support=colored
        rows.append((candidate['point_id'],candidate['xyz'],rgb,support,candidate['all_groups'],angle,candidate['error']))
    if len(rows)<settings.minimum_seed_points:raise RuntimeError(f'Insufficient static, multistation, training-colored seed geometry: {len(rows)} points; {rejected}')
    if len(rows)>settings.maximum_seed_points:
        rng=np.random.default_rng(settings.random_seed);chosen=np.sort(rng.choice(len(rows),settings.maximum_seed_points,replace=False));rows=[rows[i] for i in chosen]
    return dict(point_ids=np.asarray([x[0] for x in rows],np.int64),xyz=np.asarray([x[1] for x in rows],np.float64),rgb=np.asarray([x[2] for x in rows],np.uint8),support_station_count=np.asarray([x[3] for x in rows],np.int32),geometry_observed_station_count=np.asarray([x[4] for x in rows],np.int32),triangulation_angle_degrees=np.asarray([x[5] for x in rows],np.float32),reprojection_error_px=np.asarray([x[6] for x in rows],np.float32)),rejected


def write_point_ply(path,xyz,rgb):
    points=np.empty(len(xyz),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    for i,name in enumerate(['x','y','z']):points[name]=xyz[:,i]
    for i,name in enumerate(['red','green','blue']):points[name]=rgb[:,i]
    header=f'ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'
    with Path(path).open('wb') as stream:stream.write(header.encode('ascii'));stream.write(points.tobytes())


def package_dataset(rec,mapping,captures,job_dir,output,settings,gps_report,*,config=None):
    from .sfm import owned_file,copy_file,sha,save
    processing=read_processing_options(config or {})
    output.mkdir(parents=True,exist_ok=False);frames=[];rig_poses={};bindings=[];frame_names={}
    registered=[im for im in rec.images.values() if im.has_pose and im.name in mapping]
    registered_stations=[mapping[im.name]['station_id'] for im in registered]
    if settings.holdout_fraction==0:
        train_groups,heldout_groups=tuple(sorted(set(map(str,registered_stations)))),()
        if len(train_groups)<settings.minimum_physical_stations:
            raise ValueError('Insufficient physical stations for all-training reconstruction')
    else:
        train_groups,heldout_groups=station_split(registered_stations,holdout_fraction=settings.holdout_fraction,seed=settings.random_seed)
    for index,image in enumerate(sorted(registered,key=lambda image:image.name)):
        source=mapping[image.name];camera=rec.cameras[image.camera_id]
        if camera.model_name!='PINHOLE':raise RuntimeError('Calibrated cube camera became non-pinhole')
        c2w_cv=np.eye(4);c2w_cv[:3]=image.cam_from_world().inverse().matrix()
        station_to_world=c2w_cv@np.linalg.inv(camera_to_station(source['camera_to_station_cv']))
        pid=str(source['pano_id'])
        if pid in rig_poses and not np.allclose(station_to_world,rig_poses[pid],atol=1e-6,rtol=1e-8):raise RuntimeError('Fixed face rotations or shared capture center changed during SfM')
        rig_poses[pid]=station_to_world
        suffix=Path(source['file_path']).suffix.lower();target=f'images/{index:06d}{suffix}'
        copy_file(owned_file(job_dir,source['file_path']),output/target)
        frame=dict(file_path=target,station_id=str(source['station_id']),pano_id=pid,face=source['face'],w=int(camera.width),h=int(camera.height),fl_x=float(camera.params[0]),fl_y=float(camera.params[1]),cx=float(camera.params[2]),cy=float(camera.params[3]),transform_matrix=(c2w_cv@CV_FROM_GL).tolist(),camera_to_station_cv=source['camera_to_station_cv'],image_sha256=sha(output/target))
        binding=dict(source_image=source['file_path'],source_image_sha256=frame['image_sha256'],image=target,station_id=frame['station_id'],pano_id=pid,masks={})
        for key in ['mask_path','sfm_mask_path','sky_mask_path','ground_mask_path']:
            original=owned_file(job_dir,source[key]);relative=f'masks/{key.removesuffix("_path")}/{index:06d}.png'
            with Image.open(original) as mask:
                if mask.size!=(frame['w'],frame['h']):raise ValueError('Dataset mask dimensions disagree with camera')
            copy_file(original,output/relative);frame[key]=relative;frame[key.removesuffix('_path')+'_sha256']=sha(output/relative)
            binding['masks'][key]=dict(source=source[key],path=relative,sha256=sha(original))
        frame['foreground_mask_path']=frame['sfm_mask_path'];frame['foreground_mask_sha256']=frame['sfm_mask_sha256']
        # Preserve original-image mask provenance for panorama editing workflows.
        # These fields are absent in legacy datasets and never inferred from RGB.
        for key, folder, suffix in [('original_file_path','original_images',Path(source.get('original_file_path','x.png')).suffix),
                                    ('original_valid_mask_path','masks/original_valid','.png'),
                                    ('edit_alpha_path','masks/edit_alpha','.npy')]:
            if not source.get(key):
                continue
            original=owned_file(job_dir,source[key]);relative=f'{folder}/{index:06d}{suffix}'
            if key=='edit_alpha_path':
                alpha=np.load(original,allow_pickle=False)
                if alpha.shape!=(frame['h'],frame['w']) or not np.isfinite(alpha).all() or np.any(alpha<0) or np.any(alpha>1):
                    raise ValueError('Original edit alpha differs from the calibrated camera grid')
            else:
                with Image.open(original) as original_image:
                    if original_image.size!=(frame['w'],frame['h']):raise ValueError('Original image/mask differs from camera grid')
            copy_file(original,output/relative)
            frame[key]=relative;frame[key.removesuffix('_path')+'_sha256']=sha(output/relative)
            binding['masks'][key]=dict(source=source[key],path=relative,sha256=sha(original))
        frames.append(frame);bindings.append(binding);frame_names[image.name]=target
    arrays,rejected=seed_arrays(rec,mapping,job_dir,set(train_groups),settings)
    np.savez_compressed(output/'init_points.npz',**arrays)
    write_point_ply(output/'init.ply',arrays['xyz'],arrays['rgb'])
    train=[f for f in frames if f['station_id'] in train_groups];heldout=[f for f in frames if f['station_id'] in heldout_groups]
    base=dict(camera_model='PINHOLE',camera_convention='OpenGL_c2w',coordinate_frame='EDN',units='metres',world_up=[0,-1,0],ply_file_path='init.ply')
    base['processing_options']=processing
    save(output/'transforms_train.json',dict(base,frames=train));save(output/'transforms_heldout.json',dict(base,frames=heldout))
    from .sfm_depth import write_depth_sidecar
    depth_manifest=write_depth_sidecar(rec,mapping,frame_names,job_dir,output,arrays,train_groups,heldout_groups,settings,gps_report)
    manifest=dict(schema_version=1,coordinate_frame='EDN',units='metres',camera_convention='OpenGL_c2w',world_up=[0,-1,0],training_station_ids=list(train_groups),heldout_station_ids=list(heldout_groups),seed_color_station_ids=list(train_groups),seed_colors_exclude_heldout=True,geometry_uses_heldout_images=True,evaluation_protocol='Appearance holdout by physical station with known visually reconstructed poses; heldout images can influence geometric feature matching/BA but never seed RGB or photometric optimization',physical_stations=len(train_groups)+len(heldout_groups),capture_rigs=len(rig_poses),training_images=len(train),heldout_images=len(heldout),initial_points=len(arrays['xyz']),rejected_seeds=rejected,seed_policy=dict(minimum_training_station_count=settings.minimum_seed_stations,minimum_angle_degrees=settings.minimum_seed_angle_degrees,maximum_reprojection_error_px=settings.maximum_reprojection_error_px,rgb='3x3 static foreground patches, median per physical group then across training groups'),mask_convention='mask_path: original photometric validity incl sky; sfm_mask_path/foreground_mask_path: excludes dynamic, rig occlusion, sky; white means valid',stations=captures,metric_alignment=gps_report,files={name:sha(output/name) for name in ['transforms_train.json','transforms_heldout.json','init_points.npz','init.ply']},source_bindings=bindings)
    manifest['sparse_depth_manifest']='sparse_depth_manifest.json'
    if settings.holdout_fraction==0:
        manifest.update(training_split='all_train',holdout_fraction=0.,evaluation_status='not_run',
            evaluation_protocol='All registered physical stations provide RGB and actual-track depth supervision; no heldout appearance evaluation',
            geometry_uses_heldout_images=False,quality_comparison_enabled=False)
    manifest['processing_options']=processing
    manifest['mask_conventions']=mask_conventions(processing)
    manifest['mask_convention']='; '.join(f'{key}: {value}' for key,value in manifest['mask_conventions'].items())+'; foreground_mask_path aliases sfm_mask_path. No separate rig-occluder detector or mask is claimed.'
    manifest['sparse_depth_observation_counts']=depth_manifest['observation_counts']
    for name in ['sparse_depth_manifest.json','sparse_depth_observations.npz']:manifest['files'][name]=sha(output/name)
    save(output/'dataset_manifest.json',manifest)
    return {key:manifest[key] for key in ['coordinate_frame','units','camera_convention','physical_stations','capture_rigs','training_images','heldout_images','initial_points','training_station_ids','heldout_station_ids','seed_colors_exclude_heldout']}
