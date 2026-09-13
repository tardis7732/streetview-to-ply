"""Sparse camera-Z targets from actual SfM tracks, never projected visibility."""
from __future__ import annotations
import numpy as np
from PIL import Image


def observation_arrays(rec, mapping, frame_names, job_dir, seeds, train_groups,
                       heldout_groups, maximum_reprojection_error_px):
    """Export accepted seed-track observations; heldout rows are evaluation-only.

    ``frame_names`` maps COLMAP image names to packaged transform file_path.
    Seed support and parallax are those of accepted *training* observations.
    A projected point lacking an image track never enters this table.
    """
    train_groups=set(map(str,train_groups));heldout_groups=set(map(str,heldout_groups))
    if train_groups & heldout_groups:raise ValueError('Sparse-depth station splits overlap')
    rejected=dict(unbound=0,invalid_track=0,geometry=0,foreground_mask=0)
    by_image={}; rows=[]
    for seed_index, point_id in enumerate(seeds['point_ids']):
        point=rec.point3D(int(point_id))
        for track in point.track.elements:
            by_image.setdefault(int(track.image_id),[]).append((seed_index,int(point_id),int(track.point2D_idx)))
    delta=np.array([(x,y) for y in [-1,0,1] for x in [-1,0,1]])
    for image_id,records in sorted(by_image.items()):
        image=rec.image(image_id);source=mapping.get(image.name)
        if source is None or image.name not in frame_names or not image.has_pose:
            rejected['unbound']+=len(records);continue
        station=str(source['station_id'])
        if station not in train_groups|heldout_groups:
            rejected['unbound']+=len(records);continue
        split='train' if station in train_groups else 'heldout'
        pose=np.asarray(image.cam_from_world().matrix(),np.float64)
        camera=rec.cameras[image.camera_id]
        if camera.model_name!='PINHOLE':raise ValueError('Sparse-depth export requires calibrated PINHOLE cameras')
        fx,fy,cx,cy=np.asarray(camera.params,np.float64)
        with Image.open(job_dir/source['sfm_mask_path']) as file:mask=np.asarray(file.convert('L'))
        if mask.shape!=(camera.height,camera.width):raise ValueError('Sparse-depth foreground mask dimensions disagree')
        for seed_index,point_id,point2d_index in records:
            observation=image.points2D[point2d_index]
            if int(observation.point3D_id)!=point_id:
                rejected['invalid_track']+=1;continue
            xy=np.asarray(observation.xy,np.float64)
            xyz=np.asarray(seeds['xyz'][seed_index],np.float64)
            local=pose[:,:3]@xyz+pose[:,3]
            if not np.isfinite(xy).all() or not np.isfinite(local).all() or local[2]<=0:
                rejected['geometry']+=1;continue
            projected=local[:2]/local[2]*[fx,fy]+[cx,cy]
            error=float(np.linalg.norm(projected-xy))
            if error>maximum_reprojection_error_px:
                rejected['geometry']+=1;continue
            pixel=np.floor(xy).astype(np.int64)
            if not (1<=pixel[0]<camera.width-1 and 1<=pixel[1]<camera.height-1):
                rejected['foreground_mask']+=1;continue
            if not np.all(mask[pixel[1]+delta[:,1],pixel[0]+delta[:,0]]>0):
                rejected['foreground_mask']+=1;continue
            rows.append((frame_names[image.name],station,split,point_id,xy.copy(),float(local[2]),
                         int(seeds['support_station_count'][seed_index]),
                         float(seeds['triangulation_angle_degrees'][seed_index]),error))
    fields=[('frame_name',str),('station_id',str),('split',str),('point_id',np.int64),
            ('xy',np.float64),('depth_z',np.float64),('support_station_count',np.int32),
            ('triangulation_angle_degrees',np.float32),('reprojection_error_px',np.float32)]
    result={key:np.asarray([row[i] for row in rows],dtype=dtype) for i,(key,dtype) in enumerate(fields)}
    result['xy']=result['xy'].reshape(-1,2)
    return result,rejected


def write_depth_sidecar(rec,mapping,frame_names,job_dir,output,seeds,train_groups,
                        heldout_groups,settings,metric_alignment):
    from .sfm import sha,save
    arrays,rejected=observation_arrays(rec,mapping,frame_names,job_dir,seeds,train_groups,
                                    heldout_groups,settings.maximum_reprojection_error_px)
    filename='sparse_depth_observations.npz';np.savez_compressed(output/filename,**arrays)
    manifest=dict(schema_version=1,status='supported_observations' if len(arrays['depth_z']) else 'insufficient_observations',
        coordinate_frame='EDN',units='metres',depth_convention='camera_z',pixel_center_offset=.5,
        observation_kind='actual_sfm_tracks',geometry_scope='transductive_shared_sfm',
        npz=filename,sha256=sha(output/filename),
        transforms_train_sha256=sha(output/'transforms_train.json'),
        transforms_heldout_sha256=sha(output/'transforms_heldout.json'),
        seed_npz_sha256=sha(output/'init_points.npz'),
        train_station_ids=list(train_groups),heldout_station_ids=list(heldout_groups),
        observation_counts={split:int(np.sum(arrays['split']==split)) for split in ['train','heldout']},
        rejected_observations=rejected,metric_alignment=metric_alignment,
        frame_name_convention='Exact file_path in packaged transforms; paths relative to dataset directory',
        xy_convention='Original continuous COLMAP image coordinates; first pixel center is (0.5,0.5); no resizing or rasterization',
        mask_policy='Actual observed 3x3 patch fully inside positive sfm_mask_path; no sky/dynamic observations',
        support_policy='Selected seed training physical-station count and training parallax; duplicate cube faces never increase station support',
        use_policy='Train rows only for optimization. Heldout rows only for transductive geometric consistency assessment, never independent sensor truth; shared SfM geometry and poses can use all images',
        seed_colors_exclude_heldout=True)
    save(output/'sparse_depth_manifest.json',manifest)
    return manifest
