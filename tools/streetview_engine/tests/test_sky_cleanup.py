import copy
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools.streetview_engine import export
from tools.streetview_engine.sky_cleanup import (read_sky_cleanup_options,run_sky_cleanup,
    select_cleanup_rows,verify_sky_cleanup_report,build_mask_manifest)


def fixture(root, *, depth=False):
    root.mkdir(parents=True)
    width=height=32;K=np.array([[10.,0,16],[0,10,16],[0,0,1]])
    frames=[];records=[]
    for station,x in enumerate((-1.,0.,1.)):
        pose=np.eye(4);pose[0,3]=x
        image=root/f'image{station}.png';Image.new('RGB',(width,height),'gray').save(image)
        sky=np.zeros((height,width),np.uint8);sky[:16]=255
        sky_path=root/f'sky{station}.png';Image.fromarray(sky).save(sky_path)
        valid=root/f'valid{station}.png';Image.fromarray(np.full((height,width),255,np.uint8)).save(valid)
        alpha=np.zeros((height,width),np.float32);alpha[:,22:28]=1
        alpha_path=root/f'alpha{station}.npy';np.save(alpha_path,alpha)
        frame=dict(station_id=str(station),face='F',file_path=image.name,transform_matrix=pose.tolist(),w=width,h=height,
            fl_x=10.,fl_y=10.,cx=16.,cy=16.,image_sha256=export.sha256(image))
        frames.append(frame)
        row=dict(image=image.name,station_id=str(station),face='F',width=width,height=height,K=K.tolist(),camera_from_world=np.linalg.inv(pose).tolist())
        for name,path in [('source_rgb',image),('source_sky',sky_path),('source_generic_valid',valid),('source_edit_alpha',alpha_path)]:
            row[name+'_path']=path.name;row[name+'_sha256']=export.sha256(path)
        if depth:
            target=root/f'depth{station}.npz'
            np.savez(target,depth_z=np.full((height,width),2.),evidence_valid=np.ones((height,width),bool),K=K,
                source_image_sha256=np.asarray(frame['image_sha256']),station_id=np.asarray(str(station)),image=np.asarray(image.name),
                world_frame=np.asarray('EDN'),unit=np.asarray('metres'),depth_convention=np.asarray('camera_z'),pixel_center_offset=np.asarray(.5),
                camera_from_world=np.linalg.inv(pose),metric_calibration_accepted=np.asarray(False))
            row.update(depth_path=target.name,depth_sha256=export.sha256(target))
        records.append(row)
    camera=root/'cameras.json'
    export.write_json(camera,dict(coordinate_frame='EDN',units='metres',camera_convention='OpenCV_c2w',world_up=[0,-1,0],frames=frames))
    manifest=root/'inputs.json'
    export.write_json(manifest,dict(schema_version=1,cameras_sha256=export.sha256(camera),frames=records))
    means=np.array([[0,-5,10],[0,5,10],[8,-5,10],[0,0,-10],[0,5,10.]])
    source=root/'source.ply'
    export.write_model(source,means=means,log_scales=np.log(np.array([[.01]*3]*4+[[2.]*3])),
        quats=np.tile([1.,0,0,0],(5,1)),opacity_logits=np.arange(5),sh0=np.arange(15).reshape(5,3)/10,shN=np.zeros((5,0,3)))
    return source,camera,manifest


def test_native_masks_alpha_unknown_and_hard_cap_with_exact_export(tmp_path):
    source,camera,manifest=fixture(tmp_path/'input')
    report=run_sky_cleanup(source,camera,manifest,tmp_path/'result',dict(protect_down=False))
    assert report['removed_rows']==2 and report['remaining_rows']==3
    with np.load(report['selection']['path']) as data: np.testing.assert_array_equal(data['removed_indices'],[0,4])
    with np.load(report['evidence']['path']) as data:
        np.testing.assert_array_equal(data['sky_counts'],[3,0,0,0,0])
        np.testing.assert_array_equal(data['non_sky_counts'],[0,3,0,0,3])
    assert report['ground_plane_inferred'] is False
    assert verify_sky_cleanup_report(report,source,report['artifact']['path'],camera,report['evidence']['path'],report['selection']['path'])['exact_retained_row_bytes']
    assert (tmp_path/'result/artifact_cameras.json').is_file()


def test_depth_only_abstains_non_sky_and_invalid_retains_raw(tmp_path):
    source,camera,manifest=fixture(tmp_path/'input',depth=True)
    report=run_sky_cleanup(source,camera,manifest,tmp_path/'depth',dict(protect_down=False,depth_non_sky_abstention=True))
    with np.load(report['evidence']['path']) as data:
        np.testing.assert_array_equal(data['sky_counts'],[3,0,0,0,0])
        np.testing.assert_array_equal(data['non_sky_counts'],[0,0,0,0,0])
    assert report['all_depth_metric_calibration_accepted'] is False
    assert 'heuristic' in report['depth_scope']
    doc=json.loads(manifest.read_text());row=doc['frames'][0];path=manifest.parent/row['depth_path']
    with np.load(path) as data: arrays={name:data[name].copy() for name in data.files}
    arrays['evidence_valid'][:]=False;np.savez(path,**arrays)
    row['depth_sha256']=export.sha256(path);export.write_json(manifest,doc)
    result=run_sky_cleanup(source,camera,manifest,tmp_path/'invalid',dict(protect_down=False,depth_non_sky_abstention=True))
    with np.load(result['evidence']['path']) as data: np.testing.assert_array_equal(data['non_sky_counts'],[0,1,0,0,1])


def test_policy_invariant_under_similarity_and_duplicate_station_faces(tmp_path):
    _,camera,_=fixture(tmp_path/'input')
    frames=json.loads(camera.read_text())['frames']
    means=np.array([[0,-1,2.],[0,1,2.],[0,1,2.],[0,-1,2.],[0,-1,2.]])
    scales=np.log(np.array([[.01]*3,[.2]*3,[2.]*3,[.2]*3,[.2]*3]))
    counts=dict(sky=np.array([3,0,0,0,3]),non_sky=np.array([0,3,3,3,0]),down=np.array([0,0,1,1,1]))
    options=read_sky_cleanup_options()
    baseline,details=select_cleanup_rows(means,scales,frames,[0,-1,0],counts,options)
    np.testing.assert_array_equal(baseline,[True,False,True,False,True])
    rng=np.random.default_rng(18);rotation,_=np.linalg.qr(rng.normal(size=(3,3)))
    if np.linalg.det(rotation)<0:rotation[:,0]*=-1
    for scale in (1e-4,.3,1000.):
        shift=scale*np.array([31,-12,17.]);new=[]
        for frame in frames:
            frame=copy.deepcopy(frame);pose=np.array(frame['transform_matrix']);pose[:3,:3]=rotation@pose[:3,:3]
            pose[:3,3]=scale*(rotation@pose[:3,3])+shift;frame['transform_matrix']=pose.tolist();new.append(frame)
        new=new[::-1]+[copy.deepcopy(new[0])]*23
        actual,_=select_cleanup_rows(scale*(means@rotation.T)+shift,scales+math_log(scale),new,rotation@np.array([0,-1,0]),counts,options)
        np.testing.assert_array_equal(actual,baseline)
    uncapped,_=select_cleanup_rows(means,scales,frames,[0,-1,0],counts,dict(options,hard_size_ratio=None))
    assert not uncapped[2]  # hard cap overrides both Down and below-camera preservation.


def math_log(value):
    return float(np.log(value))


@pytest.mark.parametrize('change',['hash','grid','roster','depth_pose','no_down'])
def test_malformed_native_evidence_fails_before_output(tmp_path,change):
    source,camera,manifest=fixture(tmp_path/'input',depth=True)
    doc=json.loads(manifest.read_text());options=dict(protect_down=False,depth_non_sky_abstention=True)
    if change=='hash':doc['frames'][0]['source_sky_sha256']='0'*64
    elif change=='grid':doc['frames'][0]['K'][0][0]=11
    elif change=='roster':doc['frames'].pop()
    elif change=='no_down':options['protect_down']=True
    else:
        row=doc['frames'][0];path=manifest.parent/row['depth_path']
        with np.load(path) as data:arrays={name:data[name].copy() for name in data.files}
        arrays['camera_from_world'][0,3]+=1;np.savez(path,**arrays);row['depth_sha256']=export.sha256(path)
    export.write_json(manifest,doc)
    with pytest.raises(ValueError):run_sky_cleanup(source,camera,manifest,tmp_path/'out',options)
    assert not (tmp_path/'out').exists()


def test_bound_input_tampering_is_detected_after_export(tmp_path):
    source,camera,manifest=fixture(tmp_path/'input')
    report=run_sky_cleanup(source,camera,manifest,tmp_path/'result',dict(protect_down=False))
    manifest.write_text('{}')
    with pytest.raises(ValueError,match='bound input changed'):
        verify_sky_cleanup_report(report,source,report['artifact']['path'],camera,report['evidence']['path'],report['selection']['path'])


def test_registered_dataset_adapter_uses_original_validity_and_preserves_depth_binding(tmp_path):
    source,camera,manifest=fixture(tmp_path/'dataset',depth=True)
    original=json.loads(manifest.read_text());doc=json.loads(camera.read_text())
    for frame,row in zip(doc['frames'],original['frames']):
        frame.update(original_file_path=row['source_rgb_path'],original_file_sha256=row['source_rgb_sha256'],
                     original_valid_mask_path=row['source_generic_valid_path'],original_valid_mask_sha256=row['source_generic_valid_sha256'],
                     edit_alpha_path=row['source_edit_alpha_path'],edit_alpha_sha256=row['source_edit_alpha_sha256'],
                     sky_mask_path=row['source_sky_path'],sky_mask_sha256=row['source_sky_sha256'])
        # Training validity deliberately differs: adapter must never substitute it.
        frame['mask_path']=row['source_sky_path'];frame['mask_sha256']=row['source_sky_sha256']
    export.write_json(camera,doc)
    depth_manifest=manifest.parent/'depth.json'
    export.write_json(depth_manifest,dict(status='completed',cameras_sha256=export.sha256(camera),frames=[
        {key:row[key] for key in ('image','station_id','face','depth_path','depth_sha256')} for row in original['frames']]))
    target=build_mask_manifest(manifest.parent,camera,tmp_path/'native.json',depth_manifest)
    adapted=json.loads(target.read_text())
    assert adapted['frames'][0]['source_generic_valid_path'].endswith('valid0.png')
    assert adapted['frames'][0]['source_original_path'].endswith('image0.png')
    report=run_sky_cleanup(source,camera,target,tmp_path/'out',dict(protect_down=False,depth_non_sky_abstention=True))
    assert report['removed_rows']==2 and str(depth_manifest.resolve()) in report['input_bindings']
    depth_manifest.write_text('{}')
    with pytest.raises(ValueError,match='bound input changed'):
        verify_sky_cleanup_report(report,source,report['artifact']['path'],camera,report['evidence']['path'],report['selection']['path'])


def test_adapter_never_substitutes_training_mask_for_missing_original_validity(tmp_path):
    _,camera,_=fixture(tmp_path/'dataset')
    with pytest.raises(ValueError,match='original cleanup input'):
        build_mask_manifest(camera.parent,camera,tmp_path/'bad.json')
    assert not (tmp_path/'bad.json').exists()
