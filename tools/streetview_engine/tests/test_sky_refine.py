"""CPU contract tests; real CUDA smoke requires explicit environment opt-in."""
import copy
from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from tools.streetview_engine.export import sha256, validate_ply, write_model
from tools.streetview_engine.imaging import cube_camera_to_station_cv
from tools.streetview_engine import sky_refine
from tools.streetview_engine.sky_environment import SkyEnvironmentConfig
from tools.streetview_engine.tests.test_sky_environment import fixture, save_json


def model(path, count=4):
    write_model(path, means=np.array([[i*.1,2.,0.] for i in range(count)]),
        log_scales=np.log(np.tile([.5,.01,.5],(count,1))),quats=np.tile([1.,0,0,0],(count,1)),
        opacity_logits=np.full(count,2.),sh0=np.zeros((count,1,3)),shN=np.zeros((count,8,3)))


def test_foreground_payload_is_exact_even_noncanonical_attributes(tmp_path):
    source=tmp_path/'source.ply';sky=tmp_path/'sky.ply';candidate=tmp_path/'candidate.ply'
    model(source);model(sky,2)
    info,_,offset=sky_refine._layout(source)
    rows=np.memmap(source,mode='r+',dtype='<f4',offset=offset,shape=(4,len(info['fields'])))
    rows[:,info['fields'].index('nx')]=3.25
    rows[:,info['fields'].index('rot_0')]=1.234567
    rows.flush();rows._mmap.close()
    source_hash=sha256(source)
    identity=sky_refine.append_frozen_foreground(source,sky,candidate)
    assert sha256(source)==source_hash and identity['unchanged']
    assert identity['source_foreground_payload_sha256']==identity['candidate_foreground_payload_sha256']
    assert validate_ply(candidate)['vertex_count']==6
    original,_=sky_refine.read_gaussian_model(source);combined,_=sky_refine.read_gaussian_model(candidate)
    for key in original:np.testing.assert_array_equal(original[key],combined[key][:4])
    with pytest.raises(FileExistsError):sky_refine.append_frozen_foreground(source,sky,candidate)


def test_sky_appearance_writer_preserves_all_geometry_bytes(tmp_path):
    initial=tmp_path/'initial.ply';fitted=tmp_path/'fitted.ply';model(initial)
    before,info=sky_refine.read_gaussian_model(initial)
    colors=np.full((4,1,3),.25);opacity=np.arange(4,dtype=float)
    sky_refine.write_sky_appearance(initial,fitted,colors,opacity)
    after,_=sky_refine.read_gaussian_model(fitted)
    for key in ['means','scales','quats','shN']:np.testing.assert_array_equal(before[key],after[key])
    np.testing.assert_array_equal(after['sh0'],colors);np.testing.assert_array_equal(after['opacities'],opacity)
    assert sha256(initial)==info['sha256']
    with pytest.raises(ValueError):sky_refine.write_sky_appearance(initial,tmp_path/'bad.ply',colors[:1],opacity)


def reports():
    before=[];after=[];expected=[];holes={}
    def metric(count,sse,covered):return dict(static_pixels=count,static_sse=sse,covered_pixels=covered,alpha_threshold=.5)
    for group in range(2):
        for face in ['F','D']:
            name=f'{group}_{face}'
            binding=dict(frame=name,station_id=f'heldout_{group}',reference_rgb_sha256='rgb_'+name,
                reference_mask_sha256='mask_'+name,reference_sky_mask_sha256='sky_'+name,reference_foreground_mask_sha256='fg_'+name)
            if face=='F':
                left=dict(binding,**metric(100,10.,90),foreground=metric(80,6.,80),sky=metric(20,4.,10))
                right=dict(binding,**metric(100,8.,100),foreground=metric(80,6.,80),sky=metric(20,2.,20))
                counts=dict(all=100,foreground=80,sky=20,down=0)
            else:
                left=dict(binding,**metric(100,3.,100),foreground=metric(100,3.,100),sky=metric(0,0.,0),down=metric(100,3.,100))
                right=copy.deepcopy(left);counts=dict(all=100,foreground=100,sky=0,down=100)
            before.append(left);after.append(right);expected.append(dict(binding,category_pixels=counts))
            holes[name]=dict(all=0,foreground=0,sky=0,down=0)
    source='source_sha';candidate='candidate_sha'
    identity=dict(unchanged=True,source_sha256=source,candidate_sha256=candidate,
        source_foreground_payload_sha256='payload',candidate_foreground_payload_sha256='payload')
    return (dict(status='measured',model_sha256=source,views=before),dict(status='measured',model_sha256=candidate,views=after),
        dict(expected_manifest=dict(discovery_station_ids=['train_a','train_b'],evaluation_frames=expected),
            new_holes=holes,foreground_identity=identity,source_sha256=source,candidate_sha256=candidate))


def test_gate_requires_actual_sky_benefit_and_preserved_foreground():
    before,after,args=reports()
    result=sky_refine.compare_sky_candidate(before,after,**args)
    assert result['accepted'] and result['sky_psnr_gain_db']>3 and result['sky_coverage_gain']==.5
    assert 'geometric' in result['scope']


@pytest.mark.parametrize('fault',['missing_view','duplicate','missing_foreground','changed_down','hole','mask_hash','source_hash','changed_geometry','same_sky','coverage','threshold','bool_count','heldout_leak'])
def test_gate_fails_closed_for_unsafe_or_unbound_comparison(fault):
    before,after,args=reports()
    if fault=='missing_view':after['views'].pop()
    elif fault=='duplicate':after['views'][-1]=copy.deepcopy(after['views'][0])
    elif fault=='missing_foreground':after['views'][0].pop('foreground')
    elif fault=='changed_down':after['views'][1]['down']['static_sse']=30.
    elif fault=='hole':args['new_holes']['0_F']['foreground']=1
    elif fault=='mask_hash':after['views'][0]['reference_sky_mask_sha256']='changed'
    elif fault=='source_hash':before['model_sha256']='changed'
    elif fault=='changed_geometry':args['foreground_identity']['candidate_foreground_payload_sha256']='changed'
    elif fault=='same_sky':
        for index,row in enumerate(after['views']):row['sky']=copy.deepcopy(before['views'][index]['sky'])
    elif fault=='coverage':
        for index,row in enumerate(after['views']):row['sky']['covered_pixels']=before['views'][index]['sky']['covered_pixels']
    elif fault=='threshold':after['views'][0]['sky']['alpha_threshold']=.1
    elif fault=='bool_count':after['views'][0]['foreground']['static_pixels']=True
    elif fault=='heldout_leak':args['expected_manifest']['discovery_station_ids'].append('heldout_0')
    assert not sky_refine.compare_sky_candidate(before,after,**args)['accepted']


def test_empty_evaluation_and_no_down_cannot_approve():
    before,after,args=reports();args['expected_manifest']['evaluation_frames']=[]
    assert not sky_refine.compare_sky_candidate(before,after,**args)['accepted']
    before,after,args=reports()
    for row in args['expected_manifest']['evaluation_frames']:row['category_pixels']['down']=0
    for report in [before,after]:
        for row in report['views']:row.pop('down',None)
    result=sky_refine.compare_sky_candidate(before,after,**args)
    assert 'insufficient_down_evaluation' in result['reasons']


def test_already_covered_sky_still_needs_real_color_improvement():
    before,after,args=reports()
    for report in [before,after]:
        for row in report['views']:row['sky']['covered_pixels']=row['sky']['static_pixels']
    assert sky_refine.compare_sky_candidate(before,after,**args)['accepted']


def test_heldout_poses_enlarge_radius_without_reading_heldout_photos(tmp_path):
    train=fixture(tmp_path)
    heldout=copy.deepcopy(train[:1]);heldout[0]['file_path']='absent-heldout-photo.png'
    heldout[0]['transform_matrix'][0][3]=10.
    foreground=dict(means=np.array([[0.,0,2.]]),scales=np.log(np.full((1,3),.1)))
    config=SkyEnvironmentConfig(grid_size=8,tangent_sigma_cells=.35)
    sky,report=sky_refine._expanded_sky(tmp_path,train,heldout,foreground,config)
    assert len(sky.means)>0
    assert report['expanded_view_region']['radius_m']==10.
    assert report['expanded_view_region']['maximum_parallax_degrees_bound']<=config.maximum_parallax_degrees+1e-10
    assert report['expanded_view_region']['heldout_appearance_used'] is False


def test_disabled_stage_and_invalid_settings_do_not_start_cuda(tmp_path):
    assert sky_refine.run({},tmp_path,{})['status']=='disabled'
    with pytest.raises(ValueError):sky_refine.run({},tmp_path,{'sky_refine':{'enabled':'true'}})
    for values in [dict(steps=True),dict(resolution=0),dict(min_evaluation_stations=1),dict(color_lr=0.),dict(minimum_sky_psnr_gain_db=0.),dict(require_down_evaluation='yes')]:
        with pytest.raises(ValueError):sky_refine.SkyRefineConfig(**values)


def cuda_fixture(root):
    """Synthetic completed foreground only, clearly scoped to renderer testing."""
    dataset=root/'sfm/dataset';dataset.mkdir(parents=True)
    train=[];heldout=[];photos={}
    for index,x in enumerate([-1.5,-.5,.5,1.5]):
        for face in ['F','D']:
            name=f'{index}_{face}';is_sky=face=='F'
            rgb=np.full((64,64,3),0,np.uint8);rgb[:]=[96,144,220] if is_sky else [128,128,128]
            arrays=dict(file_path=rgb,mask_path=np.full((64,64),255,np.uint8),
                sky_mask_path=np.full((64,64),255 if is_sky else 0,np.uint8),
                foreground_mask_path=np.full((64,64),0 if is_sky else 255,np.uint8))
            paths={}
            for key,value in arrays.items():
                path=name+'_'+key+'.png';Image.fromarray(value).save(dataset/path)
                paths[key]=path;photos[path]=sha256(dataset/path)
            camera=cube_camera_to_station_cv(face)
            pose=camera@np.diag([1.,-1.,-1.,1.]);pose[0,3]=x
            row=dict(**paths,station_id='physical_'+str(index),pano_id=str(index),face=face,w=64,h=64,fl_x=32.,fl_y=32.,cx=32.,cy=32.,
                transform_matrix=pose.tolist(),camera_to_station_cv=camera.tolist(),image_sha256=photos[paths['file_path']],
                mask_sha256=photos[paths['mask_path']],sky_mask_sha256=photos[paths['sky_mask_path']],
                foreground_mask_sha256=photos[paths['foreground_mask_path']])
            (train if index<2 else heldout).append(row)
    base=dict(camera_convention='OpenGL_c2w',coordinate_frame='EDN',units='metres')
    save_json(dataset/'transforms_train.json',dict(base,frames=train));save_json(dataset/'transforms_heldout.json',dict(base,frames=heldout))
    dm=dict(base,training_station_ids=['physical_0','physical_1'],heldout_station_ids=['physical_2','physical_3'],
        seed_color_station_ids=['physical_0','physical_1'],seed_colors_exclude_heldout=True,
        files={name:sha256(dataset/name) for name in ['transforms_train.json','transforms_heldout.json']})
    save_json(dataset/'dataset_manifest.json',dm)
    training_root=root/'training';training_root.mkdir();model(training_root/'model.ply')
    inputs={name:sha256(dataset/name) for name in ['dataset_manifest.json','transforms_train.json','transforms_heldout.json']}
    inputs['photos_and_masks']=photos
    save_json(training_root/'manifest.json',dict(status='completed',artifact=validate_ply(training_root/'model.ply'),
        provenance=dict(inputs=inputs),fixture='synthetic renderer test; not a trained real-scene quality result'))
    return training_root,dataset


@pytest.mark.skipif(os.environ.get('STREETVIEW_SKY_CUDA_SMOKE')!='1',reason='Explicit opt-in required; normal unit tests never launch GPU work')
def test_real_cuda_sky_fit_exports_fixed_foreground_and_measured_reports(tmp_path):
    torch=pytest.importorskip('torch');pytest.importorskip('gsplat')
    assert torch.cuda.is_available(),'Opt-in smoke requires actual CUDA'
    training_root,dataset=cuda_fixture(tmp_path)
    original=sha256(training_root/'model.ply')
    result=sky_refine.refine_sky(training_root,dataset,tmp_path/'sky_candidate',
        config=sky_refine.SkyRefineConfig(steps=2,resolution=32,log_every=1),
        sky_config=SkyEnvironmentConfig(grid_size=8,tangent_sigma_cells=.35))
    assert result['status']=='completed' and result['nonzero_sky_gradient']
    assert result['baseline_unchanged'] and result['promoted'] is False
    assert sha256(training_root/'model.ply')==original
    assert result['foreground_identity']['unchanged']
    assert (tmp_path/'sky_candidate'/result['candidate_ply']).is_file()
    before=json.loads((tmp_path/'sky_candidate/source_reference/heldout_metrics.json').read_text())
    after=json.loads((tmp_path/'sky_candidate/candidate/heldout_metrics.json').read_text())
    assert len(before['views'])==len(after['views'])==4
    assert all('sky' in row and 'foreground' in row for row in after['views'])
