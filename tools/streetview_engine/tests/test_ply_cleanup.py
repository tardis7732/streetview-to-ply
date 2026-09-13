import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tools.streetview_engine import export
from tools.streetview_engine.ply_cleanup import (main, read_crop_options, run_ply_cleanup,
    verify_cleanup_report, write_output_camera_reference)
from tools.streetview_engine.size_filter import run_size_filter

SIZE = dict(enabled=True,max_sigma_camera_radius_ratio=.5)
CROP = dict(enabled=True,radius_camera_radius_ratio=1.)


def fixture(root, scale=1., rotation=None, translation=None, duplicates=0):
    root.mkdir(parents=True)
    rotation = np.eye(3) if rotation is None else np.asarray(rotation)
    translation = np.zeros(3) if translation is None else np.asarray(translation)
    frames=[]
    for station,x in enumerate((-2.,2.)):
        pose=np.eye(4); pose[:3,:3]=rotation; pose[:3,3]=scale*(rotation@np.array([x,0,0]))+translation
        frames.append(dict(station_id=str(station),face='F',transform_matrix=pose.tolist()))
    frames.extend(copy.deepcopy(frames[0]) for _ in range(duplicates))
    frames.reverse()
    camera=root/'cameras.json'
    up=rotation@np.array([0.,-1.,0.])
    export.write_json(camera,dict(coordinate_frame='EDN',units='metres',camera_convention='OpenGL_c2w',
        world_up=up.tolist(),w=1000,fl_x=500,frames=frames))
    ply=root/'source.ply'
    means=np.array([[-.5,0,0],[1.,1000,0],[3.,0,0],[0,0,0]])
    export.write_model(ply,means=scale*(means@rotation.T)+translation,log_scales=np.log(np.array([[.1]*3]*3+[[10.]*3])*scale),
        quats=np.tile([1.,0,0,0],(4,1)),opacity_logits=np.arange(4),sh0=np.arange(12).reshape(4,3)/10,shN=np.zeros((4,0,3)))
    payload=ply.read_bytes().replace(b'world_up 0 -1 0',('world_up '+' '.join(str(x) for x in up)).encode())
    ply.write_bytes(payload)
    return ply,camera


def test_crop_union_and_vertical_unlimited_coordinate_scale_invariance(tmp_path):
    for index,scale in enumerate((1e-4,1.,1e3)):
        rotation=np.array([[0,1.,0],[0,0,1.],[1.,0,0]])
        ply,camera=fixture(tmp_path/str(index),scale=scale,rotation=rotation,translation=scale*np.array([31.,-5,13]),duplicates=index*20)
        original=ply.read_bytes()
        result=run_ply_cleanup(ply,camera,tmp_path/f'out{index}',SIZE,CROP)
        assert (result['removed_rows'],result['remaining_rows'])==(2,2)
        assert result['size_removed_rows']==1 and result['crop_additional_removed_rows']==1
        assert result['crop_geometry']['height']=='unlimited'
        with np.load(result['selection']['path']) as selection:
            np.testing.assert_array_equal(selection['removed_indices'],[2,3])
        assert verify_cleanup_report(result,ply,result['artifact']['path'],camera,result['selection']['path'])['exact_retained_row_bytes']
        assert ply.read_bytes()==original


def test_disabled_crop_is_identical_legacy_size_filter_and_all_disabled_copy(tmp_path):
    ply,camera=fixture(tmp_path/'input')
    legacy=run_size_filter(ply,camera,tmp_path/'legacy',SIZE)
    cleanup=run_ply_cleanup(ply,camera,tmp_path/'cleanup',SIZE,{})
    assert legacy['artifact']['sha256']==cleanup['artifact']['sha256']
    untouched=run_ply_cleanup(ply,camera,tmp_path/'full',dict(SIZE,enabled=False),{})
    assert untouched['removed_rows']==0 and Path(untouched['artifact']['path']).read_bytes()==ply.read_bytes()


def test_crop_only_leaves_large_gaussian_and_output_camera_chains(tmp_path):
    ply,camera=fixture(tmp_path/'input')
    # Legacy PLY format with no coordinate declaration needs an explicit source binding.
    ply.write_bytes(ply.read_bytes().replace(b'comment coordinates EDN metres; world_up 0.0 -1.0 0.0\n',b''))
    doc=json.loads(camera.read_text());doc['ply_binding']=dict(sha256=export.sha256(ply),coordinate_frame='EDN',units='metres')
    export.write_json(camera,doc)
    result=run_ply_cleanup(ply,camera,tmp_path/'crop',dict(SIZE,enabled=False),CROP)
    assert (result['removed_rows'],result['remaining_rows'])==(1,3)
    rebound=write_output_camera_reference(result,camera,tmp_path/'bound.json')
    next_result=run_ply_cleanup(result['artifact']['path'],rebound,tmp_path/'again',dict(SIZE,enabled=False),{})
    assert result['artifact']['sha256']==next_result['artifact']['sha256']


def test_unknown_up_abstains_and_down_faces_can_define_up(tmp_path):
    ply,camera=fixture(tmp_path/'input')
    ply.write_bytes(ply.read_bytes().replace(b'; world_up 0.0 -1.0 0.0',b''))
    doc=json.loads(camera.read_text());doc.pop('world_up');export.write_json(camera,doc)
    with pytest.raises(ValueError,match='world_up'):
        run_ply_cleanup(ply,camera,tmp_path/'unknown',SIZE,CROP)
    assert not (tmp_path/'unknown').exists()
    # Down GL forward = +Y, so GL local +Z points upward (-Y).
    rotation=[[1,0,0],[0,0,-1],[0,1,0]]
    for frame in doc['frames']:
        frame['face']='D'
        pose=np.asarray(frame['transform_matrix']);pose[:3,:3]=rotation;frame['transform_matrix']=pose.tolist()
    export.write_json(camera,doc)
    result=run_ply_cleanup(ply,camera,tmp_path/'down',SIZE,CROP)
    assert result['crop_geometry']['world_up']==[0.,-1.,0.]
    assert result['remaining_rows']==2


@pytest.mark.parametrize('value',[None,[],{'enabled':1},{'radius_camera_radius_ratio':True},
    {'radius_camera_radius_ratio':float('nan')},{'radius_camera_radius_ratio':0},{'radius_camera_radius_ratio':101},{'unknown':True}])
def test_invalid_crop_options(value):
    with pytest.raises(ValueError):read_crop_options({'crop':value})


def test_empty_crop_and_selection_tampering_are_rejected(tmp_path):
    ply,camera=fixture(tmp_path/'input')
    with pytest.raises(ValueError,match='all Gaussians'):
        run_ply_cleanup(ply,camera,tmp_path/'empty',SIZE,dict(CROP,radius_camera_radius_ratio=.01))
    assert not (tmp_path/'empty').exists()
    result=run_ply_cleanup(ply,camera,tmp_path/'crop',SIZE,CROP)
    np.savez_compressed(result['selection']['path'],removed_indices=np.array([0,3],dtype='<i8'),source_vertex_count=np.array(4,dtype='<i8'))
    with pytest.raises(ValueError,match='binding changed'):
        verify_cleanup_report(result,ply,result['artifact']['path'],camera,result['selection']['path'])


def test_cli_crop_and_default_preserve(tmp_path,capsys):
    ply,camera=fixture(tmp_path/'input')
    assert main(['--input',str(ply),'--cameras',str(camera),'--output-dir',str(tmp_path/'cli'),
        '--no-size-filter','--crop-radius-ratio','1'])==0
    result=json.loads(capsys.readouterr().out)
    assert result['remaining_rows']==3 and result['training_started'] is False
    assert Path(result['output_camera_json']).is_file()
    config=tmp_path/'config.json'
    config.write_text(json.dumps(dict(source_ply='input/source.ply',camera_json='input/cameras.json',size_filter_enabled=False)))
    assert main(['--config',str(config),'--output-dir',str(tmp_path/'full')])==0
    result=json.loads(capsys.readouterr().out)
    assert result['artifact']['sha256']==export.sha256(ply)
