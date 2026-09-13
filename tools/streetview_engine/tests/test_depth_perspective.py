"""Known-camera perspective range, no invented fisheye or camera re-estimation."""
import json
from unittest.mock import patch
import numpy as np
from PIL import Image
import pytest

from tools.streetview_engine import depth_prior as d
from tools.streetview_engine.tests.test_depth_prior import frame, empty_observations
from tools.streetview_engine.imaging import FACES


def raw_plane(f, distance=2.):
    K,_,_=d.camera(f);yy,xx=np.indices((f['h'],f['w']))
    rays=np.stack([xx+.5,yy+.5,np.ones_like(xx)],-1)@np.linalg.inv(K).T
    radial=distance*np.linalg.norm(rays,axis=-1)
    rays/=np.linalg.norm(rays,axis=-1,keepdims=True)
    projection=dict(width=f['w'],height=f['h'],intrinsics_pixel_edge=K.tolist(),camera_to_station_cv=f['camera_to_station_cv'])
    return d.validate_perspective_raw(dict(radial_distance_model=radial,confidence=np.ones_like(radial),geometry_rays=rays),projection)[0]


@pytest.mark.parametrize('face',FACES)
def test_range_sampling_and_metric_calibration_for_every_oriented_face(tmp_path,face):
    f=frame(face=face,side=64)
    Image.new('RGB',(64,64),(90,90,90)).save(tmp_path/f['file_path'])
    Image.new('L',(64,64),255).save(tmp_path/f['sfm_mask_path'])
    raw=raw_plane(f);K,_,_=d.camera(f)
    yy,xx=np.meshgrid(np.linspace(8.5,55.5,15),np.linspace(8.5,55.5,15));xy=np.column_stack([xx.ravel(),yy.ravel()]);n=len(xy)
    obs=dict(frame_name=np.full(n,f['file_path']),station_id=np.full(n,'a'),split=np.full(n,'train'),point_id=np.arange(n),xy=xy,depth_z=np.full(n,4.),support_station_count=np.full(n,3),triangulation_angle_degrees=np.full(n,10.),reprojection_error_px=np.full(n,.2))
    options=d.DepthPriorSettings(input_projection='perspective',output_side=32,maximum_anchor_distance_fraction=.2)
    field,report=d.calibrated_field(f,tmp_path,raw,obs,options)
    assert report['accepted'] and field['valid'].sum()>50
    np.testing.assert_allclose(field['depth_z'][field['valid']],4.,atol=.005)
    # A rotated face samples its own camera, never wraps into another view.
    back=-np.asarray(f['camera_to_station_cv'])[:3,2]
    assert np.isnan(d.sample_raw(raw,'radial_distance_model',back))
    held,_=d.calibrated_field(dict(f,split='heldout'),tmp_path,raw,obs,options)
    assert not held['valid'].any()


def test_resized_known_k_and_official_integer_center_conversion(tmp_path):
    f=frame(side=64);image=tmp_path/f['file_path'];Image.new('RGB',(64,64)).save(image);f['image_sha256']=d._sha(image)
    rgb,projection=d.perspective_input(f,tmp_path,32)
    assert rgb.shape==(32,32,3)
    K=np.array(projection['intrinsics_pixel_edge']);assert K[0,0]==K[0,2]==16
    import torch
    class Model:
        feature_extractor=type('Features',(),{'_unisharp_last_unik3d_output':None})()
        def __call__(self,**kwargs):
            assert kwargs['camera_model']=='pinhole'
            np.testing.assert_allclose(kwargs['camera_intrinsics'].numpy()[0],[[16,0,15.5],[0,16,15.5],[0,0,1]])
            assert kwargs['depth_gt'] is None
            self.feature_extractor._unisharp_last_unik3d_output={'confidence':torch.ones(1,1,32,32)}
            return dict(unik3d_distance=torch.ones(1,1,32,32),distance_layers=torch.ones(1,1,32,32),geometry_rays=torch.zeros(1,3,32,32))
    model=d.UniSharpInference.__new__(d.UniSharpInference);model.torch=torch;model.device=torch.device('cpu');model.model=Model()
    model.infer(rgb,intrinsics_pixel_edge=K)
    assert K[0,2]==16  # Adapter must not mutate the dataset K.
    raw=raw_plane(frame(side=32));wrong=dict(projection);wrong['intrinsics_pixel_edge']=[[16,0,15.5],[0,16,15.5],[0,0,1]]
    with pytest.raises(ValueError,match='rays disagree'):d.validate_perspective_raw(raw,wrong)


def test_perspective_run_six_inferences_and_bound_cache(tmp_path):
    dataset=tmp_path/'sfm/dataset';dataset.mkdir(parents=True);frames=[]
    for face in FACES:
        f=frame(face=face)
        Image.new('RGB',(32,32),(64,96,128)).save(dataset/f['file_path']);Image.new('L',(32,32),255).save(dataset/f['sfm_mask_path'])
        f.update(image_sha256=d._sha(dataset/f['file_path']),sfm_mask_sha256=d._sha(dataset/f['sfm_mask_path']));frames.append(f)
    for split in ['train','heldout']:d._save(dataset/('transforms_'+split+'.json'),dict(frames=frames if split=='train' else []))
    np.savez_compressed(dataset/'sparse_depth_observations.npz',**empty_observations())
    sidecar=dict(status='insufficient_observations',npz='sparse_depth_observations.npz',sha256=d._sha(dataset/'sparse_depth_observations.npz'),coordinate_frame='EDN',units='metres',depth_convention='camera_z',pixel_center_offset=.5,observation_kind='actual_sfm_tracks',geometry_scope='transductive_shared_sfm',train_station_ids=['a'],heldout_station_ids=[],**{'transforms_'+split+'_sha256':d._sha(dataset/('transforms_'+split+'.json')) for split in ['train','heldout']})
    d._save(dataset/'sparse_depth_manifest.json',sidecar)
    manifest=dict(training_station_ids=['a'],heldout_station_ids=[],coordinate_frame='EDN',units='metres',camera_convention='OpenGL_c2w',files={name:d._sha(dataset/name) for name in ['transforms_train.json','transforms_heldout.json','sparse_depth_manifest.json','sparse_depth_observations.npz']})
    d._save(dataset/'dataset_manifest.json',manifest);d._save(tmp_path/'sfm/manifest.json',dict(status='completed',dataset_manifest_sha256=d._sha(dataset/'dataset_manifest.json')))
    original={str(p):d._sha(p) for p in dataset.iterdir()};calls=[]
    class FakeModel:
        metadata=dict(test_double=True)
        def __init__(self,*args):pass
        def infer(self,rgb,intrinsics_pixel_edge=None):
            calls.append(intrinsics_pixel_edge);return raw_plane(frame()),dict(seconds=0.)
        def close(self):pass
    options=dict(input_projection='perspective',perspective_side=32,output_side=16)
    with patch.object(d,'model_assets',return_value={}),patch.object(d,'UniSharpInference',FakeModel):result=d.run(dict(panorama_ids=['a']),tmp_path,options)
    assert len(calls)==6 and all(x is not None for x in calls)
    assert result['new_inference_images']==6 and result['new_inference_captures']==1 and len(result['entries'])==6
    assert result['accepted_pixels']==0 and result['input_projection']=='perspective'
    with patch.object(d,'model_assets',return_value={}),patch.object(d,'UniSharpInference',side_effect=AssertionError('No inference on verified resume')):
        resumed=d.run(dict(panorama_ids=['a']),tmp_path,dict(options,resume_raw=True))
        assert resumed['reused_raw_images']==6 and resumed['new_inference_images']==0
        with pytest.raises(ValueError,match='fingerprint differs'):d.run(dict(panorama_ids=['a']),tmp_path,dict(options,input_projection='spherical',resume_raw=True))
    assert original=={str(p):d._sha(p) for p in dataset.iterdir()}


@pytest.mark.parametrize('kwargs',[dict(input_projection='fisheye'),dict(input_projection=None),dict(perspective_side=True),dict(perspective_side=0)])
def test_bad_projection_rejected(kwargs):
    with pytest.raises(ValueError):d.DepthPriorSettings(**kwargs)
