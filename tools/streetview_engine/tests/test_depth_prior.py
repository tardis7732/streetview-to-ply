"""CPU geometric/contracts tests. No actual-model or GPU claim is made here."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from tools.streetview_engine import depth_prior as d
from tools.streetview_engine.imaging import FACES,cube_camera_to_station_cv


def frame(face='F',station='a',center=(0,0,0),side=32):
    local=cube_camera_to_station_cv(face);cv=local.copy();cv[:3,3]=center
    return dict(file_path=f'{station}_{face}.png',sfm_mask_path=f'{station}_{face}_mask.png',station_id=station,pano_id=station,face=face,split='train',w=side,h=side,fl_x=side/2,fl_y=side/2,cx=side/2,cy=side/2,camera_to_station_cv=local.tolist(),transform_matrix=(cv@np.diag([1,-1,-1,1])).tolist())


def field(station,center,z=4.,side=32):
    f=frame(station=station,center=center,side=side);K,w2c,c2w=d.camera(f)
    yy,xx=np.indices((side,side));rays=np.stack([xx+.5,yy+.5,np.ones_like(xx)],-1)@np.linalg.inv(K).T
    valid=np.ones((side,side),bool);valid[:2]=False;valid[-2:]=False;valid[:,:2]=False;valid[:,-2:]=False
    return dict(frame=f,K=K,w2c=w2c,c2w=c2w,ray=rays,depth_z=np.full((side,side),z,np.float32),valid=valid,confidence=np.full((side,side),.4,np.float32),rgb=np.full((side,side,3),.3,np.float32))


def empty_observations():
    return dict(frame_name=np.array([],str),station_id=np.array([],str),split=np.array([],str),point_id=np.array([],np.int64),xy=np.empty((0,2),np.float64),depth_z=np.array([],float),support_station_count=np.array([],np.int32),triangulation_angle_degrees=np.array([],float),reprojection_error_px=np.array([],float))


class DepthMathTests(unittest.TestCase):
    def test_all_six_calibrated_faces_and_rotated_station_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);side=64;yy,xx=np.indices((side,side));local=np.stack([(xx+.5-side/2)/(side/2),(yy+.5-side/2)/(side/2),np.ones_like(xx)],-1)
            for rotation in [np.eye(3),Rotation.from_euler('xyz',[.3,-.2,.4]).as_matrix()]:
                frames=[]
                for face in FACES:
                    f=frame(face=face,side=side);pose=np.array(f['camera_to_station_cv']);pose[:3,:3]=rotation@pose[:3,:3];f['camera_to_station_cv']=pose.tolist()
                    rays=local@pose[:3,:3].T;rays/=np.linalg.norm(rays,axis=-1,keepdims=True)
                    Image.fromarray(np.round((rays+1)*127.5).astype(np.uint8)).save(root/f['file_path']);frames.append(f)
                erp,report=d.assemble_erp(frames,root,128);expected=(d.erp_rays(128)+1)/2
                self.assertLess(float(np.max(np.abs(erp/255-expected))),.025)
                self.assertEqual(set(report['face_pixel_counts']),set(FACES));self.assertTrue(all(x>0 for x in report['face_pixel_counts'].values()))
            with self.assertRaisesRegex(ValueError,'six'):d.assemble_erp(frames[:-1],root,128)

    def test_longitude_wrap_and_poles_do_not_wrap_vertically(self):
        image=np.broadcast_to(np.linspace(0,1,32)[:,None],(32,64)).copy()
        values=d.sample_erp(image,np.array([[0,-1,0],[0,1,0],[0,0,-1]]))
        np.testing.assert_allclose(values,[0,1,.5],atol=1e-8)
        raw=dict(radial_distance_model=np.ones((32,64)),confidence=np.ones((32,64)),geometry_rays=d.erp_rays(64))
        _,error=d.validate_raw(raw,64);self.assertLess(error,1e-6)
        raw['geometry_rays'][...,1]*=-1
        with self.assertRaisesRegex(ValueError,'rays disagree'):d.validate_raw(raw,64)

    def test_calibration_unit_invariance_and_no_test_point_fitting(self):
        options=d.DepthPriorSettings();ids=np.arange(200);raw=np.linspace(2,30,200);metric=raw*3.5
        report,fit,test=d.calibrate_scale(raw,metric,ids,options)
        self.assertTrue(report['accepted']);self.assertAlmostEqual(report['scale'],3.5)
        changed,_,_=d.calibrate_scale(raw,metric*1024,ids,options)
        self.assertAlmostEqual(changed['scale'],report['scale']*1024)
        bad=metric.copy();bad[test]*=4
        rejected,fit2,test2=d.calibrate_scale(raw,bad,ids,options)
        self.assertFalse(rejected['accepted']);self.assertEqual(rejected['scale'],report['scale'])
        np.testing.assert_array_equal(fit,fit2);np.testing.assert_array_equal(test,test2)
        with self.assertRaisesRegex(ValueError,'Duplicate'):d.calibrate_scale(raw,metric,np.zeros(200,int),options)

    def test_other_station_agreement_duplicate_collapse_and_occlusion(self):
        options=d.DepthPriorSettings(output_side=32);target=field('target',(0,0,0));left=field('left',(-.8,0,0));right=field('right',(.8,0,0))
        groups=d.select_sources(target,[target,left,right],options)
        valid,count,_=d.multiview_support(target,groups,options)
        self.assertGreater(valid.sum(),200);self.assertEqual(int(count.max()),2)
        only_one,count,_=d.multiview_support(target,[('left',[left,left]),('left',[left])],options)
        self.assertFalse(only_one.any());self.assertEqual(int(count.max()),1)
        # Target lies behind the source's visible plane. Occlusion gives no vote.
        blocked=field('right',(.8,0,0),z=2)
        rejected,_,_=d.multiview_support(target,[('left',[left]),('right',[blocked])],options)
        self.assertFalse(rejected.any())
        mismatch=dict(right,rgb=np.full_like(right['rgb'],.9))
        rejected,_,_=d.multiview_support(target,[('left',[left]),('right',[mismatch])],options)
        self.assertFalse(rejected.any())

    def test_multiview_coordinate_rotation_and_1024_scale_invariance(self):
        options=d.DepthPriorSettings(output_side=32);items=[field('target',(0,0,0)),field('left',(-.8,0,0)),field('right',(.8,0,0))]
        initial,count,_=d.multiview_support(items[0],[('left',[items[1]]),('right',[items[2]])],options)
        rotation=Rotation.from_euler('xyz',[.3,-.2,.4]).as_matrix();shift=np.array([31,-17,5])
        for f in items:
            f['c2w'][:3,3]=1024*rotation@f['c2w'][:3,3]+shift;f['c2w'][:3,:3]=rotation@f['c2w'][:3,:3];f['w2c']=np.linalg.inv(f['c2w']);f['depth_z']*=1024
        result,count2,_=d.multiview_support(items[0],[('left',[items[1]]),('right',[items[2]])],options)
        np.testing.assert_array_equal(result,initial);np.testing.assert_array_equal(count2,count)

    def test_calibrated_camera_z_actual_tracks_masks_and_heldout(self):
        options=d.DepthPriorSettings(output_side=32,erp_width=256,maximum_anchor_distance_fraction=.2)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);f=frame();Image.new('RGB',(32,32),(90,90,90)).save(root/f['file_path']);Image.new('L',(32,32),255).save(root/f['sfm_mask_path'])
            rays=d.erp_rays(256);radial=2/np.maximum(np.abs(rays[...,2]),.1)
            raw=dict(radial_distance_model=radial,confidence=np.ones(radial.shape))
            yy,xx=np.meshgrid(np.linspace(4.5,27.5,15),np.linspace(4.5,27.5,15));xy=np.column_stack([xx.ravel(),yy.ravel()]);n=len(xy)
            obs=dict(frame_name=np.full(n,f['file_path']),station_id=np.full(n,'a'),split=np.full(n,'train'),point_id=np.arange(n),xy=xy,depth_z=np.full(n,4.),support_station_count=np.full(n,3),triangulation_angle_degrees=np.full(n,10.),reprojection_error_px=np.full(n,.2))
            accepted,report=d.calibrated_field(f,root,raw,obs,options)
            self.assertTrue(report['accepted']);self.assertGreater(accepted['valid'].sum(),50)
            self.assertLess(float(np.max(np.abs(accepted['depth_z'][accepted['valid']]-4))),.02)
            unsupported,_=d.calibrated_field(f,root,raw,empty_observations(),options);self.assertFalse(unsupported['valid'].any())
            held,_=d.calibrated_field(dict(f,split='heldout'),root,raw,obs,options);self.assertFalse(held['valid'].any())
            Image.new('L',(32,32),0).save(root/f['sfm_mask_path'])
            masked,_=d.calibrated_field(f,root,raw,obs,options);self.assertFalse(masked['valid'].any())

    def test_configuration_rejects_weak_evidence_and_bad_values(self):
        for kwargs in [dict(minimum_other_station_support=1),dict(erp_width=63),dict(maximum_relative_depth_error=float('nan')),dict(maximum_source_stations=1),dict(output_dir='../escape')]:
            with self.assertRaises(ValueError):d.DepthPriorSettings(**kwargs)

    def test_duplicate_frame_point_rows_select_actual_lowest_error_once(self):
        obs=dict(frame_name=np.array(['a','a','a','b']),station_id=np.array(['s','s','s','t']),split=np.array(['train']*4),point_id=np.array([7,7,8,7]),xy=np.array([[4.2,5.3],[4.1,5.1],[9,10],[1,2.]]),depth_z=np.array([8.,8.,7.,9.]),support_station_count=np.array([3]*4),triangulation_angle_degrees=np.array([5.]*4),reprojection_error_px=np.array([.9,.2,.1,.3]))
        result,report=d.deduplicate_observations(obs)
        self.assertEqual(report['duplicate_groups'],1);self.assertEqual(report['removed_rows'],1);self.assertEqual(len(result['point_id']),3)
        np.testing.assert_array_equal(result['xy'][0],obs['xy'][1]);self.assertEqual(int(result['support_station_count'][0]),3)
        reversed_obs={key:value[::-1] for key,value in obs.items()};again,_=d.deduplicate_observations(reversed_obs)
        for key in result:np.testing.assert_array_equal(result[key],again[key])
        conflicting={key:value.copy() for key,value in obs.items()};conflicting['depth_z'][1]=10
        rejected,report=d.deduplicate_observations(conflicting)
        self.assertEqual(report['conflicting_groups_rejected'],1)
        self.assertFalse(np.any((rejected['frame_name']=='a')&(rejected['point_id']==7)))

    def test_operator_assets_are_hash_checked_before_any_model_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);repo=root/'repo';snapshot=root/'snapshot';snapshot.mkdir()
            for name in ['unisharp/models/unisharp_feature.py','UniK3D/unik3d/models/unik3d.py']:
                path=repo/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# synthetic asset-binding fixture')
            checkpoint=root/'checkpoint.pt';checkpoint.write_bytes(b'not loaded by this test');(snapshot/'model.safetensors').write_bytes(b'not loaded');(snapshot/'config.json').write_text('{}')
            options=d.DepthPriorSettings(repo_path=str(repo),checkpoint_path=str(checkpoint),checkpoint_sha256=d._sha(checkpoint),unik3d_snapshot_path=str(snapshot),unik3d_sha256=d._sha(snapshot/'model.safetensors'))
            settings_path=root/'operator.json'
            settings_path.write_text(json.dumps({'depth_prior':d.asdict(options)}),encoding='utf8')
            options=d.DepthPriorSettings(**json.loads(settings_path.read_text(encoding='utf8'))['depth_prior'])
            assets=d.model_assets(options);self.assertEqual(len(assets['source_files']),2)
            checkpoint.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checkpoint hash mismatch'):d.model_assets(options)


class DiagnosticOrchestrationTests(unittest.TestCase):
    def test_empty_evidence_outputs_all_six_faces_without_mutating_dataset(self):
        # This tests packaging via an explicit test double, not model accuracy.
        class FakeModel:
            metadata=dict(test_double=True,forward='synthetic CPU test only')
            def __init__(self,*args):pass
            def infer(self,image):
                h,w=image.shape[:2]
                return dict(radial_distance_model=np.full((h,w),5,np.float32),geometry_rays=d.erp_rays(w),confidence=np.ones((h,w),np.float32)),dict(seconds=0.,first_layer_equals_unik3d=True)
            def close(self):pass
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);dataset=root/'sfm/dataset';dataset.mkdir(parents=True);frames=[]
            for face in FACES:
                f=frame(face);f['split']='train';Image.new('RGB',(32,32),(64,96,128)).save(dataset/f['file_path']);Image.new('L',(32,32),255).save(dataset/f['sfm_mask_path'])
                f['image_sha256']=d._sha(dataset/f['file_path']);f['sfm_mask_sha256']=d._sha(dataset/f['sfm_mask_path']);frames.append(f)
            for split in ['train','heldout']:d._save(dataset/('transforms_'+split+'.json'),dict(frames=frames if split=='train' else []))
            np.savez_compressed(dataset/'sparse_depth_observations.npz',**empty_observations())
            sidecar=dict(status='insufficient_observations',npz='sparse_depth_observations.npz',sha256=d._sha(dataset/'sparse_depth_observations.npz'),coordinate_frame='EDN',units='metres',depth_convention='camera_z',pixel_center_offset=.5,observation_kind='actual_sfm_tracks',geometry_scope='transductive_shared_sfm',train_station_ids=['a'],heldout_station_ids=[],**{'transforms_'+split+'_sha256':d._sha(dataset/('transforms_'+split+'.json')) for split in ['train','heldout']})
            d._save(dataset/'sparse_depth_manifest.json',sidecar)
            manifest=dict(training_station_ids=['a'],heldout_station_ids=[],metric_alignment=dict(status='synthetic_fixture'),coordinate_frame='EDN',units='metres',camera_convention='OpenGL_c2w',files={name:d._sha(dataset/name) for name in ['transforms_train.json','transforms_heldout.json','sparse_depth_manifest.json','sparse_depth_observations.npz']})
            d._save(dataset/'dataset_manifest.json',manifest);d._save(root/'sfm/manifest.json',dict(status='completed',dataset_manifest_sha256=d._sha(dataset/'dataset_manifest.json')))
            immutable={str(path):d._sha(path) for path in dataset.iterdir()}
            with self.assertRaisesRegex(ValueError,'roster'):d.load_inputs(dict(panorama_ids=['other']),root)
            with patch.object(d,'model_assets',return_value={}),patch.object(d,'UniSharpInference',FakeModel):
                result=d.run(dict(panorama_ids=['a']),root,dict(erp_width=64,output_side=16))
            self.assertEqual(result['validation_status'],'insufficient_supported_depth');self.assertEqual(result['accepted_pixels'],0)
            self.assertEqual(set(result['face_counts']),set(FACES));self.assertEqual(len(result['entries']),6)
            for entry in result['entries']:
                with np.load(root/'depth_prior'/entry['npz'],allow_pickle=False) as archive:
                    self.assertFalse(archive['valid'].any());self.assertTrue(np.all(archive['depth_z']==0));self.assertEqual(str(archive['depth_convention']),'camera_z')
                    self.assertEqual(d._sha(root/'depth_prior'/entry['npz']),entry['sha256'])
            self.assertEqual(immutable,{str(path):d._sha(path) for path in dataset.iterdir()})
            with patch.object(d,'model_assets',return_value={}),patch.object(d,'UniSharpInference',side_effect=AssertionError('Verified full raw cache must not construct the model')):
                resumed=d.run(dict(panorama_ids=['a']),root,dict(erp_width=64,output_side=16,resume_raw=True))
            self.assertEqual(resumed['reused_raw_captures'],1);self.assertEqual(resumed['new_inference_captures'],0)
            self.assertIn('dataset_manifest.json',resumed['source_dataset_files_sha256'])
            self.assertEqual(resumed['entries'][0]['status'],'insufficient_support')
            raw_path=root/'depth_prior'/resumed['raw_entries'][0]['npz'];raw_bytes=raw_path.read_bytes();raw_path.write_bytes(raw_bytes+b'changed')
            with patch.object(d,'model_assets',return_value={}):
                with self.assertRaisesRegex(ValueError,'Raw cache NPZ hash mismatch'):d.run(dict(panorama_ids=['a']),root,dict(erp_width=64,output_side=16,resume_raw=True))
            raw_path.write_bytes(raw_bytes)
            raw_path.with_suffix('.json').unlink()
            with patch.object(d,'model_assets',return_value={}):
                with self.assertRaisesRegex(ValueError,'no committed per-capture fingerprint'):d.run(dict(panorama_ids=['a']),root,dict(erp_width=64,output_side=16,resume_raw=True))
            Image.new('L',(32,32),0).save(dataset/frames[0]['sfm_mask_path'])
            with self.assertRaisesRegex(ValueError,'input hash mismatch'):d.load_inputs(dict(panorama_ids=['a']),root)


if __name__=='__main__':unittest.main()
