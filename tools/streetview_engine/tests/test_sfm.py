import importlib.util
import hashlib
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from tools.streetview_engine.sfm import SfMSettings,prepare_inputs,extract_and_configure_rig,visual_reconstruction,gps_adjust,sha,validate_prepared_selection
from tools.streetview_engine.sfm import recover_capture_rigs,reuse_visual_cache,save
from tools.streetview_engine.sfm_geometry import camera_to_station,candidate_capture_pairs,gps_local_enu,fit_horizontal_similarity,training_observation_color,maximum_ray_angle_degrees,EDN_FROM_ENU
from tools.streetview_engine.sfm_dataset import package_dataset
from tools.streetview_geometry.contracts import station_split


def rig_recovery_fixture(pc,root,points_per_camera=20,disjoint=True):
    """Real COLMAP database/model with disjoint per-face 2D/3D support."""
    (root/'sfm').mkdir(parents=True)
    db=pc.Database.open(root/'sfm/database.db')
    options=pc.SyntheticDatasetOptions(num_rigs=1,num_cameras_per_rig=3,num_frames_per_rig=4,
        num_points3D=90,num_points2D_without_point3D=0,sensor_from_rig_translation_stddev=0.,
        sensor_from_rig_rotation_stddev=0.,camera_width=96,camera_height=96,
        camera_model_id=pc.CameraModelId.PINHOLE,camera_params=[70,70,48,48],camera_has_prior_focal_length=True)
    pc.set_random_seed(0)
    rec=pc.synthesize_dataset(options,db);frame=max(rec.reg_frame_ids());truth=rec.frame(frame).rig_from_world.matrix().copy()
    targets=sorted([im for im in rec.images.values() if im.frame_id==frame],key=lambda im:im.camera_id)
    ids=sorted(rec.points3D)
    for sensor,image in enumerate(targets):
        start=sensor*points_per_camera if disjoint else 0
        keep=set(ids[start:start+points_per_camera])
        for other in rec.images.values():
            if other.frame_id==frame:continue
            geometry=db.read_two_view_geometry(image.image_id,other.image_id)
            matches=geometry.inlier_matches
            good=[int(image.points2D[int(pair[0])].point3D_id) in keep for pair in matches]
            geometry.inlier_matches=matches[np.asarray(good,bool)]
            db.update_two_view_geometry(image.image_id,other.image_id,geometry)
    mapping={im.name:dict(pano_id='capture_'+str(im.frame_id),station_id='station_'+str(im.frame_id)) for im in rec.images.values()}
    rec.deregister_frame(frame);db.close()
    return rec,mapping,frame,truth


def cached_sfm_fixture(pc,root):
    rec,mapping,_,_=rig_recovery_fixture(pc,root)
    db=pc.Database.open(root/'sfm/database.db');rows=[];feature_rows=[]
    try:
        for im in db.read_all_images():
            row=mapping[im.name];camera=db.read_camera(im.camera_id)
            image_path=f'collection/{im.image_id}.png';mask_path=f'prepared/{im.image_id}.png'
            for path,color,mode in [(image_path,(40,60,80),'RGB'),(mask_path,255,'L')]:
                (root/path).parent.mkdir(parents=True,exist_ok=True);Image.new(mode,(96,96),color).save(root/path)
            row.update(file_path=image_path,source_sha256=sha(root/image_path),sfm_mask_path=mask_path,sfm_mask_sha256=sha(root/mask_path),
                w=camera.width,h=camera.height,fl_x=camera.params[0],fl_y=camera.params[1],cx=camera.params[2],cy=camera.params[3],camera_to_station_cv=np.eye(4).tolist())
            rows.append(row);feature_rows.append(dict(image=im.name))
    finally:db.close()
    save(root/'prepared/manifest.json',dict(status='complete',frames=rows))
    settings=SfMSettings(device='cpu',threads=1)
    signature=dict(prepared_sha256=sha(root/'prepared/manifest.json'),job_config_sha256=hashlib.sha256(b'frozen fixture').hexdigest(),settings=asdict(settings))
    # Legacy source lacks the new rescue options; canonical defaults must work.
    signature['settings'].pop('recovery_rounds');signature['settings'].pop('recovery_seconds')
    save(root/'sfm/input_signature.json',signature);save(root/'sfm/failure.json',dict(status='failed',**signature))
    save(root/'sfm/feature_report.json',dict(pycolmap_version=pc.__version__,images=feature_rows,feature_device='cpu',cpu_fallback=False))
    (root/'sfm/match_pairs.txt').write_text('actual cached pairs are in the database\n',encoding='utf8')
    (root/'sfm/visual_models/0').mkdir(parents=True);rec.write(root/'sfm/visual_models/0')
    return mapping,settings,signature


class NumericalSfMTests(unittest.TestCase):
    def test_prepared_status_and_exact_frozen_roster_binding(self):
        prepared=dict(status='complete',stations=[dict(pano_id='a'),dict(pano_id='b')],frames=[dict(pano_id='a'),dict(pano_id='b')])
        validate_prepared_selection(dict(panorama_ids=['a','b']),prepared)
        with self.assertRaisesRegex(ValueError,'not complete'):validate_prepared_selection(dict(panorama_ids=['a','b']),dict(prepared,status='partial'))
        with self.assertRaisesRegex(ValueError,'roster'):validate_prepared_selection(dict(panorama_ids=['a','c']),prepared)
        with self.assertRaisesRegex(ValueError,'roster'):validate_prepared_selection(dict(panorama_ids=['a','b']),dict(prepared,frames=[dict(pano_id='a')]))
        with self.assertRaisesRegex(ValueError,'unique'):validate_prepared_selection(dict(panorama_ids=['a','a']),prepared)

    def test_rig_transform_and_shared_center_validation(self):
        pose=np.eye(4);pose[:3,:3]=Rotation.from_euler('x',90,degrees=True).as_matrix()
        camera_to_station(pose);np.testing.assert_allclose(pose[:3,:3]@[0,0,1],[0,-1,0],atol=1e-12)
        pose[0,3]=.01
        with self.assertRaisesRegex(ValueError,'center'):camera_to_station(pose)

    def test_visual_unit_and_coordinate_gauge_invariance(self):
        source=np.array([[-10,0,0],[0,8,1],[12,-2,2],[7,6,-1]],float)
        upright=Rotation.from_euler('zyx',[.3,-.4,.2]).as_matrix()
        target=3*(source@upright.T)+[11,-8,4]
        a,r,t,_=fit_horizontal_similarity(source,target,upright=upright)
        np.testing.assert_allclose(a*(source@r.T)+t,target,atol=1e-10)
        gauge=Rotation.from_euler('xyz',[.8,-.1,.2]).as_matrix();changed=(source@gauge.T)*1024
        b,r2,t2,_=fit_horizontal_similarity(changed,target,upright=upright@gauge.T)
        np.testing.assert_allclose(b*(changed@r2.T)+t2,target,atol=1e-10)
        self.assertAlmostEqual(a,b*1024)

    def test_gps_height_absence_does_not_create_height_observation(self):
        nodes=[dict(lat=0.,lng=x,station_id=str(i),pano_id=str(i)) for i,x in enumerate([0.,.0001,.0002])]
        gps,known,origin=gps_local_enu(nodes)
        self.assertFalse(known.any());self.assertTrue(origin['altitude_origin_is_arbitrary'])
        self.assertTrue(np.all(gps[:,2]==0));self.assertGreater(np.ptp(gps[:,0]),20)
        source=np.array([[0,0,0],[1,1,3],[2,0,-2]],float);target=source.copy();target[:,:2]*=8;target[:,2]=0
        scale,rotation,shift,_=fit_horizontal_similarity(source,target)
        self.assertGreater(np.ptp((scale*(source@rotation.T)+shift)[:,2]),30)

    def test_pairs_and_holdout_use_physical_groups_not_faces_or_dates(self):
        nodes=[dict(pano_id=p,station_id=g,lat=0,lng=l) for p,g,l in [('a1','a',0),('a2','a',0),('b','b',.0001),('c','c',.0002)]]
        pairs=candidate_capture_pairs(nodes,neighbors=2)
        self.assertNotIn((0,1),pairs)
        a=station_split([p['station_id'] for p in nodes],seed=3)
        self.assertEqual(a,station_split([p['station_id'] for p in nodes]*6,seed=3))
        self.assertFalse(set(a[0])&set(a[1]))

    def test_training_color_excludes_holdout_and_balances_groups(self):
        obs=[('a',[10,20,30])]*6+[('b',[30,40,50]),('held',[255,0,255])]
        color,count=training_observation_color(obs,{'a','b'})
        np.testing.assert_array_equal(color,[20,30,40]);self.assertEqual(count,2)
        self.assertIsNone(training_observation_color(obs,{'a'}))
        self.assertEqual(maximum_ray_angle_degrees([0,0,10],[[0,0,0],[0,0,0]]),0)

    def test_degenerate_visual_scale_and_bad_configuration_rejected(self):
        with self.assertRaisesRegex(ValueError,'baseline'):fit_horizontal_similarity(np.ones((3,3)),np.arange(9).reshape(3,3))
        with self.assertRaises(ValueError):SfMSettings(gps_horizontal_sigma_m=float('nan'))
        with self.assertRaises(ValueError):SfMSettings(minimum_seed_stations=1)


@unittest.skipUnless(importlib.util.find_spec('pycolmap'),'Optional pycolmap not installed')
class ActualPycolmapTests(unittest.TestCase):
    def test_duplicate_face_votes_do_not_satisfy_unique_point_threshold(self):
        import pycolmap as pc
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);rec,mapping,_,_=rig_recovery_fixture(pc,root,points_per_camera=20,disjoint=False)
            recovered,report=recover_capture_rigs(pc,root/'sfm',rec,mapping,SfMSettings(threads=1,recovery_seconds=10))
            self.assertEqual(recovered.num_reg_frames(),3);self.assertEqual(report['status'],'insufficient_verified_support')

    def test_budget_expiration_cannot_accept_pose_without_bundle_validation(self):
        import pycolmap as pc
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);rec,mapping,_,_=rig_recovery_fixture(pc,root)
            with patch('tools.streetview_engine.sfm.time.monotonic',side_effect=[0,0,20,20]):
                recovered,report=recover_capture_rigs(pc,root/'sfm',rec,mapping,SfMSettings(threads=1,recovery_seconds=10))
            self.assertTrue(report['attempts'][0]['registered'])
            self.assertEqual(recovered.num_reg_frames(),3);self.assertEqual(report['status'],'insufficient_verified_support')

    def test_verified_cache_copies_mutable_database_and_recovers_without_rematching(self):
        import pycolmap as pc
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'source';mapping,settings,signature=cached_sfm_fixture(pc,source)
            root=Path(temp)/'new';output=root/'sfm';output.mkdir(parents=True)
            before=sha(source/'sfm/database.db')
            rec,_,receipt=reuse_visual_cache(pc,source,root,output,mapping,settings,signature)
            self.assertEqual(receipt['status'],'verified_copied');self.assertEqual(receipt['database_sha256_at_copy'],before)
            self.assertEqual(receipt['database_sha256'],sha(output/'database.db'))
            recovered,report=recover_capture_rigs(pc,output,rec,mapping,settings)
            self.assertEqual(recovered.num_reg_frames(),4);self.assertEqual(report['minimum_pose_inliers'],30)
            # Writing only the new database must never alter its source.
            db=pc.Database.open(output/'database.db')
            try:db.clear_matches()
            finally:db.close()
            self.assertEqual(sha(source/'sfm/database.db'),before)

    def test_cache_rejects_changed_config_settings_source_and_database_calibration(self):
        import pycolmap as pc
        for mutation in ['config','settings','photo','calibration']:
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as temp:
                source=Path(temp)/'source';mapping,settings,signature=cached_sfm_fixture(pc,source)
                root=Path(temp)/'new';output=root/'sfm';output.mkdir(parents=True)
                if mutation=='config':signature=dict(signature,job_config_sha256='different')
                elif mutation=='settings':settings=SfMSettings(device='cpu',threads=1,max_features=256)
                elif mutation=='photo':(source/next(iter(mapping.values()))['file_path']).write_bytes(b'changed input')
                else:
                    db=pc.Database.open(source/'sfm/database.db')
                    try:
                        camera=db.read_all_cameras()[0];camera.params[0]+=1;db.update_camera(camera)
                    finally:db.close()
                with self.assertRaises(ValueError):reuse_visual_cache(pc,source,root,output,mapping,settings,signature)

    def test_pooling_missing_rig_keeps_native_inlier_threshold_and_fixed_calibration(self):
        import pycolmap as pc
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);rec,mapping,frame,truth=rig_recovery_fixture(pc,root,points_per_camera=20)
            self.assertEqual(rec.num_reg_frames(),3)
            recovered,report=recover_capture_rigs(pc,root/'sfm',rec,mapping,SfMSettings(threads=1,recovery_seconds=10))
            self.assertEqual(report['status'],'recovered');self.assertEqual(recovered.num_reg_frames(),4)
            self.assertEqual(report['minimum_pose_inliers'],30)
            self.assertTrue(all(v<30 for v in report['attempts'][0]['per_face_visible_points']))
            self.assertGreaterEqual(report['recovered_support'][0]['unique_observed_points'],30)
            np.testing.assert_allclose(recovered.frame(frame).rig_from_world.matrix(),truth,atol=1e-5)
            self.assertEqual(rec.num_reg_frames(),3)  # accepted proposal never mutates its input model
            for rig_id,rig in rec.rigs.items():
                for sensor in rig.sensor_ids():
                    if sensor==rig.ref_sensor_id:continue
                    np.testing.assert_allclose(rig.sensor_from_rig(sensor).matrix(),recovered.rigs[rig_id].sensor_from_rig(sensor).matrix(),atol=1e-12)

    def test_missing_rig_with_too_little_evidence_remains_unregistered(self):
        import pycolmap as pc
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);rec,mapping,_,_=rig_recovery_fixture(pc,root,points_per_camera=7)
            recovered,report=recover_capture_rigs(pc,root/'sfm',rec,mapping,SfMSettings(threads=1,recovery_seconds=10))
            self.assertEqual(recovered.num_reg_frames(),3);self.assertEqual(report['status'],'insufficient_verified_support')
            self.assertFalse(any(row['registered'] for row in report['attempts']))

    def test_actual_mapper_options_accept_default_float_seconds(self):
        import pycolmap as pc
        from tools.streetview_engine.sfm import incremental_options
        self.assertEqual(incremental_options(pc,SfMSettings()).max_runtime_seconds,1800)
        self.assertEqual(incremental_options(pc,SfMSettings(mapper_seconds=.2)).max_runtime_seconds,1)
        self.assertEqual(incremental_options(pc,SfMSettings(mapper_seconds=5.2)).max_runtime_seconds,6)

    def test_real_sift_and_fixed_two_face_rig_binding(self):
        import pycolmap as pc
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'sfm';out.mkdir();frames=[];stations=[]
            random=np.random.default_rng(19);texture=random.integers(0,256,(96,96,3),dtype=np.uint8)
            for capture in range(3):
                stations.append(dict(pano_id=str(capture),station_id=str(capture),lat=0.,lng=capture*.0001))
                for face,angle in [('F',0),('R',90)]:
                    path=f'prepared/{capture}_{face}.png';(root/'prepared').mkdir(exist_ok=True);Image.fromarray(texture).save(root/path)
                    mask=f'prepared/{capture}_{face}_mask.png';Image.new('L',(96,96),255).save(root/mask)
                    pose=np.eye(4);pose[:3,:3]=Rotation.from_euler('y',angle,degrees=True).as_matrix()
                    frames.append(dict(file_path=path,source_sha256=sha(root/path),mask_path=mask,mask_sha256=sha(root/mask),sfm_mask_path=mask,sfm_mask_sha256=sha(root/mask),sky_mask_path=mask,sky_mask_sha256=sha(root/mask),ground_mask_path=mask,ground_mask_sha256=sha(root/mask),pano_id=str(capture),station_id=str(capture),face=face,w=96,h=96,fl_x=48,fl_y=48,cx=48,cy=48,camera_to_station_cv=pose.tolist()))
            mapping,captures,faces,_=prepare_inputs(root,out,dict(frames=frames,stations=stations))
            report=extract_and_configure_rig(pc,out,mapping,faces,SfMSettings(device='cpu',threads=1,max_features=256))
            self.assertEqual(report['capture_rigs'],3);self.assertEqual(report['sensors'],2)
            self.assertTrue(all(row['keypoints']>0 for row in report['images']))
            database=pc.Database.open(out/'database.db')
            try:
                rigs=database.read_all_rigs();self.assertEqual(len(rigs),1)
                rig=rigs[0];sensors=[sensor for sensor in rig.sensor_ids() if sensor!=rig.ref_sensor_id]
                expected=np.linalg.inv(camera_to_station(frames[1]['camera_to_station_cv']))[:3,:3]
                np.testing.assert_allclose(rig.sensor_from_rig(sensors[0]).rotation.matrix(),expected,atol=1e-12)
            finally:database.close()
            # Identical photographs have no measured parallax. GPS coordinates
            # must not be accepted as an alternative reconstructed camera set.
            with self.assertRaisesRegex(RuntimeError,'Visual SfM found no|Insufficient visual'):
                visual_reconstruction(pc,out,mapping,captures,SfMSettings(device='cpu',threads=1,max_features=256,mapper_seconds=5),'cpu')
            Image.new('RGB',(96,96),(0,0,0)).save(root/frames[0]['file_path'])
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                prepare_inputs(root,root/'tampered',dict(frames=frames,stations=stations))

    def test_actual_pose_prior_ba_and_training_only_dataset(self):
        import pycolmap as pc
        rec=pc.Reconstruction();rec.add_camera_with_trivial_rig(pc.Camera(camera_id=1,model='PINHOLE',width=96,height=96,params=[70,70,48,48]))
        centers=np.array([[-9,0,-3],[0,.1,0],[10,.2,-2],[5,.05,5]],float)
        rng=np.random.default_rng(2);points=rng.uniform([-4,-2,23],[4,2,33],(35,3))
        metadata=[];mapping={};train,heldout=station_split([str(i) for i in range(4)],seed=0)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'prepared').mkdir()
            for i,center in enumerate(centers):
                local=points-center;xy=local[:,:2]/local[:,2:]*70+48;name=f'{i}.png'
                image=pc.Image(name=name,keypoints=xy,camera_id=1,image_id=i+1)
                rec.add_image_with_trivial_frame(image,pc.Rigid3d(pc.Rotation3d(np.eye(3)),-center))
                metadata.append(dict(pano_id=str(i),station_id=str(i),lat=center[2]/111319.49,lng=center[0]/111319.49,heading=0.))
                file=f'prepared/{i}.png';mask=f'prepared/{i}_mask.png';sky=f'prepared/{i}_sky.png'
                Image.new('RGB',(96,96),(220,0,250) if str(i) in heldout else (20,40,60)).save(root/file)
                Image.new('L',(96,96),255).save(root/mask);Image.new('L',(96,96),0).save(root/sky)
                mapping[name]=dict(file_path=file,mask_path=mask,sfm_mask_path=mask,sky_mask_path=sky,ground_mask_path=sky,pano_id=str(i),station_id=str(i),face='F',w=96,h=96,fl_x=70,fl_y=70,cx=48,cy=48,camera_to_station_cv=np.eye(4).tolist())
            for j,point in enumerate(points):
                rec.add_point3D(point+rng.normal(0,.015,3),pc.Track([pc.TrackElement(i+1,j) for i in range(4)]),np.array([255,0,255],np.uint8))
            rec.update_point_3d_errors();before=rec.compute_mean_reprojection_error()
            settings=SfMSettings(threads=1,gps_ba_iterations=30,gps_ba_seconds=10,minimum_gps_extent_sigma_ratio=1.)
            report=gps_adjust(pc,rec,mapping,metadata,settings)
            self.assertEqual(report['physical_prior_count'],4);self.assertEqual(report['height_prior_count'],0)
            self.assertLess(rec.compute_mean_reprojection_error(),before)
            rec.transform(pc.Sim3d(1.,pc.Rotation3d(EDN_FROM_ENU),np.zeros(3)))
            result=package_dataset(rec,mapping,metadata,root,root/'dataset',settings,report)
            manifest=json.loads((root/'dataset/dataset_manifest.json').read_text())
            self.assertTrue(manifest['seed_colors_exclude_heldout']);self.assertEqual(set(manifest['seed_color_station_ids']),set(train))
            self.assertFalse(set(result['training_station_ids'])&set(result['heldout_station_ids']))
            with np.load(root/'dataset/init_points.npz') as archive:
                self.assertGreaterEqual(len(archive['xyz']),4);np.testing.assert_array_equal(archive['rgb'],np.tile([20,40,60],(len(archive['rgb']),1)))
                self.assertTrue(np.all(archive['support_station_count']>=2))
                seed_arrays={key:archive[key] for key in archive.files}
            depth_manifest=json.loads((root/'dataset/sparse_depth_manifest.json').read_text())
            self.assertEqual(depth_manifest['observation_kind'],'actual_sfm_tracks')
            self.assertEqual(depth_manifest['geometry_scope'],'transductive_shared_sfm')
            self.assertEqual(depth_manifest['transforms_heldout_sha256'],sha(root/'dataset/transforms_heldout.json'))
            with np.load(root/'dataset/sparse_depth_observations.npz',allow_pickle=False) as depths:
                self.assertEqual(len(depths['point_id']),35*4)
                self.assertEqual(set(depths['station_id'][depths['split']=='heldout']),set(heldout))
                self.assertTrue(np.all(depths['depth_z']>0))
                self.assertTrue(np.all(depths['reprojection_error_px']<settings.maximum_reprojection_error_px))
                frames=json.loads((root/'dataset/transforms_train.json').read_text())['frames']+json.loads((root/'dataset/transforms_heldout.json').read_text())['frames']
                for frame in frames:
                    selected=depths['frame_name']==frame['file_path'];c2w=np.asarray(frame['transform_matrix'])@np.diag([1,-1,-1,1]);w2c=np.linalg.inv(c2w)
                    for pid,xy,z in zip(depths['point_id'][selected],depths['xy'][selected],depths['depth_z'][selected]):
                        expected=w2c[:3,:3]@rec.point3D(int(pid)).xyz+w2c[:3,3]
                        self.assertAlmostEqual(z,expected[2]);original=rec.image(int(frame['pano_id'])+1)
                        track=next(t for t in rec.point3D(int(pid)).track.elements if t.image_id==original.image_id)
                        np.testing.assert_array_equal(xy,original.points2D[track.point2D_idx].xy)
            # A matching projection cannot replace an absent actual track, and
            # a foreground-mask rejection removes only that observation.
            from tools.streetview_engine.sfm_depth import observation_arrays
            frame_names={name:f'images/{i:06d}.png' for i,name in enumerate(sorted(mapping))}
            held_image=rec.image(int(heldout[0])+1)
            xy=held_image.points2D[0].xy;mask_file=root/mapping[held_image.name]['sfm_mask_path']
            with Image.open(mask_file) as photo:mask=np.array(photo)
            mask[int(np.floor(xy[1])),int(np.floor(xy[0]))]=0;Image.fromarray(mask).save(mask_file)
            table,rejections=observation_arrays(rec,mapping,frame_names,root,seed_arrays,train,heldout,settings.maximum_reprojection_error_px)
            self.assertGreaterEqual(rejections['foreground_mask'],1)
            first_id=int(held_image.points2D[0].point3D_id)
            self.assertFalse(np.any((table['point_id']==first_id)&(table['station_id']==heldout[0])))
            self.assertTrue(np.any((table['point_id']==first_id)&(table['split']=='train')))
            absent_id=int(held_image.points2D[1].point3D_id)
            rec.delete_observation(held_image.image_id,1)
            table,_=observation_arrays(rec,mapping,frame_names,root,seed_arrays,train,heldout,settings.maximum_reprojection_error_px)
            self.assertFalse(np.any((table['point_id']==absent_id)&(table['station_id']==heldout[0])))
            self.assertTrue(np.any((table['point_id']==absent_id)&(table['split']=='train')))


if __name__=='__main__':unittest.main()
