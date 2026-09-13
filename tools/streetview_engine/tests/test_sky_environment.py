"""Synthetic photos only: no network, model downloads, GPU or scene fixture."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from tools.streetview_engine.imaging import sha256
from tools.streetview_engine import sky_environment
from tools.streetview_engine.sky_environment import SkyEnvironmentConfig, build_sky_environment


OPTIONS = SkyEnvironmentConfig(grid_size=8, tangent_sigma_cells=.35, boundary_margin_pixels=1.1)


def save_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding='utf8')


def fixture(root, *, groups=2, colors=None, sky=None, static=None, valid=None, world_rotation=None, scale=1., shift=None, duplicates=0):
    root.mkdir(parents=True, exist_ok=True)
    frame_rows = []
    rotation = np.eye(3) if world_rotation is None else world_rotation
    shift = np.zeros(3) if shift is None else np.asarray(shift)
    for group_index in range(groups):
        group = 'physical_' + str(group_index)
        center = np.array([group_index - (groups-1)/2, 0., 0.])
        for capture in range(1 + (duplicates if group_index == 0 else 0)):
            prefix = f'g{group_index}_c{capture}'
            color = [96, 144, 220] if colors is None else colors[group_index]
            rgb = np.broadcast_to(np.asarray(color, np.uint8), (128, 128, 3)).copy()
            sky_mask = np.full((128, 128), 255, np.uint8) if sky is None else sky[group_index]
            foreground = np.zeros((128, 128), np.uint8) if static is None else static[group_index]
            photometric = np.full((128, 128), 255, np.uint8) if valid is None else valid[group_index]
            paths = {}
            for key, value in [('file_path', rgb), ('mask_path', photometric), ('sky_mask_path', sky_mask), ('sfm_mask_path', foreground)]:
                name = prefix + '_' + key + '.png'
                Image.fromarray(value).save(root/name)
                paths[key] = name
            pose = np.eye(4)
            pose[:3, :3] = rotation @ np.diag([1., -1., -1.])
            pose[:3, 3] = scale*(rotation@center) + shift
            row = dict(**paths, image_sha256=sha256(root/paths['file_path']),
                       mask_sha256=sha256(root/paths['mask_path']), sky_mask_sha256=sha256(root/paths['sky_mask_path']),
                       sfm_mask_sha256=sha256(root/paths['sfm_mask_path']), station_id=group,
                       pano_id=prefix, face='F', w=128, h=128, fl_x=64., fl_y=64., cx=64., cy=64.,
                       transform_matrix=pose.tolist(), camera_to_station_cv=np.eye(4).tolist())
            frame_rows.append(row)
    transforms = dict(camera_convention='OpenGL_c2w', frames=frame_rows)
    save_json(root/'transforms_train.json', transforms)
    save_json(root/'dataset_manifest.json', dict(training_station_ids=[f'physical_{i}' for i in range(groups)],
        heldout_station_ids=['heldout_only'], files={'transforms_train.json': sha256(root/'transforms_train.json')}))
    return frame_rows


def covariance(result):
    rotations = Rotation.from_quat(result.quats[:, [1, 2, 3, 0]]).as_matrix()
    return (rotations*np.exp(result.log_scales[:, None, :])**2) @ np.swapaxes(rotations, 1, 2)


class SkyEnvironmentTests(unittest.TestCase):
    def test_supported_photo_colors_covariance_and_exact_ancestry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); frames = fixture(root); untouched = copy.deepcopy(frames)
            before = {p.name: sha256(p) for p in root.iterdir()}
            result = build_sky_environment(root, frames, config=OPTIONS)
            self.assertGreater(len(result.means), 0)
            self.assertEqual(result.shN.shape, (len(result.means), 8, 3))
            np.testing.assert_array_equal(result.support_counts, 2)
            np.testing.assert_allclose(result.sh0[:, 0]*.28209479177387814+.5, np.tile(np.array([96,144,220])/255, (len(result.means),1)), atol=1e-7)
            np.testing.assert_allclose(np.linalg.norm(result.quats, axis=1), 1, atol=1e-10)
            self.assertTrue((np.linalg.eigvalsh(covariance(result)) > 0).all())
            self.assertEqual(result.provenance['status'], 'candidate')
            self.assertFalse(result.provenance['geometry_is_measured'])
            self.assertFalse(result.provenance['quality_accepted'])
            for row in range(len(result.means)):
                source_frames = result.observation_frame_indices[result.observation_row_indices == row]
                self.assertEqual({frames[i]['station_id'] for i in source_frames}, {'physical_0', 'physical_1'})
            self.assertTrue(((result.observation_pixels_xy >= 0) & (result.observation_pixels_xy < 128)).all())
            self.assertEqual(frames, untouched)
            self.assertEqual(before, {p.name: sha256(p) for p in root.iterdir()})
            with self.assertRaises(ValueError): result.means[0, 0] = 0
            report = result.provenance; report['status'] = 'changed'
            self.assertEqual(result.provenance['status'], 'candidate')

    def test_rotation_translation_and_scale_covariance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a = root/'a'; b = root/'b'
            fixture(a)
            rotation = Rotation.from_rotvec([.4, -.2, .7]).as_matrix()
            scale = 7.5; shift = np.array([15., -7., 23.])
            fixture(b, world_rotation=rotation, scale=scale, shift=shift)
            first = build_sky_environment(a, config=OPTIONS)
            second = build_sky_environment(b, config=OPTIONS)
            np.testing.assert_array_equal(first.grid_indices, second.grid_indices)
            np.testing.assert_array_equal(first.support_counts, second.support_counts)
            np.testing.assert_allclose(second.means, scale*(first.means@rotation.T)+shift, rtol=1e-11, atol=1e-9)
            expected_covariance = scale**2*(rotation@covariance(first)@rotation.T)
            np.testing.assert_allclose(covariance(second), expected_covariance, rtol=1e-7, atol=1e-7)
            self.assertAlmostEqual(second.provenance['shell_radius_m']/first.provenance['shell_radius_m'], scale)

    def test_order_and_duplicate_capture_do_not_increase_group_support(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); a=root/'a';b=root/'b'
            frames=fixture(a);fixture(b, duplicates=3)
            first=build_sky_environment(a, config=OPTIONS)
            reordered=build_sky_environment(a, list(reversed(frames)), config=OPTIONS)
            duplicated=build_sky_environment(b, config=OPTIONS)
            np.testing.assert_array_equal(first.grid_indices, reordered.grid_indices)
            np.testing.assert_array_equal(first.grid_indices, duplicated.grid_indices)
            np.testing.assert_array_equal(first.support_counts, duplicated.support_counts)
            np.testing.assert_allclose(first.sh0, duplicated.sh0)

    def test_single_station_sky_stays_unsupported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            fixture(root, sky=[np.full((128,128),255,np.uint8),np.zeros((128,128),np.uint8)])
            result=build_sky_environment(root, config=OPTIONS)
            self.assertEqual(len(result.means),0)
            self.assertEqual(result.provenance['status'],'insufficient_evidence')
            self.assertEqual(result.observation_pixels_xy.shape,(0,2))

    def test_static_negative_veto_reduces_supported_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);full=np.full((128,128),255,np.uint8);zero=np.zeros((128,128),np.uint8)
            fixture(root, groups=3, sky=[full,full,zero], static=[zero,zero,full])
            result=build_sky_environment(root, config=OPTIONS)
            self.assertEqual(len(result.means),0)
            self.assertGreater(result.provenance['static_conflict_cells'],0)
            relaxed=build_sky_environment(root, config=replace(OPTIONS,maximum_static_negative_fraction=.34))
            self.assertGreater(len(relaxed.means),0)
            np.testing.assert_array_equal(relaxed.static_negative_counts,1)

    def test_disagreeing_colors_and_invalid_photometric_pixels_are_not_filled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);a=root/'a';b=root/'b'
            fixture(a, colors=[[255,0,0],[0,0,255]])
            result=build_sky_environment(a,config=OPTIONS)
            self.assertEqual(len(result.means),0)
            self.assertGreater(result.provenance['color_disagreement_cells'],0)
            fixture(b,valid=[np.zeros((128,128),np.uint8)]*2)
            self.assertEqual(len(build_sky_environment(b,config=OPTIONS).means),0)

    def test_projected_footprint_stays_away_from_mixed_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);sky=np.zeros((128,128),np.uint8);sky[:,:64]=255
            fixture(root,sky=[sky,sky],static=[255-sky,255-sky])
            result=build_sky_environment(root,config=OPTIONS)
            self.assertGreater(len(result.means),0)
            self.assertTrue((result.observation_pixels_xy[:,0]<60).all())
            self.assertTrue((result.static_negative_counts==0).all())

    def test_holdout_and_subset_or_hash_changes_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);frames=fixture(root)
            with self.assertRaisesRegex(ValueError,'complete saved training roster'):
                build_sky_environment(root,frames[:1],config=OPTIONS)
            changed=copy.deepcopy(frames);changed[0]['station_id']='heldout_only'
            with self.assertRaises(ValueError): build_sky_environment(root,changed,config=OPTIONS)
            with self.assertRaisesRegex(ValueError,'heldout roster'):
                build_sky_environment(root,config=OPTIONS,heldout_station_ids=['invented'])
            (root/frames[0]['sky_mask_path']).write_bytes(b'changed source')
            with self.assertRaisesRegex(ValueError,'hash mismatch'): build_sky_environment(root,config=OPTIONS)

    def test_shared_center_aliases_are_not_independent_stations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);frames=fixture(root,groups=3)
            frames[1]['transform_matrix']=copy.deepcopy(frames[0]['transform_matrix'])
            save_json(root/'transforms_train.json',dict(camera_convention='OpenGL_c2w',frames=frames))
            manifest=json.loads((root/'dataset_manifest.json').read_text())
            manifest['files']['transforms_train.json']=sha256(root/'transforms_train.json')
            save_json(root/'dataset_manifest.json',manifest)
            with self.assertRaisesRegex(ValueError,'share a reconstructed camera center'):
                build_sky_environment(root,config=OPTIONS)

    def test_midbuild_roster_mutation_invalidates_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);fixture(root)
            original=sky_environment._grid
            def change_source(*args):
                path=root/'transforms_train.json'
                path.write_text(path.read_text()+' ')
                return original(*args)
            with patch.object(sky_environment,'_grid',side_effect=change_source):
                with self.assertRaisesRegex(ValueError,'roster changed'):
                    build_sky_environment(root,config=OPTIONS)

    def test_radius_parallax_foreground_extent_and_far_clip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);frames=fixture(root)
            result=build_sky_environment(root,config=OPTIONS,foreground_extent_m=1000.)
            report=result.provenance
            self.assertGreater(report['shell_radius_m'],1000.)
            self.assertEqual(report['foreground_enclosure'],'caller_validated_extent')
            origin=np.array(report['shell_center'])
            reference=(result.means-origin)/report['shell_radius_m']
            for frame in frames:
                rays=result.means-np.asarray(frame['transform_matrix'])[:3,3]
                rays/=np.linalg.norm(rays,axis=1)[:,None]
                angles=np.rad2deg(np.arccos(np.clip(np.sum(reference*rays,axis=1),-1,1)))
                self.assertLessEqual(angles.max(),OPTIONS.maximum_parallax_degrees+1e-8)
            with self.assertRaisesRegex(ValueError,'far plane'):
                build_sky_environment(root,config=replace(OPTIONS,far_plane_m=10.))
            with self.assertRaisesRegex(ValueError,'CPU memory'):
                build_sky_environment(root,config=replace(OPTIONS,maximum_color_values=1))

    def test_config_and_nonbinary_masks_are_rejected(self):
        for args in [dict(grid_size=True),dict(minimum_physical_stations=1),dict(maximum_rgb_disagreement=float('nan')),dict(initial_opacity=1.),dict(far_plane_m=-1.)]:
            with self.subTest(args=args), self.assertRaises(ValueError):SkyEnvironmentConfig(**args)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            fixture(root,sky=[np.full((128,128),128,np.uint8)]*2)
            with self.assertRaisesRegex(ValueError,'binary'):build_sky_environment(root,config=OPTIONS)


if __name__=='__main__':unittest.main()
