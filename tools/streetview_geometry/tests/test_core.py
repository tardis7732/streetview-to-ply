import copy
import unittest
import numpy as np
from tools.streetview_geometry.contracts import CameraSet, characteristic_length, radial_to_camera_z, station_split
from tools.streetview_geometry.free_space import FreeSpaceConfig, aggregate_footprint_samples, classify_samples, select_free_space
from tools.streetview_geometry.evaluation import validate_render_comparison


def frames():
    result = []
    for group in range(5):
        for face in range(2):
            pose = np.eye(4); pose[:3, 3] = [group * 2., 0., 0.]
            result.append(dict(file_path=f'camera_{group}_{face}.jpg', station_index=str(group),
                transform_matrix=pose.tolist(), w=256, h=128, fl_x=128, fl_y=128, cx=128, cy=64))
    return result


class ContractsTests(unittest.TestCase):
    def test_group_split_and_units(self):
        camera = CameraSet.from_frames(frames(), convention='opengl', world_up=[0, -1, 0])
        self.assertEqual(camera.characteristic_length, 2.)
        discovery, evaluation = station_split(camera.station_ids)
        self.assertFalse(set(discovery) & set(evaluation))
        self.assertEqual(set(discovery) | set(evaluation), set(camera.station_ids))
        self.assertEqual(station_split(camera.station_ids * 2), (discovery, evaluation))
        centers = camera.camera_to_world_cv[:, :3, 3]
        rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        self.assertAlmostEqual(characteristic_length(centers @ rotation * 100 + 817), 200.)

    def test_invalid_station_rig_rejected(self):
        source = frames(); source[1]['transform_matrix'][0][3] += .2
        with self.assertRaises(ValueError):
            CameraSet.from_frames(source, convention='opengl', world_up=[0, 1, 0])

    def test_different_ids_at_same_center_are_not_independent(self):
        source = frames()
        source[1]['station_index'] = 'same_center_new_id'
        with self.assertRaisesRegex(ValueError, 'share a capture center'):
            CameraSet.from_frames(source, convention='opengl', world_up=[0, 1, 0])

    def test_radial_is_not_z_and_inputs_immutable(self):
        rays = np.array([[0., 0., 1.], [1., 0., 1.]])
        rays.setflags(write=False)
        np.testing.assert_allclose(radial_to_camera_z(np.array([2., 2.]), rays), [2., np.sqrt(2)])
        np.testing.assert_allclose(radial_to_camera_z(np.array([20., 20.]), rays * 7), [20., np.sqrt(200)])


class FreeSpaceTests(unittest.TestCase):
    def test_interval_support_occlusion_unknown_and_scale(self):
        values = dict(center_z=np.array([2., 10., 20., 2.]), sigma_z=np.array([.1, .1, .1, 0.]),
            surface_z=np.full(4, 10.), surface_sigma=np.full(4, .1), valid=[1, 1, 1, 0])
        result = classify_samples(**values)
        self.assertEqual(result['free'].tolist(), [True, False, False, False])
        self.assertEqual(result['support'].tolist(), [False, True, False, False])
        self.assertEqual(result['behind'].tolist(), [False, False, True, False])
        scaled = classify_samples(**{k: v if k == 'valid' else v * 1000 for k, v in values.items()})
        for key in result:
            np.testing.assert_array_equal(result[key], scaled[key])

    def test_cube_faces_are_not_independent_stations(self):
        free = np.ones((1, 9), bool); support = np.zeros_like(free)
        votes = aggregate_footprint_samples(free, support, free, free, ['one'] * 9)
        result = select_free_space([17], votes['station_ids'], votes['free'], votes['support'], votes['sparse_free'])
        self.assertEqual(result['indices'].size, 0)
        with self.assertRaises(ValueError):
            select_free_space([17], ['same'] * 3, free[:, :3], support[:, :3], free[:, :3])

    def test_three_stations_and_support_veto(self):
        free = np.array([[1, 1, 1, 0], [1, 1, 1, 0]], bool)
        support = np.array([[0, 0, 0, 0], [0, 0, 0, 1]], bool)
        result = select_free_space([20, 21], ['a', 'b', 'c', 'd'], free, support, free)
        np.testing.assert_array_equal(result['indices'], [20])
        with self.assertRaises(ValueError):
            FreeSpaceConfig(min_sparse_stations=4)
        with self.assertRaises(ValueError):
            FreeSpaceConfig(min_free_stations=3.5)
        with self.assertRaises(ValueError):
            select_free_space([20.5, 21.], ['a', 'b', 'c', 'd'], free, support, free)
        with self.assertRaises(ValueError):
            select_free_space([20, 21], ['a', 'b', 'c', 'd'], free.astype(float) * np.nan, support, free)


class EvaluationTests(unittest.TestCase):
    def good(self):
        return dict(source_sha256='source', candidate_sha256='candidate', discovery_station_ids=['train'],
            expected_evaluation_frames=['f1', 'f2'], views=[dict(frame=f'f{i}', station_id=f'eval{i}',
                reference_rgb_sha256='photo', reference_mask_sha256='mask',
                before=dict(static_sse=1., static_pixels=1000), after=dict(static_sse=1., static_pixels=1000),
                new_hole_pixels=0) for i in (1, 2)])

    def gate(self, report):
        trusted = self.good()
        manifest = dict(discovery_station_ids=trusted['discovery_station_ids'],
            evaluation_frames=[{key: row[key] for key in ('frame', 'station_id', 'reference_rgb_sha256', 'reference_mask_sha256')} for row in trusted['views']])
        return validate_render_comparison(report, source_sha256='source', candidate_sha256='candidate', expected_manifest=manifest)

    def test_complete_unchanged_pass(self):
        self.assertTrue(self.gate(self.good())['accepted'])

    def test_fail_closed_for_missing_or_leaked_views(self):
        report = self.good(); report['views'].pop()
        self.assertIn('incomplete_or_duplicate_evaluation', self.gate(report)['reasons'])
        report = self.good(); report['views'][0]['station_id'] = 'train'
        self.assertIn('physical_station_holdout_leakage', self.gate(report)['reasons'])
        report = self.good(); report['candidate_sha256'] = 'stale'
        self.assertIn('artifact_hash_mismatch', self.gate(report)['reasons'])

    def test_report_cannot_shorten_roster_or_relabel_reference(self):
        report = self.good(); report['views'].pop(); report['expected_evaluation_frames'].pop()
        self.assertIn('reported_roster_differs_from_trusted_manifest', self.gate(report)['reasons'])
        report = self.good(); report['views'][0]['reference_rgb_sha256'] = 'different_photo'
        self.assertIn('reference_binding_mismatch', self.gate(report)['reasons'])
        report = self.good(); report['views'][0]['station_id'] = 'different_station'
        self.assertIn('frame_station_binding_mismatch', self.gate(report)['reasons'])
        result = validate_render_comparison(self.good(), source_sha256='source', candidate_sha256='candidate')
        self.assertFalse(result['accepted'])

    def test_local_holes_and_rgb_regression(self):
        report = self.good(); report['views'][0]['after']['static_sse'] = 2.
        self.assertFalse(self.gate(report)['accepted'])
        report = self.good(); report['views'][0]['new_hole_pixels'] = 2
        self.assertIn('new_static_holes', self.gate(report)['reasons'])


if __name__ == '__main__':
    unittest.main()
