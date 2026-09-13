"""Independent pixel convention, semantic uncertainty and station invariants."""
import unittest

import torch

from tools.streetview_geometry.semantic_center import (
    SemanticCenterConfig, prepare_semantic_center, observe_semantic_center,
    merge_semantic_center_station_views, count_semantic_center_stations,
)


def t(value):
    return torch.tensor(value, dtype=torch.float64)


class SemanticCenterTests(unittest.TestCase):
    def sample(self, means, mask, *, valid=None, radius=0, view=None, K=None, chunk=65536):
        p = prepare_semantic_center(mask, valid, config=SemanticCenterConfig(radius))
        return observe_semantic_center(t(means), torch.eye(4, dtype=torch.float64) if view is None else view,
                                       torch.eye(3, dtype=torch.float64) if K is None else K, p, chunk_rows=chunk)

    def test_pixel_centres_cells_and_half_open_image_border(self):
        mask = torch.zeros((3, 4), dtype=torch.bool)
        mask[1, 1] = True
        # K is identity, z=1: projected u,v are direct native edge coordinates.
        xy = [(1.5, 1.5), (1.01, 1.99), (1.99, 1.01), (2., 1.5),
              (.99, 1.5), (0., .5), (3.999, 2.999), (4., 1.), (-.0001, 1.), (1., 3.)]
        result = self.sample([(x, y, 1.) for x, y in xy], mask)
        self.assertEqual(result['sky_view'].tolist(), [True, True, True]+[False]*7)
        self.assertEqual(result['in_image'].tolist(), [True]*7+[False]*3)
        self.assertTrue(result['non_sky_view'][3:7].all())
        self.assertFalse((result['sky_view'] & result['non_sky_view']).any())

    def test_intrinsics_focal_and_principal_offset_no_extra_half_pixel(self):
        mask = torch.zeros((4, 5), dtype=torch.bool)
        mask[1, 2] = True
        K = t([[4, 0, 2.5], [0, 2, 1.5], [0, 0, 1]])
        result = self.sample([(0, 0, 10), (1.25, 0, 10), (-1.26, 0, 10)], mask, K=K)
        self.assertEqual(result['sky_view'].tolist(), [True, False, False])

    def test_independent_erosion_keeps_boundaries_unknown(self):
        mask = torch.zeros((11, 16), dtype=torch.bool)
        mask[:, :8] = True
        # Radius2: sky centres x=2..5; non-sky x=10..13. Both outer borders unknown.
        points = [(x+.5, 5.5, 1) for x in range(16)]
        result = self.sample(points, mask, radius=2)
        self.assertEqual(result['sky_view'].nonzero().flatten().tolist(), [2, 3, 4, 5])
        self.assertEqual(result['non_sky_view'].nonzero().flatten().tolist(), [10, 11, 12, 13])
        self.assertTrue(result['unknown_view'][6:10].all())

    def test_edit_or_generic_invalid_is_neither_label_and_erodes_neighbours(self):
        mask = torch.ones((11, 11), dtype=torch.bool)
        valid = torch.ones_like(mask)
        valid[5, 5] = False
        points = [(5.5, 5.5, 1), (7.5, 5.5, 1), (8.5, 5.5, 1)]
        result = self.sample(points, mask, valid=valid, radius=2)
        self.assertEqual(result['sky_view'].tolist(), [False, False, True])
        self.assertFalse(result['non_sky_view'].any())
        result = self.sample(points, ~mask, valid=valid, radius=2)
        self.assertEqual(result['non_sky_view'].tolist(), [False, False, True])
        self.assertFalse(result['sky_view'].any())

    def test_native_single_pixel_hole_is_not_resampled_away(self):
        mask = torch.ones((1536, 1536), dtype=torch.bool)
        mask[767, 901] = False
        r = self.sample([(901.9, 767.1, 1), (900.9, 767.1, 1)], mask)
        self.assertEqual(r['non_sky_view'].tolist(), [True, False])
        self.assertEqual(r['sky_view'].tolist(), [False, True])

    def test_behind_camera_zero_nonfinite_and_overflow_are_unknown(self):
        mask = torch.ones((3, 3), dtype=torch.bool)
        points = [(1, 1, -1), (1, 1, 0), (float('nan'), 1, 1),
                  (float('inf'), 1, 1), (1e308, 1, 1e-308), (1, 1, 1)]
        result = self.sample(points, mask)
        self.assertEqual(result['sky_view'].tolist(), [False]*5+[True])
        self.assertTrue(result['unknown_view'][:5].all())
        self.assertTrue(result['front_view'][4])
        self.assertFalse(result['in_image'][4])

    def test_nadir_has_no_implicit_exception_or_height_rule(self):
        view = torch.diag(t([1, -1, -1, 1]))
        K = t([[3, 0, 5.5], [0, 3, 5.5], [0, 0, 1]])
        mask = torch.zeros((11, 11), dtype=torch.bool)
        result = self.sample([(0, 0, -2)], mask, view=view, K=K, radius=2)
        self.assertTrue(result['non_sky_view'][0])
        self.assertFalse(result['sky_view'][0])
        result = self.sample([(0, 0, -2)], ~mask, view=view, K=K, radius=2)
        self.assertTrue(result['sky_view'][0]) # Uses input labels, not a face name.

    def test_rigid_coordinate_and_positive_scale_invariance(self):
        torch.manual_seed(41)
        means = torch.randn(701, 3, dtype=torch.float64)
        means[:, 2] += 3
        mask = torch.zeros((64, 64), dtype=torch.bool)
        mask[:, :30] = True
        valid = torch.ones_like(mask)
        valid[16:19, 8:20] = False
        K = t([[16, 0, 32.5], [0, 16, 32.5], [0, 0, 1]])
        p = prepare_semantic_center(mask, valid)
        original = observe_semantic_center(means, torch.eye(4, dtype=torch.float64), K, p, chunk_rows=97)
        Q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
        if torch.det(Q) < 0:
            Q[:, 0] *= -1
        shift = t([15.3, -30.7, 8.2])
        for scale in [1e-4, .01, 1., 10., 1e3]:
            transformed = scale*(means @ Q.T)+shift
            view = torch.eye(4, dtype=torch.float64)
            view[:3, :3] = Q.T
            view[:3, 3] = -Q.T @ shift
            result = observe_semantic_center(transformed, view, K, p, chunk_rows=13)
            for key in ['sky_view', 'non_sky_view', 'front_view', 'in_image', 'known_view']:
                self.assertTrue(torch.equal(original[key], result[key]), (scale, key))

    def test_chunk_size_invariance_and_bounded_diagnostics(self):
        mask = torch.rand((16, 16), generator=torch.Generator().manual_seed(7)) > .5
        means = [(x+.2, y+.7, 1) for y in range(16) for x in range(16)]
        a = self.sample(means, mask, chunk=1)
        b = self.sample(means, mask, chunk=17)
        c = self.sample(means, mask, chunk=10000)
        for key in ['sky_view', 'non_sky_view', 'known_view', 'unknown_view', 'in_image', 'front_view']:
            self.assertTrue(torch.equal(a[key], b[key]))
            self.assertTrue(torch.equal(a[key], c[key]))
        self.assertEqual(b['diagnostics']['max_projection_batch'], 17)

    def test_six_cube_faces_one_physical_station_and_500_stations_int32(self):
        directions = t([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
        means = directions*8
        K = t([[16,0,16],[0,16,16],[0,0,1]])
        p = prepare_semantic_center(torch.ones((32,32),dtype=torch.bool))
        observations = []
        for z in directions:
            arbitrary = t([0,1,0]) if abs(z[1]) < .5 else t([0,0,1])
            x = torch.linalg.cross(arbitrary, z)
            x /= torch.linalg.vector_norm(x)
            y = torch.linalg.cross(z, x)
            view = torch.eye(4, dtype=torch.float64)
            view[:3,:3] = torch.stack((x,y,z))
            observations.append(observe_semantic_center(means, view, K, p))
        station = merge_semantic_center_station_views(observations)
        duplicate = merge_semantic_center_station_views(observations*13)
        self.assertTrue(torch.equal(station['sky'], duplicate['sky']))
        self.assertTrue(station['sky'].all())
        single = count_semantic_center_stations({'station': station})
        self.assertEqual(single['sky_station_count'].tolist(), [1]*6)
        many = count_semantic_center_stations({str(i):station for i in range(500)})
        self.assertEqual(many['sky_station_count'].dtype, torch.int32)
        self.assertEqual(many['sky_station_count'].tolist(), [500]*6)
        self.assertEqual(many['non_sky_station_count'].tolist(), [0]*6)

    def test_face_conflict_is_explicit_and_not_counted_as_extra_station(self):
        a = {'sky_view':torch.tensor([True,False]), 'non_sky_view':torch.tensor([False,True])}
        b = {'sky_view':torch.tensor([False,False]), 'non_sky_view':torch.tensor([True,False])}
        merged = merge_semantic_center_station_views([a,b,a,b])
        result = count_semantic_center_stations({'s': merged})
        self.assertEqual(result['sky_station_count'].tolist(), [1,0])
        self.assertEqual(result['non_sky_station_count'].tolist(), [1,1])
        self.assertEqual(result['known_station_count'].tolist(), [1,1])
        self.assertEqual(result['conflict_station_count'].tolist(), [1,0])

    def test_snapshot_and_prepared_mutation_guard(self):
        mask = torch.ones((4,4), dtype=torch.bool)
        valid = torch.ones_like(mask)
        p = prepare_semantic_center(mask, valid, config=SemanticCenterConfig(0))
        mask[:] = False
        valid[:] = False
        r = observe_semantic_center(t([[1.5,1.5,1]]),torch.eye(4),torch.eye(3),p)
        self.assertTrue(r['sky_view'][0])
        p.sky[1,1] = False
        with self.assertRaisesRegex(ValueError,'changed in place'):
            observe_semantic_center(t([[1.5,1.5,1]]),torch.eye(4),torch.eye(3),p)

    def test_reject_bad_camera_mask_config_and_colliding_station_ids(self):
        for radius in [-1, True, .5]:
            with self.assertRaises(ValueError):
                SemanticCenterConfig(radius)
        with self.assertRaises(ValueError):
            prepare_semantic_center(torch.ones(4,4))
        mask = torch.ones((4,4), dtype=torch.bool)
        view = torch.eye(4, dtype=torch.float64)
        view[0,0] = 2
        with self.assertRaises(ValueError):
            self.sample([(1,1,1)],mask,view=view)
        with self.assertRaises(ValueError):
            self.sample([(1,1,1)],mask,chunk=0)
        station = {'sky':torch.tensor([True]), 'non_sky':torch.tensor([False])}
        with self.assertRaises(ValueError):
            count_semantic_center_stations({1:station,'1':station})

    def test_empty_means_and_all_unknown_masks(self):
        p = prepare_semantic_center(torch.ones((1,1),dtype=torch.bool))
        result = observe_semantic_center(torch.zeros((0,3)),torch.eye(4),torch.eye(3),p)
        self.assertEqual(result['sky_view'].shape, (0,))
        self.assertEqual(result['diagnostics']['max_projection_batch'], 0)
        result = observe_semantic_center(t([[.5,.5,1]]),torch.eye(4),torch.eye(3),p)
        self.assertTrue(result['unknown_view'][0]) # Radius2 consumes entire 1px image.


if __name__ == '__main__':
    unittest.main()
