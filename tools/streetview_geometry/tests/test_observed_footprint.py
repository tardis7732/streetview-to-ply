"""Independent original-pixel evidence tests for complete observed footprints."""
import unittest

import numpy as np
import torch

from tools.streetview_geometry.free_space import FreeSpaceConfig, select_free_space
from tools.streetview_geometry.observed_footprint import observe
from tools.streetview_geometry.sparse_prune import reduce_station


def t(value):
    return torch.tensor(value, dtype=torch.float64)


class ObservedFootprintTests(unittest.TestCase):
    def fixture(self, entries, *, sx=1., sy=1., sz=.1):
        depth = torch.zeros((11, 11), dtype=torch.float64)
        valid = torch.zeros_like(depth, dtype=torch.bool)
        strict = valid.clone()
        for x, y, z, is_strict in entries:
            depth[y, x] = z; valid[y, x] = True; strict[y, x] = is_strict
        return dict(means=t([[0, 0, 10]]), covariance=torch.diag(t([sx*sx, sy*sy, sz*sz]))[None],
                    view=torch.eye(4, dtype=torch.float64)[:3], K=t([[10, 0, 5.5], [0, 10, 5.5], [0, 0, 1]]),
                    depth=depth, valid=valid, strict=strict, config=FreeSpaceConfig())

    def test_three_original_free_pixels_vote_once_each(self):
        f = self.fixture([(5, 5, 20, True), (6, 5, 20, True), (5, 6, 20, True)])
        result = observe(**f)
        self.assertTrue(result['free_view'][0])
        self.assertEqual(result['strict_free_pixels'].tolist(), [3])
        self.assertEqual(result['valid_pixels'].tolist(), [3])
        self.assertFalse(result['blocked_view'][0])

    def test_sparse_pixel_inside_bounding_box_but_outside_ellipse_does_not_count(self):
        # sx=2, sy=1: dx=3, dy=2 is within the 2-sigma rectangle
        # but q=(3/2)^2+2^2=6.25, outside the actual two-sigma ellipse.
        f = self.fixture([(5, 5, 20, True), (6, 5, 20, True), (8, 7, 20, True)], sx=2)
        result = observe(**f)
        self.assertEqual(result['strict_free_pixels'].tolist(), [2])
        self.assertEqual(result['valid_pixels'].tolist(), [3])
        self.assertFalse(result['free_view'][0])

    def test_three_sigma_surface_edge_vetoes_core_free_evidence(self):
        f = self.fixture([(5, 5, 20, True), (6, 5, 20, True), (5, 6, 20, True), (8, 5, 10, False)])
        result = observe(**f)
        self.assertEqual(result['strict_free_pixels'].tolist(), [3])
        self.assertTrue(result['support_view'][0])
        self.assertTrue(result['blocked_view'][0])
        self.assertFalse(result['free_view'][0])

    def test_occluded_edge_is_nonfree_veto_but_not_surface_support(self):
        f = self.fixture([(5, 5, 20, True), (6, 5, 20, True), (5, 6, 20, True), (8, 5, 5, False)])
        result = observe(**f)
        self.assertTrue(result['blocked_view'][0])
        self.assertFalse(result['support_view'][0])
        self.assertFalse(result['free_view'][0])

    def test_surface_outside_three_sigma_does_not_veto(self):
        f = self.fixture([(5, 5, 20, True), (6, 5, 20, True), (5, 6, 20, True), (9, 5, 10, True)])
        result = observe(**f)
        self.assertTrue(result['free_view'][0])
        self.assertEqual(result['valid_pixels'].tolist(), [3])

    def test_loose_free_unknown_and_invalid_depth_do_not_invent_strict_pixels(self):
        f = self.fixture([(5, 5, 20, True), (6, 5, 20, True), (5, 6, 20, False), (4, 5, float('nan'), True)])
        f['depth'][4, 5] = 20  # finite but invalid original cell
        result = observe(**f)
        self.assertEqual(result['strict_free_pixels'].tolist(), [2])
        self.assertEqual(result['valid_pixels'].tolist(), [3])
        self.assertFalse(result['free_view'][0])
        self.assertFalse(result['blocked_view'][0])

    def test_near_plane_crossing_behind_and_degenerate_geometry_abstain(self):
        f = self.fixture([(5, 5, 20, True), (6, 5, 20, True), (5, 6, 20, True)])
        f['means'] = t([[0, 0, .1], [0, 0, -10], [0, 0, 10], [float('nan'), 0, 10]])
        f['covariance'] = f['covariance'].repeat(4, 1, 1)
        f['covariance'][2] = 0
        result = observe(**f)
        torch.testing.assert_close(result['blocked_view'], torch.zeros(4, dtype=torch.bool))
        torch.testing.assert_close(result['free_view'], torch.zeros(4, dtype=torch.bool))
        self.assertEqual(result['strict_free_pixels'].tolist(), [0]*4)

    def test_global_rotation_translation_and_scale_preserve_original_pixel_votes(self):
        f = self.fixture([(5, 5, 20, True), (6, 5, 20, True), (5, 6, 20, True), (8, 5, 10, False)], sx=1.2, sy=.9)
        baseline = observe(**f)
        # Exact orthogonal rational rotation avoids any near-threshold fixture.
        Q = t([[.36, -.48, .8], [.8, .6, 0], [-.48, .64, .6]])
        for scale in (1e-4, 2.3, 1e3):
            with self.subTest(scale=scale):
                translation = t([17, -23, 41])*scale
                changed = dict(f)
                changed['means'] = scale*(f['means']@Q.T)+translation
                changed['covariance'] = scale**2*(Q@f['covariance']@Q.T)
                changed['view'] = torch.cat((Q.T, (-Q.T@translation)[:, None]), 1)
                changed['depth'] = scale*f['depth']
                result = observe(**changed)
                for key in baseline:
                    torch.testing.assert_close(result[key], baseline[key])

    def test_camera_axis_rotation_changes_correct_projected_ellipse_axis(self):
        f = self.fixture([(5, 5, 20, True), (7, 5, 20, True), (3, 5, 20, True)], sx=.1, sy=.1, sz=2)
        f['means'] = t([[-10, 0, 0]])
        f['view'] = t([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]])
        result = observe(**f)
        self.assertEqual(result['strict_free_pixels'].tolist(), [3])
        self.assertTrue(result['free_view'][0])

    def test_empty_original_depth_has_zero_evidence(self):
        result = observe(**self.fixture([]))
        self.assertFalse(result['free_view'][0])
        self.assertFalse(result['blocked_view'][0])
        self.assertEqual(result['valid_pixels'].tolist(), [0])

    def test_back_and_perpendicular_cube_faces_do_not_veto_three_front_stations(self):
        f = self.fixture([])
        f['depth'].fill_(20); f['valid'].fill_(True); f['strict'].fill_(True)
        rotations = [t([[1,0,0],[0,1,0],[0,0,1]]),
                     t([[-1,0,0],[0,1,0],[0,0,-1]]),
                     t([[0,0,-1],[0,1,0],[1,0,0]]),
                     t([[0,0,1],[0,1,0],[-1,0,0]]),
                     t([[1,0,0],[0,0,1],[0,-1,0]]),
                     t([[1,0,0],[0,0,-1],[0,1,0]])]
        free, blocked, support, groups = [], [], [], []
        for station, center in enumerate((t([0,0,0]), t([1,0,0]), t([-1,0,0]))):
            for face, R in enumerate(rotations):
                changed = dict(f, view=torch.cat((R, (-R@center)[:, None]), 1))
                result = observe(**changed)
                self.assertFalse(result['blocked_view'][0])
                self.assertEqual(bool(result['free_view'][0]), face == 0)
                free.append(bool(result['free_view'][0])); blocked.append(False)
                support.append(bool(result['support_view'][0])); groups.append(str(station))
        flags = reduce_station(np.array([free]), np.array([blocked]), np.array([support]), groups)
        proposed = select_free_space([0], flags['station_ids'], flags['free'], flags['support'], flags['free'])
        np.testing.assert_array_equal(proposed['indices'], [0])


if __name__ == '__main__':
    unittest.main()
