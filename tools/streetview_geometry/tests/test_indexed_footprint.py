"""Indexed evidence equals a full original-pixel oracle, including missed centres."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tools.streetview_geometry.free_space import FreeSpaceConfig, select_free_space
from tools.streetview_geometry.indexed_footprint import observe_indexed, prepare_geometry
from tools.streetview_geometry.observed_footprint import observe
from tools.streetview_geometry.sparse_prune import reduce_station, sample_uv, project


def t(value):
    return torch.tensor(value, dtype=torch.float64)


KEYS = ('free_view', 'blocked_view', 'support_view', 'strict_free_pixels', 'valid_pixels')


class IndexedFootprintTests(unittest.TestCase):
    def fixture(self, entries=(), *, sx=1., sy=1., sz=.1, size=11):
        depth = torch.zeros((size, size), dtype=torch.float64)
        valid = torch.zeros_like(depth, dtype=torch.bool)
        strict = valid.clone()
        for x, y, z, is_strict in entries:
            depth[y, x] = z; valid[y, x] = True; strict[y, x] = is_strict
        return dict(means=t([[0, 0, 10]]), covariance=torch.diag(t([sx*sx, sy*sy, sz*sz]))[None],
                    view=torch.eye(4, dtype=torch.float64)[:3], K=t([[10, 0, 5.5], [0, 10, 5.5], [0, 0, 1]]),
                    depth=depth, valid=valid, strict=strict, config=FreeSpaceConfig())

    def check(self, fixture, **options):
        expected = observe(**fixture)
        result = observe_indexed(**fixture, **options)
        for key in KEYS:
            torch.testing.assert_close(result[key], expected[key], rtol=0, atol=0, msg=key)
        d = result['diagnostics']
        self.assertEqual(d['gaussian_rows'], len(fixture['means']))
        self.assertEqual(d['rows_with_depth_observations'], int((result['valid_pixels'] > 0).sum()))
        self.assertEqual(d['inside_depth_pairs'], int(result['valid_pixels'].sum()))
        self.assertLessEqual(d['max_tile_batch'], d['pair_budget'])
        self.assertLessEqual(d['max_depth_batch'], d['pair_budget'])
        return result

    def test_center_missing_but_three_observed_footprint_pixels_are_free(self):
        f = self.fixture([(4, 5, 20, True), (6, 5, 20, True), (5, 6, 20, True)])
        _, uv, _ = project(f['means'], f['covariance'], f['view'], f['K'])
        self.assertFalse(sample_uv(uv, f['depth'], f['valid'])[1][0])
        result = self.check(f, tile_size=3, pair_budget=2)
        self.assertEqual(result['strict_free_pixels'].tolist(), [3])
        self.assertTrue(result['free_view'][0])

    def test_exact_two_three_sigma_boundaries_and_tile_boundaries(self):
        f = self.fixture([(3, 5, 20, True), (7, 5, 20, True), (5, 3, 20, True),
                          (5, 7, 20, True), (8, 5, 10, False), (9, 5, 10, True)])
        for tile in (1, 2, 4, 16):
            with self.subTest(tile=tile):
                r = self.check(f, tile_size=tile, pair_budget=5)
                self.assertEqual(r['strict_free_pixels'].tolist(), [4])
                self.assertEqual(r['valid_pixels'].tolist(), [5])
                self.assertTrue(r['support_view'][0])
                self.assertTrue(r['blocked_view'][0])

    def test_anisotropic_ellipse_not_bounding_rectangle(self):
        f = self.fixture([(5, 5, 20, True), (6, 5, 20, True), (8, 7, 20, True), (10, 8, 10, False)], sx=2)
        r = self.check(f, tile_size=4, pair_budget=3)
        self.assertEqual(r['strict_free_pixels'].tolist(), [2])
        self.assertEqual(r['valid_pixels'].tolist(), [3])
        self.assertFalse(r['blocked_view'][0])

    def test_ulp_perturbations_at_axis_aligned_pixel_bounds_match_oracle(self):
        f = self.fixture([(2, 5, 10, False), (3, 5, 20, True), (7, 5, 20, True),
                          (8, 5, 10, False), (5, 2, 10, False), (5, 8, 10, False)])
        # Vary covariance and principal point independently by adjacent
        # representable doubles on both sides of an exact pixel boundary.
        for direction in (-torch.inf, torch.inf):
            for field in ('covariance', 'principal_point'):
                with self.subTest(direction=direction, field=field):
                    changed = dict(f, covariance=f['covariance'].clone(), K=f['K'].clone())
                    target = changed['covariance'][:, 0, 0] if field == 'covariance' else changed['K'][0, 2:3]
                    target.copy_(torch.nextafter(target, torch.full_like(target, direction)))
                    self.check(changed, tile_size=1, pair_budget=3)

    def test_off_image_center_with_visible_ellipse_is_not_screened(self):
        f = self.fixture([(0, 5, 20, True), (1, 5, 20, True), (0, 6, 20, True)], sx=3)
        f['means'][0, 0] = -7  # projected centre -1.5, entire centre outside image
        r = self.check(f, tile_size=3, pair_budget=2)
        self.assertTrue(r['free_view'][0])
        self.assertEqual(r['strict_free_pixels'].tolist(), [3])

    def test_invalid_nonpositive_depth_and_loose_pixels_do_not_create_support(self):
        f = self.fixture([(5, 5, 20, True), (6, 5, 20, True), (5, 6, 20, False),
                          (4, 5, float('nan'), True), (5, 4, -1, True), (6, 6, float('inf'), True)])
        f['depth'][4, 4] = 10  # invalid cell despite finite depth
        r = self.check(f, tile_size=2, pair_budget=1)
        self.assertEqual(r['strict_free_pixels'].tolist(), [2])
        self.assertEqual(r['valid_pixels'].tolist(), [3])
        self.assertFalse(r['free_view'][0]); self.assertFalse(r['blocked_view'][0])

    def test_near_plane_behind_degenerate_nonfinite_and_outside_abstain(self):
        f = self.fixture([(5, 5, 20, True), (6, 5, 20, True), (5, 6, 20, True)])
        f['means'] = t([[0, 0, .1], [0, 0, -10], [0, 0, 10], [float('nan'), 0, 10],
                        [0, 0, 10], [0, 0, 10], [1e100, 0, 10], [100, 100, 10]])
        f['covariance'] = f['covariance'].repeat(8, 1, 1)
        f['covariance'][2] = 0
        f['covariance'][4, 0, 1] = .1  # non-symmetric covariance
        f['covariance'][5, 0, 0] = float('inf')
        r = self.check(f, tile_size=2, pair_budget=4)
        self.assertEqual(r['valid_pixels'].tolist(), [0]*8)
        self.assertFalse(r['blocked_view'].any())

    def test_giant_finite_ellipse_scans_whole_image_without_radius_cap(self):
        f = self.fixture(sx=100, sy=100, size=8)
        f['depth'].fill_(20); f['valid'].fill_(True); f['strict'].fill_(True)
        r = self.check(f, tile_size=2, pair_budget=3)
        self.assertEqual(r['strict_free_pixels'].tolist(), [64])
        self.assertEqual(r['diagnostics']['tile_pairs'], 16)
        self.assertEqual(r['diagnostics']['depth_pairs'], 64)

    def test_random_sparse_camera_anisotropy_matches_all_pairs_oracle(self):
        generator = torch.Generator().manual_seed(941)
        for seed in range(3):
            with self.subTest(seed=seed):
                n, h, w = 67, 31, 43
                means = torch.randn((n, 3), generator=generator, dtype=torch.float64)*4
                means[:, 2] += 9
                bases = torch.randn((n, 3, 3), generator=generator, dtype=torch.float64)*.3
                cov = bases @ bases.transpose(-1, -2)+torch.eye(3, dtype=torch.float64)*.005
                angle = .21*(seed+1)
                R = t([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
                view = torch.cat((R, t([.7, -.3, .1])[:, None]), 1)
                depth = 3+torch.rand((h, w), generator=generator, dtype=torch.float64)*25
                valid = torch.rand((h, w), generator=generator) < .13
                strict = valid & (torch.rand((h, w), generator=generator) < .65)
                f = dict(means=means, covariance=cov, view=view, K=t([[29, 0, 20.4], [0, 19, 14.2], [0, 0, 1]]),
                         depth=depth, valid=valid, strict=strict, config=FreeSpaceConfig())
                self.check(f, tile_size=7, pair_budget=197)

    def test_row_order_row_chunks_tile_size_pair_budget_invariant(self):
        f = self.fixture([(5, 5, 20, True), (6, 5, 20, True), (5, 6, 20, True), (8, 5, 10, False)])
        f['means'] = t([[0, 0, 10], [.2, -.3, 6], [2, 1, 12], [-3, 1, 8], [0, 0, -10]])
        f['covariance'] = f['covariance'].repeat(5, 1, 1)
        baseline = self.check(f)
        order = torch.tensor([4, 2, 0, 3, 1])
        for tile, budget in ((1, 1), (2, 7), (4, 17), (128, 10000)):
            changed = dict(f, means=f['means'][order], covariance=f['covariance'][order])
            r = self.check(changed, tile_size=tile, pair_budget=budget)
            for key in KEYS:
                torch.testing.assert_close(r[key], baseline[key][order], rtol=0, atol=0)
        pieces = [observe_indexed(**dict(f, means=f['means'][a:b], covariance=f['covariance'][a:b]),
                                   tile_size=3, pair_budget=2) for a, b in ((0, 2), (2, 3), (3, 5))]
        for key in KEYS:
            torch.testing.assert_close(torch.cat([r[key] for r in pieces]), baseline[key], rtol=0, atol=0)

    def test_global_rotation_translation_scale_preserve_native_evidence(self):
        f = self.fixture([(5, 5, 20, True), (6, 5, 20, True), (5, 6, 20, True), (8, 5, 10, False)], sx=1.2, sy=.9)
        baseline = self.check(f, tile_size=2, pair_budget=3)
        Q = t([[.36, -.48, .8], [.8, .6, 0], [-.48, .64, .6]])
        for scale in (1e-4, 2.3, 1e3):
            with self.subTest(scale=scale):
                translation = t([17, -23, 41])*scale
                changed = dict(f, means=scale*(f['means']@Q.T)+translation,
                               covariance=scale**2*(Q@f['covariance']@Q.T),
                               view=torch.cat((Q.T, (-Q.T@translation)[:, None]), 1), depth=scale*f['depth'])
                r = self.check(changed, tile_size=5, pair_budget=2)
                for key in KEYS:
                    torch.testing.assert_close(r[key], baseline[key], rtol=0, atol=0)

    def test_all_six_faces_three_physical_stations_do_not_inflate_or_veto(self):
        f = self.fixture()
        f['depth'].fill_(20); f['valid'].fill_(True); f['strict'].fill_(True)
        rotations = [t([[1,0,0],[0,1,0],[0,0,1]]), t([[-1,0,0],[0,1,0],[0,0,-1]]),
                     t([[0,0,-1],[0,1,0],[1,0,0]]), t([[0,0,1],[0,1,0],[-1,0,0]]),
                     t([[1,0,0],[0,0,1],[0,-1,0]]), t([[1,0,0],[0,0,-1],[0,1,0]])]
        free, blocked, support, groups = [], [], [], []
        for station, center in enumerate((t([0,0,0]), t([1,0,0]), t([-1,0,0]))):
            for face, R in enumerate(rotations):
                r = self.check(dict(f, view=torch.cat((R, (-R@center)[:, None]), 1)), tile_size=3, pair_budget=37)
                self.assertFalse(r['blocked_view'][0])
                self.assertEqual(bool(r['free_view'][0]), face == 0)
                free.append(bool(r['free_view'][0])); blocked.append(bool(r['blocked_view'][0]))
                support.append(bool(r['support_view'][0])); groups.append(str(station))
        arrays = [np.array([x]) for x in (free, blocked, support)]
        flags = reduce_station(*arrays, groups)
        self.assertEqual(int(flags['free'].sum()), 3)
        proposed = select_free_space([0], flags['station_ids'], flags['free'], flags['support'], flags['free'])
        np.testing.assert_array_equal(proposed['indices'], [0])
        duplicate = np.array(list(range(18))+list(range(18)))
        duplicated = reduce_station(*(x[:, duplicate] for x in arrays), np.array(groups)[duplicate])
        for key in flags:
            np.testing.assert_array_equal(duplicated[key], flags[key])
        single = reduce_station(*(x[:, :6] for x in arrays), groups[:6])
        self.assertEqual(select_free_space([0], single['station_ids'], single['free'], single['support'], single['free'])['indices'].size, 0)

    def test_empty_rows_and_empty_evidence(self):
        self.check(self.fixture())
        f = self.fixture([(5, 5, 20, True)])
        f['means'] = f['means'][:0]; f['covariance'] = f['covariance'][:0]
        self.check(f, tile_size=1, pair_budget=1)

    def test_prepared_geometry_matches_uncached_across_cameras(self):
        f = self.fixture([(5, 5, 20, True), (6, 5, 20, True), (5, 6, 20, True)])
        cache = prepare_geometry(f['means'], f['covariance'])
        before = cache.positive.clone()
        for offset in (0., .1, -.2):
            changed = dict(f, view=f['view'].clone())
            changed['view'][0, 3] = offset
            self.check(changed, prepared=cache)
        torch.testing.assert_close(cache.positive, before)

    def test_prepared_geometry_rejects_different_or_in_place_changed_inputs(self):
        for mutation in ('new_means', 'new_covariance', 'means', 'covariance', 'cache'):
            with self.subTest(mutation=mutation):
                f = self.fixture([(5, 5, 20, True)])
                cache = prepare_geometry(f['means'], f['covariance'])
                if mutation == 'new_means':
                    f['means'] = f['means'].clone()
                elif mutation == 'new_covariance':
                    f['covariance'] = f['covariance'].clone()
                elif mutation == 'cache':
                    cache.safe_means.add_(1)
                else:
                    f[mutation].add_(.01)
                with self.assertRaises(ValueError):
                    observe_indexed(**f, prepared=cache)

    def test_solver_workspace_batches_and_cached_validation_are_bounded(self):
        f = self.fixture([(5, 5, 20, True), (6, 5, 20, True), (5, 6, 20, True)])
        n = 8205  # deliberately larger than twice the allowed solver batch
        f['means'] = f['means'].repeat(n, 1)
        f['covariance'] = f['covariance'].repeat(n, 1, 1)
        original_eig, original_inv = torch.linalg.eigvalsh, torch.linalg.inv
        calls = []
        def eig(matrix):
            calls.append(('eig', len(matrix), matrix.shape[-1]))
            self.assertLessEqual(len(matrix), 4096)
            return original_eig(matrix)
        def inv(matrix):
            calls.append(('inv', len(matrix), matrix.shape[-1]))
            self.assertLessEqual(len(matrix), 4096)
            return original_inv(matrix)
        with patch('torch.linalg.eigvalsh', side_effect=eig), patch('torch.linalg.inv', side_effect=inv):
            cache = prepare_geometry(f['means'], f['covariance'])
            self.assertEqual(calls, [('eig', 4096, 3), ('eig', 4096, 3), ('eig', 13, 3)])
            count = len(calls)
            for _ in range(2):
                r = observe_indexed(**f, prepared=cache, pair_budget=4096)
                self.assertEqual(r['strict_free_pixels'].tolist(), [3]*n)
            self.assertEqual(len(calls), count)  # well-conditioned 2D uses no solver

    def test_projected_degeneracy_threshold_falls_back_to_brute_solver(self):
        eps = torch.finfo(torch.float64).eps
        f = self.fixture([(5, 5, 20, True)])
        angle = .37
        R = t([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        for ratio in (32*eps, 64*eps, 128*eps, 1e-10, 1e-6):
            for rotate in (False, True):
                with self.subTest(ratio=ratio, rotate=rotate):
                    covariance = torch.diag(t([1., ratio, .01]))[None]
                    if rotate:
                        covariance = R@covariance@R.T
                    self.check(dict(f, covariance=covariance), pair_budget=3)

    def test_invalid_index_settings_and_strict_mask_rejected(self):
        for setting in (dict(tile_size=0), dict(tile_size=True), dict(pair_budget=0), dict(pair_budget=1.5)):
            with self.assertRaises(ValueError):
                observe_indexed(**self.fixture(), **setting)
        f = self.fixture(); f['strict'][0, 0] = True
        with self.assertRaises(ValueError):
            observe_indexed(**f)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
