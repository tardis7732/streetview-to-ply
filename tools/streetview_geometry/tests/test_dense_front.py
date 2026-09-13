"""Dense front-of-surface evidence: geometry bounds, unknowns and station votes."""
import unittest
from dataclasses import replace

import torch

from tools.streetview_geometry.dense_front import (
    DenseFrontConfig, _perspective_bounds, _query, merge_station_views,
    observe_dense_front, prepare_depth, prepare_geometry, select_dense_front,
)


def t(value):
    return torch.tensor(value, dtype=torch.float64)


class DenseFrontTests(unittest.TestCase):
    def fixture(self, means=((0, 0, 5),), sigma=(.02, .02, .02), size=32):
        means = t(means)
        return dict(means=means, covariance=torch.diag(t(sigma).square()).repeat(len(means), 1, 1),
                    view=torch.eye(4, dtype=torch.float64),
                    K=t([[20, 0, size/2+.5], [0, 20, size/2+.5], [0, 0, 1]]),
                    depth=t(10.).expand(size, size).clone(),
                    valid=torch.ones((size, size), dtype=torch.bool))

    def observe(self, f, config=DenseFrontConfig(), uncertainty=0., **options):
        prepared = prepare_depth(f['depth'], f['valid'], uncertainty, config=config)
        return observe_dense_front(f['means'], f['covariance'], f['view'], f['K'],
                                   prepared, config=config, **options)

    def test_single_native_pixel_can_certify_small_foreground_floater(self):
        result = self.observe(self.fixture())
        self.assertEqual(result['footprint_pixels'].tolist(), [1])
        self.assertEqual(result['valid_pixels'].tolist(), [1])
        self.assertEqual(result['free_view'].tolist(), [True])
        self.assertFalse(result['possible_support_view'][0])

    def test_normal_surface_and_behind_surface_are_not_free(self):
        result = self.observe(self.fixture(means=((0,0,10), (0,0,12), (0,0,5))))
        self.assertEqual(result['free_view'].tolist(), [False, False, True])
        self.assertEqual(result['possible_support_view'].tolist(), [True, False, False])
        self.assertEqual(result['occluded_view'].tolist(), [False, True, False])

    def test_nearer_occluder_at_footprint_border_protects(self):
        f = self.fixture(sigma=(.1, .1, .02))
        f['depth'][16,17] = 4.
        self.assertEqual(f['depth'][16,16].item(), 10.)
        result = self.observe(f)
        self.assertFalse(result['free_view'][0])
        self.assertTrue(result['possible_support_view'][0])

    def test_missing_center_is_coverage_unknown_not_a_center_gate(self):
        f = self.fixture(sigma=(.1, .1, .02))
        f['valid'][16,16] = False
        strict = self.observe(f)
        relaxed = self.observe(f, replace(DenseFrontConfig(), minimum_valid_fraction=.8))
        self.assertEqual(strict['footprint_pixels'].tolist(), [9])
        self.assertAlmostEqual(strict['valid_fraction'][0].item(), 8/9, places=6)
        self.assertTrue(strict['unknown_view'][0])
        self.assertTrue(relaxed['free_view'][0])

    def test_masked_sky_is_unknown_even_when_pyramid_has_nearby_depth(self):
        f = self.fixture(sigma=(.1, .1, .02))
        f['valid'][15:18,15:18] = False
        result = self.observe(f)
        self.assertTrue(result['unknown_view'][0])
        self.assertFalse(result['free_view'][0])
        self.assertFalse(result['possible_support_view'][0])

    def test_any_projected_extent_outside_image_abstains(self):
        f = self.fixture(means=((-4.,0,5),), sigma=(.2,.1,.02))
        result = self.observe(f)
        self.assertTrue(result['projectable_view'][0])
        self.assertFalse(result['fully_in_image'][0])
        self.assertFalse(result['free_view'][0])
        self.assertTrue(result['unknown_view'][0])

    def test_camera_crossing_huge_and_invalid_gaussians_abstain(self):
        f = self.fixture(means=((0,0,5), (0,0,-1), (0,0,5), (float('nan'),0,5)))
        f['covariance'][0] = torch.eye(3, dtype=torch.float64)*4
        f['covariance'][2] = torch.diag(t([100.,100.,.01]))
        result = self.observe(f, chunk_rows=1)
        self.assertEqual(result['free_view'].tolist(), [False]*4)
        self.assertEqual(result['unknown_view'].tolist(), [True]*4)
        self.assertEqual(result['projectable_view'].tolist(), [False,False,True,False])

    def test_explicit_absolute_uncertainty_and_confidence_are_distinct(self):
        f = self.fixture()
        high_uncertainty = self.observe(f, uncertainty=6.)
        self.assertFalse(high_uncertainty['free_view'][0])
        self.assertTrue(high_uncertainty['possible_support_view'][0])
        cfg = replace(DenseFrontConfig(), minimum_confidence=.8)
        confidence = torch.ones_like(f['depth'])*.7
        prepared = prepare_depth(f['depth'], f['valid'], 0., confidence=confidence, config=cfg)
        result = observe_dense_front(f['means'],f['covariance'],f['view'],f['K'],prepared,config=cfg)
        self.assertTrue(result['unknown_view'][0])
        with self.assertRaises(ValueError):
            prepare_depth(f['depth'], f['valid'], 0., config=cfg)

    def test_invalid_depth_or_uncertainty_is_unknown(self):
        for value, uncertainty in ((float('nan'),0.), (0.,0.), (float('inf'),0.), (10.,-1.), (10.,float('nan'))):
            with self.subTest(value=value, uncertainty=uncertainty):
                f = self.fixture()
                f['depth'].fill_(value)
                self.assertTrue(self.observe(f, uncertainty=uncertainty)['unknown_view'][0])

    def test_asymmetric_bounds_are_preserved_with_separate_relative_margin(self):
        f = self.fixture()
        lower, upper = torch.full_like(f['depth'],8.), torch.full_like(f['depth'],16.)
        prepared = prepare_depth(f['depth'],f['valid'],depth_lower_z=lower,depth_upper_z=upper)
        self.assertEqual(prepared.lower_pyramid[0][16,16].item(),7.)
        self.assertEqual(prepared.upper_pyramid[0][16,16].item(),17.)
        result = observe_dense_front(f['means'],f['covariance'],f['view'],f['K'],prepared)
        self.assertTrue(result['free_view'][0])
        # A symmetric max-side error of 6 would discard this valid free vote.
        self.assertFalse(self.observe(f,uncertainty=6.)['free_view'][0])

    def test_asymmetric_bounds_take_hull_and_reject_invalid_interval_pixels(self):
        f = self.fixture()
        lower, upper = torch.full_like(f['depth'],11.), torch.full_like(f['depth'],14.)
        prepared = prepare_depth(f['depth'],f['valid'],depth_lower_z=lower,depth_upper_z=upper)
        self.assertEqual(prepared.lower_pyramid[0][16,16].item(),9.)
        self.assertEqual(prepared.upper_pyramid[0][16,16].item(),15.)
        lower.fill_(15.)
        prepared = prepare_depth(f['depth'],f['valid'],depth_lower_z=lower,depth_upper_z=upper)
        self.assertFalse(prepared.valid.any())
        with self.assertRaises(ValueError):
            prepare_depth(f['depth'],f['valid'],1.,depth_lower_z=lower,depth_upper_z=upper)
        with self.assertRaises(ValueError):
            prepare_depth(f['depth'],f['valid'],depth_lower_z=lower)

    def test_half_pixel_convention_and_cell_boundary_occluder(self):
        f = self.fixture()
        f['depth'][16,16] = 4.
        self.assertTrue(self.observe(f)['occluded_view'][0])
        # Move the projected centre to x=17.0: its footprint crosses cells 16,17.
        f['means'][0,0] = .125
        result = self.observe(f)
        self.assertEqual(result['footprint_pixels'].tolist(), [2])
        self.assertTrue(result['possible_support_view'][0])
        self.assertFalse(result['free_view'][0])

    def test_rotated_anisotropic_exact_bounds_enclose_projected_ellipsoid(self):
        generator = torch.Generator().manual_seed(18)
        angles = t([.3,.7,1.2])
        for angle in angles:
            rotation = t([[torch.cos(angle), 0, torch.sin(angle)], [0,1,0], [-torch.sin(angle),0,torch.cos(angle)]])
            axes = t([.2,.07,.4])
            covariance = rotation @ torch.diag(axes.square()) @ rotation.T
            mu = t([[1.2,-.7,5.]])
            K = t([[40,0,30.5],[0,35,24.5],[0,0,1]])
            bounds, front, back, valid = _perspective_bounds(mu,covariance[None],K,3.)
            directions = torch.randn((5000,3), generator=generator, dtype=torch.float64)
            directions /= directions.norm(dim=1,keepdim=True)
            points = mu+3*(directions*axes)@rotation.T
            uv = points[:,:2]/points[:,2,None]*t([40,35])+t([30.5,24.5])
            self.assertTrue(valid[0])
            self.assertTrue((uv[:,0] >= bounds[0,0]).all() and (uv[:,0] <= bounds[0,1]).all())
            self.assertTrue((uv[:,1] >= bounds[0,2]).all() and (uv[:,1] <= bounds[0,3]).all())
            self.assertTrue((points[:,2] >= front[0]).all() and (points[:,2] <= back[0]).all())

    def test_world_rotation_translation_scale_invariance(self):
        f = self.fixture(means=((0,0,5), (0,0,10), (.1,.2,12)), sigma=(.05,.1,.02))
        expected = self.observe(f, uncertainty=.2)
        angle = .73
        rotation = t([[math_cos(angle),0,math_sin(angle)],[0,1,0],[-math_sin(angle),0,math_cos(angle)]])
        for scale in (1e-4, .3, 1000.):
            translation = t([7,-4,12])*scale
            updated = dict(f)
            updated['means'] = scale*f['means']@rotation.T+translation
            updated['covariance'] = scale*scale*rotation@f['covariance']@rotation.T
            view = torch.eye(4,dtype=torch.float64)
            view[:3,:3] = rotation.T
            view[:3,3] = -rotation.T@translation
            updated['view'] = view
            updated['depth'] = f['depth']*scale
            actual = self.observe(updated, uncertainty=.2*scale)
            for key in ('free_view','possible_support_view','occluded_view','unknown_view','valid_pixels','footprint_pixels'):
                torch.testing.assert_close(actual[key], expected[key], rtol=0,atol=0,msg=key)

    def test_pyramid_min_max_conservatively_bound_exact_rectangles(self):
        generator = torch.Generator().manual_seed(31)
        for shape in ((19,31),(1,7),(9,1),(1,1)):
            depth = torch.rand(shape,generator=generator,dtype=torch.float64)*10+1
            valid = torch.rand(shape,generator=generator)>.15
            cfg = replace(DenseFrontConfig(),relative_depth_margin=0.)
            prepared = prepare_depth(depth,valid,0.,config=cfg)
            for _ in range(30):
                x = torch.randint(shape[1],(2,),generator=generator).sort().values
                y = torch.randint(shape[0],(2,),generator=generator).sort().values
                count,lower,upper = _query(prepared,x[:1],x[1:],y[:1],y[1:])
                crop = depth[y[0]:y[1]+1,x[0]:x[1]+1]
                mask = valid[y[0]:y[1]+1,x[0]:x[1]+1]
                self.assertEqual(count.item(),int(mask.sum()))
                if mask.any():
                    self.assertLessEqual(lower.item(),crop[mask].min().item())
                    self.assertGreaterEqual(upper.item(),crop[mask].max().item())

    def test_chunking_order_and_prepared_geometry_do_not_change_results(self):
        generator = torch.Generator().manual_seed(29)
        f = self.fixture(means=tuple((0,0,5) for _ in range(57)),sigma=(.2,.1,.03))
        f['means'][:,:2] = torch.randn((57,2),generator=generator,dtype=torch.float64)*2
        f['means'][:,2] = torch.rand(57,generator=generator,dtype=torch.float64)*15+.1
        cache = prepare_geometry(f['means'],f['covariance'])
        expected = self.observe(f,prepared_geometry=cache,chunk_rows=1000)
        for chunk in (1,7,32):
            actual = self.observe(f,prepared_geometry=cache,chunk_rows=chunk)
            self.assertLessEqual(actual['diagnostics']['maximum_geometry_batch'],chunk)
            for key in expected:
                if key != 'diagnostics':
                    torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0,msg=key)
        order = torch.randperm(57,generator=generator)
        permuted = dict(f,means=f['means'][order],covariance=f['covariance'][order])
        actual = self.observe(permuted,chunk_rows=9)
        for key in expected:
            if key != 'diagnostics':
                torch.testing.assert_close(actual[key],expected[key][order],rtol=0,atol=0,msg=key)

    def test_cube_faces_are_one_physical_station_and_any_support_protects(self):
        free_view = dict(free_view=torch.tensor([True,True]),possible_support_view=torch.tensor([False,False]))
        support_view = dict(free_view=torch.tensor([False,False]),possible_support_view=torch.tensor([False,True]))
        station = merge_station_views([free_view]*6)
        one_station = select_dense_front({'capture_a':station})
        self.assertEqual(one_station['free_station_count'].tolist(),[1,1])
        self.assertEqual(one_station['remove'].tolist(),[False,False])
        station_with_support = merge_station_views([free_view,support_view])
        result = select_dense_front({'capture_a':station,'capture_b':station,'capture_c':station_with_support})
        self.assertEqual(result['free_station_count'].tolist(),[3,2])
        self.assertEqual(result['possible_support_station_count'].tolist(),[0,1])
        self.assertEqual(result['remove'].tolist(),[True,False])

    def test_occlusion_and_unknown_views_are_neutral_and_counts_reach_500(self):
        free = dict(free_view=torch.tensor([True]),possible_support_view=torch.tensor([False]))
        neutral = dict(free_view=torch.tensor([False]),possible_support_view=torch.tensor([False]))
        station = merge_station_views([free,neutral])
        result = select_dense_front({str(i):station for i in range(500)})
        self.assertEqual(result['free_station_count'].item(),500)
        self.assertTrue(result['remove'][0])

    def test_preparation_snapshot_and_mutation_guards(self):
        f = self.fixture()
        prepared = prepare_depth(f['depth'],f['valid'],0.)
        f['depth'].fill_(1.)
        self.assertTrue(observe_dense_front(f['means'],f['covariance'],f['view'],f['K'],prepared)['free_view'][0])
        prepared.lower_pyramid[0][0,0] = 12.
        with self.assertRaisesRegex(ValueError,'modified'):
            observe_dense_front(f['means'],f['covariance'],f['view'],f['K'],prepared)

    def test_invalid_policy_camera_and_ambiguous_station_ids_rejected(self):
        for kwargs in (dict(minimum_valid_fraction=0),dict(gaussian_sigma=0),dict(minimum_free_stations=1),dict(minimum_valid_pixels=True)):
            with self.assertRaises(ValueError):
                DenseFrontConfig(**kwargs)
        f = self.fixture()
        f['view'][0,0] = 2.
        with self.assertRaises(ValueError):
            self.observe(f)
        station = dict(free=torch.tensor([True]),possible_support=torch.tensor([False]))
        with self.assertRaises(ValueError):
            select_dense_front({1:station,'1':station})


def math_sin(x):
    return torch.sin(torch.tensor(x,dtype=torch.float64)).item()


def math_cos(x):
    return torch.cos(torch.tensor(x,dtype=torch.float64)).item()


if __name__ == '__main__':
    unittest.main()
