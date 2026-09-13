"""Depth-only non-sky abstention: projection, conservative maxima and invariants."""
import unittest

import torch

from tools.streetview_geometry.center_occlusion import prepare_center_occlusion, observe_center_occlusion
from tools.streetview_geometry.semantic_center import SemanticCenterConfig, prepare_semantic_center, observe_semantic_center


def t(value):
    return torch.tensor(value, dtype=torch.float64)


class CenterOcclusionTests(unittest.TestCase):
    def sample(self, means, threshold=None, valid=None, K=None, view=None, chunk=65536):
        threshold = torch.full((9,9),5.,dtype=torch.float64) if threshold is None else threshold
        valid = torch.ones_like(threshold,dtype=torch.bool) if valid is None else valid
        prepared = prepare_center_occlusion(threshold, valid)
        return observe_center_occlusion(t(means), torch.eye(4,dtype=torch.float64) if view is None else view,
                                        t([[4,0,4.5],[0,4,4.5],[0,0,1]]) if K is None else K,
                                        prepared, chunk_rows=chunk)

    def test_only_hidden_center_abstains_foreground_equal_and_sky_unchanged(self):
        result = self.sample([(0,0,2),(0,0,5),(0,0,6),(0,0,10)])
        self.assertEqual(result['occluded_view'].tolist(), [False,False,True,True])
        raw_non_sky = torch.tensor([True,True,True,False])
        raw_sky = torch.tensor([False,False,False,True])
        non_sky_before, sky_before = raw_non_sky.clone(), raw_sky.clone()
        retained = raw_non_sky & ~result['occluded_view']
        self.assertEqual(retained.tolist(), [True,True,False,False])
        self.assertTrue(torch.equal(raw_non_sky, non_sky_before))
        self.assertTrue(torch.equal(raw_sky, sky_before))
        self.assertTrue(raw_sky[3]) # Occluded flag is NEVER a request to clear sky.

    def test_all_nine_pixels_maximum_protects_depth_discontinuity(self):
        for y in [3,4,5]:
            for x in [3,4,5]:
                depth = torch.full((9,9),2.,dtype=torch.float64)
                depth[y,x] = 12.
                result = self.sample([(0,0,6),(0,0,12),(0,0,13)], depth)
                self.assertEqual(result['occluded_view'].tolist(), [False,False,True], (y,x))
        depth = torch.full((9,9),2.,dtype=torch.float64)
        depth[4,6] = 100 # Exactly outside the 3x3; must not expand arbitrarily.
        self.assertTrue(self.sample([(0,0,6)],depth)['occluded_view'][0])

    def test_any_invalid_of_nine_retains_raw_non_sky(self):
        for y in [3,4,5]:
            for x in [3,4,5]:
                valid = torch.ones((9,9),dtype=torch.bool)
                valid[y,x] = False
                result = self.sample([(0,0,100)], valid=valid)
                self.assertFalse(result['threshold_known_view'][0], (y,x))
                self.assertFalse(result['occluded_view'][0], (y,x))

    def test_nonfinite_nonpositive_threshold_is_unknown_even_if_marked_valid(self):
        for value in [float('nan'),float('inf'),0.,-1.]:
            depth = torch.full((9,9),2.,dtype=torch.float64)
            depth[3,5] = value
            result = self.sample([(0,0,100)],depth)
            self.assertFalse(result['threshold_known_view'][0])
            self.assertFalse(result['occluded_view'][0])

    def test_depth_pixel_centres_and_floor_lookup_no_half_pixel_shift(self):
        depth = torch.full((7,8),2.,dtype=torch.float64)
        depth[:,5:] = 10
        # x=3 neighbourhood max2, x=4 max10; K identity gives direct x/z=u.
        points = [(u*6,3.5*6,6) for u in [3.01,3.5,3.99,4.,4.01,4.5]]
        result = self.sample(points,depth,K=torch.eye(3,dtype=torch.float64))
        self.assertEqual(result['occluded_view'].tolist(), [True,True,True,False,False,False])

    def test_different_native_semantic_and_depth_resolutions_share_rays(self):
        native_mask = torch.zeros((64,64),dtype=torch.bool)
        native_mask[:, :24] = True
        semantic = prepare_semantic_center(native_mask, config=SemanticCenterConfig(0))
        native_K = t([[32,0,32],[0,32,32],[0,0,1]])
        depth_K = t([[8,0,8],[0,8,8],[0,0,1]])
        means = t([[-1,0,6],[1,0,6],[-3,0,6],[0,0,1]])
        raw = observe_semantic_center(means,torch.eye(4),native_K,semantic)
        self.assertEqual(raw['sky_view'].tolist(), [False,False,True,False])
        depth = torch.full((16,16),2.,dtype=torch.float64)
        small = observe_center_occlusion(means,torch.eye(4),depth_K,prepare_center_occlusion(depth,torch.ones_like(depth,dtype=torch.bool)))
        large_depth = depth.repeat_interleave(4,0).repeat_interleave(4,1)
        large = observe_center_occlusion(means,torch.eye(4),native_K,prepare_center_occlusion(large_depth,torch.ones_like(large_depth,dtype=torch.bool)))
        self.assertTrue(torch.equal(small['occluded_view'],large['occluded_view']))
        self.assertEqual((raw['non_sky_view'] & ~small['occluded_view']).tolist(), [False,False,False,True])
        self.assertEqual(raw['sky_view'].tolist(), [False,False,True,False])

    def test_image_borders_incomplete_neighbourhood_outside_and_behind_unknown(self):
        depth = torch.ones((5,5),dtype=torch.float64)
        # z=2; means x/y doubled so u/v are direct coordinates.
        uv = [(0.,2.5),(.99,2.5),(1.,2.5),(3.99,2.5),(4.,2.5),(5.,2.5),(-.01,2.5),(2.5,0.),(2.5,4.)]
        points = [(x*2,y*2,2) for x,y in uv]+[(0,0,0),(0,0,-2),(float('nan'),0,2)]
        result = self.sample(points,depth,K=torch.eye(3,dtype=torch.float64))
        self.assertEqual(result['occluded_view'].tolist(),[False,False,True,True,False,False,False,False,False,False,False,False])
        self.assertFalse(result['threshold_known_view'][-3:].any())

    def test_rotation_translation_positive_scale_invariance(self):
        torch.manual_seed(931)
        means = torch.randn(901,3,dtype=torch.float64)
        means[:,2] = means[:,2].abs()+3
        depth = 2+torch.rand((32,32),dtype=torch.float64)*3
        valid = torch.ones_like(depth,dtype=torch.bool)
        valid[12,16] = False
        K = t([[12,0,16.5],[0,12,16.5],[0,0,1]])
        result = observe_center_occlusion(means,torch.eye(4),K,prepare_center_occlusion(depth,valid))
        Q,_ = torch.linalg.qr(torch.randn((3,3),dtype=torch.float64))
        if torch.det(Q) < 0:
            Q[:,0] *= -1
        shift = t([10.5,-42.7,6.3])
        view = torch.eye(4,dtype=torch.float64)
        view[:3,:3] = Q.T
        view[:3,3] = -Q.T@shift
        for scale in [1e-4,.01,1.,10.,1e3]:
            transformed = scale*(means@Q.T)+shift
            p = prepare_center_occlusion(depth*scale,valid)
            changed = observe_center_occlusion(transformed,view,K,p,chunk_rows=17)
            for key in ['occluded_view','threshold_known_view','front_view','in_image']:
                self.assertTrue(torch.equal(result[key],changed[key]), (scale,key))

    def test_chunking_and_empty_inputs(self):
        points = [(0,0,1+i*.01) for i in range(1400)]
        a,b = self.sample(points,chunk=13),self.sample(points,chunk=2000)
        for key in ['occluded_view','threshold_known_view','front_view','in_image']:
            self.assertTrue(torch.equal(a[key],b[key]))
        self.assertEqual(a['diagnostics']['max_projection_batch'],13)
        p = prepare_center_occlusion(torch.ones((1,1)),torch.ones((1,1),dtype=torch.bool))
        e = observe_center_occlusion(torch.zeros((0,3)),torch.eye(4),torch.eye(3),p)
        self.assertEqual(e['occluded_view'].shape,(0,))
        e = observe_center_occlusion(t([[1,1,2]]),torch.eye(4),torch.eye(3),p)
        self.assertFalse(e['threshold_known_view'][0])

    def test_prepared_snapshot_and_mutation_protection(self):
        depth = torch.ones((5,5),dtype=torch.float64)
        valid = torch.ones_like(depth,dtype=torch.bool)
        p = prepare_center_occlusion(depth,valid)
        depth[:] = 100
        valid[:] = False
        result = observe_center_occlusion(t([[5,5,2]]),torch.eye(4),torch.eye(3),p)
        self.assertTrue(result['occluded_view'][0])
        p.threshold_max_z[2,2] = 100
        with self.assertRaisesRegex(ValueError,'changed in place'):
            observe_center_occlusion(t([[5,5,2]]),torch.eye(4),torch.eye(3),p)

    def test_bad_camera_threshold_validity_and_chunk_rejected(self):
        with self.assertRaises(ValueError):
            prepare_center_occlusion(torch.ones((3,3),dtype=torch.int32),torch.ones((3,3),dtype=torch.bool))
        with self.assertRaises(ValueError):
            prepare_center_occlusion(torch.ones((3,3)),torch.ones((2,3),dtype=torch.bool))
        with self.assertRaises(ValueError):
            self.sample([(0,0,3)],chunk=True)
        view = torch.eye(4,dtype=torch.float64)
        view[2,2] = -1
        with self.assertRaises(ValueError):
            self.sample([(0,0,3)],view=view)


if __name__ == '__main__':
    unittest.main()
