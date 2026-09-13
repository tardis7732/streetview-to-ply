"""CPU synthetic correctness checks; no scene files, CUDA, or gsplat required."""
import math
import unittest

import pytest
torch = pytest.importorskip('torch')

from tools.streetview_geometry.losses import (
    alpha_coverage_loss, camera_z, depth_moment_loss,
    local_plane_loss, normal_standard_deviation,
    ray_free_space_loss, render_depth_moments,
)


class GeometryLossTests(unittest.TestCase):
    def test_fixed_confidence_excludes_zero_and_nonfinite_weights(self):
        z = torch.tensor([2., 100., 1000.], requires_grad=True)
        loss = depth_moment_loss(z, z.square(), torch.ones(3), torch.ones(3),
                                 torch.ones(3, dtype=torch.bool), torch.tensor([1., 0., float('nan')]))
        self.assertAlmostEqual(float(loss.detach()), 1.)
        loss.backward()
        torch.testing.assert_close(z.grad, torch.tensor([2., 0., 0.]))

    def test_pixel_shapes_are_not_silently_broadcast(self):
        with self.assertRaises(ValueError):
            depth_moment_loss(torch.ones(2, 2), torch.ones(2, 2), torch.ones(2, 2, 1),
                              torch.ones(2, 2), torch.ones(2, 2, dtype=torch.bool))

    def test_robust_depth_is_pixelwise_and_outlier_independent(self):
        def gradient(outlier, pixelwise):
            z = torch.tensor([1.2, outlier], requires_grad=True)
            loss = depth_moment_loss(z, z.square(), torch.ones(2), torch.ones(2),
                                     torch.ones(2, dtype=torch.bool), robust=pixelwise)
            if not pixelwise:
                loss = torch.sqrt(1+loss)-1
            loss.backward()
            self.assertTrue(torch.isfinite(z.grad).all())
            self.assertTrue(torch.isfinite(loss))
            return z.grad
        mild = gradient(2., True)
        extreme = gradient(10000., True)
        global_extreme = gradient(10000., False)
        torch.testing.assert_close(mild[0], extreme[0])
        self.assertGreater(float(extreme[0]), 1000*float(global_extreme[0]))

    def test_moments_expose_layers_hidden_by_the_mean(self):
        z = torch.tensor([5., 15.], requires_grad=True)
        weights = torch.tensor([.5, .5])
        m1 = (weights*z).sum().reshape(1)
        m2 = (weights*z.square()).sum().reshape(1)
        self.assertEqual(float((m1-10).square().detach()), 0)
        loss = depth_moment_loss(m1, m2, torch.ones(1), torch.tensor([10.]), torch.tensor([True]))
        self.assertAlmostEqual(float(loss.detach()), .25)
        loss.backward()
        self.assertLess(float(z.grad[0]), 0)
        self.assertGreater(float(z.grad[1]), 0)

    def test_pixel_mask_nan_targets_empty_mask_and_cancellation(self):
        m1 = torch.tensor([10., 7.], requires_grad=True)
        m2 = torch.tensor([100.-1e-5, 49.], requires_grad=True)
        loss = depth_moment_loss(m1, m2, torch.ones(2), torch.tensor([10., float('nan')]),
                                 torch.ones(2, dtype=torch.bool))
        self.assertEqual(float(loss.detach()), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(m1.grad).all())
        self.assertEqual(float(m1.grad[1]), 0)
        empty = depth_moment_loss(m1, m2, torch.ones(2), torch.ones(2), torch.zeros(2, dtype=torch.bool))
        self.assertEqual(float(empty.detach()), 0)

    def test_render_wrapper_preserves_depth_feature_gradient(self):
        means = torch.tensor([[0., 0., 5.], [0., 0., 15.]], requires_grad=True)
        seen = {}

        def synthetic_renderer(**kwargs):
            # A deterministic feature compositor checks the wrapper contract,
            # not CUDA rasterization or visibility correctness.
            seen.update(kwargs)
            color = (kwargs['colors'] * kwargs['opacities'][:, None]).sum(0).reshape(1, 1, 1, -1)
            alpha = kwargs['opacities'].sum().reshape(1, 1, 1, 1)
            return color, alpha, {}

        result = render_depth_moments(means, torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1),
                                      torch.ones(2, 3), torch.tensor([.5, .5]), torch.eye(4),
                                      torch.eye(3), 1, 1, rasterizer=synthetic_renderer)
        loss = depth_moment_loss(result['first_moment'], result['second_moment'], result['alpha'],
                                 torch.full((1, 1), 10.), torch.ones((1, 1), dtype=torch.bool))
        loss.backward()
        self.assertEqual(seen['sh_degree'], None)
        self.assertEqual(seen['render_mode'], 'RGB')
        self.assertEqual(seen['colors'].shape[-1], 3)
        self.assertIsNone(seen.get('backgrounds'))  # None is zero feature background.
        torch.testing.assert_close(result['feature_mass'], result['alpha'])
        self.assertLess(float(means.grad[0, 2]), 0)
        self.assertGreater(float(means.grad[1, 2]), 0)
        translated_view = torch.eye(4)
        translated_view[2, 3] = 2
        torch.testing.assert_close(camera_z(means, translated_view), torch.tensor([7., 17.]))

    def test_free_space_is_one_sided_and_preserves_gradients(self):
        z = torch.tensor([5., 10., 15.], requires_grad=True)
        alpha = torch.full((3,), .4, requires_grad=True)
        loss = ray_free_space_loss(z, torch.full((3,), 10.), alpha, torch.ones(3, dtype=torch.bool))
        loss.backward()
        self.assertGreater(float(loss.detach()), 0)
        self.assertLess(float(z.grad[0]), 0)
        self.assertGreater(float(alpha.grad[0]), 0)
        torch.testing.assert_close(z.grad[1:], torch.zeros(2))
        torch.testing.assert_close(alpha.grad[1:], torch.zeros(2))

    def test_quaternion_covariance_and_rotation_gradient(self):
        scales = torch.tensor([[1., 2., 3.]], requires_grad=True)
        n = torch.tensor([[0., 1., 0.]])
        identity = torch.tensor([[1., 0., 0., 0.]])
        quarter_turn = torch.tensor([[math.sqrt(.5), 0., 0., math.sqrt(.5)]])
        torch.testing.assert_close(normal_standard_deviation(scales, identity, n), torch.tensor([2.]))
        torch.testing.assert_close(normal_standard_deviation(scales, quarter_turn, n), torch.tensor([1.]))
        q = torch.tensor([[math.cos(math.pi/8), 0., 0., math.sin(math.pi/8)]], requires_grad=True)
        normal_standard_deviation(scales, q, n).sum().backward()
        self.assertGreater(float(q.grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(q.grad).all())

    def test_ground_mean_tails_and_fixed_different_planes(self):
        means = torch.tensor([[0., -.5, 0.], [0., -100., 0.], [0., 2., 0.]], requires_grad=True)
        scales = torch.tensor([[1., .3, 1.], [1., 1., 1.], [10., .01, 10.]], requires_grad=True)
        quats = torch.tensor([[1., 0., 0., 0.]]).repeat(3, 1).requires_grad_()
        normals = torch.tensor([[0., 1., 0.], [0., 1., 0.], [0., 2., 0.]])
        terms = local_plane_loss(means, scales, quats, normals, torch.tensor([0., 0., -4.]),
                                 torch.tensor([True, False, True]))
        self.assertGreater(float(terms['tail_loss'].detach()), 0)
        terms['loss'].backward()
        self.assertLess(float(means.grad[0, 1]), 0)
        self.assertGreater(float(scales.grad[0, 1]), 0)
        torch.testing.assert_close(means.grad[1:], torch.zeros(2, 3))
        torch.testing.assert_close(scales.grad[1:], torch.zeros(2, 3))
        self.assertEqual(float(scales.grad[0, 0]), 0)  # in-plane extent untouched
        self.assertEqual(float(scales.grad[0, 2]), 0)

    def test_coverage_prevents_fading_escape(self):
        alpha = torch.tensor([.6, 1.], requires_grad=True)
        loss = alpha_coverage_loss(alpha, torch.ones(2), torch.ones(2, dtype=torch.bool))
        loss.backward()
        self.assertLess(float(alpha.grad[0]), 0)
        self.assertEqual(float(alpha.grad[1]), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
