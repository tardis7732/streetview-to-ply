"""Independent geometric/evidence checks for sparse-depth pruning proposals."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from tools.streetview_geometry.free_space import FreeSpaceConfig, select_free_space
from tools.streetview_geometry.sparse_prune import (
    classify, footprint_uv, project, quaternion_covariance, reduce_station,
    sample_uv, unique_pixels, write_subset,
)


DTYPE = torch.float64


def tensor(value):
    return torch.tensor(value, dtype=DTYPE)


def rotation(axis, angle):
    axis = tensor(axis)
    axis /= torch.linalg.vector_norm(axis)
    x, y, z = axis
    zero = x * 0
    cross = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    return torch.eye(3, dtype=DTYPE) + np.sin(angle)*cross + (1-np.cos(angle))*(cross@cross)


class SparseProjectionTests(unittest.TestCase):
    def test_covariance_projection_uses_camera_axes_and_anisotropic_intrinsics(self):
        view = tensor([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]])
        K = tensor([[100, 0, 192], [0, 150, 128], [0, 0, 1]])
        covariance = torch.diag(tensor([.04, .09, .16]))[None]
        xyz, uv, sigma = project(tensor([[-10, 0, 0]]), covariance, view, K)
        torch.testing.assert_close(xyz, tensor([[0, 0, 10]]))
        torch.testing.assert_close(uv, tensor([[192, 128]]))
        torch.testing.assert_close(sigma, tensor([.2]))
        samples = footprint_uv(xyz, covariance, view, K, uv)
        self.assertEqual(tuple(samples.shape), (1, 9, 2))
        self.assertAlmostEqual(float((samples[0, :, 0]-192).abs().max()), 8.)
        self.assertAlmostEqual(float((samples[0, :, 1]-128).abs().max()), 9.)

    def test_global_similarity_preserves_pixels_footprint_and_decisions(self):
        camera_rotation = rotation([.3, -.2, 1], .43)
        camera_translation = tensor([.2, -.3, .4])
        view = torch.cat((camera_rotation, camera_translation[:, None]), dim=1)
        K = tensor([[431, 0, 511.5], [0, 377, 383.5], [0, 0, 1]])
        camera_xyz = tensor([[1.2, -.4, 12], [-.8, .6, 18], [0, 0, 25]])
        means = (camera_xyz-camera_translation) @ camera_rotation
        bases = tensor([[[.09, .03, -.02], [0, .07, .01], [.01, 0, .04]],
                        [[.1, -.02, 0], [.03, .05, .01], [0, .02, .08]],
                        [[.08, 0, .01], [.02, .08, 0], [.01, 0, .09]]])
        covariance = bases @ bases.transpose(-1, -2)
        xyz, uv, sigma = project(means, covariance, view, K)
        footprint = footprint_uv(xyz, covariance, view, K, uv)
        valid = torch.ones(3, dtype=torch.bool)
        depth = tensor([20, 18, 12])
        expected = classify(xyz[:, 2], sigma, depth, valid, FreeSpaceConfig())
        world_rotation = rotation([1, -2, .7], .83)
        for scale in (1e-4, 1., 1e3):
            with self.subTest(scale=scale):
                translation = tensor([17, -23, 41]) * scale
                changed_means = scale * (means @ world_rotation.T) + translation
                changed_covariance = scale**2 * (world_rotation @ covariance @ world_rotation.T)
                new_rotation = camera_rotation @ world_rotation.T
                new_translation = scale * camera_translation - new_rotation @ translation
                changed_view = torch.cat((new_rotation, new_translation[:, None]), dim=1)
                changed_xyz, changed_uv, changed_sigma = project(changed_means, changed_covariance, changed_view, K)
                torch.testing.assert_close(changed_xyz, scale*xyz, rtol=1e-11, atol=1e-11*scale)
                torch.testing.assert_close(changed_uv, uv, rtol=1e-11, atol=1e-9)
                torch.testing.assert_close(changed_sigma, scale*sigma, rtol=1e-11, atol=1e-11*scale)
                changed_footprint = footprint_uv(changed_xyz, changed_covariance, changed_view, K, changed_uv)
                torch.testing.assert_close(changed_footprint, footprint, rtol=1e-11, atol=1e-9)
                actual = classify(changed_xyz[:, 2], changed_sigma, scale*depth, valid, FreeSpaceConfig())
                for before, after in zip(expected, actual):
                    torch.testing.assert_close(before, after)

    def test_repeated_projected_eigenvalues_do_not_rotate_the_sample_axes(self):
        K = tensor([[128, 0, 128], [0, 128, 128], [0, 0, 1]])
        view = torch.eye(4, dtype=DTYPE)[:3]
        means = tensor([[0, 0, 8]])
        covariance = torch.eye(3, dtype=DTYPE)[None] * .04
        xyz, uv, sigma = project(means, covariance, view, K)
        before = footprint_uv(xyz, covariance, view, K, uv)
        Q = rotation([.3, .7, -.9], .73)
        shifted_means = means @ Q.T + tensor([10, -2, 7])
        shifted_view = torch.cat((Q.T, (-Q.T @ tensor([10, -2, 7]))[:, None]), dim=1)
        shifted_covariance = Q @ covariance @ Q.T
        xyz2, uv2, _ = project(shifted_means, shifted_covariance, shifted_view, K)
        after = footprint_uv(xyz2, shifted_covariance, shifted_view, K, uv2)
        torch.testing.assert_close(after, before, rtol=1e-11, atol=1e-10)
        torch.testing.assert_close(before[0, 1], tensor([121.6, 128]))
        torch.testing.assert_close(before[0, 5], tensor([128, 121.6]))

    def test_quaternion_normalization_and_sign_preserve_covariance(self):
        q = tensor([[np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)]])
        scales = tensor([[.2, .3, .4]])
        expected = torch.diag(tensor([.09, .04, .16]))[None]
        torch.testing.assert_close(quaternion_covariance(q*7, scales), expected)
        torch.testing.assert_close(quaternion_covariance(-q, scales), expected)


class SparseSamplingTests(unittest.TestCase):
    def test_half_pixel_centers_borders_and_nonfinite_uv_are_not_clamped_evidence(self):
        depth = torch.arange(1, 13, dtype=DTYPE).reshape(3, 4)
        valid = torch.ones((3, 4), dtype=torch.bool)
        valid[1, 1] = False
        uv = tensor([[[0, 0], [.5, .5], [3.9999, 2.9999], [4, 2.5],
                      [-.0001, .5], [.5, 3], [float('nan'), 1],
                      [float('inf'), 1], [-float('inf'), 1], [1.5, 1.5]]])
        values, keep, pixels = sample_uv(uv, depth, valid)
        torch.testing.assert_close(keep, torch.tensor([[True, True, True, False, False, False, False, False, False, False]]))
        torch.testing.assert_close(pixels, torch.tensor([[0, 0, 11, -1, -1, -1, -1, -1, -1, -1]]))
        torch.testing.assert_close(values[keep], tensor([1, 1, 12]))

    def test_unknown_nonpositive_and_nonfinite_depth_never_votes(self):
        depth = tensor([[0, -1, float('nan'), float('inf'), 7]])
        valid = torch.ones_like(depth, dtype=torch.bool)
        uv = tensor([[[i+.5, .5] for i in range(5)]])
        _, keep, pixels = sample_uv(uv, depth, valid)
        torch.testing.assert_close(keep, torch.tensor([[False, False, False, False, True]]))
        torch.testing.assert_close(pixels, torch.tensor([[-1, -1, -1, -1, 4]]))

    def test_duplicate_count_ignores_invalid_earlier_occurrences(self):
        ids = torch.tensor([[0, 0, 1, 1, 0, -1], [4, 4, 4, 5, 6, 6]])
        valid = torch.tensor([[False, True, True, False, True, False], [True, True, True, True, True, True]])
        expected = torch.tensor([[False, True, True, False, False, False], [True, False, False, True, True, False]])
        torch.testing.assert_close(unique_pixels(ids, valid), expected)
        torch.testing.assert_close(valid, torch.tensor([[False, True, True, False, True, False], [True, True, True, True, True, True]]))

    def test_whole_depth_extent_support_and_behind_are_not_removed(self):
        z = tensor([2, 2, 10, 15, -2, 2])
        sigma = tensor([.1, 3, .1, .1, .1, .1])
        depth = tensor([10, 10, 10, 10, 10, 10])
        valid = torch.tensor([True, True, True, True, True, False])
        free, support = classify(z, sigma, depth, valid, FreeSpaceConfig())
        torch.testing.assert_close(free, torch.tensor([True, False, False, False, False, False]))
        torch.testing.assert_close(support, torch.tensor([False, True, True, False, False, False]))


class PhysicalStationTests(unittest.TestCase):
    def fixture(self):
        groups = ['a', 'a', 'b', 'c', 'd']
        free = np.array([[1, 0, 1, 1, 0], [1, 0, 1, 1, 0], [1, 0, 1, 1, 0]], bool)
        blocked = np.array([[0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], bool)
        support = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]], bool)
        return groups, free, blocked, support

    def test_nonfree_other_face_veto_and_any_surface_support_prevent_removal(self):
        groups, free, blocked, support = self.fixture()
        result = reduce_station(free, blocked, support, groups)
        np.testing.assert_array_equal(result['station_ids'], ['a', 'b', 'c', 'd'])
        np.testing.assert_array_equal(result['free'], [[0, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0]])
        selected = select_free_space(np.arange(3), result['station_ids'], result['free'], result['support'], result['free'])
        np.testing.assert_array_equal(selected['indices'], [2])

    def test_view_reordering_and_duplicate_faces_do_not_inflate_station_votes(self):
        groups, free, blocked, support = self.fixture()
        original = reduce_station(free, blocked, support, groups)
        indices = [4, 3, 2, 1, 0, 0, 2, 0]
        changed = reduce_station(free[:, indices], blocked[:, indices], support[:, indices], np.array(groups)[indices])
        for key in original:
            np.testing.assert_array_equal(original[key], changed[key])
        one_station = reduce_station(np.ones((1, 6), bool), np.zeros((1, 6), bool), np.zeros((1, 6), bool), ['same']*6)
        proposed = select_free_space([0], one_station['station_ids'], one_station['free'], one_station['support'], one_station['free'])
        self.assertEqual(proposed['indices'].size, 0)

    def test_float_nan_and_missing_station_metadata_are_rejected(self):
        with self.assertRaises(ValueError):
            reduce_station(np.array([[float('nan')]]), np.zeros((1, 1), bool), np.zeros((1, 1), bool), ['a'])
        with self.assertRaises(ValueError):
            reduce_station(np.zeros((1, 2), bool), np.zeros((1, 2), bool), np.zeros((1, 2), bool), ['a'])


class ExactSubsetTests(unittest.TestCase):
    def test_binary_sh2_subset_keeps_opacity_every_attribute_and_order_exactly(self):
        # Construct bytes independently of the production writer and cover
        # both PLY endian conventions and header newline formats.
        names = ['x', 'y', 'z'] + [f'f_dc_{i}' for i in range(3)]
        names += [f'f_rest_{i}' for i in range(24)] + ['opacity']
        names += [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)]
        keep = np.array([False, True, False, True, True, False])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for index, (byte_order, label, newline) in enumerate((
                ('<', 'binary_little_endian', b'\n'),
                ('>', 'binary_big_endian', b'\r\n'),
            )):
                with self.subTest(byte_order=byte_order):
                    rows = np.zeros(6, dtype=[(name, byte_order+'f4') for name in names])
                    for field_index, name in enumerate(names):
                        rows[name] = np.arange(6, dtype=np.float32)*.13 + field_index*.17
                    rows['opacity'] = [-7., -.0, .25, -3.5, 2., .001]
                    lines = [b'ply', f'format {label} 1.0'.encode(), b'comment preserve this metadata', b'element vertex 6']
                    lines += [f'property float {name}'.encode() for name in names] + [b'end_header']
                    header = newline.join(lines)+newline
                    source, destination = root/f'{index}_source.ply', root/f'{index}_subset.ply'
                    source_bytes = header+rows.tobytes()
                    source.write_bytes(source_bytes)
                    source_sha = hashlib.sha256(source_bytes).hexdigest()
                    result = write_subset(source, destination, keep, source_sha)
                    expected_header = header.replace(b'element vertex 6', b'element vertex 3')
                    expected_payload = rows[keep].tobytes()
                    self.assertEqual(destination.read_bytes(), expected_header+expected_payload)
                    self.assertEqual(source.read_bytes(), source_bytes)
                    self.assertEqual(result['rows'], 3)
                    self.assertEqual(result['retained_payload_sha256'], hashlib.sha256(expected_payload).hexdigest())
                    with self.assertRaises(ValueError):
                        write_subset(source, destination, keep, source_sha)
                    self.assertEqual(destination.read_bytes(), expected_header+expected_payload)
                    bad_destination = root/f'{index}_wrong_hash.ply'
                    with self.assertRaises(ValueError):
                        write_subset(source, bad_destination, keep, '0'*64)
                    self.assertFalse(bad_destination.exists())


if __name__ == '__main__':
    unittest.main()
