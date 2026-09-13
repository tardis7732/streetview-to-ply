"""Validated camera conventions, physical panorama groups and scene units."""
from dataclasses import dataclass
import hashlib
import numpy as np


@dataclass(frozen=True)
class CameraSet:
    frame_names: tuple[str, ...]
    station_ids: tuple[str, ...]
    camera_to_world_cv: np.ndarray
    intrinsics: np.ndarray
    image_sizes_hw: np.ndarray
    world_up: np.ndarray
    characteristic_length: float

    @classmethod
    def from_frames(cls, frames, *, convention, world_up, station_key='station_index'):
        if convention not in {'opengl', 'opencv'}:
            raise ValueError('Explicit opengl/opencv camera convention is required')
        if not frames:
            raise ValueError('No cameras')
        up = np.asarray(world_up, dtype=np.float64)
        if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) == 0:
            raise ValueError('A finite nonzero world-up vector is required')
        up = up / np.linalg.norm(up)
        names, groups, poses, intrinsics, sizes = [], [], [], [], []
        for frame in frames:
            if station_key not in frame or frame[station_key] is None:
                raise ValueError('Every face must identify its physical panorama station')
            group = str(frame[station_key])
            if not group:
                raise ValueError('Empty physical station ID')
            name = str(frame['file_path'])
            if name in names:
                raise ValueError(f'Duplicate image: {name}')
            pose = np.array(frame['transform_matrix'], dtype=np.float64, copy=True)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError(f'Invalid camera transform: {name}')
            if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-9, rtol=0):
                raise ValueError('Camera transform must be affine')
            rotation = pose[:3, :3]
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6):
                raise ValueError('Camera rotation must be proper and orthonormal')
            if convention == 'opengl':
                pose[:3, :3] = rotation @ np.diag([1., -1., -1.])
            width, height = int(frame['w']), int(frame['h'])
            fx, fy, cx, cy = [float(frame[key]) for key in ('fl_x', 'fl_y', 'cx', 'cy')]
            if min(width, height, fx, fy) <= 0 or not np.isfinite([fx, fy, cx, cy]).all():
                raise ValueError('Invalid intrinsics or image size')
            names.append(name); groups.append(group); poses.append(pose)
            intrinsics.append([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
            sizes.append([height, width])
        poses = np.stack(poses)
        representative = {}
        for group, pose in zip(groups, poses):
            representative.setdefault(group, pose[:3, 3])
        centers = np.array(list(representative.values()))
        if len(centers) < 2:
            raise ValueError('At least two physical stations are needed for multiview geometry')
        scale = characteristic_length(centers)
        # Different provider IDs at one capture center are not independent
        # geometric baselines (including repeated cubes or capture dates).
        for start in range(0, len(centers), 256):
            distances = np.linalg.norm(centers[start:start + 256, None] - centers[None], axis=-1)
            local = np.arange(len(distances))
            distances[local, start + local] = np.inf
            if np.any(distances <= scale * 1e-6):
                raise ValueError('Distinct station IDs share a capture center; merge co-located captures before multiview voting')
        for group, pose in zip(groups, poses):
            if np.linalg.norm(pose[:3, 3] - representative[group]) > scale * 1e-6:
                raise ValueError('Faces sharing a physical panorama must share a camera center')
        return cls(tuple(names), tuple(groups), poses, np.array(intrinsics), np.array(sizes), up, scale)


def characteristic_length(camera_centers):
    """Median nearest distinct-station distance; transforms with scene units."""
    centers = np.asarray(camera_centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or not np.isfinite(centers).all():
        raise ValueError('Expected finite camera centers [N,3]')
    unique = np.unique(centers, axis=0)
    if len(unique) < 2:
        raise ValueError('No independent camera baseline')
    # Chunked distances avoid allocating N*N for a long road route.
    nearest = []
    for start in range(0, len(unique), 256):
        distances = np.linalg.norm(unique[start:start + 256, None] - unique[None], axis=-1)
        distances[distances == 0] = np.inf
        nearest.extend(np.min(distances, axis=1))
    return float(np.median(nearest))


def station_split(station_ids, *, holdout_fraction=.2, seed=0):
    """Deterministic group split; all faces at one location stay together."""
    groups = sorted(set(map(str, station_ids)))
    if len(groups) < 3 or not 0 < holdout_fraction < 1:
        raise ValueError('A split needs >=3 physical stations and a fraction in (0,1)')
    order = sorted(groups, key=lambda key: hashlib.sha256(f'{seed}\0{key}'.encode()).digest())
    count = min(len(groups) - 2, max(1, int(np.ceil(len(groups) * holdout_fraction))))
    return tuple(sorted(order[count:])), tuple(sorted(order[:count]))


def radial_to_camera_z(radial_distance, camera_rays):
    """Convert first-surface radial distance using each ray, without assuming unit rays."""
    distance = np.asarray(radial_distance, dtype=np.float64)
    rays = np.asarray(camera_rays, dtype=np.float64)
    if rays.shape != distance.shape + (3,):
        raise ValueError('Ray shape must be distance.shape+(3,)')
    length = np.linalg.norm(rays, axis=-1)
    if not np.isfinite(distance).all() or not np.isfinite(rays).all() or np.any(distance < 0) or np.any(length == 0):
        raise ValueError('Distances/rays must be finite and rays nonzero')
    return distance * rays[..., 2] / length
