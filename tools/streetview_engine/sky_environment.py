"""Optional, photo-supported finite sky shell; no training starts on import.

The shell is a PLY rendering approximation to angular radiance, not measured
sky geometry. Its geometry must stay separate from foreground MCMC, depth
losses and foreground shape statistics. This module only builds a candidate;
it neither edits a baseline nor accepts visual quality.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation

from .imaging import FACES, cube_camera_to_station_cv, inside, sha256


@dataclass(frozen=True)
class SkyEnvironmentConfig:
    grid_size: int = 48
    minimum_physical_stations: int = 2
    maximum_static_negative_fraction: float = 0.0
    maximum_rgb_disagreement: float = 0.12
    maximum_parallax_degrees: float = 0.05
    tangent_sigma_cells: float = 0.65
    radial_to_tangent_sigma: float = 0.001
    footprint_sigma: float = 3.0
    boundary_margin_pixels: float = 1.0
    initial_opacity: float = 0.8
    sh_degree: int = 2
    far_plane_m: float | None = None
    maximum_color_values: int = 50_000_000

    def __post_init__(self):
        for key, lo, hi in [('grid_size', 2, 256), ('minimum_physical_stations', 2, 10000),
                            ('sh_degree', 0, 3), ('maximum_color_values', 1, 1_000_000_000)]:
            value = getattr(self, key)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError('Invalid integer sky setting: ' + key)
        for key in ['maximum_static_negative_fraction', 'maximum_rgb_disagreement']:
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Invalid sky fraction: ' + key)
        for key in ['maximum_parallax_degrees', 'tangent_sigma_cells', 'radial_to_tangent_sigma',
                    'footprint_sigma', 'boundary_margin_pixels', 'initial_opacity']:
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError('Sky setting must be finite positive: ' + key)
        if self.maximum_parallax_degrees >= 45 or self.initial_opacity >= 1 or self.radial_to_tangent_sigma >= 1:
            raise ValueError('Invalid sky angular/opacity/thickness setting')
        if self.far_plane_m is not None and (isinstance(self.far_plane_m, bool) or not math.isfinite(self.far_plane_m) or self.far_plane_m <= 0):
            raise ValueError('far_plane_m must be finite positive or None')


def _readonly(value, dtype=None):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SkyEnvironment:
    means: np.ndarray
    log_scales: np.ndarray
    quats: np.ndarray
    opacity_logits: np.ndarray
    sh0: np.ndarray
    shN: np.ndarray
    grid_indices: np.ndarray
    support_counts: np.ndarray
    static_negative_counts: np.ndarray
    color_disagreement: np.ndarray
    observation_row_indices: np.ndarray
    observation_frame_indices: np.ndarray
    observation_pixels_xy: np.ndarray
    _provenance_json: str

    @property
    def provenance(self):
        """A fresh JSON value; caller edits cannot change the recorded report."""
        return json.loads(self._provenance_json)


def _ids(values, label):
    if not isinstance(values, (list, tuple)) or any(not isinstance(x, str) or not x for x in values) or len(set(values)) != len(values):
        raise ValueError(label + ' must be unique nonempty string IDs')
    return set(values)


def _pose(frame):
    pose = np.asarray(frame['transform_matrix'], np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8, rtol=0):
        raise ValueError('Sky source needs a finite OpenGL camera-to-world pose')
    r = pose[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-6, rtol=0) or not np.isclose(np.linalg.det(r), 1, atol=1e-6):
        raise ValueError('Sky source rotation must be proper')
    return pose[:3, :3] @ np.diag([1., -1., -1.]), pose[:3, 3].copy()


def _load_sources(dataset, frames, heldout_station_ids):
    """Bind discovery to the saved train roster, hashes and physical split."""
    manifest_path = inside(dataset, 'dataset_manifest.json')
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    train_ids = _ids(manifest.get('training_station_ids'), 'Training physical station IDs')
    heldout_ids = _ids(manifest.get('heldout_station_ids'), 'Heldout physical station IDs')
    if train_ids & heldout_ids or not train_ids or not heldout_ids:
        raise ValueError('Sky discovery requires a nonempty disjoint physical-station split')
    if heldout_station_ids and set(heldout_station_ids) != heldout_ids:
        raise ValueError('Caller heldout roster differs from saved dataset')
    transforms_path = inside(dataset, 'transforms_train.json')
    if manifest.get('files', {}).get('transforms_train.json') != sha256(transforms_path):
        raise ValueError('Sky training roster hash differs from dataset manifest')
    transforms = json.loads(transforms_path.read_text(encoding='utf8'))
    if transforms.get('camera_convention') != 'OpenGL_c2w':
        raise ValueError('Sky builder requires explicitly declared OpenGL_c2w')
    saved = transforms.get('frames', [])
    if not saved or any(not isinstance(x, dict) for x in saved):
        raise ValueError('Sky builder requires actual saved training frames')
    if frames is None:
        frames = saved
    # Bind every field, including masks and poses; no subset or replacement
    # reference can quietly improve the apparent directional support.
    key = lambda x: str(x['file_path'])
    if json.dumps(sorted(frames, key=key), sort_keys=True, allow_nan=False) != json.dumps(sorted(saved, key=key), sort_keys=True, allow_nan=False):
        raise ValueError('Sky discovery frames differ from the complete saved training roster')
    frames = sorted(frames, key=lambda f: (str(f['station_id']), str(f.get('pano_id', '')), str(f['file_path'])))
    if len({f['file_path'] for f in frames}) != len(frames):
        raise ValueError('Duplicate sky source image')
    if {f['station_id'] for f in frames} != train_ids or any(f['station_id'] in heldout_ids for f in frames):
        raise ValueError('Sky discovery contains heldout or undeclared physical stations')
    bindings = []
    for frame in frames:
        _pose(frame)
        if type(frame['w']) is not int or type(frame['h']) is not int or min(frame['w'], frame['h']) < 2:
            raise ValueError('Invalid sky image dimensions')
        k = np.asarray([frame[n] for n in ['fl_x', 'fl_y', 'cx', 'cy']], np.float64)
        if not np.isfinite(k).all() or np.any(k[:2] <= 0):
            raise ValueError('Invalid sky image calibration')
        source_hash = frame.get('image_sha256', frame.get('source_sha256'))
        foreground_key = 'foreground_mask_path' if frame.get('foreground_mask_path') else 'sfm_mask_path'
        hashes = {}
        for path_key, expected in [('file_path', source_hash), ('mask_path', frame.get('mask_sha256')),
                                   ('sky_mask_path', frame.get('sky_mask_sha256')),
                                   (foreground_key, frame.get(foreground_key.removesuffix('_path') + '_sha256'))]:
            path = inside(dataset, frame[path_key])
            actual = sha256(path)
            if actual != expected:
                raise ValueError('Sky input hash mismatch: ' + str(frame[path_key]))
            hashes[path_key] = dict(path=frame[path_key], sha256=actual)
        bindings.append(dict(station_id=frame['station_id'], pano_id=frame.get('pano_id'),
                             files=hashes, w=frame['w'], h=frame['h'],
                             intrinsics=k.tolist(), transform_matrix=frame['transform_matrix']))
    return frames, bindings, train_ids, heldout_ids, sha256(manifest_path), sha256(transforms_path)


def _grid(size, basis):
    """Cube grid follows an actual training rig, hence rotates with the scene."""
    a = (np.arange(size, dtype=np.float64) + .5) * (2 / size) - 1
    x, y = np.meshgrid(a, a)
    local = np.stack([x.ravel(), y.ravel(), np.ones(size*size)], axis=1)
    length = np.linalg.norm(local, axis=1)
    directions, tangent_x, spacing, weights = [], [], [], []
    # Derivatives of the normalized pinhole ray give local angular spacing.
    unit = local / length[:, None]
    dx = (np.array([1., 0, 0]) - unit*unit[:, :1]) / length[:, None]
    dy = (np.array([0., 1, 0]) - unit*unit[:, 1:2]) / length[:, None]
    step = np.minimum(np.linalg.norm(dx, axis=1), np.linalg.norm(dy, axis=1)) * (2 / size)
    for face in FACES:
        rotation = basis @ cube_camera_to_station_cv(face)[:3, :3]
        directions.append(unit @ rotation.T)
        tangent_x.append((dx / np.linalg.norm(dx, axis=1)[:, None]) @ rotation.T)
        spacing.append(step)
        weights.append((2/size)**2 / length**3)
    return tuple(np.concatenate(items) for items in [directions, tangent_x, spacing, weights])


def _distance(mask):
    # Padding makes image edges unsupported rather than infinite valid space.
    return distance_transform_edt(np.pad(mask, 1, constant_values=False))[1:-1, 1:-1]


def _project(means, covariances, frame):
    rotation, center = _pose(frame)
    points = (means - center) @ rotation
    z = points[:, 2]
    good = z > 0
    pixel = np.full((len(means), 2), -1., np.float64)
    pixel[good, 0] = frame['fl_x'] * points[good, 0] / z[good] + frame['cx']
    pixel[good, 1] = frame['fl_y'] * points[good, 1] / z[good] + frame['cy']
    good &= np.isfinite(pixel).all(axis=1) & (pixel[:, 0] >= 0) & (pixel[:, 0] < frame['w']) & (pixel[:, 1] >= 0) & (pixel[:, 1] < frame['h'])
    indices = np.flatnonzero(good)
    p = points[indices]; depth = p[:, 2]
    jacobian = np.zeros((len(indices), 2, 3), np.float64)
    jacobian[:, 0, 0] = frame['fl_x']/depth
    jacobian[:, 1, 1] = frame['fl_y']/depth
    jacobian[:, 0, 2] = -frame['fl_x']*p[:, 0]/depth**2
    jacobian[:, 1, 2] = -frame['fl_y']*p[:, 1]/depth**2
    world_jacobian = jacobian @ rotation.T
    covariance = world_jacobian @ covariances[indices] @ np.swapaxes(world_jacobian, 1, 2)
    sigma = np.sqrt(np.maximum(np.linalg.eigvalsh(covariance)[:, -1], 0))
    return indices, np.floor(pixel[indices]).astype(np.int32), sigma


def build_sky_environment(dataset_root, training_frames=None, *, config=None,
                          heldout_station_ids=(), foreground_extent_m=None):
    """Build a fixed candidate from complete TRAIN photos, never heldout RGB.

    foreground_extent_m, when supplied, is a validated enclosing radius about
    the returned shell center (including foreground covariance extent). With
    None the report explicitly leaves foreground enclosure unverified. The
    valid view region is the reported camera ball; no infinite-background or
    arbitrary novel-view guarantee is implied by finite PLY serialization.
    """
    options = config or SkyEnvironmentConfig()
    if not isinstance(options, SkyEnvironmentConfig):
        raise TypeError('config must be SkyEnvironmentConfig')
    root = Path(dataset_root).resolve()
    frames, bindings, train_ids, heldout_ids, manifest_hash, roster_hash = _load_sources(root, training_frames, heldout_station_ids)
    groups = sorted(train_ids)
    count = 6*options.grid_size**2
    if count*len(groups)*3 > options.maximum_color_values:
        raise ValueError('Sky grid/group color evidence exceeds configured CPU memory bound')
    centers_by_group = {group: np.unique(np.asarray([_pose(f)[1] for f in frames if f['station_id'] == group]), axis=0) for group in groups}
    group_centers = np.asarray([centers_by_group[g].mean(axis=0) for g in groups])
    center = group_centers.mean(axis=0)
    cameras = np.concatenate(list(centers_by_group.values()))
    camera_radius = float(np.linalg.norm(cameras-center, axis=1).max())
    if not np.isfinite(camera_radius) or camera_radius <= np.finfo(float).tiny:
        raise ValueError('Sky support needs spatially distinct physical camera groups')
    # Distinct identifiers at the same reconstructed camera center do not
    # establish independent evidence, even if they name dates or cube aliases.
    for a, group in enumerate(groups):
        for other in groups[a+1:]:
            distance = np.linalg.norm(centers_by_group[group][:, None] - centers_by_group[other][None], axis=-1)
            if np.any(distance <= camera_radius*1e-6):
                raise ValueError('Distinct physical groups share a reconstructed camera center')
    first = frames[0]
    basis, _ = _pose(first)
    if 'camera_to_station_cv' in first:
        local = np.asarray(first['camera_to_station_cv'], np.float64)
        from .sfm_geometry import camera_to_station
        camera_to_station(local)
        basis = basis @ local[:3, :3].T
    directions, tangent, spacing, solid_angles = _grid(options.grid_size, basis)
    angular_sigma = spacing*options.tangent_sigma_cells
    radial_fraction = angular_sigma*options.radial_to_tangent_sigma
    inward = options.footprint_sigma*float(radial_fraction.max())
    if inward >= .5:
        raise ValueError('Sky radial covariance is too thick to represent a distant shell')
    radius = camera_radius / math.sin(math.radians(options.maximum_parallax_degrees))
    if foreground_extent_m is not None:
        if isinstance(foreground_extent_m, bool) or not math.isfinite(foreground_extent_m) or foreground_extent_m < 0:
            raise ValueError('Foreground extent must be a finite nonnegative enclosing radius')
        radius = max(radius, foreground_extent_m*1.01/(1-inward))
    means = center + radius*directions
    sigmas = radius*np.stack([angular_sigma, angular_sigma, radial_fraction], axis=1)
    rotations = np.stack([tangent, np.cross(directions, tangent), directions], axis=2)
    covariances = (rotations*sigmas[:, None, :]**2) @ np.swapaxes(rotations, 1, 2)
    # All axes contribute to the far-clip requirement; a tangent footprint is
    # not bounded by radial thickness alone.
    required_far = radius + camera_radius + options.footprint_sigma*float(sigmas.max())
    if options.far_plane_m is not None and required_far >= options.far_plane_m:
        raise ValueError('Finite sky shell/covariance does not fit configured renderer far plane')
    quantized = means.astype(np.float32).astype(np.float64)
    positional_error = np.linalg.norm(quantized-means, axis=1).max()
    if not np.isfinite(quantized).all() or positional_error/(radius-camera_radius) > math.sin(math.radians(options.maximum_parallax_degrees))*.1:
        raise ValueError('Float32 PLY positions cannot preserve the requested angular tolerance')
    colors = np.full((len(groups), count, 3), np.nan, np.float32)
    negative = np.zeros((len(groups), count), bool)
    observations = []
    for group_index, group in enumerate(groups):
        sums = np.zeros((count, 3), np.float64)
        votes = np.zeros(count, np.int32)
        for frame_index, (frame, binding) in enumerate(zip(frames, bindings)):
            if frame['station_id'] != group:
                continue
            loaded = {}
            for key, item in binding['files'].items():
                with Image.open(inside(root, item['path'])) as image:
                    if image.size != (frame['w'], frame['h']):
                        raise ValueError('Sky photo/mask size differs from calibration')
                    value = np.asarray(image.convert('RGB' if key == 'file_path' else 'L'))
                    if key != 'file_path' and np.any((value != 0) & (value != 255)):
                        raise ValueError('Sky builder needs explicit binary 0/255 masks')
                    loaded[key] = value
            photometric = loaded['mask_path'] == 255
            foreground_key = 'foreground_mask_path' if 'foreground_mask_path' in loaded else 'sfm_mask_path'
            static = (loaded[foreground_key] == 255) & photometric
            sky = (loaded['sky_mask_path'] == 255) & photometric & ~static
            sky_distance = _distance(sky)
            static_distance = distance_transform_edt(~static) if static.any() else np.full(static.shape, np.inf)
            indices, pixels, projected_sigma = _project(means, covariances, frame)
            x, y = pixels.T
            margin = options.footprint_sigma*projected_sigma + options.boundary_margin_pixels
            positive = sky_distance[y, x] > margin
            conflict = (static_distance[y, x] <= margin) & photometric[y, x]
            negative[group_index, indices[conflict]] = True
            chosen = indices[positive]
            rgb = loaded['file_path'][y[positive], x[positive]].astype(np.float64)/255
            sums[chosen] += rgb
            votes[chosen] += 1
            observations.append((frame_index, chosen, pixels[positive].copy()))
        supported = votes > 0
        colors[group_index, supported] = sums[supported]/votes[supported, None]
    positive_groups = np.isfinite(colors[:, :, 0])
    support = positive_groups.sum(axis=0)
    negatives = negative.sum(axis=0)
    eligible = support >= options.minimum_physical_stations
    center_color = np.zeros((count, 3), np.float64)
    disagreement = np.full(count, np.inf)
    if eligible.any():
        center_color[eligible] = np.nanmedian(colors[:, eligible], axis=0)
        discrepancy = np.abs(colors[:, eligible] - center_color[eligible][None])
        disagreement[eligible] = np.nanmax(discrepancy, axis=(0, 2))
    negative_fraction = negatives/np.maximum((positive_groups | negative).sum(axis=0), 1)
    accepted = eligible & (negative_fraction <= options.maximum_static_negative_fraction) & (disagreement <= options.maximum_rgb_disagreement)
    chosen = np.flatnonzero(accepted)
    lookup = np.full(count, -1, np.int32);lookup[chosen] = np.arange(len(chosen))
    observation_rows, observation_frames, observation_pixels = [], [], []
    for frame_index, indices, pixels in observations:
        keep = accepted[indices]
        if keep.any():
            observation_rows.append(lookup[indices[keep]])
            observation_frames.append(np.full(int(keep.sum()), frame_index, np.int32))
            observation_pixels.append(pixels[keep])
    joined = lambda parts, shape: np.concatenate(parts) if parts else np.empty(shape, np.int32)
    # Recheck references after processing, including a changed mask that would
    # otherwise leave a falsely bound support report.
    for binding in bindings:
        for item in binding['files'].values():
            if sha256(inside(root, item['path'])) != item['sha256']:
                raise ValueError('Sky source changed while building candidate')
    if sha256(inside(root, 'dataset_manifest.json')) != manifest_hash or sha256(inside(root, 'transforms_train.json')) != roster_hash:
        raise ValueError('Sky dataset/roster changed while building candidate')
    report = dict(schema_version=1, status='candidate' if len(chosen) else 'insufficient_evidence',
        config=asdict(options), source_dataset_manifest_sha256=manifest_hash, source_training_roster_sha256=roster_hash,
        training_station_ids=groups, heldout_station_ids=sorted(heldout_ids), source_bindings=bindings,
        grid_basis_camera=first['file_path'], grid_basis_world=basis.tolist(), shell_center=center.tolist(),
        camera_enclosing_radius_m=camera_radius, shell_radius_m=radius, required_renderer_far_plane_m=required_far,
        valid_view_region=dict(center=center.tolist(), radius_m=camera_radius, description='Camera enclosing ball; larger novel-view displacements require a new angular-error bound'),
        maximum_parallax_degrees_bound=math.degrees(math.asin(camera_radius/radius)),
        float32_position_angular_error_bound_degrees=math.degrees(math.asin(min(1., positional_error/(radius-camera_radius)))),
        foreground_enclosure='caller_validated_extent' if foreground_extent_m is not None else 'UNVERIFIED: only camera enclosure was supplied',
        foreground_extent_m=foreground_extent_m, total_grid_cells=count, accepted_cells=len(chosen),
        insufficient_station_support_cells=int((~eligible).sum()),
        static_conflict_cells=int((eligible & (negative_fraction > options.maximum_static_negative_fraction)).sum()),
        color_disagreement_cells=int((eligible & (disagreement > options.maximum_rgb_disagreement)).sum()),
        supported_cell_solid_angle_fraction=float(solid_angles[accepted].sum()/solid_angles.sum()),
        colors='Mean valid source pixels within each physical station, then median across stations; max channel deviation from station median bounds disagreement',
        static_policy='Projected covariance footprint conflicts veto by physical station. An occluding building can cause a conservative veto; this is not evidence of measured sky depth.',
        support_policy='Only complete TRAIN roster, high-confidence sky core and photometric-valid footprint; no heldout RGB/masks, extrapolation or unsupported-cell fill',
        component='shared_angular_sky_fixed_geometry', geometry_is_measured=False, quality_accepted=False,
        integration='Keep separate from foreground MCMC/depth/scale regularization; concatenate for full alpha*T RGB rendering and PLY export; evaluate independent frozen foreground and sky views',
        limitations=['Finite PLY approximates angular radiance within a bounded view region.',
            'Semantic agreement and RGB consistency do not establish true sky depth or cross-location quality.',
            'Unsupported directions remain absent; cell support is not rendered alpha coverage.',
            'Foreground holes, occlusion, clouds and appearance changes require actual candidate/control rendering.'])
    quats = Rotation.from_matrix(rotations[chosen]).as_quat()[:, [3, 0, 1, 2]] if len(chosen) else np.empty((0, 4))
    return SkyEnvironment(
        means=_readonly(means[chosen]), log_scales=_readonly(np.log(sigmas[chosen])), quats=_readonly(quats),
        opacity_logits=_readonly(np.full(len(chosen), math.log(options.initial_opacity/(1-options.initial_opacity)))),
        sh0=_readonly(((center_color[chosen]-.5)/.28209479177387814)[:, None, :]),
        shN=_readonly(np.zeros((len(chosen), (options.sh_degree+1)**2-1, 3))),
        grid_indices=_readonly(chosen, np.int64), support_counts=_readonly(support[chosen], np.int32),
        static_negative_counts=_readonly(negatives[chosen], np.int32), color_disagreement=_readonly(disagreement[chosen]),
        observation_row_indices=_readonly(joined(observation_rows, (0,)), np.int32),
        observation_frame_indices=_readonly(joined(observation_frames, (0,)), np.int32),
        observation_pixels_xy=_readonly(joined(observation_pixels, (0, 2)), np.int32),
        _provenance_json=json.dumps(report, sort_keys=True, allow_nan=False))
