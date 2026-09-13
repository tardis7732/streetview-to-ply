"""Diagnostic native semantic labels at projected Gaussian MEANS only.

This deliberately ignores covariance and visibility. A sky-labelled centre is
not proof that the full ellipsoid lies against sky, and a non-sky label is not
measured surface support: a foreground object may occlude the Gaussian. The
caller must validate a multi-station deletion policy separately. This module
neither selects/deletes Gaussians nor invents sky depth, height or ground planes.

K uses pixel-edge coordinates: native pixel (x,y) has centre (x+.5,y+.5),
and a projected continuous coordinate (u,v) samples (floor(u),floor(v)).
No resize, wrap, rounding to integer-centred pixels, or half-pixel shift occurs.
"""
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .dense_front import _camera_check


@dataclass(frozen=True)
class SemanticCenterConfig:
    erosion_radius_px: int = 2

    def __post_init__(self):
        r = self.erosion_radius_px
        if not isinstance(r, int) or isinstance(r, bool) or r < 0:
            raise ValueError('Erosion radius must be a nonnegative integer in native pixels')


def _versions(tensors):
    try:
        return tuple(t._version for t in tensors)
    except RuntimeError as exc:
        raise ValueError('Prepared masks require tracked tensor versions') from exc


@dataclass(frozen=True)
class PreparedSemanticCenter:
    sky: torch.Tensor
    non_sky: torch.Tensor
    config: SemanticCenterConfig
    versions: tuple

    def validate(self, device):
        if self.sky.device != device or self.non_sky.device != device:
            raise ValueError('Prepared masks and means must share a device')
        if _versions((self.sky, self.non_sky)) != self.versions:
            raise ValueError('Prepared semantic masks changed in place')


@torch.no_grad()
def prepare_semantic_center(sky, semantic_valid=None, *, config=SemanticCenterConfig()):
    """Snapshot and erode sky AND valid non-sky separately on the native grid.

    ``semantic_valid`` should exclude edited/invalid pixels, not exclude sky as
    a dense-depth validity map often does. Outside-image pixels and invalid/edit
    pixels are unknown. Both class boundaries are eroded; erased sky does not
    become non-sky. A score must already have been converted to a boolean label
    using the source model's documented threshold; no probability is assumed.
    """
    if not isinstance(config, SemanticCenterConfig):
        raise ValueError('Expected SemanticCenterConfig')
    if not isinstance(sky, torch.Tensor) or sky.dtype != torch.bool or sky.ndim != 2 or min(sky.shape) < 1:
        raise ValueError('Native sky must be a nonempty boolean torch [H,W]')
    if semantic_valid is None:
        semantic_valid = torch.ones_like(sky)
    if not isinstance(semantic_valid, torch.Tensor) or semantic_valid.shape != sky.shape or semantic_valid.dtype != torch.bool:
        raise ValueError('Semantic validity must match the native boolean grid')
    semantic_valid = semantic_valid.to(device=sky.device)
    radius = config.erosion_radius_px

    def erode(mask):
        if not radius:
            return mask.clone()
        padded = F.pad(mask.float()[None, None], (radius, radius, radius, radius), value=0.)
        return -F.max_pool2d(-padded, 2*radius+1, stride=1)[0, 0] > .5

    positive = erode(sky & semantic_valid)
    negative = erode(~sky & semantic_valid)
    return PreparedSemanticCenter(positive, negative, config, _versions((positive, negative)))


@torch.no_grad()
def observe_semantic_center(means, view, K, prepared, *, chunk_rows=65536):
    """Return per-row centre labels using bounded float64 projection batches.

    ``view`` is a rigid world-to-camera matrix (3x4 or 4x4), with positive camera
    Z forward, and K must correspond to the prepared native mask resolution.
    Invalid, behind-camera, off-image and unknown-mask rows never vote. There is
    no world-unit near-plane threshold, keeping positive-scale invariance.
    Output flags use O(N) boolean memory and intermediate projection O(chunk).
    Covariance is intentionally not consumed, so image-edge-crossing ellipsoids
    can supply centre evidence; this weaker evidence is labelled explicitly.
    """
    if not isinstance(chunk_rows, int) or isinstance(chunk_rows, bool) or chunk_rows < 1:
        raise ValueError('chunk_rows must be a positive integer')
    if not isinstance(means, torch.Tensor) or means.ndim != 2 or means.shape[1] != 3 or not means.is_floating_point():
        raise ValueError('Expected floating torch means [N,3]')
    if not isinstance(prepared, PreparedSemanticCenter):
        raise ValueError('Use prepare_semantic_center for native masks')
    if not isinstance(view, torch.Tensor) or not isinstance(K, torch.Tensor):
        raise ValueError('Camera matrices must be torch tensors')
    device = means.device
    prepared.validate(device)
    view, K = _camera_check(view, K, device)
    h, w = prepared.sky.shape
    n = len(means)
    flags = {key: torch.zeros(n, dtype=torch.bool, device=device) for key in
             ('sky_view', 'non_sky_view', 'known_view', 'front_view', 'in_image')}
    max_batch = 0
    for first in range(0, n, chunk_rows):
        last = min(n, first+chunk_rows)
        max_batch = max(max_batch, last-first)
        source = means[first:last].to(dtype=torch.float64)
        camera = source @ view[:, :3].T + view[:, 3]
        front = torch.isfinite(camera).all(1) & (camera[:, 2] > 0)
        flags['front_view'][first:last] = front
        safe_z = torch.where(front, camera[:, 2], torch.ones_like(camera[:, 2]))
        uv = camera[:, :2] / safe_z[:, None]
        uv = uv * torch.stack((K[0, 0], K[1, 1])) + K[:2, 2]
        inside = front & torch.isfinite(uv).all(1)
        inside &= (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        flags['in_image'][first:last] = inside
        selected = inside.nonzero().flatten()
        if not len(selected):
            continue
        xy = uv[selected].floor().to(dtype=torch.int64)
        sky = prepared.sky[xy[:, 1], xy[:, 0]]
        non_sky = prepared.non_sky[xy[:, 1], xy[:, 0]]
        index = first + selected
        flags['sky_view'][index] = sky
        flags['non_sky_view'][index] = non_sky
        flags['known_view'][index] = sky | non_sky
    flags['unknown_view'] = ~flags['known_view']
    return dict(**flags, diagnostics={
        'status': 'diagnostic_center_semantics_not_footprint_or_visibility_truth',
        'rows': n, 'native_width': w, 'native_height': h, 'max_projection_batch': max_batch,
        'policy': asdict(prepared.config),
        'counts': {key: int(value.sum().item()) for key, value in flags.items()},
    })


def _boolean_vector(value, reference=None):
    if not isinstance(value, torch.Tensor) or value.dtype != torch.bool or value.ndim != 1:
        raise ValueError('Expected one-dimensional boolean row flags')
    if reference is not None and (value.shape != reference.shape or value.device != reference.device):
        raise ValueError('Row evidence shape/device mismatch')
    return value


def merge_semantic_center_station_views(view_evidence):
    """OR cube views: one physical station, at most one vote PER label.

    If different faces label the same row differently, keep both flags as a
    station conflict. No class wins silently and no face becomes a new station.
    """
    entries = list(view_evidence)
    if not entries:
        raise ValueError('At least one face observation required')
    first = _boolean_vector(entries[0]['sky_view'])
    sky = torch.zeros_like(first)
    non_sky = torch.zeros_like(first)
    for entry in entries:
        sky |= _boolean_vector(entry['sky_view'], first)
        non_sky |= _boolean_vector(entry['non_sky_view'], first)
    return dict(sky=sky, non_sky=non_sky, known=sky | non_sky, conflict=sky & non_sky)


def count_semantic_center_stations(station_evidence):
    """Sum already grouped physical-station flags in int32; no deletion policy."""
    if not hasattr(station_evidence, 'items') or not station_evidence:
        raise ValueError('Expected physical station ID to merged evidence mapping')
    items = list(station_evidence.items())
    ids = tuple(str(key) for key, _ in items)
    if len(set(ids)) != len(ids):
        raise ValueError('Physical station IDs collide as strings')
    if len(ids) > torch.iinfo(torch.int32).max:
        raise ValueError('Station count exceeds int32 capacity')
    first = _boolean_vector(items[0][1]['sky'])
    counts = {key: torch.zeros_like(first, dtype=torch.int32) for key in
              ('sky_station_count', 'non_sky_station_count', 'known_station_count', 'conflict_station_count')}
    for _, entry in items:
        sky = _boolean_vector(entry['sky'], first)
        non_sky = _boolean_vector(entry['non_sky'], first)
        counts['sky_station_count'] += sky
        counts['non_sky_station_count'] += non_sky
        counts['known_station_count'] += sky | non_sky
        counts['conflict_station_count'] += sky & non_sky
    return dict(**counts, station_ids=ids,
                diagnostics={'physical_stations': len(ids),
                             'status': 'diagnostic_center_semantics_no_selection_policy'})
