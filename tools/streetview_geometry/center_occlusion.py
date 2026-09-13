"""Diagnostic foreground-occlusion abstention for raw non-sky CENTRE votes.

This is neither a sky detector nor a free-space/surface certificate. The caller
may clear a raw non-sky vote when ``occluded_view`` is true; raw sky votes must
remain unchanged. Invalid depth, incomplete neighbourhoods and failed camera
projections retain raw non-sky evidence. There is no Gaussian deletion here.

The input threshold is already in camera-Z units (for example a supplied depth
upper bound). Its interpretation and calibration belong to the caller. We take
the MAXIMUM over all nine pixels in a 3x3 neighbourhood, only when ALL nine are
valid, finite and positive. Thus a nearby discontinuity does not use the nearer
pixel alone to assert that a centre is hidden behind a foreground surface.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .dense_front import _camera_check


def _versions(tensors):
    try:
        return tuple(t._version for t in tensors)
    except RuntimeError as exc:
        raise ValueError('Prepared depth requires tracked tensor versions') from exc


@dataclass(frozen=True)
class PreparedCenterOcclusion:
    threshold_max_z: torch.Tensor
    neighbourhood_valid: torch.Tensor
    versions: tuple

    def validate(self, device):
        if self.threshold_max_z.device != device or self.neighbourhood_valid.device != device:
            raise ValueError('Prepared depth and means must share a device')
        if _versions((self.threshold_max_z, self.neighbourhood_valid)) != self.versions:
            raise ValueError('Prepared depth changed in place')


@torch.no_grad()
def prepare_center_occlusion(threshold_z, valid):
    """Snapshot a native depth threshold; strict 3x3 ALL-valid / MAX filter.

    No resampling occurs. An invalid pixel or the outside-image boundary makes
    every touching neighbourhood unknown. Nonpositive/nonfinite thresholds are
    invalid even when the supplied validity mask says otherwise.
    """
    if not isinstance(threshold_z, torch.Tensor) or threshold_z.ndim != 2 or not threshold_z.is_floating_point() or min(threshold_z.shape) < 1:
        raise ValueError('Expected nonempty floating threshold_z [H,W]')
    if not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool or valid.shape != threshold_z.shape:
        raise ValueError('Validity must be a matching boolean native depth grid')
    threshold = threshold_z.to(dtype=torch.float64)
    finite = torch.isfinite(threshold) & (threshold > 0) & valid.to(threshold.device)
    safe = torch.where(finite, threshold, torch.zeros_like(threshold))
    padded_valid = F.pad(finite.float()[None, None], (1,1,1,1), value=0.)
    all_valid = -F.max_pool2d(-padded_valid, 3, stride=1)[0,0] > .5
    maximum = F.max_pool2d(safe[None, None], 3, stride=1, padding=1)[0,0]
    maximum = torch.where(all_valid, maximum, torch.full_like(maximum, torch.nan))
    return PreparedCenterOcclusion(maximum, all_valid, _versions((maximum, all_valid)))


@torch.no_grad()
def observe_center_occlusion(means, view, K_depth, prepared, *, chunk_rows=65536):
    """Mark centre hidden only when camera Z is strictly above known threshold.

    K_depth is calibrated to the DEPTH grid, which may differ from the semantic
    mask's native resolution. It uses edge coordinates: pixel centres are
    (x+.5,y+.5), so continuous (u,v) samples (floor(u),floor(v)). ``view`` is a
    rigid world-to-camera matrix with positive camera Z forward. Means, camera
    translations and threshold_z must use consistent units. No world-scale
    near-plane epsilon, height plane, opacity or covariance is used.

    Apply only as ``retained_non_sky = raw_non_sky & ~occluded_view``. Neither
    the raw flag arrays nor Gaussian parameters are consumed or modified here.
    Occlusion is a diagnostic depth-model hypothesis, not measured truth.
    """
    if not isinstance(chunk_rows, int) or isinstance(chunk_rows, bool) or chunk_rows < 1:
        raise ValueError('chunk_rows must be a positive integer')
    if not isinstance(means, torch.Tensor) or means.ndim != 2 or means.shape[1] != 3 or not means.is_floating_point():
        raise ValueError('Expected floating torch means [N,3]')
    if not isinstance(prepared, PreparedCenterOcclusion):
        raise ValueError('Use prepare_center_occlusion for depth thresholds')
    if not isinstance(view, torch.Tensor) or not isinstance(K_depth, torch.Tensor):
        raise ValueError('Camera matrices must be torch tensors')
    device = means.device
    prepared.validate(device)
    view, K = _camera_check(view, K_depth, device)
    h, w = prepared.threshold_max_z.shape
    n = len(means)
    flags = {key: torch.zeros(n, dtype=torch.bool, device=device) for key in
             ('occluded_view', 'threshold_known_view', 'front_view', 'in_image')}
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
        uv = uv * torch.stack((K[0,0], K[1,1])) + K[:2,2]
        inside = front & torch.isfinite(uv).all(1)
        inside &= (uv[:,0] >= 0) & (uv[:,0] < w) & (uv[:,1] >= 0) & (uv[:,1] < h)
        flags['in_image'][first:last] = inside
        selected = inside.nonzero().flatten()
        if not len(selected):
            continue
        xy = uv[selected].floor().to(dtype=torch.int64)
        known = prepared.neighbourhood_valid[xy[:,1], xy[:,0]]
        threshold = prepared.threshold_max_z[xy[:,1], xy[:,0]]
        index = first+selected
        flags['threshold_known_view'][index] = known
        flags['occluded_view'][index] = known & (camera[selected,2] > threshold)
    return dict(**flags, diagnostics={
        'status': 'diagnostic_center_occlusion_non_sky_abstention_only',
        'rows': n, 'depth_width': w, 'depth_height': h, 'max_projection_batch': max_batch,
        'threshold_window': '3x3 maximum with all nine finite-positive-valid',
        'unknown_action': 'retain raw non-sky; do not create sky votes',
        'counts': {key: int(value.sum().item()) for key, value in flags.items()},
    })
