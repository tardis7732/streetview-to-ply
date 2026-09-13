"""Project unmodified full-ERP boolean masks into calibrated native cameras.

ERP longitude wraps left/right; latitude uses the same pole-edge replication as
the existing native compositor. Camera rays sample pixel centers (x+.5,y+.5).
The default excludes every native center receiving a positive bilinear weight
from a true ERP texel. This is conservative for those four contributors, not a
claim of continuous coverage of the entire native-pixel footprint.
"""
from __future__ import annotations

import numpy as np

from .imaging import FACES
from .panorama_cube import validate_cameras
from .panorama_native_mapping import native_uv_alpha

SAMPLING = ('positive_bilinear', 'nearest')


def projection_policy(sampling='positive_bilinear'):
    if sampling not in SAMPLING:
        raise ValueError('Unsupported ERP boolean mask sampling')
    return dict(sampling=sampling, pixel_center_offset=.5,
        erp_u='atan2(station_x,station_z)/(2*pi)+.5; periodic left/right',
        erp_v='asin(station_y)/pi+.5; replicated pole edges',
        positive_weight_contributors_only=sampling=='positive_bilinear',
        entire_pixel_footprint_conservative=False, input_mask_changed=False,
        geometry='original calibrated camera_to_station_cv; no generated-image registration')


def project_erp_mask(mask, camera, *, sampling='positive_bilinear', chunk_rows=32):
    """Return a native H×W bool mask, aligned to the untouched native RGB grid.

``positive_bilinear`` is exactly ``native_uv_alpha(camera, mask)[1] > 0``.
``nearest`` selects the ERP texel containing the projected center, using floor
in pixel-edge coordinates. Neither mode changes or dilates the source mask.
No panorama-to-generated-image Transform/Scale enters this geometry.
"""
    projection_policy(sampling)
    mask = np.asarray(mask)
    if mask.dtype != np.bool_ or mask.ndim != 2 or mask.shape[0] <= 0 or mask.shape[1] != 2*mask.shape[0]:
        raise ValueError('ERP mask must be a nonempty 2:1 boolean array')
    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise ValueError('chunk_rows must be a positive integer')
    if not isinstance(camera, dict) or any(type(camera.get(key)) is not int or camera[key] <= 0 for key in ('w','h')):
        raise ValueError('Native camera dimensions must be positive integers')
    # Reuse the compositor's validated center-ray projection, wrapping and poles.
    # Boolean input is deliberately converted only for the existing interpolation.
    uv, alpha = native_uv_alpha(camera, mask.astype(np.float64), chunk_rows=chunk_rows)
    if sampling == 'positive_bilinear':
        return alpha > 0
    height,width = mask.shape
    x = np.floor(uv[...,0]*width).astype(np.int64) % width
    y = np.clip(np.floor(uv[...,1]*height).astype(np.int64), 0, height-1)
    return mask[y,x].copy()


def project_erp_mask_to_cubes(mask, cameras, *, sampling='positive_bilinear', chunk_rows=32):
    """Project one station ERP mask to all six validated 90-degree cube faces."""
    cameras = list(cameras)
    if any(not isinstance(camera,dict) for camera in cameras) or {camera.get('face') for camera in cameras} != set(FACES):
        raise ValueError('All six named native cube cameras are required')
    validate_cameras(cameras)
    by_face = {camera['face']:camera for camera in cameras}
    return {face:project_erp_mask(mask,by_face[face],sampling=sampling,chunk_rows=chunk_rows) for face in FACES}
