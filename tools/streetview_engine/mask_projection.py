"""Binary supervision support for an explicitly specified RGB remap or resize.

This module does not select faces, project rays, resample RGB, or infer masks.
The caller supplies the *same pixel-center maps and border policy as RGB*.
True always means a source pixel is allowed to supervise the reconstruction.
Existing dataset generation and training callers are intentionally unchanged.
"""
from __future__ import annotations

import numpy as np
from numbers import Integral


def remap_valid_mask(mask, map_x, map_y, *, border_mode="constant"):
    """Return bool destination support; every contributing source must be valid.

    ``map_x``/``map_y`` use OpenCV pixel-center coordinates: integer (0,0)
    selects the first pixel center. They are cast to float32 just as the RGB
    ``cv2.remap(..., INTER_LINEAR)`` call must be. ``mask`` must be a 2D bool
    array; convert a 0/255 file with ``raw == 255``, not an implicit threshold.

    ``border_mode='replicate'`` clamps contributors to the edge, matching cube
    RGB assembly. ``'constant'`` treats every out-of-bounds contributor as
    unsupported, even if RGB uses a nonzero constant background. Neither mode
    wraps a cube face or an ERP seam. For periodic ERP resampling, explicitly
    pad RGB and mask horizontally and offset their maps identically before
    calling this helper; the vertical poles must retain the chosen RGB policy.

    Require all nonzero *continuous* bilinear contributors of the float32
    maps. OpenCV's fixed-point convertMaps/INTER_LINEAR path rounds fractions
    to a 1/32-pixel table; some float-map paths (observed in OpenCV 5.0) retain
    finer fractions. Continuous support conservatively covers either path,
    including a tiny-weight neighbor that the table rounds away. Do not
    replace this with uint8 mask interpolation, a loose coverage threshold,
    or float coverage == 1: rounding can hide a small invalid contribution.

    All inputs must be finite and within OpenCV's signed-short remap limits.
    Geometric coverage/face ownership remains the caller's responsibility:
    replicated edge pixels alone are not proof a target ray is observed.
    """
    source = np.asarray(mask)
    if source.dtype.kind != "b" or source.ndim != 2 or not source.size:
        raise ValueError("mask must be a nonempty 2D bool array; True means valid")
    if any(size >= 32767 for size in source.shape):
        raise ValueError("Source dimensions exceed OpenCV remap limits")
    if border_mode not in ("constant", "replicate"):
        raise ValueError("border_mode must be constant or replicate; no implicit seam wrapping")
    arrays = [np.asarray(value) for value in (map_x, map_y)]
    if any(a.dtype.kind not in "fiu" or a.ndim != 2 or not a.size for a in arrays):
        raise ValueError("Maps must be nonempty 2D real numeric arrays")
    if arrays[0].shape != arrays[1].shape or any(size >= 32767 for size in arrays[0].shape):
        raise ValueError("Map shapes must match and fit OpenCV remap limits")
    if any(not np.isfinite(a).all() or np.any(np.abs(a.astype(np.float64)) > 32760) for a in arrays):
        raise ValueError("Maps must be finite and within signed-short remap coordinates")
    x, y = [np.ascontiguousarray(a, dtype=np.float32) for a in arrays]
    accepted = np.ones(x.shape, dtype=bool)
    lo_x = np.floor(x).astype(np.int64)
    lo_y = np.floor(y).astype(np.int64)
    # Integer comparisons, not multiplied floating weights, preserve arbitrarily
    # small nonzero contributors near an integer (including negative subnormals).
    fractional_x = x.astype(np.float64) != lo_x
    fractional_y = y.astype(np.float64) != lo_y
    height, width = source.shape
    for step_x, step_y, active in (
            (0, 0, np.ones(x.shape, bool)), (1, 0, fractional_x),
            (0, 1, fractional_y), (1, 1, fractional_x & fractional_y)):
        sx, sy = lo_x + step_x, lo_y + step_y
        supported = source[np.clip(sy, 0, height-1), np.clip(sx, 0, width-1)]
        if border_mode == "constant":
            supported = supported & (sx >= 0) & (sx < width) & (sy >= 0) & (sy < height)
        accepted &= ~active | supported
    return accepted


def resize_lanczos_valid_mask(mask, output_shape):
    """Conservative support for Pillow's default LANCZOS RGB resize.

    ``output_shape`` is (height, width), unlike Pillow's size argument. True
    source pixels provide valid supervision/evidence. Require every source
    pixel in the separable Lanczos-3 support envelope to be valid, including
    negative lobes and zero-valued taps inside the envelope. Downsampling
    widens the source-space radius by source_size/output_size. Source edges
    truncate the support as Pillow does; they are not invalid padding.

    Exact identity dimensions select only the original pixel on that axis;
    a same-size resize returns an independent copy of the original mask.
    This matches ``Image.resize(..., Resampling.LANCZOS)`` with the full image
    box and default ``reducing_gap=None``. A crop, prior reduction pass, or
    different interpolation kernel requires its own support policy.

    Lanczos has signed coefficients: resizing a float or uint8 mask and
    testing coverage is not a valid all-contributor check. An integral image
    of excluded pixels tests the complete support rectangle without relying
    on coefficient cancellation, quantization or a color-dependent threshold.
    """
    source = np.asarray(mask)
    if source.dtype.kind != 'b' or source.ndim != 2 or not source.size:
        raise ValueError('mask must be a nonempty 2D bool array; True means valid')
    if isinstance(output_shape, np.ndarray) and output_shape.ndim != 1:
        raise ValueError('output_shape must be positive integer (height, width)')
    if (not isinstance(output_shape, (tuple, list, np.ndarray)) or len(output_shape) != 2
            or any(isinstance(n, bool) or not isinstance(n, Integral) or n < 1 for n in output_shape)):
        raise ValueError('output_shape must be positive integer (height, width)')
    shape = tuple(map(int, output_shape))
    if source.shape == shape:
        return source.copy()
    if source.all():
        return np.ones(shape, dtype=bool)
    if not source.any():
        return np.zeros(shape, dtype=bool)

    def bounds(source_size, output_size):
        if source_size == output_size:
            lower = np.arange(output_size, dtype=np.int64)
            return lower, lower+1
        scale = source_size/output_size
        center = (np.arange(output_size, dtype=np.float64)+.5)*scale
        support = 3*max(1., scale)
        lower = np.maximum(0, np.ceil(center-support-.5).astype(np.int64))
        upper = np.minimum(source_size, np.floor(center+support-.5).astype(np.int64)+1)
        return lower, upper

    y0, y1 = bounds(source.shape[0], shape[0])
    x0, x1 = bounds(source.shape[1], shape[1])
    integral = np.pad((~source).astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    excluded = (integral[y1[:, None], x1[None, :]]-integral[y0[:, None], x1[None, :]]
                -integral[y1[:, None], x0[None, :]]+integral[y0[:, None], x0[None, :]])
    return excluded == 0
