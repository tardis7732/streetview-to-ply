"""Resolution-preserving spherical mask compositing of an existing generated ERP.

Only alpha support changes RGB. Object/shadow cores are never faded back into
the original; feathering expands outward by angular distance on the sphere.
No inference, local alignment, global image blur, crop, or implicit resize.
"""
from dataclasses import asdict, dataclass
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class FeatherSettings:
    feather_pixels: float | None = None
    feather_ratio: float = 4 / 2048
    dilation_pixels: float | None = None
    dilation_ratio: float = 2 / 2048
    blend_space: str = "linear"
    generated_resampling: str = "reject"
    chunk_rows: int = 32

    def __post_init__(self):
        for name in ("feather_ratio", "dilation_ratio"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= .25:
                raise ValueError(name + " must be a finite width ratio in [0,.25]")
        for name in ("feather_pixels", "dilation_pixels"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError(name + " must be finite and nonnegative or None")
        if self.blend_space not in ("linear", "srgb"):
            raise ValueError("blend_space must be linear or srgb")
        if self.generated_resampling not in ("reject", "spherical_bilinear"):
            raise ValueError("Generated resampling must be explicitly reject or spherical_bilinear")
        if type(self.chunk_rows) is not int or not 1 <= self.chunk_rows <= 1024:
            raise ValueError("chunk_rows must be an integer in 1..1024")

    def pixels(self, width):
        feather = self.feather_pixels if self.feather_pixels is not None else self.feather_ratio * width
        dilation = self.dilation_pixels if self.dilation_pixels is not None else self.dilation_ratio * width
        if feather + dilation > width / 4:
            raise ValueError("Total outward support must not exceed a 90-degree spherical radius")
        return float(feather), float(dilation)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def binary(array, shape=None):
    array = np.asarray(array)
    if array.ndim != 2 or (shape is not None and array.shape != shape):
        raise ValueError("Mask must have the exact original ERP dimensions")
    if array.dtype == np.bool_:
        return array
    if not np.all((array == 0) | (array == 255)):
        raise ValueError("Binary masks require exact 0/255, white=edit")
    return array == 255


def validate_erp(array, *, generated=False):
    array = np.asarray(array)
    if (array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] not in (3, 4) or
            array.shape[1] != 2 * array.shape[0] or array.shape[0] < 4):
        raise ValueError("Full ERP must be uint8 RGB/RGBA with exact original 2:1 dimensions")
    if generated and array.shape[2] == 4 and not np.all(array[..., 3] == 255):
        raise ValueError("Generated RGB must be opaque; transparent/premultiplied colors are not silently used")
    return array


def sphere_points(rows, columns, height, width):
    """Canonical ERP pixel-center rays, X right / Y down / Z forward."""
    latitude = ((np.asarray(rows, np.float64) + .5) / height - .5) * np.pi
    longitude = ((np.asarray(columns, np.float64) + .5) / width - .5) * (2 * np.pi)
    coslat = np.cos(latitude)
    return np.stack((coslat * np.sin(longitude), np.sin(latitude), coslat * np.cos(longitude)), axis=-1)


def spherical_alpha(object_mask, shadow_mask=None, settings=None):
    """Distance to nearest core pixel center on S², not flat ERP raster distance.

    A nearest-neighbor query on unit directions gives exact nearest chord
    distance, monotonically equivalent to great-circle angular distance. Using
    all core centers avoids a latitude-dependent contour approximation. Queries
    are chunked and bounded to the requested angular support; longitude wrapping
    and crossing either pole follow automatically from 3D directions.
    """
    settings = settings or FeatherSettings()
    core = binary(object_mask).copy()
    h, w = core.shape
    if w != 2*h or h < 4:
        raise ValueError("Mask must cover a complete 2:1 ERP")
    if shadow_mask is not None:
        core |= binary(shadow_mask, core.shape)
    feather, dilation = settings.pixels(w)
    alpha = core.astype(np.float32)
    if not core.any() or core.all() or feather + dilation == 0:
        return alpha, core
    core_y, core_x = np.nonzero(core)
    tree = cKDTree(sphere_points(core_y, core_x, h, w), compact_nodes=True, balanced_tree=True)
    angular_radius = (feather + dilation) * 2 * np.pi / w
    chord_radius = 2 * np.sin(angular_radius / 2)
    for start in range(0, h, settings.chunk_rows):
        stop = min(start + settings.chunk_rows, h)
        yy, xx = np.nonzero(~core[start:stop])
        if not len(yy):
            continue
        yy = yy + start
        queries = sphere_points(yy, xx, h, w)
        chord, _ = tree.query(queries, k=1, distance_upper_bound=chord_radius + 1e-12, workers=1)
        near = np.isfinite(chord)
        if not near.any():
            continue
        distance_pixels = 2 * np.arcsin(np.clip(chord[near] / 2, 0, 1)) * w / (2*np.pi)
        if feather == 0:
            value = (distance_pixels <= dilation + 1e-10).astype(np.float64)
        else:
            t = np.clip((distance_pixels - dilation) / feather, 0, 1)
            value = 1 - t*t*(3 - 2*t)
        alpha[yy[near], xx[near]] = value.astype(np.float32)
    alpha[core] = 1.
    return alpha, core

