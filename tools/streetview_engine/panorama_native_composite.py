"""Raster registration of generated panoramas onto immutable target images.

M_normalized maps target ERP centers ((x+.5)/W, (y+.5)/H, 1)
to generated ERP normalized coordinates. No estimation or per-face fitting.
A candidate is a diagnostic; geometric safety is not registration acceptance.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image

COORDINATES = "target ERP normalized pixel centers -> generated ERP normalized pixel centers"


class CompositeRejected(ValueError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def array_sha256(array):
    """SHA over contiguous array bytes; shape/dtype are bound separately."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def support_sha256(alpha):
    return array_sha256(np.ascontiguousarray(np.asarray(alpha) > 0, dtype=np.uint8))


def srgb_to_linear(rgb):
    """Decode standard sRGB values into linear-light values."""
    value = np.asarray(rgb).astype(np.float32) / 255.0
    return np.where(value <= .04045, value / 12.92, ((value + .055) / 1.055) ** 2.4)


def linear_to_srgb(value):
    """Same rounded uint8 transfer function as the existing compositor."""
    value = np.clip(value, 0., 1.)
    encoded = np.where(value <= .0031308, value * 12.92, 1.055 * value ** (1. / 2.4) - .055)
    return np.clip(np.rint(encoded * 255.), 0, 255).astype(np.uint8)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _rgb(value, name):
    value = np.asarray(value)
    _require(value.dtype == np.uint8 and value.ndim == 3 and value.shape[2] == 3,
             name + " must be HxWx3 uint8 RGB")
    _require(min(value.shape[:2]) > 0, name + " is empty")
    return value


def _alpha(value, shape):
    value = np.asarray(value)
    _require(value.shape == tuple(shape) and np.issubdtype(value.dtype, np.floating),
             "alpha must be a floating point array matching the target grid")
    _require(np.isfinite(value).all() and ((value >= 0) & (value <= 1)).all(),
             "alpha must be finite in [0,1]")
    return value


def _matrix(value):
    value = np.asarray(value, dtype=np.float64)
    _require(value.shape == (2, 3) and np.isfinite(value).all(),
             "M_normalized must be a finite 2x3 matrix")
    _require(np.linalg.det(value[:, :2]) > 0, "M must preserve orientation and be invertible")
    return value


def erp_center_grid(size_wh):
    w, h = map(int, size_wh)
    _require(w > 0 and h > 0, "Invalid ERP dimensions")
    u, v = np.meshgrid((np.arange(w, dtype=np.float64) + .5) / w,
                       (np.arange(h, dtype=np.float64) + .5) / h)
    return np.stack((u, v), axis=-1)


def _uv(value, shape):
    value = np.asarray(value, dtype=np.float64)
    _require(value.shape == (*shape, 2) and np.isfinite(value).all(),
             "Target UV must be finite HxWx2")
    _require(((value[..., 0] >= 0) & (value[..., 0] <= 1) &
              (value[..., 1] >= 0) & (value[..., 1] <= 1)).all(),
             "Target UV must lie in the original normalized ERP domain")
    return value


def _sample(array, uv):
    """Periodic U, edge-replicated V, true bilinear pixel-center sampling."""
    h, w = array.shape[:2]
    x = np.remainder(uv[..., 0], 1.0) * w - .5
    y = np.clip(uv[..., 1] * h - .5, 0, h - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    tx, ty = x - x0, y - y0
    x1, y1 = (x0 + 1) % w, np.minimum(y0 + 1, h - 1)
    x0 %= w
    result = np.zeros((*uv.shape[:-1], *array.shape[2:]), dtype=np.float64)
    for iy, ix, weight in ((y0, x0, (1-tx)*(1-ty)),
                           (y0, x1, tx*(1-ty)),
                           (y1, x0, (1-tx)*ty),
                           (y1, x1, tx*ty)):
        if array.ndim == 3:
            weight = weight[..., None]
        result += array[iy, ix] * weight
    return result


def sample_fixed_target_alpha(erp_alpha, target_uv):
    """Project an existing alpha onto rays, without transforming/dilating it."""
    alpha = np.asarray(erp_alpha)
    _alpha(alpha, alpha.shape)
    _require(alpha.ndim == 2, "ERP alpha must be 2D")
    uv = np.asarray(target_uv)
    _uv(uv, uv.shape[:-1])
    # This is original target-domain alpha sampling, never generated-domain M.
    return np.clip(_sample(alpha, uv), 0, 1)


def inspect_mapping(alpha, target_uv, M_normalized, source_size_wh, target_erp_size_wh,
                    *, seam_guard_pixels=1.0, pole_guard_pixels=1.0,
                    matrix_tolerance=1e-10, allow_topology_unverified=False):
    """Reject only actual active support that violates the declared topology.

    By default, active incompatible seams/poles are rejected. Explicit
    allow_topology_unverified=True records them as unaccepted review warnings;
    it does not alter M, alpha, or sampling and cannot override invalid source V.
    Generated U wraps periodically.
    Generated V must stay inside the full bilinear center domain on active edits.
    Exact identity with equal source/target ERP sizes also accepts in-domain
    polar half-texels through the existing edge-replicating sampler. Coordinates
    outside [0,1] and every non-identity mapping retain the stricter guard.
    """
    _require(type(allow_topology_unverified) is bool,
             "Topology review opt-in must be a boolean")
    alpha = np.asarray(alpha)
    _alpha(alpha, alpha.shape)
    _require(alpha.ndim == 2, "alpha must be 2D")
    uv = _uv(target_uv, alpha.shape)
    m = _matrix(M_normalized)
    sw, sh = map(int, source_size_wh)
    tw, th = map(int, target_erp_size_wh)
    _require(min(sw, sh, tw, th) > 0, "Invalid source/target dimensions")
    _require(np.isfinite([seam_guard_pixels, pole_guard_pixels, matrix_tolerance]).all()
             and min(seam_guard_pixels, pole_guard_pixels) >= 1 and matrix_tolerance > 0,
             "Guards must be at least one pixel and tolerance positive")
    mapped = uv @ m[:, :2].T + m[:, 2]
    _require(np.isfinite(mapped).all(), "Mapped coordinates overflow")
    active = alpha > 0
    periodic = bool(abs(m[0, 0] - 1) <= matrix_tolerance and
                    abs(m[1, 0]) <= matrix_tolerance)
    # At either pole, every longitude describes the same point. A raster
    # transform preserves both pole locations only when v'=v.
    pole_compatible = bool(abs(m[1, 0]) <= matrix_tolerance and
                           abs(m[1, 1] - 1) <= matrix_tolerance and
                           abs(m[1, 2]) <= matrix_tolerance)
    seam_zone = ((uv[..., 0] <= seam_guard_pixels / tw) |
                 (uv[..., 0] >= 1 - seam_guard_pixels / tw))
    pole_zone = ((uv[..., 1] <= pole_guard_pixels / th) |
                 (uv[..., 1] >= 1 - pole_guard_pixels / th))
    eps = np.finfo(np.float64).eps * 32
    outside_centers = active & ((mapped[..., 1] < .5 / sh - eps) |
                                (mapped[..., 1] > 1 - .5 / sh + eps))
    identity_edge_policy = bool(np.array_equal(m, np.array([[1., 0., 0.], [0., 1., 0.]]))
                                and (sw, sh) == (tw, th))
    inside_erp_domain = ((mapped[..., 1] >= 0.) & (mapped[..., 1] <= 1.))
    replicated_polar = outside_centers & inside_erp_domain & identity_edge_policy
    invalid_v = outside_centers & ~replicated_polar
    invalid_seam = active & seam_zone & (not periodic)
    invalid_pole = active & pole_zone & (not pole_compatible)
    reasons = []
    topology_warnings = []
    if invalid_v.any():
        reasons.append("active_source_v_outside_bilinear_centers")
    for mask, label in ((invalid_seam, "active_target_seam_incompatible"),
                        (invalid_pole, "active_target_pole_incompatible")):
        if mask.any():
            topology_warnings.append(label)
            if not allow_topology_unverified:
                reasons.append(label)
    report = dict(
        schema="normalized_erp_composite_geometry_v1",
        coordinate_convention=COORDINATES, M_normalized=m.tolist(),
        matrix_sha256=array_sha256(m), source_size_wh=[sw, sh],
        target_erp_size_wh=[tw, th], target_grid_size_wh=[alpha.shape[1], alpha.shape[0]],
        safe_edit_support_sha256=support_sha256(alpha),
        source_shape_hw=[sh, sw], target_shape_hw=[th, tw],
        support_sha_convention="contiguous uint8(alpha > 0), row-major, no shape prefix",
        alpha_sha256=array_sha256(alpha), alpha_dtype=str(alpha.dtype),
        active_pixels=int(active.sum()),
        globally_periodic_u=periodic, globally_pole_compatible=pole_compatible,
        seam_guard_pixels=float(seam_guard_pixels), pole_guard_pixels=float(pole_guard_pixels),
        identity_polar_edge_replication_enabled=identity_edge_policy,
        identity_polar_edge_replication_pixels=int(replicated_polar.sum()),
        identity_polar_edge_policy="exact identity and equal ERP sizes: in-domain half-texels use unchanged edge replication",
        source_v_outside_center_pixels_before_identity_policy=int(outside_centers.sum()),
        invalid_source_v_pixels=int(invalid_v.sum()),
        invalid_seam_pixels=int(invalid_seam.sum()), invalid_pole_pixels=int(invalid_pole.sum()),
        source_u_wrapped_active_pixels=int((active & ((mapped[..., 0] < 0) |
                                                     (mapped[..., 0] >= 1))).sum()),
        hard_rejected=bool(reasons), rejection_reasons=reasons,
        topology_unverified=bool(topology_warnings),
        topology_warnings=topology_warnings,
        allow_topology_unverified=allow_topology_unverified,
        registration_accepted=False, review_only=True,
        source_sampling=("true bilinear, pixel centers, periodic U; exact identity/equal ERP sizes permit in-domain polar edge replication"
                         if identity_edge_policy else "true bilinear, pixel centers, periodic U; strict active V center domain"),
        registration_validated=False, training_ready=False,
    )
    if reasons:
        raise CompositeRejected("; ".join(reasons), report)
    return mapped, report


def compose_on_target_grid(original_rgb, generated_erp_rgb, target_alpha,
                           target_uv, M_normalized, target_erp_size_wh, **safety):
    """Compose on an ERP or native face grid using one panorama-level matrix.

    For a native face, supply its original native RGB and original-ERP ray UVs.
    target_alpha is the fixed target-domain alpha (e.g. sample_fixed_target_alpha).
    This function never estimates a transform, modifies intrinsics, resizes the
    original, or samples the original RGB. All six faces must use the same M.
    """
    original = _rgb(original_rgb, "original_rgb")
    generated = _rgb(generated_erp_rgb, "generated_erp_rgb")
    alpha = _alpha(target_alpha, original.shape[:2])
    uv = _uv(target_uv, original.shape[:2])
    old_hashes = [array_sha256(v) for v in (original, generated, alpha, uv)]
    mapped, report = inspect_mapping(alpha, uv, M_normalized,
                                    (generated.shape[1], generated.shape[0]),
                                    target_erp_size_wh, **safety)
    active = alpha > 0
    result = original.copy()
    if active.any():
        # Sample only active pixels. Excluded generated pixels/coordinates are
        # not used as a reason to replace original background.
        # Preserve the existing compositor's color transfer and one generated
        # sampling pass. Alpha stays in the original target domain.
        sampled = _sample(srgb_to_linear(generated).astype(np.float64), mapped[active])
        a = alpha[active, None]
        result[active] = linear_to_srgb(
            (1 - a) * srgb_to_linear(original[active]) + a * sampled)
    preserved = int(np.count_nonzero(np.any(result[~active] != original[~active], axis=-1)))
    new_hashes = [array_sha256(v) for v in (original, generated, alpha, uv)]
    _require(old_hashes == new_hashes and preserved == 0, "Immutable source invariant failed")
    report.update(schema="normalized_erp_composite_v1", status="review_candidate_completed",
                  original_rgb_sha256=old_hashes[0], generated_rgb_sha256=old_hashes[1],
                  output_rgb_sha256=array_sha256(result), input_arrays_unchanged=True,
                  alpha_unchanged=True, alpha_transformed=False, additional_feather=False,
                  color_correction=False, original_rgb_resampled=False,
                  protected_native_changed_pixels=preserved,
                  protected_target_changed_pixels=preserved,
                  original_resolution_preserved=True,
                  generated_erp_sampling_passes=1,
                  color_space="sRGB uint8 decoded with standard sRGB transfer; generated linear float64",
                  sampling_space="linear RGB, true bilinear four-term accumulation",
                  blend_space="linear RGB under unchanged original target alpha",
                  rgb_rounding="existing linear_to_srgb transfer, round-to-nearest-even uint8")
    return result, report


def compose_erp(original_erp_rgb, generated_erp_rgb, target_erp_alpha, M_normalized, **safety):
    original = _rgb(original_erp_rgb, "original_erp_rgb")
    wh = (original.shape[1], original.shape[0])
    return compose_on_target_grid(original, generated_erp_rgb, target_erp_alpha,
                                  erp_center_grid(wh), M_normalized, wh, **safety)


def save_rgb_png_exact(rgb, destination):
    """Write a fresh RGB PNG; never overwrite an earlier result."""
    rgb = _rgb(rgb, "rgb")
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        Image.fromarray(rgb, "RGB").save(stream, format="PNG")
    with Image.open(path) as image:
        _require(image.mode == "RGB" and np.array_equal(np.asarray(image), rgb),
                 "PNG bytes did not preserve the computed RGB")
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                size_wh=[rgb.shape[1], rgb.shape[0]])




def compose_registered_erp(original_erp_rgb, generated_erp_rgb, target_erp_alpha,
                           registration_report, *, allow_unvalidated_candidate=False):
    """Bind estimator evidence to the exact alpha, source size, and matrix.

    Diagnostic candidates require explicit opt-in. Unsafe sampling cannot be
    overridden. Even accepted registration is not a training promotion.
    """
    report = registration_report
    _require(type(report.get("accepted")) is bool, "Missing registration accepted decision")
    _require(report.get("sampling_safe") is True, "Registration reports unsafe sampling")
    _require(report["accepted"] or allow_unvalidated_candidate is True,
             "Unvalidated candidate requires explicit review-only opt-in")
    original = _rgb(original_erp_rgb, "original_erp_rgb")
    generated = _rgb(generated_erp_rgb, "generated_erp_rgb")
    alpha = _alpha(target_erp_alpha, original.shape[:2])
    _require(report.get("safe_edit_support_sha256") == support_sha256(alpha),
             "Registration support mask binding differs")
    _require(report.get("source_shape_hw") == list(generated.shape[:2]) and
             report.get("target_shape_hw") == list(original.shape[:2]),
             "Registration source/target shape binding differs")
    m = _matrix(report.get("M_normalized"))
    rgb, result = compose_erp(original, generated, alpha, m)
    result.update(registration_accepted=report["accepted"],
                  registration_validated=report["accepted"],
                  review_only=True, training_ready=False,
                  estimator_binding_verified=True)
    return rgb, result
