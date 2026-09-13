"""Reusable differentiable geometry losses with explicit observation masks.

Torch is the only import dependency; gsplat is loaded only by the render helper.
All scales are ACTIVATED standard deviations in metres; quaternions are wxyz.
Depth is positive camera-Z in metres, not the radial distance of an ERP map.
This optional module requires Torch. Only render_depth_moments imports gsplat;
CPU math and an injected test rasterizer do not need CUDA or scene assets.
Targets, confidence, plane associations and acceptance are supplied by callers.
These losses do not establish that a depth observation or plane is correct.

Depth moments describe Gaussian CENTER depths under the rasterizer's compositing
weights. They constrain layer spread, but do not represent a true ray integral
through each ellipsoid. The independent ground covariance loss constrains tails.
"""
from __future__ import annotations

from typing import Callable

try:
    import torch
    from torch import Tensor
except ImportError as exc:
    raise ImportError("streetview_geometry.losses requires the optional Torch dependency") from exc


def _weighted_mean(values: Tensor, valid: Tensor, confidence: Tensor | None) -> Tensor:
    """Fixed-mask mean; an empty mask gives differentiable zero.

    Shapes must agree, preventing accidental [H,W] versus [H,W,1] broadcasts.
    Confidence belongs to offline observations, not Gaussian opacity.
    """
    if values.shape != valid.shape:
        raise ValueError(f"values/mask shapes differ: {values.shape}, {valid.shape}")
    weights = valid.to(dtype=values.dtype)
    if confidence is not None:
        if confidence.shape != values.shape:
            raise ValueError("confidence must have the same shape as values")
        weights = weights * torch.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)
    safe_values = torch.where(valid & (weights > 0), values, torch.zeros_like(values))
    return (safe_values * weights).sum() / weights.sum().clamp_min(1e-12)


def camera_z(means: Tensor, viewmat: Tensor) -> Tensor:
    """One world-to-camera [4,4] matrix and [N,3] means -> [N] camera-Z."""
    if means.ndim != 2 or means.shape[-1] != 3 or viewmat.shape != (4, 4):
        raise ValueError("expected means [N,3] and viewmat [4,4]")
    return means @ viewmat[2, :3] + viewmat[2, 3]


def render_depth_moments(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    viewmat: Tensor,
    K: Tensor,
    width: int,
    height: int,
    *,
    rasterizer: Callable | None = None,
    **rasterizer_kwargs,
) -> dict[str, Tensor | dict]:
    """Render raw M1=sum(w*z), M2=sum(w*z*z), A=sum(w), each [H,W].

    Supply only finite-scene Gaussians: exclude the separate sky/background.
    Geometry and appearance passes must use matching culling/filter options.
    This pass uses sh_degree=None, render_mode='RGB', and zero background.
    Do not pass SH coefficients: ordinary N-D features carry the depth moments.
    No detach is applied to z, so gradients flow through both features and
    projection/compositing to means, covariance, and opacity. Renderer metadata
    is returned for inspection; do not use its gradients for RGB densification.
    """
    if K.shape != (3, 3):
        raise ValueError("K must be [3,3]")
    reserved = {"colors", "sh_degree", "render_mode", "backgrounds", "viewmats", "Ks"}
    if reserved.intersection(rasterizer_kwargs):
        raise ValueError(f"reserved rasterizer options: {reserved.intersection(rasterizer_kwargs)}")
    if rasterizer is None:
        from gsplat import rasterization
        rasterizer = rasterization
    z = camera_z(means, viewmat)
    # Three channels avoid special internal padding of two-channel feature
    # passes in some gsplat packed paths. The third feature is a unit mass.
    features = torch.stack((z, z.square(), torch.ones_like(z)), dim=-1)
    rendered, alpha, meta = rasterizer(
        means=means, quats=quats, scales=scales, opacities=opacities,
        colors=features, viewmats=viewmat[None], Ks=K[None],
        # Omitting backgrounds means exactly zero feature background.
        width=width, height=height,
        sh_degree=None, render_mode="RGB", **rasterizer_kwargs,
    )
    return {"first_moment": rendered[0, ..., 0],
            "second_moment": rendered[0, ..., 1], "alpha": alpha[0, ..., 0],
            "feature_mass": rendered[0, ..., 2],
            "meta": meta}


def depth_moment_loss(
    first_moment: Tensor,
    second_moment: Tensor,
    alpha: Tensor,
    target_depth: Tensor,
    valid_mask: Tensor,
    confidence: Tensor | None = None,
    *,
    relative: bool = True,
    robust: bool = False,
    alpha_epsilon: float = 1e-6,
    min_depth: float = 1e-3,
) -> Tensor:
    """E[(z-D)^2], optionally divided by D^2; zero background assumed.

    Unnormalized moments yield (M2-2*D*M1+D^2*A)/A, algebraically equal to
    variance(z)+(E[z]-D)^2. Clamp small negative cancellation error to zero.
    The raw relative loss is dimensionless; the raw absolute loss is metres^2.
    robust=True applies sqrt(1+per_pixel_loss)-1 BEFORE the weighted mean,
    so a distant outlier cannot attenuate all other pixels' gradients. It uses
    an algebraically identical stable quotient to avoid small-loss cancellation.
    In absolute mode its implicit transition scale is one metre.
    Finite target-depth validity is combined with the supplied fixed mask.
    Never derive that mask from the current alpha: maintain a separate coverage
    constraint so lowering opacity cannot escape supervision as A approaches 0.
    """
    tensors = (first_moment, second_moment, alpha, target_depth, valid_mask)
    if len({x.shape for x in tensors}) != 1:
        raise ValueError("all pixel tensors must have identical shapes")
    valid = valid_mask.bool() & torch.isfinite(target_depth) & (target_depth > min_depth)
    depth = torch.where(valid, target_depth, torch.ones_like(target_depth))
    residual = (second_moment - 2 * depth * first_moment + depth.square() * alpha).clamp_min(0)
    loss = residual / alpha.clamp_min(alpha_epsilon)
    if relative:
        loss = loss / depth.square()
    if robust:
        loss = loss / (torch.sqrt(1 + loss) + 1)
    return _weighted_mean(loss, valid, confidence)


def alpha_coverage_loss(alpha: Tensor, reference_alpha: Tensor, valid_mask: Tensor,
                        confidence: Tensor | None = None, *, tolerance: float = 0.01) -> Tensor:
    """Squared one-sided alpha drop, measured against detached baseline alpha.

    A baseline preserves existing coverage. Use an all-ones reference only on
    known opaque ground where adding missing surface coverage is intended.
    """
    if alpha.shape != reference_alpha.shape:
        raise ValueError("alpha and reference_alpha shapes must agree")
    return _weighted_mean((reference_alpha.detach() - tolerance - alpha).clamp_min(0).square(),
                          valid_mask.bool(), confidence)


def ray_free_space_loss(
    center_depth: Tensor,
    target_depth: Tensor,
    opacities: Tensor,
    valid_mask: Tensor,
    confidence: Tensor | None = None,
    *,
    margin_abs: float = 0.5,
    margin_relative: float = 0.10,
    ray_sigma: Tensor | None = None,
    sigma_multiple: float = 0.0,
    relative: bool = True,
    min_depth: float = 1e-3,
) -> Tensor:
    """Opacity-weighted one-sided free-space potential on projected candidates.

    A candidate in front of D-(margin_abs+margin_relative*D) is contradicted;
    a candidate behind D is occluded and neutral. target_depth, confidence,
    valid_mask should come from conservative multi-station prior verification.
    Detaching sampled target depths is recommended to prevent lateral motion
    chasing edges; center_depth itself MUST retain the position gradient.
    Normalization excludes opacity, so opacity reduction can suppress floaters.

    Optional ray_sigma shifts the tested point to center-k*sigma. That is an
    extent heuristic, not exact Gaussian visibility. Default tests centers only.
    Relative mode is dimensionless; the other mode is opacity * metres^2.
    """
    if len({x.shape for x in (center_depth, target_depth, opacities, valid_mask)}) != 1:
        raise ValueError("all candidate tensors must have identical shapes")
    valid = (valid_mask.bool() & torch.isfinite(target_depth) & (target_depth > min_depth)
             & torch.isfinite(center_depth) & (center_depth > min_depth))
    depth = torch.where(valid, target_depth, torch.ones_like(target_depth))
    tested = torch.where(valid, center_depth, depth)
    if ray_sigma is not None:
        if ray_sigma.shape != center_depth.shape:
            raise ValueError("ray_sigma must have the same candidate shape")
        tested = tested - sigma_multiple * ray_sigma
    violation = (depth - margin_abs - margin_relative * depth - tested).clamp_min(0)
    if relative:
        violation = violation / depth
    return _weighted_mean(opacities * violation.square(), valid, confidence)


def quaternion_to_matrix_wxyz(quats: Tensor) -> Tensor:
    """Normalize (...,4) wxyz quaternions; an all-zero quaternion maps to identity."""
    if quats.shape[-1] != 4:
        raise ValueError("quaternions must end in 4 wxyz components")
    q = quats / torch.linalg.vector_norm(quats, dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack((
        1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y),
    ), dim=-1).reshape(quats.shape[:-1] + (3, 3))


def normal_standard_deviation(scales: Tensor, quats: Tensor, normals: Tensor) -> Tensor:
    """sqrt(n^T Sigma n), with unit n and Sigma=R diag(scales^2) R^T.

    All inputs share leading dimensions. This constrains only the component
    normal to a surface, preserving large legitimate in-plane footprints.
    """
    if scales.shape != normals.shape or scales.shape[:-1] != quats.shape[:-1]:
        raise ValueError("scales/normals/quaternions must share leading dimensions")
    n = normals / torch.linalg.vector_norm(normals, dim=-1, keepdim=True).clamp_min(1e-12)
    local_n = (quaternion_to_matrix_wxyz(quats).transpose(-1, -2) @ n[..., None]).squeeze(-1)
    return (scales.square() * local_n.square()).sum(-1).clamp_min(1e-24).sqrt()


def local_plane_loss(
    means: Tensor,
    scales: Tensor,
    quats: Tensor,
    plane_normals: Tensor,
    plane_offsets: Tensor,
    ground_mask: Tensor,
    confidence: Tensor | None = None,
    *,
    mean_tolerance: float = 0.05,
    normal_sigma_max: float = 0.08,
    below_tolerance: float = 0.20,
    sigma_multiple: float = 3.0,
    distance_scale: float = 1.0,
    mean_weight: float = 1.0,
    extent_weight: float = 1.0,
    tail_weight: float = 1.0,
) -> dict[str, Tensor]:
    """Per-row fixed local plane: n points toward free space, n.dot(X)+b=0.

    Plane normals [N,3], offsets [N], and mask [N] are fixed associations,
    prepared externally from independently validated surface support.
    No opacity weighting: associated surfaces cannot evade geometry by fading.
    Unassociated rows contribute zero; no plane is inferred by this function.

    Penalize mean-to-plane error, normal sigma exceeding its bound, and the
    k-sigma lower extent: relu(k*sigma_n-signed_height-below_tolerance)^2.
    These are finite k-sigma envelopes; a Gaussian has infinite mathematical
    tails. Thresholds are configurable world-unit tolerances, not learned truth.
    Divide metre residuals by distance_scale (default 1 m) before squaring.
    """
    if distance_scale <= 0:
        raise ValueError("distance_scale must be positive")
    if (means.ndim != 2 or means.shape[-1] != 3 or plane_normals.shape != means.shape
            or scales.shape != means.shape or quats.shape != (len(means), 4)
            or plane_offsets.shape != (len(means),) or ground_mask.shape != (len(means),)):
        raise ValueError("expected row arrays: means/scales/normals [N,3], quats [N,4], offsets/mask [N]")
    # Targets/association are deliberately fixed; model parameters are not.
    lengths = torch.linalg.vector_norm(plane_normals.detach(), dim=-1)
    valid = ground_mask.bool() & torch.isfinite(lengths) & (lengths > 1e-12) & torch.isfinite(plane_offsets)
    n = torch.where(valid[:, None], plane_normals.detach(), means.new_tensor([0., 1., 0.]))
    divisor = torch.where(valid, lengths, torch.ones_like(lengths))
    n = n / divisor[:, None]
    b = torch.where(valid, plane_offsets.detach(), torch.zeros_like(plane_offsets)) / divisor
    signed_height = (means * n).sum(-1) + b
    normal_sigma = normal_standard_deviation(scales, quats, n)
    mean_residual = (signed_height.abs() - mean_tolerance).clamp_min(0) / distance_scale
    extent_residual = (normal_sigma - normal_sigma_max).clamp_min(0) / distance_scale
    tail_residual = (sigma_multiple * normal_sigma - signed_height - below_tolerance).clamp_min(0) / distance_scale
    mean_loss = _weighted_mean(mean_residual.square(), valid, confidence)
    extent_loss = _weighted_mean(extent_residual.square(), valid, confidence)
    tail_loss = _weighted_mean(tail_residual.square(), valid, confidence)
    return {"loss": mean_weight*mean_loss + extent_weight*extent_loss + tail_weight*tail_loss,
            "mean_loss": mean_loss, "extent_loss": extent_loss, "tail_loss": tail_loss,
            "mean_abs_distance_m": _weighted_mean(signed_height.abs(), valid, confidence),
            "mean_normal_sigma_m": _weighted_mean(normal_sigma, valid, confidence)}
