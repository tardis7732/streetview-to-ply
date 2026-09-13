"""Complete sparse-depth footprint evidence using a native-pixel tile index.

The index only accelerates the exact observed_footprint policy. No centre-pixel
screen, filled depth, footprint radius cap, or subsampled observations are used.
Both expanded tile pairs and expanded pixel pairs have bounded batch sizes.
"""
from dataclasses import dataclass

import torch

from .sparse_prune import classify, project


_SOLVER_BATCH = 4096


def _versions(tensors):
    try:
        return tuple(t._version for t in tensors)
    except RuntimeError as exc:
        raise ValueError('Prepared geometry requires tensors with tracked mutation versions') from exc


def _bounded_eigvalsh(matrices):
    """cuSOLVER workspace must not grow with the full scene's Gaussian count."""
    values = torch.empty(matrices.shape[:-1], dtype=matrices.dtype, device=matrices.device)
    for first in range(0, len(matrices), _SOLVER_BATCH):
        values[first:first+_SOLVER_BATCH] = torch.linalg.eigvalsh(matrices[first:first+_SOLVER_BATCH])
    return values


def _bounded_inverse(matrices):
    inverse = torch.empty_like(matrices)
    for first in range(0, len(matrices), _SOLVER_BATCH):
        inverse[first:first+_SOLVER_BATCH] = torch.linalg.inv(matrices[first:first+_SOLVER_BATCH])
    return inverse


@dataclass(frozen=True)
class PreparedGeometry:
    """Validated world geometry reusable across cameras, bound to exact inputs."""
    means_input: torch.Tensor
    covariance_input: torch.Tensor
    input_versions: tuple
    safe_means: torch.Tensor
    safe_cov: torch.Tensor
    positive: torch.Tensor
    cache_versions: tuple

    def validate(self, means, covariance):
        if self.means_input is not means or self.covariance_input is not covariance:
            raise ValueError('Prepared geometry is bound to different tensor identities')
        if self.input_versions != _versions((means, covariance)):
            raise ValueError('Prepared geometry inputs changed in place')
        if self.cache_versions != _versions((self.safe_means, self.safe_cov, self.positive)):
            raise ValueError('Prepared geometry cache changed in place')


@torch.no_grad()
def prepare_geometry(means, covariance):
    """Validate finite/symmetric/positive world covariance once, in bounded batches.

    Returned geometry is invalidated by in-place edits to either original input
    tensor. New/reordered/cloned tensors require their own preparation.
    """
    if not isinstance(means, torch.Tensor) or means.ndim != 2 or means.shape[1] != 3:
        raise ValueError('Expected torch means [rows,3]')
    if not isinstance(covariance, torch.Tensor) or covariance.shape != (len(means), 3, 3):
        raise ValueError('Expected covariance [rows,3,3]')
    versions = _versions((means, covariance))
    mm = means.to(device=means.device, dtype=torch.float64)
    cov = covariance.to(device=means.device, dtype=torch.float64)
    numeric = torch.isfinite(mm).all(1) & torch.isfinite(cov).all((1, 2))
    safe_means = torch.where(numeric[:, None], mm, torch.zeros_like(mm))
    safe_cov = torch.where(numeric[:, None, None], cov,
                           torch.eye(3, dtype=torch.float64, device=means.device)[None])
    eig3 = _bounded_eigvalsh((safe_cov+safe_cov.transpose(-1, -2))*.5)
    cov_scale = eig3.abs().amax(1)
    eps = torch.finfo(torch.float64).eps
    symmetric = (safe_cov-safe_cov.transpose(-1, -2)).abs().amax((1, 2)) <= 64*eps*cov_scale
    positive = numeric & symmetric & (eig3[:, 0] > 0)
    if versions != _versions((means, covariance)):
        raise ValueError('Input geometry changed during preparation')
    return PreparedGeometry(means, covariance, versions, safe_means, safe_cov, positive,
                            _versions((safe_means, safe_cov, positive)))


def _projected_positive(cov2):
    """Analytic 2x2 eigen test, with bounded exact fallback near degeneracy.

    Scaling avoids determinant/trace overflow. Close to the 64eps eigenvalue
    threshold, determinant cancellation could change a decision; those rows
    use the same eigensolver as the brute-force policy in bounded batches.
    """
    eps = torch.finfo(torch.float64).eps
    symmetric = (cov2+cov2.transpose(-1, -2))*.5
    magnitude = symmetric.abs().amax((1, 2))
    safe_magnitude = torch.where(magnitude > 0, magnitude, torch.ones_like(magnitude))
    normalized = symmetric/safe_magnitude[:, None, None]
    a, b, d = normalized[:, 0, 0], normalized[:, 0, 1], normalized[:, 1, 1]
    largest = (a+d+torch.hypot(a-d, 2*b))*.5
    determinant = a*d-b.square()
    smallest = determinant/torch.where(largest > 0, largest, torch.ones_like(largest))
    positive = (magnitude > 0) & (smallest > 64*eps*largest)
    borderline = (smallest <= 4096*eps*largest) | ~torch.isfinite(smallest)
    ids = borderline.nonzero().flatten()
    if len(ids):
        eig2 = _bounded_eigvalsh(symmetric[ids])
        positive[ids] = eig2[:, 0] > 64*eps*eig2[:, 1]
    return positive


def _projected_inverse(cov2):
    """Analytic inverse except ill-conditioned rows, with bounded fallback."""
    magnitude = cov2.abs().amax((1, 2))
    normalized = cov2/magnitude[:, None, None]
    a, b, c, d = normalized[:, 0, 0], normalized[:, 0, 1], normalized[:, 1, 0], normalized[:, 1, 1]
    determinant = a*d-b*c
    adjugate = torch.stack((d, -b, -c, a), dim=1).reshape(-1, 2, 2)
    inverse = (adjugate/determinant[:, None, None])/magnitude[:, None, None]
    # On highly eccentric ellipses LU avoids determinant cancellation.
    fallback = (determinant.abs() < 1e-8) | ~torch.isfinite(inverse).all((1, 2))
    ids = fallback.nonzero().flatten()
    if len(ids):
        inverse[ids] = _bounded_inverse(cov2[ids])
    return inverse


@torch.no_grad()
def observe_indexed(means, covariance, view, K, depth, valid, strict, config,
                    tile_size=16, pair_budget=1048576, prepared=None):
    """Return observe()'s five per-row tensors plus scalar ``diagnostics``.

    All geometry is float64 on means.device. Native depth pixel centres i+.5
    inside the 2-sigma ellipse supply strict free witnesses; any valid nonfree
    pixel inside the 3-sigma ellipse vetoes the view. The entire camera-Z
    3-sigma interval must be in front of the observed depth with its margin.
    """
    if not isinstance(tile_size, int) or isinstance(tile_size, bool) or tile_size < 1:
        raise ValueError('tile_size must be a positive integer')
    if not isinstance(pair_budget, int) or isinstance(pair_budget, bool) or pair_budget < 1:
        raise ValueError('pair_budget must be a positive integer')
    if not isinstance(means, torch.Tensor) or means.ndim != 2 or means.shape[1] != 3:
        raise ValueError('Expected torch means [rows,3]')
    n = len(means)
    if covariance.shape != (n, 3, 3) or view.shape not in ((3, 4), (4, 4)) or K.shape != (3, 3):
        raise ValueError('Gaussian covariance or camera shape differs')
    if depth.ndim != 2 or valid.shape != depth.shape or strict.shape != depth.shape:
        raise ValueError('Expected matching native depth and mask grids')
    if valid.dtype != torch.bool or strict.dtype != torch.bool:
        raise ValueError('Depth evidence masks must be boolean')
    device = means.device
    if prepared is not None:
        if not isinstance(prepared, PreparedGeometry):
            raise ValueError('prepared must come from prepare_geometry')
        prepared.validate(means, covariance)
    view, K, depth = [x.to(device=device, dtype=torch.float64) for x in (view, K, depth)]
    valid, strict = valid.to(device=device), strict.to(device=device)
    if torch.any(strict & ~valid):
        raise ValueError('Strict evidence must be a subset of valid evidence')
    if not torch.isfinite(view).all() or not torch.isfinite(K).all():
        raise ValueError('Finite calibrated camera required')
    if K[0, 0] <= 0 or K[1, 1] <= 0 or torch.any(K[[0, 1], [1, 0]] != 0):
        raise ValueError('Positive pinhole focal lengths and zero skew required')

    counts = torch.zeros(n, dtype=torch.int64, device=device)
    free_count, blocked_count, support_count = counts.clone(), counts.clone(), counts.clone()
    diagnostics = dict(gaussian_rows=n, front_projectable_rows=0,
                       footprint_image_rows=0, rows_with_depth_observations=0,
                       native_valid_depth_pixels=0, tile_pairs=0, depth_pairs=0,
                       inside_depth_pairs=0, max_tile_batch=0, max_depth_batch=0,
                       tile_size=tile_size, pair_budget=pair_budget)

    def finish():
        blocked, supported = blocked_count > 0, support_count > 0
        diagnostics['rows_with_depth_observations'] = int((counts > 0).sum().item())
        diagnostics['inside_depth_pairs'] = int(counts.sum().item())
        return dict(free_view=(free_count >= config.min_footprint_samples) & ~blocked & ~supported,
                    blocked_view=blocked, support_view=supported,
                    strict_free_pixels=free_count, valid_pixels=counts,
                    diagnostics=diagnostics)

    original = valid & torch.isfinite(depth) & (depth > 0)
    locations = original.nonzero(as_tuple=False)
    diagnostics['native_valid_depth_pixels'] = len(locations)
    if not n or not len(locations):
        return finish()

    # World covariance validity is independent of the camera. Validate it once
    # per scene when a prepared cache is supplied, never in a full-N solver.
    if prepared is None:
        prepared = prepare_geometry(means, covariance)
    safe_means, safe_cov = prepared.safe_means, prepared.safe_cov
    eps = torch.finfo(torch.float64).eps
    positive = prepared.positive.clone()
    xyz, uv, sigma_z = project(safe_means, safe_cov, view, K)
    positive &= torch.isfinite(xyz).all(1) & torch.isfinite(uv).all(1)
    positive &= torch.isfinite(sigma_z) & (xyz[:, 2]-config.gaussian_sigma*sigma_z > 0)
    safe_z = torch.where(positive, xyz[:, 2], torch.ones_like(xyz[:, 2]))
    zero = torch.zeros_like(safe_z)
    jx = torch.stack([K[0, 0]/safe_z, zero, -K[0, 0]*xyz[:, 0]/safe_z.square()], -1) @ view[:3, :3]
    jy = torch.stack([zero, K[1, 1]/safe_z, -K[1, 1]*xyz[:, 1]/safe_z.square()], -1) @ view[:3, :3]
    J = torch.stack([jx, jy], 1)
    cov2 = J @ safe_cov @ J.transpose(-1, -2)
    finite_cov2 = torch.isfinite(cov2).all((1, 2))
    safe_cov2 = torch.where(finite_cov2[:, None, None], cov2,
                            torch.eye(2, dtype=torch.float64, device=device)[None])
    projected_ids = (positive & finite_cov2).nonzero().flatten()
    projected_valid = torch.zeros_like(positive)
    if len(projected_ids):
        projected_valid[projected_ids] = _projected_positive(safe_cov2[projected_ids])
    positive &= projected_valid
    diagnostics['front_projectable_rows'] = int(positive.sum().item())
    if not positive.any():
        return finish()

    # A coordinate-wise 3-sigma ellipse bound is sqrt(C_xx), sqrt(C_yy),
    # independent of ellipse rotation. Clip floats BEFORE int conversion to
    # avoid overflow for enormous yet finite projected ellipses.
    h, w = depth.shape
    extent = 3*safe_cov2.diagonal(dim1=-2, dim2=-1).clamp_min(0).sqrt()
    # This is only a broad phase: round its bounds outward so arithmetic
    # cancellation or a last-bit covariance/inverse discrepancy cannot cull
    # an original pixel that the exact q test accepts on an ellipse boundary.
    # The guard does NOT change the q<=4/q<=9 acceptance thresholds.
    guard = (64*eps)*uv.abs()+(64*eps)*extent+64*eps
    neg_inf = torch.full_like(uv, -torch.inf)
    pos_inf = torch.full_like(uv, torch.inf)
    lower = torch.ceil(torch.nextafter(uv-extent-.5-guard, neg_inf))
    upper = torch.floor(torch.nextafter(uv+extent-.5+guard, pos_inf))
    overlap = positive & (lower <= upper).all(1)
    overlap &= (lower[:, 0] < w) & (upper[:, 0] >= 0)
    overlap &= (lower[:, 1] < h) & (upper[:, 1] >= 0)
    row_ids = overlap.nonzero().flatten()
    diagnostics['footprint_image_rows'] = len(row_ids)
    if not len(row_ids):
        return finish()
    lo = lower[row_ids].clamp_min(0)
    hi = upper[row_ids].clamp_min(0)
    lo[:, 0].clamp_max_(w-1); lo[:, 1].clamp_max_(h-1)
    hi[:, 0].clamp_max_(w-1); hi[:, 1].clamp_max_(h-1)
    lo, hi = lo.long()//tile_size, hi.long()//tile_size
    widths = hi[:, 0]-lo[:, 0]+1
    tile_counts = widths*(hi[:, 1]-lo[:, 1]+1)
    tile_ends = tile_counts.cumsum(0)
    tile_starts = tile_ends-tile_counts
    total_tiles = int(tile_ends[-1].item())
    diagnostics['tile_pairs'] = total_tiles
    inverse = _projected_inverse(safe_cov2[row_ids])

    # CSR indexes original observations once; each pixel belongs to one tile,
    # so expansion cannot count duplicate native witnesses for any Gaussian.
    tiles_x = (w+tile_size-1)//tile_size
    tiles_y = (h+tile_size-1)//tile_size
    tile_of_pixel = (locations[:, 0]//tile_size)*tiles_x + locations[:, 1]//tile_size
    order = torch.argsort(tile_of_pixel, stable=True)
    pixel_counts = torch.bincount(tile_of_pixel, minlength=tiles_x*tiles_y)
    pixel_ends = pixel_counts.cumsum(0)
    pixel_starts = pixel_ends-pixel_counts
    pixels = locations[order][:, [1, 0]].to(torch.float64)+.5
    observed_z = depth[original][order]
    observed_strict = strict[original][order]

    for tile_first in range(0, total_tiles, pair_budget):
        tile_last = min(tile_first+pair_budget, total_tiles)
        diagnostics['max_tile_batch'] = max(diagnostics['max_tile_batch'], tile_last-tile_first)
        serial = torch.arange(tile_first, tile_last, dtype=torch.int64, device=device)
        local_rows = torch.searchsorted(tile_ends, serial, right=True)
        offset = serial-tile_starts[local_rows]
        tx = lo[local_rows, 0]+offset % widths[local_rows]
        ty = lo[local_rows, 1]+offset//widths[local_rows]
        tile_ids = ty*tiles_x+tx
        nonempty = pixel_counts[tile_ids] > 0
        tile_ids, local_rows = tile_ids[nonempty], local_rows[nonempty]
        if not len(tile_ids):
            continue
        pair_counts = pixel_counts[tile_ids]
        pair_ends = pair_counts.cumsum(0)
        pair_starts = pair_ends-pair_counts
        total_pairs = int(pair_ends[-1].item())
        diagnostics['depth_pairs'] += total_pairs
        for first in range(0, total_pairs, pair_budget):
            last = min(first+pair_budget, total_pairs)
            diagnostics['max_depth_batch'] = max(diagnostics['max_depth_batch'], last-first)
            serial = torch.arange(first, last, dtype=torch.int64, device=device)
            slot = torch.searchsorted(pair_ends, serial, right=True)
            local = local_rows[slot]
            rows = row_ids[local]
            pix = pixel_starts[tile_ids[slot]]+serial-pair_starts[slot]
            delta = pixels[pix]-uv[rows]
            q = torch.einsum('ni,nij,nj->n', delta, inverse[local], delta)
            inside3 = torch.isfinite(q) & (q >= 0) & (q <= 9.)
            inside2 = inside3 & (q <= 4.)
            free, support = classify(xyz[rows, 2], sigma_z[rows], observed_z[pix], inside3, config)
            counts.scatter_add_(0, rows, inside3.long())
            free_count.scatter_add_(0, rows, (inside2 & free & observed_strict[pix]).long())
            blocked_count.scatter_add_(0, rows, (inside3 & ~free).long())
            support_count.scatter_add_(0, rows, support.long())
    return finish()
