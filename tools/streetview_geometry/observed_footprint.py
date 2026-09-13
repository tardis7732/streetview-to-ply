"""Sparse depth witnesses over complete observed projected Gaussian footprints.

Only original valid pixel centres are visited. Unknown pixels remain unknown;
ellipses select observations and never spread or interpolate depth evidence.
"""
import torch

from .sparse_prune import classify, project


def observe(means, covariance, view, K, depth, valid, strict, config):
    """Return per-row evidence from original pixels inside 2/3-sigma ellipses.

Three-sigma depth extents must be wholly in front of an observed surface to
vote free. At least ``config.min_footprint_samples`` distinct strict pixels
inside the two-sigma ellipse are required. Any loose nonfree observation in
the three-sigma ellipse vetoes the view, including occluded/behind evidence.

    Invalid or camera-plane-crossing projections abstain in that view. They
    cannot invent a veto in another cube face with no observed footprint.
    Computation is float64 on the input means' CPU/CUDA device.
"""
    if not isinstance(means, torch.Tensor) or means.ndim != 2 or means.shape[1] != 3:
        raise ValueError('Expected torch means [rows,3]')
    n = len(means)
    if covariance.shape != (n, 3, 3) or view.shape not in ((3, 4), (4, 4)) or K.shape != (3, 3):
        raise ValueError('Gaussian covariance or camera shape differs')
    if depth.ndim != 2 or valid.shape != depth.shape or strict.shape != depth.shape:
        raise ValueError('Expected matching native depth and mask grids')
    if valid.dtype != torch.bool or strict.dtype != torch.bool:
        raise ValueError('Depth evidence masks must be boolean')
    if torch.any(strict & ~valid):
        raise ValueError('Strict evidence must be a subset of valid evidence')
    device = means.device
    means, covariance, view, K, depth = [x.to(device=device, dtype=torch.float64)
                                       for x in (means, covariance, view, K, depth)]
    valid, strict = valid.to(device=device), strict.to(device=device)
    if not torch.isfinite(view).all() or not torch.isfinite(K).all():
        raise ValueError('Finite calibrated camera required')
    if K[0, 0] <= 0 or K[1, 1] <= 0 or torch.any(K[[0, 1], [1, 0]] != 0):
        raise ValueError('Positive pinhole focal lengths and zero skew required')

    counts = torch.zeros(n, dtype=torch.int64, device=device)
    free_count = counts.clone()
    blocked = torch.zeros(n, dtype=torch.bool, device=device)
    supported = blocked.clone()
    output = dict(free_view=blocked.clone(), blocked_view=blocked,
                  support_view=supported, strict_free_pixels=free_count,
                  valid_pixels=counts)
    if not n:
        return output

    # Per-row invalid geometry cannot contaminate the batched decomposition.
    numeric = torch.isfinite(means).all(1) & torch.isfinite(covariance).all((1, 2))
    safe_means = torch.where(numeric[:, None], means, torch.zeros_like(means))
    safe_cov = torch.where(numeric[:, None, None], covariance,
                           torch.eye(3, dtype=torch.float64, device=device)[None])
    eig3 = torch.linalg.eigvalsh((safe_cov+safe_cov.transpose(-1, -2))*.5)
    cov_scale = eig3.abs().amax(1)
    eps = torch.finfo(torch.float64).eps
    symmetric = (safe_cov-safe_cov.transpose(-1, -2)).abs().amax((1, 2)) <= 64*eps*cov_scale
    positive = (eig3[:, 0] > 0) & symmetric
    xyz, uv, sigma_z = project(safe_means, safe_cov, view, K)
    positive &= numeric & torch.isfinite(xyz).all(1) & torch.isfinite(uv).all(1)
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
    eig2 = torch.linalg.eigvalsh((safe_cov2+safe_cov2.transpose(-1, -2))*.5)
    positive &= finite_cov2 & (eig2[:, 0] > 64*eps*eig2[:, 1])
    original = valid & torch.isfinite(depth) & (depth > 0)
    locations = original.nonzero(as_tuple=False)
    if not len(locations) or not positive.any():
        return output
    pixels = locations[:, [1, 0]].to(torch.float64)+.5
    observed_z = depth[original]
    observed_strict = strict[original]
    row_ids = positive.nonzero().flatten()
    inverse = torch.linalg.inv(safe_cov2[row_ids])
    # Bounds memory to about 1M row/pixel pairs regardless of scene size.
    row_chunk, pixel_chunk = 64, 16384
    for start in range(0, len(row_ids), row_chunk):
        ids = row_ids[start:start+row_chunk]
        inv = inverse[start:start+row_chunk]
        for first in range(0, len(pixels), pixel_chunk):
            last = first+pixel_chunk
            delta = pixels[None, first:last]-uv[ids, None]
            q = torch.einsum('npi,nij,npj->np', delta, inv, delta)
            inside3 = torch.isfinite(q) & (q >= 0) & (q <= 9.)
            inside2 = inside3 & (q <= 4.)
            free, support = classify(xyz[ids, 2, None], sigma_z[ids, None],
                                     observed_z[None, first:last], inside3, config)
            counts[ids] += inside3.sum(1)
            free_count[ids] += (inside2 & free & observed_strict[None, first:last]).sum(1)
            blocked[ids] |= (inside3 & ~free).any(1)
            supported[ids] |= support.any(1)
    output['free_view'] = (free_count >= config.min_footprint_samples) & ~blocked & ~supported
    return output
