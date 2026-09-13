"""Conservative, bounded front-of-predicted-surface pruning evidence.

This is a hard-removal *proposal*, not an opacity loss. Inputs are calibrated
camera-Z predictions in the same units as Gaussian geometry. Predictions are
not measurements or ground-truth free-space certificates.

Every Gaussian is considered. The exact perspective rectangle enclosing its
k-sigma ellipsoid is intersected with native dense-depth pixel *cells*. Thus a
subpixel Gaussian can use one native pixel, and depth at an off-centre part of
the footprint is never silently omitted. Pixel cells [i,i+1] have centre i+.5.

For bounded work we query a dyadic min/max pyramid over a rectangle SUPerset.
The entire ellipsoid's camera-Z interval must be in front of every predicted
lower bound in that superset. This can retain extra geometry, never manufacture
a free decision compared with an exact rectangle minimum. Range overlap is
only POSSIBLE surface support, not a ray/ellipsoid surface intersection. A
nearer occluder, an unrelated surface in the rectangle, or the pyramid superset
can overprotect geometry. Unknown pixels are not filled; insufficient native
coverage or any footprint outside the image abstains from a free vote.
"""
from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F

from .indexed_footprint import PreparedGeometry, prepare_geometry


@dataclass(frozen=True)
class DenseFrontConfig:
    gaussian_sigma: float = 3.0
    relative_depth_margin: float = 0.10
    uncertainty_multiplier: float = 1.0
    minimum_valid_fraction: float = 0.95
    minimum_valid_pixels: int = 1
    minimum_confidence: float = 0.0
    minimum_free_stations: int = 3

    def __post_init__(self):
        numbers = (self.gaussian_sigma, self.relative_depth_margin,
                   self.uncertainty_multiplier, self.minimum_valid_fraction,
                   self.minimum_confidence)
        if not all(math.isfinite(x) for x in numbers):
            raise ValueError('Policy values must be finite')
        if self.gaussian_sigma <= 0 or self.relative_depth_margin < 0 or self.uncertainty_multiplier < 0:
            raise ValueError('Positive Gaussian extent and nonnegative margins required')
        if not 0 < self.minimum_valid_fraction <= 1 or not 0 <= self.minimum_confidence <= 1:
            raise ValueError('Coverage/confidence must be fractions in their valid ranges')
        for value in (self.minimum_valid_pixels, self.minimum_free_stations):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError('Observation minima must be positive integer counts')
        if self.minimum_free_stations < 2:
            raise ValueError('Removal requires independent physical stations')


def _version(tensors):
    try:
        return tuple(x._version for x in tensors)
    except RuntimeError as exc:
        raise ValueError('Prepared depth requires tracked tensor mutation versions') from exc


@dataclass(frozen=True)
class PreparedDepth:
    height: int
    width: int
    valid: torch.Tensor
    integral_valid: torch.Tensor
    lower_pyramid: tuple
    upper_pyramid: tuple
    interval_source: str
    config: DenseFrontConfig
    versions: tuple

    def validate(self, config, device):
        if config != self.config:
            raise ValueError('Depth preparation and observation policy differ')
        if self.valid.device != device:
            raise ValueError('Prepare depth on the same device as Gaussian means')
        if self.versions != _version((self.valid, self.integral_valid,
                                      *self.lower_pyramid, *self.upper_pyramid)):
            raise ValueError('Prepared depth cache was modified')


@torch.no_grad()
def prepare_depth(depth_z, valid, uncertainty_z=None, *, confidence=None,
                  depth_lower_z=None, depth_upper_z=None,
                  config=DenseFrontConfig()):
    """Prepare one native image without resizing, interpolation or depth filling.

    ``uncertainty_z`` is an explicit nonnegative interval half-width, NOT a
    confidence score or assumed standard deviation. It may be a scalar tensor
    or [H,W]. Relative margins are added separately. Confidence, when supplied,
    is only an eligibility gate in [0,1], never converted to depth uncertainty.
    Alternatively supply BOTH ``depth_lower_z`` and ``depth_upper_z`` as native
    asymmetric empirical bounds, with ``uncertainty_z=None``. Their interval is
    conservatively enlarged to contain the point prediction; the multiplier
    scales each side's distance from that prediction, retaining asymmetry.
    The prepared snapshot does not share input tensor storage.
    """
    if not isinstance(depth_z, torch.Tensor) or depth_z.ndim != 2 or min(depth_z.shape) < 1:
        raise ValueError('Native depth must be a nonempty torch [H,W] tensor')
    if not isinstance(valid, torch.Tensor) or valid.shape != depth_z.shape or valid.dtype != torch.bool:
        raise ValueError('Native validity must be a matching boolean tensor')
    device = depth_z.device
    depth = depth_z.to(dtype=torch.float64)
    mask = valid.to(device=device).clone()
    mask &= torch.isfinite(depth) & (depth > 0)
    if depth_lower_z is not None or depth_upper_z is not None:
        if depth_lower_z is None or depth_upper_z is None or uncertainty_z is not None:
            raise ValueError('Supply both asymmetric bounds OR an uncertainty half-width')
        low = torch.as_tensor(depth_lower_z, dtype=torch.float64, device=device)
        high = torch.as_tensor(depth_upper_z, dtype=torch.float64, device=device)
        if low.shape != depth.shape or high.shape != depth.shape:
            raise ValueError('Asymmetric bounds must match the native grid')
        mask &= torch.isfinite(low) & torch.isfinite(high) & (low > 0) & (high >= low)
        # An empirical interval may not straddle a biased point prediction.
        # Taking the hull preserves that evidence and only makes pruning harder.
        raw_lower = depth + config.uncertainty_multiplier*(torch.minimum(low, depth)-depth)
        raw_upper = depth + config.uncertainty_multiplier*(torch.maximum(high, depth)-depth)
        interval_source = 'asymmetric_explicit_bounds_hull_with_point_prediction'
    else:
        if uncertainty_z is None:
            raise ValueError('Explicit uncertainty half-width or asymmetric bounds required')
        uncertainty = torch.as_tensor(uncertainty_z, dtype=torch.float64, device=device)
        if uncertainty.ndim and uncertainty.shape != depth.shape:
            raise ValueError('Uncertainty must be scalar or native [H,W]')
        uncertainty = torch.broadcast_to(uncertainty, depth.shape)
        mask &= torch.isfinite(uncertainty) & (uncertainty >= 0)
        raw_lower = depth - config.uncertainty_multiplier*uncertainty
        raw_upper = depth + config.uncertainty_multiplier*uncertainty
        interval_source = 'symmetric_explicit_interval_half_width'
    if confidence is not None:
        score = torch.as_tensor(confidence, dtype=torch.float64, device=device)
        if score.shape != depth.shape:
            raise ValueError('Confidence must match the native grid')
        mask &= torch.isfinite(score) & (score >= config.minimum_confidence) & (score >= 0) & (score <= 1)
    elif config.minimum_confidence > 0:
        raise ValueError('A positive confidence gate requires explicit confidence')
    margin = config.relative_depth_margin * depth
    lower = torch.where(mask, raw_lower - margin, torch.inf)
    upper = torch.where(mask, raw_upper + margin, -torch.inf)
    lower_levels, upper_levels = [lower], [upper]
    while max(lower_levels[-1].shape) > 1:
        lo, hi = lower_levels[-1], upper_levels[-1]
        lower_levels.append(-F.max_pool2d(-lo[None, None], 2, 2, ceil_mode=True)[0, 0])
        upper_levels.append(F.max_pool2d(hi[None, None], 2, 2, ceil_mode=True)[0, 0])
    integral = F.pad(mask.to(torch.int64).cumsum(0).cumsum(1), (1, 0, 1, 0))
    lo, hi = tuple(lower_levels), tuple(upper_levels)
    return PreparedDepth(*depth.shape, mask, integral, lo, hi, interval_source, config,
                         _version((mask, integral, *lo, *hi)))


def _camera_check(view, K, device):
    if view.shape not in ((3, 4), (4, 4)) or K.shape != (3, 3):
        raise ValueError('Expected world-to-camera [3,4]/[4,4] and pinhole K [3,3]')
    view, K = (x.to(device=device, dtype=torch.float64) for x in (view, K))
    if not torch.isfinite(view).all() or not torch.isfinite(K).all():
        raise ValueError('Camera must be finite')
    if view.shape == (4, 4) and not torch.equal(view[3], view.new_tensor([0, 0, 0, 1])):
        raise ValueError('Invalid homogeneous world-to-camera bottom row')
    rotation = view[:3, :3]
    if not torch.allclose(rotation @ rotation.T, torch.eye(3, dtype=view.dtype, device=device), rtol=1e-6, atol=1e-8):
        raise ValueError('World-to-camera must contain a rigid rotation, not scale')
    if not torch.allclose(torch.det(rotation), view.new_tensor(1.), rtol=1e-6, atol=1e-8):
        raise ValueError('World-to-camera rotation must be proper')
    if K[0, 0] <= 0 or K[1, 1] <= 0 or K[0, 1] != 0 or K[1, 0] != 0 or not torch.equal(K[2], K.new_tensor([0, 0, 1])):
        raise ValueError('Positive zero-skew pinhole camera required')
    return view[:3], K


def _perspective_bounds(mu, covariance, K, k):
    """Exact perspective ratio extrema of a positive-Z ellipsoid.

    For ratio r=x/z the tangent equation is
    (z²-k²Czz)r² - 2(xz-k²Cxz)r + (x²-k²Cxx)=0.
    This is not the first-order Gaussian rasterizer Jacobian ellipse.
    """
    z = mu[:, 2]
    sigma_z = covariance[:, 2, 2].clamp_min(0).sqrt()
    front, back = z - k * sigma_z, z + k * sigma_z
    denominator = z.square() - k*k * covariance[:, 2, 2]
    safe_den = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    bounds, numeric = [], (front > 0) & torch.isfinite(front) & torch.isfinite(back)
    eps = torch.finfo(mu.dtype).eps
    for axis in (0, 1):
        x, cxx, cxz, czz = mu[:, axis], covariance[:, axis, axis], covariance[:, axis, 2], covariance[:, 2, 2]
        terms = torch.stack((cxx*z.square(), -2*cxz*x*z, czz*x.square(), -k*k*(cxx*czz-cxz.square())))
        discriminant = terms.sum(0)
        tolerance = 128*eps*terms.abs().sum(0)
        numeric &= discriminant >= -tolerance
        midpoint = (x*z-k*k*cxz)/safe_den
        radius = k*discriminant.clamp_min(0).sqrt()/safe_den
        lower = K[axis, axis]*(midpoint-radius)+K[axis, 2]
        upper = K[axis, axis]*(midpoint+radius)+K[axis, 2]
        # Outward error guard, including a one-ULP guard at exact cell edges.
        guard = 128*eps*(lower.abs()+upper.abs()+1)
        lower = torch.nextafter(lower-guard, torch.full_like(lower, -torch.inf))
        upper = torch.nextafter(upper+guard, torch.full_like(upper, torch.inf))
        numeric &= torch.isfinite(lower) & torch.isfinite(upper)
        bounds.extend((lower, upper))
    return torch.stack(bounds, 1), front, back, numeric


def _query(prepared, left, right, top, bottom):
    """Exact valid count, conservative depth extrema from <=4 dyadic blocks."""
    integral = prepared.integral_valid
    count = integral[bottom+1, right+1]-integral[top, right+1]-integral[bottom+1, left]+integral[top, left]
    span = torch.maximum(right-left+1, bottom-top+1)
    levels = torch.ceil(torch.log2(span.to(torch.float64))).to(torch.int64)
    lower = torch.full_like(span, torch.inf, dtype=torch.float64)
    upper = torch.full_like(span, -torch.inf, dtype=torch.float64)
    for level, (lo, hi) in enumerate(zip(prepared.lower_pyramid, prepared.upper_pyramid)):
        rows = (levels == level).nonzero().flatten()
        if not len(rows):
            continue
        x0, x1 = left[rows] >> level, right[rows] >> level
        y0, y1 = top[rows] >> level, bottom[rows] >> level
        lower[rows] = torch.stack((lo[y0,x0], lo[y0,x1], lo[y1,x0], lo[y1,x1])).amin(0)
        upper[rows] = torch.stack((hi[y0,x0], hi[y0,x1], hi[y1,x0], hi[y1,x1])).amax(0)
    return count, lower, upper


@torch.no_grad()
def observe_dense_front(means, covariance, view, K, prepared_depth, *,
                        config=DenseFrontConfig(), prepared_geometry=None,
                        chunk_rows=65536):
    """Return per-Gaussian dense depth evidence; this function never edits PLY.

    ``free_view`` requires complete in-image ellipsoid projection, configured
    native rectangle coverage, and back_Z < minimum predicted lower bound.
    ``possible_support_view`` is a deliberately conservative interval overlap.
    ``occluded_view`` (all queried surfaces ahead) and ``unknown_view`` are
    neutral. No Gaussian size cap, centre-pixel gate, or depth pixel subsampling.
    Workspace is O(chunk_rows + native depth); returned arrays are O(rows).
    """
    if not isinstance(chunk_rows, int) or isinstance(chunk_rows, bool) or chunk_rows < 1:
        raise ValueError('chunk_rows must be a positive integer')
    if not isinstance(prepared_depth, PreparedDepth):
        raise ValueError('Use prepare_depth for native dense evidence')
    if not isinstance(means, torch.Tensor) or means.ndim != 2 or means.shape[1] != 3:
        raise ValueError('Expected means [rows,3]')
    prepared_depth.validate(config, means.device)
    if prepared_geometry is None:
        prepared_geometry = prepare_geometry(means, covariance)
    if not isinstance(prepared_geometry, PreparedGeometry):
        raise ValueError('Use prepare_geometry for geometry validation')
    prepared_geometry.validate(means, covariance)
    view, K = _camera_check(view, K, means.device)
    n, device = len(means), means.device
    flags = {key: torch.zeros(n, dtype=torch.bool, device=device) for key in
             ('free_view', 'possible_support_view', 'occluded_view', 'unknown_view',
              'projectable_view', 'fully_in_image', 'coverage_eligible')}
    coverage = torch.zeros(n, dtype=torch.float32, device=device)
    valid_pixels = torch.zeros(n, dtype=torch.int64, device=device)
    footprint_pixels = valid_pixels.clone()
    h, w = prepared_depth.height, prepared_depth.width
    rotation, translation = view[:, :3], view[:, 3]
    for first in range(0, n, chunk_rows):
        last = min(first+chunk_rows, n)
        mu = prepared_geometry.safe_means[first:last] @ rotation.T + translation
        cov = rotation @ prepared_geometry.safe_cov[first:last] @ rotation.T
        bounds, front, back, numeric = _perspective_bounds(mu, cov, K, config.gaussian_sigma)
        numeric &= prepared_geometry.positive[first:last]
        flags['projectable_view'][first:last] = numeric
        in_image = numeric & (bounds[:,0] >= 0) & (bounds[:,1] <= w) & (bounds[:,2] >= 0) & (bounds[:,3] <= h)
        flags['fully_in_image'][first:last] = in_image
        overlaps = numeric & (bounds[:,1] >= 0) & (bounds[:,0] < w) & (bounds[:,3] >= 0) & (bounds[:,2] < h)
        rows = overlaps.nonzero().flatten()
        if not len(rows):
            continue
        bb = bounds[rows]
        # Clamp before integer conversion: a near-plane ellipse can have huge bounds.
        left, right = bb[:,0].clamp(0, w-1).floor().long(), bb[:,1].clamp(0, w-1).floor().long()
        top, bottom = bb[:,2].clamp(0, h-1).floor().long(), bb[:,3].clamp(0, h-1).floor().long()
        count, lower, upper = _query(prepared_depth, left, right, top, bottom)
        area = (right-left+1)*(bottom-top+1)
        fraction = count.to(torch.float64)/area
        eligible = in_image[rows] & (count >= config.minimum_valid_pixels) & (fraction >= config.minimum_valid_fraction)
        has_depth = count > 0
        free = eligible & (back[rows] < lower)
        support = has_depth & (front[rows] <= upper) & (back[rows] >= lower)
        occluded = has_depth & (front[rows] > upper)
        index = first+rows
        flags['free_view'][index] = free
        flags['possible_support_view'][index] = support
        flags['occluded_view'][index] = occluded
        flags['coverage_eligible'][index] = eligible
        coverage[index], valid_pixels[index], footprint_pixels[index] = fraction.float(), count, area
    flags['unknown_view'] = ~(flags['free_view'] | flags['possible_support_view'] | flags['occluded_view'])
    diagnostics = {key: int(value.sum().item()) for key, value in flags.items()}
    diagnostics.update(gaussian_rows=n, native_valid_pixels=int(prepared_depth.valid.sum().item()),
                       policy=asdict(config), maximum_geometry_batch=min(n, chunk_rows),
                       footprint='exact_perspective_ellipsoid_bounding_pixel_cells',
                       depth_query='conservative_dyadic_rectangle_superset',
                       depth_uncertainty=prepared_depth.interval_source,
                       support='possible_interval_support_not_measured_ray_intersection',
                       status='predicted_depth_evidence_requires_render_validation')
    return dict(**flags, valid_fraction=coverage, valid_pixels=valid_pixels,
                footprint_pixels=footprint_pixels, diagnostics=diagnostics)


def merge_station_views(view_evidence):
    """Collapse cube faces from ONE physical station; occlusion/unknown neutral.

    Caller groups by actual capture station, not by image/cube-face name.
    Any possible support vetoes this station's free vote and protects globally.
    """
    values = list(view_evidence)
    if not values:
        raise ValueError('At least one view is required')
    free = values[0]['free_view'].clone()
    support = values[0]['possible_support_view'].clone()
    if free.ndim != 1 or free.dtype != torch.bool or support.shape != free.shape or support.dtype != torch.bool:
        raise ValueError('Expected matching one-dimensional boolean view evidence')
    for item in values[1:]:
        f, s = item['free_view'], item['possible_support_view']
        if f.shape != free.shape or s.shape != free.shape or f.dtype != torch.bool or s.dtype != torch.bool or f.device != free.device or s.device != free.device:
            raise ValueError('View evidence shape, dtype or device differs')
        free |= f
        support |= s
    return dict(free=free & ~support, possible_support=support)


def select_dense_front(station_evidence, *, config=DenseFrontConfig()):
    """Aggregate an ID->evidence mapping of independent physical stations.

    Output is a proposed boolean removal mask, never in-place opacity editing.
    The station key must identify the physical capture position; six faces at
    the same capture must first be merged with ``merge_station_views``.
    """
    if not hasattr(station_evidence, 'items') or not station_evidence:
        raise ValueError('Use a nonempty mapping of physical station IDs to evidence')
    items = list(station_evidence.items())
    if len({str(key) for key, _ in items}) != len(items):
        raise ValueError('Physical station IDs must remain unique as strings')
    first = items[0][1]['free']
    if first.ndim != 1 or first.dtype != torch.bool:
        raise ValueError('Station flags must be one-dimensional boolean tensors')
    free_count = torch.zeros_like(first, dtype=torch.int32)
    support_count = torch.zeros_like(first, dtype=torch.int32)
    for _, item in items:
        free, support = item['free'], item['possible_support']
        if free.shape != first.shape or support.shape != first.shape or free.dtype != torch.bool or support.dtype != torch.bool or free.device != first.device or support.device != first.device:
            raise ValueError('Station evidence shape, dtype or device differs')
        if torch.any(free & support):
            raise ValueError('A station cannot simultaneously support and vote free')
        free_count += free
        support_count += support
    remove = (free_count >= config.minimum_free_stations) & (support_count == 0)
    return dict(remove=remove, free_station_count=free_count,
                possible_support_station_count=support_count,
                station_ids=tuple(str(key) for key, _ in items),
                diagnostics=dict(physical_stations=len(items), proposed=int(remove.sum().item()),
                                 policy=asdict(config), status='proposed_requires_render_validation'))
