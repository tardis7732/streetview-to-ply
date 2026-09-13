"""Conservative free-space decisions from independent physical observations.

No visibility is unknown. A sample behind another surface is occluded/neutral.
The caller must generate footprint evidence using calibrated cameras and static
masks; counts alone cannot certify depth quality or camera calibration.
"""
from dataclasses import asdict, dataclass
import numpy as np


def _boolean_evidence(value):
    array = np.asarray(value)
    if array.dtype == np.bool_:
        return array
    if not np.issubdtype(array.dtype, np.integer) or not np.isin(array, [0, 1]).all():
        raise ValueError('Evidence flags must be bool or integer 0/1; NaN and probabilities are not votes')
    return array.astype(bool)


@dataclass(frozen=True)
class FreeSpaceConfig:
    min_free_stations: int = 3
    min_sparse_stations: int = 3
    min_footprint_samples: int = 3
    relative_depth_margin: float = .10
    uncertainty_sigma: float = 3.
    gaussian_sigma: float = 3.

    def __post_init__(self):
        if any(not isinstance(v, (int, np.integer)) or isinstance(v, bool) for v in (self.min_free_stations, self.min_sparse_stations, self.min_footprint_samples)):
            raise ValueError('Observation minima must be integer counts')
        if self.min_free_stations < 2 or not 0 <= self.min_sparse_stations <= self.min_free_stations:
            raise ValueError('Free-space policy needs independent stations and a valid sparse minimum')
        if self.min_footprint_samples < 1:
            raise ValueError('Footprint evidence must contain samples')
        if not np.isfinite([self.relative_depth_margin, self.uncertainty_sigma, self.gaussian_sigma]).all() or min(self.relative_depth_margin, self.uncertainty_sigma, self.gaussian_sigma) < 0:
            raise ValueError('Margins must be finite and nonnegative')


def classify_samples(center_z, sigma_z, surface_z, surface_sigma, valid, *, config=FreeSpaceConfig()):
    """Classify one calibrated footprint sample using intervals in the same units.

Missing/invalid samples return all false. Interval overlap supports a possible
surface and vetoes removal; this favors retaining uncertain geometry.
"""
    values = np.broadcast_arrays(*[np.asarray(v, dtype=np.float64) for v in (center_z, sigma_z, surface_z, surface_sigma)])
    center, spread, depth, uncertainty = values
    valid = np.broadcast_to(_boolean_evidence(valid), center.shape).copy()
    valid &= np.logical_and.reduce([np.isfinite(v) for v in values])
    valid &= (center > 0) & (depth > 0) & (spread >= 0) & (uncertainty >= 0)
    margin = config.relative_depth_margin * depth + config.uncertainty_sigma * uncertainty
    front = center - config.gaussian_sigma * spread
    back = center + config.gaussian_sigma * spread
    free = valid & (back < depth - margin)
    behind = valid & (front > depth + margin)
    support = valid & ~free & ~behind
    return dict(free=free, support=support, behind=behind, valid=valid)


def aggregate_footprint_samples(free, support, sparse, valid, sample_station_ids, *, config=FreeSpaceConfig()):
    """Collapse [row,sample] evidence to [row,physical-station] votes.

Samples must be distinct depth pixels after upstream deduplication. Duplicate
pixel coordinates need to be removed by the calibrated evidence producer.
Any surface support or valid nonfree sample within a station vetoes its free vote.
"""
    arrays = [_boolean_evidence(a) for a in (free, support, sparse, valid)]
    if any(a.ndim != 2 or a.shape != arrays[0].shape for a in arrays):
        raise ValueError('All sample arrays must share [rows,samples] shape')
    free, support, sparse, valid = arrays
    groups = np.asarray(list(map(str, sample_station_ids)))
    if groups.shape != (free.shape[1],):
        raise ValueError('Station IDs must describe sample columns')
    if np.any((free | support) & ~valid) or np.any(free & support):
        raise ValueError('Contradictory sample classifications')
    unique = np.unique(groups)
    fs = np.zeros((free.shape[0], len(unique)), bool)
    ss = fs.copy(); ps = fs.copy()
    for column, group in enumerate(unique):
        mask = groups == group
        fs[:, column] = (valid[:, mask].sum(axis=1) >= config.min_footprint_samples) & ((free[:, mask] | ~valid[:, mask]).all(axis=1))
        ps[:, column] = support[:, mask].any(axis=1)
        ss[:, column] = fs[:, column] & (sparse[:, mask] & free[:, mask]).any(axis=1)
    return dict(station_ids=unique, free=fs, support=ps, sparse_free=ss)


def select_free_space(row_ids, station_ids, free, support, sparse_free, *, config=FreeSpaceConfig()):
    """Propose rows for counterfactual rendering; never delete a PLY here."""
    rows = np.asarray(row_ids)
    if not np.issubdtype(rows.dtype, np.integer):
        raise ValueError('PLY row IDs must have integer dtype; fractional indices are invalid')
    rows = rows.astype(np.int64, copy=False)
    groups = tuple(map(str, station_ids))
    free, support, sparse = [_boolean_evidence(v) for v in (free, support, sparse_free)]
    shape = (len(rows), len(groups))
    if rows.ndim != 1 or len(np.unique(rows)) != len(rows) or np.any(rows < 0):
        raise ValueError('Unique nonnegative current PLY row IDs are required')
    if len(groups) != len(set(groups)):
        raise ValueError('Evidence columns must be independent physical stations, not cube faces')
    if any(v.shape != shape for v in (free, support, sparse)):
        raise ValueError('Evidence matrix shape differs from row/station metadata')
    if np.any(sparse & ~free) or np.any(support & free):
        raise ValueError('Inconsistent station evidence')
    fc, sc, pc = free.sum(axis=1), sparse.sum(axis=1), support.sum(axis=1)
    accepted = (fc >= config.min_free_stations) & (sc >= config.min_sparse_stations) & (pc == 0)
    return dict(indices=rows[accepted], free_station_count=fc[accepted], sparse_station_count=sc[accepted],
        diagnostics=dict(policy=asdict(config), rows_evaluated=len(rows), physical_stations=len(groups),
            proposed=int(accepted.sum()), surface_supported=int((pc > 0).sum()),
            insufficient_evidence=int(((pc == 0) & ~accepted).sum()),
            status='proposed_requires_independent_render_validation'))
