"""Scene-independent candidate-selection holdout gates with explicit binding."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class EvaluationConfig:
    max_pooled_psnr_loss_db: float = .10
    max_single_view_psnr_loss_db: float = .50
    max_new_hole_fraction: float = .001
    min_evaluation_stations: int = 2

    def __post_init__(self):
        values = [self.max_pooled_psnr_loss_db, self.max_single_view_psnr_loss_db, self.max_new_hole_fraction]
        if not all(math.isfinite(v) and v >= 0 for v in values) or self.max_new_hole_fraction > 1 or self.min_evaluation_stations < 1:
            raise ValueError('Invalid evaluation policy')


def validate_render_comparison(report, *, source_sha256, candidate_sha256, expected_manifest=None, config=EvaluationConfig()):
    """Accept only complete, bound, disjoint selection-holdout observations.

This gate is an appearance regression guard, not a proof of geometry correctness
or cross-location performance. A candidate can only be accepted for a pipeline
stage if its separate geometric/semantic evidence also passed.
"""
    problems = []
    if expected_manifest is None:
        return dict(accepted=False, reasons=['missing_independently_supplied_evaluation_manifest'],
            policy=asdict(config), scope='candidate_selection_holdout_appearance_guard_not_geometry_or_cross_location_proof')
    # This manifest must come from the pipeline's immutable scene/split inputs,
    # not from a list the render report can shorten after seeing bad results.
    roster = expected_manifest.get('evaluation_frames', [])
    trusted_names = [str(row.get('frame', '')) for row in roster]
    if not roster or '' in trusted_names or len(trusted_names) != len(set(trusted_names)):
        problems.append('invalid_trusted_evaluation_roster')
    bindings = {str(row.get('frame', '')): row for row in roster}
    if report.get('source_sha256') != source_sha256 or report.get('candidate_sha256') != candidate_sha256:
        problems.append('artifact_hash_mismatch')
    expected = set(trusted_names)
    discovery = set(map(str, expected_manifest.get('discovery_station_ids', [])))
    if not expected or not discovery:
        problems.append('missing_split_or_frame_manifest')
    if set(map(str, report.get('expected_evaluation_frames', []))) != expected:
        problems.append('reported_roster_differs_from_trusted_manifest')
    if set(map(str, report.get('discovery_station_ids', []))) != discovery:
        problems.append('reported_split_differs_from_trusted_manifest')
    rows = report.get('views', [])
    names = [str(v.get('frame', '')) for v in rows]
    if len(names) != len(set(names)) or set(names) != expected:
        problems.append('incomplete_or_duplicate_evaluation')
    groups = {str(v.get('station_id', '')) for v in rows}
    if '' in groups or groups & discovery:
        problems.append('physical_station_holdout_leakage')
    before_sum, after_sum, pixels = 0., 0., 0
    worst_loss = 0.; worst_holes = 0.
    measured_groups = set()
    measured_views = 0; unmeasured_views = 0
    for row in rows:
        binding = bindings.get(str(row.get('frame', '')))
        if binding is None:
            problems.append('unexpected_evaluation_frame')
        elif str(row.get('station_id', '')) != str(binding.get('station_id', '')):
            problems.append('frame_station_binding_mismatch')
        if binding is not None:
            for key in ('reference_rgb_sha256', 'reference_mask_sha256'):
                if not binding.get(key) or row.get(key) != binding[key]:
                    problems.append('reference_binding_mismatch')
        before, after = row.get('before', {}), row.get('after', {})
        keys = ('static_sse', 'static_pixels')
        if any(not isinstance(metric, dict) or any(key not in metric for key in keys) for metric in (before, after)):
            problems.append('missing_metrics'); continue
        count = before['static_pixels']
        bse, ase = before['static_sse'], after['static_sse']
        new_holes = row.get('new_hole_pixels')
        if type(count) is not int or count < 0 or type(after['static_pixels']) is not int or count != after['static_pixels'] or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0 for v in (bse, ase)) or type(new_holes) is not int or not 0 <= new_holes <= count:
            problems.append('invalid_or_incomparable_metrics'); continue
        if not row.get('reference_rgb_sha256') or not row.get('reference_mask_sha256'):
            problems.append('unbound_reference_inputs')
        # A report must not erase an inconvenient view by merely claiming an
        # empty mask. New manifests bind counts computed from immutable input
        # masks before rendering. Legacy positive-count reports remain valid.
        if binding is not None and 'static_pixels' in binding:
            trusted_count = binding['static_pixels']
            if type(trusted_count) is not int or trusted_count < 0 or trusted_count != count:
                problems.append('reference_mask_pixel_count_mismatch'); continue
        if count == 0:
            if binding is None or type(binding.get('static_pixels')) is not int or binding['static_pixels'] != 0:
                problems.append('unverified_empty_reference_mask'); continue
            zero_errors = bse == 0 and ase == 0 and new_holes == 0
            zero_coverage = all('covered_pixels' not in metric or
                (type(metric['covered_pixels']) is int and metric['covered_pixels'] == 0) for metric in (before, after))
            unmeasured = all(metric.get(key) is None for metric in (before, after)
                for key in ('masked_psnr_db', 'masked_l1', 'alpha_mean', 'coverage_fraction'))
            if not zero_errors or not zero_coverage or not unmeasured:
                problems.append('invalid_empty_mask_metrics'); continue
            unmeasured_views += 1
            continue
        measured_views += 1
        measured_groups.add(str(row.get('station_id', '')))
        before_sum += bse; after_sum += ase; pixels += count
        loss = 10 * math.log10(max(ase, count * 1e-12) / max(bse, count * 1e-12))
        worst_loss = max(worst_loss, loss)
        worst_holes = max(worst_holes, new_holes / count)
    pooled_loss = 10 * math.log10(max(after_sum, pixels * 1e-12) / max(before_sum, pixels * 1e-12)) if pixels else None
    if len(measured_groups) < config.min_evaluation_stations:
        problems.append('insufficient_evaluation_stations')
    if not pixels:
        problems.append('no_measured_evaluation_pixels')
    if pooled_loss is not None and pooled_loss > config.max_pooled_psnr_loss_db:
        problems.append('pooled_rgb_regression')
    if worst_loss > config.max_single_view_psnr_loss_db:
        problems.append('local_rgb_regression')
    if worst_holes > config.max_new_hole_fraction:
        problems.append('new_static_holes')
    return dict(accepted=not problems, reasons=sorted(set(problems)), policy=asdict(config),
        pooled_psnr_loss_db=pooled_loss, worst_view_psnr_loss_db=worst_loss if pixels else None,
        worst_new_hole_fraction=worst_holes if pixels else None, evaluation_stations=len(measured_groups),
        evaluation_roster_stations=len(groups), measured_views=measured_views, unmeasured_views=unmeasured_views,
        scope='candidate_selection_holdout_appearance_guard_not_geometry_or_cross_location_proof')
