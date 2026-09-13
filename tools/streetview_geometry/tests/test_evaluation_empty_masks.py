"""A verified fully masked view is unmeasured, never perfect or disposable."""
import copy

import numpy as np
import pytest

from tools.streetview_engine.quality import masked_metrics
from tools.streetview_geometry.evaluation import validate_render_comparison


def fixture():
    rows = []
    roster = []
    rgb = np.full((2, 2, 3), .4, np.float32)
    prediction = np.full_like(rgb, .5)
    alpha = np.ones((2, 2), np.float32)
    for station in ('heldout_a', 'heldout_b'):
        for name, mask in (('side', np.ones((2, 2), bool)), ('sky', np.zeros((2, 2), bool))):
            binding = dict(frame=f'{station}/{name}', station_id=station,
                reference_rgb_sha256=f'photo-{station}-{name}', reference_mask_sha256=f'mask-{station}-{name}',
                static_pixels=int(mask.sum()))
            metrics = masked_metrics(prediction, rgb, alpha, mask)
            row = {key: binding[key] for key in ('frame', 'station_id', 'reference_rgb_sha256', 'reference_mask_sha256')}
            row.update(before=copy.deepcopy(metrics), after=copy.deepcopy(metrics), new_hole_pixels=0)
            rows.append(row)
            roster.append(binding)
    expected = dict(discovery_station_ids=['train'], evaluation_frames=roster)
    report = dict(source_sha256='source', candidate_sha256='candidate', discovery_station_ids=['train'],
        expected_evaluation_frames=[row['frame'] for row in rows], views=rows)
    return report, expected


def gate(report, expected):
    return validate_render_comparison(report, expected_manifest=expected,
        source_sha256='source', candidate_sha256='candidate')


def empty(row):
    for arm in ('before', 'after'):
        row[arm].update(static_pixels=0, static_sse=0., covered_pixels=0,
            masked_psnr_db=None, masked_l1=None, alpha_mean=None, coverage_fraction=None)
    row['new_hole_pixels'] = 0


def test_verified_empty_masks_keep_roster_but_never_inflate_scores_or_evidence():
    report, expected = fixture()
    result = gate(report, expected)
    assert result['accepted'], result
    assert result['pooled_psnr_loss_db'] == 0.
    assert result['measured_views'] == 2 and result['unmeasured_views'] == 2
    assert result['evaluation_stations'] == 2
    # Dropping truly empty frames from a separately trusted diagnostic fixture
    # must produce identical numeric scores, without making that omission legal
    # against the original immutable roster.
    measured_report, measured_expected = copy.deepcopy(report), copy.deepcopy(expected)
    measured_report['views'] = [row for row in measured_report['views'] if row['before']['static_pixels']]
    measured_report['expected_evaluation_frames'] = [row['frame'] for row in measured_report['views']]
    measured_expected['evaluation_frames'] = [row for row in measured_expected['evaluation_frames'] if row['static_pixels']]
    measured = gate(measured_report, measured_expected)
    for key in ('pooled_psnr_loss_db', 'worst_view_psnr_loss_db', 'worst_new_hole_fraction', 'evaluation_stations'):
        assert result[key] == measured[key]
    assert not gate(measured_report, expected)['accepted']


def test_report_cannot_make_a_bad_view_unmeasured():
    report, expected = fixture()
    empty(report['views'][0])
    result = gate(report, expected)
    assert not result['accepted']
    assert 'reference_mask_pixel_count_mismatch' in result['reasons']


def test_legacy_manifest_without_pixel_counts_rejects_unverified_empty_claims():
    report, expected = fixture()
    for binding in expected['evaluation_frames']:
        binding.pop('static_pixels')
    result = gate(report, expected)
    assert not result['accepted']
    assert 'unverified_empty_reference_mask' in result['reasons']


@pytest.mark.parametrize('change', [
    lambda row: row['before'].update(static_pixels=1),
    lambda row: row['after'].update(static_pixels=1),
    lambda row: row['before'].update(static_pixels=False),
    lambda row: row['after'].update(static_pixels=0.),
    lambda row: row['before'].update(static_sse=1.),
    lambda row: row['after'].update(static_sse=1.),
    lambda row: row['before'].update(static_sse=False),
    lambda row: row.update(new_hole_pixels=1),
    lambda row: row.update(new_hole_pixels=False),
    lambda row: row['before'].update(covered_pixels=1),
    lambda row: row['after'].update(covered_pixels=False),
    lambda row: row['after'].update(masked_psnr_db=120.),
    lambda row: row['before'].update(masked_l1=0.),
    lambda row: row['after'].update(alpha_mean=0.),
    lambda row: row['before'].update(coverage_fraction=0.),
])
def test_empty_views_require_matching_zero_counts_errors_holes_and_unmeasured_means(change):
    report, expected = fixture()
    change(report['views'][1])
    assert not gate(report, expected)['accepted']


@pytest.mark.parametrize('value', [-1, 0., False, None, '0', 1])
def test_empty_counts_must_be_independently_trusted_nonnegative_integers(value):
    report, expected = fixture()
    expected['evaluation_frames'][1]['static_pixels'] = value
    assert not gate(report, expected)['accepted']


@pytest.mark.parametrize('change,reason', [
    (lambda row: row.update(reference_rgb_sha256='other'), 'reference_binding_mismatch'),
    (lambda row: row.update(reference_mask_sha256='other'), 'reference_binding_mismatch'),
    (lambda row: row.update(station_id='other'), 'frame_station_binding_mismatch'),
    (lambda row: row.update(station_id='train'), 'physical_station_holdout_leakage'),
])
def test_empty_views_still_require_full_hash_station_and_split_binding(change, reason):
    report, expected = fixture()
    change(report['views'][1])
    result = gate(report, expected)
    assert not result['accepted'] and reason in result['reasons']


def test_empty_only_station_does_not_satisfy_minimum_physical_station_evidence():
    report, expected = fixture()
    empty(report['views'][2])
    expected['evaluation_frames'][2]['static_pixels'] = 0
    result = gate(report, expected)
    assert not result['accepted']
    assert 'insufficient_evaluation_stations' in result['reasons']
    assert result['evaluation_stations'] == 1 and result['evaluation_roster_stations'] == 2


def test_all_empty_evaluation_fails_without_zero_error_or_perfect_score_claim():
    report, expected = fixture()
    for row, binding in zip(report['views'], expected['evaluation_frames']):
        empty(row)
        binding['static_pixels'] = 0
    result = gate(report, expected)
    assert not result['accepted']
    assert 'no_measured_evaluation_pixels' in result['reasons']
    assert result['evaluation_stations'] == 0 and result['unmeasured_views'] == 4
    for key in ('pooled_psnr_loss_db', 'worst_view_psnr_loss_db', 'worst_new_hole_fraction'):
        assert result[key] is None


def test_empty_views_cannot_hide_regression_in_remaining_measured_view():
    report, expected = fixture()
    report['views'][0]['after']['static_sse'] *= 2
    result = gate(report, expected)
    assert not result['accepted']
    assert 'local_rgb_regression' in result['reasons']
