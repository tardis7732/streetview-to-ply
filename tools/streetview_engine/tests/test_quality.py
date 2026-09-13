import copy
import numpy as np
import pytest

from tools.streetview_engine.quality import compare_candidate, compare_depth_consistency, depth_metrics, geometry_statistics, masked_metrics, paired_new_holes, summarize_views


def test_metrics_actual_static_pixels_coverage_and_masks():
    reference = np.zeros((2, 2, 3), np.float32)
    prediction = np.ones_like(reference) * .1
    prediction[0, 0] = 1
    mask = np.array([[False, True], [True, True]])
    alpha = np.array([[0, .8], [.4, .9]])
    row = masked_metrics(prediction, reference, alpha, mask)
    assert row["static_pixels"] == 3
    assert row["static_sse"] == pytest.approx(.09)
    assert row["masked_psnr_db"] == pytest.approx(20)
    assert row["coverage_fraction"] == pytest.approx(2/3)
    summary = summarize_views([dict(row, frame="a", station_id="physical_a", sky=row)])
    assert summary["physical_stations"] == 1 and summary["sky"]["static_pixels"] == 3


def reports():
    refs = [dict(frame=f"view{i}", station_id=f"heldout{i}", reference_rgb_sha256=f"rgb{i}", reference_mask_sha256=f"mask{i}") for i in range(2)]
    before = dict(model_sha256="source", views=[dict(row, static_pixels=100, static_sse=3., covered_pixels=100) for row in refs])
    after = dict(model_sha256="candidate", views=[dict(row, static_pixels=100, static_sse=2., covered_pixels=100) for row in refs])
    roster = dict(discovery_station_ids=["train"], evaluation_frames=refs)
    evidence = dict(status="passed", source_sha256="source", candidate_sha256="candidate", independent_multi_station=True,
                    heldout_geometry_improved=True, artifact_sha256="separately-bound-depth-evidence")
    return before, after, roster, evidence


def test_better_rgb_alone_cannot_certify_geometry_candidate():
    before, after, roster, evidence = reports()
    result = compare_candidate(before, after, expected_manifest=roster, new_hole_pixels={"view0":0, "view1":0})
    assert result["appearance"]["accepted"] and not result["accepted"] and result["selected"] == "baseline"
    accepted = compare_candidate(before, after, expected_manifest=roster, new_hole_pixels={"view0":0, "view1":0}, geometry_evidence=evidence)
    assert accepted["accepted"]


@pytest.mark.parametrize("kind", ["hole", "missing", "station", "source_hash"])
def test_candidate_gate_rejects_actual_regression_or_unbound_reference(kind):
    before, after, roster, evidence = reports()
    holes = {"view0":0, "view1":0}
    if kind == "hole":
        holes["view0"] = 2
    elif kind == "missing":
        after["views"].pop()
    elif kind == "station":
        after["views"][0]["station_id"] = "train"
    elif kind == "source_hash":
        before["views"][0]["reference_rgb_sha256"] = "different"
    result = compare_candidate(before, after, expected_manifest=roster, new_hole_pixels=holes, geometry_evidence=evidence)
    assert not result["accepted"]


def test_large_particle_diagnostic_is_scale_invariant_not_deletion():
    means = np.array([[0, 0, 0], [1, 1, 1]], float)
    scales = np.array([[.1, .1, .1], [2, 2, 2]])
    a = geometry_statistics(means, scales, np.array([.5, .6]), scene_radius_m=10)
    b = geometry_statistics(means*20 + 100, scales*20, np.array([.5, .6]), scene_radius_m=200)
    assert a["large_sigma_count"] == b["large_sigma_count"] == 1
    assert not a["cleanup_applied"] and a["surface_depth"]["status"] == "unassessed"


def test_new_holes_come_from_paired_float_alpha(tmp_path):
    before, after, _, _ = reports()
    left, right = tmp_path / "a", tmp_path / "b"
    left.mkdir(); right.mkdir()
    for index in range(2):
        name = f"alpha{index}.npz"
        before["views"][index]["alpha_npz"] = name
        after["views"][index]["alpha_npz"] = name
        np.savez(left/name, alpha=np.ones((2, 2)), mask=np.ones((2, 2), bool))
        np.savez(right/name, alpha=np.array([[.1, 1], [1, 1]]), mask=np.ones((2, 2), bool))
    assert paired_new_holes(left, right, before, after) == {"view0":1, "view1":1}


def test_depth_metric_detects_spread_hidden_by_correct_mean_and_is_scale_invariant():
    target, alpha = np.full((2, 2), 5.), np.ones((2, 2))
    before = depth_metrics(target, np.full((2, 2), 27.), alpha, target, np.ones((2, 2), bool))
    scaled = depth_metrics(target*7, np.full((2, 2), 27.)*49, alpha, target*7, np.ones((2, 2), bool))
    assert before["relative_target_rmse"] == 0
    assert before["relative_center_depth_variance"] == pytest.approx(.08)
    assert scaled["relative_center_depth_variance"] == pytest.approx(before["relative_center_depth_variance"])
    missing = depth_metrics(np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), target, np.ones((2, 2), bool))
    assert missing["relative_expected_squared_error"] == 1 and missing["coverage_fraction"] == 0


def depth_reports():
    reports = []
    for name, second in (("source", 27.), ("candidate", 25.5)):
        metric = depth_metrics(np.array([5.]), np.array([second]), np.ones(1), np.array([5.]), np.ones(1, bool))
        reports.append(dict(model_sha256=name, split="heldout", used_for_optimization=False,
            geometry_scope="transductive_shared_sfm", optimization_station_ids=["train"], observation_manifest_sha256="bound-observations",
            views=[dict(frame=f"view{i}", station_id=f"heldout{i}", target_roster_sha256=f"target{i}", **metric) for i in range(2)]))
    return reports


def depth_roster(report):
    return dict(observation_manifest_sha256=report["observation_manifest_sha256"],
                evaluation_frames=[{key: row[key] for key in ("frame", "station_id", "target_roster_sha256", "target_pixels")} for row in report["views"]])


def test_transductive_consistency_can_accept_with_full_rgb_nohole_guard():
    depth_before, depth_after = depth_reports()
    compared = compare_depth_consistency(depth_before, depth_after, expected_manifest=depth_roster(depth_before))
    assert compared["accepted"] and compared["relative_improvement"] == pytest.approx(.75)
    assert "not_independent_geometry_truth" in compared["scope"]
    before, after, roster, _ = reports()
    roster["depth_evaluation"] = depth_roster(depth_before)
    result = compare_candidate(before, after, expected_manifest=roster, new_hole_pixels={"view0":0, "view1":0},
                               depth_baseline=depth_before, depth_candidate=depth_after)
    assert result["accepted"]


@pytest.mark.parametrize("kind", ["leak", "roster", "holes", "worse"])
def test_depth_consistency_rejects_leakage_missing_targets_or_coverage(kind):
    before, after = depth_reports()
    if kind == "leak":
        after["used_for_optimization"] = True
    elif kind == "roster":
        after["views"][0]["target_roster_sha256"] = "different"
    elif kind == "holes":
        after["views"][0]["covered_pixels"] = 0
    else:
        after["views"][0]["relative_moment_sum"] = 100
    assert not compare_depth_consistency(before, after, expected_manifest=depth_roster(before))["accepted"]


def test_depth_consistency_needs_trusted_roster_and_cannot_trade_bias_for_spread():
    before, after = depth_reports()
    assert not compare_depth_consistency(before, after)["accepted"]
    metric = depth_metrics(np.array([6.]), np.array([36.]), np.ones(1), np.array([5.]), np.ones(1, bool))
    for row in after["views"]:
        row.update(metric)
    compared = compare_depth_consistency(before, after, expected_manifest=depth_roster(before))
    assert not compared["accepted"] and "heldout_depth_target_error_regression" in compared["reasons"]
