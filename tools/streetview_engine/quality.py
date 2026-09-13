"""Measured appearance/coverage and evidence-gated model comparisons.

Statistics about large Gaussians are descriptive. They never authorize row
deletion, and RGB agreement alone never certifies surface geometry.
"""
import math
from pathlib import Path

import numpy as np

from tools.streetview_geometry.evaluation import EvaluationConfig, validate_render_comparison


def _psnr(sse, count):
    return -10 * math.log10(max(float(sse) / (3 * count), 1e-12)) if count else None


def masked_metrics(prediction, reference, alpha, mask, *, alpha_threshold=.5):
    prediction, reference = np.asarray(prediction), np.asarray(reference)
    alpha, mask = np.asarray(alpha), np.asarray(mask, bool)
    if prediction.shape != reference.shape or prediction.shape != mask.shape + (3,) or alpha.shape != mask.shape:
        raise ValueError("RGB, alpha and mask shapes differ")
    if not 0 <= alpha_threshold <= 1:
        raise ValueError("Invalid alpha threshold")
    if not all(np.isfinite(x).all() for x in (prediction, reference, alpha)):
        raise ValueError("Nonfinite rendered/reference values")
    if np.any((reference < 0) | (reference > 1)) or np.any((alpha < -1e-5) | (alpha > 1 + 1e-5)):
        raise ValueError("Expected normalized sRGB and alpha")
    count = int(mask.sum())
    # Quantization-independent display-space comparison, also used by exports.
    delta = (np.clip(prediction, 0, 1).astype(np.float64) - reference.astype(np.float64))[mask]
    sse = float(np.square(delta).sum())
    return dict(static_pixels=count, static_sse=sse, masked_psnr_db=_psnr(sse, count),
                masked_l1=float(np.abs(delta).mean()) if count else None,
                alpha_mean=float(alpha[mask].mean()) if count else None,
                covered_pixels=int((mask & (alpha >= alpha_threshold)).sum()),
                coverage_fraction=float((alpha[mask] >= alpha_threshold).mean()) if count else None,
                alpha_threshold=alpha_threshold)


def summarize_views(views):
    def aggregate(rows):
        count = sum(row["static_pixels"] for row in rows)
        sse = sum(row["static_sse"] for row in rows)
        covered = sum(row["covered_pixels"] for row in rows)
        return dict(static_pixels=count, static_sse=sse, pooled_psnr_db=_psnr(sse, count),
                    coverage_fraction=covered / count if count else None, views=len(rows))
    groups = {}
    for view in views:
        groups.setdefault(str(view["station_id"]), []).append(view)
    stations = {key: aggregate(rows) for key, rows in sorted(groups.items())}
    station_values = [x["pooled_psnr_db"] for x in stations.values() if x["pooled_psnr_db"] is not None]
    result = aggregate(views)
    result.update(stations=stations, physical_stations=len(stations),
                  station_mean_psnr_db=float(np.mean(station_values)) if station_values else None,
                  worst_view=min(views, key=lambda x: x["masked_psnr_db"] if x["masked_psnr_db"] is not None else math.inf)["frame"] if views else None)
    for category in ("foreground", "sky", "down"):
        rows = [row[category] for row in views if category in row and row[category]["static_pixels"]]
        result[category] = aggregate(rows)
    return result


def geometry_statistics(means, scales, opacities, *, scene_radius_m, large_sigma_fraction=.05):
    means, scales, opacities = map(np.asarray, (means, scales, opacities))
    if means.ndim != 2 or means.shape[1] != 3 or scales.shape != means.shape or opacities.shape != (len(means),):
        raise ValueError("Invalid Gaussian array shapes")
    if not len(means) or not all(np.isfinite(x).all() for x in (means, scales, opacities)) or np.any(scales <= 0):
        raise ValueError("Invalid Gaussian parameters")
    if not math.isfinite(scene_radius_m) or scene_radius_m <= 0 or not math.isfinite(large_sigma_fraction) or large_sigma_fraction <= 0:
        raise ValueError("Invalid scene-normalized diagnostic threshold")
    max_sigma = scales.max(axis=1)
    return dict(gaussians=len(means), bounds_edn_m=[means.min(0).tolist(), means.max(0).tolist()],
                principal_sigma_m={str(q): float(np.percentile(max_sigma, q)) for q in (50, 95, 99, 100)},
                large_sigma_fraction=large_sigma_fraction,
                large_sigma_count=int((max_sigma / scene_radius_m > large_sigma_fraction).sum()),
                opacity_mean=float(opacities.mean()),
                cleanup_applied=False, interpretation="descriptive_scale_statistics_not_floater_classification",
                surface_depth=dict(status="unassessed", reason="No independent validated surface depth supplied"),
                free_space=dict(status="unassessed", reason="No independent multi-station free-space witnesses supplied"))


def depth_metrics(first_moment, second_moment, alpha, target_z, valid, confidence=None, *, alpha_threshold=.5):
    """Camera-Z center-layer spread and target agreement on a fixed target set.

    Relative statistics are invariant to uniform scene-unit changes. Missing
    rays get a unit relative target error and a separate coverage count; they
    cannot look improved merely because their moment mass faded to zero.
    """
    m1, m2, a, target = [np.asarray(x, np.float64) for x in (first_moment, second_moment, alpha, target_z)]
    valid = np.asarray(valid, bool)
    if len({x.shape for x in (m1, m2, a, target, valid)}) != 1:
        raise ValueError("Depth moment/target shapes differ")
    if not all(np.isfinite(x).all() for x in (m1, m2, a)) or np.any(a < -1e-6) or not 0 <= alpha_threshold <= 1:
        raise ValueError("Invalid rendered depth moments")
    valid = valid & np.isfinite(target) & (target > 0)
    weights = np.ones_like(target) if confidence is None else np.asarray(confidence, np.float64)
    if weights.shape != target.shape or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Invalid depth confidence")
    valid &= weights > 0
    count = int(valid.sum())
    if not count:
        return dict(target_pixels=0, confidence_sum=0., relative_error_sum=0., relative_variance_sum=0., relative_moment_sum=0., covered_pixels=0)
    z, opacity, w = target[valid], a[valid], weights[valid]
    mean = m1[valid]/np.maximum(opacity, 1e-8)
    variance = np.maximum(m2[valid]/np.maximum(opacity, 1e-8)-mean**2, 0)
    observed = opacity > 1e-6
    error = np.where(observed, ((mean-z)/z)**2, 1.)
    spread = np.where(observed, variance/z**2, 0.)
    return dict(target_pixels=count, confidence_sum=float(w.sum()), relative_error_sum=float((w*error).sum()),
                relative_variance_sum=float((w*spread).sum()), relative_moment_sum=float((w*(error+spread)).sum()),
                covered_pixels=int((opacity >= alpha_threshold).sum()), alpha_threshold=alpha_threshold,
                relative_target_rmse=float(np.sqrt((w*error).sum()/w.sum())),
                relative_center_depth_variance=float((w*spread).sum()/w.sum()),
                relative_expected_squared_error=float((w*(error+spread)).sum()/w.sum()),
                coverage_fraction=float((opacity >= alpha_threshold).mean()))


def summarize_depth_views(views):
    def aggregate(rows):
        count = sum(x["target_pixels"] for x in rows)
        weight = sum(x["confidence_sum"] for x in rows)
        sums = {key: sum(x[key] for x in rows) for key in ("relative_error_sum", "relative_variance_sum", "relative_moment_sum", "covered_pixels")}
        return dict(target_pixels=count, confidence_sum=weight, **sums,
                    relative_target_rmse=math.sqrt(sums["relative_error_sum"]/weight) if weight else None,
                    relative_center_depth_variance=sums["relative_variance_sum"]/weight if weight else None,
                    relative_expected_squared_error=sums["relative_moment_sum"]/weight if weight else None,
                    coverage_fraction=sums["covered_pixels"]/count if count else None)
    groups = {}
    for row in views:
        groups.setdefault(str(row["station_id"]), []).append(row)
    return dict(**aggregate(views), physical_stations=len(groups), stations={key: aggregate(rows) for key, rows in sorted(groups.items())})


def compare_depth_consistency(baseline, candidate, *, min_relative_improvement=.01,
                              max_station_relative_regression=.05, max_coverage_loss=.001, min_stations=2,
                              expected_manifest=None, max_relative_target_mse_increase=1e-4):
    """Bound transductive heldout-observation consistency, not surveyed truth."""
    thresholds = (min_relative_improvement, max_station_relative_regression, max_coverage_loss, max_relative_target_mse_increase)
    if any(not math.isfinite(x) or x < 0 for x in thresholds) or min_relative_improvement >= 1 or max_coverage_loss > 1 or min_stations < 1:
        raise ValueError("Invalid depth consistency policy")
    reasons = []
    if expected_manifest is None:
        return dict(accepted=False, reasons=["missing_trusted_depth_target_roster"], scope="transductive_heldout_observation_consistency_not_independent_geometry_truth")
    trusted = {row["frame"]: row for row in expected_manifest.get("evaluation_frames", [])}
    if not trusted or len(trusted) != len(expected_manifest.get("evaluation_frames", [])):
        reasons.append("invalid_trusted_depth_target_roster")
    for report in (baseline, candidate):
        if report.get("split") != "heldout" or report.get("used_for_optimization") is not False or report.get("geometry_scope") != "transductive_shared_sfm":
            reasons.append("invalid_geometry_validation_scope")
        groups = {str(row["station_id"]) for row in report.get("views", [])}
        if groups & set(map(str, report.get("optimization_station_ids", []))) or len(groups) < min_stations:
            reasons.append("insufficient_or_leaked_physical_station_holdout")
    if not baseline.get("observation_manifest_sha256") or baseline.get("observation_manifest_sha256") != candidate.get("observation_manifest_sha256"):
        reasons.append("observation_manifest_mismatch")
    if expected_manifest.get("observation_manifest_sha256") != baseline.get("observation_manifest_sha256"):
        reasons.append("trusted_observation_manifest_mismatch")
    before = {row["frame"]: row for row in baseline.get("views", [])}
    after = {row["frame"]: row for row in candidate.get("views", [])}
    if not before or set(before) != set(after) or len(before) != len(baseline.get("views", [])) or len(after) != len(candidate.get("views", [])):
        reasons.append("depth_target_roster_mismatch")
    if set(before) != set(trusted) or set(after) != set(trusted):
        reasons.append("trusted_depth_target_roster_mismatch")
    for name in before.keys() & after.keys():
        a, b = before[name], after[name]
        if any(a.get(key) != b.get(key) for key in ("station_id", "target_roster_sha256", "target_pixels", "confidence_sum")) or not a.get("target_roster_sha256"):
            reasons.append("depth_target_binding_mismatch")
        if name not in trusted or any(a.get(key) != trusted[name].get(key) for key in ("station_id", "target_roster_sha256", "target_pixels")):
            reasons.append("trusted_depth_target_binding_mismatch")
        for row in (a, b):
            if row.get("target_pixels", 0) <= 0 or row.get("confidence_sum", 0) <= 0 or any(not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key]) or row[key] < 0 for key in ("relative_error_sum", "relative_variance_sum", "relative_moment_sum", "covered_pixels")):
                reasons.append("invalid_depth_statistics")
    if reasons:
        return dict(accepted=False, reasons=sorted(set(reasons)), scope="transductive_heldout_observation_consistency_not_independent_geometry_truth")
    a, b = summarize_depth_views(list(before.values())), summarize_depth_views(list(after.values()))
    improvement = 1 - b["relative_expected_squared_error"]/max(a["relative_expected_squared_error"], 1e-12)
    if improvement < min_relative_improvement:
        reasons.append("no_heldout_geometric_consistency_improvement")
    if b["relative_error_sum"]/b["confidence_sum"] > a["relative_error_sum"]/a["confidence_sum"] + max_relative_target_mse_increase:
        reasons.append("heldout_depth_target_error_regression")
    if a["coverage_fraction"]-b["coverage_fraction"] > max_coverage_loss:
        reasons.append("heldout_depth_coverage_regression")
    for station in a["stations"]:
        before_station, after_station = a["stations"][station], b["stations"][station]
        if after_station["relative_expected_squared_error"] > max(before_station["relative_expected_squared_error"], 1e-12)*(1+max_station_relative_regression):
            reasons.append("local_geometric_consistency_regression")
        if after_station["relative_error_sum"]/after_station["confidence_sum"] > before_station["relative_error_sum"]/before_station["confidence_sum"] + max_relative_target_mse_increase:
            reasons.append("local_depth_target_error_regression")
        if before_station["coverage_fraction"]-after_station["coverage_fraction"] > max_coverage_loss:
            reasons.append("local_depth_coverage_regression")
    return dict(accepted=not reasons, reasons=sorted(set(reasons)), relative_improvement=improvement,
                before=a, after=b, source_sha256=baseline.get("model_sha256"), candidate_sha256=candidate.get("model_sha256"),
                observation_manifest_sha256=baseline["observation_manifest_sha256"],
                scope="transductive_heldout_observation_consistency_not_independent_geometry_truth")


def compare_candidate(baseline, candidate, *, expected_manifest, new_hole_pixels,
                      geometry_evidence=None, depth_baseline=None, depth_candidate=None, policy=EvaluationConfig()):
    """Compare full actual heldout render reports, with a separate geometry gate.

    Caller computes per-view newly uncovered pixels from paired alpha maps.
    Geometry evidence must be independently produced and bound to both PLYs;
    a smaller file, mean depth, or an oversized-splat threshold is insufficient.
    """
    source_sha, candidate_sha = baseline["model_sha256"], candidate["model_sha256"]
    before = {row["frame"]: row for row in baseline["views"]}
    after = {row["frame"]: row for row in candidate["views"]}
    if len(before) != len(baseline["views"]) or len(after) != len(candidate["views"]):
        raise ValueError("Duplicate render frame")
    report = dict(source_sha256=source_sha, candidate_sha256=candidate_sha,
                  expected_evaluation_frames=[row["frame"] for row in expected_manifest["evaluation_frames"]],
                  discovery_station_ids=expected_manifest["discovery_station_ids"], views=[])
    for name, row in after.items():
        if name not in before:
            continue
        report["views"].append(dict(frame=name, station_id=row["station_id"], before=before[name], after=row,
            reference_rgb_sha256=row["reference_rgb_sha256"], reference_mask_sha256=row["reference_mask_sha256"],
            new_hole_pixels=new_hole_pixels.get(name)))
    appearance = validate_render_comparison(report, source_sha256=source_sha, candidate_sha256=candidate_sha,
                                            expected_manifest=expected_manifest, config=policy)
    baseline_references_match = all(name in before and all(before[name].get(key) == row.get(key)
        for key in ("station_id", "reference_rgb_sha256", "reference_mask_sha256")) for name, row in after.items())
    if not baseline_references_match or set(before) != set(after):
        appearance["accepted"] = False
        appearance["reasons"].append("baseline_reference_or_roster_mismatch")
    evidence = geometry_evidence or {}
    geometry_ok = (evidence.get("status") == "passed" and evidence.get("source_sha256") == source_sha
                   and evidence.get("candidate_sha256") == candidate_sha and evidence.get("independent_multi_station") is True
                   and evidence.get("heldout_geometry_improved") is True and bool(evidence.get("artifact_sha256")))
    if depth_baseline is not None or depth_candidate is not None:
        if depth_baseline is None or depth_candidate is None:
            raise ValueError("Both baseline and candidate depth reports are required")
        evidence = compare_depth_consistency(depth_baseline, depth_candidate, expected_manifest=expected_manifest.get("depth_evaluation"))
        geometry_ok = evidence["accepted"] and depth_baseline.get("model_sha256") == source_sha and depth_candidate.get("model_sha256") == candidate_sha
    return dict(accepted=bool(appearance["accepted"] and geometry_ok), appearance=appearance,
                geometry=dict(status="passed" if geometry_ok else "unassessed_or_rejected", evidence=evidence),
                selected="candidate" if appearance["accepted"] and geometry_ok else "baseline",
                scope="bound_holdout_guard_plus_independent_geometry_evidence_not_cross_location_proof")


def paired_new_holes(baseline_directory, candidate_directory, baseline, candidate, *, threshold=.5):
    """Compute actual coverage losses from saved float alpha maps and masks."""
    if not 0 <= threshold <= 1:
        raise ValueError("Invalid alpha threshold")
    before = {row["frame"]: row for row in baseline["views"]}
    result = {}
    for row in candidate["views"]:
        if row["frame"] not in before:
            continue
        arrays = []
        for root, entry in ((Path(baseline_directory), before[row["frame"]]), (Path(candidate_directory), row)):
            path = (root / entry["alpha_npz"]).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("Alpha artifact escapes run directory")
            with np.load(path, allow_pickle=False) as data:
                arrays.append((data["alpha"].copy(), data["mask"].astype(bool)))
        (a, mask_a), (b, mask_b) = arrays
        if a.shape != b.shape or not np.array_equal(mask_a, mask_b):
            raise ValueError("Coverage references differ")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Nonfinite alpha")
        result[row["frame"]] = int((mask_a & (a >= threshold) & (b < threshold)).sum())
    return result
