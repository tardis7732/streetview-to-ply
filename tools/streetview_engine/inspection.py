"""Read-only, scale-aware Gaussian/SfM diagnostics; no cleanup or training.

CLI: python -m tools.streetview_engine.inspection --ply scene.ply --dataset
sfm/dataset --output inspection.json [--components sky_candidate/manifest.json]
[--ground]. Flags identify unverified review candidates, never junk or deletion
instructions. Sparse support is shared-SfM evidence, not independent truth.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

from .export import sha256, write_json


@dataclass(frozen=True)
class InspectionConfig:
    chunk_size: int = 65536
    workers: int = 4
    spacing_neighbors: int = 8
    minimum_spacing_neighbors: int = 3
    minimum_station_support: int = 2
    minimum_angle_degrees: float = 2.
    maximum_reprojection_px: float = 2.
    extent_sigma_multiple: float = 3.
    review_extent_baseline_ratio: float = .5
    review_axis_ratio: float = 100.
    review_support_spacing_ratio: float = 4.
    max_example_rows: int = 32
    maximum_baseline_stations: int = 1024
    ground: bool = False

    def __post_init__(self):
        for key, lower in (("chunk_size", 1), ("workers", 1), ("spacing_neighbors", 1), ("minimum_spacing_neighbors", 1),
                           ("minimum_station_support", 2), ("max_example_rows", 0), ("maximum_baseline_stations", 2)):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < lower:
                raise ValueError("Invalid inspection setting: " + key)
        if self.minimum_spacing_neighbors > self.spacing_neighbors or not isinstance(self.ground, bool):
            raise ValueError("Invalid spacing/ground configuration")
        for key in ("minimum_angle_degrees", "maximum_reprojection_px", "extent_sigma_multiple", "review_extent_baseline_ratio", "review_axis_ratio", "review_support_spacing_ratio"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid inspection setting: " + key)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def _bound(root, relative, bindings, expected=None):
    relative = Path(relative)
    path = (root/relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Missing or escaped inspection input: " + str(relative))
    actual = sha256(path)
    if expected is not None and actual != expected:
        raise ValueError("Inspection source hash changed: " + str(relative))
    bindings[relative.as_posix()] = actual
    return path


def _distribution(values):
    values = np.asarray(values, np.float64).ravel()
    valid = values[np.isfinite(values)]
    return dict(finite_count=int(len(valid)), unavailable_or_nonfinite=int(len(values)-len(valid)),
                quantiles=dict(zip(("min", "p01", "p10", "p50", "p90", "p99", "max"), map(float, np.quantile(valid, [0, .01, .1, .5, .9, .99, 1])))) if len(valid) else {})


def camera_baseline(frames, maximum_stations=1024):
    """Median inter-station distance; one centroid per physical station.

    Faces are averaged within a capture, then captures within a station solely
    for normalization. No camera pose is modified or forced to share a center.
    A deterministic ID-selected subset bounds pair storage for large datasets.
    """
    groups = {}
    for frame in frames:
        pose = np.asarray(frame["transform_matrix"], np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]) or not np.allclose(pose[:3, :3].T@pose[:3, :3], np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-6):
            raise ValueError("Invalid camera rigid pose")
        station = str(frame["station_id"])
        capture = str(frame.get("pano_id", frame["file_path"]))
        groups.setdefault(station, {}).setdefault(capture, []).append(pose[:3, 3])
    ids = sorted(groups, key=lambda value: hashlib.sha256(value.encode()).digest())[:maximum_stations]
    centers = np.asarray([np.mean([np.mean(values, axis=0) for values in groups[station].values()], axis=0) for station in ids])
    distances = pdist(centers) if len(centers) >= 2 else np.empty(0)
    positive = distances[np.isfinite(distances) & (distances > 0)]
    return dict(status="measured" if len(positive) else "insufficient_separated_stations", physical_station_count=len(groups),
                sampled_station_count=len(ids), sampled_station_ids=ids, positive_pair_count=len(positive),
                nonfinite_pair_count=int((~np.isfinite(distances)).sum()),
                metres=float(np.median(positive)) if len(positive) else None,
                method="median positive inter-physical-station centroid distance; capture/face-balanced; not a ground or height estimate")


def _header(path):
    """Strict standard float PLY layout; nonfinite payloads are reported later."""
    fields, count = [], None
    with Path(path).open("rb") as stream:
        if stream.readline().strip() != b"ply" or stream.readline().strip() != b"format binary_little_endian 1.0":
            raise ValueError("Inspection requires binary little-endian Gaussian PLY")
        while True:
            line = stream.readline(8192)
            if not line or stream.tell() > 65536:
                raise ValueError("Malformed or truncated PLY header")
            parts = line.decode("ascii").split()
            if not parts or parts[0] in ("comment", "obj_info"):
                continue
            if parts == ["end_header"]:
                break
            if len(parts) == 3 and parts[:2] == ["element", "vertex"] and count is None:
                count = int(parts[2])
            elif len(parts) == 3 and parts[0] == "property" and parts[1] in ("float", "float32") and parts[2] not in fields:
                fields.append(parts[2])
            else:
                raise ValueError("Unsupported Gaussian PLY declaration")
        offset = stream.tell()
    required = {"x", "y", "z", "opacity", *(f"scale_{i}" for i in range(3)), *(f"rot_{i}" for i in range(4)), *(f"f_dc_{i}" for i in range(3))}
    rest = [name for name in fields if name.startswith("f_rest_")]
    basis = len(rest)//3+1
    if count is None or count <= 0 or not required <= set(fields) or len(rest)%3 or set(rest) != {f"f_rest_{i}" for i in range(len(rest))} or math.isqrt(basis)**2 != basis or Path(path).stat().st_size != offset+count*len(fields)*4:
        raise ValueError("Gaussian PLY count/SH fields/payload size disagree")
    return count, fields, offset, math.isqrt(basis)-1


def _foreground_prefix(path, count, fields, offset, digest, component_manifest):
    """Only an explicitly declared, SHA-bound appended angular sky is excluded."""
    if component_manifest is None:
        return count, dict(status="not_declared", excluded_rows=0, interpretation="No sky inferred from Gaussian size, depth or color")
    manifest_path = Path(component_manifest).resolve(strict=True)
    manifest = _read(manifest_path)
    extra_bindings = {}
    published = manifest.get("quality", {}).get("sky_component")
    if published is not None:
        selection = manifest.get("selection", {})
        if manifest.get("status") != "completed" or manifest.get("artifact", {}).get("sha256") != digest or selection.get("accepted_model_sha256") != digest or published.get("kind") != "shared_angular_sky_fixed_geometry" or published.get("geometry_is_measured") is not False or published.get("foreground_statistics_exclude_sky") is not True:
            raise ValueError("Published sky component/model binding differs")
        comparison_path = _bound(manifest_path.parent, selection["comparison_report"], extra_bindings, selection["comparison_sha256"])
        identity = _read(comparison_path).get("foreground_identity", {})
        if published.get("row_start") != published.get("foreground_rows") or published.get("row_end_exclusive") != count or published.get("combined_rows") != count or identity.get("foreground_rows") != published.get("foreground_rows") or identity.get("sky_rows") != published.get("rows") or identity.get("source_sha256") != manifest.get("foreground_artifact", {}).get("sha256"):
            raise ValueError("Published sky component suffix/identity differs")
        manifest = dict(manifest, foreground_identity=identity, candidate_sha256=digest)
    identity = manifest.get("foreground_identity")
    if identity is not None:  # sky_refine.py's exact-prefix declaration
        if manifest.get("status") != "completed" or manifest.get("candidate_sha256") != digest or identity.get("candidate_sha256") != digest or identity.get("unchanged") is not True or identity.get("identity") != "exact_original_payload_prefix_including_all_attributes":
            raise ValueError("Sky component identity/model binding differs")
        start, sky_count = identity.get("foreground_rows"), identity.get("sky_rows")
        expected_prefix = identity.get("candidate_foreground_payload_sha256")
        if not expected_prefix or expected_prefix != identity.get("source_foreground_payload_sha256"):
            raise ValueError("Sky component foreground payload binding differs")
    else:
        components = manifest.get("components", [])
        if manifest.get("schema_version") != 1 or manifest.get("model_sha256") != digest or len(components) != 1 or components[0].get("role") != "shared_angular_sky_fixed_geometry":
            raise ValueError("Expected one explicit SHA-bound appended sky component")
        start, sky_count = components[0].get("start_row"), components[0].get("row_count")
        expected_prefix = None
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in (start, sky_count)) or start+sky_count != count or not sky_count:
        raise ValueError("Declared sky must be one nonempty appended suffix")
    if expected_prefix:
        hasher = hashlib.sha256()
        with Path(path).open("rb") as stream:
            stream.seek(offset)
            remaining = start*len(fields)*4
            while remaining:
                block = stream.read(min(remaining, 8*1024*1024))
                if not block:
                    raise ValueError("Truncated foreground payload")
                hasher.update(block)
                remaining -= len(block)
        if hasher.hexdigest() != expected_prefix:
            raise ValueError("Declared foreground prefix payload changed")
    return start, dict(status="explicit_appended_sky_excluded", manifest_path=str(manifest_path), manifest_sha256=sha256(manifest_path),
                       extra_files_sha256=extra_bindings, excluded_start_row=start, excluded_rows=sky_count, role="shared_angular_sky_fixed_geometry", quality_accepted_by_inspection=False)


def _seed_support(seeds, options):
    xyz = np.asarray(seeds["xyz"], np.float64)
    arrays = [np.asarray(seeds[key]) for key in ("support_station_count", "triangulation_angle_degrees", "reprojection_error_px")]
    if xyz.ndim != 2 or xyz.shape[1] != 3 or any(value.shape != (len(xyz),) for value in arrays) or not np.issubdtype(arrays[0].dtype, np.integer):
        raise ValueError("SfM seed support array contract differs")
    support, angle, error = arrays
    valid = np.isfinite(xyz).all(1) & np.isfinite(angle) & np.isfinite(error) & (support >= options.minimum_station_support) & (angle >= options.minimum_angle_degrees) & (error >= 0) & (error <= options.maximum_reprojection_px)
    points = np.unique(xyz[valid], axis=0)
    tree = cKDTree(points) if len(points) else None
    spacing = np.full(len(points), np.nan)
    if len(points) > options.minimum_spacing_neighbors:
        distances = tree.query(points, k=min(options.spacing_neighbors+1, len(points)), workers=options.workers)[0][:, 1:]
        spacing = np.median(distances, axis=1)
        spacing[~np.isfinite(spacing) | (spacing <= 0)] = np.nan
    report = dict(input_points=len(xyz), finite_points=int(np.isfinite(xyz).all(1).sum()), supported_points=int(valid.sum()), unique_supported_positions=len(points),
        support_station_count_semantics="SfM-exported TRAIN physical-station support, not cube-face counts or visibility at arbitrary views",
        local_spacing_m=_distribution(spacing), spacing_neighbors=options.spacing_neighbors,
        limitation="Euclidean nearest sparse-point spacing is a density normalization, not an observed surface resolution or depth/occlusion certificate")
    return tree, spacing, report


def inspect_arrays(chunks, seeds, baseline_m, *, options=None):
    """Chunked numeric core. Each chunk has row_ids/means/log_scales/quats/finite.

    Caller supplies foreground rows only. Never mutates input arrays. Stores
    O(N) scalar statistics and queries the seed tree in bounded chunks.
    """
    options = options or InspectionConfig()
    tree, spacing, support = _seed_support(seeds, options)
    measured = {key: [] for key in ("principal_min", "principal_middle", "principal_max", "axis_ratio", "extent_baseline_ratio", "nearest_support_m", "nearest_support_spacing_ratio")}
    flags = {key: dict(count=0, example_row_ids=[], classification="unverified_review_candidate") for key in ("large_extent", "high_axis_ratio", "far_from_sparse_support")}
    counts = dict(rows=0, finite_all_fields=0, finite_geometry_fields=0, valid_covariance_rows=0, zero_or_invalid_quaternion_rows=0, invalid_activated_scale_rows=0)
    baseline_valid = baseline_m is not None and np.isfinite(baseline_m) and baseline_m > 0
    for chunk in chunks:
        means, logs, quats = [np.asarray(chunk[key], np.float64) for key in ("means", "log_scales", "quats")]
        ids = np.asarray(chunk["row_ids"], np.int64)
        if means.shape != logs.shape or means.shape != (len(ids), 3) or quats.shape != (len(ids), 4) or np.asarray(chunk["finite"]).shape != (len(ids),):
            raise ValueError("Inspection Gaussian chunk shape mismatch")
        finite = np.isfinite(means).all(1) & np.isfinite(logs).all(1) & np.isfinite(quats).all(1)
        norm = np.linalg.norm(quats, axis=1)
        quat_valid = np.isfinite(norm) & (norm > 1e-8)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            scales = np.sort(np.exp(logs), axis=1)
            native_scales = np.exp(logs.astype(np.float32))
        scale_valid = np.isfinite(scales).all(1) & (scales > 0).all(1) & np.isfinite(native_scales).all(1) & (native_scales > 0).all(1)
        valid = finite & quat_valid & scale_valid
        counts["rows"] += len(ids)
        counts["finite_all_fields"] += int(np.asarray(chunk["finite"], bool).sum())
        counts["finite_geometry_fields"] += int(finite.sum())
        counts["valid_covariance_rows"] += int(valid.sum())
        counts["zero_or_invalid_quaternion_rows"] += int((~quat_valid).sum())
        counts["invalid_activated_scale_rows"] += int((~scale_valid).sum())
        s, xyz, ids = scales[valid], means[valid], ids[valid]
        with np.errstate(over="ignore", invalid="ignore"):
            ratio = s[:, 2]/s[:, 0]
        extent = 2*options.extent_sigma_multiple*s[:, 2]/baseline_m if baseline_valid else np.full(len(ids), np.nan)
        nearest = np.full(len(ids), np.nan)
        normalized = np.full(len(ids), np.nan)
        if tree is not None and len(ids):
            nearest, index = tree.query(xyz, k=1, workers=options.workers)
            supported = np.isfinite(nearest) & (index < len(spacing))
            normalized[supported] = nearest[supported]/spacing[index[supported]]
        values = dict(principal_min=s[:, 0], principal_middle=s[:, 1], principal_max=s[:, 2], axis_ratio=ratio,
                      extent_baseline_ratio=extent, nearest_support_m=nearest, nearest_support_spacing_ratio=normalized)
        for name, value in values.items():
            measured[name].append(value)
        for name, selected in (("large_extent", extent > options.review_extent_baseline_ratio), ("high_axis_ratio", ratio > options.review_axis_ratio), ("far_from_sparse_support", normalized > options.review_support_spacing_ratio)):
            flags[name]["count"] += int(selected.sum())
            remaining = max(0, options.max_example_rows-len(flags[name]["example_row_ids"]))
            flags[name]["example_row_ids"].extend(map(int, ids[selected][:remaining]))
    distributions = {name: _distribution(np.concatenate(value) if value else []) for name, value in measured.items()}
    return dict(counts=counts, distributions=distributions, sparse_support=support, review_candidates=flags,
                extent_definition="2*k*maximum principal standard deviation / robust physical-station baseline; finite envelope heuristic, not visibility",
                no_geometry_mutation=True, classification_policy="Flags are unverified. Large walls, sparse texture and uncertain poses can produce legitimate flagged rows. No removal or repair is authorized by these statistics.")


def ground_inventory(dataset, frames, seeds, baseline, bindings, manifest_files):
    """Count actual TRAIN observations inside hash-bound 3x3 ground cores.

    Never infer a plane, use heldout ground to certify training, or manufacture
    visibility by projecting unobserved seeds. Duplicate faces cannot increase
    physical-station votes. Neighbor support counts are diagnostic only.
    """
    from PIL import Image
    path = dataset/"sparse_depth_manifest.json"
    if not path.is_file():
        return dict(status="unassessed", reason="No actual-observation sidecar")
    metadata = _read(_bound(dataset, path.name, bindings, manifest_files.get(path.name)))
    if metadata.get("status") not in ("supported_observations", "insufficient_observations") or metadata.get("units") not in ("metres", "meters", "m") or metadata.get("observation_kind") != "actual_sfm_tracks" or metadata.get("geometry_scope") != "transductive_shared_sfm" or metadata.get("depth_convention") != "camera_z" or metadata.get("pixel_center_offset") != .5:
        raise ValueError("Ground inventory requires actual half-pixel SfM camera-Z tracks")
    for split in ("train", "heldout"):
        if metadata.get(f"transforms_{split}_sha256") != bindings[f"transforms_{split}.json"]:
            raise ValueError("Ground observation pose binding differs")
    if metadata.get("seed_npz_sha256") != bindings["init_points.npz"]:
        raise ValueError("Ground observation seed binding differs")
    source = _bound(dataset, metadata["npz"], bindings, metadata["sha256"])
    with np.load(source, allow_pickle=False) as archive:
        rows = {key: archive[key].copy() for key in ("frame_name", "station_id", "split", "point_id", "xy", "depth_z", "reprojection_error_px")}
    length = len(rows["point_id"])
    if any(len(value) != length for value in rows.values()) or rows["xy"].shape != (length, 2):
        raise ValueError("Malformed ground observation table")
    if not np.issubdtype(rows["point_id"].dtype, np.integer) or any(not np.isfinite(rows[key]).all() for key in ("xy", "depth_z", "reprojection_error_px")) or np.any(rows["depth_z"] <= 0) or np.any(rows["reprojection_error_px"] < 0):
        raise ValueError("Invalid actual ground observation values")
    roster = {frame["file_path"]: frame for frame in frames}
    for name, station, split in set(zip(rows["frame_name"].tolist(), rows["station_id"].tolist(), rows["split"].tolist())):
        if name not in roster or str(station) != str(roster[name]["station_id"]) or split != roster[name]["split"]:
            raise ValueError("Ground observation station/split differs from bound frame")
    valid_ground, observations, down_observations = [], 0, 0
    per_frame = []
    votes = {}
    for frame in frames:
        if frame["split"] != "train":
            continue
        indices = np.flatnonzero(rows["frame_name"] == frame["file_path"])
        # Select one actual row for each (frame, point), never count duplicates.
        indices = indices[np.argsort(rows["reprojection_error_px"][indices], kind="stable")]
        _, first = np.unique(rows["point_id"][indices], return_index=True)
        indices = indices[first]
        observations += len(indices)
        if str(frame.get("face", "")).lower() in ("d", "down"):
            down_observations += len(indices)
        if not frame.get("ground_mask_path") or not frame.get("ground_mask_sha256"):
            per_frame.append(dict(frame=frame["file_path"], status="ground_mask_not_supplied", observed_tracks=len(indices)))
            continue
        ground_path = _bound(dataset, frame["ground_mask_path"], bindings, frame["ground_mask_sha256"])
        static_path = _bound(dataset, frame["sfm_mask_path"], bindings, frame["sfm_mask_sha256"])
        with Image.open(ground_path) as im:
            ground = np.asarray(im.convert("L")) == 255
        with Image.open(static_path) as im:
            static = np.asarray(im.convert("L")) == 255
        if ground.shape != (frame["h"], frame["w"]) or static.shape != ground.shape:
            raise ValueError("Ground/static mask resolution differs")
        finite = np.isfinite(rows["xy"][indices]).all(1) & np.isfinite(rows["depth_z"][indices]) & (rows["depth_z"][indices] > 0)
        indices = indices[finite]
        xy = np.floor(rows["xy"][indices]).astype(int)
        inside = (xy[:, 0] >= 1) & (xy[:, 1] >= 1) & (xy[:, 0] < frame["w"]-1) & (xy[:, 1] < frame["h"]-1)
        indices, xy = indices[inside], xy[inside]
        valid = np.ones(len(indices), bool)
        mask = ground & static
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                valid &= mask[xy[:, 1]+dy, xy[:, 0]+dx]
        selected = indices[valid]
        valid_ground.extend(map(int, selected))
        for index in selected:
            votes.setdefault(int(rows["point_id"][index]), set()).add(str(frame["station_id"]))
        per_frame.append(dict(frame=frame["file_path"], station_id=str(frame["station_id"]), face=frame.get("face"), status="measured",
                              ground_mask_fraction=float(mask.mean()), observed_tracks=len(indices), ground_observed_tracks=len(selected)))
    point_ids = np.asarray(seeds["point_ids"])
    if point_ids.shape != (len(seeds["xyz"]),) or len(np.unique(point_ids)) != len(point_ids) or not set(votes) <= set(map(int, point_ids)):
        raise ValueError("Ground observation point IDs are not bound to unique seed IDs")
    two = np.array([len(votes.get(int(point), ())) >= 2 for point in point_ids])
    strong = two & (seeds["support_station_count"] >= 3) & (seeds["triangulation_angle_degrees"] >= 5) & (seeds["reprojection_error_px"] >= 0) & (seeds["reprojection_error_px"] <= 2) & np.isfinite(seeds["xyz"]).all(1)
    support_points = np.asarray(seeds["xyz"])[strong]
    neighbors = {}
    if len(support_points) and baseline["metres"]:
        tree = cKDTree(support_points)
        for ratio in (.025, .05, .1, .25):
            counts = tree.query_ball_point(support_points, ratio*baseline["metres"], return_length=True)
            neighbors[str(ratio)] = dict(max_points=int(counts.max()), centers_with_at_least_six_points=int((counts >= 6).sum()))
    return dict(status="measured", split="train", input_track_rows=length, unique_train_observations=observations,
        unique_train_down_observations=down_observations, train_ground_observations=len(valid_ground), train_ground_unique_points=len(votes),
        train_ground_points_two_physical_stations=int(two.sum()), strong_ground_points=int(strong.sum()),
        strong_criteria=dict(minimum_train_stations=3, minimum_angle_degrees=5, maximum_reprojection_px=2, minimum_ground_station_votes=2),
        strong_neighbor_counts_by_camera_baseline_radius=neighbors, frames=per_frame,
        interpretation="Actual TRAIN track + 3x3 semantic/static evidence only; ground class does not certify a flat surface or floor height; no plane inferred")


def inspect_scene(ply, dataset, *, components=None, config=None):
    options = config or InspectionConfig()
    ply, dataset = Path(ply).resolve(strict=True), Path(dataset).resolve(strict=True)
    bindings = {}
    manifest = _read(_bound(dataset, "dataset_manifest.json", bindings))
    if manifest.get("camera_convention") != "OpenGL_c2w" or manifest.get("units") not in ("metres", "meters", "m"):
        raise ValueError("Inspection dataset must declare OpenGL camera poses in metres")
    files = manifest.get("files", {})
    if not {"transforms_train.json", "transforms_heldout.json", "init_points.npz"} <= set(files):
        raise ValueError("Inspection requires hash-bound transforms and seed files")
    frames = []
    for split in ("train", "heldout"):
        path = _bound(dataset, f"transforms_{split}.json", bindings, files[f"transforms_{split}.json"])
        frames.extend(dict(frame, split=split) for frame in _read(path)["frames"])
    if len({f["file_path"] for f in frames}) != len(frames) or {str(f["station_id"]) for f in frames if f["split"] == "train"} & {str(f["station_id"]) for f in frames if f["split"] == "heldout"}:
        raise ValueError("Inspection frame roster duplicate or physical-station split leakage")
    baseline = camera_baseline(frames, options.maximum_baseline_stations)
    with np.load(_bound(dataset, "init_points.npz", bindings, files["init_points.npz"]), allow_pickle=False) as archive:
        seeds = {key: archive[key].copy() for key in ("xyz", "support_station_count", "triangulation_angle_degrees", "reprojection_error_px", "point_ids")}
    digest = sha256(ply)
    count, fields, offset, degree = _header(ply)
    foreground, component_report = _foreground_prefix(ply, count, fields, offset, digest, components)
    row_data = np.memmap(ply, mode="r", dtype="<f4", offset=offset, shape=(count, len(fields)))
    totals = dict(rows=count, finite_all_fields=0, nonfinite_rows=0)
    try:
        for start in range(0, count, options.chunk_size):
            totals["finite_all_fields"] += int(np.isfinite(row_data[start:start+options.chunk_size]).all(1).sum())
        totals["nonfinite_rows"] = count-totals["finite_all_fields"]
        indices = lambda names: [fields.index(name) for name in names]
        xyz_i = indices(("x", "y", "z"))
        scale_i = indices(f"scale_{i}" for i in range(3))
        quat_i = indices(f"rot_{i}" for i in range(4))
        def chunks():
            for start in range(0, foreground, options.chunk_size):
                stop = min(foreground, start+options.chunk_size)
                part = row_data[start:stop]
                yield dict(row_ids=np.arange(start, stop), means=part[:, xyz_i], log_scales=part[:, scale_i], quats=part[:, quat_i], finite=np.isfinite(part).all(1))
        result = inspect_arrays(chunks(), seeds, baseline["metres"], options=options)
    finally:
        row_data._mmap.close()
        del row_data
    ground = ground_inventory(dataset, frames, seeds, baseline, bindings, files) if options.ground else dict(status="not_requested")
    if sha256(ply) != digest or any(sha256(dataset/name) != expected for name, expected in bindings.items()):
        raise ValueError("Inspection input changed during read-only audit")
    if components is not None and sha256(components) != component_report["manifest_sha256"]:
        raise ValueError("Sky declaration changed during inspection")
    if components is not None and any(sha256(Path(components).resolve().parent/name) != expected for name, expected in component_report["extra_files_sha256"].items()):
        raise ValueError("Sky component identity evidence changed during inspection")
    return dict(schema_version=1, status="inspected", input_ply=dict(path=str(ply), sha256=digest, bytes=ply.stat().st_size, sh_degree=degree),
        dataset_root=str(dataset), dataset_files_sha256=bindings, inspection_code_sha256=sha256(__file__), config=asdict(options),
        metric_scale_evidence=manifest.get("metric_alignment", manifest.get("scale_evidence", "not_supplied")),
        all_rows=totals, declared_components=component_report, camera_baseline=baseline, foreground=result, ground_support=ground,
        quality_accepted=False, geometry_mutated=False, limitations=["Shared SfM support is transductive geometric evidence, not an independent sensor measurement.",
        "Unknown/occluded/untextured regions can be far from sparse seeds. Review flags cannot distinguish those from floaters.",
        "Explicit sky suffixes are excluded from foreground statistics; no other component is inferred from size or position."])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--components", type=Path)
    parser.add_argument("--ground", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output == args.ply.resolve() or output.is_relative_to(args.dataset.resolve()) or (args.components and output == args.components.resolve()):
        parser.error("Report output must be separate from the input PLY, dataset and component manifest")
    result = inspect_scene(args.ply, args.dataset, components=args.components, config=InspectionConfig(ground=args.ground, chunk_size=args.chunk_size))
    write_json(output, result)
    print(f"Read-only inspection: {result['all_rows']['rows']} rows; {result['foreground']['counts']['rows']} foreground; {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
