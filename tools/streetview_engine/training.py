"""Portable seed-to-Gaussian gsplat stage with physical-station holdout.

CUDA/Torch/gsplat are imported only by run(). This module starts no GPU work on
import. Camera geometry and metric alignment are supplied by the SfM stage.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .export import sha256, write_json, write_model, validate_ply
from .quality import depth_metrics, geometry_statistics, masked_metrics, summarize_depth_views, summarize_views
from .processing_options import read_processing_options


def _now():
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class TrainingSettings:
    steps: int = 6000
    resolution: int = 768
    max_splats: int = 1000000
    seed: int = 42
    strategy: str = "mcmc"
    opacity_reg: float = .01
    scale_reg: float = .01
    ssim_weight: float = .2
    learning_rate_scale: float = 1.
    noise_lr: float = 500000.
    checkpoint_every: int = 1000
    log_every: int = 100
    cpu_workers: int = 4
    sh_degree: int = 2
    alpha_threshold: float = .5
    large_sigma_fraction: float = .05
    depth_moment_weight: float = 0.
    depth_coverage_weight: float = .01
    depth_resolution: int = 384
    depth_start_fraction: float = .1
    depth_ramp_fraction: float = .1
    depth_max_reprojection_px: float = 2.
    depth_min_support: int = 2
    depth_min_angle_degrees: float = 2.
    depth_collision_relative_tolerance: float = .02
    depth_min_pixels: int = 1
    depth_use_validated_prior: bool = False
    depth_prior_relative_weight: float = .25
    device: str = "cuda"

    def __post_init__(self):
        if self.strategy not in ("mcmc", "3dgs"):
            raise ValueError("Invalid training strategy; expected mcmc or 3dgs")
        for name, lower, upper in (("steps", 1, 1000000), ("resolution", 16, 8192), ("max_splats", 4, 10000000),
                                   ("checkpoint_every", 1, 1000000), ("log_every", 1, 1000000), ("cpu_workers", 1, 128), ("sh_degree", 0, 3),
                                   ("depth_resolution", 16, 8192), ("depth_min_support", 2, 10000), ("depth_min_pixels", 1, 100000000)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError("Invalid training setting: " + name)
        for name in ("opacity_reg", "scale_reg", "noise_lr", "ssim_weight", "alpha_threshold", "depth_moment_weight", "depth_coverage_weight", "depth_start_fraction", "depth_ramp_fraction", "depth_min_angle_degrees", "depth_collision_relative_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid training setting: " + name)
        if self.ssim_weight > 1 or self.alpha_threshold > 1 or not math.isfinite(self.learning_rate_scale) or self.learning_rate_scale <= 0 or not math.isfinite(self.large_sigma_fraction) or self.large_sigma_fraction <= 0:
            raise ValueError("Invalid training weights/thresholds")
        if self.device != "cuda" or isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("Training requires device=cuda and a uint32 seed")
        if self.depth_start_fraction > 1 or not 0 < self.depth_ramp_fraction <= 1 or self.depth_min_angle_degrees > 180 or not math.isfinite(self.depth_max_reprojection_px) or self.depth_max_reprojection_px <= 0 or self.depth_collision_relative_tolerance > 1:
            raise ValueError("Invalid sparse-depth settings")
        if self.depth_moment_weight > 0 and self.depth_coverage_weight <= 0:
            raise ValueError("Depth moments require positive fixed-reference coverage weight")
        if not isinstance(self.depth_use_validated_prior, bool) or isinstance(self.depth_prior_relative_weight, bool) or not math.isfinite(self.depth_prior_relative_weight) or not 0 < self.depth_prior_relative_weight <= 1:
            raise ValueError("Invalid validated depth-prior settings")

    @classmethod
    def from_inputs(cls, config, settings):
        values = {key: config[source] for key, source in (("steps", "training_steps"), ("resolution", "resolution"), ("max_splats", "max_splats")) if source in config}
        overrides = settings.get("training", {})
        if not isinstance(overrides, dict) or set(overrides) - {item.name for item in fields(cls)}:
            raise ValueError("Unknown training settings")
        values.update(overrides)
        return cls(**values)


def create_strategy(options, params, optimizers):
    """Construct the native algorithm; regularization remains an explicit setting.

    DefaultStrategy uses normalized coordinates (scene_scale=1), classic signed
    gradients and its native thresholds. This opt-in comparison does not change
    the MCMC schedule, parameter activation, optimizer rates or photometric loss.
    """
    factor = options.steps / 30000
    schedule = dict(refine_start_iter=max(1, round(500*factor)),
                    refine_every=max(1, round(100*factor)), verbose=False)
    if options.strategy == "mcmc":
        from gsplat import MCMCStrategy
        strategy = MCMCStrategy(cap_max=options.max_splats, noise_lr=options.noise_lr,
            refine_stop_iter=max(2, round(25000*factor)), **schedule)
        state = strategy.initialize_state()
    else:
        from gsplat import DefaultStrategy
        strategy = DefaultStrategy(refine_stop_iter=max(2, round(15000*factor)),
            reset_every=max(1, round(3000*factor)), refine_scale2d_stop_iter=0, **schedule)
        state = strategy.initialize_state(scene_scale=1.)
        state["_streetview_reset_compatibility"] = default_reset_compatibility(strategy)
    strategy.check_sanity(params, optimizers)
    return strategy, state


def default_reset_compatibility(strategy):
    """Recognize the upstream reset-condition defect without guessing by version.

    A fixed implementation runs only its own reset. The exact known malformed
    AST runs a positive-cadence reset after the unmodified native callback. An
    unknown implementation fails initialization instead of risking two resets.
    """
    import ast
    import inspect
    import textwrap
    source = textwrap.dedent(inspect.getsource(type(strategy).step_post_backward))
    tree = ast.parse(source)
    def is_reset(node):
        return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "reset_opa"
    calls = [node for node in ast.walk(tree) if is_reset(node)]
    conditions = [node.test for node in ast.walk(tree) if isinstance(node, ast.If)
                  and any(isinstance(stmt, ast.Expr) and is_reset(stmt.value) for stmt in node.body)]
    if len(calls) != 1 or len(conditions) != 1:
        raise RuntimeError("Unrecognized native DefaultStrategy opacity reset implementation")
    condition = ast.dump(conditions[0], include_attributes=False)
    def signature(expression):
        return ast.dump(ast.parse(expression, mode="eval").body, include_attributes=False)
    broken = signature("step % self.reset_every == 0 & step > 0")
    fixed = {signature(expression) for expression in (
        "step % self.reset_every == 0 and step > 0",
        "step > 0 and step % self.reset_every == 0",
        "(step % self.reset_every == 0) & (step > 0)",
        "(step > 0) & (step % self.reset_every == 0)")}
    if condition != broken and condition not in fixed:
        raise RuntimeError("Unrecognized native DefaultStrategy opacity reset condition")
    return dict(apply_missing_reset=condition == broken,
                native_post_backward_sha256=hashlib.sha256(source.encode("utf8")).hexdigest(),
                recognition="exact_reset_condition_AST", resets_applied=0)


def strategy_pre_backward(options, strategy, params, optimizers, state, step, info):
    # Only DefaultStrategy requires retaining the packed means2d gradient. Leave
    # the historical MCMC call sequence unchanged.
    if options.strategy == "3dgs":
        strategy.step_pre_backward(params, optimizers, state, step, info)


def strategy_post_backward(options, strategy, params, optimizers, state, step, info, *, lr):
    if options.strategy == "mcmc":
        strategy.step_post_backward(params, optimizers, state, step, info, lr=lr)
        return
    # Native v1.5.3: with 2D-radius growth disabled, small/large gradient masks
    # are disjoint. Duplicating adds one row; splitting replaces one with two.
    # Hence one native event produces <=2N rows before opacity/scale pruning.
    # This bounds row count, not temporary allocator/optimizer memory overhead.
    if strategy.refine_scale2d_stop_iter != 0:
        raise ValueError("3dgs budget proof requires native 2D-radius growth disabled")
    before = len(params["means"])
    if before > options.max_splats:
        raise RuntimeError("3dgs max_splats budget exceeded before native refinement")
    guarded = before > options.max_splats // 2
    thresholds = strategy.grow_grad2d, strategy.grow_scale2d
    try:
        if guarded:
            strategy.grow_grad2d = strategy.grow_scale2d = math.inf
        strategy.step_post_backward(params, optimizers, state, step, info, packed=True)
    finally:
        strategy.grow_grad2d, strategy.grow_scale2d = thresholds
    compatibility = state["_streetview_reset_compatibility"]
    if compatibility["apply_missing_reset"] and 0 < step < strategy.refine_stop_iter and step % strategy.reset_every == 0:
        from gsplat.strategy.ops import reset_opa
        reset_opa(params=params, optimizers=optimizers, state=state, value=strategy.prune_opa*2.)
        compatibility["resets_applied"] += 1
    after = len(params["means"])
    if after > options.max_splats or after > (before if guarded else 2*before):
        raise RuntimeError("3dgs native growth violated max_splats/2N budget; no rows were arbitrarily removed")
    if not after:
        raise RuntimeError("3dgs native pruning removed every Gaussian")


def _file(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Missing dataset file or escaped path: " + str(relative))
    return path


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def validate_splits(train, heldout):
    if not train:
        raise ValueError("No training frames")
    seen = set()
    for frame in train + heldout:
        if frame.get("station_id") is None or isinstance(frame.get("station_id"), bool) or not str(frame.get("station_id", "")):
            raise ValueError("Frame lacks physical station_id")
        if frame["file_path"] in seen:
            raise ValueError("Duplicate frame across dataset splits")
        seen.add(frame["file_path"])
        c2w = np.asarray(frame["transform_matrix"], np.float64)
        if c2w.shape != (4, 4) or not np.isfinite(c2w).all() or not np.allclose(c2w[3], [0, 0, 0, 1], atol=1e-7) or not np.allclose(c2w[:3, :3].T @ c2w[:3, :3], np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(c2w[:3, :3]), 1, atol=1e-5):
            raise ValueError("Invalid OpenGL c2w pose")
        for key in ("w", "h", "fl_x", "fl_y", "cx", "cy"):
            if isinstance(frame[key], bool) or not np.isfinite(frame[key]) or (key in ("w", "h", "fl_x", "fl_y") and frame[key] <= 0):
                raise ValueError("Invalid frame calibration")
    training_stations = {str(frame["station_id"]) for frame in train}
    heldout_stations = {str(frame["station_id"]) for frame in heldout}
    if training_stations & heldout_stations:
        raise ValueError("Physical-station holdout leakage")
    return training_stations, heldout_stations


def validate_dataset_manifest(manifest, training_stations, heldout_stations):
    if manifest.get("coordinate_frame") != "EDN" or manifest.get("units") not in ("metres", "meters", "m") or manifest.get("camera_convention") != "OpenGL_c2w":
        raise ValueError("Dataset must explicitly declare OpenGL c2w in EDN metres")
    colors = set(map(str, manifest.get("seed_color_station_ids", [])))
    if manifest.get("seed_colors_exclude_heldout") is not True or not colors or not colors <= training_stations or colors & heldout_stations:
        raise ValueError("Seed colors must be explicitly bound to training-only physical stations")








def normalization(frames):
    # Equal physical-station weight; different capture rigs retain their own
    # actual c2w translations and are never forced into a common camera center.
    groups = {}
    for frame in frames:
        groups.setdefault(str(frame["station_id"]), []).append(np.asarray(frame["transform_matrix"], np.float64)[:3, 3])
    centers = np.asarray([np.mean(x, axis=0) for x in groups.values()])
    center = centers.mean(axis=0)
    radius = float(np.linalg.norm(centers - center, axis=1).max() * 1.1)
    if not np.isfinite(radius) or radius <= 1e-8:
        raise ValueError("Training requires multiple spatially separated physical stations")
    return center, radius


def station_uniform_schedule(frames, steps, seed):
    groups = {}
    for index, frame in enumerate(frames):
        groups.setdefault(str(frame["station_id"]), []).append(index)
    if not groups or steps < 1:
        raise ValueError("Sampling requires frames and positive steps")
    ordered = [groups[key] for key in sorted(groups)]
    rng = np.random.default_rng(seed)
    stations = rng.integers(0, len(ordered), size=steps)
    return np.asarray([ordered[group][int(rng.integers(len(ordered[group])))] for group in stations], np.int64)


def sparse_observation_maps(rows, frame, foreground_mask, options):
    """Rasterize actual observations once at native target resolution.

    No projected-only visibility assumptions or sparse-map upsampling. Pixels
    containing materially different observed depths are excluded conservatively.
    """
    original_w, original_h = int(frame["w"]), int(frame["h"])
    if np.asarray(foreground_mask).shape != (original_h, original_w):
        raise ValueError("Sparse depth foreground mask/calibration mismatch")
    factor = min(1., options.depth_resolution / max(original_w, original_h), options.resolution / max(original_w, original_h))
    w, h = max(1, round(original_w*factor)), max(1, round(original_h*factor))
    xy, z = np.asarray(rows["xy"], np.float64), np.asarray(rows["depth_z"], np.float64)
    support, angle, error = [np.asarray(rows[key]) for key in ("support_station_count", "triangulation_angle_degrees", "reprojection_error_px")]
    point_ids = np.asarray(rows["point_id"], np.int64)
    count = len(z)
    if xy.shape != (count, 2) or any(x.shape != (count,) for x in (support, angle, error, point_ids)):
        raise ValueError("Sparse depth observation row shapes differ")
    if not np.issubdtype(support.dtype, np.integer):
        raise ValueError("Physical station support must be integer counts")
    valid_rows = np.isfinite(xy).all(1) & np.isfinite(z) & (z > 0) & np.isfinite(angle) & np.isfinite(error)
    valid_rows &= (support >= options.depth_min_support) & (angle >= options.depth_min_angle_degrees) & (error >= 0) & (error <= options.depth_max_reprojection_px)
    pixel = np.floor(np.where(np.isfinite(xy), xy, -2)).astype(np.int64)
    valid_rows &= (pixel[:, 0] >= 1) & (pixel[:, 0] < original_w-1) & (pixel[:, 1] >= 1) & (pixel[:, 1] < original_h-1)
    selected = np.flatnonzero(valid_rows)
    for dx, dy in ((x, y) for x in (-1, 0, 1) for y in (-1, 0, 1)):
        selected = selected[np.asarray(foreground_mask, bool)[pixel[selected, 1]+dy, pixel[selected, 0]+dx]]
    uv = np.floor(xy[selected] * np.array([w/original_w, h/original_h])).astype(np.int64)
    flat = uv[:, 1]*w + uv[:, 0]
    depth = np.zeros((h, w), np.float32)
    confidence = np.zeros((h, w), np.float32)
    ids = np.full((h, w), -1, np.int64)
    order = np.argsort(flat, kind="stable")
    flat, selected = flat[order], selected[order]
    ambiguous = 0
    ambiguous_mask = np.zeros((h, w), bool)
    for group in np.split(np.arange(len(flat)), np.flatnonzero(np.diff(flat))+1) if len(flat) else []:
        candidates = selected[group]
        local_z = z[candidates]
        if (local_z.max()-local_z.min())/local_z.min() > options.depth_collision_relative_tolerance:
            ambiguous += 1
            ambiguous_mask.flat[int(flat[group[0]])] = True
            continue
        chosen = candidates[np.argmin(local_z)]
        p = int(flat[group[0]])
        depth.flat[p] = z[chosen]
        confidence.flat[p] = min(float(support[chosen])/3, 1.) * min(float(angle[chosen])/10, 1.) * math.exp(-.5*(float(error[chosen])/options.depth_max_reprojection_px)**2)
        ids.flat[p] = point_ids[chosen]
    valid = confidence > 0
    roster = hashlib.sha256(ids[valid].astype("<i8").tobytes() + np.flatnonzero(valid).astype("<i8").tobytes() + depth[valid].astype("<f4").tobytes()).hexdigest()
    return dict(depth_z=depth, confidence=confidence, valid=valid, point_ids=ids, width=w, height=h,
                target_roster_sha256=roster, actual_observation_rows=count, accepted_pixels=int(valid.sum()),
                ambiguous_collision_pixels=ambiguous, ambiguous_mask=ambiguous_mask)


def load_sparse_depth(dataset, train_frames, heldout_frames, options, *, train_max_resolution=None):
    """Load SHA-bound actual SfM observations, keeping optimization/eval apart."""
    from PIL import Image
    path = dataset / "sparse_depth_manifest.json"
    if not path.is_file():
        if options.depth_moment_weight:
            raise ValueError("Depth candidate requires actual-observation sparse_depth_manifest.json")
        return {}, dict(status="unassessed", reason="No actual SfM observation sidecar")
    manifest = _read(path)
    if manifest.get("status") == "insufficient_observations" and not options.depth_moment_weight:
        return {}, dict(status="unassessed", manifest_sha256=sha256(path), reason="SfM reports insufficient actual observations")
    expected = dict(status="supported_observations", coordinate_frame="EDN", depth_convention="camera_z",
                    observation_kind="actual_sfm_tracks", geometry_scope="transductive_shared_sfm")
    if any(manifest.get(key) != value for key, value in expected.items()) or manifest.get("units") not in ("metres", "meters", "m") or manifest.get("pixel_center_offset") != .5:
        raise ValueError("Sparse depth provenance/convention is incompatible")
    for split in ("train", "heldout"):
        if manifest.get(f"transforms_{split}_sha256") != sha256(dataset / f"transforms_{split}.json"):
            raise ValueError("Sparse depth camera transform binding differs")
    train_ids, heldout_ids = validate_splits(train_frames, heldout_frames)
    if set(map(str, manifest.get("train_station_ids", []))) != train_ids or set(map(str, manifest.get("heldout_station_ids", []))) != heldout_ids:
        raise ValueError("Sparse depth physical-station split differs")
    npz = _file(dataset, manifest["npz"])
    if sha256(npz) != manifest.get("sha256"):
        raise ValueError("Sparse depth observation artifact changed")
    if manifest.get("seed_npz_sha256") != sha256(dataset / "init_points.npz"):
        raise ValueError("Sparse depth seed provenance differs")
    keys = ("frame_name", "station_id", "split", "point_id", "xy", "depth_z", "support_station_count", "triangulation_angle_degrees", "reprojection_error_px")
    with np.load(npz, allow_pickle=False) as data:
        rows = {key: data[key].copy() for key in keys}
    count = len(rows["depth_z"])
    if any(len(value) != count for value in rows.values()):
        raise ValueError("Sparse depth table row lengths differ")
    roster = {frame["file_path"]: (frame, split) for split, frames in (("train", train_frames), ("heldout", heldout_frames)) for frame in frames}
    if not set(map(str, rows["frame_name"])) <= set(roster):
        raise ValueError("Sparse depth refers to an unregistered dataset frame")
    result = {}
    totals = {split: dict(pixels=0, frames=0) for split in ("train", "heldout")}
    for name in sorted(set(map(str, rows["frame_name"]))):
        frame, split = roster[name]
        chosen = rows["frame_name"].astype(str) == name
        if set(map(str, rows["station_id"][chosen])) != {str(frame["station_id"])} or set(map(str, rows["split"][chosen])) != {split}:
            raise ValueError("Sparse depth row station/split mismatch")
        foreground_path = frame.get("foreground_mask_path") or frame.get("sfm_mask_path")
        if not foreground_path:
            raise ValueError("Sparse depth requires a separate static foreground mask")
        with Image.open(_file(dataset, foreground_path)) as image:
            foreground = np.asarray(image.convert("L")) == 255
        frame_options = replace(options, depth_resolution=min(options.depth_resolution, train_max_resolution)) if split == "train" and train_max_resolution is not None else options
        mapped = sparse_observation_maps({key: value[chosen] for key, value in rows.items()}, frame, foreground, frame_options)
        if not mapped["accepted_pixels"] and (split == "heldout" or not mapped["ambiguous_collision_pixels"]):
            continue
        mapped.update(split=split, frame=name, station_id=str(frame["station_id"]))
        result[name] = mapped
        totals[split]["pixels"] += mapped["accepted_pixels"]
        totals[split]["frames"] += 1
    if options.depth_moment_weight and totals["train"]["pixels"] < options.depth_min_pixels:
        raise ValueError("Insufficient accepted TRAIN depth observations; candidate cannot silently skip depth")
    return result, dict(status="supported_observations", manifest_sha256=sha256(path), observation_sha256=sha256(npz),
                        totals=totals, geometry_scope="transductive_shared_sfm", validation_scope="heldout_observation_consistency_not_independent_sensor_truth",
                        optimization_source="actual observed TRAIN foreground pixels only", mask_policy="static foreground 3x3 original pixels; no sky; ambiguous depth collisions excluded")


def _resize_accepted_prior(arrays, width, height, tolerance):
    """Downsample accepted support conservatively; never expand learned pixels.

    Integer blocks use exact all-valid/min-support and max/min depth checks.
    Other ratios use a covering conservative neighborhood (which can abstain
    more than necessary). No interpolation can bridge invalid/edge pixels.
    """
    from PIL import Image
    from scipy.ndimage import minimum_filter, maximum_filter
    sh, sw = arrays["valid"].shape
    if width > sw or height > sh:
        raise ValueError("Validated learned depth must never be upsampled")
    if (height, width) == (sh, sw):
        return {key: value.copy() for key, value in arrays.items()}
    if sh % height == 0 and sw % width == 0:
        shape = (height, sh//height, width, sw//width)
        block = lambda value: value.reshape(shape)
        valid = block(arrays["valid"]).all(axis=(1, 3))
        lo = block(arrays["depth_z"]).min(axis=(1, 3))
        hi = block(arrays["depth_z"]).max(axis=(1, 3))
        depth = block(arrays["depth_z"]).mean(axis=(1, 3))
        count = block(arrays["source_count"]).min(axis=(1, 3))
        confidence = block(arrays["confidence"]).min(axis=(1, 3))
    else:
        y = np.minimum(((np.arange(height)+.5)*sh/height).astype(int), sh-1)
        x = np.minimum(((np.arange(width)+.5)*sw/width).astype(int), sw-1)
        size = (2*math.ceil(sh/height)+1, 2*math.ceil(sw/width)+1)
        sample = lambda value: value[np.ix_(y, x)]
        low = lambda value: sample(minimum_filter(value, size=size, mode="constant", cval=0))
        valid = low(arrays["valid"])
        lo, hi = low(arrays["depth_z"]), sample(maximum_filter(arrays["depth_z"], size=size, mode="constant", cval=0))
        depth = np.asarray(Image.fromarray(arrays["depth_z"]).resize((width, height), Image.Resampling.BOX))
        count, confidence = low(arrays["source_count"]), low(arrays["confidence"])
    valid = valid & (lo > 0) & ((hi-lo) <= tolerance*lo)
    return dict(depth_z=np.where(valid, depth, 0).astype(np.float32), valid=valid,
                confidence=np.where(valid, confidence, 0).astype(np.float32), source_count=np.where(valid, count, 0))


def load_validated_depth_prior(job_dir, dataset, train_frames, heldout_frames, options):
    """Load only accepted UniSHARP TRAIN pixels with portable source bindings.

    Hash verification also runs for zero-weight baselines so matched arms bind
    identical immutable inputs. Neither raw nor candidate_* arrays are read.
    Heldout learned maps never become optimization or evaluation targets.
    """
    from PIL import Image
    prior = Path(job_dir) / "depth_prior"
    path = prior / "manifest.json"
    required = options.depth_use_validated_prior and options.depth_moment_weight > 0
    if not path.is_file():
        if required:
            raise ValueError("Validated depth-prior candidate requires depth_prior/manifest.json")
        return {}, dict(status="not_supplied", used_for_optimization=False), {}
    manifest = _read(path)
    expected = dict(schema_version=1, status="complete", stage="depth_prior", coordinate_frame="EDN", units="metres",
                    depth_convention="camera_z", pixel_center_offset=.5, camera_convention="OpenCV_world_to_camera",
                    geometry_scope="transductive_shared_sfm_calibration")
    if any(manifest.get(key) != value for key, value in expected.items()) or manifest.get("validation_status") not in ("multistation_consistent_subset", "insufficient_supported_depth"):
        raise ValueError("Validated depth-prior manifest convention/status is incompatible")
    train_ids, heldout_ids = validate_splits(train_frames, heldout_frames)
    if set(map(str, manifest.get("train_station_ids", []))) != train_ids or set(map(str, manifest.get("heldout_station_ids", []))) != heldout_ids:
        raise ValueError("Validated depth-prior physical-station split differs")
    minimum = manifest.get("minimum_other_station_support")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 2:
        raise ValueError("Validated depth requires >=2 OTHER physical stations")
    bindings = {"manifest.json": sha256(path)}
    for filename, key in (("input_manifest.json", "input_manifest_sha256"), ("calibration_report.json", "calibration_report_sha256"), ("raw/provenance.json", "provenance_sha256"), ("observation_deduplication.json", "observation_deduplication_sha256")):
        if filename == "observation_deduplication.json" and key not in manifest:
            continue
        actual = sha256(_file(prior, filename))
        if actual != manifest.get(key):
            raise ValueError("Validated depth-prior provenance artifact changed: " + filename)
        bindings[filename] = actual
    sources = manifest.get("source_dataset_files_sha256", {})
    essential = {"dataset_manifest.json", "init_points.npz", "transforms_train.json", "transforms_heldout.json", "sparse_depth_manifest.json", "sparse_depth_observations.npz"}
    essential.update(frame[key] for frame in train_frames+heldout_frames for key in ("file_path", "sfm_mask_path"))
    if not isinstance(sources, dict) or not essential <= set(sources):
        raise ValueError("Validated depth lacks portable source photo/SfM bindings")
    for name, expected_sha in sources.items():
        if Path(name).is_absolute() or ".." in Path(name).parts or sha256(_file(dataset, name)) != expected_sha:
            raise ValueError("Validated depth-prior source changed: " + name)
    for frame in train_frames+heldout_frames:
        if frame.get("image_sha256") != sources[frame["file_path"]] or frame.get("sfm_mask_sha256") != sources[frame["sfm_mask_path"]]:
            raise ValueError("Validated depth source disagrees with dataset frame hashes")
    for split in ("train", "heldout"):
        if manifest.get("source_transforms_sha256", {}).get(split) != sources[f"transforms_{split}.json"]:
            raise ValueError("Validated depth camera source binding differs")
    roster = {frame["file_path"]: (frame, split) for split, group in (("train", train_frames), ("heldout", heldout_frames)) for frame in group}
    entries = manifest.get("entries", [])
    if len(entries) != len(roster) or {entry.get("frame_name") for entry in entries} != set(roster):
        raise ValueError("Validated depth frame roster differs")
    maps, counts = {}, dict(accepted_native_pixels=0, accepted_training_pixels=0, frames=0)
    source_histogram = {}
    for entry in entries:
        name = entry["frame_name"]
        frame, split = roster[name]
        if entry.get("split") != split or str(entry.get("station_id")) != str(frame["station_id"]) or str(entry.get("pano_id")) != str(frame.get("pano_id")):
            raise ValueError("Validated depth entry station/capture/split differs")
        if entry.get("image_sha256") != sources[frame["file_path"]] or entry.get("sfm_mask_sha256") != sources[frame["sfm_mask_path"]]:
            raise ValueError("Validated depth entry source photo/mask binding differs")
        count = entry.get("valid_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0 or entry.get("status") != ("accepted" if count else "insufficient_support"):
            raise ValueError("Validated depth entry acceptance status differs")
        if split == "heldout" and count:
            raise ValueError("Heldout learned prior must not contain accepted calibration")
        counts["accepted_native_pixels"] += count
        if not count:
            continue
        npz = _file(prior, entry["npz"])
        if Path(entry["npz"]).is_absolute() or ".." in Path(entry["npz"]).parts:
            raise ValueError("Accepted learned artifact requires a portable confined path")
        actual_sha = sha256(npz)
        if actual_sha != entry.get("sha256"):
            raise ValueError("Accepted learned depth artifact changed")
        if entry["npz"] in bindings:
            raise ValueError("Duplicate accepted learned artifact path")
        bindings[entry["npz"]] = actual_sha
        width, height = entry.get("width"), entry.get("height")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (width, height)):
            raise ValueError("Invalid learned depth dimensions")
        K = np.array([[frame["fl_x"]*width/frame["w"], 0, frame["cx"]*width/frame["w"]], [0, frame["fl_y"]*height/frame["h"], frame["cy"]*height/frame["h"]], [0, 0, 1]])
        view = np.linalg.inv(np.asarray(frame["transform_matrix"]) @ np.diag([1., -1., -1., 1.]))
        with np.load(npz, allow_pickle=False) as archive:
            for key, expected_value in dict(world_frame="EDN", units="metres", depth_convention="camera_z", pixel_center_offset=.5, frame_name=name, station_id=str(frame["station_id"]), pano_id=str(frame.get("pano_id")), split="train", evidence="calibrated_UniSHARP_first_surface_multistation_consistency").items():
                if archive[key].shape != () or archive[key].item() != expected_value:
                    raise ValueError("Accepted learned depth metadata differs: " + key)
            for key, expected_value in (("K", K), ("camera_from_world", view)):
                if np.asarray(entry[key]).shape != expected_value.shape or archive[key].shape != expected_value.shape or not np.allclose(entry[key], expected_value, atol=1e-7, rtol=1e-7) or not np.allclose(archive[key], expected_value, atol=1e-7, rtol=1e-7):
                    raise ValueError("Accepted learned depth camera/K differs")
            arrays = {key: archive[key].copy() for key in ("depth_z", "valid", "confidence", "source_count")}
        if any(value.shape != (height, width) for value in arrays.values()) or arrays["valid"].dtype != bool or not np.issubdtype(arrays["source_count"].dtype, np.integer):
            raise ValueError("Accepted learned depth array shape/type differs")
        valid = arrays["valid"]
        support = entry.get("support_by_other_station", {})
        if not isinstance(support, dict) or not set(support) <= (train_ids-{str(frame["station_id"])}) or any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= width*height for v in support.values()) or sum(v > 0 for v in support.values()) < minimum:
            raise ValueError("Learned depth source roster must contain OTHER training stations")
        if int(valid.sum()) != count or not np.isfinite(arrays["depth_z"][valid]).all() or np.any(arrays["depth_z"][valid] <= 0) or not np.isfinite(arrays["confidence"][valid]).all() or np.any(arrays["confidence"][valid] <= 0) or np.any(arrays["confidence"][valid] > .5+1e-7) or np.any(arrays["source_count"][valid] < minimum) or np.any(arrays["source_count"][valid] > sum(v > 0 for v in support.values())):
            raise ValueError("Accepted learned pixels violate depth/confidence/OTHER-station support")
        with Image.open(_file(dataset, frame["sfm_mask_path"])) as image:
            if image.size != (frame["w"], frame["h"]):
                raise ValueError("Learned depth foreground mask dimensions differ")
            foreground = np.asarray(image.convert("F").resize((width, height), Image.Resampling.BOX)) >= 254.999
        if np.any(valid & ~foreground):
            raise ValueError("Accepted learned depth includes excluded sky/dynamic pixels")
        factor = min(1., options.depth_resolution/max(frame["w"], frame["h"]), options.resolution/max(frame["w"], frame["h"]), width/frame["w"], height/frame["h"])
        w, h = max(1, round(frame["w"]*factor)), max(1, round(frame["h"]*factor))
        mapped = _resize_accepted_prior(arrays, w, h, options.depth_collision_relative_tolerance)
        n = int(mapped["valid"].sum())
        if n:
            mapped.update(width=w, height=h, frame=name, split="train", station_id=str(frame["station_id"]), accepted_pixels=n)
            maps[name] = mapped
            counts["frames"] += 1
            counts["accepted_training_pixels"] += n
            values, numbers = np.unique(mapped["source_count"][mapped["valid"]], return_counts=True)
            for value, number in zip(values, numbers):
                source_histogram[str(int(value))] = source_histogram.get(str(int(value)), 0)+int(number)
    if manifest.get("accepted_pixels") != counts["accepted_native_pixels"] or (counts["accepted_native_pixels"] > 0) != (manifest["validation_status"] == "multistation_consistent_subset"):
        raise ValueError("Validated depth manifest acceptance totals differ")
    if required and counts["accepted_training_pixels"] < options.depth_min_pixels:
        raise ValueError("No accepted learned TRAIN depth pixels; cannot claim UniSHARP supervision")
    provenance = dict(status=manifest["validation_status"], manifest_sha256=bindings["manifest.json"], **counts,
        source_count_histogram=source_histogram, minimum_other_station_support=minimum, used_for_optimization=bool(required),
        relative_loss_weight=options.depth_prior_relative_weight, geometry_scope=manifest["geometry_scope"],
        uncertainty="Confidence is nominal local calibration quality, not probability or sensor uncertainty; separately normalized robust learned loss is downweighted. No rejected/raw maps or heldout learned targets are used.",
        consistency_thresholds={key: manifest.get("settings", {}).get(key) for key in ("maximum_relative_depth_error", "maximum_roundtrip_pixels", "maximum_rgb_patch_error")})
    return maps if options.depth_use_validated_prior else {}, provenance, bindings


def merge_training_depth(sparse, learned):
    """Copy TRAIN maps only; actual observations win and ambiguous ones veto.

    The separate sparse dictionary and all heldout evaluation arrays are left
    untouched. source_kind=1 means actual track; 2 means validated learned.
    """
    result, stats = {}, dict(actual_pixels=0, learned_pixels=0, sparse_overrides=0, ambiguous_vetoes=0)
    for name in sorted({name for name, item in sparse.items() if item["split"] == "train"} | set(learned)):
        actual, prior = sparse.get(name), learned.get(name)
        if any(item is not None and item["split"] != "train" for item in (actual, prior)):
            raise ValueError("Heldout depth cannot enter merged training targets")
        reference = actual if actual is not None else prior
        shape = (reference["height"], reference["width"])
        if prior is not None and prior["valid"].shape != shape:
            raise ValueError("Rerasterize actual TRAIN observations at learned native resolution before merging")
        out = {key: reference[key] for key in ("frame", "station_id", "split", "width", "height")}
        out.update(depth_z=np.zeros(shape, np.float32), confidence=np.zeros(shape, np.float32), valid=np.zeros(shape, bool), point_ids=np.full(shape, -1, np.int64), source_kind=np.zeros(shape, np.uint8), source_count=np.zeros(shape, np.uint16))
        if prior is not None:
            for key in ("depth_z", "confidence", "valid", "source_count"):
                out[key] = prior[key].copy()
            out["source_kind"][prior["valid"]] = 2
        if actual is not None:
            ambiguous = actual.get("ambiguous_mask", np.zeros(shape, bool))
            stats["ambiguous_vetoes"] += int((out["valid"] & ambiguous).sum())
            stats["sparse_overrides"] += int((out["valid"] & actual["valid"]).sum())
            for key in ("depth_z", "confidence", "valid", "source_kind", "source_count"):
                out[key][ambiguous] = 0
            for key in ("depth_z", "confidence", "valid", "point_ids"):
                out[key][actual["valid"]] = actual[key][actual["valid"]]
            out["source_kind"][actual["valid"]] = 1
            out["source_count"][actual["valid"]] = 0  # OTHER-station count belongs only to learned evidence.
        out["accepted_pixels"] = int(out["valid"].sum())
        if not out["accepted_pixels"]:
            continue
        out["target_roster_sha256"] = hashlib.sha256(b"".join(out[key].tobytes() for key in ("depth_z", "confidence", "valid", "point_ids", "source_kind", "source_count"))).hexdigest()
        result[name] = out
        stats["actual_pixels"] += int((out["source_kind"] == 1).sum())
        stats["learned_pixels"] += int((out["source_kind"] == 2).sum())
    return result, stats


def prepare_frame(frame, dataset, resolution, center, radius, *, allow_empty=False):
    from PIL import Image
    from scipy.ndimage import minimum_filter
    from .mask_projection import resize_lanczos_valid_mask
    factor = min(1., resolution / max(frame["w"], frame["h"]))
    w, h = max(1, round(frame["w"] * factor)), max(1, round(frame["h"] * factor))
    sx, sy = w / frame["w"], h / frame["h"]
    K = np.array([[frame["fl_x"] * sx, 0, frame["cx"] * sx], [0, frame["fl_y"] * sy, frame["cy"] * sy], [0, 0, 1]], np.float32)
    c2w = np.asarray(frame["transform_matrix"], np.float64)
    rotation = c2w[:3, :3] @ np.diag([1., -1., -1.])
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = rotation.T
    view[:3, 3] = -rotation.T @ ((c2w[:3, 3] - center) / radius)
    rgb_path, mask_path = _file(dataset, frame["file_path"]), _file(dataset, frame["mask_path"])
    with Image.open(rgb_path) as image:
        if image.size != (frame["w"], frame["h"]):
            raise ValueError("Source photo size differs from calibration")
        rgb = np.array(image.convert("RGB").resize((w, h), Image.Resampling.LANCZOS))

    def load_mask(path):
        with Image.open(path) as image:
            if image.size != (frame["w"], frame["h"]):
                raise ValueError("Mask size differs from photo/calibration")
            source_valid = np.array(image.convert("L")) == 255
        return resize_lanczos_valid_mask(source_valid, (h, w))

    mask = load_mask(mask_path)
    if not mask.any() and not allow_empty:
        raise ValueError("Photo has no valid photometric supervision: " + frame["file_path"])
    result = dict(meta=frame, frame=frame["file_path"], station_id=str(frame["station_id"]), w=w, h=h, K=K, view=view,
                  rgb=rgb, mask=mask, ssim_mask=minimum_filter(mask.astype(np.uint8), size=11, mode="constant", cval=0).astype(bool),
                  reference_rgb_sha256=sha256(rgb_path), reference_mask_sha256=sha256(mask_path))
    result['mask_resampling'] = dict(method='all_source_pixels_in_pillow_lanczos3_support_envelope',
        source_shape=[frame['h'], frame['w']], output_shape=[h, w],
        same_size_identity=(h, w) == (frame['h'], frame['w']), source_border='truncated',
        rgb_filter='PIL.Image.Resampling.LANCZOS', reducing_gap=None,
        signed_coefficients='positive_and_negative_contributors_required',
        evidence_policy='photometric, foreground and sky each require full source support')
    for category, key in (("foreground", "foreground_mask_path"), ("sky", "sky_mask_path")):
        if frame.get(key):
            result[category] = load_mask(_file(dataset, frame[key])) & mask
    return result


def supervised_frame_schedule(frames, steps, seed):
    """Skip fully masked TRAIN faces without silently dropping a station."""
    eligible = np.asarray([i for i, frame in enumerate(frames) if frame['mask'].any()], np.int64)
    stations = {str(frame['station_id']) for frame in frames}
    observed = {str(frames[i]['station_id']) for i in eligible}
    if not stations or stations != observed:
        raise ValueError('Insufficient photometric evidence: a training physical station has no unmasked face')
    return eligible[station_uniform_schedule([frames[i] for i in eligible], steps, seed)]


def masked_losses(prediction, target, mask, ssim_mask):
    import torch
    import torch.nn.functional as F
    l1 = ((prediction - target).abs() * mask[..., None]).sum() / (mask.sum() * 3).clamp_min(1)
    x, y = prediction.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None]
    coords = torch.arange(11, device=x.device, dtype=x.dtype) - 5
    kernel = torch.exp(-coords.square() / (2 * 1.5**2))
    kernel = kernel / kernel.sum()
    window = (kernel[:, None] * kernel[None, :])[None, None].repeat(15, 1, 1, 1)
    statistics = F.conv2d(torch.cat([x, y, x*x, y*y, x*y], 1), window, padding=5, groups=15)
    mx, my, xx, yy, xy = statistics.split(3, dim=1)
    ssim = ((2*mx*my + .01**2) * (2*(xy-mx*my) + .03**2)) / ((mx.square()+my.square()+.01**2) * (xx-mx.square()+yy-my.square()+.03**2)).clamp_min(1e-12)
    valid = ssim_mask[None, None]
    dssim = ((1-ssim) * valid).sum() / (valid.sum()*3).clamp_min(1)
    return l1, dssim


def _evaluate(params, frames, output, rasterization, torch, settings, radius, model_sha):
    from PIL import Image
    directory = output / "evaluation"
    directory.mkdir(exist_ok=True)
    rows = []
    with torch.no_grad():
        for index, frame in enumerate(frames):
            color, alpha, _ = _render(params, frame, settings.sh_degree, rasterization, radius)
            prediction = color[0].clamp(0, 1).cpu().numpy()
            a = alpha[0, ..., 0].clamp(0, 1).cpu().numpy()
            reference = frame["rgb"].astype(np.float32) / 255
            row = dict(frame=frame["frame"], station_id=frame["station_id"],
                       reference_rgb_sha256=frame["reference_rgb_sha256"], reference_mask_sha256=frame["reference_mask_sha256"],
                       **masked_metrics(prediction, reference, a, frame["mask"], alpha_threshold=settings.alpha_threshold))
            if 'mask_resampling' in frame:
                row['mask_resampling'] = frame['mask_resampling']
            for category in ("foreground", "sky"):
                if category in frame:
                    row[category] = masked_metrics(prediction, reference, a, frame[category], alpha_threshold=settings.alpha_threshold)
            if str(frame["meta"].get("face", "")).lower() in ("d", "down"):
                row["down"] = masked_metrics(prediction, reference, a, frame["mask"], alpha_threshold=settings.alpha_threshold)
            stem = f"{index:06d}"
            Image.fromarray(np.rint(prediction * 255).astype(np.uint8)).save(directory / (stem + ".png"))
            np.savez_compressed(directory / (stem + ".npz"), alpha=a, mask=frame["mask"])
            row.update(rgb_png=f"evaluation/{stem}.png", alpha_npz=f"evaluation/{stem}.npz", width=frame["w"], height=frame["h"])
            rows.append(row)
    return dict(status="measured" if rows else "unassessed", model_sha256=model_sha, views=rows,
                summary=summarize_views(rows), scope="heldout_RGB_at_known_SfM_poses_not_geometry_or_cross_location_validation")


def _render(params, frame, degree, rasterization, radius):
    import torch
    return rasterization(means=params["means"], quats=params["quats"], scales=params["scales"].exp(),
        opacities=params["opacities"].sigmoid(), colors=torch.cat([params["sh0"], params["shN"]], dim=1),
        viewmats=frame["view_gpu"][None], Ks=frame["K_gpu"][None], width=frame["w"], height=frame["h"],
        sh_degree=degree, render_mode="RGB", packed=True, near_plane=.05/radius, far_plane=1e7/radius, rasterize_mode="classic")


def _render_sparse_moments(params, frame, target, rasterization, radius):
    from tools.streetview_geometry.losses import render_depth_moments
    K = frame["K_gpu"].clone()
    K[0] *= target["width"]/frame["w"]
    K[1] *= target["height"]/frame["h"]
    return render_depth_moments(params["means"], params["quats"], params["scales"].exp(), params["opacities"].sigmoid(),
        frame["view_gpu"], K, target["width"], target["height"], rasterizer=rasterization,
        packed=True, near_plane=.05/radius, far_plane=1e7/radius, rasterize_mode="classic")


def training_depth_losses(moments, target, radius, options):
    """Separately normalize actual-track and learned losses, sharing one pass.

    Dense pseudo-depth pixel count cannot amplify its relative loss budget.
    Nominal learned confidence weights pixels only within that smaller budget.
    """
    import torch
    from tools.streetview_geometry.losses import depth_moment_loss, alpha_coverage_loss
    if target["split"] != "train":
        raise ValueError("Heldout depth was selected for optimization")
    device = moments["alpha"].device
    depth = torch.as_tensor(target["depth_z"], device=device)/radius
    valid = torch.as_tensor(target["valid"], device=device)
    confidence = torch.as_tensor(target["confidence"], device=device)
    kind = torch.as_tensor(target["source_kind"], device=device)
    reference = torch.as_tensor(target["reference_alpha"], device=device)
    terms = {}
    for name, value in (("actual", 1), ("learned", 2)):
        selected = valid & (kind == value)
        terms[name] = depth_moment_loss(moments["first_moment"], moments["second_moment"], moments["alpha"],
            depth, selected, confidence, relative=True, robust=True, min_depth=1e-10)
        terms[name+"_coverage"] = alpha_coverage_loss(moments["alpha"], reference, selected, confidence)
    terms["depth"] = terms["actual"] + options.depth_prior_relative_weight*terms["learned"]
    terms["coverage"] = terms["actual_coverage"] + options.depth_prior_relative_weight*terms["learned_coverage"]
    return terms


def _evaluate_sparse_depth(params, frames, sparse, provenance, output, rasterization, torch, options, radius, model_sha, train_stations):
    rows = []
    with torch.no_grad():
        for frame in frames:
            target = sparse.get(frame["frame"])
            if target is None or target["split"] != "heldout":
                continue
            moments = _render_sparse_moments(params, frame, target, rasterization, radius)
            metrics = depth_metrics(moments["first_moment"].cpu().numpy()*radius, moments["second_moment"].cpu().numpy()*radius**2,
                moments["alpha"].cpu().numpy(), target["depth_z"], target["valid"], target["confidence"], alpha_threshold=options.alpha_threshold)
            rows.append(dict(frame=frame["frame"], station_id=frame["station_id"], target_roster_sha256=target["target_roster_sha256"], **metrics))
    report = dict(status="measured" if rows else "unassessed", model_sha256=model_sha, split="heldout", used_for_optimization=False,
        geometry_scope="transductive_shared_sfm", optimization_station_ids=sorted(train_stations),
        observation_manifest_sha256=provenance.get("manifest_sha256"), views=rows, summary=summarize_depth_views(rows),
        interpretation="Heldout actual-track center-depth consistency at shared SfM geometry/poses; not independent sensor or surface truth")
    write_json(output / "heldout_depth_metrics.json", report)
    return report


def evaluation_manifest(heldout, sparse, sparse_provenance, train_stations, processing):
    """Identical trusted roster for the initial and optimized model renders."""
    result = dict(discovery_station_ids=sorted(train_stations), evaluation_frames=[
        dict({key: frame[key] for key in ("frame", "station_id", "reference_rgb_sha256", "reference_mask_sha256")},
             static_pixels=int(frame['mask'].sum())) for frame in heldout])
    result["processing_options"] = processing
    result["depth_evaluation"] = dict(observation_manifest_sha256=sparse_provenance.get("manifest_sha256"),
        evaluation_frames=[dict(frame=name, station_id=target["station_id"], target_roster_sha256=target["target_roster_sha256"],
                               target_pixels=target["accepted_pixels"]) for name, target in sorted(sparse.items()) if target["split"] == "heldout"])
    return result




def run(config, job_dir, settings):
    """Train one real SfM baseline arm; depth candidates use separate runs."""
    if any(settings.get(key) is not None for key in ("initialization", "gaussian_initializer", "spherical_initializer")):
        raise ValueError("Multi-view generation uses SfM initialization only")
    options = TrainingSettings.from_inputs(config, settings)
    processing = read_processing_options(config)
    root = Path(job_dir).resolve()
    dataset, output = root / "sfm/dataset", root / "training"
    manifest_path = output / "manifest.json"
    train_raw = _read(dataset / "transforms_train.json")["frames"]
    heldout_raw = _read(dataset / "transforms_heldout.json")["frames"]
    train_stations, heldout_stations = validate_splits(train_raw, heldout_raw)
    dataset_manifest = _read(dataset / "dataset_manifest.json")
    if read_processing_options(dataset_manifest) != processing:
        raise ValueError('Dataset processing options differ from frozen training configuration')
    validate_dataset_manifest(dataset_manifest, train_stations, heldout_stations)
    if dataset_manifest.get("diagnostic_point_geometry") is not None:
        raise ValueError("Product training requires an SfM dataset, not diagnostic point geometry")
    inputs = {name: sha256(dataset / name) for name in ("transforms_train.json", "transforms_heldout.json", "init_points.npz", "dataset_manifest.json")}
    paths = sorted({frame[key] for frame in train_raw + heldout_raw for key in
                    ("file_path", "mask_path", "foreground_mask_path", "sky_mask_path") if frame.get(key)})
    inputs["photos_and_masks"] = {name: sha256(_file(dataset, name)) for name in paths}
    if (dataset / "sparse_depth_manifest.json").is_file():
        depth_manifest = _read(dataset / "sparse_depth_manifest.json")
        inputs["sparse_depth_manifest.json"] = sha256(dataset / "sparse_depth_manifest.json")
        inputs["sparse_depth_observations"] = sha256(_file(dataset, depth_manifest["npz"]))
    learned, learned_provenance, learned_bindings = load_validated_depth_prior(root, dataset, train_raw, heldout_raw, options)
    if learned_bindings:
        inputs["validated_depth_prior"] = learned_bindings
    signature_data = dict(inputs=inputs, settings=asdict(options), processing_options=processing, code_sha256=sha256(__file__),
                          quality_code_sha256=sha256(Path(__file__).with_name("quality.py")), export_code_sha256=sha256(Path(__file__).with_name("export.py")),
                          mask_projection_code_sha256=sha256(Path(__file__).with_name("mask_projection.py")),
                          mask_resampling_policy='all_source_pixels_in_pillow_lanczos3_support_envelope')
    signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
    if manifest_path.exists():
        previous = _read(manifest_path)
        if previous.get("signature") != signature:
            raise ValueError("Existing training differs; use a new job directory")
        if previous.get("status") == "completed":
            if validate_ply(output / "model.ply")["sha256"] != previous["artifact"]["sha256"]:
                raise ValueError("Completed model changed")
            return previous
        raise ValueError("Interrupted training is retained; explicit resume is not implemented")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Training directory is not empty and has no matching manifest")
    center, radius = normalization(train_raw)
    with np.load(dataset / "init_points.npz", allow_pickle=False) as data:
        xyz, rgb = data["xyz"].astype(np.float64), data["rgb"].copy()
        support = data["support_station_count"].copy()
    if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape or rgb.dtype != np.uint8 or support.shape != (len(xyz),) or not np.issubdtype(support.dtype, np.integer):
        raise ValueError("Seed contract requires xyz Nx3, rgb uint8 Nx3 and support_station_count integers")
    if len(xyz) < 4 or not np.isfinite(xyz).all() or np.any(support < 2):
        raise ValueError("Need at least four finite seeds supported by >=2 physical stations")
    training_resolution = min(max(item["width"], item["height"]) for item in learned.values()) if learned else None
    sparse, sparse_provenance = load_sparse_depth(dataset, train_raw, heldout_raw, options, train_max_resolution=training_resolution)
    training_depth, merged_provenance = merge_training_depth(sparse, learned)
    if options.depth_use_validated_prior and options.depth_moment_weight and merged_provenance["learned_pixels"] < options.depth_min_pixels:
        raise ValueError("No learned TRAIN pixels remain after actual-observation precedence; cannot claim UniSHARP supervision")
    # Sky-only cube faces remain in the fixed evaluation roster with zero
    # measured pixels, but cannot consume optimizer updates without RGB data.
    train = [prepare_frame(frame, dataset, options.resolution, center, radius, allow_empty=True) for frame in train_raw]
    heldout = [prepare_frame(frame, dataset, options.resolution, center, radius, allow_empty=True) for frame in heldout_raw]
    schedule = supervised_frame_schedule(train, options.steps, options.seed)
    # Import only after complete CPU contract checking. No renderer replacement
    # or CPU mock may manufacture a successful training manifest.
    import torch
    from gsplat import rasterization
    import gsplat
    from scipy.spatial import cKDTree
    if not torch.cuda.is_available():
        raise RuntimeError("Actual CUDA Torch and gsplat are required for training")
    torch.cuda.set_device(0)
    torch.set_num_threads(options.cpu_workers)
    torch.manual_seed(options.seed)
    np.random.seed(options.seed)
    output.mkdir(parents=True, exist_ok=True)
    status = dict(status="running", signature=signature, provenance=signature_data, started_utc=_now(), completed_steps=0,
                  training_station_ids=sorted(train_stations), heldout_station_ids=sorted(heldout_stations),
                  coordinate_frame="EDN", units="metres", world_up=[0, -1, 0],
                  metric_scale_evidence=dataset_manifest.get("metric_alignment", dataset_manifest.get("scale_evidence", "not supplied")),
                  processing_options=processing,
                  sky_policy=("Predicted sky regions excluded from RGB loss and sky append disabled; this is not a guarantee of zero residual sky Gaussians" if processing['remove_sky'] else "Photometric sky retained in shared Gaussian RGB loss; optional shared sky requires a separately verified candidate"),
                  sparse_depth=sparse_provenance,
                  validated_depth_prior=learned_provenance, merged_training_depth=merged_provenance,
                  runtime=dict(torch=torch.__version__, gsplat=gsplat.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name()))
    write_json(manifest_path, status)
    start = time.monotonic()
    try:
        for frame in train + heldout:
            frame["view_gpu"] = torch.tensor(frame["view"], device="cuda")
            frame["K_gpu"] = torch.tensor(frame["K"], device="cuda")
        seed_keep = np.arange(len(xyz))
        if len(xyz) > options.max_splats:
            seed_keep = np.sort(np.random.default_rng(options.seed).choice(len(xyz), options.max_splats, replace=False))
            xyz, rgb = xyz[seed_keep], rgb[seed_keep]
        np.save(output / "initial_seed_indices.npy", seed_keep)
        distances = cKDTree(xyz).query(xyz, k=4, workers=options.cpu_workers)[0][:, 1:]
        sigma = np.maximum(np.sqrt(np.mean(distances**2, axis=1)) / radius, 1e-7)
        count, rest_count = len(xyz), (options.sh_degree + 1)**2 - 1
        params = torch.nn.ParameterDict({
            "means": torch.nn.Parameter(torch.tensor((xyz-center)/radius, dtype=torch.float32, device="cuda")),
            "scales": torch.nn.Parameter(torch.tensor(np.repeat(np.log(sigma)[:, None], 3, axis=1), dtype=torch.float32, device="cuda")),
            "quats": torch.nn.Parameter(torch.rand((count, 4), device="cuda")),
            "opacities": torch.nn.Parameter(torch.full((count,), math.log(.1/.9), device="cuda")),
            "sh0": torch.nn.Parameter(torch.tensor(((rgb.astype(np.float32)/255-.5)/.28209479177387814)[:, None], device="cuda")),
            "shN": torch.nn.Parameter(torch.zeros((count, rest_count, 3), device="cuda"))})
        rates = {key: value * options.learning_rate_scale for key, value in dict(means=1.6e-4, scales=.005, quats=.001, opacities=.05, sh0=.0025, shN=.0025/20).items()}
        optimizers = {key: torch.optim.Adam([value], lr=rates[key], eps=1e-15) for key, value in params.items()}
        factor = options.steps / 30000
        strategy, strategy_state = create_strategy(options, params, optimizers)
        depth_gradient_verified = False
        depth_supervised_steps = 0
        learned_depth_gradient_verified = False
        learned_depth_supervised_steps = 0
        learned_depth_pixel_observations = 0
        if options.depth_moment_weight:
            # Freeze actual initial-model alpha before ANY training update.
            # It is not an opaque-plane target and does not certify new floor.
            with torch.no_grad():
                for frame in train:
                    target_depth = training_depth.get(frame["frame"])
                    if target_depth is not None:
                        target_depth["reference_alpha"] = _render_sparse_moments(params, frame, target_depth, rasterization, radius)["alpha"].cpu().numpy()
            depth_directory = output / "depth_targets"
            depth_directory.mkdir()
            target_records = []
            saved_targets = {**{name: item for name, item in sparse.items() if item["split"] == "heldout"}, **training_depth}
            for index, (name, target_depth) in enumerate(sorted(saved_targets.items())):
                filename = f"{index:06d}.npz"
                arrays_to_save = {key: target_depth[key] for key in ("depth_z", "confidence", "valid", "point_ids")}
                arrays_to_save.update({key: target_depth[key] for key in ("source_kind", "source_count") if key in target_depth})
                if "reference_alpha" in target_depth:
                    arrays_to_save["initial_reference_alpha"] = target_depth["reference_alpha"]
                np.savez_compressed(depth_directory / filename, **arrays_to_save)
                target_records.append(dict(frame=name, split=target_depth["split"], npz=filename,
                    sha256=sha256(depth_directory / filename), target_roster_sha256=target_depth["target_roster_sha256"]))
            write_json(depth_directory / "manifest.json", dict(source=sparse_provenance, learned_source=learned_provenance, merged_training=merged_provenance, targets=target_records,
                coverage_reference="detached actual initial-model alpha, never invented opaque geometry",
                optimizer_uses="train targets only; heldout targets saved for reproducible evaluation"))
        np.save(output / "training_frame_schedule.npy", schedule)
        status.update(initial_gaussians=count, normalization_center_edn_m=center.tolist(), scene_radius_m=radius,
                      sampling="uniform physical station, then uniform frame with photometric supervision within that station",
                      excluded_training_frames=[dict(frame=frame['frame'],station_id=frame['station_id'],reason='no_unmasked_photometric_pixels') for frame in train if not frame['mask'].any()],
                      selected_seed_indices_sha256=sha256(output / "initial_seed_indices.npy"),
                      strategy=dict(name="gsplat." + type(strategy).__name__, **vars(strategy)))
        if options.strategy == "3dgs":
            status["strategy_adapter"] = dict(scene_scale=1., packed=True,
                growth_budget="Native <=2N event; disable growth above max_splats//2, retain native pruning",
                comparison_scope="Opt-in algorithm; no quality improvement inferred from strategy choice",
                regularization_policy="Explicit opacity_reg/scale_reg values; never silently changed",
                opacity_reset_compatibility=strategy_state["_streetview_reset_compatibility"])
        write_json(manifest_path, status)

        def checkpoint(step):
            temporary = output / "checkpoint_last.pt.tmp"
            torch.save(dict(signature=signature, completed_steps=step, params={key: value.detach().cpu() for key, value in params.items()},
                optimizers={key: value.state_dict() for key, value in optimizers.items()}, strategy_state=strategy_state,
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), center_edn_m=center, radius_m=radius), temporary)
            temporary.replace(output / "checkpoint_last.pt")

        with (output / "metrics_history.jsonl").open("w", encoding="utf8") as history:
            for step, frame_index in enumerate(schedule):
                frame = train[int(frame_index)]
                target = torch.tensor(frame["rgb"], dtype=torch.float32, device="cuda") / 255
                mask = torch.tensor(frame["mask"], device="cuda")
                ssim_mask = torch.tensor(frame["ssim_mask"], device="cuda")
                degree = min(options.sh_degree, step // max(1, round(1000*factor)))
                lr = rates["means"] * .01**(step/options.steps)
                optimizers["means"].param_groups[0]["lr"] = lr
                for optimizer in optimizers.values():
                    optimizer.zero_grad(set_to_none=True)
                rendered, alpha, info = _render(params, frame, degree, rasterization, radius)
                strategy_pre_backward(options, strategy, params, optimizers, strategy_state, step, info)
                l1, dssim = masked_losses(rendered[0], target, mask, ssim_mask)
                regularization = options.opacity_reg*params["opacities"].sigmoid().mean() + options.scale_reg*params["scales"].exp().mean()
                loss = (1-options.ssim_weight)*l1 + options.ssim_weight*dssim + regularization
                prior_loss = rendered[0, 0, 0, 0]*0
                actual_depth_loss = prior_loss
                coverage_loss = prior_loss
                learned_loss = prior_loss
                effective_depth_weight = 0.
                gradient_record = {}
                target_depth = training_depth.get(frame["frame"])
                ramp = max(0., min(1., (step+1-options.steps*options.depth_start_fraction)/max(1., options.steps*options.depth_ramp_fraction)))
                if options.depth_moment_weight and target_depth is not None and ramp > 0:
                    if target_depth["split"] != "train":
                        raise ValueError("Heldout depth was selected for optimization")
                    moments = _render_sparse_moments(params, frame, target_depth, rasterization, radius)
                    terms = training_depth_losses(moments, target_depth, radius, options)
                    prior_loss, coverage_loss, learned_loss = terms["depth"], terms["coverage"], terms["learned"]
                    actual_depth_loss = terms["actual"]
                    effective_depth_weight = options.depth_moment_weight*ramp
                    geometry_objective = effective_depth_weight*prior_loss + options.depth_coverage_weight*ramp*coverage_loss
                    loss = loss + geometry_objective
                    depth_supervised_steps += 1
                    learned_pixels = int((target_depth["source_kind"] == 2).sum())
                    if learned_pixels:
                        learned_depth_supervised_steps += 1
                        learned_depth_pixel_observations += learned_pixels
                        if not learned_depth_gradient_verified:
                            learned_objective = options.depth_prior_relative_weight*(effective_depth_weight*terms["learned"] + options.depth_coverage_weight*ramp*terms["learned_coverage"])
                            gradients = torch.autograd.grad(learned_objective, [params[key] for key in ("means", "scales", "opacities")], retain_graph=True, allow_unused=True)
                            learned_gradients = {key: float(gradient.norm()) if gradient is not None else 0. for key, gradient in zip(("learned_depth_gradient_means", "learned_depth_gradient_log_scales", "learned_depth_gradient_opacity_logits"), gradients)}
                            if not all(math.isfinite(value) for value in learned_gradients.values()):
                                raise FloatingPointError("Nonfinite validated learned depth gradient")
                            learned_depth_gradient_verified = learned_gradients["learned_depth_gradient_means"] > 0
                            gradient_record.update(learned_gradients)
                    if not depth_gradient_verified:
                        gradients = torch.autograd.grad(geometry_objective, [params[key] for key in ("means", "scales", "opacities")], retain_graph=True, allow_unused=True)
                        gradient_record.update({key: float(gradient.norm()) if gradient is not None else 0. for key, gradient in zip(("depth_gradient_means", "depth_gradient_log_scales", "depth_gradient_opacity_logits"), gradients)})
                        if not all(math.isfinite(value) for value in gradient_record.values()):
                            raise FloatingPointError("Nonfinite sparse depth gradient")
                        depth_gradient_verified = gradient_record["depth_gradient_means"] > 0
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                for optimizer in optimizers.values():
                    optimizer.step()
                strategy_post_backward(options, strategy, params, optimizers, strategy_state, step, info, lr=lr)
                completed = step + 1
                if completed == 1 or completed % options.log_every == 0 or completed == options.steps or gradient_record:
                    if not all(torch.isfinite(value).all() for value in params.values()):
                        raise FloatingPointError("Nonfinite Gaussian parameter")
                    row = dict(event="training", step=completed, station_id=frame["station_id"], frame=frame["frame"],
                               gaussians=len(params["means"]), loss=float(loss.detach()), rgb_l1=float(l1.detach()),
                               dssim=float(dssim.detach()), regularization=float(regularization.detach()), elapsed_s=time.monotonic()-start,
                               sparse_depth_loss=float(actual_depth_loss.detach()), depth_moment_loss=float(prior_loss.detach()), depth_coverage_loss=float(coverage_loss.detach()),
                               learned_depth_loss=float(learned_loss.detach()), learned_depth_supervised_steps=learned_depth_supervised_steps,
                               learned_depth_pixel_observations=learned_depth_pixel_observations, learned_depth_gradient_verified=learned_depth_gradient_verified,
                               depth_effective_weight=effective_depth_weight, depth_supervised_steps=depth_supervised_steps,
                               depth_gradient_verified=depth_gradient_verified, **gradient_record)
                    history.write(json.dumps(row, allow_nan=False) + "\n")
                    history.flush()
                    print(json.dumps(row), flush=True)
                    status.update(completed_steps=completed, current_gaussians=len(params["means"]), updated_utc=_now())
                    write_json(manifest_path, status)
                if completed % options.checkpoint_every == 0:
                    checkpoint(completed)
        checkpoint(options.steps)
        if options.depth_moment_weight and (not depth_supervised_steps or not depth_gradient_verified):
            raise RuntimeError("Depth candidate has no verified nonzero geometry gradient; not a completed candidate")
        if options.depth_use_validated_prior and options.depth_moment_weight and (not learned_depth_supervised_steps or not learned_depth_gradient_verified):
            raise RuntimeError("Learned depth candidate has no verified nonzero learned geometry gradient")
        arrays = {key: value.detach().cpu().numpy() for key, value in params.items()}
        means_m = arrays["means"].astype(np.float64)*radius + center
        log_scales_m = arrays["scales"].astype(np.float64) + math.log(radius)
        artifact = write_model(output / "model.ply", means=means_m, log_scales=log_scales_m, quats=arrays["quats"],
                               opacity_logits=arrays["opacities"], sh0=arrays["sh0"], shN=arrays["shN"])
        measured = _evaluate(params, heldout, output, rasterization, torch, options, radius, artifact["sha256"])
        write_json(output / "heldout_metrics.json", measured)
        depth_assessment = _evaluate_sparse_depth(params, heldout, sparse, sparse_provenance, output,
            rasterization, torch, options, radius, artifact["sha256"], train_stations)
        trusted_evaluation = evaluation_manifest(heldout, sparse, sparse_provenance, train_stations, processing)
        write_json(output / "evaluation_manifest.json", trusted_evaluation)
        geometry = geometry_statistics(means_m, np.exp(log_scales_m), 1/(1+np.exp(-np.clip(arrays["opacities"], -80, 80))),
                                       scene_radius_m=radius, large_sigma_fraction=options.large_sigma_fraction)
        geometry["heldout_observation_consistency"] = dict(status=depth_assessment["status"], summary=depth_assessment["summary"],
            scope=depth_assessment["interpretation"], independent_geometry_truth=False)
        status.update(status="completed", completed_steps=options.steps, completed_utc=_now(), duration_s=time.monotonic()-start,
                      artifact=artifact, checkpoint_sha256=sha256(output / "checkpoint_last.pt"),
                      depth_gradient_verified=depth_gradient_verified, depth_supervised_steps=depth_supervised_steps,
                      learned_depth_gradient_verified=learned_depth_gradient_verified, learned_depth_supervised_steps=learned_depth_supervised_steps,
                      learned_depth_pixel_observations=learned_depth_pixel_observations,
                      selection=(dict(accepted_model=None, accepted_model_sha256=None, selected="candidate_pending_comparison",
                                      candidate_model="model.ply", candidate_model_sha256=artifact["sha256"],
                                      basis="Depth-regularized candidate trained; matched baseline and heldout consistency/appearance comparison required")
                                 if options.depth_moment_weight else
                                 dict(accepted_model="model.ply", accepted_model_sha256=artifact["sha256"], selected="baseline",
                                      basis="Completed finite baseline; no quality candidate compared or claimed superior")),
                      quality=dict(heldout=measured["summary"], geometry=geometry, quality_improved=False,
                                   candidate_comparison="not_run", sky_coverage=measured["summary"]["sky"],
                                   limitation=("Sky is excluded from photometric supervision; empty sky metrics are unmeasured, not perfect scores. Residual sky-like splats are not certified absent." if processing['remove_sky'] else "Foreground SfM initialization has no dedicated sky shell; uncovered sky remains measurable")))
        write_json(manifest_path, status)
        return status
    except BaseException as error:
        status.update(status="failed", error=str(error), error_type=type(error).__name__, failed_utc=_now())
        write_json(manifest_path, status)
        raise
