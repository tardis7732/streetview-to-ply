import copy
import json
import math
import os

import numpy as np
from PIL import Image
import pytest

from tools.streetview_engine.training import (TrainingSettings, masked_losses, normalization,
    load_sparse_depth, load_validated_depth_prior, merge_training_depth, training_depth_losses,
    prepare_frame, sparse_observation_maps, station_uniform_schedule, validate_dataset_manifest, validate_splits)
from tools.streetview_engine.export import run as export_run, sha256, validate_ply, write_json, write_model


def frame(station, x=0., path=None):
    pose = np.eye(4)
    pose[0, 3] = x
    return dict(station_id=station, file_path=path or f"{station}.png", mask_path="mask.png", w=32, h=32,
                fl_x=16., fl_y=16., cx=16., cy=16., transform_matrix=pose.tolist(), face="down")


def test_settings_operator_override_and_small_actual_smoke_budget():
    options = TrainingSettings.from_inputs(dict(training_steps=6000, resolution=768, max_splats=10000),
                                           dict(training=dict(steps=30, resolution=128, max_splats=100)))
    assert options.steps == 30 and options.resolution == 128 and options.max_splats == 100
    with pytest.raises(ValueError):
        TrainingSettings(steps=0)
    with pytest.raises(ValueError):
        TrainingSettings.from_inputs({}, dict(training=dict(scene_id="forbidden")))


def test_physical_station_holdout_detects_different_capture_same_station():
    train = [frame("a", path="capture1_down.png"), frame("b", 3)]
    heldout = [frame("a", path="capture2_front.png")]
    with pytest.raises(ValueError, match="Physical-station"):
        validate_splits(train, heldout)
    heldout[0]["station_id"] = "c"
    assert validate_splits(train, heldout) == ({"a", "b"}, {"c"})


def test_station_sampling_is_not_face_count_weighted():
    frames = [frame("a")] + [frame("b", 3, path=f"b{i}.png") for i in range(9)]
    indices = station_uniform_schedule(frames, 20000, 19)
    assert .48 < (indices == 0).mean() < .52
    assert np.array_equal(indices, station_uniform_schedule(frames, 20000, 19))


def test_coordinate_translation_scale_invariance_and_mask_support(tmp_path):
    Image.fromarray(np.full((32, 32, 3), 127, np.uint8)).save(tmp_path / "a.png")
    mask = np.full((32, 32), 255, np.uint8)
    mask[16, 16] = 0
    Image.fromarray(mask).save(tmp_path / "mask.png")
    originals = [frame("a", -3), frame("b", 3)]
    center, radius = normalization(originals)
    base = prepare_frame(originals[0], tmp_path, 32, center, radius)
    transformed = copy.deepcopy(originals)
    offset = np.array([10., -4., 80.])
    for item in transformed:
        pose = np.asarray(item["transform_matrix"])
        pose[:3, 3] = pose[:3, 3] * 7 + offset
        item["transform_matrix"] = pose.tolist()
    center2, radius2 = normalization(transformed)
    actual = prepare_frame(transformed[0], tmp_path, 32, center2, radius2)
    np.testing.assert_allclose(base["K"], actual["K"])
    np.testing.assert_allclose(base["view"], actual["view"], atol=1e-7)
    assert math.isclose(radius2, 7*radius)
    assert base["mask"].sum() == 1023
    assert not base["ssim_mask"][11:22, 11:22].any()
    assert not base["ssim_mask"][:5].any()


def test_masked_rgb_ssim_has_no_gradient_from_dynamic_pixels():
    torch = pytest.importorskip("torch")
    from scipy.ndimage import minimum_filter
    mask_np = np.ones((24, 24), bool)
    mask_np[11:14, 11:14] = False
    mask = torch.from_numpy(mask_np)
    ssim_mask = torch.from_numpy(minimum_filter(mask_np.astype(np.uint8), size=11, mode="constant", cval=0).astype(bool))
    reference = torch.full((24, 24, 3), .4)
    prediction = reference.clone()
    prediction[~mask] = .95
    prediction.requires_grad_()
    l1, dssim = masked_losses(prediction, reference, mask, ssim_mask)
    (l1+dssim).backward()
    assert l1.item() == 0 and abs(dssim.item()) < 1e-5
    assert torch.equal(prediction.grad[~mask], torch.zeros_like(prediction.grad[~mask]))


@pytest.mark.parametrize("face_label", ["D", "down"])
def test_evaluation_classifies_actual_engine_down_face_label(tmp_path, monkeypatch, face_label):
    torch = pytest.importorskip("torch")
    from tools.streetview_engine.training import _evaluate
    monkeypatch.setattr("tools.streetview_engine.training._render", lambda *args: (torch.full((1, 4, 4, 3), .5), torch.ones((1, 4, 4, 1)), {}))
    item = dict(frame="down.png", station_id="physical", meta=dict(face=face_label), w=4, h=4,
                rgb=np.full((4, 4, 3), 128, np.uint8), mask=np.ones((4, 4), bool), reference_rgb_sha256="rgb", reference_mask_sha256="mask")
    result = _evaluate({}, [item], tmp_path, None, torch, TrainingSettings(), 1., "model")
    assert result["summary"]["down"]["static_pixels"] == 16


def test_train_only_seed_color_provenance_required():
    manifest = dict(coordinate_frame="EDN", units="metres", camera_convention="OpenGL_c2w",
                    seed_colors_exclude_heldout=True, seed_color_station_ids=["a", "b"])
    validate_dataset_manifest(manifest, {"a", "b"}, {"c"})
    manifest["seed_color_station_ids"].append("c")
    with pytest.raises(ValueError, match="Seed colors"):
        validate_dataset_manifest(manifest, {"a", "b"}, {"c"})


def model(path):
    shN = np.arange(48, dtype=np.float32).reshape(2, 8, 3) / 100
    result = write_model(path, means=np.array([[0., 1., 2.], [3., 4., 5.]]), log_scales=np.full((2, 3), -2.),
                        quats=np.array([[2., 0, 0, 0], [1., 0, 0, 0]]), opacity_logits=np.zeros(2),
                        sh0=np.zeros((2, 1, 3)), shN=shN)
    return result, shN


def test_standard_ply_channel_order_and_bound_export(tmp_path):
    training = tmp_path / "training"
    training.mkdir()
    artifact, shN = model(training / "model.ply")
    raw = (training / "model.ply").read_bytes()
    payload = raw.split(b"end_header\n", 1)[1]
    data = np.frombuffer(payload, dtype="<f4").reshape(2, -1)
    np.testing.assert_array_equal(data[:, 9:33], shN.transpose(0, 2, 1).reshape(2, -1))
    np.testing.assert_allclose(data[:, -4:], [[1, 0, 0, 0], [1, 0, 0, 0]])
    write_json(training / "manifest.json", dict(status="completed", selection=dict(accepted_model="model.ply", accepted_model_sha256=artifact["sha256"], selected="baseline")))
    delivered = export_run({}, tmp_path, {})
    assert delivered["artifact"]["vertex_count"] == 2 and delivered["artifact"]["sh_degree"] == 2
    assert (tmp_path / "export/scene.ply").read_bytes() == raw
    assert export_run({}, tmp_path, {}) == delivered
    with (training / "model.ply").open("ab") as stream:
        stream.write(b"bad")
    with pytest.raises(ValueError):
        export_run({}, tmp_path, {})


def test_invalid_gaussian_export_never_reports_valid(tmp_path):
    artifact, _ = model(tmp_path / "model.ply")
    raw = (tmp_path / "model.ply").read_bytes()
    (tmp_path / "bad.ply").write_bytes(raw[:-1])
    with pytest.raises(ValueError, match="size"):
        validate_ply(tmp_path / "bad.ply")


def sparse_rows():
    return dict(xy=np.array([[8.5, 8.5], [8.7, 8.6], [16.5, 16.5], [24.5, 24.5]]),
                depth_z=np.array([3., 6., 4., 5.]), point_id=np.arange(4),
                support_station_count=np.array([2, 2, 3, 3]), triangulation_angle_degrees=np.full(4, 10.),
                reprojection_error_px=np.full(4, .5))


def test_sparse_actual_observations_are_not_upsampled_and_collisions_abstain():
    options = TrainingSettings(resolution=32, depth_resolution=16)
    mask = np.ones((32, 32), bool)
    mask[24, 24] = False
    maps = sparse_observation_maps(sparse_rows(), frame("a"), mask, options)
    assert maps["width"] == maps["height"] == 16
    assert maps["ambiguous_collision_pixels"] == 1
    assert maps["accepted_pixels"] == 1
    assert maps["depth_z"][8, 8] == 4
    assert maps["point_ids"][8, 8] == 2
    # A single actual observation stays a single supervision pixel.
    assert np.count_nonzero(maps["valid"]) == 1


def test_sparse_geometry_scale_invariance_and_confidence():
    rows = sparse_rows()
    rows["depth_z"][1] = 3.01
    options = TrainingSettings(resolution=32, depth_resolution=16)
    first = sparse_observation_maps(rows, frame("a"), np.ones((32, 32), bool), options)
    scaled = {key: value.copy() for key, value in rows.items()}
    scaled["depth_z"] *= 17
    second = sparse_observation_maps(scaled, frame("a"), np.ones((32, 32), bool), options)
    np.testing.assert_array_equal(first["valid"], second["valid"])
    np.testing.assert_allclose(first["depth_z"]*17, second["depth_z"], rtol=1e-6)
    np.testing.assert_array_equal(first["confidence"], second["confidence"])


def test_positive_depth_weight_requires_coverage_and_actual_observation_sidecar(tmp_path):
    with pytest.raises(ValueError, match="coverage"):
        TrainingSettings(depth_moment_weight=.01, depth_coverage_weight=0.)
    with pytest.raises(ValueError, match="actual-observation"):
        load_sparse_depth(tmp_path, [frame("a"), frame("b", 3)], [], TrainingSettings(depth_moment_weight=.01))
    assert load_sparse_depth(tmp_path, [frame("a"), frame("b", 3)], [], TrainingSettings())[1]["status"] == "unassessed"


def test_sparse_sidecar_split_hash_and_train_target_separation(tmp_path):
    train, heldout = [frame("a", -1), frame("b", 1)], [frame("c", 0)]
    for item in train + heldout:
        item["foreground_mask_path"] = "mask.png"
    Image.fromarray(np.full((32, 32), 255, np.uint8)).save(tmp_path / "mask.png")
    write_json(tmp_path / "transforms_train.json", dict(frames=train))
    write_json(tmp_path / "transforms_heldout.json", dict(frames=heldout))
    np.savez(tmp_path / "init_points.npz", xyz=np.zeros((4, 3)))
    rows = sparse_rows()
    rows.update(frame_name=np.array(["a.png", "a.png", "b.png", "c.png"]), station_id=np.array(["a", "a", "b", "c"]), split=np.array(["train", "train", "train", "heldout"]))
    np.savez(tmp_path / "sparse_depth_observations.npz", **rows)
    manifest = dict(status="supported_observations", coordinate_frame="EDN", units="metres", depth_convention="camera_z", pixel_center_offset=.5,
        observation_kind="actual_sfm_tracks", geometry_scope="transductive_shared_sfm", npz="sparse_depth_observations.npz", sha256=sha256(tmp_path / "sparse_depth_observations.npz"),
        transforms_train_sha256=sha256(tmp_path / "transforms_train.json"), transforms_heldout_sha256=sha256(tmp_path / "transforms_heldout.json"),
        seed_npz_sha256=sha256(tmp_path / "init_points.npz"), train_station_ids=["a", "b"], heldout_station_ids=["c"])
    write_json(tmp_path / "sparse_depth_manifest.json", manifest)
    mapped, provenance = load_sparse_depth(tmp_path, train, heldout, TrainingSettings(depth_moment_weight=.01, depth_resolution=16, resolution=32))
    assert mapped["c.png"]["split"] == "heldout"
    assert provenance["totals"]["train"]["pixels"] == 1
    assert provenance["totals"]["heldout"]["pixels"] == 1
    manifest["heldout_station_ids"] = ["a"]
    write_json(tmp_path / "sparse_depth_manifest.json", manifest)
    with pytest.raises(ValueError, match="split"):
        load_sparse_depth(tmp_path, train, heldout, TrainingSettings())


def learned_fixture(tmp_path, *, side=16, scale=1., accepted=True, excluded_mask_pixel=False):
    """Synthetic adapter contract only; no inference/geometry quality claim."""
    dataset, prior = tmp_path / "sfm/dataset", tmp_path / "depth_prior"
    dataset.mkdir(parents=True)
    prior.mkdir()
    mask = np.full((32, 32), 255, np.uint8)
    if excluded_mask_pixel:
        mask[8, 8] = 0
    Image.fromarray(mask).save(dataset / "mask.png")
    train = [frame(name, x*scale) for name, x in (("a", -1), ("b", 0), ("c", 1))]
    heldout = [frame("d", -.3*scale), frame("e", .3*scale)]
    for item in train+heldout:
        item.update(pano_id="capture_"+item["station_id"], sfm_mask_path="mask.png", foreground_mask_path="mask.png")
        Image.fromarray(np.full((32, 32, 3), 127, np.uint8)).save(dataset / item["file_path"])
        item.update(image_sha256=sha256(dataset/item["file_path"]), sfm_mask_sha256=sha256(dataset/"mask.png"))
    for split, items in (("train", train), ("heldout", heldout)):
        write_json(dataset / f"transforms_{split}.json", dict(frames=items))
    write_json(dataset/"dataset_manifest.json", dict(fixture=True))
    np.savez(dataset/"init_points.npz", xyz=np.zeros((4, 3)))
    rows = sparse_rows()
    rows["depth_z"] *= scale
    rows.update(frame_name=np.array(["a.png", "a.png", "a.png", "d.png"]), station_id=np.array(["a", "a", "a", "d"]), split=np.array(["train", "train", "train", "heldout"]))
    np.savez(dataset/"sparse_depth_observations.npz", **rows)
    write_json(dataset/"sparse_depth_manifest.json", dict(status="supported_observations", coordinate_frame="EDN", units="metres", depth_convention="camera_z", pixel_center_offset=.5,
        observation_kind="actual_sfm_tracks", geometry_scope="transductive_shared_sfm", npz="sparse_depth_observations.npz", sha256=sha256(dataset/"sparse_depth_observations.npz"),
        transforms_train_sha256=sha256(dataset/"transforms_train.json"), transforms_heldout_sha256=sha256(dataset/"transforms_heldout.json"),
        seed_npz_sha256=sha256(dataset/"init_points.npz"), train_station_ids=["a", "b", "c"], heldout_station_ids=["d", "e"]))
    bindings = {p.relative_to(dataset).as_posix(): sha256(p) for p in dataset.iterdir() if p.is_file()}
    for name in ("input_manifest.json", "calibration_report.json", "raw/provenance.json"):
        write_json(prior/name, dict(synthetic=True))
    entries = []
    for item in train+heldout:
        active = accepted and item is train[0]
        K = np.array([[side/2, 0, side/2], [0, side/2, side/2], [0, 0, 1]], np.float64)
        view = np.linalg.inv(np.asarray(item["transform_matrix"]) @ np.diag([1., -1., -1., 1.]))
        split = "train" if item in train else "heldout"
        entry = dict(frame_name=item["file_path"], station_id=item["station_id"], pano_id=item["pano_id"], split=split,
            status="accepted" if active else "insufficient_support", valid_count=side*side if active else 0, width=side, height=side,
            K=K.tolist(), camera_from_world=view.tolist(), image_sha256=item["image_sha256"], sfm_mask_sha256=item["sfm_mask_sha256"],
            support_by_other_station={"b": side*side, "c": side*side} if active else {}, npz=item["station_id"]+".npz")
        if active:
            # Poisoned diagnostic object data must never even be accessed.
            np.savez(prior/entry["npz"], depth_z=np.full((side, side), 9*scale, np.float32), valid=np.ones((side, side), bool),
                confidence=np.full((side, side), .25, np.float32), source_count=np.full((side, side), 2, np.uint16),
                candidate_depth_z=np.array([{"forbidden": True}], object), candidate_valid=np.ones((1, 1), bool),
                K=K, camera_from_world=view, world_frame="EDN", units="metres", depth_convention="camera_z", pixel_center_offset=.5,
                frame_name=item["file_path"], station_id=item["station_id"], pano_id=item["pano_id"], split=split,
                evidence="calibrated_UniSHARP_first_surface_multistation_consistency")
            entry["sha256"] = sha256(prior/entry["npz"])
        entries.append(entry)
    manifest = dict(schema_version=1, status="complete", stage="depth_prior", validation_status="multistation_consistent_subset" if accepted else "insufficient_supported_depth",
        coordinate_frame="EDN", units="metres", depth_convention="camera_z", pixel_center_offset=.5, camera_convention="OpenCV_world_to_camera",
        geometry_scope="transductive_shared_sfm_calibration", minimum_other_station_support=2,
        train_station_ids=["a", "b", "c"], heldout_station_ids=["d", "e"], entries=entries, accepted_pixels=side*side if accepted else 0,
        source_dataset_files_sha256=bindings, source_transforms_sha256={split: bindings[f"transforms_{split}.json"] for split in ("train", "heldout")},
        input_manifest_sha256=sha256(prior/"input_manifest.json"), calibration_report_sha256=sha256(prior/"calibration_report.json"), provenance_sha256=sha256(prior/"raw/provenance.json"))
    write_json(prior/"manifest.json", manifest)
    return dataset, prior, train, heldout, manifest


def test_validated_prior_merges_actual_first_never_candidate_or_heldout(tmp_path):
    dataset, prior, train, heldout, manifest = learned_fixture(tmp_path)
    options = TrainingSettings(depth_use_validated_prior=True, depth_moment_weight=.01, depth_resolution=16, resolution=32)
    learned, provenance, binding = load_validated_depth_prior(tmp_path, dataset, train, heldout, options)
    sparse, _ = load_sparse_depth(dataset, train, heldout, options)
    heldout_before = {key: value.copy() for key, value in sparse["d.png"].items() if isinstance(value, np.ndarray)}
    merged, stats = merge_training_depth(sparse, learned)
    assert set(merged) == {"a.png"} and set(learned) == {"a.png"}
    assert not merged["a.png"]["valid"][4, 4]  # actual observations disagree: no learned fill
    assert merged["a.png"]["depth_z"][8, 8] == 4  # actual wins learned 9m
    assert merged["a.png"]["depth_z"][1, 1] == 9
    assert merged["a.png"]["source_kind"][8, 8] == 1
    assert merged["a.png"]["source_kind"][1, 1] == 2
    assert stats == dict(actual_pixels=1, learned_pixels=254, sparse_overrides=1, ambiguous_vetoes=1)
    assert provenance["source_count_histogram"] == {"2": 256}
    assert binding["a.npz"] == manifest["entries"][0]["sha256"]
    for key, value in heldout_before.items():
        np.testing.assert_array_equal(sparse["d.png"][key], value)


@pytest.mark.parametrize("mutation,match", [
    ("artifact", "artifact changed"), ("source", "source changed"), ("camera", "camera/K"),
    ("self_station", "OTHER training"), ("heldout", "Heldout learned"),
    ("count", "support"), ("confidence", "confidence"), ("photo", "photo/mask"),
    ("status", "status"), ("path", "escaped path"), ("missing_binding", "portable"),
])
def test_validated_prior_rejects_invalid_support_provenance_and_leakage(tmp_path, mutation, match):
    dataset, prior, train, heldout, manifest = learned_fixture(tmp_path)
    entry = manifest["entries"][0]
    if mutation == "artifact":
        with (prior/"a.npz").open("ab") as stream: stream.write(b"changed")
    elif mutation == "source":
        Image.fromarray(np.zeros((32, 32, 3), np.uint8)).save(dataset/"a.png")
    elif mutation == "camera": entry["K"][0][0] += 1
    elif mutation == "self_station": entry["support_by_other_station"] = {"a": 256, "b": 256}
    elif mutation == "heldout": manifest["entries"][3].update(valid_count=1, status="accepted")
    elif mutation in ("count", "confidence"):
        with np.load(prior/"a.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files if not key.startswith("candidate_")}
        arrays["source_count" if mutation == "count" else "confidence"][2, 2] = 1 if mutation == "count" else .8
        np.savez(prior/"a.npz", **arrays)
        entry["sha256"] = sha256(prior/"a.npz")
    elif mutation == "photo": entry["image_sha256"] = "wrong"
    elif mutation == "status": entry["status"] = "candidate"
    elif mutation == "path": entry["npz"] = "../../escaped.npz"
    elif mutation == "missing_binding": del manifest["source_dataset_files_sha256"]["init_points.npz"]
    write_json(prior/"manifest.json", manifest)
    with pytest.raises(ValueError, match=match):
        load_validated_depth_prior(tmp_path, dataset, train, heldout, TrainingSettings(depth_use_validated_prior=True, depth_moment_weight=.01))


def test_empty_learned_prior_is_bound_but_zero_weight_baseline_stays_valid(tmp_path):
    dataset, prior, train, heldout, manifest = learned_fixture(tmp_path, accepted=False)
    maps, evidence, binding = load_validated_depth_prior(tmp_path, dataset, train, heldout, TrainingSettings(depth_use_validated_prior=True))
    assert maps == {} and not evidence["used_for_optimization"] and binding["manifest.json"] == sha256(prior/"manifest.json")
    with pytest.raises(ValueError, match="No accepted learned"):
        load_validated_depth_prior(tmp_path, dataset, train, heldout, TrainingSettings(depth_use_validated_prior=True, depth_moment_weight=.01))


def test_learned_foreground_is_checked_against_original_full_source_footprint(tmp_path):
    dataset, prior, train, heldout, manifest = learned_fixture(tmp_path, excluded_mask_pixel=True)
    with pytest.raises(ValueError, match="excluded sky/dynamic"):
        load_validated_depth_prior(tmp_path, dataset, train, heldout, TrainingSettings(depth_use_validated_prior=True, depth_moment_weight=.01))


def test_learned_native_grid_scale_invariance_and_sparse_heldout_grid_unchanged(tmp_path):
    outputs = []
    for label, scale in (("base", 1.), ("scaled", 17.)):
        root = tmp_path/label
        dataset, prior, train, heldout, _ = learned_fixture(root, side=16, scale=scale)
        options = TrainingSettings(depth_use_validated_prior=True, depth_moment_weight=.01, depth_resolution=32, resolution=32)
        learned, _, _ = load_validated_depth_prior(root, dataset, train, heldout, options)
        sparse, _ = load_sparse_depth(dataset, train, heldout, options, train_max_resolution=16)
        assert sparse["a.png"]["width"] == 16 and sparse["d.png"]["width"] == 32
        assert learned["a.png"]["valid"].sum() == 256  # never expands 16 to32
        outputs.append(merge_training_depth(sparse, learned)[0]["a.png"])
    np.testing.assert_array_equal(outputs[0]["valid"], outputs[1]["valid"])
    np.testing.assert_allclose(outputs[0]["depth_z"]*17, outputs[1]["depth_z"])
    np.testing.assert_array_equal(outputs[0]["confidence"], outputs[1]["confidence"])


def test_learned_downsampling_requires_complete_valid_consistent_blocks():
    from tools.streetview_engine.training import _resize_accepted_prior
    arrays = dict(depth_z=np.full((32, 32), 4., np.float32), valid=np.ones((32, 32), bool), confidence=np.full((32, 32), .4, np.float32), source_count=np.full((32, 32), 3, np.uint16))
    arrays["valid"][4, 4] = False
    arrays["depth_z"][10, 10] = 8
    arrays["source_count"][20, 20] = 2
    actual = _resize_accepted_prior(arrays, 16, 16, .02)
    assert not actual["valid"][2, 2] and not actual["valid"][5, 5]
    assert actual["source_count"][10, 10] == 2 and actual["valid"].sum() == 254


def test_dense_learned_pixel_count_does_not_overwhelm_actual_loss_budget():
    torch = pytest.importorskip("torch")
    options = TrainingSettings(depth_use_validated_prior=True, depth_prior_relative_weight=.25)
    losses = []
    for learned_count in (1, 1000):
        kinds = np.array([[1]+[2]*learned_count], np.uint8)
        target = dict(split="train", source_kind=kinds, valid=np.ones_like(kinds, bool), confidence=np.array([[1.]+[.25]*learned_count], np.float32),
                      depth_z=np.ones_like(kinds, np.float32), reference_alpha=np.ones_like(kinds, np.float32))
        z = torch.full(kinds.shape, 2., requires_grad=True)
        moments = dict(first_moment=z, second_moment=z*z, alpha=torch.ones_like(z))
        terms = training_depth_losses(moments, target, 1., options)
        terms["depth"].backward()
        assert torch.isfinite(z.grad).all() and (z.grad > 0).all()
        assert terms["depth"].item() == pytest.approx(1.25*(math.sqrt(2)-1))
        losses.append(terms["depth"].item())
        target["split"] = "heldout"
        with pytest.raises(ValueError, match="Heldout"):
            training_depth_losses(moments, target, 1., options)
    assert losses[0] == pytest.approx(losses[1])


@pytest.mark.skipif(os.environ.get("STREETVIEW_RUN_CUDA") != "1", reason="Explicit opt-in actual CUDA training; no GPU work in CPU suite")
def test_actual_cuda_training_and_export(tmp_path):
    """Independent synthetic contract fixture; never downloads scene data."""
    from tools.streetview_engine.training import run
    dataset = tmp_path / "sfm/dataset"
    dataset.mkdir(parents=True)
    Image.fromarray(np.full((32, 32), 255, np.uint8)).save(dataset / "mask.png")
    frames = []
    for index, x in enumerate((-1., 1., -.3, .3)):
        name = f"synthetic_{index}.png"
        Image.fromarray(np.full((32, 32, 3), 90, np.uint8)).save(dataset / name)
        frames.append(frame(str(index), x, name))
    train, heldout = frames[:2], frames[2:]
    write_json(dataset / "transforms_train.json", dict(frames=train))
    write_json(dataset / "transforms_heldout.json", dict(frames=heldout))
    write_json(dataset / "dataset_manifest.json", dict(coordinate_frame="EDN", units="metres", camera_convention="OpenGL_c2w",
        seed_colors_exclude_heldout=True, seed_color_station_ids=["0", "1"], metric_alignment=dict(status="synthetic_known_metres")))
    xx, yy = np.meshgrid(np.linspace(-2, 2, 8), np.linspace(-2, 2, 8))
    xyz = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, -3.)], axis=1)
    np.savez(dataset / "init_points.npz", xyz=xyz, rgb=np.full((len(xyz), 3), 90, np.uint8), support_station_count=np.full(len(xyz), 2, np.int32))
    settings = dict(training=dict(steps=30, resolution=32, max_splats=64, checkpoint_every=30, log_every=30,
                                  noise_lr=0., opacity_reg=0., scale_reg=0.))
    trained = run({}, tmp_path, settings)
    delivered = export_run({}, tmp_path, settings)
    assert trained["status"] == "completed" and trained["completed_steps"] == 30
    assert trained["quality"]["heldout"]["physical_stations"] == 2
    assert delivered["artifact"]["sha256"] == trained["selection"]["accepted_model_sha256"]
    history = [json.loads(line) for line in (tmp_path / "training/metrics_history.jsonl").read_text().splitlines()]
    assert history[-1]["rgb_l1"] < history[0]["rgb_l1"]
    assert run({}, tmp_path, settings)["signature"] == trained["signature"]
    Image.fromarray(np.zeros((32, 32, 3), np.uint8)).save(dataset / train[0]["file_path"])
    with pytest.raises(ValueError, match="Existing training differs"):
        run({}, tmp_path, settings)


@pytest.mark.skipif(os.environ.get("STREETVIEW_RUN_CUDA") != "1", reason="Explicit opt-in actual sparse-depth CUDA integration")
def test_actual_cuda_sparse_depth_candidate(tmp_path):
    """Exercise the native-resolution moment branch, not scene reconstruction."""
    from tools.streetview_engine.training import run
    dataset = tmp_path / "sfm/dataset"
    dataset.mkdir(parents=True)
    Image.fromarray(np.full((32, 32), 255, np.uint8)).save(dataset / "mask.png")
    frames = []
    for index, x in enumerate((-1., 1., -.3, .3)):
        name = f"synthetic_{index}.png"
        Image.fromarray(np.full((32, 32, 3), 90, np.uint8)).save(dataset / name)
        item = frame(str(index), x, name)
        item["foreground_mask_path"] = "mask.png"
        frames.append(item)
    train, heldout = frames[:2], frames[2:]
    write_json(dataset / "transforms_train.json", dict(frames=train))
    write_json(dataset / "transforms_heldout.json", dict(frames=heldout))
    write_json(dataset / "dataset_manifest.json", dict(coordinate_frame="EDN", units="metres", camera_convention="OpenGL_c2w",
        seed_colors_exclude_heldout=True, seed_color_station_ids=["0", "1"], metric_alignment=dict(status="synthetic_known_metres")))
    xx, yy = np.meshgrid(np.linspace(-2, 2, 8), np.linspace(-2, 2, 8))
    near = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, -3.)], axis=1)
    far = np.stack([xx.ravel()*5/3, yy.ravel()*5/3, np.full(xx.size, -5.)], axis=1)
    xyz = np.concatenate([near, far])
    np.savez(dataset / "init_points.npz", xyz=xyz, rgb=np.full((len(xyz), 3), 90, np.uint8),
             support_station_count=np.full(len(xyz), 2, np.int32))
    table = {key: [] for key in ("frame_name", "station_id", "split", "point_id", "xy", "depth_z", "support_station_count", "triangulation_angle_degrees", "reprojection_error_px")}
    for index, item in enumerate(frames):
        center = np.asarray(item["transform_matrix"])[:3, 3]
        cv = (near-center)*np.array([1, -1, -1])
        xy = cv[:, :2]/cv[:, 2:]*16 + 16
        for pid in np.flatnonzero((xy >= 2).all(1) & (xy < 30).all(1)):
            values = (item["file_path"], item["station_id"], "train" if index < 2 else "heldout", int(pid), xy[pid],
                      float(cv[pid, 2]), 2, 10., .1)
            for key, value in zip(table, values):
                table[key].append(value)
    string_keys = {"frame_name", "station_id", "split"}
    integer_keys = {"point_id", "support_station_count"}
    arrays = {key: np.asarray(value, dtype=str if key in string_keys else np.int64 if key in integer_keys else np.float64) for key, value in table.items()}
    np.savez(dataset / "sparse_depth_observations.npz", **arrays)
    write_json(dataset / "sparse_depth_manifest.json", dict(status="supported_observations", coordinate_frame="EDN", units="metres",
        depth_convention="camera_z", pixel_center_offset=.5, observation_kind="actual_sfm_tracks", geometry_scope="transductive_shared_sfm",
        npz="sparse_depth_observations.npz", sha256=sha256(dataset / "sparse_depth_observations.npz"), seed_npz_sha256=sha256(dataset / "init_points.npz"),
        transforms_train_sha256=sha256(dataset / "transforms_train.json"), transforms_heldout_sha256=sha256(dataset / "transforms_heldout.json"),
        train_station_ids=["0", "1"], heldout_station_ids=["2", "3"], fixture_scope="Synthetic declared observations for CUDA integration only"))
    settings = dict(training=dict(steps=30, resolution=32, depth_resolution=16, max_splats=128,
        checkpoint_every=30, log_every=30, noise_lr=0., opacity_reg=0., scale_reg=0.,
        depth_moment_weight=.1, depth_coverage_weight=.05, depth_start_fraction=0., depth_ramp_fraction=.1))
    trained = run({}, tmp_path, settings)
    assert trained["status"] == "completed" and trained["depth_gradient_verified"]
    assert trained["depth_supervised_steps"] == 30
    assert trained["selection"]["selected"] == "candidate_pending_comparison"
    assert trained["selection"]["accepted_model"] is None
    history = [json.loads(line) for line in (tmp_path / "training/metrics_history.jsonl").read_text().splitlines()]
    gradient = next(row for row in history if "depth_gradient_means" in row)
    assert all(np.isfinite(gradient[key]) and gradient[key] > 0 for key in
               ("depth_gradient_means", "depth_gradient_log_scales", "depth_gradient_opacity_logits"))
    measured = json.loads((tmp_path / "training/heldout_depth_metrics.json").read_text())
    assert measured["summary"]["physical_stations"] == 2 and measured["used_for_optimization"] is False
    assert {row["station_id"] for row in measured["views"]} == {"2", "3"}
    with pytest.raises(ValueError, match="accepted model"):
        export_run({}, tmp_path, settings)


@pytest.mark.skipif(os.environ.get("STREETVIEW_RUN_CUDA") != "1", reason="Explicit opt-in accepted-learned-depth CUDA integration")
def test_actual_cuda_validated_learned_depth_candidate(tmp_path):
    """Synthetic loader→merge→packed loss→gradient→manifest integration only."""
    from tools.streetview_engine.training import run
    dataset, prior, train, heldout, manifest = learned_fixture(tmp_path)
    xx, yy = np.meshgrid(np.linspace(-4, 4, 8), np.linspace(-4, 4, 8))
    near = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, -8.)], axis=1)
    far = np.stack([xx.ravel()*10/8, yy.ravel()*10/8, np.full(xx.size, -10.)], axis=1)
    xyz = np.concatenate([near, far])
    np.savez(dataset/"init_points.npz", xyz=xyz, rgb=np.full((len(xyz), 3), 127, np.uint8), support_station_count=np.full(len(xyz), 2, np.int32))
    sidecar = json.loads((dataset/"sparse_depth_manifest.json").read_text())
    sidecar["seed_npz_sha256"] = sha256(dataset/"init_points.npz")
    write_json(dataset/"sparse_depth_manifest.json", sidecar)
    write_json(dataset/"dataset_manifest.json", dict(coordinate_frame="EDN", units="metres", camera_convention="OpenGL_c2w",
        seed_colors_exclude_heldout=True, seed_color_station_ids=["a", "b", "c"], metric_alignment=dict(status="synthetic_known_metres")))
    for name in ("init_points.npz", "sparse_depth_manifest.json", "dataset_manifest.json"):
        manifest["source_dataset_files_sha256"][name] = sha256(dataset/name)
    write_json(prior/"manifest.json", manifest)
    settings = dict(training=dict(steps=30, resolution=32, depth_resolution=16, max_splats=128, noise_lr=0.,
        opacity_reg=0., scale_reg=0., checkpoint_every=30, log_every=30, depth_start_fraction=0., depth_ramp_fraction=.1,
        depth_moment_weight=.1, depth_coverage_weight=.05, depth_use_validated_prior=True, depth_prior_relative_weight=.25))
    trained = run({}, tmp_path, settings)
    assert trained["status"] == "completed" and trained["learned_depth_gradient_verified"]
    assert trained["learned_depth_supervised_steps"] > 0 and trained["learned_depth_pixel_observations"] > 0
    assert trained["merged_training_depth"]["learned_pixels"] == 254
    history = [json.loads(line) for line in (tmp_path/"training/metrics_history.jsonl").read_text().splitlines()]
    gradient = next(row for row in history if "learned_depth_gradient_means" in row)
    assert all(np.isfinite(gradient[key]) and gradient[key] > 0 for key in
               ("learned_depth_gradient_means", "learned_depth_gradient_log_scales", "learned_depth_gradient_opacity_logits"))
    measured = json.loads((tmp_path/"training/heldout_depth_metrics.json").read_text())
    assert measured["used_for_optimization"] is False and {row["station_id"] for row in measured["views"]} == {"d"}
    assert trained["selection"]["accepted_model"] is None
