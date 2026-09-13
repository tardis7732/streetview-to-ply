import copy
import hashlib
import json

import numpy as np
from PIL import Image
import pytest
from scipy.spatial.transform import Rotation

from tools.streetview_engine.export import sha256, write_json, write_model
from tools.streetview_engine.inspection import InspectionConfig, _header, camera_baseline, inspect_arrays, inspect_scene, main


def frame(station, x, *, face="F", suffix=""):
    pose = np.eye(4)
    pose[0, 3] = x
    return dict(file_path=f"{station}_{face}{suffix}.png", station_id=station, pano_id=f"capture_{station}", face=face,
                transform_matrix=pose.tolist(), w=16, h=16, fl_x=8., fl_y=8., cx=8., cy=8.)


def seeds():
    x, y = np.meshgrid(np.arange(-1., 2.), np.arange(-1., 2.))
    xyz = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    return dict(xyz=xyz, point_ids=np.arange(len(xyz)), support_station_count=np.full(len(xyz), 3, np.int32),
                triangulation_angle_degrees=np.full(len(xyz), 10.), reprojection_error_px=np.full(len(xyz), .5))


def gaussian_chunks(means, logs, quats, size=65536):
    for start in range(0, len(means), size):
        stop = min(len(means), start+size)
        yield dict(means=means[start:stop], log_scales=logs[start:stop], quats=quats[start:stop],
                   row_ids=np.arange(start, stop), finite=np.ones(stop-start, bool))


def test_rotation_translation_scale_invariance_and_nonmutation():
    points = seeds()
    frames = [frame("a", -2), frame("b", 2), frame("c", 1)]
    means = np.array([[0., 0., 0.], [1., .5, 0.], [30., 0., 0.], [0., 0., 0.]])
    logs = np.log(np.array([[.1, .2, .3], [.001, .2, .3], [.1, .1, .1], [2., 3., 4.]]))
    rot = Rotation.from_euler("xyz", np.arange(12).reshape(4, 3)/10)
    quats = rot.as_quat()[:, [3, 0, 1, 2]]
    original = [array.copy() for array in (means, logs, quats)]
    for array in (means, logs, quats): array.setflags(write=False)
    baseline = camera_baseline(frames)["metres"]
    a = inspect_arrays(gaussian_chunks(means, logs, quats, 1), points, baseline)
    rotation = Rotation.from_euler("xyz", [.7, -.9, .3])
    matrix, shift, scale = rotation.as_matrix(), np.array([21., -7., 4.]), 13.
    transformed_frames = copy.deepcopy(frames)
    for item in transformed_frames:
        pose = np.asarray(item["transform_matrix"])
        pose[:3, :3] = matrix@pose[:3, :3]
        pose[:3, 3] = scale*(matrix@pose[:3, 3])+shift
        item["transform_matrix"] = pose.tolist()
    bpoints = dict(points, xyz=scale*(points["xyz"]@matrix.T)+shift)
    baseline_b = camera_baseline(transformed_frames)["metres"]
    b = inspect_arrays(gaussian_chunks(scale*(means@matrix.T)+shift, logs+np.log(scale), (rotation*rot).as_quat()[:, [3, 0, 1, 2]], 3), bpoints, baseline_b)
    assert baseline_b == pytest.approx(scale*baseline)
    assert a["counts"] == b["counts"] and a["review_candidates"] == b["review_candidates"]
    for name in ("axis_ratio", "extent_baseline_ratio", "nearest_support_spacing_ratio"):
        np.testing.assert_allclose(list(a["distributions"][name]["quantiles"].values()), list(b["distributions"][name]["quantiles"].values()), atol=1e-12)
    for name in ("principal_min", "principal_middle", "principal_max", "nearest_support_m"):
        np.testing.assert_allclose(np.array(list(a["distributions"][name]["quantiles"].values()))*scale, list(b["distributions"][name]["quantiles"].values()), atol=1e-12)
    for array, before in zip((means, logs, quats), original): np.testing.assert_array_equal(array, before)


def test_station_and_capture_grouping_prevents_cube_face_weighting():
    frames = [frame("a", -2), frame("b", 2), frame("c", 1)]
    other_capture = frame("a", -4, suffix="other")
    other_capture["pano_id"] = "second_capture"
    expected = camera_baseline(frames+[other_capture])
    actual = camera_baseline(frames+[other_capture]+[dict(frames[0], file_path=f"extra{i}.png") for i in range(20)])
    assert actual == expected
    assert actual["physical_station_count"] == 3


def fixture(root, *, ground=False):
    dataset = root/"dataset"
    dataset.mkdir(parents=True)
    train = [frame("a", -2), frame("a", -2, face="D"), frame("b", 2), frame("c", 1)]
    heldout = [frame("d", 0)]
    mask = np.full((16, 16), 255, np.uint8)
    Image.fromarray(mask).save(dataset/"mask.png")
    for item in train+heldout:
        item.update(ground_mask_path="mask.png", ground_mask_sha256=sha256(dataset/"mask.png"), sfm_mask_path="mask.png", sfm_mask_sha256=sha256(dataset/"mask.png"))
    for split, items in (("train", train), ("heldout", heldout)):
        write_json(dataset/f"transforms_{split}.json", dict(frames=items))
    np.savez(dataset/"init_points.npz", **seeds())
    if ground:
        np.savez(dataset/"sparse_depth_observations.npz", frame_name=np.array(["a_F.png", "a_F.png", "a_D.png", "b_F.png", "c_F.png", "d_F.png"]),
            station_id=np.array(["a", "a", "a", "b", "c", "d"]), split=np.array(["train"]*5+["heldout"]), point_id=np.array([0, 0, 0, 0, 1, 0]),
            xy=np.full((6, 2), 8.5), depth_z=np.full(6, 3.), reprojection_error_px=np.array([.5, .6, .5, .5, .5, .5]))
        write_json(dataset/"sparse_depth_manifest.json", dict(status="supported_observations", coordinate_frame="EDN", units="metres", observation_kind="actual_sfm_tracks", geometry_scope="transductive_shared_sfm", depth_convention="camera_z", pixel_center_offset=.5,
            transforms_train_sha256=sha256(dataset/"transforms_train.json"), transforms_heldout_sha256=sha256(dataset/"transforms_heldout.json"), seed_npz_sha256=sha256(dataset/"init_points.npz"),
            npz="sparse_depth_observations.npz", sha256=sha256(dataset/"sparse_depth_observations.npz")))
    files = {p.name: sha256(p) for p in dataset.iterdir()}
    write_json(dataset/"dataset_manifest.json", dict(coordinate_frame="EDN", units="metres", camera_convention="OpenGL_c2w", files=files))
    means = np.array([[0., 0., 0.], [1., 0., 0.], [20., 0., 0.], [0., 0., 0.]])
    scales = np.array([[.1, .2, .3], [.001, .2, .3], [.1, .1, .1], [10000., 10000., 1.]])
    ply = root/"scene.ply"
    write_model(ply, means=means, log_scales=np.log(scales), quats=np.tile([1., 0., 0., 0.], (4, 1)), opacity_logits=np.zeros(4), sh0=np.zeros((4, 1, 3)), shN=np.zeros((4, 8, 3)))
    return dataset, ply


def test_file_hash_binding_sky_exclusion_and_read_only_cli(tmp_path):
    dataset, ply = fixture(tmp_path)
    original = {str(path): sha256(path) for path in tmp_path.rglob("*") if path.is_file()}
    declaration = tmp_path/"components.json"
    write_json(declaration, dict(schema_version=1, model_sha256=sha256(ply), components=[dict(role="shared_angular_sky_fixed_geometry", start_row=3, row_count=1)]))
    all_rows = inspect_scene(ply, dataset)
    foreground = inspect_scene(ply, dataset, components=declaration, config=InspectionConfig(chunk_size=1))
    assert all_rows["foreground"]["counts"]["rows"] == 4
    assert foreground["foreground"]["counts"]["rows"] == 3 and foreground["declared_components"]["excluded_rows"] == 1
    assert all_rows["foreground"]["distributions"]["principal_max"]["quantiles"]["max"] > 9999
    assert foreground["foreground"]["distributions"]["principal_max"]["quantiles"]["max"] < 1
    assert not foreground["quality_accepted"] and not foreground["geometry_mutated"]
    assert main(["--ply", str(ply), "--dataset", str(dataset), "--components", str(declaration), "--output", str(tmp_path/"report.json")]) == 0
    for path, value in original.items(): assert sha256(path) == value
    with pytest.raises(SystemExit): main(["--ply", str(ply), "--dataset", str(dataset), "--output", str(ply)])


@pytest.mark.parametrize("mutation", ["hash", "suffix", "role", "source", "escape"])
def test_corrupt_or_unbound_inputs_are_rejected(tmp_path, mutation):
    dataset, ply = fixture(tmp_path)
    declaration = dict(schema_version=1, model_sha256=sha256(ply), components=[dict(role="shared_angular_sky_fixed_geometry", start_row=3, row_count=1)])
    if mutation == "hash": declaration["model_sha256"] = "wrong"
    elif mutation == "suffix": declaration["components"][0]["start_row"] = 1
    elif mutation == "role": declaration["components"][0]["role"] = "guessed_big_particles"
    elif mutation == "source": (dataset/"transforms_train.json").write_text("{}")
    elif mutation == "escape":
        manifest = json.loads((dataset/"dataset_manifest.json").read_text())
        manifest["files"]["init_points.npz"] = "wrong"
        (dataset/"init_points.npz").unlink()
        write_json(dataset/"dataset_manifest.json", manifest)
    write_json(tmp_path/"components.json", declaration)
    with pytest.raises(ValueError): inspect_scene(ply, dataset, components=tmp_path/"components.json")


def test_published_sky_manifest_requires_actual_prefix_identity(tmp_path):
    dataset, ply = fixture(tmp_path)
    count, fields, offset, _ = _header(ply)
    payload = ply.read_bytes()[offset:offset+3*len(fields)*4]
    identity = dict(candidate_sha256=sha256(ply), source_sha256="source", unchanged=True, identity="exact_original_payload_prefix_including_all_attributes", foreground_rows=3, sky_rows=1,
                    source_foreground_payload_sha256=hashlib.sha256(payload).hexdigest(), candidate_foreground_payload_sha256=hashlib.sha256(payload).hexdigest())
    write_json(tmp_path/"sky_comparison.json", dict(foreground_identity=identity))
    manifest = dict(status="completed", artifact=dict(sha256=sha256(ply)), selection=dict(accepted_model_sha256=sha256(ply), comparison_report="sky_comparison.json", comparison_sha256=sha256(tmp_path/"sky_comparison.json")),
        foreground_artifact=dict(sha256="source"), quality=dict(sky_component=dict(kind="shared_angular_sky_fixed_geometry", rows=1, foreground_rows=3, combined_rows=4, row_start=3, row_end_exclusive=4, geometry_is_measured=False, foreground_statistics_exclude_sky=True)))
    write_json(tmp_path/"manifest.json", manifest)
    assert inspect_scene(ply, dataset, components=tmp_path/"manifest.json")["foreground"]["counts"]["rows"] == 3
    identity["candidate_foreground_payload_sha256"] = identity["source_foreground_payload_sha256"] = "forged"
    write_json(tmp_path/"sky_comparison.json", dict(foreground_identity=identity))
    manifest["selection"]["comparison_sha256"] = sha256(tmp_path/"sky_comparison.json")
    write_json(tmp_path/"manifest.json", manifest)
    with pytest.raises(ValueError, match="prefix payload"): inspect_scene(ply, dataset, components=tmp_path/"manifest.json")


def test_nonfinite_payload_reported_without_claiming_valid_geometry(tmp_path):
    dataset, ply = fixture(tmp_path)
    count, fields, offset, _ = _header(ply)
    raw = np.memmap(ply, mode="r+", dtype="<f4", offset=offset, shape=(count, len(fields)))
    raw[1, fields.index("f_dc_0")] = np.nan
    raw[2, [fields.index(f"rot_{i}") for i in range(4)]] = 0
    raw[3, fields.index("scale_0")] = 1000.  # finite payload, invalid activated covariance
    raw.flush(); raw._mmap.close()
    report = inspect_scene(ply, dataset)
    assert report["all_rows"]["nonfinite_rows"] == 1
    assert report["foreground"]["counts"]["zero_or_invalid_quaternion_rows"] == 1
    assert report["foreground"]["counts"]["invalid_activated_scale_rows"] == 1
    assert report["status"] == "inspected" and not report["quality_accepted"]


def test_ground_actual_tracks_only_group_faces_and_exclude_heldout(tmp_path):
    dataset, ply = fixture(tmp_path, ground=True)
    report = inspect_scene(ply, dataset, config=InspectionConfig(ground=True))
    ground = report["ground_support"]
    assert ground["input_track_rows"] == 6 and ground["unique_train_observations"] == 4
    assert ground["train_ground_observations"] == 4 and ground["train_ground_unique_points"] == 2
    assert ground["train_ground_points_two_physical_stations"] == 1 and ground["strong_ground_points"] == 1
    assert ground["unique_train_down_observations"] == 1
    assert all(row.get("station_id") != "d" for row in ground["frames"])


def test_insufficient_camera_and_sparse_support_stays_unassessed():
    means, logs, quats = np.zeros((2, 3)), np.zeros((2, 3)), np.tile([1, 0, 0, 0], (2, 1))
    points = seeds()
    points["support_station_count"][:] = 1
    report = inspect_arrays(gaussian_chunks(means, logs, quats), points, None)
    assert report["sparse_support"]["supported_points"] == 0
    assert report["distributions"]["nearest_support_spacing_ratio"]["finite_count"] == 0
    assert report["distributions"]["extent_baseline_ratio"]["finite_count"] == 0
    assert camera_baseline([frame("a", 0), frame("b", 0)])["metres"] is None


def test_chunked_million_row_array_matches_single_chunk():
    n = 1000003
    means = np.column_stack([np.linspace(-3, 3, n), np.zeros(n), np.zeros(n)])
    logs = np.full((n, 3), -3.)
    quats = np.tile([1., 0, 0, 0], (n, 1))
    small = inspect_arrays(gaussian_chunks(means, logs, quats, 4096), seeds(), 4.)
    large = inspect_arrays(gaussian_chunks(means, logs, quats, n), seeds(), 4.)
    assert small == large
