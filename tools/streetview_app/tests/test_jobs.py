"""Actual tiny subprocess workflows; no provider requests, GPU or scene jobs."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import pytest

from tools.streetview_app.jobs import Backend, JobManager, STAGES, Stage, validate_config, validate_gaussian_ply


def selection(count=2):
    ids = [f"opaque+/{i}=" for i in range(count)]
    return dict(schema_version=1, provider="naver", center=dict(lat=10., lng=20.), radius_m=100.,
                capture_policy=dict(mode="same_day", value="2024-06-12", time_start="09:00", time_end="10:00"),
                panorama_ids=ids, panoramas=[dict(id=pid, lat=10., lng=20., captured_at="2024-06-12T09:30:15",
                    capture_date="2024-06-12", capture_precision="second", heading=0, links=[], title="fixture") for pid in ids],
                timestamp_policy="Provider text is retained without inventing a timezone.", selection_verified=True)


SCRIPT = '''import json, pathlib, struct, sys, time, subprocess
stage, config_path, job_root, mode = sys.argv[1:]
root = pathlib.Path(job_root)
config = json.loads(pathlib.Path(config_path).read_text(encoding="utf8"))
assert config["selection_verified"] and config["panorama_ids"][0] == "opaque+/0="
assert sys.argv[2] == str(root / "config.json")
stages = ["collect", "preprocess", "sfm", "train", "export"]
if stages.index(stage):
    assert (root / (stages[stages.index(stage)-1] + ".json")).is_file()
print("actual fixture stage:", stage, flush=True)
if mode == "exit":
    sys.exit(17)
if mode == "sleep":
    child_code = "import pathlib,time; time.sleep(1.5); pathlib.Path('child_survived').write_text('unexpected')"
    child = subprocess.Popen([sys.executable, "-c", child_code])
    (root / "ready").write_text(str(child.pid))
    time.sleep(30)
if mode == "missing":
    sys.exit(0)
(root / (stage + ".json")).write_text(json.dumps({"stage": stage, "panorama_ids": config["panorama_ids"]}))
if stage == "export":
    out = root / "export" / "scene.ply"
    out.parent.mkdir()
    if mode == "invalid":
        out.write_bytes(b"not a Gaussian PLY")
    else:
        props = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
        header = "ply\\nformat binary_little_endian 1.0\\nelement vertex 2\\n" + "".join("property float " + p + "\\n" for p in props) + "end_header\\n"
        vertices = [[0,0,2,0,0,0,0,-2,-2,-2,1,0,0,0], [1,0,3,0,0,0,0,-2,-2,-2,1,0,0,0]]
        out.write_bytes(header.encode("ascii") + b"".join(struct.pack("<14f", *v) for v in vertices))
'''


def backend(tmp_path, *, modes=None):
    program = tmp_path / "actual fixture program.py"
    program.write_text(SCRIPT, encoding="utf8")
    modes = modes or {}
    return Backend(name="synthetic-fixture-only", stages=tuple(
        Stage(name, (sys.executable, str(program), name, "{config}", "{job_dir}", modes.get(name, "ok")),
              (name + ".json",) + (("export/scene.ply",) if name == "export" else ()),
              required_paths=(str(program),)) for name in STAGES))


def test_unconfigured_is_disabled_and_no_process_starts(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("read-only operations must not launch processes")
    monkeypatch.setattr("tools.streetview_app.jobs.subprocess.Popen", forbidden)
    manager = JobManager(tmp_path)
    cap = manager.capabilities()
    assert not cap["generation_available"] and not cap["can_start"]
    assert all(not stage["ready"] for stage in cap["stages"])
    assert manager.list() == []
    with pytest.raises(RuntimeError, match="incomplete"):
        manager.start(selection())
    assert not list(tmp_path.glob("*/config.json"))


def test_real_five_stage_workflow_and_persisted_ply(tmp_path):
    manager = JobManager(tmp_path / "jobs with spaces", backend(tmp_path))
    assert manager.capabilities()["can_start"]
    config = selection()
    job = manager.start(config)
    config["panoramas"][0]["title"] = "caller changed this"
    done = manager.wait(job["id"], 15)
    assert done["status"] == "completed", done
    assert [stage["name"] for stage in done["stages"]] == list(STAGES)
    assert all(stage["status"] == "completed" and stage["exit_code"] == 0 for stage in done["stages"])
    artifact = done["artifact"]
    ply = manager.root / job["id"] / artifact["path"]
    assert artifact["vertex_count"] == 2 and artifact["sha256"] == hashlib.sha256(ply.read_bytes()).hexdigest()
    assert artifact["validation"] == "standard_gaussian_serialization_only"
    saved = json.loads((manager.root / job["id"] / "config.json").read_text(encoding="utf8"))
    assert saved["panoramas"][0]["title"] == "fixture"
    assert saved["panorama_ids"] == selection()["panorama_ids"]
    assert saved["capture_policy"] == selection()["capture_policy"]
    assert all((manager.root / job["id"] / stage["log"]).stat().st_size for stage in done["stages"])
    manager.close()
    loaded = JobManager(manager.root)
    assert loaded.get(job["id"]) == done
    assert not loaded.capabilities()["generation_available"]  # saved jobs do not install a backend


@pytest.mark.parametrize("mode, expected", [("exit", "exit code 17"), ("missing", "declared artifact")])
def test_stage_failure_does_not_continue_or_claim_success(tmp_path, mode, expected):
    manager = JobManager(tmp_path / "jobs", backend(tmp_path, modes={"sfm": mode}))
    job = manager.start(selection())
    done = manager.wait(job["id"], 15)
    assert done["status"] == "failed" and expected in done["error"]
    assert done["artifact"] is None
    assert [stage["name"] for stage in done["stages"]] == ["collect", "preprocess", "sfm"]
    assert done["stages"][-1]["status"] == "failed"
    assert not (manager.root / job["id"] / "train.json").exists()


def test_invalid_export_is_not_downloadable(tmp_path):
    manager = JobManager(tmp_path / "jobs", backend(tmp_path, modes={"export": "invalid"}))
    job = manager.start(selection())
    done = manager.wait(job["id"], 15)
    assert done["status"] == "failed" and done["artifact"] is None
    assert "PLY" in done["error"]


def test_cancel_owns_process_tree_and_blocks_next_stage(tmp_path):
    manager = JobManager(tmp_path / "jobs", backend(tmp_path, modes={"collect": "sleep"}))
    job = manager.start(selection())
    ready = manager.root / job["id"] / "ready"
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.exists(), manager.get(job["id"])
        assert manager.capabilities()["busy"] and not manager.capabilities()["can_start"]
        with pytest.raises(RuntimeError, match="already active"):
            manager.start(selection())
        manager.cancel(job["id"])
        done = manager.wait(job["id"], 10)
        assert done["status"] == "cancelled" and done["artifact"] is None
        assert not (ready.parent / "preprocess.json").exists()
        # The fixture child writes after 1.5 seconds if a process-tree cancel
        # misses it. This checks actual descendants, not a mocked terminate().
        time.sleep(1.7)
        assert not (ready.parent / "child_survived").exists()
    finally:
        manager.close()


@pytest.mark.parametrize("change", [
    lambda x: x.update(argv=["bad"]),
    lambda x: x.update(selection_verified=False),
    lambda x: x.update(schema_version=True),
    lambda x: x["center"].update(lat=float("nan")),
    lambda x: x["capture_policy"].update(value="2024-06-13"),
    lambda x: x["capture_policy"].update(time_start="11:00"),
    lambda x: x["panoramas"][0].update(capture_precision="day"),
    lambda x: x["panoramas"][0].update(lng=25.),
    lambda x: x["panoramas"][0].update(id="mismatch"),
    lambda x: x["panorama_ids"].append(x["panorama_ids"][0]),
    lambda x: x["panorama_ids"].__setitem__(0, "bad\x00id"),
    lambda x: x.update(max_panoramas=1),
])
def test_invalid_frozen_selection_rejected(change):
    config = selection()
    change(config)
    with pytest.raises((ValueError, TypeError)):
        validate_config(config)


def test_verified_metadata_opaque_ids_and_more_than_fifty_preserved():
    config = selection(200)
    original = copy.deepcopy(config)
    result = validate_config(config)
    assert result["max_panoramas"] == 200
    for key in original:
        assert result[key] == original[key]
    result["panoramas"][0]["links"].append("new")
    assert config == original


def test_legacy_max_panoramas_has_no_upper_bound_and_never_truncates_selection():
    config = selection(500)
    config['max_panoramas'] = 2000
    result = validate_config(config)
    assert result['max_panoramas'] == 2000
    assert result['panorama_ids'] == config['panorama_ids']
    assert result['panoramas'] == config['panoramas']
    for invalid in (499, True, '2000', 2000.5):
        config['max_panoramas'] = invalid
        with pytest.raises(ValueError, match='max_panoramas'):
            validate_config(config)


def test_restart_marks_unknown_completion_interrupted_without_starting(tmp_path, monkeypatch):
    directory = tmp_path / ("a" * 32)
    directory.mkdir()
    state = dict(id=directory.name, status="running", created_utc="2024-06-12T00:00:00Z", stages=[], artifact=None)
    (directory / "state.json").write_text(json.dumps(state), encoding="utf8")
    monkeypatch.setattr("tools.streetview_app.jobs.subprocess.Popen", lambda *a, **kw: pytest.fail("must not resume"))
    result = JobManager(tmp_path).get(directory.name)
    assert result["status"] == "interrupted" and "unknown" in result["error"]


def test_stage_arguments_and_output_confinement(tmp_path):
    with pytest.raises(ValueError, match="placeholder"):
        Stage("collect", ("{config.__class__}",), ("manifest.json",))
    with pytest.raises(ValueError, match="relative"):
        Stage("collect", (sys.executable,), ("../elsewhere",))
    with pytest.raises(ValueError, match="unique"):
        Backend(stages=(Stage("sfm", (sys.executable,), ("sfm.json",)), Stage("collect", (sys.executable,), ("collect.json",))))


def test_remote_commands_do_not_claim_integrated_cancellation(tmp_path):
    configured = backend(tmp_path)
    manager = JobManager(tmp_path / "jobs", Backend(stages=configured.stages, compute="remote_adapter"))
    assert not manager.capabilities()["generation_available"]
    assert "Remote" in manager.capabilities()["stages"][0]["reasons"][0]


def test_ply_nonfinite_and_truncated_payload_rejected(tmp_path):
    manager = JobManager(tmp_path / "jobs", backend(tmp_path))
    done = manager.wait(manager.start(selection())["id"], 15)
    original = (manager.root / done["id"] / done["artifact"]["path"]).read_bytes()
    broken = tmp_path / "broken.ply"
    broken.write_bytes(original[:-1])
    with pytest.raises(ValueError, match="payload length"):
        validate_gaussian_ply(broken)
    import struct
    broken.write_bytes(original[:-4] + struct.pack("<f", float("nan")))
    with pytest.raises(ValueError, match="nonfinite"):
        validate_gaussian_ply(broken)
