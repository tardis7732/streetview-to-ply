"""Cloud adapter state/integrity regressions without SSH, GPU or network."""
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tarfile
from types import SimpleNamespace

import pytest

from tools.streetview_app.jobs import Backend, Stage, STAGES
from tools.streetview_app.remote_jobs import CloudProfile, RemoteJobManager, extract_results


OUTPUTS = dict(collect=("collection/manifest.json",), preprocess=("prepared/manifest.json",),
               sfm=("sfm/manifest.json",), train=("training/manifest.json",),
               export=("export/scene.ply", "export/report.json"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf8")


def write_result_fixture(directory, job_id, *, wrong_model_hash=False):
    export = directory / "export"
    export.mkdir(exist_ok=True)
    fields = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    header = "ply\nformat binary_little_endian 1.0\nelement vertex 1\n" + "".join("property float " + x + "\n" for x in fields) + "end_header\n"
    ply = header.encode() + struct.pack("<14f", 0, 0, 3, 0, 0, 0, 0, -2, -2, -2, 1, 0, 0, 0)
    (export / "scene.ply").write_bytes(ply)
    digest = hashlib.sha256(ply).hexdigest()
    claimed = "0"*64 if wrong_model_hash else digest
    selection = dict(accepted_model="model.ply", accepted_model_sha256=claimed, selected="baseline")
    write_json(directory / "training/manifest.json", dict(status="completed", selection=selection))
    write_json(export / "report.json", dict(status="completed", artifact=dict(path="export/scene.ply", sha256=claimed, bytes=len(ply)),
                                           selection=selection, training_manifest_sha256=hashlib.sha256((directory / "training/manifest.json").read_bytes()).hexdigest()))
    for stage in ("collect", "preprocess", "sfm"):
        write_json(directory / OUTPUTS[stage][0], dict(status="completed"))
    stages = []
    for name in STAGES:
        outputs = []
        for path in OUTPUTS[name]:
            data = (directory / path).read_bytes()
            outputs.append(dict(path=path, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        stages.append(dict(name=name, status="completed", exit_code=0, outputs=outputs))
    state = dict(id=job_id, status="completed", stage="export", stages=stages)
    write_json(directory / "remote_state.json", state)
    return state


def manager_fixture(tmp_path, monkeypatch, *, wrong_model_hash=False, late_cancel=False, cancellation_rpc_fails=False, start_ack_lost=False):
    settings = tmp_path / "settings.json"
    write_json(settings, {})
    backend = Backend(name="test-only", stages=tuple(Stage(name, (sys.executable,), OUTPUTS[name]) for name in STAGES),
        compute="remote_adapter", remote=dict(host="synthetic-host", launcher="/test/python", worker="/test/worker.py",
            jobs_root="/test/jobs", settings_path=str(settings), lease_seconds=15))
    monkeypatch.setattr("tools.streetview_app.remote_jobs.subprocess.run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=b"synthetic GPU probe", stderr=b""))
    monkeypatch.setattr("tools.streetview_app.remote_jobs.source_bundle", lambda *a, **k: (b"synthetic code bundle", {}))
    monkeypatch.setattr("tools.streetview_app.remote_jobs.time.sleep", lambda *_: None)

    class FakeRPCManager(RemoteJobManager):
        def __init__(self):
            super().__init__(tmp_path / "jobs", backend)
            self.calls = []
            self.heartbeats_after_cancel = 0

        def _rpc(self, job_id, token, action, **kwargs):
            self.calls.append(action)
            if action == "start" and start_ack_lost:
                raise RuntimeError("Simulated lost acknowledgement after remote launch")
            if action == "start" and cancellation_rpc_fails:
                self._cancelled.add(job_id)
            if action == "cancel" and cancellation_rpc_fails:
                raise RuntimeError("Simulated cancel RPC outage")
            if action == "inspect" and cancellation_rpc_fails:
                return dict(id=job_id, status="cancelled", stage="train", stages=[])
            if action == "status":
                if job_id in self._cancelled and cancellation_rpc_fails:
                    self.heartbeats_after_cancel += 1
                    if self.heartbeats_after_cancel >= 3:
                        raise RuntimeError("Bound the buggy renewal loop for this test")
                    return dict(id=job_id, status="running", stage="train", stages=[])
                return write_result_fixture(self.root / job_id, job_id, wrong_model_hash=wrong_model_hash)
            return dict(status="prepared" if action == "init" else "queued")

        def _collect_results(self, job_id, token, directory):
            if late_cancel:
                self._cancelled.add(job_id)
            return dict(sha256="f"*64, bytes=1024)

    manager = FakeRPCManager()
    job_id = "a"*32
    (manager.root / job_id).mkdir()
    manager._jobs[job_id] = dict(id=job_id, status="queued", config={}, created_utc="2026-01-01T00:00:00Z", stages=[], artifact=None)
    manager._save(manager._jobs[job_id])
    return manager, job_id


def test_cancel_rpc_failure_does_not_keep_renewing_compute_lease(tmp_path, monkeypatch):
    manager, job_id = manager_fixture(tmp_path, monkeypatch, cancellation_rpc_fails=True)
    manager._run(job_id)
    assert manager.heartbeats_after_cancel == 0, "status RPC renews the remote lease and defeats cancellation"
    assert manager.get(job_id)["status"] in ("cancelled", "interrupted")
    assert manager.get(job_id)["artifact"] is None


def test_local_cancel_during_result_transfer_is_not_overwritten_by_completed(tmp_path, monkeypatch):
    manager, job_id = manager_fixture(tmp_path, monkeypatch, late_cancel=True)
    manager._run(job_id)
    result = manager.get(job_id)
    assert result["status"] == "cancelled"
    assert result["artifact"] is None


def test_valid_ply_must_match_training_and_export_model_binding(tmp_path, monkeypatch):
    manager, job_id = manager_fixture(tmp_path, monkeypatch, wrong_model_hash=True)
    manager._run(job_id)
    result = manager.get(job_id)
    assert result["status"] != "completed", "A transfer SHA is not accepted-model provenance"
    assert result["artifact"] is None


def test_bound_success_fixture_can_complete(tmp_path, monkeypatch):
    manager, job_id = manager_fixture(tmp_path, monkeypatch)
    manager._run(job_id)
    result = manager.get(job_id)
    assert result["status"] == "completed", result
    assert result["artifact"]["sha256"] == hashlib.sha256((manager.root / job_id / "export/scene.ply").read_bytes()).hexdigest()


def test_lost_start_acknowledgement_is_uncertain_and_requests_cancel(tmp_path, monkeypatch):
    manager, job_id = manager_fixture(tmp_path, monkeypatch, start_ack_lost=True)
    manager._run(job_id)
    result = manager.get(job_id)
    assert result["status"] == "interrupted"
    assert "cancel" in manager.calls
    assert "lease" in result.get("recovery_note", "").lower()
    assert result["artifact"] is None


def make_archive(path, members):
    with tarfile.open(path, "w") as archive:
        for name, contents, kind in members:
            item = tarfile.TarInfo(name)
            if kind == "symlink":
                item.type = tarfile.SYMTYPE
                item.linkname = "../../outside"
                archive.addfile(item)
            else:
                item.size = len(contents)
                archive.addfile(item, io.BytesIO(contents))


@pytest.mark.parametrize("name,kind", [("../outside", "file"), ("export/../../outside", "file"),
    ("export/x:stream", "file"), ("export/back\\slash", "file"), ("export/link", "symlink"), ("config.json", "file")])
def test_result_paths_and_links_cannot_escape_job(tmp_path, name, kind):
    archive, destination = tmp_path / "bad.tar", tmp_path / "job"
    destination.mkdir()
    make_archive(archive, [(name, b"bad", kind)])
    with pytest.raises(ValueError):
        extract_results(archive, destination)
    assert not (tmp_path / "outside").exists()


def test_duplicate_archive_members_cannot_replace_an_earlier_artifact(tmp_path):
    archive, destination = tmp_path / "duplicates.tar", tmp_path / "job"
    destination.mkdir()
    make_archive(archive, [("export/report.json", b"first", "file"), ("export/report.json", b"second", "file")])
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        extract_results(archive, destination)


def test_cloud_profile_rejects_shell_host_or_relative_remote_path(tmp_path):
    base = dict(host="good-host", launcher="/opt/python", worker="/opt/worker.py", jobs_root="/runs/jobs", settings_path=str(tmp_path / "settings.json"))
    for changes in (dict(host="host;command"), dict(worker="../../worker.py"), dict(jobs_root="/runs/../other")):
        with pytest.raises(ValueError):
            CloudProfile(dict(base, **changes))
