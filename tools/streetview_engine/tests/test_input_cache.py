"""Real collection/preprocess cache paths; synthetic provider/model only."""
import copy
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.streetview_engine import collection, preprocess, input_cache
from tools.streetview_engine.imaging import sha256, write_json
from tools.streetview_engine.tests.test_collection import selection, source_response
from tools.streetview_engine.tests.test_preprocess import FakeSegmenter


@pytest.fixture(scope="module")
def completed_source(tmp_path_factory):
    root = tmp_path_factory.mktemp("complete_input_cache")
    config = selection()
    config["capture_policy"] = dict(mode="same_day", value="2024-03-02", time_start=None, time_end=None)
    model = root/"operator_model"
    model.mkdir()
    (model/"config.json").write_text("{}")
    (model/"model.safetensors").write_bytes(b"synthetic test fixture, not a neural model")
    settings = dict(collection=dict(max_workers=1, timeout_s=25), preprocess=dict(segmentation=dict(model_path=str(model), dynamic_dilation_px=0, core_erosion_px=0)))
    with patch.object(collection, "_request", side_effect=source_response):
        collected = collection.run(config, root, settings)
    # Emulate the old serial collector manifest, whose transport receipt was
    # absent. Its established input fingerprint excludes transport settings.
    collected.pop("transport", None)
    write_json(root/"collection/manifest.json", collected, immutable=False)
    with patch.object(preprocess, "HFSemanticSegmenter", FakeSegmenter):
        preprocess.run(config, root, settings)
    write_json(root/"config.json", config)
    write_json(root/"sfm/failure.json", dict(status="failed", reason="synthetic downstream failure"))
    (root/"owner.json").write_text('{"synthetic_private_receipt":"must_not_copy"}')
    (root/"logs").mkdir()
    (root/"logs/log.txt").write_text("not an input")
    (root/"model.ply").write_text("not an input")
    return root, config, settings


@pytest.fixture
def case(tmp_path, completed_source):
    original, config, settings = completed_source
    source = tmp_path/"source"
    shutil.copytree(original, source)
    destination = tmp_path/"new_job"
    options = copy.deepcopy(settings)
    options["collection"] = dict(max_workers=4, timeout_s=35)
    options["input_cache"] = dict(source_job_dir=str(source))
    return source, destination, copy.deepcopy(config), options


def file_hashes(root):
    return {path.relative_to(root).as_posix(): sha256(path) for path in root.rglob("*") if path.is_file()}


def test_verified_clone_uses_normal_cache_validation_and_preserves_source(case):
    source, destination, config, settings = case
    before = file_hashes(source)
    receipt = input_cache.reuse_inputs(config, destination, settings)
    assert receipt["status"] == "reused" and receipt["files_count"] > 70
    assert receipt["hardlinked_files"] + receipt["copied_files"] == receipt["files_count"]
    assert {path.name for path in destination.iterdir()} == {"collection", "prepared", "input_cache.json"}
    assert file_hashes(source) == before
    with patch.object(collection, "_request", side_effect=AssertionError("Cache must not download")), patch.object(preprocess, "HFSemanticSegmenter", side_effect=AssertionError("Cache must not infer")):
        assert collection.run(config, destination, settings) == json.loads((source/"collection/manifest.json").read_text())
        assert preprocess.run(config, destination, settings) == json.loads((source/"prepared/manifest.json").read_text())
    assert input_cache.reuse_inputs(config, destination, settings) == receipt
    assert file_hashes(source) == before


def test_copy_fallback_revalidates_every_artifact(case, monkeypatch):
    source, destination, config, settings = case
    monkeypatch.setattr(input_cache.os, "link", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("cross-device")))
    receipt = input_cache.reuse_inputs(config, destination, settings)
    assert receipt["hardlinked_files"] == 0 and receipt["copied_files"] == receipt["files_count"]
    for name, digest in receipt["files_sha256"].items(): assert sha256(destination/name) == sha256(source/name) == digest


@pytest.mark.parametrize("mutation", ["radius", "date", "ids", "model_options", "incomplete", "hash", "missing", "unrelated", "escaped_reference", "source_receipt"])
def test_invalid_cache_rejected_before_any_partial_artifact_write(case, mutation):
    source, destination, config, settings = case
    if mutation == "radius": config["radius_m"] += 1
    elif mutation == "date": config["capture_policy"]["value"] = "2024-03-03"
    elif mutation == "ids": config["panorama_ids"] = ["different_capture"]
    elif mutation == "model_options": settings["preprocess"]["segmentation"]["core_erosion_px"] = 2
    elif mutation == "incomplete":
        path = source/"prepared/manifest.json"
        value = json.loads(path.read_text()); value["status"] = "failed"
        write_json(path, value, immutable=False)
    elif mutation in ("hash", "missing"):
        frame = json.loads((source/"prepared/manifest.json").read_text())["frames"][0]
        if mutation == "hash": (source/frame["mask_path"]).write_bytes(b"changed")
        else: (source/frame["mask_path"]).unlink()
    elif mutation == "unrelated": (source/"collection/owner.json").write_text("unreferenced secret")
    elif mutation == "escaped_reference":
        path = source/"prepared/manifest.json"
        value = json.loads(path.read_text()); value["frames"][0]["mask_path"] = "../owner.json"
        write_json(path, value, immutable=False)
    elif mutation == "source_receipt":
        receipt = next((source/"collection/sources").rglob("*.source.json"))
        value = json.loads(receipt.read_text()); value["url"] = "https://invalid.test/other"
        write_json(receipt, value, immutable=False)
    with pytest.raises((ValueError, FileNotFoundError)):
        input_cache.reuse_inputs(config, destination, settings)
    assert not destination.exists() or list(destination.iterdir()) == []


def test_symlink_source_is_rejected_without_following(case, tmp_path):
    source, destination, config, settings = case
    link = source/"collection/external"
    try:
        link.symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        pytest.skip("Host does not grant symlink creation; production check also rejects reparse points")
    with pytest.raises(ValueError, match="symlink|junction"):
        input_cache.reuse_inputs(config, destination, settings)
    assert not destination.exists() or list(destination.iterdir()) == []


def test_failure_during_staging_rolls_back_only_owned_new_paths(case, monkeypatch):
    source, destination, config, settings = case
    before = file_hashes(source)
    original = input_cache.os.link
    calls = 0
    def failing_link(left, right, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3: raise KeyboardInterrupt("synthetic interruption")
        return original(left, right, **kwargs)
    monkeypatch.setattr(input_cache.os, "link", failing_link)
    with pytest.raises(KeyboardInterrupt): input_cache.reuse_inputs(config, destination, settings)
    assert list(destination.iterdir()) == [] and file_hashes(source) == before


def test_failure_after_first_directory_promotion_rolls_back_both(case, monkeypatch):
    source, destination, config, settings = case
    before = file_hashes(source)
    original = Path.rename
    def failing_rename(path, target):
        if path.name == "prepared" and path.parent.name.startswith(".input_cache_"):
            raise OSError("synthetic second promotion failure")
        return original(path, target)
    monkeypatch.setattr(Path, "rename", failing_rename)
    with pytest.raises(OSError, match="second promotion"):
        input_cache.reuse_inputs(config, destination, settings)
    assert list(destination.iterdir()) == [] and file_hashes(source) == before
    with pytest.raises(RuntimeError, match="outside"):
        input_cache._remove_owned(source, destination)


def test_no_source_or_existing_destination_overwrite(case):
    source, destination, config, settings = case
    assert input_cache.reuse_inputs(config, destination, {}) == {"status": "disabled"}
    with pytest.raises(ValueError, match="separate"):
        input_cache.reuse_inputs(config, source, settings)
    destination.mkdir()
    (destination/"collection").mkdir()
    (destination/"collection/keep.txt").write_text("existing")
    with pytest.raises(ValueError, match="overwrite"):
        input_cache.reuse_inputs(config, destination, settings)
    assert (destination/"collection/keep.txt").read_text() == "existing"


def test_input_cache_settings_reject_implicit_paths_and_unrecognized_options(case):
    source, destination, config, settings = case
    for invalid in ({"source_job_dir": "relative"}, {"source_job_dir": str(source), "copy_logs": True}, True):
        with pytest.raises(ValueError, match="absolute source_job_dir"):
            input_cache.reuse_inputs(config, destination, dict(settings, input_cache=invalid))


@pytest.mark.parametrize("stage", ["collect", "preprocess", "sfm"])
def test_dispatcher_only_calls_cache_before_collect(tmp_path, monkeypatch, stage):
    from tools.streetview_engine import __main__ as cli
    from tools.streetview_app import jobs
    write_json(tmp_path/"config.json", dict(frozen="fixture"))
    write_json(tmp_path/"settings.json", dict(input_cache=dict(source_job_dir="operator-path")))
    events = []
    monkeypatch.setattr(jobs, "validate_config", lambda value: value)
    monkeypatch.setattr(input_cache, "reuse_inputs", lambda *args: events.append("cache") or dict(status="test-only"))
    monkeypatch.setattr(cli.importlib, "import_module", lambda name: SimpleNamespace(run=lambda *args: events.append("stage") or {}))
    monkeypatch.setattr(sys, "argv", ["engine", stage, "--job-config", str(tmp_path/"config.json"), "--job-dir", str(tmp_path/"job"), "--settings", str(tmp_path/"settings.json")])
    cli.main()
    assert events == (["cache", "stage"] if stage == "collect" else ["stage"])
