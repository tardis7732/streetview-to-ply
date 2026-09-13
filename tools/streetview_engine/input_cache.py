"""Verified same-host reuse of complete collection/prepared stage inputs only.

No network, inference, SfM database, owner token, logs or PLY is copied. Hardlinks
are used for immutable artifacts when supported, otherwise verified copies.
Downstream stage writers must keep their existing immutable/cache validation.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import stat
import uuid

import numpy as np

from . import collection, preprocess
from .imaging import FACES, cube_camera_to_station_cv, fingerprint, group_physical_stations, sha256, write_json
from .processing_options import read_processing_options
from .preprocess_identity import preprocess_input_fingerprint


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def _no_links(path):
    """Reject symlinks and Windows junction/reparse points without following."""
    path = Path(path).absolute()
    for current in (path, *path.parents):
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Input cache forbids symlinks/junctions: " + str(current))
    return path


def _tree_files(root):
    result = set()
    for stage in ("collection", "prepared"):
        directory = root/stage
        _no_links(directory)
        if not directory.is_dir():
            raise ValueError("Missing completed cache stage: " + stage)
        for current, directories, files in os.walk(directory, followlinks=False):
            for name in directories+files:
                path = Path(current)/name
                _no_links(path)
                if name in files:
                    if not stat.S_ISREG(path.stat().st_mode):
                        raise ValueError("Cache artifact is not a regular file")
                    result.add(path.relative_to(root).as_posix())
    return result


def validate_cache(config, source_job_dir, settings):
    """Complete read-only preflight. Returns immutable file hashes to clone."""
    processing = read_processing_options(config)
    source = _no_links(source_job_dir)
    if not source.is_dir():
        raise ValueError("Input cache source job directory does not exist")
    available = _tree_files(source)
    hashes = {}
    def artifact(name, expected=None):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() not in available:
            raise ValueError("Missing or escaped cache artifact: " + str(name))
        path = _no_links(source/relative)
        actual = hashes.setdefault(relative.as_posix(), sha256(path))
        if expected is not None and actual != expected:
            raise ValueError("Input cache artifact hash mismatch: " + str(name))
        return path
    collected = _read(artifact("collection/manifest.json"))
    prepared = _read(artifact("prepared/manifest.json"))
    collection_input = _read(artifact("collection/input.json"))
    prepared_input = _read(artifact("prepared/input.json"))
    if collected.get("status") != "complete" or collected.get("stage") != "collect" or prepared.get("status") != "complete" or prepared.get("stage") != "preprocess":
        raise ValueError("Input cache requires both complete collection and prepared stages")
    if fingerprint(collection_input.get("config")) != fingerprint(config):
        raise ValueError("Input cache frozen config differs (selection/date/radius/settings)")
    if (source/"config.json").exists():
        _no_links(source/"config.json")
        if fingerprint(_read(source/"config.json")) != fingerprint(config):
            raise ValueError("Input cache source job config differs from frozen selection")
    ids, frozen = collection._validate_selection(config)
    collection_options = dict(settings.get("collection", settings))
    tolerance = float(collection_options.get("colocation_tolerance_m", .25))
    workers, timeout = collection_options.get("max_workers", 4), float(collection_options.get("timeout_s", 25))
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 8 or not math.isfinite(timeout) or not 1 <= timeout <= 120:
        raise ValueError("Invalid collection transport settings for cached validation")
    expected_collection = fingerprint(dict(schema_version=1, config=config, native_face_size=1024, colocation_tolerance_m=tolerance))
    if collected.get("input_fingerprint") != expected_collection or collection_input.get("input_fingerprint") != expected_collection or collection_input.get("colocation_tolerance_m") != tolerance or collected.get("colocation_tolerance_m") != tolerance:
        raise ValueError("Input cache collection fingerprint differs")
    if collected.get("face_order") != list(FACES) or collected.get("native_face_size") != 1024 or collected.get("panorama_count") != len(ids):
        raise ValueError("Input cache collection count/native cube contract differs")
    stations = collected.get("stations", [])
    if [station.get("pano_id") for station in stations] != ids:
        raise ValueError("Input cache panorama roster/order differs")
    grouping = group_physical_stations(stations, tolerance)
    faces = {}
    def provider_source(record):
        path = artifact(record["file_path"], record["sha256"])
        receipt = _read(artifact(record["file_path"]+".source.json"))
        if any(receipt.get(key) != record.get(key) for key in ("file_path", "sha256", "url", "bytes", "retrieved_at_utc")) or record.get("bytes") != path.stat().st_size:
            raise ValueError("Input cache provider source receipt differs")
        return path
    for station in stations:
        key = station["pano_id"]
        if station.get("station_id") != grouping[key] or set(station.get("faces", {})) != set(FACES):
            raise ValueError("Input cache physical-station/cube grouping differs")
        raw = _read(provider_source(station["metadata_source"]))
        if raw.get("id") != key or str((raw.get("info") or {}).get("photodate")) != str(station.get("captured_at")):
            raise ValueError("Input cache original provider ID/capture timestamp differs")
        if collection.distance_m(station, frozen[key]) > .5 or collection.distance_m(dict(lat=float(raw["latitude"]), lng=float(raw["longitude"])), station) > 1e-6:
            raise ValueError("Input cache capture location differs from frozen/original metadata")
        date = frozen[key].get("captured_at") or frozen[key].get("capture_date")
        if date and not str(station.get("captured_at") or "").startswith(str(date)):
            raise ValueError("Input cache capture date differs from selection")
        for face in FACES:
            item = station["faces"][face]
            artifact(item["file_path"], item["sha256"])
            if item.get("w") != 1024 or item.get("h") != 1024 or len(item.get("tiles", [])) != 4 or {(tile.get("x"), tile.get("y")) for tile in item["tiles"]} != {(0, 0), (0, 1), (1, 0), (1, 1)}:
                raise ValueError("Input cache native tile dimensions/roster differ")
            for tile in item["tiles"]:
                if tile.get("face") != face or tile.get("url") != collection.tile_url(key, face, tile["x"], tile["y"]):
                    raise ValueError("Input cache original tile URL/face differs")
                provider_source(tile)
            faces[key, face] = (station, item)
    options = dict(settings.get("preprocess", settings))
    semantic = dict(options.get("segmentation", options))
    policy = preprocess.mask_policy(semantic)
    provenance = preprocess.model_provenance(semantic)  # hashes local weights; no inference/import of Torch
    expected_prepared = preprocess_input_fingerprint(config, hashes["collection/manifest.json"], semantic, policy, provenance)
    if prepared.get("input_fingerprint") != expected_prepared or prepared_input.get("input_fingerprint") != expected_prepared or prepared_input.get("semantic_options") != semantic or prepared_input.get("policy") != policy or prepared_input.get("model_provenance", {}).get("files_sha256") != provenance["files_sha256"]:
        raise ValueError("Input cache preprocessing fingerprint/model/settings differ")
    if read_processing_options(prepared) != processing or read_processing_options(prepared_input) != processing:
        raise ValueError("Input cache processing options differ from frozen selection")
    if prepared.get("collection_manifest_path") != "collection/manifest.json" or prepared.get("collection_sha256") != hashes["collection/manifest.json"]:
        raise ValueError("Input cache prepared-to-collection binding differs")
    frames = prepared.get("frames", [])
    if len(frames) != len(faces) or {(frame.get("pano_id"), frame.get("face")) for frame in frames} != set(faces):
        raise ValueError("Input cache prepared six-face roster differs")
    for frame in frames:
        station, item = faces[frame["pano_id"], frame["face"]]
        if frame.get("file_path") != item["file_path"] or frame.get("source_sha256") != item["sha256"] or frame.get("station_id") != station["station_id"] or any(frame.get(key) != expected for key, expected in dict(w=1024, h=1024, fl_x=512., fl_y=512., cx=512., cy=512.).items()) or not np.array_equal(frame.get("camera_to_station_cv"), cube_camera_to_station_cv(frame["face"])):
            raise ValueError("Input cache prepared photo/camera binding differs")
        for path_key, hash_key in (("file_path", "source_sha256"), ("mask_path", "mask_sha256"), ("sfm_mask_path", "sfm_mask_sha256"), ("sky_mask_path", "sky_mask_sha256"), ("ground_mask_path", "ground_mask_sha256"), ("semantic_path", "semantic_sha256")):
            artifact(frame[path_key], frame[hash_key])
    if set(hashes) != available:
        raise ValueError("Input cache trees contain unreferenced files; refusing to copy unrelated artifacts")
    return dict(source_job_dir=str(source), config_sha256=fingerprint(config), collection_fingerprint=expected_collection,
        prepared_fingerprint=expected_prepared, files_sha256=hashes, files_count=len(hashes), bytes=sum((source/name).stat().st_size for name in hashes),
        transport_policy="collection max_workers/timeout affect transport only and are absent from the established capture fingerprint",
        source_stages_only=["collection", "prepared"])


def _remove_owned(path, destination):
    path, destination = Path(path).absolute(), Path(destination).absolute()
    if path.parent != destination or path.name not in ("collection", "prepared") and not path.name.startswith(".input_cache_"):
        raise RuntimeError("Refusing cleanup outside the owned new-job cache paths")
    if path.exists():
        _no_links(path)
        shutil.rmtree(path)


def reuse_inputs(config, job_dir, settings):
    """Called before collect only. Fully validate, stage, verify, then publish."""
    cache = settings.get("input_cache")
    if cache is None:
        return dict(status="disabled")
    if settings.get('workflow') == 'panorama_brush_refine':
        from .panorama_input_cache import reuse_inputs as reuse_panorama_inputs
        return reuse_panorama_inputs(config, job_dir, settings)
    if not isinstance(cache, dict) or set(cache) != {"source_job_dir"} or not isinstance(cache["source_job_dir"], str) or not Path(cache["source_job_dir"]).is_absolute():
        raise ValueError("input_cache requires only an absolute source_job_dir")
    source = _no_links(cache["source_job_dir"])
    destination = Path(job_dir).absolute()
    destination.mkdir(parents=True, exist_ok=True)
    _no_links(destination)
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Input cache source/destination must be separate non-nested job directories")
    verified = validate_cache(config, source, settings)  # no destination artifact writes before this passes
    receipt_path = destination/"input_cache.json"
    expected_signature = fingerprint(verified)
    if receipt_path.exists():
        _no_links(receipt_path)
        receipt = _read(receipt_path)
        if receipt.get("status") != "reused" or receipt.get("signature") != expected_signature or _tree_files(destination) != set(verified["files_sha256"]):
            raise ValueError("Existing input cache receipt/roster differs")
        if any(sha256(destination/name) != digest for name, digest in verified["files_sha256"].items()):
            raise ValueError("Existing cached destination artifact changed")
        return receipt
    if any((destination/name).exists() or (destination/name).is_symlink() for name in ("collection", "prepared")):
        raise ValueError("Refusing to overwrite existing destination collection/prepared trees")
    staging = destination/(".input_cache_"+uuid.uuid4().hex)
    lock = destination/".input_cache_lock"
    created, linked, copied = [], 0, 0
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    try:
        staging.mkdir()
        for name, digest in verified["files_sha256"].items():
            target = staging/name
            target.parent.mkdir(parents=True, exist_ok=True)
            _no_links(target.parent)
            original = _no_links(source/name)
            if target.exists() or target.is_symlink():
                raise ValueError("Cache staging destination unexpectedly exists")
            try:
                os.link(original, target, follow_symlinks=False)
                linked += 1
            except OSError:
                # Exclusive creation never follows/overwrites an existing link.
                with original.open("rb") as left, target.open("xb") as right:
                    shutil.copyfileobj(left, right, 1024*1024)
                copied += 1
            _no_links(target)
            if sha256(target) != digest:
                raise ValueError("Input cache source changed while staging")
        if _tree_files(source) != set(verified["files_sha256"]) or any(sha256(source/name) != digest for name, digest in verified["files_sha256"].items()):
            raise ValueError("Input cache source changed during verified clone")
        if _tree_files(staging) != set(verified["files_sha256"]):
            raise ValueError("Cache staging tree differs before publication")
        for stage in ("collection", "prepared"):
            if (destination/stage).exists():
                raise ValueError("Destination stage appeared during cache clone")
            (staging/stage).rename(destination/stage)
            created.append(destination/stage)
        receipt = dict(status="reused", signature=expected_signature, **verified, hardlinked_files=linked, copied_files=copied,
            destination_job_dir=str(destination), geometry_or_ownership_copied=False,
            immutability="Complete cached artifacts are read-only by contract; shared hardlink bytes must never be edited in place")
        write_json(receipt_path, receipt)
        return receipt
    except BaseException:
        for path in reversed(created):
            _remove_owned(path, destination)
        raise
    finally:
        _remove_owned(staging, destination)
        lock.unlink(missing_ok=True)
