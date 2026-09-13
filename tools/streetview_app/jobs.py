"""Persistent, explicit-start orchestration of configured streetview stages.

Backend commands are administrator configuration, never HTTP request fields.
Constructing a manager or reading capabilities starts no subprocess or network
request. No dataset-specific legacy script is silently installed as a backend.
"""
from __future__ import annotations

import array
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import string
import subprocess
import sys
import threading
import uuid


STAGES = ("collect", "preprocess", "sfm", "train", "export")
SINGLE_STAGES = ("collect", "infer", "export")
GENERATION_MODES = ("multi_view", "single_panorama")
ACTIVE = {"queued", "running", "cancelling"}


def generation_mode(config):
    mode = config.get('generation_mode', 'multi_view')
    if mode not in GENERATION_MODES:
        raise ValueError('generation_mode must be multi_view or single_panorama')
    return mode


def stage_names(mode='multi_view'):
    mode = generation_mode(dict(generation_mode=mode))
    return SINGLE_STAGES if mode == 'single_panorama' else STAGES


class _WindowsJob:
    """An owned kernel process group; no system-wide taskkill enumeration."""
    def __init__(self):
        import ctypes
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                        ("flags", wintypes.DWORD), ("min_working_set", ctypes.c_size_t),
                        ("max_working_set", ctypes.c_size_t), ("active_process_limit", wintypes.DWORD),
                        ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]

        class IOCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("basic", BasicLimits), ("io", IOCounters), ("process_memory", ctypes.c_size_t),
                        ("job_memory", ctypes.c_size_t), ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]

        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        self.api.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.api.CreateJobObjectW.restype = wintypes.HANDLE
        self.api.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.api.SetInformationJobObject.restype = wintypes.BOOL
        self.api.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.api.AssignProcessToJobObject.restype = wintypes.BOOL
        self.api.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.api.TerminateJobObject.restype = wintypes.BOOL
        self.api.CloseHandle.argtypes = [wintypes.HANDLE]
        self.api.CloseHandle.restype = wintypes.BOOL
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def attach(self, process):
        import ctypes
        if not self.api.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def terminate(self):
        import ctypes
        if self.handle and not self.api.TerminateJobObject(self.handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    os.replace(temporary, path)


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _inside(root, relative):
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("Artifact paths must be relative to the job directory")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes job directory")
    return resolved


@dataclass(frozen=True)
class Stage:
    """One direct executable argv, with explicitly declared output artifacts.

    Supported placeholders: {job_dir}, {config}, {python}. Commands run with the
    job directory as cwd. A remote adapter must itself submit, wait, collect its
    output, and implement cancellation; this manager never guesses SSH paths.
    """
    name: str
    argv: tuple[str, ...]
    outputs: tuple[str, ...]
    required_paths: tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self):
        if self.name not in set(STAGES) | set(SINGLE_STAGES) or not self.argv or not self.outputs:
            raise ValueError("Each known stage requires argv and output artifacts")
        for value in self.argv:
            if not isinstance(value, str) or "\x00" in value:
                raise ValueError("argv must contain strings without NUL")
            for _, key, spec, conversion in string.Formatter().parse(value):
                if key is not None and (key not in {"job_dir", "config", "python"} or spec or conversion):
                    raise ValueError(f"Unsupported argv placeholder: {key}")
        for output in self.outputs:
            _inside(Path.cwd(), output)

    @classmethod
    def from_dict(cls, value):
        return cls(name=value["name"], argv=tuple(value["argv"]),
                   outputs=tuple(value["outputs"]), required_paths=tuple(value.get("required_paths", ())),
                   description=value.get("description", ""))


@dataclass(frozen=True)
class Backend:
    stages: tuple[Stage, ...] = ()
    name: str = "unconfigured"
    final_ply: str = "export/scene.ply"
    compute: str = "local"
    environment: dict[str, str] = field(default_factory=dict)
    remote: dict = field(default_factory=dict)
    mode_stages: dict[str, tuple[Stage, ...]] = field(default_factory=dict)

    def __post_init__(self):
        if tuple(stage.name for stage in self.stages) != tuple(s for s in STAGES if any(x.name == s for x in self.stages)):
            raise ValueError("Stages must be unique and in collect/preprocess/sfm/train/export order")
        if set(self.mode_stages) - {'single_panorama'}:
            raise ValueError('mode_stages only accepts single_panorama; legacy stages define multi_view')
        for mode, configured in self.mode_stages.items():
            if tuple(s.name for s in configured) != tuple(name for name in stage_names(mode) if any(s.name == name for s in configured)):
                raise ValueError('Mode stages must be unique and follow their declared execution order')
        _inside(Path.cwd(), self.final_ply)
        if self.compute not in {"local", "remote_adapter"}:
            raise ValueError("compute must be local or remote_adapter")
        if any(not isinstance(k, str) or not isinstance(v, str) for k, v in self.environment.items()):
            raise ValueError("Environment entries must be strings")

    @classmethod
    def from_dict(cls, value):
        return cls(stages=tuple(Stage.from_dict(x) for x in value.get("stages", ())),
                   name=value.get("name", "configured"), final_ply=value.get("final_ply", "export/scene.ply"),
                   compute=value.get("compute", "local"), environment=dict(value.get("environment", {})), remote=dict(value.get("remote", {})),
                   mode_stages={mode: tuple(Stage.from_dict(x) for x in stages) for mode, stages in value.get('mode_stages', {}).items()})

    def stages_for(self, mode='multi_view'):
        mode = generation_mode(dict(generation_mode=mode))
        return self.stages if mode == 'multi_view' else self.mode_stages.get(mode, ())

    @classmethod
    def from_file(cls, path):
        """Read only a trusted operator-owned backend configuration file."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf8")))


def validate_config(config):
    """Check the server-frozen selection; preserve its actual capture metadata.

    HTTP callers must first use plans.freeze_selection against the provider.
    This function checks structural consistency without any network requests;
    selection_verified is a server hand-off field, not client authentication.
    """
    if not isinstance(config, dict):
        raise ValueError("Job configuration must be an object")
    allowed = {"schema_version", "center", "radius_m", "capture_policy", "max_panoramas", "training_steps",
               "resolution", "max_splats", "provider", "panorama_ids", "excluded_panorama_ids", "panoramas", "timestamp_policy", "selection_verified", "processing_options", "generation_mode", "size_filter", "depth_cleanup"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError("Unknown job configuration fields: " + ", ".join(sorted(unknown)))
    # Deep-copy through strict JSON: no caller-owned mutable references, NaN,
    # Python objects, or accidental later changes to the saved selection.
    result = json.loads(json.dumps(config, ensure_ascii=False, allow_nan=False))
    mode = generation_mode(result)
    from .plans import read_selection_options
    options = read_selection_options(config)
    if "processing_options" in config or mode == 'single_panorama':
        result["processing_options"] = options
    from ..streetview_engine.size_filter import read_size_filter_options
    size_filter = read_size_filter_options(config)
    if mode == 'single_panorama' and size_filter['enabled']:
        raise ValueError('Single panorama generation cannot enable the size filter')
    if 'size_filter' in config:
        result['size_filter'] = size_filter
    from ..streetview_engine.depth_cleanup_options import read_depth_cleanup_options
    depth_cleanup = read_depth_cleanup_options(config)
    if depth_cleanup is not None:
        result['depth_cleanup'] = depth_cleanup
    if result.get("schema_version") != 1 or isinstance(result.get("schema_version"), bool) or result.get("selection_verified") is not True:
        raise ValueError("Job requires a server-verified schema_version=1 selection")
    ids = result.get("panorama_ids", [])
    if not isinstance(ids, list) or not ids or any(not isinstance(x, str) or not 1 <= len(x) <= 2048 or "\x00" in x for x in ids):
        raise ValueError("Invalid panorama_ids")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate panorama_ids")
    panoramas = result.get("panoramas")
    if not isinstance(panoramas, list) or len(panoramas) != len(ids) or any(not isinstance(x, dict) for x in panoramas):
        raise ValueError("Frozen panorama metadata must cover every selected ID")
    if [point.get("id") for point in panoramas] != ids:
        raise ValueError("Frozen panorama IDs/order differ from selection")
    for point in panoramas:
        for key, low, high in (("lat", -90, 90), ("lng", -180, 180)):
            value = point.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError("Invalid frozen panorama coordinate: " + key)
    for key, allowed_keys in (("center", {"lat", "lng"}), ("capture_policy", {"mode", "value", "time_start", "time_end"})):
        if not isinstance(result.get(key), dict) or set(result[key]) - allowed_keys:
            raise ValueError("Invalid " + key + " fields")
    from .plans import freeze_selection

    frozen_by_id = {point["id"]: point for point in panoramas}

    class FrozenProvider:
        def get_panorama(self, pano_id):
            return frozen_by_id[pano_id]

    # Reuse the same radius/date/time-precision gates as the plan API. The
    # adapter reads only supplied metadata; this never re-queries a provider.
    checked = freeze_selection(result, FrozenProvider())
    for key in ("center", "radius_m", "capture_policy", "excluded_panorama_ids"):
        result[key] = checked[key]
    # Retain the legacy field for saved configurations. Explicitly selected
    # IDs define the entire roster; the field must never truncate that roster.
    max_panoramas = result.get("max_panoramas", max(50, len(ids)))
    if isinstance(max_panoramas, bool) or not isinstance(max_panoramas, int) or max_panoramas < len(ids):
        raise ValueError("max_panoramas must be an integer at least the selected panorama count")
    result["max_panoramas"] = max_panoramas
    for key, default, lower, upper in (("training_steps", 6000, 1, 1000000),
                                        ("resolution", 768, 64, 4096), ("max_splats", 1000000, 4, 10000000)):
        value = result.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
            raise ValueError(f"{key} must be an integer between {lower} and {upper}")
        result[key] = value
    provider = result.get("provider")
    if not isinstance(provider, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", provider):
        raise ValueError("Invalid provider identifier")
    result["provider"] = provider
    if not isinstance(result.get("timestamp_policy"), str) or not result["timestamp_policy"]:
        raise ValueError("Frozen selection must preserve its timestamp policy")
    return result


def verify_single_export(directory, final_ply, artifact=None):
    """Bind a real single-capture PLY to inference/export, with no training claim."""
    directory = Path(directory)
    artifact = artifact or validate_gaussian_ply(_inside(directory, final_ply))
    manifest_path = _inside(directory, 'single_panorama/manifest.json')
    inferred = json.loads(manifest_path.read_text(encoding='utf8'))
    exported = json.loads(_inside(directory, 'export/report.json').read_text(encoding='utf8'))
    if inferred.get('status') != 'completed' or exported.get('status') != 'completed' or exported.get('generation_mode') != 'single_panorama':
        raise ValueError('Single panorama inference/export did not complete')
    if inferred.get('artifact', {}).get('sha256') != artifact['sha256']:
        raise ValueError('Single panorama PLY differs from its inference artifact')
    if exported.get('artifact', {}).get('sha256') != artifact['sha256'] or exported['artifact'].get('path') != final_ply:
        raise ValueError('Single panorama PLY differs from its export manifest')
    if exported.get('single_panorama_manifest_sha256') != _sha(manifest_path):
        raise ValueError('Single panorama export is bound to another inference manifest')
    if exported.get('training_manifest_sha256') is not None:
        raise ValueError('Single panorama export cannot claim a training stage')
    return artifact


def validate_gaussian_ply(path):
    """Verify standard scalar-float binary Gaussian PLY, finite data and SHA.

    This is serialization validation, not reconstruction-quality acceptance.
    """
    path = Path(path)
    properties = []
    count = None
    with path.open("rb") as stream:
        if stream.readline() != b"ply\n":
            raise ValueError("Output is not a PLY file")
        total = 4
        format_seen = False
        while True:
            line = stream.readline(8193)
            total += len(line)
            if not line or len(line) > 8192 or total > 65536:
                raise ValueError("Invalid PLY header")
            tokens = line.decode("ascii").strip().split()
            if not tokens:
                continue
            if tokens[0] in {"comment", "obj_info"}:
                continue
            if tokens[0] == "format":
                if tokens != ["format", "binary_little_endian", "1.0"] or format_seen:
                    raise ValueError("Expected binary_little_endian PLY 1.0")
                format_seen = True
            elif tokens[0] == "element":
                if len(tokens) != 3 or tokens[1] != "vertex" or count is not None:
                    raise ValueError("Expected only one vertex element")
                count = int(tokens[2])
                if count < 1:
                    raise ValueError("Gaussian PLY has no vertices")
            elif tokens[0] == "property":
                if count is None or len(tokens) != 3 or tokens[1] not in {"float", "float32"} or tokens[2] in properties:
                    raise ValueError("Expected unique scalar float vertex properties")
                properties.append(tokens[2])
            elif tokens == ["end_header"]:
                break
            else:
                raise ValueError("Unknown PLY header declaration")
        required = {"x", "y", "z", "opacity", *(f"scale_{i}" for i in range(3)),
                    *(f"rot_{i}" for i in range(4)), *(f"f_dc_{i}" for i in range(3))}
        if not format_seen or count is None or not required <= set(properties):
            raise ValueError("Missing standard Gaussian PLY properties")
        payload_bytes = count * len(properties) * 4
        if path.stat().st_size - stream.tell() != payload_bytes:
            raise ValueError("PLY payload length does not match declared vertices")
        while data := stream.read(4 * 262144):
            values = array.array("f")
            values.frombytes(data)
            if sys.byteorder != "little":
                values.byteswap()
            if not all(map(math.isfinite, values)):
                raise ValueError("PLY contains nonfinite Gaussian parameters")
    return dict(vertex_count=count, properties=properties, bytes=path.stat().st_size, sha256=_sha(path),
                validation="standard_gaussian_serialization_only")


class JobManager:
    """Single-server job owner; at most one active compute job per manager.

    Persisted jobs are never automatically resumed after a server restart.
    Previously active records become interrupted; recorded external processes
    are not claimed as owned or killed by a new manager instance.
    """
    def __init__(self, storage_root, backend=None):
        self.root = Path(storage_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.backend = backend if isinstance(backend, Backend) else (Backend.from_dict(backend) if backend is not None else Backend())
        self._lock = threading.RLock()
        self._jobs = {}
        self._processes = {}
        self._windows_jobs = {}
        self._threads = {}
        self._cancelled = set()
        for record in sorted(self.root.glob("*/state.json")):
            state = json.loads(record.read_text(encoding="utf8"))
            if state.get("id") != record.parent.name:
                raise ValueError(f"Job record identity mismatch: {record}")
            if state.get("status") in ACTIVE:
                state.update(status="interrupted", finished_utc=_now(),
                             error="Server restarted. Previous process completion is unknown; no automatic resume or process termination.")
                _write_json(record, state)
            self._jobs[state["id"]] = state

    def capabilities(self, mode='multi_view'):
        """Read-only availability; does not probe GPUs or initiate SSH."""
        if mode != 'multi_view':
            raise ValueError('Only multi_view generation is supported')
        stages = []
        configured = self.backend.stages_for(mode)
        for name in stage_names(mode):
            stage = next((x for x in configured if x.name == name), None)
            reasons = []
            if stage is None:
                reasons.append("No generic stage adapter configured")
            else:
                argv0 = stage.argv[0].format(job_dir=str(self.root), config=str(self.root / "config.json"), python=sys.executable)
                if not shutil.which(argv0):
                    reasons.append("Configured executable is unavailable: " + argv0)
                reasons.extend("Required path unavailable: " + path for path in stage.required_paths if not Path(path).exists())
                if self.backend.compute == "remote_adapter":
                    reasons.append("Remote task cancellation and recovery adapter is not integrated")
            stages.append(dict(name=name, ready=not reasons, status="configured" if not reasons else "unavailable",
                               reasons=reasons, description=stage.description if stage else ""))
        complete = all(stage["ready"] for stage in stages)
        with self._lock:
            busy = any(x["status"] in ACTIVE for x in self._jobs.values())
        return dict(generation_available=complete, can_start=complete and not busy, generation_mode=mode,
                    backend=self.backend.name, compute=self.backend.compute, busy=busy, stages=stages,
                    validation="Configured commands and required paths checked; no runtime GPU or provider probe performed",
                    starts_only_on_explicit_request=True)

    def _save(self, state):
        _write_json(self.root / state["id"] / "state.json", state)

    def start(self, config, *, execution_recipe=None, reuse=None):
        config = validate_config(config)
        if generation_mode(config) != 'multi_view':
            raise ValueError('Only multi_view generation is supported')
        from .recipes import backend_snapshot
        snapshot = backend_snapshot(self.backend)
        if execution_recipe is not None:
            if execution_recipe.get('execution_snapshot') != snapshot:
                raise ValueError('Recipe code or operator settings changed')
            execution_recipe = json.loads(json.dumps(execution_recipe, allow_nan=False))
        if reuse is not None:
            # Recheck just before job creation; callers cannot bypass inspection.
            from .reuse import ReuseService
            verified = ReuseService(self).prepare(reuse['source_job_id'], reuse['from_stage'])
            if verified.pop('config') != config:
                raise ValueError('Reused job requires the identical frozen configuration')
            verified.pop('execution_recipe')
            if verified != reuse:
                raise ValueError('Reuse request changed after verification')
        with self._lock:
            capability = self.capabilities(generation_mode(config))
            if not capability["generation_available"]:
                missing = ", ".join(s["name"] for s in capability["stages"] if not s["ready"])
                raise RuntimeError("Generation backend is incomplete: " + missing)
            if capability["busy"]:
                raise RuntimeError("A generation job is already active")
            job_id = uuid.uuid4().hex
            directory = self.root / job_id
            directory.mkdir()
            (directory / "logs").mkdir()
            _write_json(directory / "config.json", config)
            state = dict(id=job_id, status="queued", created_utc=_now(), config=config,
                         config_sha256=_sha(directory / "config.json"), backend=self.backend.name,
                         compute=self.backend.compute, stage=None, stages=[], artifact=None,
                         execution_snapshot=snapshot)
            if execution_recipe is not None:
                state['execution_recipe'] = execution_recipe
            if reuse is not None:
                _write_json(directory / 'reuse_request.json', reuse)
                state['reuse'] = {key: reuse[key] for key in ('source_job_id', 'from_stage', 'reused_stages')}
            self._jobs[job_id] = state
            self._save(state)
            worker = threading.Thread(target=self._run, args=(job_id,), name="streetview-" + job_id, daemon=True)
            self._threads[job_id] = worker
            worker.start()
            return self.get(job_id)

    def get(self, job_id):
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError("Unknown job")
            return json.loads(json.dumps(self._jobs[job_id]))

    def list(self):
        with self._lock:
            return [self.get(key) for key in sorted(self._jobs, key=lambda key: self._jobs[key]["created_utc"], reverse=True)]

    def _run(self, job_id):
        directory = self.root / job_id
        try:
            mode = generation_mode(self._jobs[job_id]['config'])
            reused = {}
            reuse_path = directory / 'reuse_request.json'
            if reuse_path.is_file():
                from .reuse import copy_files
                reuse = json.loads(reuse_path.read_text(encoding='utf8'))
                copy_files(self.root / reuse['source_job_id'], directory, reuse['files'])
                _write_json(directory / 'cache_manifest.json', dict(schema_version=1, stages=reuse['cache_stages']))
                reused = {row['name']: row for row in reuse['records']}
            for stage in self.backend.stages_for(mode):
                with self._lock:
                    if job_id in self._cancelled:
                        break
                    state = self._jobs[job_id]
                    if stage.name in reused:
                        row = reused[stage.name]
                        record = dict(name=stage.name, status='completed', exit_code=0, outputs=row['outputs'],
                            reused=True, source_job_id=reuse['source_job_id'], finished_utc=_now())
                        state['stages'].append(record)
                        state['cache_manifest'] = dict(path='cache_manifest.json', sha256=_sha(directory / 'cache_manifest.json'))
                        self._save(state)
                        continue
                    argv = [part.format(job_dir=str(directory), config=str(directory / "config.json"), python=sys.executable) for part in stage.argv]
                    record = dict(name=stage.name, status="running", started_utc=_now(), argv=argv,
                                  log=f"logs/{stage.name}.log", outputs=[])
                    state.update(status="running", stage=stage.name)
                    state["stages"].append(record)
                    self._save(state)
                    environment = os.environ.copy()
                    environment.update(self.backend.environment)
                    environment.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
                    options = dict(cwd=directory, env=environment, stdin=subprocess.DEVNULL, shell=False)
                    launch_argv = argv
                    if os.name == "nt":
                        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
                        # The wrapper waits on stdin before spawning any stage
                        # descendants. Attach it to our kernel Job Object first
                        # so even very fast child launches cannot escape it.
                        launch_record = directory / (stage.name + ".command.json")
                        _write_json(launch_record, dict(argv=argv))
                        launch_argv = [sys.executable, str(Path(__file__).resolve()), "--launch", str(launch_record)]
                        options["stdin"] = subprocess.PIPE
                        self._windows_jobs[job_id] = _WindowsJob()
                    else:
                        options["start_new_session"] = True
                    with (directory / record["log"]).open("wb") as log:
                        process = subprocess.Popen(launch_argv, stdout=log, stderr=subprocess.STDOUT, **options)
                    if os.name == "nt":
                        try:
                            self._windows_jobs[job_id].attach(process)
                            process.stdin.write(b"run\n")
                            process.stdin.close()
                        except Exception:
                            process.kill()
                            process.wait()
                            raise
                    self._processes[job_id] = process
                    record["pid"] = process.pid
                    self._save(state)
                exit_code = process.wait()
                with self._lock:
                    self._processes.pop(job_id, None)
                    owned_job = self._windows_jobs.pop(job_id, None)
                    if owned_job:
                        owned_job.close()
                    record.update(exit_code=exit_code, finished_utc=_now())
                    if job_id in self._cancelled:
                        record["status"] = "cancelled"
                        break
                    if exit_code != 0:
                        record["status"] = "failed"
                        raise RuntimeError(f"{stage.name} failed with exit code {exit_code}; see {record['log']}")
                    for relative in stage.outputs:
                        output = _inside(directory, relative)
                        if not output.is_file() or output.stat().st_size == 0:
                            raise RuntimeError(f"{stage.name} did not produce declared artifact: {relative}")
                        record["outputs"].append(dict(path=relative, sha256=_sha(output), bytes=output.stat().st_size))
                    record["status"] = "completed"
                    from .reuse import record_stage_cache
                    state['cache_manifest'] = record_stage_cache(directory, stage.name, stage.outputs)
                    self._save(state)
            # Full-file serialization checks may be substantial for a real
            # export. Keep status polling and cancellation responsive.
            result = None
            with self._lock:
                should_validate = job_id not in self._cancelled
            if should_validate:
                result = validate_gaussian_ply(_inside(directory, self.backend.final_ply))
                if mode == 'single_panorama':
                    result = verify_single_export(directory, self.backend.final_ply, result)
                else:
                    from tools.streetview_engine.size_filter import read_size_filter_options, verify_filtered_export_lineage
                    filter_requested = read_size_filter_options(self._jobs[job_id]['config'])['enabled']
                    report_path = directory / 'export/report.json'
                    report = json.loads(report_path.read_text(encoding='utf8')) if report_path.is_file() else {}
                    if filter_requested or report.get('size_filter') is not None:
                        verified = verify_filtered_export_lineage(directory, self.backend.final_ply, self._jobs[job_id]['config'])
                        if verified['artifact_sha256'] != result['sha256']:
                            raise ValueError('Final PLY changed during filtered export verification')
                        result['size_filter_verified'] = verified
            with self._lock:
                state = self._jobs[job_id]
                if job_id in self._cancelled:
                    state.update(status="cancelled", finished_utc=_now())
                else:
                    state.update(status="completed", finished_utc=_now(),
                                 artifact=dict(path=self.backend.final_ply, **result))
                self._save(state)
        except Exception as error:
            with self._lock:
                owned_job = self._windows_jobs.pop(job_id, None)
                if owned_job:
                    owned_job.close()
                state = self._jobs[job_id]
                state.update(status="cancelled" if job_id in self._cancelled else "failed", error=str(error), finished_utc=_now())
                if state["stages"] and (state["stages"][-1]["status"] == "running" or state["stages"][-1]["name"] == "export"):
                    state["stages"][-1].update(status=state["status"], error=str(error), finished_utc=_now())
                self._save(state)

    def cancel(self, job_id):
        """Terminate only the process tree launched and still owned by this manager."""
        with self._lock:
            state = self.get(job_id)
            if state["status"] not in ACTIVE:
                return state
            self._cancelled.add(job_id)
            self._jobs[job_id]["status"] = "cancelling"
            self._save(self._jobs[job_id])
            process = self._processes.get(job_id)
            if process is not None and process.poll() is None:
                if os.name == "nt":
                    self._windows_jobs[job_id].terminate()
                else:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            return self.get(job_id)

    def wait(self, job_id, timeout=None):
        """Test/CLI convenience; HTTP handlers should poll get() instead."""
        worker = self._threads.get(job_id)
        if worker:
            worker.join(timeout)
        return self.get(job_id)

    def close(self):
        for job in self.list():
            if job["status"] in ACTIVE:
                self.cancel(job["id"])
        for worker in tuple(self._threads.values()):
            worker.join(timeout=5)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--launch":
        raise SystemExit("This internal launcher requires --launch <owned command.json>")
    if sys.stdin.buffer.readline() != b"run\n":
        raise SystemExit("Job Object assignment was not acknowledged")
    command = json.loads(Path(sys.argv[2]).read_text(encoding="utf8"))["argv"]
    # This Windows-only wrapper and all children are already in the owning
    # Job Object. Never detach or request CREATE_BREAKAWAY_FROM_JOB.
    child = subprocess.Popen(command, stdin=subprocess.DEVNULL, shell=False,
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    raise SystemExit(child.wait())
