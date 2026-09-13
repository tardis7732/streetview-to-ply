"""GUI job owner for the existing SSH cloud, backed by an owned lease supervisor."""
from __future__ import annotations
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time

from .jobs import ACTIVE, JobManager, _inside, _now, _sha, _write_json, validate_gaussian_ply, generation_mode, stage_names, verify_single_export

WORKSPACE = Path(__file__).resolve().parents[2]


class CloudProfile:
    def __init__(self, value):
        self.host = value.get('host', '')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}', self.host):
            raise ValueError('Cloud host must be a configured SSH alias')
        self.launcher = value.get('launcher', '')
        self.worker = value.get('worker', '')
        self.jobs_root = value.get('jobs_root', '')
        for path in (self.launcher, self.worker, self.jobs_root):
            parsed = PurePosixPath(path)
            if not parsed.is_absolute() or '..' in parsed.parts or any(c in path for c in '\r\n\x00'):
                raise ValueError('Cloud paths must be fixed absolute Linux paths')
        self.settings_path = Path(value['settings_path']).resolve()
        self.lease_seconds = int(value.get('lease_seconds', 90))
        if not 15 <= self.lease_seconds <= 600:
            raise ValueError('Invalid cloud lease')
        self.timeout_seconds = int(value.get('timeout_seconds', 21600))
        if not 60 <= self.timeout_seconds <= 86400:
            raise ValueError('Invalid cloud timeout')


def source_bundle(config, settings, stages, profile, *, workspace=WORKSPACE, reuse=None):
    """Only relevant Python source and explicit scene inputs; no caches or keys."""
    mode = generation_mode(config)
    if mode != 'multi_view':
        raise ValueError('Only multi_view generation can be submitted to the cloud')
    if stages and tuple(stage.name for stage in stages) != stage_names(mode):
        raise ValueError('Cloud stage recipe differs from the frozen generation mode')
    files = [workspace / 'tools/__init__.py']
    for package in ('streetview_engine', 'streetview_geometry'):
        files.extend((workspace / 'tools' / package).glob('*.py'))
    for name in ('__init__.py', 'jobs.py', 'plans.py', 'provider.py', 'recipes.py', 'reuse.py'):
        files.append(workspace / 'tools/streetview_app' / name)
    source_hashes = {}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for path in sorted(set(files)):
            if not path.is_file() or path.is_symlink():
                raise ValueError('Required source is missing or a symlink: ' + str(path))
            name = 'code/' + path.relative_to(workspace).as_posix()
            source_hashes[name] = _sha(path)
            archive.add(path, arcname=name, recursive=False)
        recipe = dict(generation_mode=mode, stages=[dict(name=s.name, outputs=list(s.outputs)) for s in stages],
            lease_seconds=profile.lease_seconds, timeout_seconds=profile.timeout_seconds,
            cpu_threads=settings.get('cpu_threads', 8), code_sha256=source_hashes)
        if reuse is not None:
            recipe['reuse'] = reuse
        for name, value in (('config.json', config), ('settings.json', settings), ('recipe.json', recipe)):
            payload = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode('utf8')
            entry = tarfile.TarInfo(name); entry.size = len(payload)
            archive.addfile(entry, io.BytesIO(payload))
    return buffer.getvalue(), source_hashes


def extract_results(archive_path, destination, *, limit=4 * 1024**3):
    """Accept only ordinary result artifacts inside the immutable job folder."""
    total = 0
    seen = set()
    with tarfile.open(archive_path, 'r') as archive:
        for member in archive:
            total += member.size
            if total > limit or not member.isfile():
                raise ValueError('Invalid result archive')
            prefix = PurePosixPath(member.name).parts[0]
            if prefix not in ('collection', 'prepared', 'sfm', 'training', 'single_panorama', 'export', 'logs', 'remote_state.json', 'cache_manifest.json'):
                raise ValueError('Unexpected result file: ' + member.name)
            if '\\' in member.name or ':' in member.name:
                raise ValueError('Invalid result path')
            target = _inside(destination, member.name)
            if target in seen:
                raise ValueError('Duplicate result archive path: ' + member.name)
            seen.add(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(prefix='.receiving-', dir=target.parent)
            temporary = Path(temporary_name)
            try:
                with archive.extractfile(member) as source, os.fdopen(descriptor, 'wb') as out:
                    shutil.copyfileobj(source, out, 1024 * 1024)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)


def verify_result(directory, job_id, stages, final_ply, *, mode='multi_view'):
    """Transfer integrity and reconstruction/export identity are separate gates."""
    read = lambda name: json.loads(_inside(directory, name).read_text(encoding='utf8'))
    state = read('remote_state.json')
    if state.get('id') != job_id or state.get('status') != 'completed':
        raise ValueError('Remote result job identity or completion differs')
    records = state.get('stages', [])
    if tuple(s.name for s in stages) != stage_names(mode):
        raise ValueError('Expected cloud stage recipe differs from requested mode')
    if state.get('generation_mode', mode) != mode:
        raise ValueError('Remote result generation mode differs from request')
    if [r.get('name') for r in records] != [s.name for s in stages]:
        raise ValueError('Remote result lacks the expected complete stage roster')
    for configured, record in zip(stages, records):
        if record.get('status') != 'completed' or record.get('exit_code') != 0:
            raise ValueError('Remote stage did not complete successfully')
        outputs = record.get('outputs', [])
        if [o.get('path') for o in outputs] != list(configured.outputs):
            raise ValueError('Remote stage output roster differs')
        for output in outputs:
            path = _inside(directory, output['path'])
            if not path.is_file() or path.stat().st_size != output['bytes'] or _sha(path) != output['sha256']:
                raise ValueError('Downloaded stage output changed: ' + output['path'])
    artifact = validate_gaussian_ply(_inside(directory, final_ply))
    if mode == 'single_panorama':
        return dict(path=final_ply, **verify_single_export(directory, final_ply, artifact))
    training = read('training/manifest.json')
    exported = read('export/report.json')
    if training.get('status') != 'completed' or exported.get('status') != 'completed':
        raise ValueError('Training or export manifest is incomplete')
    from tools.streetview_engine.size_filter import read_size_filter_options, verify_filtered_export_lineage
    filter_record = exported.get('size_filter')
    config_path = directory / 'config.json'
    requested_filter = read_size_filter_options(read('config.json')) if config_path.is_file() else read_size_filter_options({})
    accepted_sha = training.get('selection', {}).get('accepted_model_sha256')
    if filter_record is None:
        if requested_filter['enabled']:
            raise ValueError('Requested size filter is absent from exported result')
        if accepted_sha != artifact['sha256']:
            raise ValueError('PLY differs from the evaluated accepted model')
    else:
        verified = verify_filtered_export_lineage(directory, final_ply, read('config.json'))
        if verified['artifact_sha256'] != artifact['sha256']:
            raise ValueError('Final PLY changed during filtered export verification')
    if exported.get('artifact', {}).get('sha256') != artifact['sha256'] or exported['artifact'].get('path') != final_ply:
        raise ValueError('PLY differs from its export manifest')
    if exported.get('selection') != training.get('selection'):
        raise ValueError('Export selection differs from the accepted training selection')
    if exported.get('training_manifest_sha256') != _sha(directory / 'training/manifest.json'):
        raise ValueError('Export is bound to another training manifest')
    return dict(path=final_ply, **artifact)


class RemoteJobManager(JobManager):
    def __init__(self, storage_root, backend):
        super().__init__(storage_root, backend)
        if self.backend.compute != 'remote_adapter':
            raise ValueError('Remote manager requires compute=remote_adapter')
        self.profile = CloudProfile(self.backend.remote)

    def capabilities(self, mode='multi_view'):
        result = super().capabilities(mode)
        profile = getattr(self, 'profile', None)
        for stage in result['stages']:
            reasons = [r for r in stage['reasons'] if r != 'Remote task cancellation and recovery adapter is not integrated']
            if profile is None:
                reasons.append('Cloud profile unavailable')
            elif not profile.settings_path.is_file():
                reasons.append('Engine settings unavailable')
            if not shutil.which('ssh') or not shutil.which('scp'):
                reasons.append('SSH/SCP clients unavailable')
            stage.update(reasons=reasons, ready=not reasons, status='configured' if not reasons else 'unavailable')
        ready = all(s['ready'] for s in result['stages'])
        result.update(generation_available=ready, can_start=ready and not result['busy'],
            remote_lifecycle='owned_process_groups_with_renewable_lease',
            validation='Adapter configured; explicit start probes cloud execution. Page load makes no SSH request.')
        return result

    def _options(self):
        return dict(creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)

    def _rpc(self, job_id, token, action, *, payload=None, timeout=30):
        worker = self.profile.worker if action == 'init' else str(PurePosixPath(self.profile.jobs_root) / job_id / 'code/tools/streetview_engine/remote_worker.py')
        command = shlex.join([self.profile.launcher, worker,
            '--root', self.profile.jobs_root, '--job-id', job_id, '--token', token, action])
        process = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', self.profile.host, command],
            input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, shell=False, **self._options())
        if process.returncode:
            # Do not include the full command line/owner token in user logs.
            error = process.stderr.decode('utf8', errors='replace')[-2000:]
            raise RuntimeError(f'Cloud {action} failed ({process.returncode}): {error}')
        return json.loads(process.stdout)

    def _collect_results(self, job_id, token, directory):
        packed = self._rpc(job_id, token, 'pack', timeout=180)
        expected = PurePosixPath(self.profile.jobs_root) / job_id / 'results.tar'
        if PurePosixPath(packed['path']) != expected or not 0 < packed['bytes'] <= 4 * 1024**3:
            raise ValueError('Cloud result location/size differs from the owned job')
        archive = directory / 'cloud_results.tar'
        process = subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
            self.profile.host + ':' + shlex.quote(str(expected)), str(archive)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600, shell=False, **self._options())
        if process.returncode:
            raise RuntimeError('Cloud result download failed: ' + process.stderr.decode('utf8', errors='replace')[-1000:])
        if archive.stat().st_size != packed['bytes'] or _sha(archive) != packed['sha256']:
            raise ValueError('Cloud result transfer hash mismatch')
        extract_results(archive, directory)
        return packed

    def _run(self, job_id):
        directory = self.root / job_id
        token = secrets.token_hex(32)
        initiated = False; started = False; terminal_remote = False
        try:
            with self._lock:
                if job_id in self._cancelled:
                    self._jobs[job_id].update(status='cancelled', finished_utc=_now()); self._save(self._jobs[job_id]); return
                config = self._jobs[job_id]['config']
            mode = generation_mode(config)
            stages = self.backend.stages_for(mode)
            selected_recipe = self._jobs[job_id].get('execution_recipe')
            settings = selected_recipe['operator_settings'] if selected_recipe else json.loads(self.profile.settings_path.read_text(encoding='utf8'))
            probe = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', self.profile.host,
                'nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv,noheader'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=25, shell=False, **self._options())
            if probe.returncode:
                raise RuntimeError('Cloud GPU availability check failed: ' + probe.stderr.decode('utf8', errors='replace')[-1000:])
            _write_json(directory / 'cloud_probe.json', dict(checked_utc=_now(), gpu=probe.stdout.decode('utf8').strip()))
            reuse = None
            reuse_path = directory / 'reuse_request.json'
            if reuse_path.is_file():
                reuse = json.loads(reuse_path.read_text(encoding='utf8'))
                owner_path = self.root / reuse['source_job_id'] / 'remote_owner.json'
                owner = json.loads(owner_path.read_text(encoding='utf8'))
                if owner['host'] != self.profile.host or owner['root'] != self.profile.jobs_root:
                    raise ValueError('Reuse source belongs to another cloud workspace')
                reuse['source_owner_token'] = owner['token']
            bundle, hashes = source_bundle(config, settings, stages, self.profile, reuse=reuse)
            _write_json(directory / 'engine_settings.json', settings)
            _write_json(directory / 'code_manifest.json', hashes)
            _write_json(directory / 'remote_owner.json', dict(token=token, host=self.profile.host, root=self.profile.jobs_root, job_id=job_id))
            self._rpc(job_id, token, 'init', payload=bundle, timeout=120); initiated = True
            if job_id in self._cancelled:
                self._rpc(job_id, token, 'cancel')
                with self._lock:
                    self._jobs[job_id].update(status='cancelled', finished_utc=_now()); self._save(self._jobs[job_id])
                return
            # An acknowledgement can be lost after the supervisor was launched.
            started = True
            self._rpc(job_id, token, 'start')
            while True:
                if job_id in self._cancelled:
                    try:
                        remote = self._rpc(job_id, token, 'cancel')
                    except Exception:
                        # Observing cancellation must not renew the lease.
                        remote = self._rpc(job_id, token, 'inspect')
                else:
                    remote = self._rpc(job_id, token, 'status')
                with self._lock:
                    state = self._jobs[job_id]
                    state.update(status='cancelling' if job_id in self._cancelled and remote['status'] in ('queued', 'running') else remote['status'],
                        stage=remote.get('stage'), stages=remote.get('stages', []), remote_host=self.profile.host,
                        log_tail=remote.get('log_tail', ''), updated_utc=_now())
                    if remote.get('error'):
                        state['error'] = remote['error']
                    if remote['status'] in ('completed', 'failed', 'cancelled'):
                        # Keep the owner alive until result/log transfer finishes.
                        state.update(status='cancelling' if job_id in self._cancelled else 'running',
                            message='클라우드 결과와 기록을 내려받아 검증하고 있습니다.')
                        if remote['status'] == 'completed':
                            state['stage'] = 'export'
                    self._save(state)
                if remote['status'] in ('completed', 'failed', 'cancelled'):
                    terminal_remote = True
                    break
                time.sleep(2)
            packed = self._collect_results(job_id, token, directory)
            artifact = None
            if remote['status'] == 'completed' and job_id not in self._cancelled:
                artifact = verify_result(directory, job_id, stages, self.backend.final_ply, mode=mode)
            with self._lock:
                state = self._jobs[job_id]
                cancelled = job_id in self._cancelled
                state.update(status='cancelled' if cancelled else remote['status'], artifact=None if cancelled else artifact,
                    finished_utc=_now(), cloud_result_sha256=packed['sha256'])
                if artifact is not None and not cancelled:
                    cache_path = directory / 'cache_manifest.json'
                    if state.get('execution_snapshot') or remote.get('cache_manifest') or cache_path.exists():
                        if not cache_path.is_file() or remote.get('cache_manifest', {}).get('sha256') != _sha(cache_path):
                            raise ValueError('Remote completed-stage cache manifest binding differs')
                        state['cache_manifest'] = dict(path='cache_manifest.json', sha256=_sha(cache_path))
                    report_path = directory / 'export/report.json'
                    report = json.loads(report_path.read_text(encoding='utf8'))
                    if mode == 'single_panorama':
                        state['export_report'] = dict(path='export/report.json', sha256=_sha(report_path))
                    else:
                        state['quality_report'] = dict(path='export/report.json', sha256=_sha(report_path))
                        state['quality_summary'] = report.get('quality', {})
                state.pop('message', None)
                self._save(state)
        except Exception as error:
            if started:
                try:
                    self._rpc(job_id, token, 'cancel', timeout=15)
                except Exception:
                    pass
            with self._lock:
                state = self._jobs[job_id]
                state.update(status='interrupted' if started and not terminal_remote else ('cancelled' if job_id in self._cancelled else 'failed'),
                    error=str(error), finished_utc=_now(), artifact=None)
                if started:
                    state['recovery_note'] = f'Cloud acknowledgement may be unavailable. The owned supervisor stops compute after {self.profile.lease_seconds}s without a renewed lease.'
                self._save(state)

    def cancel(self, job_id):
        with self._lock:
            state = self.get(job_id)
            if state['status'] not in ACTIVE:
                return state
            self._cancelled.add(job_id)
            self._jobs[job_id].update(status='cancelling')
            self._save(self._jobs[job_id])
        def request():
            owner = self.root / job_id / 'remote_owner.json'
            if owner.is_file():
                try:
                    self._rpc(job_id, json.loads(owner.read_text(encoding='utf8'))['token'], 'cancel')
                except Exception:
                    # The worker checks the local cancellation flag before
                    # start; after start its renewable lease bounds failures.
                    pass
        threading.Thread(target=request, daemon=True).start()
        return self.get(job_id)
