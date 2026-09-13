"""Owned Linux pipeline supervisor with a renewable client lease.

The supervisor adopts and tracks its Linux descendants. Losing the local server expires
its lease, so an abandoned UI cannot leave indefinite GPU work behind. Control
requests require the per-job owner token. Signals use verified pidfds, never a
stored arbitrary PID/group. This standalone file is installed into the cloud root.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import hmac
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
import tarfile
import time

STAGES = ('collect', 'preprocess', 'sfm', 'train', 'export')


def verify_recipe(job, recipe):
    """The standalone supervisor accepts only the frozen mode's exact stages."""
    config = read_json(job / 'config.json')
    mode = config.get('generation_mode', 'multi_view')
    if mode != 'multi_view' or recipe.get('generation_mode', 'multi_view') != mode:
        raise ValueError('Recipe generation mode differs from frozen configuration')
    if tuple(x.get('name') for x in recipe['stages']) != STAGES:
        raise ValueError('Exactly the ordered stages for the frozen generation mode are required')
    return mode


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding='utf8'))


def inside(root, name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
        raise ValueError('Archive path is not confined')
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError('Path escapes job directory')
    return target


def extract_archive(stream, destination, *, limit=32 * 1024 * 1024):
    total = 0
    with tarfile.open(fileobj=stream, mode='r|*') as archive:
        for member in archive:
            total += member.size
            if total > limit or not member.isfile():
                raise ValueError('Only bounded ordinary files are accepted')
            target = inside(destination, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError('Duplicate archive path')
            with archive.extractfile(member) as source, target.open('xb') as output:
                while block := source.read(1024 * 1024):
                    output.write(block)


def job_path(root, job_id):
    if not re.fullmatch(r'[0-9a-f]{32}', job_id):
        raise ValueError('Invalid job identifier')
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    return inside(root, job_id)


def authenticate(job, token):
    if not re.fullmatch(r'[0-9a-f]{64}', token):
        raise ValueError('Invalid owner token')
    owner = read_json(job / 'owner.json')
    if not hmac.compare_digest(owner['token_sha256'], hashlib.sha256(token.encode()).hexdigest()):
        raise PermissionError('Job owner token mismatch')


def verify_code(job, recipe):
    """Verify the complete transferred source roster before executing it."""
    expected = recipe.get('code_sha256')
    if not isinstance(expected, dict) or not expected:
        raise ValueError('Transferred source requires a nonempty code hash manifest')
    actual = {path.relative_to(job).as_posix() for path in (job / 'code').rglob('*') if path.is_file()}
    if set(expected) != actual:
        raise ValueError('Transferred code roster differs from its hash manifest')
    for name, digest in expected.items():
        path = inside(job, name)
        if PurePosixPath(name).parts[0] != 'code' or path.suffix != '.py' or path.is_symlink() or not re.fullmatch(r'[0-9a-f]{64}', str(digest)):
            raise ValueError('Invalid transferred source manifest entry')
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                raise ValueError('Transferred source hash mismatch: ' + name)


def init_job(job, token, source):
    if not re.fullmatch(r'[0-9a-f]{64}', token):
        raise ValueError('Invalid owner token')
    job.mkdir(exist_ok=False)
    extract_archive(source, job)
    recipe = read_json(job / 'recipe.json')
    mode = verify_recipe(job, recipe)
    if not 15 <= recipe.get('lease_seconds', 90) <= 600:
        raise ValueError('Invalid lease duration')
    for stage in recipe['stages']:
        for path in stage['outputs']:
            inside(job, path)
    for path in ('config.json', 'settings.json', 'code/tools/streetview_engine/__main__.py', 'code/tools/streetview_engine/remote_worker.py'):
        if not inside(job, path).is_file():
            raise ValueError('Required job input missing: ' + path)
    verify_code(job, recipe)
    write_json(job / 'owner.json', dict(token_sha256=hashlib.sha256(token.encode()).hexdigest(), created_utc=now()))
    (job / 'lease').touch()
    write_json(job / 'remote_state.json', dict(id=job.name, status='prepared', generation_mode=mode, stage=None, stages=[], created_utc=now()))
    return read_json(job / 'remote_state.json')


def start_job(job, token):
    authenticate(job, token)
    recipe = read_json(job / 'recipe.json')
    verify_recipe(job, recipe)
    verify_code(job, recipe)
    with (job / 'started.lock').open('x') as lock:
        lock.write(now())
    (job / 'lease').touch()
    state = read_json(job / 'remote_state.json')
    state.update(status='queued'); write_json(job / 'remote_state.json', state)
    (job / 'logs').mkdir(exist_ok=True)
    worker = job / 'code/tools/streetview_engine/remote_worker.py'
    with (job / 'logs/supervisor.log').open('ab') as log:
        process = subprocess.Popen([sys.executable, str(worker), '--root', str(job.parent), '--job-id', job.name, 'supervise'],
            cwd=job / 'code', stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return dict(status='queued', supervisor_pid=process.pid)


class OwnedDescendants:
    """A dedicated Linux supervisor owns only its descendants and adoptees.

    PR_SET_CHILD_SUBREAPER keeps double-forked/setsid children in our ancestry
    when their leader exits. We discover ancestry from /proc, open a pidfd,
    recheck the process start tick, then signal that descriptor. A recycled PID
    can never receive a signal intended for an older opened process instance.
    This must only be instantiated by the dedicated single-job supervisor.
    """
    def __init__(self):
        if sys.platform != 'linux' or not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
            raise RuntimeError('Owned cloud cleanup requires Linux pidfd support')
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
            raise OSError(ctypes.get_errno(), 'Cannot establish child-subreaper ownership')
        self.pid = os.getpid()

    @staticmethod
    def _record(pid):
        try:
            raw = Path(f'/proc/{pid}/stat').read_text()
            fields = raw[raw.rfind(')') + 2:].split()
            return dict(pid=pid, ppid=int(fields[1]), state=fields[0], start=int(fields[19]))
        except (OSError, ValueError, IndexError):
            return None

    def snapshot(self):
        records = {}
        for path in Path('/proc').iterdir():
            if path.name.isdigit():
                item = self._record(int(path.name))
                if item is not None:
                    records[item['pid']] = item
        parents = {self.pid}; owned = {}
        while True:
            added = {pid for pid, row in records.items() if row['ppid'] in parents and pid not in parents}
            if not added:
                return owned
            owned.update({pid: records[pid] for pid in added})
            parents.update(added)

    def send(self, record, signum):
        descriptor = None
        try:
            descriptor = os.pidfd_open(record['pid'])
            current = self._record(record['pid'])
            if current is None or current['start'] != record['start']:
                return
            # Recheck current ancestry after opening the stable process handle.
            owned = self.snapshot().get(record['pid'])
            if owned is None or owned['start'] != record['start']:
                return
            signal.pidfd_send_signal(descriptor, signum)
        except ProcessLookupError:
            pass
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def reap_adopted(self, process):
        leader = process.pid if process is not None else None
        for pid, row in self.snapshot().items():
            if pid != leader and row['ppid'] == self.pid and row['state'] == 'Z':
                try:
                    os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    pass

    def terminate(self, process=None, *, grace_seconds=5):
        """TERM, then KILL every still-owned process, including late forks.

        The leader is reaped by its Popen instance to preserve its exit code.
        Never return a terminal job state while a live owned descendant remains.
        An uninterruptible kernel task can delay this; ownership/compute flock
        are retained until it actually exits, rather than declaring it stopped.
        """
        deadline = time.monotonic() + grace_seconds
        terminated = set()
        empty_rounds = 0
        while True:
            if process is not None:
                process.poll()
            self.reap_adopted(process)
            active = [row for row in self.snapshot().values() if row['state'] not in ('Z', 'X')]
            if not active:
                if process is not None:
                    process.wait()
                self.reap_adopted(None)
                # A fork/leader-exit can straddle one /proc directory snapshot.
                # Recheck after adoption settles before releasing ownership.
                empty_rounds += 1
                if empty_rounds >= 2:
                    return
                time.sleep(.05)
                continue
            empty_rounds = 0
            killing = time.monotonic() >= deadline
            for row in active:
                identity = (row['pid'], row['start'])
                if killing or identity not in terminated:
                    self.send(row, signal.SIGKILL if killing else signal.SIGTERM)
                    terminated.add(identity)
            time.sleep(.05)


def stop_child(process, owned):
    owned.terminate(process)


def prepare_reuse(job, recipe):
    """Copy a same-owner completed sibling only after every cached byte verifies."""
    request = recipe.get('reuse')
    if request is None:
        return {}
    from tools.streetview_app.reuse import copy_files
    source = job_path(job.parent, request['source_job_id'])
    if source == job:
        raise ValueError('Reuse source and destination must differ')
    authenticate(source, request['source_owner_token'])
    original = read_json(source / 'remote_state.json')
    if original.get('status') != 'completed':
        raise ValueError('Cloud reuse source did not complete')
    if read_json(source / 'config.json') != read_json(job / 'config.json') or read_json(source / 'settings.json') != read_json(job / 'settings.json'):
        raise ValueError('Cloud cache configuration/operator settings differ')
    original_recipe = read_json(source / 'recipe.json')
    if original_recipe['code_sha256'] != recipe['code_sha256']:
        raise ValueError('Cloud cache source code versions differ')
    verify_code(source, original_recipe)
    cache_path = source / 'cache_manifest.json'
    with cache_path.open('rb') as stream:
        actual_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual_hash != request['cache_manifest_sha256'] or original.get('cache_manifest', {}).get('sha256') != actual_hash:
        raise ValueError('Cloud cache manifest binding differs')
    names = [stage['name'] for stage in recipe['stages']]
    if request['from_stage'] not in names[1:] or request['reused_stages'] != names[:names.index(request['from_stage'])]:
        raise ValueError('Cloud reuse is not an ordered completed prefix')
    cache = read_json(cache_path)
    expected_files = {}
    for name in request['reused_stages']:
        if cache['stages'][name] != request['cache_stages'][name]:
            raise ValueError('Cloud cached stage inventory differs')
        expected_files.update(cache['stages'][name])
    if expected_files != request['files']:
        raise ValueError('Cloud copied file roster differs')
    copy_files(source, job, expected_files)
    write_json(job / 'cache_manifest.json', dict(schema_version=1, stages=request['cache_stages']))
    records = {record['name']: record for record in original['stages'] if record['name'] in request['reused_stages']}
    if set(records) != set(request['reused_stages']) or any(record.get('status') != 'completed' for record in records.values()):
        raise ValueError('Cloud reuse source stage records are incomplete')
    return records


def supervise(job):
    if sys.platform != 'linux':
        raise RuntimeError('Cloud supervisor requires Linux descendant ownership')
    recipe = read_json(job / 'recipe.json')
    state = read_json(job / 'remote_state.json')
    import fcntl
    compute_lock = (job.parent / 'compute.lock').open('a+')
    try:
        fcntl.flock(compute_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        state.update(status='failed', error='Another owned application job is still active on this cloud root', finished_utc=now())
        write_json(job / 'remote_state.json', state)
        compute_lock.close()
        return
    lease_seconds = recipe.get('lease_seconds', 90)
    started = time.monotonic(); process = None; owned = None
    cancelled = False
    interrupted = False
    def interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True
    previous_handlers = {kind: signal.signal(kind, interrupt) for kind in (signal.SIGTERM, signal.SIGINT)}
    try:
        verify_recipe(job, recipe)
        verify_code(job, recipe)
        reused = prepare_reuse(job, recipe)
        owned = OwnedDescendants()
        for stage in recipe['stages']:
            if interrupted or (job / 'cancel').exists():
                cancelled = True; break
            name = stage['name']
            if name in reused:
                from tools.streetview_app.reuse import record_stage_cache
                state['stages'].append(dict(name=name, status='completed', exit_code=0,
                    reused=True, source_job_id=recipe['reuse']['source_job_id'],
                    outputs=reused[name]['outputs'], finished_utc=now()))
                state['cache_manifest'] = record_stage_cache(job, name, stage['outputs'])
                write_json(job / 'remote_state.json', state)
                continue
            record = dict(name=name, status='running', started_utc=now(), log=f'logs/{name}.log', outputs=[])
            state['stages'].append(record); state.update(status='running', stage=name)
            write_json(job / 'remote_state.json', state)
            argv = [sys.executable, '-B', '-m', 'tools.streetview_engine', name,
                    '--job-config', str(job / 'config.json'), '--job-dir', str(job), '--settings', str(job / 'settings.json')]
            environment = os.environ.copy()
            environment['PYTHONPATH'] = str(job / 'code') + os.pathsep + environment.get('PYTHONPATH', '')
            environment.update(PYTHONUNBUFFERED='1', PYTHONUTF8='1', OMP_NUM_THREADS=str(recipe.get('cpu_threads', 8)))
            with (job / record['log']).open('ab') as log:
                process = subprocess.Popen(argv, cwd=job / 'code', env=environment, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, shell=False, start_new_session=True)
                while process.poll() is None:
                    if interrupted or (job / 'cancel').exists():
                        cancelled = True; stop_child(process, owned); break
                    if time.time() - (job / 'lease').stat().st_mtime > lease_seconds:
                        stop_child(process, owned); raise RuntimeError('Owner lease expired; cloud compute stopped')
                    if time.monotonic() - started > recipe.get('timeout_seconds', 21600):
                        stop_child(process, owned); raise RuntimeError('Configured job time limit exceeded')
                    time.sleep(.5)
                record['exit_code'] = process.wait()
            # A successfully exited CLI may still leave detached descendants.
            # Cleanup is mandatory before the next stage or terminal state.
            stop_child(process, owned)
            process = None
            if cancelled:
                record.update(status='cancelled', finished_utc=now()); break
            if record['exit_code'] != 0:
                raise RuntimeError(f'{name} exited with code {record["exit_code"]}; see {record["log"]}')
            for relative in stage['outputs']:
                path = inside(job, relative)
                if not path.is_file() or not path.stat().st_size:
                    raise RuntimeError(f'{name} did not produce {relative}')
                with path.open('rb') as stream:
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                record['outputs'].append(dict(path=relative, bytes=path.stat().st_size, sha256=digest))
            record.update(status='completed', finished_utc=now())
            from tools.streetview_app.reuse import record_stage_cache
            state['cache_manifest'] = record_stage_cache(job, name, stage['outputs'])
            write_json(job / 'remote_state.json', state)
        state.update(status='cancelled' if cancelled else 'completed', finished_utc=now())
    except Exception as error:
        if owned is not None:
            stop_child(process, owned)
        state.update(status='failed', error=str(error), finished_utc=now())
        if state['stages'] and state['stages'][-1]['status'] == 'running':
            state['stages'][-1].update(status='failed', error=str(error), finished_utc=now())
    finally:
        if owned is not None:
            owned.terminate(process)
        write_json(job / 'remote_state.json', state)
        compute_lock.close()
        for kind, handler in previous_handlers.items():
            signal.signal(kind, handler)


def status(job, token, *, renew=True):
    authenticate(job, token)
    if renew:
        (job / 'lease').touch()
    state = read_json(job / 'remote_state.json')
    stage = state.get('stage')
    if stage in STAGES:
        path = job / f'logs/{stage}.log'
        if path.is_file():
            with path.open('rb') as stream:
                stream.seek(max(0, path.stat().st_size - 6000))
                state['log_tail'] = stream.read().decode('utf8', errors='replace')
    return state


def pack_results(job, token):
    authenticate(job, token)
    state = read_json(job / 'remote_state.json')
    if state['status'] in ('prepared', 'queued', 'running'):
        raise RuntimeError('Cannot collect results while job is active')
    paths = [job / 'remote_state.json']
    if (job / 'cache_manifest.json').is_file():
        paths.append(job / 'cache_manifest.json')
    for prefix in ('collection', 'prepared', 'sfm', 'training'):
        directory = job / prefix
        paths.extend(directory.glob('*.json'))
    for prefix in ('export', 'logs', 'training/evaluation'):
        directory = job / prefix
        paths.extend(p for p in directory.rglob('*') if p.is_file())
    archive_path = job / 'results.tar'
    with tarfile.open(archive_path, 'w') as archive:
        for path in sorted(set(paths)):
            if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(job):
                archive.add(path, arcname=path.relative_to(job).as_posix(), recursive=False)
    with archive_path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(archive_path), bytes=archive_path.stat().st_size, sha256=digest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--token', default='')
    parser.add_argument('action', choices=('init', 'start', 'supervise', 'status', 'inspect', 'cancel', 'pack'))
    args = parser.parse_args(); job = job_path(args.root, args.job_id)
    if args.action == 'init':
        result = init_job(job, args.token, sys.stdin.buffer)
    elif args.action == 'start':
        result = start_job(job, args.token)
    elif args.action == 'supervise':
        supervise(job); return
    elif args.action == 'status':
        result = status(job, args.token)
    elif args.action == 'inspect':
        result = status(job, args.token, renew=False)
    elif args.action == 'cancel':
        authenticate(job, args.token); (job / 'cancel').touch(); result = status(job, args.token, renew=False)
    else:
        result = pack_results(job, args.token)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
