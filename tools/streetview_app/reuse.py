"""Verified immutable completed-stage copies; never optimizer checkpoint resume."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import stat

from .recipes import backend_snapshot, digest, file_hash

STAGE_DIRS = {'collect': ('collection',), 'preprocess': ('prepared',),
              'sfm': ('sfm',), 'train': ('training', 'depth_prior', 'cleanup'),
              'infer': ('single_panorama',), 'export': ('export',)}


def safe_path(root, name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
        raise ValueError('Invalid cache artifact path')
    root = Path(root).resolve()
    target = root / path
    if not target.resolve().is_relative_to(root):
        raise ValueError('Cache path escapes job directory')
    for component in (target, *target.parents):
        if component == root:
            break
        if component.exists():
            info = component.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
                raise ValueError('Cache symlinks and junctions are not accepted')
    return target


def stage_inventory(directory, stage, outputs):
    directory = Path(directory)
    paths = set(outputs)
    for folder in STAGE_DIRS[stage]:
        root = safe_path(directory, folder)
        if root.exists():
            for path in root.rglob('*'):
                safe_path(directory, path.relative_to(directory).as_posix())
                if path.is_file():
                    paths.add(path.relative_to(directory).as_posix())
    rows = {}
    for name in sorted(paths):
        path = safe_path(directory, name)
        if not path.is_file():
            raise ValueError('Completed stage cache is missing: ' + name)
        rows[name] = dict(sha256=file_hash(path), bytes=path.stat().st_size)
    return rows


def record_stage_cache(directory, stage, outputs):
    path = Path(directory) / 'cache_manifest.json'
    manifest = json.loads(path.read_text(encoding='utf8')) if path.exists() else dict(schema_version=1, stages={})
    manifest['stages'][stage] = stage_inventory(directory, stage, outputs)
    from .jobs import _write_json
    _write_json(path, manifest)
    return dict(path='cache_manifest.json', sha256=file_hash(path))


def verify_files(directory, files):
    for name, row in files.items():
        path = safe_path(directory, name)
        if not path.is_file() or path.stat().st_size != row['bytes'] or file_hash(path) != row['sha256']:
            raise ValueError('Completed cache artifact changed: ' + name)


def copy_files(source, target, files):
    """Ordinary copies isolate previous jobs from later stage writers."""
    verify_files(source, files)
    for name, row in files.items():
        src, dst = safe_path(source, name), safe_path(target, name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with src.open('rb') as left, dst.open('xb') as right:
            shutil.copyfileobj(left, right, 1024 * 1024)
        if file_hash(dst) != row['sha256']:
            raise ValueError('Cache changed during copy: ' + name)
    verify_files(source, files)


class ReuseService:
    def __init__(self, manager):
        self.manager = manager

    def _source(self, job_id):
        state = self.manager.get(job_id)
        if state['status'] != 'completed':
            raise ValueError('Only completed jobs can supply reusable stages')
        if state.get('execution_snapshot') != backend_snapshot(self.manager.backend):
            raise ValueError('Source job lacks a matching verified code/operator snapshot')
        directory = self.manager.root / job_id
        config_path = directory / 'config.json'
        if file_hash(config_path) != state['config_sha256'] or json.loads(config_path.read_text(encoding='utf8')) != state['config']:
            raise ValueError('Source frozen configuration changed')
        binding = state.get('cache_manifest')
        if not binding or binding.get('path') != 'cache_manifest.json':
            raise ValueError('This earlier job has no complete-stage cache manifest')
        path = directory / 'cache_manifest.json'
        if not path.is_file() or file_hash(path) != binding['sha256']:
            raise ValueError('Source cache manifest changed')
        manifest = json.loads(path.read_text(encoding='utf8'))
        return state, directory, manifest

    def inspect(self, job_id):
        from .jobs import generation_mode, stage_names
        job = self.manager.get(job_id)
        choices = []
        reason = None
        try:
            state, directory, manifest = self._source(job_id)
        except (ValueError, OSError) as error:
            reason = str(error)
        names = stage_names(generation_mode(job['config']))
        for index, stage in enumerate(names[1:], 1):
            why = reason
            files = {}
            if why is None:
                try:
                    for prefix in names[:index]:
                        if not manifest['stages'].get(prefix):
                            raise ValueError('Missing complete cache for ' + prefix)
                        files.update(manifest['stages'][prefix])
                    if self.manager.backend.compute == 'local':
                        verify_files(directory, files)
                    else:
                        # Full images/checkpoints stay on cloud; the owned worker
                        # rehashes all bytes before copying or skipping a stage.
                        if not (directory / 'remote_owner.json').is_file():
                            raise ValueError('Cloud cache ownership receipt is missing')
                except (ValueError, OSError, KeyError) as error:
                    why = str(error)
            choices.append(dict(from_stage=stage, available=why is None, reason=why,
                reused_stages=list(names[:index]), files_count=len(files), bytes=sum(r['bytes'] for r in files.values())))
        return dict(job_id=job_id, stages=choices, exact_settings_only=True,
            optimizer_checkpoint_resume=False, cloud_reverified_before_execution=self.manager.backend.compute != 'local')

    def prepare(self, job_id, from_stage, overrides=None):
        from .jobs import generation_mode, stage_names
        if overrides:
            raise ValueError('Stage reuse requires identical settings; use the PLY cleanup panel for export changes')
        state, directory, manifest = self._source(job_id)
        names = stage_names(generation_mode(state['config']))
        if from_stage not in names[1:]:
            raise ValueError('Restart stage must follow collect')
        prefix = names[:names.index(from_stage)]
        files = {}
        records = []
        for name in prefix:
            rows = manifest['stages'].get(name)
            if not rows:
                raise ValueError('Missing complete cache for ' + name)
            files.update(rows)
            record = next((x for x in state['stages'] if x['name'] == name), None)
            if not record or record['status'] != 'completed':
                raise ValueError('Source stage did not complete')
            records.append(record)
        if self.manager.backend.compute == 'local':
            verify_files(directory, files)
        return dict(source_job_id=job_id, from_stage=from_stage, reused_stages=list(prefix),
            files=files, records=records, cache_manifest_sha256=state['cache_manifest']['sha256'],
            cache_stages={name: manifest['stages'][name] for name in prefix},
            config=state['config'], execution_recipe=state.get('execution_recipe'),
            source_execution_identity=state['execution_snapshot']['identity_sha256'])

    def start(self, job_id, from_stage, overrides=None):
        prepared = self.prepare(job_id, from_stage, overrides)
        return self.manager.start(prepared.pop('config'), execution_recipe=prepared.pop('execution_recipe'), reuse=prepared)
