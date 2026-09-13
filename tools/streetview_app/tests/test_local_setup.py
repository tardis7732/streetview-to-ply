import json
import os
import subprocess
import sys

from tools.streetview_app.jobs import Backend, JobManager, STAGES
from tools.streetview_app.local_setup import backend_profile
from tools.streetview_app.recipes import backend_snapshot, operator_settings


def test_local_stage_imports_from_job_directory_without_ssh(tmp_path):
    settings = tmp_path / 'operator {local}.json'
    settings.write_text('{}', encoding='utf-8')
    backend = Backend.from_dict(backend_profile(settings))
    job = tmp_path / 'job with spaces'
    job.mkdir()
    assert backend.compute == 'local' and not backend.remote
    assert tuple(s.name for s in backend.stages) == STAGES
    argv = [part.format(job_dir=str(job), config=str(job/'config.json'), python=sys.executable)
            for part in backend.stages[0].argv]
    assert argv[-1] == str(settings)
    # Help exercises the real module entry point in the job's cwd, without
    # fetching provider data or loading GPU models.
    result = subprocess.run(argv + ['--help'], cwd=job,
        env={**os.environ, **backend.environment}, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    manager = JobManager(tmp_path/'jobs', backend)
    assert manager.capabilities()['generation_available']


def test_local_operator_settings_are_used_and_changes_invalidate_snapshot(tmp_path):
    settings = tmp_path/'operator.json'
    settings.write_text(json.dumps({'workflow': 'panorama_brush_refine'}), encoding='utf-8')
    backend = Backend.from_dict(backend_profile(settings))
    assert operator_settings(backend)['workflow'] == 'panorama_brush_refine'
    before = backend_snapshot(backend)['identity_sha256']
    settings.write_text(json.dumps({'workflow': 'panorama_brush_refine', 'changed': True}), encoding='utf-8')
    assert backend_snapshot(backend)['identity_sha256'] != before
