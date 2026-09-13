import hashlib
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from tools.streetview_app.ply_filters import PlyFilterManager
from tools.streetview_app.tests.test_server import app_server, post
from tools.streetview_engine.tests.test_size_filter import fixture


def wait_job(manager, job_id):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        job = manager.get(job_id)
        if job['status'] not in ('queued', 'running'):
            return job
        time.sleep(.02)
    pytest.fail('CPU fixture did not finish')


def inputs(tmp_path):
    ply, cameras = fixture(tmp_path / 'inputs')
    return dict(source_ply=str(ply), camera_json=str(cameras), max_sigma_camera_radius_ratio=.5)


def test_explicit_api_filter_and_verified_download_without_training(app_server, tmp_path):
    app, base = app_server
    payload = inputs(tmp_path)
    original = Path(payload['source_ply']).read_bytes()
    defaults_path = app.filters.defaults_path
    defaults_path.write_text(json.dumps(payload), encoding='utf-8')
    try:
        with urlopen(base + '/api/ply-filters') as response:
            initial = json.load(response)
        assert initial == dict(jobs=[], defaults=payload)
        assert app.jobs.list() == []
        status, response = post(base + '/api/ply-filters', payload, origin=base)
        assert status == 201
        job = wait_job(app.filters, response['job']['id'])
        assert job['status'] == 'completed', job
        assert (job['source_rows'], job['removed_rows'], job['remaining_rows']) == (4, 2, 2)
        assert job['original_preserved'] and job['training_started'] is False
        assert app.jobs.list() == []
        assert Path(payload['source_ply']).read_bytes() == original
        with urlopen(base + job['download_url']) as download:
            data = download.read()
            assert download.headers['Content-Disposition'].startswith('attachment;')
        assert hashlib.sha256(data).hexdigest() == job['artifact']['sha256']
        assert b'element vertex 2\n' in data
        saved = app.filters.root / job['id'] / job['artifact']['path']
        saved.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
        with pytest.raises(HTTPError) as error:
            urlopen(base + job['download_url'])
        assert error.value.code == 409
    finally:
        app.filters.close()


def test_api_rejects_bad_input_and_foreign_origin(app_server, tmp_path):
    app, base = app_server
    payload = inputs(tmp_path)
    try:
        with pytest.raises(HTTPError) as error:
            post(base + '/api/ply-filters', payload, origin='https://unrelated.example')
        assert error.value.code == 403
        for changes in [dict(max_sigma_camera_radius_ratio=True), dict(max_sigma_camera_radius_ratio=0),
                        dict(source_ply='relative.ply'), dict(source_ply='//host/share/source.ply'),
                        dict(unrecognized=True), dict(size_filter_enabled=1),
                        dict(crop={'enabled':True,'radius_camera_radius_ratio':0}), dict(crop={'unknown':True})]:
            with pytest.raises(HTTPError) as error:
                post(base + '/api/ply-filters', dict(payload, **changes), origin=base)
            assert error.value.code == 400
        assert app.filters.list() == [] and app.jobs.list() == []
    finally:
        app.filters.close()


def test_crop_api_and_reusable_output_camera_reference(app_server, tmp_path):
    from tools.streetview_engine.tests.test_ply_cleanup import fixture as crop_fixture
    app, base = app_server
    ply, camera = crop_fixture(tmp_path/'crop-input')
    payload = dict(source_ply=str(ply),camera_json=str(camera),size_filter_enabled=False,
                   crop=dict(enabled=True,radius_camera_radius_ratio=1.))
    try:
        _,response = post(base+'/api/ply-filters',payload,origin=base)
        job=wait_job(app.filters,response['job']['id'])
        assert job['status']=='completed',job
        assert job['size_removed_rows']==0 and job['crop_additional_removed_rows']==1
        assert job['remaining_rows']==3 and job['crop_geometry']['height']=='unlimited'
        assert Path(job['output_camera_json']).is_file()
        # The resulting file has a usable camera binding for another explicit cleanup.
        artifact=app.filters.root/job['id']/job['artifact']['path']
        _,response=post(base+'/api/ply-filters',dict(source_ply=str(artifact),camera_json=job['output_camera_json'],
                          size_filter_enabled=False),origin=base)
        again=wait_job(app.filters,response['job']['id'])
        assert again['status']=='completed',again
        assert again['artifact']['sha256']==job['artifact']['sha256'] and again['removed_rows']==0
        assert app.jobs.list()==[]
    finally:
        app.filters.close()


def test_no_empty_output_is_published_and_interrupted_jobs_do_not_restart(tmp_path):
    payload = inputs(tmp_path)
    root = tmp_path / 'jobs'
    manager = PlyFilterManager(root)
    try:
        job = manager.start(dict(payload, max_sigma_camera_radius_ratio=.0001))
        result = wait_job(manager, job['id'])
        assert result['status'] == 'failed' and result['artifact'] is None
        assert 'download_url' not in result
        assert not (root / job['id'] / 'result' / 'scene.ply').exists()
    finally:
        manager.close()
    saved = root / job['id'] / 'state.json'
    state = json.loads(saved.read_text(encoding='utf-8'))
    state['status'] = 'running'
    saved.write_text(json.dumps(state), encoding='utf-8')
    reopened = PlyFilterManager(root)
    try:
        assert reopened.get(job['id'])['status'] == 'interrupted'
        assert not (root / job['id'] / 'result').exists()
    finally:
        reopened.close()
