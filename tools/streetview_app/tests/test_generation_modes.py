"""Retired generation modes cannot enter provider or compute work."""
import json
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from tools.streetview_app.cloud_setup import backend_profile
from tools.streetview_app.jobs import JobManager, STAGES
from tools.streetview_app.plans import freeze_selection
from tools.streetview_app.remote_jobs import source_bundle
from tools.streetview_app.tests.test_jobs import selection
from tools.streetview_app.tests.test_server import app_server, payload, post


def test_single_selection_fails_before_provider_access():
    class NeverProvider:
        def get_panorama(self, _):
            pytest.fail('Retired mode reached the provider')
    with pytest.raises(ValueError, match='multi_view'):
        freeze_selection(payload(generation_mode='single_panorama'), NeverProvider())


def test_manager_and_cloud_bundle_cannot_start_single_mode(tmp_path):
    config = dict(selection(1), generation_mode='single_panorama')
    manager = JobManager(tmp_path/'jobs')
    with pytest.raises(ValueError, match='multi_view'):
        manager.start(config)
    with pytest.raises(ValueError, match='multi_view'):
        source_bundle(config, {}, (), SimpleNamespace())
    assert manager.list() == []
    assert list((tmp_path/'jobs').iterdir()) == []


def test_http_status_has_only_multi_view_and_single_posts_fail(app_server):
    app, base = app_server
    with urlopen(base+'/api/status') as response:
        assert set(json.load(response)['capabilities']['generation_modes']) == {'multi_view'}
    for route in ('/api/plans', '/api/jobs'):
        with pytest.raises(HTTPError) as error:
            post(base+route, payload(generation_mode='single_panorama'), base)
        assert error.value.code == 400
    assert not app.jobs.list() and not app.plans.list()


def test_cloud_profile_registers_only_multi_view_with_explicit_host(tmp_path):
    settings = tmp_path/'settings.json'; settings.write_text('{}')
    profile = backend_profile(settings, host='my-gpu')
    assert [stage['name'] for stage in profile['stages']] == list(STAGES)
    assert not profile.get('mode_stages')
    assert profile['remote']['host'] == 'my-gpu'
    assert profile['remote']['worker'] == '/srv/streetview-to-ply/code/tools/streetview_engine/remote_worker.py'
