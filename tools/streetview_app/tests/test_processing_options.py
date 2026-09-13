"""Per-job masks survive selection, storage, workers and the cloud bundle.

Only synthetic provider metadata and tiny local fixture processes are used;
these tests neither contact a provider nor start cloud/GPU reconstruction.
"""
import copy
import hashlib
import io
import json
import tarfile
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from tools.streetview_app.jobs import JobManager, STAGES, validate_config
from tools.streetview_app.plans import PlanStore, freeze_selection
from tools.streetview_app.remote_jobs import source_bundle
from tools.streetview_app.tests.test_jobs import backend, selection
from tools.streetview_app.tests.test_server import Provider, app_server, payload, post
from tools.streetview_engine.processing_options import read_processing_options


@pytest.mark.parametrize('remove_sky,mask_dynamic', [(False, False), (False, True), (True, False), (True, True)])
def test_all_mask_options_survive_plan_download_job_stages_and_cloud_bundle(app_server, tmp_path, remove_sky, mask_dynamic):
    app, base = app_server
    expected = dict(remove_sky=remove_sky, mask_dynamic=mask_dynamic)
    source = selection()
    app.provider.points = {point['id']: point for point in source['panoramas']}
    request = {key: source[key] for key in ('center', 'radius_m', 'capture_policy', 'panorama_ids')}
    request['processing_options'] = expected.copy()
    request['excluded_panorama_ids'] = ['excluded+/A=']
    configured = backend(tmp_path)
    # Record what each actual fixture process received through config.json.
    program = tmp_path / 'actual fixture program.py'
    script = program.read_text(encoding='utf8')
    marker = 'config = json.loads(pathlib.Path(config_path).read_text(encoding="utf8"))'
    assert marker in script
    program.write_text(script.replace(marker, marker + '\n(root / (stage + ".options.json")).write_text(json.dumps(config["processing_options"]))'), encoding='utf8')
    app.jobs = JobManager(tmp_path / 'configured_jobs', configured)
    try:
        code, saved = post(base + '/api/plans', request, origin=base)
        assert code == 201
        plan = saved['plan']
        assert plan['processing_options'] == expected
        assert app.plans.get(plan['id'])['processing_options'] == expected
        with urlopen(base + '/api/plans') as response:
            assert json.load(response)['plans'][0]['processing_options'] == expected
        with urlopen(base + saved['download_url']) as response:
            assert json.load(response)['processing_options'] == expected
        assert not app.jobs.list(), 'Saving options must not start generation'

        code, started = post(base + '/api/jobs', request, origin=base)
        assert code == 201
        done = app.jobs.wait(started['job']['id'], 15)
        assert done['status'] == 'completed', done
        assert done['config']['processing_options'] == expected
        assert app.jobs.list()[0]['config']['processing_options'] == expected
        directory = app.jobs.root / done['id']
        config = json.loads((directory / 'config.json').read_text(encoding='utf8'))
        assert config['processing_options'] == expected
        assert config['panorama_ids'] == source['panorama_ids']
        assert config['excluded_panorama_ids'] == ['excluded+/A=']
        for stage in STAGES:
            assert json.loads((directory / (stage + '.options.json')).read_text()) == expected

        packed, hashes = source_bundle(config, {}, configured.stages,
            SimpleNamespace(lease_seconds=90, timeout_seconds=600))
        with tarfile.open(fileobj=io.BytesIO(packed), mode='r:gz') as archive:
            transmitted = json.load(archive.extractfile('config.json'))
            helper = 'code/tools/streetview_engine/processing_options.py'
            assert hashlib.sha256(archive.extractfile(helper).read()).hexdigest() == hashes[helper]
        assert transmitted == config
        assert read_processing_options(transmitted) == expected
    finally:
        app.jobs.close()


@pytest.mark.parametrize('options', [None, False, True, [], (), 'true', 1,
    {'unknown': True}, {'remove_sky': False, 'extra': False},
    *({key: bad} for key in ('remove_sky', 'mask_dynamic') for bad in (None, 0, 1, 'false', 'true', [], {}))])
def test_invalid_processing_options_rejected_before_provider_requests(options):
    class NoRequests:
        def get_panorama(self, _):
            pytest.fail('Invalid processing options must be rejected before metadata requests')

    config = selection()
    config['processing_options'] = options
    for validate in (read_processing_options, validate_config):
        with pytest.raises(ValueError):
            validate(config)
    with pytest.raises(ValueError):
        freeze_selection(payload(processing_options=options), NoRequests())


def test_invalid_http_options_create_no_plan_or_job(app_server, tmp_path):
    app, base = app_server
    app.jobs = JobManager(tmp_path / 'configured_jobs', backend(tmp_path))
    try:
        for route in ('/api/plans', '/api/jobs'):
            with pytest.raises(HTTPError) as error:
                post(base + route, payload(processing_options={'remove_sky': 'false'}), origin=base)
            assert error.value.code == 400
        assert app.plans.list() == []
        assert app.jobs.list() == []
    finally:
        app.jobs.close()


@pytest.mark.parametrize('config', [None, [], 1, True, 'options'])
def test_options_reader_requires_object(config):
    with pytest.raises(ValueError):
        read_processing_options(config)


@pytest.mark.parametrize('options,expected', [
    ({}, dict(remove_sky=False, mask_dynamic=True)),
    ({'remove_sky': True}, dict(remove_sky=True, mask_dynamic=True)),
    ({'mask_dynamic': False}, dict(remove_sky=False, mask_dynamic=False)),
])
def test_explicit_partial_options_use_legacy_field_defaults_without_mutation(options, expected):
    config = selection()
    config['processing_options'] = options.copy()
    original = copy.deepcopy(config)
    read = read_processing_options(config)
    assert read == expected
    normalized = validate_config(config)
    assert normalized['processing_options'] == expected
    frozen = freeze_selection(payload(processing_options=options), Provider())
    assert frozen['processing_options'] == expected
    read['remove_sky'] = not read['remove_sky']
    normalized['processing_options']['mask_dynamic'] = not normalized['processing_options']['mask_dynamic']
    assert config == original


def test_legacy_configs_keep_serialized_options_absent_and_hashes_stable(tmp_path):
    source = selection()
    original = copy.deepcopy(source)
    normalized = validate_config(source)
    before = json.dumps(normalized, sort_keys=True).encode()
    assert read_processing_options(normalized) == dict(remove_sky=False, mask_dynamic=True)
    assert 'processing_options' not in source and 'processing_options' not in normalized
    assert json.dumps(normalized, sort_keys=True).encode() == before
    assert source == original
    frozen = freeze_selection(payload(), Provider())
    assert 'processing_options' not in frozen
    store = PlanStore(tmp_path / 'plans')
    saved = store.save(frozen)
    assert 'processing_options' not in store.get(saved['id'])
    assert 'processing_options' not in store.list()[0]
    packed, _ = source_bundle(normalized, {}, (), SimpleNamespace(lease_seconds=90, timeout_seconds=600))
    with tarfile.open(fileobj=io.BytesIO(packed), mode='r:gz') as archive:
        transmitted = json.load(archive.extractfile('config.json'))
    assert transmitted == normalized and 'processing_options' not in transmitted
