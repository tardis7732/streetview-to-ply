"""Explicit input exclusions survive plans/jobs and never enter collection.

Provider traffic is replaced with synthetic native tiles; there is no cloud or
GPU workload. Local fixture processes verify the job configuration hand-off.
"""
import copy
import io
import json
import tarfile
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import urlopen

import pytest

from tools.streetview_app.jobs import JobManager, STAGES, validate_config
from tools.streetview_app.plans import PlanStore, freeze_selection
from tools.streetview_app.remote_jobs import source_bundle
from tools.streetview_app.tests.test_jobs import backend, selection
from tools.streetview_app.tests.test_server import Provider, app_server, payload, post


def test_saved_exclusions_are_provenance_and_request_no_metadata(app_server, monkeypatch):
    app, base = app_server
    original_get = app.provider.get_panorama
    calls = []

    def selected_only(pano_id):
        calls.append(pano_id)
        return original_get(pano_id)

    monkeypatch.setattr(app.provider, 'get_panorama', selected_only)
    excluded = ['omitted+/A=']  # Deliberately unavailable from this provider.
    code, response = post(base + '/api/plans', payload(excluded_panorama_ids=excluded), origin=base)
    assert code == 201 and calls == ['first']
    plan = response['plan']
    assert plan['panorama_ids'] == ['first']
    assert [point['id'] for point in plan['panoramas']] == ['first']
    assert plan['excluded_panorama_ids'] == excluded
    assert app.plans.get(plan['id']) == plan
    assert app.plans.list()[0]['excluded_panorama_ids'] == excluded
    with urlopen(base + response['download_url']) as download:
        assert json.load(download)['excluded_panorama_ids'] == excluded
    assert not app.jobs.list()


@pytest.mark.parametrize('excluded', [None, 'other', 123, [None], [''], ['bad\x00id'],
    ['x' * 2049], ['duplicate', 'duplicate'], ['first']])
def test_invalid_or_overlapping_exclusions_rejected_before_provider(excluded):
    class NoRequests:
        def get_panorama(self, _):
            pytest.fail('Invalid selection must be rejected before metadata requests')

    with pytest.raises(ValueError):
        freeze_selection(payload(excluded_panorama_ids=excluded), NoRequests())
    config = selection()
    config['excluded_panorama_ids'] = [config['panorama_ids'][0]] if excluded == ['first'] else excluded
    with pytest.raises(ValueError):
        validate_config(config)


def test_all_excluded_selection_cannot_be_saved_or_generated(app_server, tmp_path):
    app, base = app_server
    app.jobs = JobManager(tmp_path / 'configured_jobs', backend(tmp_path))
    request = payload(panorama_ids=[], excluded_panorama_ids=['first'])
    try:
        for route in ('/api/plans', '/api/jobs'):
            with pytest.raises(HTTPError) as error:
                post(base + route, request, origin=base)
            assert error.value.code == 400
        assert app.plans.list() == [] and app.jobs.list() == []
    finally:
        app.jobs.close()


def test_job_and_each_stage_receive_only_included_ids(app_server, tmp_path):
    app, base = app_server
    frozen = selection()
    # New UI supplies date mode/value only; legacy clock filters still work in
    # their existing tests but are not required by this generation request.
    frozen['capture_policy'] = dict(mode='same_day', value='2024-06-12')
    app.provider.points = {point['id']: point for point in frozen['panoramas']}
    app.jobs = JobManager(tmp_path / 'configured_jobs', backend(tmp_path))
    request = {key: frozen[key] for key in ('center', 'radius_m', 'capture_policy', 'panorama_ids')}
    request['excluded_panorama_ids'] = ['omitted+/A=', 'omitted/B==']
    try:
        code, response = post(base + '/api/jobs', request, origin=base)
        assert code == 201
        done = app.jobs.wait(response['job']['id'], 15)
        assert done['status'] == 'completed', done
        root = app.jobs.root / done['id']
        config = json.loads((root / 'config.json').read_text(encoding='utf8'))
        assert config['excluded_panorama_ids'] == request['excluded_panorama_ids']
        assert config['panorama_ids'] == frozen['panorama_ids']
        assert [p['id'] for p in config['panoramas']] == frozen['panorama_ids']
        assert config['capture_policy']['time_start'] is None
        assert config['capture_policy']['time_end'] is None
        for stage in STAGES:
            assert json.loads((root / (stage + '.json')).read_text())['panorama_ids'] == frozen['panorama_ids']
    finally:
        app.jobs.close()


def test_exclusion_normalization_copies_caller_and_preserves_old_config():
    original = selection()
    assert validate_config(original)['excluded_panorama_ids'] == []
    assert 'excluded_panorama_ids' not in original
    original['excluded_panorama_ids'] = ['omitted']
    saved = copy.deepcopy(original)
    normalized = validate_config(original)
    normalized['excluded_panorama_ids'].append('another')
    assert original == saved


def test_five_hundred_selected_and_uncapped_exclusions_survive_plan_cloud_and_collection_roster(tmp_path):
    from tools.streetview_engine.collection import _validate_selection

    count = 500
    source = selection(count)
    request = {key: source[key] for key in ('center', 'radius_m', 'capture_policy', 'panorama_ids')}
    excluded = [f'omitted+/{index}=' for index in range(1018)]
    request['excluded_panorama_ids'] = excluded
    points = {point['id']: point for point in source['panoramas']}
    calls = []

    class SelectedProvider:
        def get_panorama(self, panorama_id):
            calls.append(panorama_id)
            return points[panorama_id]

    frozen = freeze_selection(request, SelectedProvider())
    assert calls == source['panorama_ids']
    store = PlanStore(tmp_path / 'plans')
    saved = store.save(frozen)
    assert store.get(saved['id'])['excluded_panorama_ids'] == excluded
    assert store.get(saved['id'])['panorama_ids'] == source['panorama_ids']

    config = validate_config(frozen)
    assert config['max_panoramas'] == count
    packed, _ = source_bundle(config, {}, (), SimpleNamespace(lease_seconds=90, timeout_seconds=600))
    with tarfile.open(fileobj=io.BytesIO(packed), mode='r:gz') as archive:
        transmitted = json.load(archive.extractfile('config.json'))
    roster, metadata = _validate_selection(transmitted)
    assert roster == source['panorama_ids']
    assert list(metadata) == source['panorama_ids']
    assert transmitted['excluded_panorama_ids'] == excluded
    assert not set(excluded) & set(metadata)
    assert transmitted['panoramas'] == frozen['panoramas']


@pytest.mark.parametrize('count', [501, 1001])
def test_large_selected_rosters_preserved_by_plan_job_and_collection(count):
    from tools.streetview_engine.collection import _validate_selection
    config = selection(count)
    provider = Provider()
    provider.points = {point['id']: dict(point, capture_precision='second', capture_date=point['captured_at'][:10]) for point in config['panoramas']}
    frozen = freeze_selection(config, provider)
    assert len(frozen['panorama_ids']) == count
    assert len(validate_config(frozen)['panorama_ids']) == count
    assert len(_validate_selection(frozen)[0]) == count


def test_actual_collection_and_cloud_bundle_exclude_omitted_inputs(tmp_path, monkeypatch):
    from tools.streetview_engine import collection
    from tools.streetview_engine.tests.test_collection import selection as collection_selection, source_response

    request = collection_selection()
    request['capture_policy'] = dict(mode='same_day', value='2024-03-02')
    excluded = ['omitted+/B==']
    request['excluded_panorama_ids'] = excluded
    provider = Provider()
    provider.points = {point['id']: dict(point, capture_date='2024-03-02', capture_precision='second') for point in request['panoramas']}
    config = validate_config(freeze_selection(request, provider))
    # Exercise the exact config serialization used for an explicit cloud job,
    # without making SSH requests or submitting any work.
    packed, _ = source_bundle(config, {}, (), SimpleNamespace(lease_seconds=90, timeout_seconds=600))
    with tarfile.open(fileobj=io.BytesIO(packed), mode='r:gz') as archive:
        transmitted = json.load(archive.extractfile('config.json'))
    assert transmitted == config
    calls = []

    def selected_sources_only(url, timeout, session=None):
        calls.append(url)
        assert quote(excluded[0], safe='') not in url
        assert quote(config['panorama_ids'][0], safe='') in url
        return source_response(url, timeout, session)

    monkeypatch.setattr(collection, '_request', selected_sources_only)
    result = collection.run(transmitted, tmp_path, {'max_workers': 1})
    assert len(calls) == 25  # One selected metadata record plus 24 native tiles.
    assert result['panorama_count'] == 1
    assert [point['pano_id'] for point in result['stations']] == config['panorama_ids']
    assert len(list((tmp_path / 'collection/images').glob('*.png'))) == 6
    stored_input = json.loads((tmp_path / 'collection/input.json').read_text(encoding='utf8'))
    assert stored_input['config']['excluded_panorama_ids'] == excluded
    assert collection.run(transmitted, tmp_path, {'max_workers': 1}) == result
    assert len(calls) == 25
