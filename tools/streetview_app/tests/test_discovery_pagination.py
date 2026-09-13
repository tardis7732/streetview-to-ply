"""Continuation uses synthetic provider graphs; no Naver bulk requests."""
import json
import os
from http.server import ThreadingHTTPServer
import threading
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

import pytest

from tools.streetview_app.jobs import JobManager
from tools.streetview_app.provider import NaverProvider, capture_fields
from tools.streetview_app.server import Application, make_handler


class ChainProvider(NaverProvider):
    def __init__(self, root, count=1205, **kwargs):
        super().__init__(root, **kwargs)
        self.count = count
        self.calls = []

    def _json(self, url):
        self.calls.append(url)
        if '/nearby/' in url:
            return {'features': [{'properties': {'id': 'n0'}}]}
        if '/around/' in url:
            return {'panoramas': {'street': []}}
        raise AssertionError(url)

    def _history(self, anchor):
        return []

    def get_panorama(self, pano_id):
        self.calls.append(pano_id)
        index = int(pano_id[1:])
        assert 0 <= index < self.count
        return dict(id=pano_id, lat=35., lng=128., title='Fixture',
            projection='cubic', **capture_fields('2024-03-04 12:34:56.125'),
            links=[dict(id=f'n{i}', lat=35., lng=128.)
                for i in (index - 1, index + 1, index + 2) if 0 <= i < self.count])


def test_more_than_one_thousand_disjoint_pages_without_missing_ids(tmp_path):
    provider = ChainProvider(tmp_path)
    token = None
    seen = set()
    page_count = 0
    while True:
        page = provider.discover(35., 128., 100.125, max_nodes=200, continuation=token)
        ids = {p['id'] for p in page['panoramas']}
        assert len(ids) == len(page['panoramas']) <= 200
        assert not (seen & ids)
        seen.update(ids)
        page_count += 1
        assert page['nodes_examined'] <= 200
        assert page['nodes_examined_total'] == len(seen)
        assert page['coverage'] == 'bounded_connected_search'
        assert page['radius_m'] == 100.125
        assert page['truncated'] == bool(page['continuation'])
        assert page['date_options'] == [dict(value='2024-03-04', label='2024-03-04',
            count=len(seen), precision='day')]
        assert all(p['captured_at'] == '2024-03-04 12:34:56.125'
            and p['capture_precision'] == 'second' for p in page['panoramas'])
        token = page['continuation']
        if token is None:
            break
        assert page_count < 10
    assert seen == {f'n{i}' for i in range(1205)}
    assert page_count == 7
    assert sum(call.startswith('n') for call in provider.calls) == 1205


def test_batch_guard_is_configurable_and_not_a_total_500_limit(tmp_path):
    provider = ChainProvider(tmp_path, request_limit=700)
    first = provider.discover(35, 128, 100, max_nodes=600)
    assert len(first['panoramas']) == 600
    second = provider.discover(35, 128, 100, max_nodes=100000,
        continuation=first['continuation'])
    assert second['page_size'] == 700
    assert len(second['panoramas']) == 605
    assert second['nodes_examined_total'] == 1205
    assert second['continuation'] is None


def test_continuation_replay_survives_provider_restart_without_advancing(tmp_path):
    provider = ChainProvider(tmp_path, count=9)
    first = provider.discover(35, 128, 100, max_nodes=2)
    token = first['continuation']
    second = provider.discover(35, 128, 100, max_nodes=2, continuation=token)
    resumed = ChainProvider(tmp_path, count=9)
    replay = resumed.discover(35, 128, 100, max_nodes=20, continuation=token)
    assert replay == second
    assert resumed.calls == []
    third = resumed.discover(35, 128, 100, max_nodes=20, continuation=second['continuation'])
    assert {p['id'] for page in (first, second, third) for p in page['panoramas']} == {f'n{i}' for i in range(9)}
    assert sum(len(page['panoramas']) for page in (first, second, third)) == 9


@pytest.mark.parametrize('changes', [dict(lat=35.00000001), dict(lng=128.00000001),
    dict(radius_m=100.00000001), dict(date='2024-03')])
def test_continuation_scope_cannot_change(tmp_path, changes):
    provider = ChainProvider(tmp_path, count=4)
    first = provider.discover(35, 128, 100, max_nodes=1)
    args = dict(lat=35, lng=128, radius_m=100, continuation=first['continuation'])
    args.update(changes)
    calls = provider.calls[:]
    with pytest.raises(ValueError, match='바뀌었습니다'):
        provider.discover(**args)
    assert provider.calls == calls


@pytest.mark.parametrize('token', ['https://example.com/metadata', '../anything',
    'a' * 63, 'g' * 64, 'a' * 64])
def test_invalid_or_unknown_token_never_accesses_provider(tmp_path, token):
    provider = ChainProvider(tmp_path)
    with pytest.raises(ValueError, match='다시 검색'):
        provider.discover(35, 128, 100, continuation=token)
    assert provider.calls == []


def test_unissued_page_token_and_expired_search_are_explicit(tmp_path):
    provider = ChainProvider(tmp_path, count=4, discovery_ttl_s=30)
    first = provider.discover(35, 128, 100, max_nodes=1)
    token = first['continuation']
    calls = provider.calls[:]
    with pytest.raises(ValueError, match='올바르지'):
        provider.discover(35, 128, 100, continuation=token[:32] + '0' * 32)
    path = provider._discovery_dir / (token[:32] + '.sqlite3')
    expired = path.stat().st_mtime - 60
    os.utime(path, (expired, expired))
    with pytest.raises(ValueError, match='만료'):
        provider.discover(35, 128, 100, continuation=token)
    assert provider.calls == calls


def test_repeated_searches_do_not_cap_sessions_and_expired_data_is_reclaimed(tmp_path):
    provider = ChainProvider(tmp_path, count=4, discovery_ttl_s=30)
    first = provider.discover(35, 128, 100, max_nodes=1)
    for _ in range(35):
        provider.discover(35, 128, 100, max_nodes=1)
    second = provider.discover(35, 128, 100, continuation=first['continuation'])
    assert second['nodes_examined_total'] == 4
    path = provider._discovery_dir / (first['continuation'][:32] + '.sqlite3')
    expired = path.stat().st_mtime - 60
    os.utime(path, (expired, expired))
    provider.discover(35, 128, 100, max_nodes=1)
    assert not path.exists()
    assert len(list(provider._discovery_dir.glob('*.sqlite3'))) == 36


def test_failed_initialization_does_not_leave_session_and_messages_are_unicode(tmp_path):
    class FailingProvider(ChainProvider):
        def _json(self, url):
            raise RuntimeError('Synthetic initialization failure')
    provider = FailingProvider(tmp_path)
    with pytest.raises(RuntimeError, match='Synthetic'):
        provider.discover(35, 128, 100)
    assert not list(provider._discovery_dir.glob('*.sqlite3'))
    assert provider._active_discovery == {}
    import inspect
    source = inspect.getsource(NaverProvider)
    assert '???' not in source and '\ufffd' not in source


def test_matching_disconnected_history_is_not_dropped_between_pages(tmp_path):
    class HistoryProvider(ChainProvider):
        def _history(self, anchor):
            return [dict(id=f'n{i}', lat=35., lng=128., title='Fixture',
                **capture_fields('2020-01')) for i in (1, 2, 3)] + [
                dict(id='outside', lat=36., lng=128., **capture_fields('2020-01'))]

        def get_panorama(self, pano_id):
            p = super().get_panorama(pano_id)
            p['links'] = []
            p.update(capture_fields('2024-03-04' if pano_id == 'n0' else '2020-01'))
            return p

    provider = HistoryProvider(tmp_path, count=4)
    pages = []
    token = None
    while True:
        page = provider.discover(35, 128, 100, date='2020-01', max_nodes=1, continuation=token)
        pages.append(page)
        token = page['continuation']
        if token is None:
            break
    assert [p['id'] for page in pages for p in page['panoramas']] == ['n1', 'n2', 'n3']
    assert all(p['captured_at'] == '2020-01' and p['capture_precision'] == 'month'
        for page in pages for p in page['panoramas'])
    assert pages[-1]['date_options'] == [
        dict(value='2024-03-04', label='2024-03-04', count=1, precision='day'),
        dict(value='2020-01', label='2020-01', count=3, precision='month')]


def test_http_continuation_contract_and_scope_error(tmp_path):
    provider = ChainProvider(tmp_path / 'cache', count=5)
    app = Application(tmp_path, provider=provider, jobs=JobManager(tmp_path / 'jobs'))
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(app))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}/api/discover?'
    params = dict(lat=35, lng=128, radius_m=100, max_nodes=2)
    try:
        with urlopen(base + urlencode(params)) as response:
            first = json.load(response)
        params['continuation'] = first['continuation']
        with urlopen(base + urlencode(params)) as response:
            second = json.load(response)
        assert len(second['panoramas']) == 2
        assert second['nodes_examined_total'] == 4
        params['radius_m'] = 101
        with pytest.raises(HTTPError) as error:
            urlopen(base + urlencode(params))
        assert error.value.code == 400
        assert '바뀌었습니다' in json.load(error.value)['error']
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        app.jobs.close()


def test_failed_metadata_remains_explicit_on_later_successful_pages(tmp_path):
    class PartialProvider(ChainProvider):
        def get_panorama(self, pano_id):
            if pano_id == 'n1':
                raise LookupError('Synthetic missing panorama')
            return super().get_panorama(pano_id)
    provider = PartialProvider(tmp_path, count=6)
    first = provider.discover(35, 128, 100, max_nodes=2)
    assert first['metadata_failures_total'] == 1
    second = provider.discover(35, 128, 100, continuation=first['continuation'])
    assert second['metadata_failures_total'] == 1
    assert second['continuation'] is None
    assert first['warnings'] == second['warnings']
    assert '다시 검색' in second['warnings'][0]
    assert {p['id'] for page in (first, second) for p in page['panoramas']} == {'n0', 'n2', 'n3', 'n4', 'n5'}
