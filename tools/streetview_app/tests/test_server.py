import json
import hashlib
from http.server import ThreadingHTTPServer
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import pytest
from tools.streetview_app.jobs import JobManager
from tools.streetview_app.plans import freeze_selection
from tools.streetview_app.server import Application, make_handler


class Provider:
    def __init__(self):
        self.points = {'first': dict(id='first', lat=35., lng=128., captured_at='2025-04-06 10:30:00', capture_date='2025-04-06', capture_precision='second'),
            'month': dict(id='month', lat=35., lng=128., captured_at='2025-04', capture_date='2025-04', capture_precision='month'),
            'far': dict(id='far', lat=36., lng=128., captured_at='2025-04-06', capture_date='2025-04-06', capture_precision='day')}
    def get_panorama(self, pano_id):
        return self.points[pano_id]
    def discover(self, lat, lng, radius_m, **_):
        return dict(panoramas=[self.points['first']], date_options=[])
    def cube_face(self, pano_id, face):
        if pano_id not in self.points or face not in ('front','back','left','right','up','down'):
            raise ValueError('Invalid panorama/face')
        return b'test-image', 'image/jpeg'


def payload(**changes):
    result = dict(center=dict(lat=35., lng=128.), radius_m=100,
        capture_policy=dict(mode='same_day', value='2025-04-06'), panorama_ids=['first'])
    result.update(changes); return result


@pytest.fixture
def app_server(tmp_path):
    app = Application(tmp_path, provider=Provider(), jobs=JobManager(tmp_path/'jobs'))
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(app))
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    yield app, f'http://127.0.0.1:{server.server_port}'
    server.shutdown(); server.server_close(); thread.join()


def post(url, data, origin=None):
    headers = {'Content-Type':'application/json'}
    if origin: headers['Origin'] = origin
    request = Request(url, json.dumps(data).encode(), headers=headers)
    with urlopen(request, timeout=5) as response:
        return response.status, json.load(response)


def test_live_routes_save_verified_plan_no_generation(app_server):
    app, base = app_server
    with urlopen(base+'/api/status') as response:
        assert json.load(response)['capabilities']['generate'] is False
    status, value = post(base+'/api/plans', payload(), origin=base)
    assert status == 201 and value['plan']['selection_verified']
    assert len(app.plans.list()) == 1 and app.jobs.list() == []
    with urlopen(base+value['download_url']) as response:
        assert response.headers.get('Content-Disposition').startswith('attachment')
        assert json.load(response)['panoramas'][0]['captured_at'] == '2025-04-06 10:30:00'
    with pytest.raises(HTTPError) as error:
        post(base+'/api/jobs', payload(), origin=base)
    assert error.value.code == 409 and app.jobs.list() == []


def test_quality_report_requires_completed_bound_artifact(app_server):
    app, base = app_server
    job_id = 'b' * 32
    directory = app.jobs.root / job_id / 'export'
    directory.mkdir(parents=True)
    raw = b'{"status":"completed","quality":{"quality_improved":false}}'
    (directory / 'report.json').write_bytes(raw)
    app.jobs._jobs[job_id] = dict(id=job_id, status='running',
        quality_report=dict(path='export/report.json', sha256=hashlib.sha256(raw).hexdigest()))
    with pytest.raises(HTTPError) as error:
        urlopen(base + '/api/jobs/' + job_id + '/report')
    assert error.value.code == 409
    app.jobs._jobs[job_id]['status'] = 'completed'
    with urlopen(base + '/api/jobs/' + job_id + '/report') as response:
        assert response.read() == raw
        assert response.headers['Content-Disposition'].startswith('attachment')
    (directory / 'report.json').write_bytes(b'changed')
    with pytest.raises(HTTPError) as error:
        urlopen(base + '/api/jobs/' + job_id + '/report')
    assert error.value.code == 409


def test_foreign_origin_cannot_start_or_save(app_server):
    app, base = app_server
    with pytest.raises(HTTPError) as error:
        post(base+'/api/plans', payload(), origin='https://unrelated.example')
    assert error.value.code == 403 and app.plans.list() == []


def test_metadata_overrides_client_claims():
    with pytest.raises(ValueError, match='촬영일'):
        freeze_selection(payload(panorama_ids=['month']), Provider())
    with pytest.raises(ValueError, match='반경'):
        freeze_selection(payload(panorama_ids=['far']), Provider())
    result = freeze_selection(dict(payload(), panoramas=[dict(id='first', captured_at='invented')]), Provider())
    assert result['panoramas'][0]['captured_at'] == '2025-04-06 10:30:00'


def test_capture_clock_filter_is_explicit_and_fails_unknown():
    policy = dict(mode='same_day', value='2025-04-06', time_start='10:00', time_end='11:00')
    assert freeze_selection(payload(capture_policy=policy), Provider())['selection_verified']
    policy['time_end'] = '10:15'
    with pytest.raises(ValueError, match='시각 범위'):
        freeze_selection(payload(capture_policy=policy), Provider())
    policy = dict(mode='same_month', value='2025-04', time_start='10:00', time_end='11:00')
    with pytest.raises(ValueError, match='촬영 시각이 없는'):
        freeze_selection(payload(capture_policy=policy, panorama_ids=['month']), Provider())


def test_path_traversal_and_bad_preview_rejected(app_server):
    _, base = app_server
    with pytest.raises(HTTPError) as error:
        urlopen(base+'/%2e%2e/server.py')
    assert error.value.code == 404
    with pytest.raises(HTTPError) as error:
        urlopen(base+'/api/cube/first/invalid')
    assert error.value.code == 400


@pytest.mark.parametrize('mutation', [dict(lat=float('nan')), dict(lng=float('inf')), dict(id='mismatch')])
def test_invalid_provider_metadata_is_rejected(mutation):
    provider = Provider()
    provider.points['first'].update(mutation)
    with pytest.raises(ValueError):
        freeze_selection(payload(), provider)


@pytest.mark.parametrize('change', [dict(center=None), dict(capture_policy=[]), dict(radius_m=True)])
def test_malformed_selection_returns_400(app_server, change):
    app, base = app_server
    with pytest.raises(HTTPError) as error:
        post(base + '/api/plans', payload(**change))
    assert error.value.code == 400
    assert not app.plans.list()


def test_configured_http_job_exports_only_verified_actual_ply(app_server, tmp_path):
    from tools.streetview_app.tests.test_jobs import backend, selection
    app, base = app_server
    frozen = selection()
    app.provider.points = {point['id']: point for point in frozen['panoramas']}
    app.jobs = JobManager(tmp_path / 'configured_jobs', backend(tmp_path))
    request = {key: frozen[key] for key in ('center', 'radius_m', 'capture_policy', 'panorama_ids')}
    status, response = post(base + '/api/jobs', request, origin=base)
    assert status == 201
    done = app.jobs.wait(response['job']['id'], 15)
    assert done['status'] == 'completed', done
    url = base + '/api/jobs/' + done['id'] + '/download'
    with urlopen(url) as response:
        data = response.read()
        assert data.startswith(b'ply\nformat binary_little_endian')
        assert 'attachment' in response.headers['Content-Disposition']
    path = app.jobs.root / done['id'] / done['artifact']['path']
    path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    with pytest.raises(HTTPError) as error:
        urlopen(url)
    assert error.value.code == 409
    app.jobs.close()
