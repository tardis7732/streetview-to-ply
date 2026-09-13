from io import BytesIO
from pathlib import Path
from urllib.parse import unquote

from PIL import Image
import pytest

from tools.streetview_app.provider import NaverProvider, capture_fields, valid_id


def basic(pano_id, lng=128., links=(), captured='2024-03-04 12:34:56'):
    return dict(id=pano_id, latitude=35., longitude=lng, proj_type='cubic',
        camera_angle=[0, 90, 0], info=dict(photodate=captured, timeline_id='a', title='Test'),
        links=[dict(id=i, latitude=35., longitude=x) for i, x in links])


class FixtureProvider(NaverProvider):
    def __init__(self, root):
        super().__init__(root)
        self.calls = []
        self.data = {'a': basic('a', links=[('b', 128.0001), ('outside', 130.)]),
            'b': basic('b', 128.0001, [('a', 128.), ('c', 128.0002)]),
            'c': basic('c', 128.0002),
            'old': basic('old', captured='2020-01-02 03:04:05')}

    def _json(self, url):
        self.calls.append(url)
        if '/nearby/' in url:
            return dict(features=[dict(properties=dict(id='a'))])
        if '/timeline/' in url:
            return dict(timeline=dict(panoramas=[['id', 'lon', 'lat', 'type', 'date'],
                ['old', 128., 35., '13', '2020-01-02 03:04:05.0']]))
        if '/around/' in url:
            return dict(panoramas=dict(street=[]))
        return self.data[unquote(url.split('/basic/')[1].split('?')[0])]


@pytest.mark.parametrize('raw,precision,day', [
    ('2024-03-04 12:34:56.0', 'second', '2024-03-04'),
    ('2024-03-04', 'day', '2024-03-04'),
    ('2024-03', 'month', '2024-03'),
    ('2024-02-31', 'unknown', None),
    (None, 'unknown', None),
])
def test_capture_precision_no_invented_timestamp(raw, precision, day):
    result = capture_fields(raw)
    assert result['captured_at'] == raw
    assert result['capture_precision'] == precision
    assert result['capture_date'] == day


def test_connected_discovery_filters_radius_and_retains_history(tmp_path):
    provider = FixtureProvider(tmp_path)
    result = provider.discover(35, 128, 100, max_nodes=20)
    assert {p['id'] for p in result['panoramas']} == {'a', 'b', 'c'}
    assert {d['value'] for d in result['date_options']} == {'2024-03-04', '2020-01-02'}
    assert not result['truncated']
    assert not any('/basic/outside' in url for url in provider.calls)


def test_historical_date_is_expanded_and_time_retained(tmp_path):
    result = FixtureProvider(tmp_path).discover(35, 128, 100, date='2020-01-02')
    assert [p['id'] for p in result['panoramas']] == ['old']
    assert result['panoramas'][0]['captured_at'] == '2020-01-02 03:04:05'
    assert result['panoramas'][0]['heading'] == pytest.approx(90)


def test_page_boundary_has_explicit_continuation(tmp_path):
    result = FixtureProvider(tmp_path).discover(35, 128, 100, max_nodes=1)
    assert len(result['panoramas']) == 1
    assert result['truncated'] and result['continuation']
    assert result['warnings'] == []


@pytest.mark.parametrize('lat,lng,radius', [(float('nan'), 128, 100), (35, 128, float('inf')), (91, 0, 100), (0, 0, 9)])
def test_invalid_discovery_never_requests(tmp_path, lat, lng, radius):
    provider = FixtureProvider(tmp_path)
    with pytest.raises(ValueError):
        provider.discover(lat, lng, radius)
    assert not provider.calls


@pytest.mark.parametrize('projection', ['cubic', 'equirect'])
def test_cube_face_orientation_and_opaque_id_cache(tmp_path, monkeypatch, projection):
    provider = FixtureProvider(tmp_path)
    opaque = 'base64/+value=='
    provider.data[opaque] = basic(opaque)
    provider.data[opaque]['proj_type'] = projection
    colors = [(20, 0, 0), (50, 0, 0), (80, 0, 0), (110, 0, 0), (140, 0, 0), (170, 0, 0)]
    strip = Image.new('RGB', (1536, 256))
    for n, color in enumerate(colors):
        strip.paste(color, (n * 256, 0, (n + 1) * 256, 256))
    buffer = BytesIO(); strip.save(buffer, 'PNG')
    calls = []
    class Response:
        content = buffer.getvalue()
        def raise_for_status(self):
            pass
    def get(url, **kwargs):
        calls.append(url); return Response()
    monkeypatch.setattr('tools.streetview_app.provider.requests.get', get)
    for face, index in zip(('F', 'R', 'B', 'L', 'U', 'D'), (1, 2, 3, 0, 5, 4)):
        payload, mime = provider.cube_face(opaque, face)
        image = Image.open(BytesIO(payload))
        assert image.size == (256, 256) and mime == 'image/jpeg'
        assert abs(image.getpixel((128, 128))[0] - colors[index][0]) <= 2
    assert len(calls) == 1 and '%2F' in calls[0]
    assert len(list(tmp_path.glob('*.jpg'))) == 6
    with pytest.raises(ValueError):
        provider.cube_face(opaque, '../F')
    with pytest.raises(ValueError):
        valid_id('../secret')


def test_changed_preview_shape_is_rejected(tmp_path, monkeypatch):
    provider = FixtureProvider(tmp_path)
    buffer = BytesIO(); Image.new('RGB', (100, 100)).save(buffer, 'PNG')
    class Response:
        content = buffer.getvalue()
        def raise_for_status(self):
            pass
    monkeypatch.setattr('tools.streetview_app.provider.requests.get', lambda *a, **k: Response())
    with pytest.raises(ValueError, match='형식'):
        provider.cube_face('a', 'F')
    assert not list(tmp_path.glob('*.jpg'))
