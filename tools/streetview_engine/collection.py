"""Collect original Naver medium cube tiles into six lossless 1024px faces.

Only an explicit run downloads imagery. Opaque IDs come from a frozen selection;
no location-specific defaults or preview-image substitutions are permitted.
"""
from __future__ import annotations

from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import hashlib
import json
import math
from pathlib import Path
import re
import threading
from urllib.parse import quote

import numpy as np
from PIL import Image
import requests

from .imaging import FACES, distance_m, fingerprint, group_physical_stations, inside, png_bytes, sha256, write_bytes, write_json


def valid_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_+/=-]{1,240}', value):
        raise ValueError('Invalid opaque panorama ID')
    return value


def tile_url(pano_id, face, x, y):
    if face not in FACES or type(x) is not int or type(y) is not int or x not in (0, 1) or y not in (0, 1):
        raise ValueError('Medium native cubes have two by two tiles per face')
    return f'https://panorama.pstatic.net/image/{quote(valid_id(pano_id), safe="")}/512/M/{face.lower()}/{x+1}/{y+1}'


def _request(url, timeout_s, session=None):
    if session is None:
        with requests.Session() as owned:
            return _request(url, timeout_s, owned)
    with session.get(url, headers={'Referer': 'https://map.naver.com/', 'User-Agent': 'Mozilla/5.0'}, timeout=(10, timeout_s), stream=True) as response:
        response.raise_for_status()
        content = bytearray()
        for block in response.iter_content(65536):
            content.extend(block)
            if len(content) > 8 * 1024 * 1024:
                raise ValueError('Provider response exceeds the bounded tile/metadata limit')
        return bytes(content)


def _cached_request(root, name, url, timeout_s, session=None):
    path = inside(root, name)
    record_path = path.with_name(path.name + '.source.json')
    if path.exists() or record_path.exists():
        if not path.is_file() or not record_path.is_file():
            raise ValueError('Incomplete cached source; preserve it and use a fresh job directory')
        record = json.loads(record_path.read_text(encoding='utf8'))
        if record.get('url') != url or record.get('sha256') != sha256(path):
            raise ValueError('Cached provider source failed URL/hash verification')
        return path.read_bytes(), record
    content = _request(url, timeout_s, session)
    record = dict(url=url, retrieved_at_utc=datetime.now(timezone.utc).isoformat(), sha256=hashlib.sha256(content).hexdigest(), bytes=len(content), file_path=path.relative_to(root).as_posix())
    write_bytes(path, content)
    write_json(record_path, record)
    return content, record


class _RequestPool:
    """Run-scoped worker-local sessions; never share Session state across threads.

    One metadata task or four tile tasks are submitted at a time. No retry loop,
    background process, or cross-run connection pool is introduced. Streams are
    fully consumed/closed by _request so keep-alive connections can be reused.
    """
    def __init__(self, max_workers):
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='streetview-tiles')
        self.local = threading.local()
        self.sessions = []
        self.lock = threading.Lock()

    def _fetch(self, arguments):
        session = getattr(self.local, 'session', None)
        if session is None:
            session = requests.Session()
            # Each Session has exactly one owning worker. Two origin pools
            # accommodate metadata and image hosts, with no automatic retries.
            for scheme in ('http://', 'https://'):
                session.mount(scheme, requests.adapters.HTTPAdapter(pool_connections=2, pool_maxsize=1, max_retries=0, pool_block=True))
            self.local.session = session
            with self.lock:
                self.sessions.append(session)
        return _cached_request(*arguments, session=session)

    def fetch(self, *arguments):
        return self.executor.submit(self._fetch, arguments).result()

    def fetch_face(self, arguments):
        if len(arguments) != 4:
            raise ValueError('Only one four-tile face may be submitted at a time')
        # map returns results in input order, independently of completion order.
        return list(self.executor.map(self._fetch, arguments))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.executor.shutdown(wait=True, cancel_futures=True)
        finally:
            for session in self.sessions:
                session.close()


def assemble_face(tiles):
    """Stitch native pixel arrays without resampling, warping or color edits."""
    if set(tiles) != {(0, 0), (1, 0), (0, 1), (1, 1)}:
        raise ValueError('A cube face requires every native tile exactly once')
    face = np.empty((1024, 1024, 3), dtype=np.uint8)
    for (x, y), content in tiles.items():
        with Image.open(BytesIO(content)) as image:
            image.load()
            if image.size != (512, 512) or image.format not in ('JPEG', 'PNG'):
                raise ValueError('Unexpected native tile dimensions or image format')
            face[y*512:(y+1)*512, x*512:(x+1)*512] = np.asarray(image.convert('RGB'))
    return face


def _validate_selection(config):
    ids = config.get('panorama_ids')
    rows = config.get('panoramas')
    if not isinstance(ids, list) or not ids:
        raise ValueError('Frozen selection requires at least one panorama ID')
    for key in ids:
        valid_id(key)
    if len(set(ids)) != len(ids):
        raise ValueError('Frozen selection requires unique panorama IDs')
    if not isinstance(rows, list):
        raise ValueError('Frozen per-panorama metadata is required')
    metadata = {row.get('pano_id', row.get('id')): row for row in rows}
    if len(metadata) != len(rows) or set(ids) != set(metadata):
        raise ValueError('Frozen metadata and selected panorama IDs differ')
    center, radius = config.get('center'), config.get('radius_m')
    from .selection_limits import MIN_RADIUS_M
    if not isinstance(center, dict) or isinstance(radius, bool) or not isinstance(radius, (int, float)) or not math.isfinite(radius) or radius < MIN_RADIUS_M:
        raise ValueError('Frozen center and radius are required')
    for row in rows:
        if distance_m(center, row) > radius + 1.:
            raise ValueError('Frozen panorama lies outside the selected radius')
    return ids, metadata


def _collect_station(root, key, frozen, timeout, network):
    token = 'pano_' + fingerprint(key)[:20]
    encoded = quote(key, safe='')
    content, provenance = network.fetch(root, f'collection/sources/{token}/metadata.json', f'https://panorama.map.naver.com/metadataV3/basic/{encoded}?lang=ko', timeout)
    raw = json.loads(content)
    if raw.get('id') != key:
        raise ValueError('Provider metadata ID does not match selected panorama')
    station = dict(frozen, id=key, pano_id=key, lat=float(raw['latitude']), lng=float(raw['longitude']), camera_angle=raw.get('camera_angle'), projection=raw.get('proj_type'), provider_altitude=raw.get('altitude'), captured_at=(raw.get('info') or {}).get('photodate'), metadata_source=provenance)
    if distance_m(station, frozen) > .5:
        raise ValueError('Provider capture position changed after selection')
    selected_date = frozen.get('captured_at') or frozen.get('capture_date')
    if selected_date and not str(station['captured_at'] or '').startswith(str(selected_date)):
        raise ValueError('Provider capture date changed after selection')
    if station['projection'] not in ('cubic', 'equirect'):
        raise ValueError('Unknown provider projection; native cube geometry is unverified')
    station['faces'] = {}
    positions = [(x, y) for y in range(2) for x in range(2)]
    for face in FACES:
        arguments = [(root, f'collection/sources/{token}/{face}_{x}_{y}.jpg', tile_url(key, face, x, y), timeout) for x, y in positions]
        results = network.fetch_face(arguments)
        tiles, sources = {}, []
        for (x, y), (data, source) in zip(positions, results):
            tiles[x, y] = data
            sources.append(dict(source, face=face, x=x, y=y))
        pixels = assemble_face(tiles)
        relative = f'collection/images/{token}_{face}.png'
        path = inside(root, relative)
        write_bytes(path, png_bytes(pixels))
        station['faces'][face] = dict(file_path=relative, sha256=sha256(path), w=1024, h=1024, tiles=sources, color_processing='native decoded RGB; lossless PNG; no resizing or image edits')
    return station


def run(config: dict, job_dir: Path, settings: dict) -> dict:
    options = dict(settings.get('collection', settings))
    if options.get('native_source') is not None:
        from .native_collection import run as import_native_source
        return import_native_source(config, job_dir, settings)
    root = Path(job_dir).resolve()
    ids, frozen = _validate_selection(config)
    tolerance = float(options.get('colocation_tolerance_m', .25))
    timeout = float(options.get('timeout_s', 25))
    max_workers = options.get('max_workers', 4)
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or not 1 <= max_workers <= 8:
        raise ValueError('Collection max_workers must be an integer from 1 to 8')
    if not math.isfinite(timeout) or not 1 <= timeout <= 120:
        raise ValueError('Collection timeout must be 1..120 seconds')
    group_physical_stations(list(frozen.values()), tolerance)
    signature = fingerprint(dict(schema_version=1, config=config, native_face_size=1024, colocation_tolerance_m=tolerance))
    manifest_path = inside(root, 'collection/manifest.json')
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding='utf8'))
        if prior.get('input_fingerprint') != signature or prior.get('status') != 'complete':
            raise ValueError('Existing collection belongs to another selection/settings')
        for station in prior['stations']:
            metadata = station['metadata_source']
            if sha256(inside(root, metadata['file_path'])) != metadata['sha256']:
                raise ValueError('Original provider metadata changed after completion')
            for face in FACES:
                item = station['faces'][face]
                if sha256(inside(root, item['file_path'])) != item['sha256']:
                    raise ValueError('Collected image changed after completion')
                for tile in item['tiles']:
                    if sha256(inside(root, tile['file_path'])) != tile['sha256']:
                        raise ValueError('Original provider tile changed after completion')
        return prior
    write_json(inside(root, 'collection/input.json'), dict(input_fingerprint=signature, config=config, colocation_tolerance_m=tolerance))
    stations = []
    # Worker count affects transport only, not the immutable capture signature.
    with _RequestPool(max_workers) as network:
        for key in ids:
            stations.append(_collect_station(root, key, frozen[key], timeout, network))
            print(json.dumps(dict(stage='collect', completed_panoramas=len(stations), total_panoramas=len(ids))), flush=True)
    assignment = group_physical_stations(stations, tolerance)
    for station in stations:
        station['station_id'] = assignment[station['pano_id']]
    manifest = dict(schema_version=1, status='complete', stage='collect', provider='naver', input_fingerprint=signature,
        face_order=list(FACES), native_face_size=1024, tile_size=512, tiles_per_face=[2, 2], source_projection='native_provider_cubemap',
        station_frame='front_camera_opencv_X_right_Y_down_Z_forward', stations=stations,
        physical_station_count=len(set(assignment.values())), panorama_count=len(stations), colocation_tolerance_m=tolerance,
        physical_grouping='complete-link horizontal provider-GPS tolerance; evidence groups only; separate pano_id rigs retain separate camera poses',
        source_images_preserved=True, approximate_gps_not_surveyed=True,
        transport=dict(max_workers=max_workers, maximum_in_flight_tiles_per_face=min(max_workers,4), session_count=len(network.sessions), session_policy='one reusable Session per worker; closed at end of explicit run', retry_policy='no automatic request retries'))
    write_json(manifest_path, manifest)
    return manifest
