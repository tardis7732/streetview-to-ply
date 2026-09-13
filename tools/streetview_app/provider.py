"""Resumable Naver metadata discovery and cached low-resolution cube previews.

Provider capture text is retained without assigning a timezone. These public
web endpoints are an adapter, not a promise of a stable official bulk API.
No full-resolution training tiles or GPU jobs are downloaded by this module.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import date as calendar_date, datetime
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
import time
from urllib.parse import quote
import uuid

import requests
from PIL import Image

from .plans import distance_m


HEADERS = {'Referer': 'https://map.naver.com/', 'User-Agent': 'Mozilla/5.0'}
FACES = dict(zip(('F', 'R', 'B', 'L', 'U', 'D'), (1, 2, 3, 0, 5, 4)))


def valid_id(value):
    # Opaque provider identifiers may contain base64 characters, including '/'.
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_+/=-]{1,240}', value):
        raise ValueError('거리뷰 ID가 올바르지 않습니다.')
    return value


def capture_fields(value):
    raw = str(value or '').strip()
    precision = 'unknown'; day = None
    try:
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?', raw):
            datetime.fromisoformat(raw)
            precision, day = 'second', raw[:10]
        elif re.fullmatch(r'\d{4}-\d{2}-\d{2}', raw):
            calendar_date.fromisoformat(raw)
            precision, day = 'day', raw
        elif re.fullmatch(r'\d{4}-\d{2}', raw):
            calendar_date.fromisoformat(raw + '-01')
            precision, day = 'month', raw
    except ValueError:
        pass
    return dict(captured_at=raw or None, capture_date=day, capture_precision=precision)


class NaverProvider:
    def __init__(self, cache_dir, *, timeout_s=12, cache_ttl_s=3600, request_limit=200,
                 discovery_ttl_s=3600):
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s; self.cache_ttl_s = cache_ttl_s
        # This bounds one HTTP request, never the total number of selected or
        # discovered panoramas. The persistent frontier has no node-count cap.
        self.request_limit = max(1, int(request_limit))
        self.discovery_ttl_s = max(1, float(discovery_ttl_s))
        self._discovery_dir = self.cache_dir / 'discovery'
        self._discovery_dir.mkdir(exist_ok=True)
        self._discovery_lock = threading.Lock()
        self._active_discovery = {}
        self._preview_lock = threading.Lock()

    def _cache_path(self, key, suffix):
        return self.cache_dir / (hashlib.sha256(key.encode('utf8')).hexdigest() + suffix)

    def _write(self, path, data):
        temporary = path.with_suffix('.' + uuid.uuid4().hex + '.tmp')
        try:
            temporary.write_bytes(data)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def _json(self, url):
        path = self._cache_path(url, '.json')
        if path.is_file() and time.time() - path.stat().st_mtime < self.cache_ttl_s:
            try:
                return json.loads(path.read_bytes())
            except (ValueError, OSError):
                pass
        response = requests.get(url, headers=HEADERS, timeout=(4, self.timeout_s))
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or payload.get('error'):
            raise LookupError('네이버에서 해당 거리뷰 자료를 제공하지 않았습니다.')
        self._write(path, json.dumps(payload, ensure_ascii=False).encode('utf8'))
        return payload

    @staticmethod
    def _coordinates(lat, lng):
        lat, lng = float(lat), float(lng)
        if not math.isfinite(lat) or not math.isfinite(lng) or not (-90 <= lat <= 90 and -180 <= lng <= 180):
            raise ValueError('좌표가 올바르지 않습니다.')
        return lat, lng

    def _normalise(self, raw):
        pano_id = valid_id(raw['id'])
        lat, lng = self._coordinates(raw['latitude'], raw['longitude'])
        info = raw.get('info') or {}
        links = []
        for link in raw.get('links') or []:
            try:
                link_lat, link_lng = self._coordinates(link['latitude'], link['longitude'])
                links.append(dict(id=valid_id(link['id']), lat=link_lat, lng=link_lng))
            except (KeyError, ValueError, TypeError):
                continue
        angles = raw.get('camera_angle') or [0, 0, 0]
        try:
            # This conversion is also used by the installed streetlevel adapter.
            from streetlevel.naver.parse import _convert_pano_rotation
            heading = math.degrees(_convert_pano_rotation(angles)[0]) % 360
        except (ImportError, ValueError, TypeError, IndexError):
            heading = None
        return dict(id=pano_id, lat=lat, lng=lng,
            **capture_fields(info.get('photodate')), heading=heading,
            title=str(info.get('title') or info.get('description') or '거리뷰'),
            description=str(info.get('description') or ''), links=links,
            timeline_id=info.get('timeline_id'), projection=raw.get('proj_type'),
            camera_angle=raw.get('camera_angle'), provider_altitude=raw.get('altitude'),
            preview_projection='cubemap', preview_face_size=256,
            faces={face: f'/api/cube/{quote(pano_id, safe="")}/{face}' for face in FACES})

    def get_panorama(self, pano_id):
        encoded = quote(valid_id(pano_id), safe='')
        raw = self._json(f'https://panorama.map.naver.com/metadataV3/basic/{encoded}?lang=ko')
        result = self._normalise(raw)
        if result['id'] != pano_id:
            raise ValueError('요청한 거리뷰와 응답 ID가 다릅니다.')
        return result

    def _history(self, anchor):
        timeline_id = anchor.get('timeline_id') or anchor['id']
        encoded = quote(valid_id(timeline_id), safe='')
        raw = self._json(f'https://panorama.map.naver.com/metadata/timeline/{encoded}')
        result = []
        for row in (raw.get('timeline') or {}).get('panoramas', []):
            try:
                pano_id = valid_id(row[0])
                lat, lng = self._coordinates(row[2], row[1])
                result.append(dict(id=pano_id, lat=lat, lng=lng, **capture_fields(row[4]),
                    heading=None, title=anchor['title'], description='', links=[],
                    faces={face: f'/api/cube/{quote(pano_id, safe="")}/{face}' for face in FACES}))
            except (IndexError, ValueError, TypeError, KeyError):
                continue
        return result

    def _discovery_session(self, scope, continuation):
        """Only an opaque, server-issued token can address a saved search.

        SQLite keeps the frontier, visited IDs, and replayable pages on disk,
        rather than retaining an unbounded Python graph between HTTP requests.
        Expired searches are deleted when a new search starts. Live searches
        are never evicted to make room for another search.
        """
        with self._discovery_lock:
            if continuation is not None:
                if not isinstance(continuation, str) or not re.fullmatch(r'[0-9a-f]{64}', continuation):
                    raise ValueError('거리뷰 이어보기 정보가 올바르지 않습니다. 다시 검색해 주세요.')
                path = self._discovery_dir / (continuation[:32] + '.sqlite3')
                if not path.is_file() or time.time() - path.stat().st_mtime >= self.discovery_ttl_s:
                    raise ValueError('거리뷰 검색이 만료되었습니다. 다시 검색해 주세요.')
                self._active_discovery[path] = self._active_discovery.get(path, 0) + 1
                return path, continuation[32:], False
            for path in self._discovery_dir.glob('*.sqlite3'):
                if path not in self._active_discovery and time.time() - path.stat().st_mtime >= self.discovery_ttl_s:
                    try:
                        path.unlink()
                    except OSError:
                        pass
            path = self._discovery_dir / (uuid.uuid4().hex + '.sqlite3')
            page_key = uuid.uuid4().hex
            with closing(sqlite3.connect(path)) as db:
                db.executescript("""
                    CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE nodes (id TEXT PRIMARY KEY, payload TEXT,
                        capture_date TEXT, status INTEGER NOT NULL);
                    CREATE INDEX pending_nodes ON nodes(status);
                    CREATE TABLE pages (token TEXT PRIMARY KEY, response TEXT);
                """)
                db.execute('INSERT INTO meta VALUES (?, ?)', ('scope', json.dumps(scope)))
                db.execute('INSERT INTO pages VALUES (?, NULL)', (page_key,))
                db.commit()
            self._active_discovery[path] = 1
            return path, page_key, True

    def discover(self, lat, lng, radius_m, *, max_nodes=200, date=None, continuation=None):
        lat, lng = self._coordinates(lat, lng)
        from ..streetview_engine.selection_limits import read_radius_m
        radius_m = read_radius_m(radius_m)
        if not isinstance(max_nodes, int) or isinstance(max_nodes, bool) or max_nodes < 1:
            raise ValueError('한 번에 불러올 거리뷰 수는 1 이상이어야 합니다.')
        if date is not None:
            if not isinstance(date, str) or not re.fullmatch(r'\d{4}-\d{2}(?:-\d{2})?', date):
                raise ValueError('촬영일은 YYYY-MM-DD 또는 YYYY-MM 형식입니다.')
            calendar_date.fromisoformat(date if len(date) == 10 else date + '-01')
        scope = dict(lat=lat, lng=lng, radius_m=radius_m, date=date)
        path, page_key, fresh = self._discovery_session(scope, continuation)
        limit = min(max_nodes, self.request_limit)
        radius = lambda p: distance_m(lat, lng, p['lat'], p['lng']) <= radius_m
        matches = lambda p: date is None or (p.get('capture_date') or '').startswith(date)
        warnings = []
        result = dict(center=dict(lat=lat, lng=lng), radius_m=radius_m,
            panoramas=[], date_options=[], truncated=False, continuation=None,
            warnings=warnings, coverage='bounded_connected_search', nodes_examined=0,
            nodes_examined_total=0, page_size=limit)
        deadline = time.monotonic() + 45
        db = None
        completed = False
        try:
            db = sqlite3.connect(path, timeout=0)
            # A second request for the same page must not advance the graph
            # again. Concurrent callers get a retryable error; later retries
            # receive the exact page committed by the first caller.
            try:
                db.execute('BEGIN IMMEDIATE')
            except sqlite3.OperationalError as error:
                raise ValueError('거리뷰를 불러오는 중입니다. 잠시 후 다시 시도해 주세요.') from error
            stored = db.execute('SELECT value FROM meta WHERE key=?', ('scope',)).fetchone()
            if stored is None or json.loads(stored[0]) != scope:
                raise ValueError('검색 위치·반경·촬영일이 바뀌었습니다. 다시 검색해 주세요.')
            page = db.execute('SELECT response FROM pages WHERE token=?', (page_key,)).fetchone()
            if page is None:
                raise ValueError('거리뷰 이어보기 정보가 올바르지 않습니다. 다시 검색해 주세요.')
            if page[0] is not None:
                result = json.loads(page[0])
                db.commit()
                path.touch()
                completed = True
                return result

            def catalogue(p):
                if radius(p):
                    db.execute('INSERT OR IGNORE INTO nodes VALUES (?, NULL, ?, 2)',
                        (p['id'], p.get('capture_date')))

            def enqueue(p):
                payload = json.dumps(p, ensure_ascii=False)
                db.execute('INSERT OR IGNORE INTO nodes VALUES (?, ?, ?, 0)',
                    (p['id'], payload, p.get('capture_date') if radius(p) else None))
                db.execute('UPDATE nodes SET status=0, payload=? WHERE id=? AND status=2',
                    (payload, p['id']))

            if fresh:
                nearest = self._json(f'https://map.naver.com/p/api/panorama/nearby/{lng}/{lat}')
                features = nearest.get('features') or []
                if features:
                    anchor = self.get_panorama(features[0]['properties']['id'])
                    catalogue(anchor)
                    try:
                        history = self._history(anchor)
                    except (LookupError, requests.RequestException, ValueError):
                        history = []
                        warnings.append('일부 과거 촬영 목록을 가져오지 못했습니다.')
                    for p in history:
                        catalogue(p)
                    # Historical captures are initially date choices. A date
                    # query queues every matching known version, including
                    # disconnected versions, so none are lost between pages.
                    alternatives = sorted((p for p in history if date and matches(p) and radius(p)),
                        key=lambda p: distance_m(lat, lng, p['lat'], p['lng']))
                    if date and not matches(anchor):
                        if alternatives:
                            anchor = self.get_panorama(alternatives[0]['id'])
                        else:
                            warnings.append('중심 지점에 해당 촬영일이 없어 주변 연결 지점에서 확인합니다.')
                    enqueue(anchor)
                    for p in alternatives:
                        enqueue(p)
                    try:
                        around = self._json(f'https://panorama.map.naver.com/metadataV3/around/{quote(anchor["id"], safe="")}?lang=ko')
                        for raw in (around.get('panoramas') or {}).get('street', []):
                            try:
                                p_lat, p_lng = self._coordinates(raw['latitude'], raw['longitude'])
                                p = dict(id=valid_id(raw['id']), lat=p_lat, lng=p_lng)
                                if radius(p):
                                    enqueue(p)
                            except (KeyError, ValueError, TypeError):
                                continue
                    except (LookupError, requests.RequestException, KeyError, ValueError, TypeError):
                        warnings.append('보조 주변 목록을 가져오지 못해 연결 경로만 탐색했습니다.')
                else:
                    warnings.append('이 지점 주변에서 거리뷰를 찾지 못했습니다.')
                db.execute('INSERT INTO meta VALUES (?, ?)', ('warnings', json.dumps(warnings, ensure_ascii=False)))
            else:
                saved_warnings = db.execute('SELECT value FROM meta WHERE key=?', ('warnings',)).fetchone()
                if saved_warnings:
                    warnings.extend(json.loads(saved_warnings[0]))

            def retrieve(p):
                if 'capture_precision' in p and 'projection' in p:
                    return p
                try:
                    return self.get_panorama(p['id'])
                except (LookupError, requests.RequestException, KeyError, ValueError, TypeError):
                    return None

            failures = 0
            with ThreadPoolExecutor(max_workers=6) as pool:
                while result['nodes_examined'] < limit and time.monotonic() < deadline:
                    rows = db.execute('SELECT id, payload FROM nodes WHERE status=0 ORDER BY rowid LIMIT ?',
                        (min(6, limit - result['nodes_examined']),)).fetchall()
                    if not rows:
                        break
                    for (pano_id, _), p in zip(rows, pool.map(retrieve, (json.loads(row[1]) for row in rows))):
                        result['nodes_examined'] += 1
                        db.execute('UPDATE nodes SET status=1, payload=NULL WHERE id=?', (pano_id,))
                        if p is None:
                            failures += 1
                            continue
                        if radius(p):
                            db.execute('UPDATE nodes SET capture_date=? WHERE id=?', (p.get('capture_date'), pano_id))
                            if matches(p):
                                result['panoramas'].append(p)
                        for link in p.get('links', []):
                            if radius(link):
                                enqueue(link)
            result['panoramas'].sort(key=lambda p: (distance_m(lat, lng, p['lat'], p['lng']), p['id']))
            result['nodes_examined_total'] = db.execute('SELECT COUNT(*) FROM nodes WHERE status=1').fetchone()[0]
            previous_failures = db.execute('SELECT value FROM meta WHERE key=?', ('metadata_failures',)).fetchone()
            failure_total = failures + (int(previous_failures[0]) if previous_failures else 0)
            result['metadata_failures_total'] = failure_total
            db.execute('INSERT OR REPLACE INTO meta VALUES (?, ?)', ('metadata_failures', str(failure_total)))
            counts = db.execute('SELECT capture_date, COUNT(*) FROM nodes WHERE capture_date IS NOT NULL GROUP BY capture_date ORDER BY capture_date DESC')
            result['date_options'] = [dict(value=d, label=d, count=count,
                precision='day' if len(d) == 10 else 'month') for d, count in counts]
            pending = db.execute('SELECT 1 FROM nodes WHERE status=0 LIMIT 1').fetchone() is not None
            if pending:
                next_key = uuid.uuid4().hex
                db.execute('INSERT INTO pages VALUES (?, NULL)', (next_key,))
                result['continuation'] = path.stem + next_key
                result['truncated'] = True
            if failure_total:
                warnings.append(f'총 {failure_total}개 지점의 정보를 가져오지 못했습니다. 빠진 지점을 다시 확인하려면 다시 검색해 주세요.')
            if not result['panoramas']:
                warnings.append('이번 조회에서 선택 조건에 맞는 지점을 찾지 못했습니다.')
            db.execute('UPDATE pages SET response=? WHERE token=?',
                (json.dumps(result, ensure_ascii=False), page_key))
            db.commit()
            path.touch()
            completed = True
            return result
        finally:
            if db is not None:
                db.close()
            with self._discovery_lock:
                self._active_discovery[path] -= 1
                if self._active_discovery[path] == 0:
                    del self._active_discovery[path]
                    if fresh and not completed:
                        path.unlink(missing_ok=True)

    def cube_face(self, pano_id, face):
        valid_id(pano_id); face = str(face).upper()
        if face not in FACES:
            raise ValueError('지원하지 않는 큐브 면입니다.')
        destination = self._cache_path(pano_id + ':' + face, '.jpg')
        with self._preview_lock:
            if not destination.is_file():
                self.get_panorama(pano_id)
                # Naver also supplies this native cube preview for ERP source
                # panoramas. Validate the actual preview, not the source tag.
                url = f'https://panorama.pstatic.net/image/{quote(pano_id, safe="")}/512/P'
                response = requests.get(url, headers=HEADERS, timeout=(4, self.timeout_s))
                response.raise_for_status()
                with Image.open(BytesIO(response.content)) as source:
                    if source.size != (1536, 256):
                        raise ValueError('거리뷰 미리보기 형식이 변경되었습니다.')
                    source = source.convert('RGB')
                    for label, index in FACES.items():
                        buffer = BytesIO()
                        source.crop((index * 256, 0, (index + 1) * 256, 256)).save(buffer, 'JPEG', quality=94)
                        self._write(self._cache_path(pano_id + ':' + label, '.jpg'), buffer.getvalue())
            return destination.read_bytes(), 'image/jpeg'
