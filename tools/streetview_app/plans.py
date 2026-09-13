"""Validate selections against provider metadata and save immutable plans."""
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
import re
import uuid

from ..streetview_engine.processing_options import read_processing_options
from ..streetview_engine.size_filter import read_size_filter_options
from ..streetview_engine.selection_limits import read_radius_m
from ..streetview_engine.depth_cleanup_options import read_depth_cleanup_options


def distance_m(lat1, lng1, lat2, lng2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1; dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371008.8 * 2 * math.asin(math.sqrt(min(1., max(0., a))))


def _number(value, low, high, label):
    if isinstance(value, bool):
        raise ValueError(f'{label} 값이 올바르지 않습니다.')
    number = float(value)
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError(f'{label} 범위는 {low}~{high}입니다.')
    return number


def read_selection_options(payload):
    """New selections always use the current multi-view generation pipeline."""
    from .jobs import generation_mode
    if generation_mode(payload) != 'multi_view':
        raise ValueError('Only multi_view generation is supported')
    return read_processing_options(payload)


def selection_settings(payload):
    if not isinstance(payload, dict):
        raise ValueError('선택 설정은 JSON 객체여야 합니다.')
    options = read_selection_options(payload)
    size_filter = read_size_filter_options(payload)
    depth_cleanup = read_depth_cleanup_options(payload)
    from .jobs import generation_mode
    generation = generation_mode(payload)
    center = payload.get('center', {})
    if not isinstance(center, dict):
        raise ValueError('중심 좌표 형식이 올바르지 않습니다.')
    lat = _number(center.get('lat'), -90, 90, '위도')
    lng = _number(center.get('lng'), -180, 180, '경도')
    radius = read_radius_m(payload.get('radius_m'))
    policy = payload.get('capture_policy', {})
    if not isinstance(policy, dict):
        raise ValueError('촬영 조건 형식이 올바르지 않습니다.')
    mode = policy.get('mode', 'same_day'); value = policy.get('value')
    if mode not in ('same_day', 'same_month', 'any'):
        raise ValueError('촬영 시기 선택이 올바르지 않습니다.')
    if mode == 'same_day':
        if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
            raise ValueError('실제 제공된 촬영일을 선택해 주세요.')
        date.fromisoformat(value)
    elif mode == 'same_month':
        if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}', value):
            raise ValueError('실제 제공된 촬영월을 선택해 주세요.')
        date.fromisoformat(value + '-01')
    else:
        value = None
    time_start, time_end = policy.get('time_start') or None, policy.get('time_end') or None
    for item in (time_start, time_end):
        if item is not None and (not isinstance(item, str) or not re.fullmatch(r'(?:[01]\d|2[0-3]):[0-5]\d', item)):
            raise ValueError('촬영 시각은 HH:MM 형식입니다.')
    if time_start and time_end and time_start > time_end:
        raise ValueError('종료 시각은 시작 시각보다 늦어야 합니다.')
    ids = payload.get('panorama_ids', [])
    if not isinstance(ids, list) or not ids or any(not isinstance(v, str) or not 1 <= len(v) <= 2048 or '\x00' in v for v in ids):
        raise ValueError('학습할 거리뷰 지점을 1개 이상 선택해 주세요.')
    if len(set(ids)) != len(ids):
        raise ValueError('선택한 지점에 중복 ID가 있습니다.')
    excluded = payload.get('excluded_panorama_ids', [])
    if not isinstance(excluded, list) or any(not isinstance(v, str) or not 1 <= len(v) <= 2048 or '\x00' in v for v in excluded):
        raise ValueError('제외한 지점 ID 목록은 유효한 문자열이어야 합니다.')
    if len(set(excluded)) != len(excluded):
        raise ValueError('제외한 지점에 중복 ID가 있습니다.')
    if set(ids) & set(excluded):
        raise ValueError('학습할 지점과 제외한 지점이 겹칩니다.')
    settings = dict(center=dict(lat=lat, lng=lng), radius_m=radius,
        capture_policy=dict(mode=mode, value=value, time_start=time_start, time_end=time_end),
        panorama_ids=list(ids), excluded_panorama_ids=list(excluded))
    if 'processing_options' in payload:
        settings['processing_options'] = options
    if 'size_filter' in payload:
        settings['size_filter'] = size_filter
    if depth_cleanup is not None:
        settings['depth_cleanup'] = depth_cleanup
    if 'generation_mode' in payload:
        settings['generation_mode'] = generation
    for key, lower, upper in [('training_steps', 1, 1000000), ('resolution', 64, 4096), ('max_splats', 4, 10000000)]:
        if key in payload:
            value = payload[key]
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f'{key} 값은 {lower}~{upper} 범위의 정수여야 합니다.')
            settings[key] = value
    return settings


def _capture_parts(panorama):
    raw = str(panorama.get('captured_at') or panorama.get('capture_date') or '')
    precision = str(panorama.get('capture_precision') or 'unknown')
    match = re.match(r'^(\d{4}-\d{2})(?:-(\d{2}))?(?:[T ](\d{2}:\d{2})(?::\d{2})?)?', raw)
    if match is None or precision == 'unknown':
        return None, None, None
    month = match[1]
    day = f'{month}-{match[2]}' if match[2] and precision not in ('year', 'month') else None
    clock = match[3] if precision not in ('year', 'month', 'day', 'date') else None
    return month, day, clock


def freeze_selection(payload, provider):
    """Verify input IDs only; explicit exclusions are saved provenance.

    Excluded IDs are neither collected nor assigned a training/evaluation split.
    Their metadata is deliberately not requested or represented as verified.
    """
    settings = selection_settings(payload)
    points = []
    for pano_id in settings['panorama_ids']:
        # The client cannot supply its own coordinates/dates to bypass filters.
        panorama = dict(provider.get_panorama(pano_id))
        if panorama.get('id') != pano_id:
            raise ValueError('요청한 거리뷰와 응답 ID가 다릅니다.')
        lat = _number(panorama['lat'], -90, 90, '거리뷰 위도')
        lng = _number(panorama.get('lng', panorama.get('lon')), -180, 180, '거리뷰 경도')
        d = distance_m(settings['center']['lat'], settings['center']['lng'], lat, lng)
        if not math.isfinite(d) or d > settings['radius_m'] + 1.:
            raise ValueError('선택한 지점 중 지정 반경 밖의 자료가 있습니다. 다시 검색해 주세요.')
        month, day, clock = _capture_parts(panorama)
        policy = settings['capture_policy']
        if policy['mode'] == 'same_day' and day != policy['value']:
            raise ValueError('촬영일이 다른 지점이 포함돼 있습니다.')
        if policy['mode'] == 'same_month' and month != policy['value']:
            raise ValueError('촬영월이 다른 지점이 포함돼 있습니다.')
        if policy['time_start'] or policy['time_end']:
            if clock is None:
                raise ValueError('정확한 촬영 시각이 없는 지점은 시간 필터에 포함할 수 없습니다.')
            if (policy['time_start'] and clock < policy['time_start']) or (policy['time_end'] and clock > policy['time_end']):
                raise ValueError('선택한 촬영 시각 범위 밖의 지점이 있습니다.')
        points.append({key: panorama.get(key) for key in ('id', 'lat', 'lng', 'captured_at', 'capture_date', 'capture_precision', 'heading', 'camera_angle', 'provider_altitude', 'projection', 'links', 'title')})
    return dict(schema_version=1, provider='naver', **settings, panoramas=points,
        timestamp_policy='Provider capture text is retained; no timezone or capture time is invented.',
        selection_verified=True)


class PlanStore:
    def __init__(self, root):
        self.root = Path(root).resolve(); self.root.mkdir(parents=True, exist_ok=True)

    def save(self, frozen):
        plan_id = uuid.uuid4().hex
        plan = dict(frozen, id=plan_id, created_at=datetime.now(timezone.utc).isoformat(), status='saved_plan')
        path = self.root / (plan_id + '.json')
        with path.open('x', encoding='utf8') as stream:
            json.dump(plan, stream, indent=2, ensure_ascii=False, allow_nan=False)
        return plan

    def get(self, plan_id):
        if not re.fullmatch(r'[0-9a-f]{32}', plan_id):
            raise KeyError('저장된 설정이 없습니다.')
        path = self.root / (plan_id + '.json')
        if not path.is_file():
            raise KeyError('저장된 설정이 없습니다.')
        return json.loads(path.read_text(encoding='utf8'))

    def list(self):
        result = []
        for path in self.root.glob('*.json'):
            try:
                plan = json.loads(path.read_text(encoding='utf8'))
                row = {key: plan.get(key) for key in ('id', 'created_at', 'center', 'radius_m', 'capture_policy', 'panorama_ids', 'excluded_panorama_ids', 'status')}
                if 'processing_options' in plan:
                    row['processing_options'] = plan['processing_options']
                if 'size_filter' in plan:
                    row['size_filter'] = plan['size_filter']
                if 'depth_cleanup' in plan:
                    row['depth_cleanup'] = plan['depth_cleanup']
                if 'generation_mode' in plan:
                    row['generation_mode'] = plan['generation_mode']
                result.append(row)
            except (OSError, ValueError):
                continue
        return sorted(result, key=lambda row: row['created_at'] or '', reverse=True)
