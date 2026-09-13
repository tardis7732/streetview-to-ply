"""Explicit local CPU PLY filtering jobs, independent of reconstruction jobs."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import threading
import uuid


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


class PlyFilterManager:
    """All inputs are explicitly supplied local files; GET never starts work."""
    def __init__(self, root, defaults_path=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.defaults_path = Path(defaults_path) if defaults_path else None
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ply-size-filter')
        self._jobs = {}
        self._closed = False
        for path in self.root.glob('*/state.json'):
            try:
                state = json.loads(path.read_text(encoding='utf-8'))
                if (not isinstance(state, dict) or state.get('id') != path.parent.name
                        or not re.fullmatch(r'[0-9a-f]{32}', state['id'])
                        or not isinstance(state.get('created_at'), str)
                        or state.get('status') not in ('queued', 'running', 'completed', 'failed', 'interrupted')):
                    continue
                if state.get('status') in ('queued', 'running'):
                    state.update(status='interrupted', error='서버가 종료되어 정리가 중단됐습니다. 다시 실행해 주세요.', finished_at=_now())
                    _write(path, state)
                self._jobs[state['id']] = state
            except (OSError, ValueError, KeyError):
                continue

    def defaults(self):
        if not self.defaults_path or not self.defaults_path.is_file():
            return {}
        try:
            raw = json.loads(self.defaults_path.read_text(encoding='utf-8'))
            from ..streetview_engine.size_filter import read_size_filter_options
            options = read_size_filter_options({'size_filter': {'enabled': True,
                'max_sigma_camera_radius_ratio': raw.get('max_sigma_camera_radius_ratio', .5)}})
            paths = {key: raw.get(key, '') for key in ('source_ply', 'camera_json')}
            if any(not isinstance(value, str) or not Path(value).is_file() for value in paths.values()):
                return {}
            result = dict(**paths, max_sigma_camera_radius_ratio=options['max_sigma_camera_radius_ratio'])
            if 'size_filter_enabled' in raw:
                if type(raw['size_filter_enabled']) is not bool:
                    return {}
                result['size_filter_enabled'] = raw['size_filter_enabled']
            if 'crop' in raw:
                from ..streetview_engine.ply_cleanup import read_crop_options
                result['crop'] = read_crop_options(raw)
            return result
        except (OSError, ValueError, TypeError, KeyError):
            return {}

    @staticmethod
    def _public(state):
        result = json.loads(json.dumps(state))
        if result.get('status') == 'completed':
            result['download_url'] = f"/api/ply-filters/{result['id']}/download"
        return result

    def get(self, job_id):
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError('정리 작업을 찾을 수 없습니다.')
            return self._public(self._jobs[job_id])

    def list(self):
        with self._lock:
            return [self._public(s) for s in sorted(self._jobs.values(), key=lambda s: s['created_at'], reverse=True)]

    def start(self, payload):
        from ..streetview_engine.size_filter import read_size_filter_options
        from ..streetview_engine.ply_cleanup import read_crop_options
        if not isinstance(payload, dict) or set(payload) - {'source_ply', 'camera_json', 'max_sigma_camera_radius_ratio', 'size_filter_enabled', 'crop'}:
            raise ValueError('정리 설정 형식이 올바르지 않습니다.')
        paths = {}
        for key, extension in [('source_ply', '.ply'), ('camera_json', '.json')]:
            raw = payload.get(key)
            if not isinstance(raw, str) or not raw.strip() or '\x00' in raw:
                raise ValueError('PLY와 카메라 JSON의 로컬 파일 경로를 입력해 주세요.')
            if raw.strip().replace('\\', '/').startswith('//'):
                raise ValueError('네트워크 공유 경로 대신 로컬 파일을 선택해 주세요.')
            path = Path(raw.strip()).expanduser()
            if not path.is_absolute() or path.suffix.lower() != extension or not path.is_file():
                raise ValueError(f'존재하는 {extension} 파일의 절대 경로가 필요합니다.')
            resolved = path.resolve()
            if str(resolved).replace('\\', '/').startswith('//'):
                raise ValueError('네트워크 공유 경로 대신 로컬 파일을 선택해 주세요.')
            paths[key] = str(resolved)
        if Path(paths['source_ply']).stat().st_size > 8 * 1024**3 or Path(paths['camera_json']).stat().st_size > 128 * 1024**2:
            raise ValueError('현재 정리 기능은 PLY 8GB, 카메라 JSON 128MB까지 지원합니다.')
        options = read_size_filter_options({'size_filter': {'enabled': payload.get('size_filter_enabled', True),
            'max_sigma_camera_radius_ratio': payload.get('max_sigma_camera_radius_ratio', .5)}})
        crop = read_crop_options(payload)
        with self._lock:
            if self._closed:
                raise ValueError('서버가 종료 중입니다.')
            if any(s['status'] in ('queued', 'running') for s in self._jobs.values()):
                raise ValueError('다른 PLY 정리가 진행 중입니다. 완료 후 다시 실행해 주세요.')
            job_id = uuid.uuid4().hex
            directory = self.root / job_id
            directory.mkdir()
            state = dict(id=job_id, status='queued', created_at=_now(), **paths, options=options, crop=crop,
                         artifact=None, removed_rows=None, remaining_rows=None, training_started=False)
            self._jobs[job_id] = state
            _write(directory / 'state.json', state)
            self._executor.submit(self._run, job_id)
            return self._public(state)

    def _run(self, job_id):
        directory = self.root / job_id
        with self._lock:
            state = self._jobs[job_id]
            state.update(status='running', started_at=_now())
            _write(directory / 'state.json', state)
        try:
            from ..streetview_engine.ply_cleanup import run_ply_cleanup, verify_cleanup_report, write_output_camera_reference
            output = directory / 'result'
            report = run_ply_cleanup(state['source_ply'], state['camera_json'], output, state['options'], state.get('crop'))
            verify_cleanup_report(report, Path(state['source_ply']), Path(report['artifact']['path']),
                                      output / 'cameras.json', output / 'selection.npz')
            output_camera = write_output_camera_reference(report, output / 'cameras.json', output / 'artifact_cameras.json')
            artifact = dict(report['artifact'])
            target = Path(artifact['path']).resolve()
            if not target.is_relative_to(directory.resolve()) or _sha(target) != artifact['sha256']:
                raise ValueError('정리 결과 파일 검증에 실패했습니다.')
            artifact['path'] = target.relative_to(directory).as_posix()
            with self._lock:
                state.update(status='completed', finished_at=_now(), artifact=artifact,
                             removed_rows=report['removed_rows'], remaining_rows=report['remaining_rows'],
                             source_rows=report['source']['vertex_count'],
                             threshold_sigma_scene_units=report['threshold_sigma_scene_units'],
                             camera_reference=report['camera_reference'],
                             output_camera_json=str(output_camera),
                             crop_geometry=report.get('crop_geometry'),
                             size_removed_rows=report.get('size_removed_rows', report['removed_rows']),
                             crop_additional_removed_rows=report.get('crop_additional_removed_rows', 0),
                             report=dict(path='result/report.json', sha256=_sha(output / 'report.json')),
                             original_preserved=True)
        except Exception as error:
            with self._lock:
                state.update(status='failed', finished_at=_now(), error=str(error), artifact=None)
        finally:
            with self._lock:
                _write(directory / 'state.json', state)

    def close(self):
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=True)
