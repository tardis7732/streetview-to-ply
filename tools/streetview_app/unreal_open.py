"""Explicit, hash-bound Unreal previews of completed local Gaussian artifacts.

HTTP callers must resolve a completed artifact first. Executable/project settings
come only from an operator-owned profile; no user-supplied commands are accepted.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid

import numpy as np


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def _local_file(raw, suffix):
    if not isinstance(raw, str) or '\x00' in raw or raw.replace('\\', '/').startswith('//'):
        raise ValueError('기존 로컬 파일의 절대 경로가 필요합니다.')
    path = Path(raw)
    if not path.is_absolute() or path.suffix.lower() != suffix or not path.is_file():
        raise ValueError('기존 로컬 파일의 절대 경로가 필요합니다.')
    if str(path.resolve()).replace('\\', '/').startswith('//'):
        raise ValueError('네트워크 공유 경로는 지원하지 않습니다.')
    return path


def _profile(path):
    raw = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    required = {'executable', 'project', 'map_root'}
    optional = {'minimum_free_gpu_mb', 'max_active_previews', 'timeout_seconds', 'camera_limit'}
    if not isinstance(raw, dict) or not required <= raw.keys() or raw.keys() - required - optional:
        raise ValueError('언리얼 실행 프로필 형식이 올바르지 않습니다.')
    exe, project = _local_file(raw['executable'], '.exe'), _local_file(raw['project'], '.uproject')
    if exe.name.lower() != 'unrealeditor.exe':
        raise ValueError('프로필 실행 파일은 UnrealEditor.exe여야 합니다.')
    if not str(project).isascii():
        raise ValueError('MLSLabsRenderer 파일 입출력에는 ASCII 프로젝트 경로(또는 등록된 정션)가 필요합니다.')
    if not re.fullmatch(r'/Game/[A-Za-z0-9_]+(?:/[A-Za-z0-9_]+)*', raw['map_root']):
        raise ValueError('전용 비교 맵의 /Game 패키지 경로가 필요합니다.')
    project_data = json.loads(project.read_text(encoding='utf-8-sig'))
    enabled = {p['Name'] for p in project_data.get('Plugins', []) if p.get('Enabled') is True}
    if not {'MLSLabsRenderer', 'PythonScriptPlugin', 'EditorScriptingUtilities'} <= enabled:
        raise ValueError('프로젝트에 MLSLabsRenderer, PythonScriptPlugin, EditorScriptingUtilities가 필요합니다.')
    result = dict(raw, executable=str(exe), project=str(project))
    for key, default, low, high in [('minimum_free_gpu_mb', 2048, 256, 65536),
                                   ('max_active_previews', 1, 1, 4),
                                   ('timeout_seconds', 360, 30, 1800), ('camera_limit', 12, 1, 100)]:
        value = raw.get(key, default)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f'잘못된 언리얼 프로필 값: {key}')
        result[key] = value
    return result


def _gpu_free_mb():
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=10, check=True, creationflags=flags)
    # Unreal's adapter selection is deliberately not guessed on multi-GPU hosts.
    values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    if len(values) != 1:
        raise ValueError('현재 자동 미리보기는 NVIDIA GPU 한 개가 있는 PC에서 지원합니다.')
    return values[0]


def _pid_alive(pid):
    if type(pid) is not int or pid <= 0:
        return False
    if os.name == 'nt':
        import ctypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(ctypes.c_void_p(handle))
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def prepare_artifact(resolved, camera_limit=12):
    """Validate a descriptor produced by the server's completed-artifact resolver."""
    from ..streetview_engine.size_filter import _ply
    allowed = {'source_ply', 'camera_json', 'source_sha256', 'camera_sha256', 'coordinate_frame',
               'units', 'camera_convention', 'label', 'source_ref'}
    if not isinstance(resolved, dict) or resolved.keys() - allowed:
        raise ValueError('완료된 PLY와 검증된 카메라 참조가 필요합니다.')
    if resolved.get('coordinate_frame') != 'EDN' or resolved.get('units') not in ('metres', 'meters', 'm'):
        raise ValueError('현재 언리얼 자동 열기는 EDN 좌표·미터 단위만 지원합니다.')
    convention = resolved.get('camera_convention')
    if convention not in ('OpenGL_c2w', 'OpenCV_c2w'):
        raise ValueError('카메라의 c2w 좌표 규약이 필요합니다.')
    source = _local_file(resolved.get('source_ply'), '.ply').resolve()
    camera_path = _local_file(resolved.get('camera_json'), '.json').resolve()
    if source.stat().st_size > 8 * 1024**3 or camera_path.stat().st_size > 128 * 1024**2:
        raise ValueError('미리보기 파일 크기 제한을 초과했습니다.')
    for path, key in [(source, 'source_sha256'), (camera_path, 'camera_sha256')]:
        if not re.fullmatch('[0-9a-f]{64}', str(resolved.get(key, ''))) or _sha(path) != resolved[key]:
            raise ValueError('완료 후 PLY 또는 카메라 파일이 변경됐습니다.')
    info, header, _ = _ply(source)
    declaration = re.search(rb'^comment coordinates ([A-Za-z0-9_]+) (metres|meters|m)(?:;|\s|$)', header, re.M)
    if declaration and declaration.group(1) != b'EDN':
        raise ValueError('PLY 좌표 선언이 등록된 EDN 좌표와 다릅니다.')
    document = json.loads(camera_path.read_text(encoding='utf-8-sig'))
    if not isinstance(document, dict) or not isinstance(document.get('frames'), list) or not document['frames']:
        raise ValueError('카메라 프레임이 없습니다.')
    for key, expected in [('coordinate_frame', 'EDN'), ('camera_convention', convention)]:
        if key in document and document[key] != expected:
            raise ValueError('카메라 좌표 선언이 등록된 정보와 다릅니다.')
    if 'units' in document and document['units'] not in ('m', 'meters', 'metres'):
        raise ValueError('카메라 단위가 미터가 아닙니다.')
    frames = []
    for i, frame in enumerate(document['frames']):
        matrix = np.asarray(frame.get('transform_matrix'), dtype=float)
        if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
                or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5, rtol=0)
                or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5, rtol=0)):
            raise ValueError('카메라 변환 행렬이 올바르지 않습니다.')
        if convention == 'OpenCV_c2w':
            matrix[:3, 1:3] *= -1
        values = {}
        for key in ('w', 'h', 'fl_x', 'fl_y'):
            value = frame.get(key, document.get(key))
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError('양수의 카메라 해상도·초점거리가 필요합니다.')
            values[key] = float(value)
        cx, cy = frame.get('cx', document.get('cx', values['w']/2)), frame.get('cy', document.get('cy', values['h']/2))
        if (not np.isclose(values['fl_x'], values['fl_y'], rtol=1e-5)
                or not np.isclose(cx, values['w']/2, rtol=0, atol=.01)
                or not np.isclose(cy, values['h']/2, rtol=0, atol=.01)
                or any(abs(float(frame.get(k, document.get(k, 0)))) > 1e-8 for k in ('k1', 'k2', 'k3', 'k4', 'p1', 'p2'))):
            raise ValueError('언리얼 미리보기에는 왜곡 보정된 중앙 주점의 원근 카메라가 필요합니다.')
        fov = math.degrees(2 * math.atan(values['w'] / (2 * values['fl_x'])))
        if not 5 <= fov < 170:
            raise ValueError('지원 범위를 벗어난 카메라 시야각입니다.')
        frames.append(dict(index=i, transform_matrix=matrix.tolist(), **values,
                           station_id=str(frame.get('station_id', i)), face=str(frame.get('face', ''))))
    # Equal station weight selects a central physical viewpoint; faces do not bias its center.
    stations = {}
    for frame in frames:
        stations.setdefault(frame['station_id'], []).append(frame)
    centers = {station: np.mean([np.asarray(f['transform_matrix'])[:3, 3] for f in rows], axis=0)
               for station, rows in stations.items()}
    center = np.mean(list(centers.values()), axis=0)
    station = min(centers, key=lambda key: (float(np.linalg.norm(centers[key]-center)), key))
    preferred = min(stations[station], key=lambda f: (f['face'] not in ('F', 'front'),
                    abs(f['transform_matrix'][1][2]), f['index']))
    selected = [preferred]
    for i in np.linspace(0, len(frames)-1, min(camera_limit, len(frames)), dtype=int):
        if frames[i]['index'] != preferred['index'] and len(selected) < camera_limit:
            selected.append(frames[i])
    for i, frame in enumerate(selected):
        frame['label'] = f'ReviewCamera_{i+1:02d}'
    label = re.sub('[^A-Za-z0-9_]+', '_', str(resolved.get('label', 'Streetview')))[:48].strip('_') or 'Streetview'
    return dict(source_ply=str(source), camera_json=str(camera_path), source_sha256=info['sha256'],
                camera_sha256=resolved['camera_sha256'], vertex_count=info['vertex_count'], coordinate_frame='EDN',
                units='metres', camera_convention='OpenGL_c2w', label=label, cameras=selected,
                source_ref=resolved.get('source_ref'))


class UnrealOpenManager:
    def __init__(self, root, profile_path, *, launcher=None, gpu_probe=None):
        self.root, self.profile_path = Path(root).resolve(), Path(profile_path)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock, self._stop = threading.RLock(), threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='unreal-preview')
        self._launcher, self._gpu_probe = launcher or subprocess.Popen, gpu_probe or _gpu_free_mb
        self._jobs, self._processes = {}, {}
        for file in self.root.glob('*/state.json'):
            try:
                state = json.loads(file.read_text(encoding='utf-8'))
                if not re.fullmatch('[0-9a-f]{32}', file.parent.name) or state['id'] != file.parent.name:
                    continue
                if state['status'] in ('queued', 'running'):
                    state.update(status='interrupted', error='서버가 종료되어 열기 결과를 확인하지 못했습니다.')
                    _write(file, state)
                self._jobs[state['id']] = state
            except (ValueError, KeyError, OSError, TypeError):
                continue

    def capability(self):
        try:
            if sys.platform != 'win32':
                raise ValueError('현재 언리얼 자동 열기는 Windows에서 지원합니다.')
            profile = _profile(self.profile_path)
            return dict(available=True, renderer='MLSLabsRenderer', coordinate_frame='EDN', units='metres',
                        creates_new_level=True, minimum_free_gpu_mb=profile['minimum_free_gpu_mb'])
        except (OSError, ValueError, TypeError, KeyError) as error:
            return dict(available=False, reason=str(error))

    def get(self, job_id):
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError('언리얼 열기 작업을 찾을 수 없습니다.')
            result = json.loads(json.dumps(self._jobs[job_id]))
            result['editor_running'] = _pid_alive(result.get('pid'))
            return result

    def list(self):
        with self._lock:
            return [self.get(key) for key in sorted(self._jobs, key=lambda k: self._jobs[k]['created_at'], reverse=True)]

    def start(self, resolved):
        if not self.capability()['available']:
            raise ValueError(self.capability()['reason'])
        profile = _profile(self.profile_path)
        artifact = prepare_artifact(resolved, profile['camera_limit'])
        with self._lock:
            if self._stop.is_set():
                raise ValueError('서버가 종료 중입니다.')
            active = [s for s in self._jobs.values() if s['status'] in ('queued', 'running') or _pid_alive(s.get('pid'))]
            if len(active) >= profile['max_active_previews']:
                raise ValueError('이 도구에서 연 언리얼 창을 닫은 뒤 다시 열어 주세요.')
            job_id = uuid.uuid4().hex
            directory = self.root / job_id
            directory.mkdir()
            state = dict(id=job_id, status='queued', created_at=_now(), source_ref=artifact['source_ref'],
                         label=artifact['label'], source_sha256=artifact['source_sha256'], pid=None,
                         map=profile['map_root'] + '/L_' + artifact['label'] + '_' + job_id[:12])
            self._jobs[job_id] = state
            _write(directory/'state.json', state)
            self._executor.submit(self._run, job_id, profile, artifact)
            return self.get(job_id)

    def _run(self, job_id, profile, artifact):
        directory = self.root / job_id
        state = self._jobs[job_id]
        try:
            free = self._gpu_probe()
            if free < profile['minimum_free_gpu_mb']:
                raise ValueError(f"GPU 여유 메모리가 부족합니다 ({free} MB). 기존 언리얼 창을 닫은 뒤 다시 열어 주세요.")
            project = Path(profile['project'])
            # Use the operator's project spelling (including an ASCII junction when configured).
            runtime = project.parent / 'Saved' / 'StreetviewToolPreview' / job_id
            runtime.mkdir(parents=True, exist_ok=False)
            renderer_ply = runtime/'scene.ply'
            shutil.copyfile(artifact['source_ply'], renderer_ply)
            if _sha(renderer_ply) != artifact['source_sha256']:
                raise ValueError('언리얼용 PLY 복사 검증에 실패했습니다.')
            config_path, report_path = runtime/'config.json', runtime/'result.json'
            config = dict(artifact, id=job_id, map=state['map'], report_path=str(report_path),
                          renderer_ply=str(renderer_ply), screenshot_path=str(runtime/'preview.png'), project=str(project))
            _write(config_path, config)
            template = Path(__file__).with_name('unreal_preview_script.py').read_text(encoding='utf-8')
            script = runtime/'open_preview.py'
            script.write_text('CONFIG_PATH = ' + repr(str(config_path)) + '\n' + template, encoding='utf-8')
            command = [profile['executable'], profile['project'], '/Engine/Maps/Entry', '-d3d12',
                       '-NoSplash', '-NoSound', '-unattended', '-NoLoadStartupPackages',
                       '-ExecutePythonScript=' + str(script), '-abslog=' + str(runtime/'Unreal.log'),
                       '-ini:Engine:[/Script/Engine.RendererSettings]:r.RayTracing=False',
                       '-ini:Engine:[/Script/Engine.RendererSettings]:r.PostProcessing.PropagateAlpha=True',
                       '-ExecCmds=r.AntiAliasingMethod 0,r.ScreenPercentage 100,r.DefaultFeature.AutoExposure 0,'
                       'r.DefaultFeature.MotionBlur 0,r.DynamicGlobalIlluminationMethod 0,r.ReflectionMethod 0']
            # This is the user-requested interactive editor, so its window stays visible.
            options = dict(cwd=str(project.parent), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.name == 'nt':
                startup = subprocess.STARTUPINFO()
                startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startup.wShowWindow = 1
                options.update(startupinfo=startup, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            process = self._launcher(command, **options)
            self._processes[job_id] = process
            with self._lock:
                state.update(status='running', started_at=_now(), pid=process.pid, free_gpu_mb=free,
                             runtime_dir=str(runtime), result_path=str(report_path), source_ply=artifact['source_ply'])
                _write(directory/'state.json', state)
            deadline = time.monotonic() + profile['timeout_seconds']
            while not self._stop.wait(.5):
                if report_path.is_file():
                    report = json.loads(report_path.read_text(encoding='utf-8'))
                    if report.get('status') == 'failed':
                        raise ValueError(report.get('error', '언리얼 맵 생성에 실패했습니다.'))
                    if report.get('status') == 'completed':
                        self._verify_result(report, config, process.pid)
                        with self._lock:
                            state.update(status='completed', finished_at=_now(), screenshot_path=config['screenshot_path'],
                                         screenshot_sha256=report['screenshot_sha256'], cameras=len(config['cameras']),
                                         result_sha256=_sha(report_path), rendered=True)
                            _write(directory/'state.json', state)
                        return
                if process.poll() is not None:
                    raise ValueError('미리보기 확인 전에 언리얼이 종료됐습니다. 실행 로그를 확인해 주세요.')
                if time.monotonic() > deadline:
                    raise ValueError('언리얼 미리보기 확인 시간이 초과됐습니다. 창과 실행 로그를 확인해 주세요.')
            raise ValueError('서버가 종료되어 열기 결과 확인이 중단됐습니다. 언리얼 창은 유지됩니다.')
        except Exception as error:
            with self._lock:
                state.update(status='interrupted' if self._stop.is_set() else 'failed', error=str(error), finished_at=_now())
                _write(directory/'state.json', state)

    @staticmethod
    def _verify_result(report, config, pid):
        if (report.get('id') != config['id'] or report.get('pid') != pid
                or report.get('map') != config['map'] or report.get('source_sha256') != config['source_sha256']
                or report.get('camera_sha256') != config['camera_sha256']
                or not report.get('serialized_verified') or not report.get('source_unchanged')):
            raise ValueError('언리얼 결과와 요청한 PLY·카메라 정보가 일치하지 않습니다.')
        image = Path(config['screenshot_path'])
        if not image.is_file() or image.stat().st_size < 1000 or image.read_bytes()[:8] != b'\x89PNG\r\n\x1a\n':
            raise ValueError('언리얼 미리보기 이미지가 없습니다.')
        if _sha(image) != report.get('screenshot_sha256') or _sha(config['source_ply']) != config['source_sha256']:
            raise ValueError('언리얼 결과 파일이 변경됐습니다.')
        if _sha(config['renderer_ply']) != config['source_sha256']:
            raise ValueError('언리얼에서 참조하는 PLY 복사본이 변경됐습니다.')

    def close(self):
        self._stop.set()
        self._executor.shutdown(wait=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Open a verified Gaussian PLY in a new Unreal comparison level.')
    parser.add_argument('--profile', default=str(Path(__file__).with_name('data')/'unreal_profile.json'))
    parser.add_argument('--source-ply', required=True)
    parser.add_argument('--camera-json', required=True)
    parser.add_argument('--size-filter-report', help='Required for a filtered PLY whose camera binding refers to its parent.')
    parser.add_argument('--output-dir', required=True, help='New local preview job directory (editor remains open).')
    parser.add_argument('--label', default='Streetview')
    args = parser.parse_args(argv)
    from ..streetview_engine.size_filter import _ply, _camera_reference, verify_size_filter_report
    source, header, _ = _ply(args.source_ply)
    if args.size_filter_report:
        path = Path(args.size_filter_report).resolve()
        report = json.loads(path.read_text(encoding='utf-8'))
        verify_size_filter_report(report, report['source']['path'], args.source_ply, args.camera_json,
                                  path.with_name('selection.npz'))
        camera = report['camera_reference']
    else:
        camera = _camera_reference(args.camera_json, source, header)
    output = Path(args.output_dir).resolve()
    if output.exists():
        parser.error('--output-dir must be a new directory')
    descriptor = dict(source_ply=str(Path(args.source_ply).resolve()), camera_json=str(Path(args.camera_json).resolve()),
                      source_sha256=source['sha256'], camera_sha256=_sha(args.camera_json),
                      coordinate_frame=camera['coordinate_frame'], units=camera['units'],
                      camera_convention=camera['camera_convention'], label=args.label)
    manager = UnrealOpenManager(output, args.profile)
    try:
        job = manager.start(descriptor)
        while job['status'] in ('queued', 'running'):
            time.sleep(.5)
            job = manager.get(job['id'])
        print(json.dumps(job, ensure_ascii=False, indent=2))
        return 0 if job['status'] == 'completed' else 1
    finally:
        manager.close()


if __name__ == '__main__':
    raise SystemExit(main())
