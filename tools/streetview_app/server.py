"""Loopback-only local UI server. No backend job starts on GET or page load."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import threading
from urllib.parse import parse_qs, unquote, urlsplit
import webbrowser
from .plans import PlanStore, freeze_selection
from ..streetview_engine.selection_limits import read_radius_m

STATIC = Path(__file__).parent / 'static'


class Application:
    def __init__(self, data_dir, provider=None, jobs=None, filters=None, unreal=None):
        self.data_dir = Path(data_dir).resolve(); self.data_dir.mkdir(parents=True, exist_ok=True)
        if provider is None:
            from .provider import NaverProvider
            provider = NaverProvider(self.data_dir / 'cache')
        if jobs is None:
            from .jobs import JobManager
            backend_path = self.data_dir / 'backend.json'
            backend = json.loads(backend_path.read_text(encoding='utf8')) if backend_path.exists() else None
            if backend and backend.get('compute') == 'remote_adapter':
                from .remote_jobs import RemoteJobManager
                jobs = RemoteJobManager(self.data_dir / 'jobs', backend=backend)
            else:
                jobs = JobManager(self.data_dir / 'jobs', backend=backend)
        self.provider = provider; self.jobs = jobs; self.plans = PlanStore(self.data_dir / 'plans')
        from .ply_filters import PlyFilterManager
        self.filters = filters if filters is not None else PlyFilterManager(
            self.data_dir / 'ply_filters', self.data_dir / 'ply_filter_defaults.json')
        from .recipes import RecipeStore
        from .reuse import ReuseService
        from .unreal_open import UnrealOpenManager
        self.recipes = RecipeStore(self.data_dir / 'recipes', self.jobs.backend)
        self.reuse = ReuseService(self.jobs)
        self.unreal = unreal if unreal is not None else UnrealOpenManager(self.data_dir / 'unreal_opens', self.data_dir / 'unreal_profile.json')

    def public_recipe(self, recipe):
        from .recipes import depth_cleanup_capability
        result = dict(recipe)
        if recipe.get('settings', {}).get('generation_mode', 'multi_view') != 'multi_view':
            result['executable'] = False
            result['unavailable_reasons'] = [*recipe.get('unavailable_reasons', []),
                '단일 파노라마 생성 기능은 종료되었습니다.']
        result['depth_cleanup'] = depth_cleanup_capability(
            recipe.get('operator_settings'), recipe.get('settings'))
        preview = recipe.get('reference_thumbnail')
        if isinstance(preview, dict) and preview.get('path') and preview.get('sha256'):
            result['preview_url'] = f"/api/recipes/{recipe['id']}/preview"
        return result

    def require_product_generation(self, config, recipe_id=None):
        """Archived single-panorama artifacts remain readable, never restarted."""
        from .jobs import generation_mode
        if generation_mode(config) != 'multi_view':
            raise ValueError('단일 파노라마 생성 기능은 종료되었습니다. 주변 공간 생성을 사용해 주세요.')
        if recipe_id:
            recipe = self.recipes.get(recipe_id)
            if generation_mode(recipe.get('settings', {})) != 'multi_view':
                raise ValueError('이 프리셋은 종료된 단일 파노라마 생성 방식입니다.')

    def default_recipe_id(self):
        path = self.data_dir / 'ui_preferences.json'
        try:
            recipe_id = json.loads(path.read_text(encoding='utf8')).get('default_recipe_id')
            self.recipes.get(recipe_id)
            return recipe_id
        except (OSError, ValueError, TypeError, KeyError):
            return None

    def status(self):
        from .recipes import depth_cleanup_capability, operator_settings
        try:
            depth_cleanup = depth_cleanup_capability(operator_settings(self.jobs.backend))
        except (OSError, ValueError, TypeError, KeyError):
            depth_cleanup = dict(available=False, default_enabled=False)
        modes = {'multi_view': self.jobs.capabilities('multi_view')}
        ready = {mode: bool(value.get('generation_available', value.get('can_start', False))) for mode, value in modes.items()}
        busy = {mode: bool(value.get('busy', False)) for mode, value in modes.items()}
        reasons = {mode: '다른 생성 작업이 진행 중입니다.' if busy[mode] else (None if ready[mode] else '선택한 생성 방식의 엔진이 연결되지 않았습니다.') for mode in modes}
        return dict(name='Streetview to PLY', version='0.1.0', depth_cleanup=depth_cleanup,
            capabilities=dict(discover=True, preview=True, save_plan=True, generate=ready['multi_view'] and not busy['multi_view'],
                              generation_modes={mode: ready[mode] and not busy[mode] for mode in modes}),
            disabled_reasons=dict(generate=reasons['multi_view'], **reasons),
            jobs=self.jobs.list(), saved_plans=len(self.plans.list()))


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        server_version = 'StreetviewPLY/0.1'

        def log_message(self, *_):
            pass

        def _send(self, status, payload, content_type='application/json; charset=utf-8', *, filename=None):
            if not isinstance(payload, bytes):
                payload = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf8')
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('X-Content-Type-Options', 'nosniff')
            cache = 'no-store' if content_type.startswith('application/json') else ('no-cache' if content_type.startswith('text/') or 'javascript' in content_type else 'private, max-age=300')
            self.send_header('Cache-Control', cache)
            if filename:
                self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
            self.end_headers(); self.wfile.write(payload)

        def _error(self, status, message):
            self._send(status, dict(error=message))

        def _download_ply(self, job, *, storage_root=None):
            artifact = job.get('artifact') or {}
            job_root = (Path(storage_root if storage_root is not None else app.jobs.root) / job['id']).resolve()
            path = (job_root / artifact.get('path', '')).resolve()
            if not path.is_relative_to(job_root) or not path.is_file():
                return self._error(404, '생성된 PLY 파일이 없습니다.')
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if digest != artifact.get('sha256'):
                return self._error(409, '완료 후 결과 파일이 변경돼 다운로드를 보류했습니다.')
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Length', str(path.stat().st_size))
            self.send_header('Content-Disposition', f'attachment; filename="streetview-{job["id"][:8]}.ply"')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    self.wfile.write(block)

        def _local_request(self):
            host = self.headers.get('Host', '')
            try:
                hostname = urlsplit('http://' + host).hostname
            except ValueError:
                return False
            if hostname not in ('127.0.0.1', 'localhost', '::1'):
                return False
            origin = self.headers.get('Origin')
            if origin:
                parsed = urlsplit(origin)
                if parsed.scheme != 'http' or parsed.netloc != host:
                    return False
            return True

        def do_GET(self):
            if not self._local_request():
                return self._error(403, '이 프로그램의 로컬 화면에서만 사용할 수 있습니다.')
            parsed = urlsplit(self.path); path = unquote(parsed.path)
            query = parse_qs(parsed.query)
            one = lambda name, default=None: query.get(name, [default])[0]
            try:
                if path == '/api/status':
                    return self._send(200, app.status())
                if path == '/api/discover':
                    result = app.provider.discover(float(one('lat')), float(one('lng', one('lon'))), read_radius_m(one('radius_m', '100')),
                        max_nodes=int(one('max_nodes', '200')), date=one('date'), continuation=one('continuation'))
                    return self._send(200, result)
                if path == '/api/panorama':
                    return self._send(200, app.provider.get_panorama(one('id')))
                if path.startswith('/api/cube/'):
                    parts = path[len('/api/cube/'):].rsplit('/', 1)
                    if len(parts) != 2:
                        return self._error(400, '거리뷰 요청이 올바르지 않습니다.')
                    payload, mime = app.provider.cube_face(parts[0], parts[1])
                    return self._send(200, payload, mime)
                if path == '/api/jobs':
                    return self._send(200, dict(jobs=app.jobs.list()))
                if path == '/api/recipes':
                    return self._send(200, dict(recipes=[app.public_recipe(row) for row in app.recipes.list()], default_recipe_id=app.default_recipe_id()))
                recipe_match = re.fullmatch(r'/api/recipes/([0-9a-f]{32})(/preview)?', path)
                if recipe_match:
                    recipe = app.recipes.get(recipe_match[1])
                    if recipe_match[2]:
                        preview = recipe.get('reference_thumbnail') or {}
                        candidate = Path(preview.get('path', ''))
                        if not candidate.is_file() or candidate.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
                            return self._error(404, '등록된 대표 이미지가 없습니다.')
                        raw = candidate.read_bytes()
                        if hashlib.sha256(raw).hexdigest() != preview.get('sha256'):
                            return self._error(409, '대표 이미지가 변경됐습니다.')
                        return self._send(200, raw, 'image/png' if candidate.suffix.lower() == '.png' else 'image/jpeg')
                    return self._send(200, app.public_recipe(recipe))
                if path == '/api/unreal-opens':
                    return self._send(200, dict(capability=app.unreal.capability(), jobs=app.unreal.list()))
                unreal_match = re.fullmatch(r'/api/unreal-opens/([0-9a-f]{32})', path)
                if unreal_match:
                    return self._send(200, app.unreal.get(unreal_match[1]))
                reuse_match = re.fullmatch(r'/api/jobs/([0-9a-f]{32})/reuse', path)
                if reuse_match:
                    app.require_product_generation(app.jobs.get(reuse_match[1]).get('config', {}))
                    return self._send(200, app.reuse.inspect(reuse_match[1]))
                if path == '/api/ply-filters':
                    return self._send(200, dict(jobs=app.filters.list(), defaults=app.filters.defaults()))
                filter_match = re.fullmatch(r'/api/ply-filters/([0-9a-f]{32})(/download)?', path)
                if filter_match:
                    job = app.filters.get(filter_match[1])
                    if filter_match[2]:
                        if job.get('status') != 'completed':
                            return self._error(409, 'PLY 정리가 아직 완료되지 않았습니다.')
                        return self._download_ply(job, storage_root=app.filters.root)
                    return self._send(200, job)
                if path.startswith('/api/jobs/'):
                    job_id = path.split('/')[3]
                    job = app.jobs.get(job_id)
                    if path.endswith('/download'):
                        if job.get('status') != 'completed':
                            return self._error(409, 'PLY 생성이 완료되지 않았습니다.')
                        return self._download_ply(job)
                    if path.endswith('/report'):
                        report = job.get('quality_report')
                        if job.get('status') != 'completed' or not report:
                            return self._error(409, '생성 결과의 품질 보고서가 아직 없습니다.')
                        report_path = app.jobs.root / job['id'] / 'export/report.json'
                        raw = report_path.read_bytes()
                        if hashlib.sha256(raw).hexdigest() != report.get('sha256'):
                            return self._error(409, '완료 후 품질 보고서가 변경되었습니다.')
                        return self._send(200, raw, filename=f"streetview-quality-{job['id'][:8]}.json")
                    return self._send(200, job)
                if path == '/api/plans':
                    return self._send(200, dict(plans=app.plans.list()))
                if path.startswith('/api/plans/'):
                    plan_id = path.split('/')[3]
                    plan = app.plans.get(plan_id)
                    return self._send(200, plan, filename=f'streetview-plan-{plan_id[:8]}.json' if path.endswith('/download') else None)
                relative = 'index.html' if path == '/' else path.lstrip('/')
                candidate = (STATIC / relative).resolve()
                if not candidate.is_relative_to(STATIC.resolve()) or not candidate.is_file():
                    return self._error(404, '페이지가 없습니다.')
                mime = mimetypes.guess_type(candidate.name)[0] or 'application/octet-stream'
                if mime.startswith('text/') or mime in ('application/javascript',):
                    mime += '; charset=utf-8'
                return self._send(200, candidate.read_bytes(), mime)
            except (ValueError, TypeError) as error:
                return self._error(400, str(error))
            except (KeyError, FileNotFoundError):
                return self._error(404, '요청한 자료를 찾을 수 없습니다.')
            except Exception as error:
                return self._error(502, '자료를 가져오지 못했습니다. 잠시 후 다시 시도해 주세요. ' + type(error).__name__)

        def do_POST(self):
            if not self._local_request():
                # Drain a bounded body before closing. Windows otherwise resets
                # the connection and hides the actual 403 response from clients.
                try:
                    rejected_length = int(self.headers.get('Content-Length', '0'))
                    if 0 < rejected_length <= 1024 * 1024:
                        self.connection.settimeout(5)
                        self.rfile.read(rejected_length)
                except (ValueError, OSError):
                    pass
                return self._error(403, '다른 사이트에서는 이 프로그램의 작업을 시작할 수 없습니다.')
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 1024 * 1024 or self.headers.get_content_type() != 'application/json':
                    return self._error(400, 'JSON 요청만 사용할 수 있습니다.')
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    return self._error(400, '요청 형식이 올바르지 않습니다.')
                path = urlsplit(self.path).path
                if path == '/api/recipes':
                    if not {'name', 'config'} <= payload.keys() or set(payload) - {'name', 'config', 'base_recipe_id'} or not isinstance(payload['config'], dict):
                        raise ValueError('프리셋 이름과 생성 설정이 필요합니다.')
                    app.require_product_generation(payload['config'], payload.get('base_recipe_id'))
                    request = dict(payload, config=freeze_selection(payload['config'], app.provider))
                    return self._send(201, dict(recipe=app.public_recipe(app.recipes.save(request))))
                if path in ('/api/artifacts/resolve', '/api/unreal-opens'):
                    from .artifacts import resolve_artifact
                    resolved = resolve_artifact(app, payload)
                    if path == '/api/artifacts/resolve':
                        return self._send(200, resolved)
                    return self._send(201, dict(job=app.unreal.start(resolved)))
                if path == '/api/plans':
                    recipe_id = payload.pop('recipe_id', None)
                    app.require_product_generation(payload, recipe_id)
                    frozen = freeze_selection(payload, app.provider)
                    if recipe_id:
                        frozen = app.recipes.resolve(recipe_id, frozen)['config']
                    plan = app.plans.save(frozen)
                    return self._send(201, dict(plan=plan, download_url=f"/api/plans/{plan['id']}/download"))
                if path == '/api/ply-filters':
                    return self._send(201, dict(job=app.filters.start(payload)))
                if path == '/api/jobs':
                    from .jobs import generation_mode
                    app.require_product_generation(payload, payload.get('recipe_id'))
                    mode = generation_mode(payload)
                    status = app.status()
                    if not status['capabilities']['generation_modes'][mode]:
                        return self._error(409, status['disabled_reasons'][mode] + ' 생성 설정을 먼저 저장할 수 있습니다.')
                    recipe_id = payload.pop('recipe_id', None)
                    frozen = freeze_selection(payload, app.provider)
                    arguments = app.recipes.resolve(recipe_id, frozen) if recipe_id else dict(config=frozen)
                    if not recipe_id:
                        from ..streetview_engine.depth_cleanup_options import read_depth_cleanup_options, validate_depth_cleanup_support
                        option = read_depth_cleanup_options(frozen)
                        if option is not None and option['enabled']:
                            from .recipes import operator_settings
                            validate_depth_cleanup_support(frozen, operator_settings(app.jobs.backend))
                    return self._send(201, dict(job=app.jobs.start(**arguments)))
                reuse_match = re.fullmatch(r'/api/jobs/([0-9a-f]{32})/reuse', path)
                if reuse_match:
                    if set(payload) - {'from_stage', 'overrides'}:
                        raise ValueError('지원하지 않는 재사용 설정입니다.')
                    app.require_product_generation(app.jobs.get(reuse_match[1]).get('config', {}))
                    return self._send(201, dict(job=app.reuse.start(reuse_match[1], payload.get('from_stage'), payload.get('overrides'))))
                match = re.fullmatch(r'/api/jobs/([0-9a-f]+)/cancel', path)
                if match:
                    return self._send(200, dict(job=app.jobs.cancel(match[1])))
                return self._error(404, '지원하지 않는 요청입니다.')
            except (ValueError, TypeError, KeyError) as error:
                return self._error(400, str(error))
            except Exception as error:
                return self._error(500, '요청을 완료하지 못했습니다. ' + type(error).__name__)
    return Handler


def main():
    parser = argparse.ArgumentParser(description='Local streetview selection and PLY job interface')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).parent / 'data')
    parser.add_argument('--open', action='store_true')
    args = parser.parse_args()
    app = Application(args.data_dir)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(app))
    url = f'http://127.0.0.1:{server.server_port}/'
    args.data_dir.mkdir(parents=True, exist_ok=True)
    (args.data_dir / 'server.json').write_text(json.dumps(dict(url=url, port=server.server_port)), encoding='utf8')
    print(url, flush=True)
    if args.open:
        threading.Timer(.2, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.unreal.close()
        app.filters.close()
        app.jobs.close()
        server.server_close()


if __name__ == '__main__':
    main()
