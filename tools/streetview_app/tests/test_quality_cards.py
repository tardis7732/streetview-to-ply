"""Opt-in plain job UI tests with synthetic jobs and a temporary mock server.

STREETVIEW_QUALITY_CARD_BROWSER=1 python -m pytest this_file.py -q

No real job, report, artifact, provider, or cloud service is accessed. The
historical environment variable is retained for existing browser test runners.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
from urllib.parse import urlsplit

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get('STREETVIEW_QUALITY_CARD_BROWSER') != '1',
    reason='Explicit isolated browser opt-in; normal tests do not launch Chrome',
)
APP = Path(__file__).resolve().parents[1]
JOB_ID = 'a' * 32


def sample(status='completed'):
    # Deliberately keep legacy report metadata in the response: it must not
    # cause a report fetch, quality badge, technical metrics, or report link.
    return dict(id=JOB_ID, status=status, stage='export',
                created_utc='2026-09-11T00:00:00+00:00',
                artifact=dict(sha256='b' * 64),
                quality_report=dict(sha256='c' * 64))


@pytest.fixture(scope='module')
def browser():
    sync = pytest.importorskip('playwright.sync_api')
    with sync.sync_playwright() as playwright:
        instance = playwright.chromium.launch(channel='chrome', headless=True)
        yield instance
        instance.close()


@pytest.fixture
def ui(browser):
    state = dict(job=None, report_requests=0, posts=[])

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, body, mime='application/json', code=200):
            raw = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header('Content-Type', mime + '; charset=utf-8')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            state['posts'].append((self.path, body))
            if self.path == f'/api/jobs/{JOB_ID}/cancel':
                state['job']['status'] = 'cancelling'
                self.reply(dict(job=state['job']))
            else:
                self.reply(dict(error='Unexpected mock POST'), code=405)

        def do_GET(self):
            parsed = urlsplit(self.path)
            if parsed.path == '/api/status':
                self.reply(dict(capabilities=dict(generate=False), jobs=[state['job']]))
            elif parsed.path.endswith('/report'):
                state['report_requests'] += 1
                self.reply(dict(error='Report UI has been removed'), code=409)
            else:
                path = (APP / 'static' / (parsed.path.lstrip('/') or 'index.html')).resolve()
                if not path.is_relative_to((APP / 'static').resolve()) or not path.is_file():
                    self.send_error(404)
                    return
                mime = {'.html': 'text/html', '.js': 'application/javascript',
                        '.css': 'text/css'}.get(path.suffix, 'application/octet-stream')
                self.reply(path.read_bytes(), mime)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    page = browser.new_page(viewport=dict(width=1600, height=1040))
    page.set_default_timeout(10000)
    page.route('https://**/*', lambda route: route.abort())
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    state.update(page=page, url=f'http://127.0.0.1:{server.server_address[1]}')
    try:
        yield state
    finally:
        page.close()
        server.shutdown()
        server.server_close()
        thread.join(2)
        assert not errors, errors
        assert state['report_requests'] == 0


def load(ui, job=None):
    ui['job'] = sample() if job is None else job
    ui['page'].goto(ui['url'], wait_until='domcontentloaded')
    ui['page'].locator('.job-card').wait_for()
    return ui['page']


def assert_plain_card(page):
    assert page.locator('.job-quality, .quality-metrics, .quality-caveat').count() == 0
    assert page.locator('.job-actions a[href$="/report"]').count() == 0
    assert page.locator(f'.job-actions a[href="/api/jobs/{JOB_ID}"]').count() == 0
    body = page.locator('.job-card').inner_text()
    assert all(word not in body for word in ('품질', 'PSNR', '보고서', '작업 정보'))


def test_completed_job_keeps_only_status_stage_and_ply_link_even_with_report_metadata(ui):
    from playwright.sync_api import expect
    page = load(ui)
    expect(page.locator('.job-status')).to_have_text('파일 생성 완료')
    expect(page.locator('.job-meta')).to_contain_text('PLY 내보내기')
    expect(page.locator('.job-actions a')).to_have_text('PLY 다운로드 ↓')
    expect(page.locator('.job-actions a')).to_have_attribute('href', f'/api/jobs/{JOB_ID}/download')
    assert page.locator('.job-actions button').count() == 0
    # Older jobs without persisted choices must not borrow the current UI defaults.
    assert page.locator('.job-options').count() == 0
    for _ in range(2):
        with page.expect_response('**/api/status'):
            page.locator('#refresh-jobs').click()
        assert_plain_card(page)
    assert not ui['posts']
    page.set_viewport_size(dict(width=390, height=844))
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')




@pytest.mark.parametrize('status,label', [
    ('failed', '실패'), ('interrupted', '중단됨'),
    ('cancelled', '취소됨'), ('cancelling', '취소 중'),
])
def test_noncompleted_job_does_not_show_download_or_finished_report_controls(ui, status, label):
    from playwright.sync_api import expect
    job = sample(status)
    job['error'] = 'MOCK: 입력 자료를 읽지 못했습니다.'
    page = load(ui, job)
    expect(page.locator('.job-status')).to_have_text(label)
    expect(page.locator('.job-card')).to_contain_text(job['error'])
    assert page.locator('.job-actions a, .job-actions button').count() == 0
    assert_plain_card(page)
    assert not ui['posts']


@pytest.mark.parametrize('status,label,progress', [('queued', '대기 중', .25), ('running', '진행 중', 25)])
def test_active_job_keeps_progress_and_only_explicit_cancel_posts_to_mock(ui, status, label, progress):
    from playwright.sync_api import expect
    job = sample(status)
    job.update(stage='train', progress=progress)
    page = load(ui, job)
    expect(page.locator('.job-status')).to_have_text(label)
    expect(page.locator('.job-meta')).to_contain_text('Gaussian 학습')
    expect(page.get_by_role('progressbar')).to_have_attribute('aria-valuenow', '25')
    expect(page.locator('.job-actions button')).to_have_text('작업 취소')
    assert page.locator('.job-actions a').count() == 0
    assert_plain_card(page)
    assert not ui['posts']
    with page.expect_response(f'**/api/jobs/{JOB_ID}/cancel'):
        page.locator('.job-actions button').click()
    expect(page.locator('.job-status')).to_have_text('취소 중')
    assert ui['posts'] == [(f'/api/jobs/{JOB_ID}/cancel', {})]
    assert page.locator('.job-actions button').count() == 0


def test_completed_job_without_artifact_does_not_invent_download(ui):
    from playwright.sync_api import expect
    job = sample()
    job.pop('artifact')
    page = load(ui, job)
    expect(page.locator('.job-status')).to_have_text('파일 생성 완료')
    assert page.locator('.job-actions a, .job-actions button').count() == 0
    assert_plain_card(page)
    assert not ui['posts']


@pytest.mark.parametrize('remove_sky,mask_dynamic', [(False, False), (False, True), (True, False), (True, True)])
def test_job_choices_are_persisted_options_independent_of_current_checkbox_values(ui, remove_sky, mask_dynamic):
    from playwright.sync_api import expect
    job = sample()
    job['config'] = dict(processing_options=dict(remove_sky=remove_sky, mask_dynamic=mask_dynamic))
    page = load(ui, job)
    summary = ('하늘 제거' if remove_sky else '하늘 유지') + ' · ' + (
        '인물·차량 마스크 켬' if mask_dynamic else '인물·차량 마스크 끔')
    expect(page.locator('.job-options')).to_have_text(summary)
    page.locator('#remove-sky').set_checked(not remove_sky)
    page.locator('#mask-dynamic').set_checked(not mask_dynamic)
    with page.expect_response('**/api/status'):
        page.locator('#refresh-jobs').click()
    expect(page.locator('.job-options')).to_have_text(summary)
    expect(page.locator('.job-actions a')).to_have_text('PLY 다운로드 ↓')
    assert_plain_card(page)
    assert not ui['posts']
    page.set_viewport_size(dict(width=390, height=844))
    expect(page.locator('.job-options')).to_be_visible()
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
