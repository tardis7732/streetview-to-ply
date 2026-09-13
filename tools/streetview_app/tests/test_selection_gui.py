"""Isolated selection GUI tests: mock metadata, images, and POST receiver only.

STREETVIEW_SELECTION_BROWSER=1 python -m pytest this_file.py -q

No provider request, real generation server, cloud job, or generated PLY is used.
External links are intercepted in the test browser; these tests verify the URL
and navigation behavior, not the external provider's handling of the URL.
"""
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlsplit

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get('STREETVIEW_SELECTION_BROWSER') != '1',
    reason='Explicit isolated browser opt-in; normal tests do not launch Chrome',
)
APP = Path(__file__).resolve().parents[1]
IDS = ['10000000000000000001', '10000000000000000002',
       '10000000000000000003', '10000000000000000004']
DATES = ['2026-08-20', '2026-08-20', '2026-08-19', '2026-07-10']
# A synthetic 1 x 1 PNG, not a streetview image.
PNG = base64.b64decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aWQ0AAAAASUVORK5CYII='
)


@pytest.fixture(scope='module')
def browser():
    sync = pytest.importorskip('playwright.sync_api')
    with sync.sync_playwright() as playwright:
        instance = playwright.chromium.launch(channel='chrome', headless=True)
        yield instance
        instance.close()


@pytest.fixture
def ui(browser):
    panoramas = [dict(id=pano_id, lat=35.1 + index * .00001,
                      lng=128.1, title=f'MOCK 촬영 지점 {index + 1}',
                      capture_date=day, captured_at=day + 'T12:30:00',
                      capture_precision='second', heading=37.5)
                 for index, (pano_id, day) in enumerate(zip(IDS, DATES))]
    state = dict(posts=[], gets=[], panoramas=panoramas, discovery_pages=None,
                 discovery_codes={}, post_release=None, capabilities=dict(generate=True), disabled_reasons={},
                 filter_available=True, filter_jobs=[], filter_defaults=None)

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
            if state['post_release'] is not None:
                state['post_release'].wait(10)
            if self.path == '/api/plans':
                self.reply(dict(plan=dict(id='mock-plan'),
                                download_url='/mock-plan.json'))
            elif self.path == '/api/jobs':
                self.reply(dict(job=dict(id='mock-job', status='queued')))
            elif self.path == '/api/ply-filters':
                job = dict(id='mock-filter', status='completed', source_ply=body['source_ply'],
                    camera_json=body['camera_json'], options=dict(max_sigma_camera_radius_ratio=body['max_sigma_camera_radius_ratio']),
                    removed_rows=7, remaining_rows=93, artifact=dict(path='filtered.ply'))
                state['filter_jobs'].insert(0, job)
                self.reply(dict(job=job), code=201)
            else:
                self.reply(dict(error='Unexpected mock POST'), code=404)

        def do_GET(self):
            state['gets'].append(self.path)
            parsed = urlsplit(self.path)
            if parsed.path == '/api/status':
                self.reply(dict(capabilities=state['capabilities'], disabled_reasons=state['disabled_reasons'], jobs=[]))
            elif parsed.path == '/api/ply-filters':
                if state['filter_available']:
                    self.reply(dict(jobs=state['filter_jobs'], defaults=state['filter_defaults']))
                else:
                    self.reply(dict(error='Not connected'), code=404)
            elif parsed.path == '/api/discover':
                token = parse_qs(parsed.query).get('continuation', [''])[0]
                if state['discovery_pages'] is not None:
                    result = state['discovery_pages'].get(token)
                    code = state['discovery_codes'].get(token, 200)
                    self.reply(result or dict(error='Unknown mock continuation'),
                               code=code if result else 404)
                else:
                    self.reply(dict(panoramas=state['panoramas'], date_options=[],
                                    warnings=['MOCK DATA: isolated GUI validation']))
            elif parsed.path == '/api/panorama':
                pano_id = parse_qs(parsed.query)['id'][0]
                details = next(p for p in state['panoramas'] if p['id'] == pano_id)
                self.reply(dict(**details, faces={face: '/mock-face.png'
                                                 for face in 'FRBLUD'}))
            elif parsed.path == '/mock-face.png':
                self.reply(PNG, 'image/png')
            else:
                relative = parsed.path.lstrip('/') or 'index.html'
                path = (APP / 'static' / relative).resolve()
                if not path.is_relative_to((APP / 'static').resolve()) or not path.is_file():
                    self.send_error(404)
                    return
                mime = {'.html': 'text/html', '.js': 'application/javascript',
                        '.css': 'text/css'}.get(path.suffix, 'application/octet-stream')
                self.reply(path.read_bytes(), mime)

    # Windows can return low ephemeral ports (e.g. 4045) that Chromium blocks.
    # Keep this isolated fixture on a browser-safe high port.
    for port in range(49152, 49280):
        try:
            server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
            break
        except OSError:
            continue
    else:
        raise RuntimeError('Could not allocate a browser-safe local fixture port')
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    context = browser.new_context(viewport=dict(width=1600, height=1040))

    def intercept_external(route):
        if urlsplit(route.request.url).hostname == 'map.naver.com':
            route.fulfill(status=200, content_type='text/html',
                          body='<title>MOCK external navigation target</title>')
        else:
            route.abort()

    context.route('https://**/*', intercept_external)
    page = context.new_page()
    page.set_default_timeout(10000)
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    state.update(page=page, url=f'http://127.0.0.1:{server.server_address[1]}')
    try:
        yield state
    finally:
        if state['post_release'] is not None:
            state['post_release'].set()
        context.close()
        server.shutdown()
        server.server_close()
        thread.join(2)
        assert not errors, errors


def load(ui):
    from playwright.sync_api import expect
    page = ui['page']
    page.goto(ui['url'], wait_until='domcontentloaded')
    expect(page.locator('#connection-label')).to_have_text('로컬 서버 연결됨')
    return page


def discover(ui, expected_count='2'):
    from playwright.sync_api import expect
    page = load(ui)
    page.locator('#latitude').fill('35.1')
    page.locator('#longitude').fill('128.1')
    page.locator('#coordinate-form button').click()
    page.locator('#discover').click()
    expect(page.locator('#selected-count')).to_have_text(expected_count)
    expect(page.locator('#generate')).to_be_enabled()
    return page


def row(page, index):
    return page.locator(f'.station-row[data-panorama-id="{IDS[index]}"]')


def mock_screenshot(page, name):
    if os.environ.get('STREETVIEW_SELECTION_SCREENSHOTS') == '1':
        directory = APP / 'validation' / 'selection_controls_mock'
        directory.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(directory / name), full_page=True)




def test_size_filter_generation_preserves_separate_options(ui):
    from playwright.sync_api import expect
    ui['capabilities']['generation_modes'] = dict(multi_view=True)
    page = discover(ui)
    expect(page.locator('#size-filter-enabled')).not_to_be_checked()
    expect(page.locator('#size-filter-percent')).to_be_disabled()
    page.locator('#size-filter-enabled').check()
    page.locator('#size-filter-percent').fill('25')
    assert not ui['posts']
    with page.expect_response('**/api/plans'):
        page.locator('#save-plan').click()
    assert ui['posts'][-1][1]['size_filter'] == dict(enabled=True,max_sigma_camera_radius_ratio=.25)
    page.locator('#size-filter-percent').fill('0')
    expect(page.locator('#generate')).to_be_disabled()
    expect(page.locator('#save-plan')).to_be_disabled()
    page.locator('#size-filter-percent').fill('25')
    expect(page.locator('#save-plan')).to_be_enabled()


def test_existing_ply_defaults_edit_submit_and_download_without_generation(ui):
    from playwright.sync_api import expect
    ui['filter_defaults'] = dict(source_ply='D:/synthetic/source.ply',camera_json='D:/synthetic/cameras.json',max_sigma_camera_radius_ratio=.5)
    page = load(ui)
    expect(page.locator('#ply-filter-source')).to_have_value('D:/synthetic/source.ply')
    expect(page.locator('#ply-filter-submit')).to_be_enabled()
    assert not ui['posts']
    page.locator('#ply-filter-source').fill('D:/other/input.ply')
    page.locator('#ply-filter-percent').fill('12.5')
    page.locator('#refresh-jobs').click()
    expect(page.locator('#ply-filter-source')).to_have_value('D:/other/input.ply')
    with page.expect_response('**/api/ply-filters'):
        page.locator('#ply-filter-submit').click()
    expect(page.locator('#ply-filter-jobs')).to_contain_text('7개 삭제 · 93개 유지')
    assert ui['posts'] == [('/api/ply-filters',dict(source_ply='D:/other/input.ply',camera_json='D:/synthetic/cameras.json',max_sigma_camera_radius_ratio=.125))]
    expect(page.locator('#ply-filter-jobs a')).to_have_attribute('href','/api/ply-filters/mock-filter/download')
    expect(page.locator('#ply-filter-submit')).to_be_enabled()
    mock_screenshot(page,'size_filter_completed.png')


def test_existing_ply_api_unavailable_does_not_block_map_or_generation(ui):
    from playwright.sync_api import expect
    ui['filter_available'] = False
    page = discover(ui)
    expect(page.locator('#ply-filter-submit')).to_be_disabled()
    expect(page.locator('#ply-filter-note')).to_contain_text('연결할 수 없습니다')
    expect(page.locator('#generate')).to_be_enabled()
    assert not ui['posts']


def test_initial_controls_and_discovery_select_only_current_date_without_clock_filter(ui):
    from playwright.sync_api import expect
    page = load(ui)
    expect(page.locator('#naver-roadview')).to_have_attribute('aria-disabled', 'true')
    assert page.locator('#naver-roadview').get_attribute('href') is None
    expect(page.locator('#toggle-preview-selection')).to_be_disabled()
    expect(page.locator('#save-plan')).to_be_disabled()
    assert page.locator('#time-start, #time-end, input[type="time"]').count() == 0
    expect(page.locator('#load-more')).to_be_hidden()
    expect(page.locator('#remove-sky')).to_be_checked()
    expect(page.locator('#mask-dynamic')).to_be_checked()
    expect(page.locator('#remove-sky')).to_be_enabled()
    expect(page.locator('#mask-dynamic')).to_be_enabled()
    page = discover(ui)
    expect(page.locator('#station-count')).to_have_text('2')
    expect(page.locator('#excluded-count')).to_have_text('0')
    expect(row(page, 0).locator('input')).to_be_checked()
    expect(row(page, 1).locator('input')).to_be_checked()
    assert not ui['posts']








def test_exclusion_preview_and_external_link_are_independent_of_generation(ui):
    from playwright.sync_api import expect
    page = discover(ui)
    row(page, 0).locator('input').uncheck()
    expect(row(page, 0)).to_have_class('station-row excluded')
    expect(page.locator('#selected-count')).to_have_text('1')
    expect(page.locator('#excluded-count')).to_have_text('1')
    row(page, 0).locator('.station-open').click()
    expect(page.locator('#preview-meta')).to_contain_text('저해상도 미리보기')
    expect(page.locator('#toggle-preview-selection')).to_have_text('학습에 포함')
    expect(row(page, 0).locator('input')).not_to_be_checked()
    link = page.locator('#naver-roadview')
    expect(link).to_have_attribute('aria-disabled', 'false')
    expect(link).to_have_attribute('target', '_blank')
    assert {'noopener', 'noreferrer'} <= set(link.get_attribute('rel').split())
    target = urlsplit(link.get_attribute('href'))
    assert (target.scheme, target.netloc, target.path) == ('https', 'map.naver.com', '/p/')
    assert parse_qs(target.query)['p'] == [f'{IDS[0]},38,0,80,Float']
    with page.expect_popup() as popup:
        link.click()
    popup.value.wait_for_load_state('domcontentloaded')
    assert parse_qs(urlsplit(popup.value.url).query)['p'][0].startswith(IDS[0] + ',')
    popup.value.close()
    assert not ui['posts']
    page.locator('#toggle-preview-selection').click()
    expect(row(page, 0).locator('input')).to_be_checked()
    expect(page.locator('#toggle-preview-selection')).to_have_text('학습에서 제외')
    page.locator('#toggle-preview-selection').click()
    expect(page.locator('#excluded-count')).to_have_text('1')
    assert not ui['posts']
    mock_screenshot(page, 'desktop_mock_selection.png')


def test_date_modes_same_scope_rescans_preserve_exclusions_but_new_scope_clears(ui):
    from playwright.sync_api import expect
    page = discover(ui)
    row(page, 0).locator('input').uncheck()
    page.locator('[data-mode="same_month"]').click()
    expect(page.locator('#selected-count')).to_have_text('2')
    expect(row(page, 0).locator('input')).not_to_be_checked()
    page.locator('[data-mode="any"]').click()
    expect(page.locator('#selected-count')).to_have_text('3')
    page.locator('[data-mode="same_day"]').click()
    page.locator('#capture-date').select_option('2026-08-19')
    expect(page.locator('#selected-count')).to_have_text('1')
    page.locator('#capture-date').select_option('2026-08-20')
    expect(row(page, 0).locator('input')).not_to_be_checked()
    with page.expect_response('**/api/discover?*'):
        page.locator('#rescan-date').click()
    expect(page.locator('#discover')).to_be_enabled()
    expect(row(page, 0).locator('input')).not_to_be_checked()
    last_discovery = [url for url in ui['gets'] if url.startswith('/api/discover?')][-1]
    assert parse_qs(urlsplit(last_discovery).query)['date'] == ['2026-08-20']
    with page.expect_response('**/api/discover?*'):
        page.locator('#discover').click()
    expect(page.locator('#discover')).to_be_enabled()
    expect(row(page, 0).locator('input')).not_to_be_checked()
    page.locator('#radius-value').fill('200')
    expect(page.locator('#save-plan')).to_be_disabled()
    expect(row(page, 0).locator('input')).to_be_disabled()
    page.locator('#discover').click()
    expect(page.locator('#selected-count')).to_have_text('2')
    expect(page.locator('#excluded-count')).to_have_text('0')
    expect(row(page, 0).locator('input')).to_be_checked()
    assert not ui['posts']


@pytest.mark.parametrize('control,path', [('save-plan', '/api/plans'), ('generate', '/api/jobs')])
@pytest.mark.parametrize('remove_sky,mask_dynamic', [(False, False), (False, True), (True, False), (True, True)])
def test_only_explicit_submit_sends_included_ids_exclusions_and_processing_options(ui, control, path, remove_sky, mask_dynamic):
    from playwright.sync_api import expect
    page = discover(ui)
    row(page, 0).locator('input').uncheck()
    provider_requests = [url for url in ui['gets'] if url.startswith(('/api/discover?', '/api/panorama?'))]
    page.locator('#remove-sky').set_checked(remove_sky)
    page.locator('#mask-dynamic').set_checked(mask_dynamic)
    assert provider_requests == [url for url in ui['gets'] if url.startswith(('/api/discover?', '/api/panorama?'))]
    assert not ui['posts']
    with page.expect_response('**' + path):
        page.locator('#' + control).click()
    expect(page.locator('#' + control)).to_be_enabled()
    assert len(ui['posts']) == 1
    posted_path, body = ui['posts'][0]
    assert posted_path == path
    assert body == dict(center=dict(lat=35.1, lng=128.1), radius_m=100,
                        generation_mode='multi_view',
                        training_steps=6000, resolution=768, max_splats=1000000,
                        capture_policy=dict(mode='same_day', value='2026-08-20'),
                        panorama_ids=[IDS[1]], excluded_panorama_ids=[IDS[0]],
                        processing_options=dict(remove_sky=remove_sky, mask_dynamic=mask_dynamic),
                        size_filter=dict(enabled=False,max_sigma_camera_radius_ratio=.5))


def test_processing_options_changes_hide_saved_link_without_implicit_requests(ui):
    from playwright.sync_api import expect
    page = discover(ui)
    provider_requests = [url for url in ui['gets'] if url.startswith(('/api/discover?', '/api/panorama?'))]
    for expected_posts, option in enumerate(('remove-sky', 'mask-dynamic'), start=1):
        with page.expect_response('**/api/plans'):
            page.locator('#save-plan').click()
        expect(page.locator('#saved-link')).to_be_visible()
        assert len(ui['posts']) == expected_posts
        page.locator('#' + option).uncheck()
        expect(page.locator('#saved-link')).to_be_hidden()
        expect(page.locator('#save-plan')).to_be_enabled()
        expect(page.locator('#generate')).to_be_enabled()
        assert len(ui['posts']) == expected_posts
        assert provider_requests == [url for url in ui['gets'] if url.startswith(('/api/discover?', '/api/panorama?'))]
    assert all(path == '/api/plans' for path, _ in ui['posts'])
    assert ui['posts'][0][1]['processing_options'] == dict(remove_sky=True, mask_dynamic=True)
    assert ui['posts'][1][1]['processing_options'] == dict(remove_sky=False, mask_dynamic=True)


@pytest.mark.parametrize('control,path', [('save-plan', '/api/plans'), ('generate', '/api/jobs')])
def test_processing_options_are_locked_only_while_explicit_submission_is_pending(ui, control, path):
    from playwright.sync_api import expect
    page = discover(ui)
    release = threading.Event()
    ui['post_release'] = release
    with page.expect_request('**' + path):
        page.locator('#' + control).click()
    expect(page.locator('#remove-sky')).to_be_disabled()
    expect(page.locator('#mask-dynamic')).to_be_disabled()
    assert len(ui['posts']) == 1
    release.set()
    expect(page.locator('#remove-sky')).to_be_enabled()
    expect(page.locator('#mask-dynamic')).to_be_enabled()
    expect(page.locator('#' + control)).to_be_enabled()
    assert len(ui['posts']) == 1


def test_all_excluded_disables_save_generate_and_select_all_restores(ui):
    from playwright.sync_api import expect
    page = discover(ui)
    page.locator('#clear-selection').click()
    expect(page.locator('#selected-count')).to_have_text('0')
    expect(page.locator('#excluded-count')).to_have_text('2')
    expect(page.locator('#save-plan')).to_be_disabled()
    expect(page.locator('#generate')).to_be_disabled()
    page.locator('[data-mode="same_month"]').click()
    expect(page.locator('#selected-count')).to_have_text('1')
    page.locator('[data-mode="same_day"]').click()
    expect(page.locator('#generate')).to_be_disabled()
    page.locator('#select-all').click()
    expect(page.locator('#selected-count')).to_have_text('2')
    expect(page.locator('#excluded-count')).to_have_text('0')
    expect(page.locator('#save-plan')).to_be_enabled()
    expect(page.locator('#generate')).to_be_enabled()
    assert not ui['posts']


def test_mobile_selection_preview_and_long_external_link_do_not_overflow(ui):
    from playwright.sync_api import expect
    page = discover(ui)
    page.set_viewport_size(dict(width=390, height=844))
    row(page, 0).locator('.station-open').click()
    expect(page.locator('#preview-meta')).to_contain_text('저해상도 미리보기')
    page.locator('#toggle-preview-selection').click()
    expect(page.locator('#excluded-count')).to_have_text('1')
    for control in ('remove-sky', 'mask-dynamic', 'save-plan', 'generate'):
        expect(page.locator('#' + control)).to_be_visible()
        expect(page.locator('#' + control)).to_be_enabled()
    expect(page.locator('#remove-sky')).to_be_checked()
    expect(page.locator('#mask-dynamic')).to_be_checked()
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
    assert not ui['posts']
    mock_screenshot(page, 'mobile_mock_selection.png')








def test_continuation_retry_preserves_selection_and_exact_discovery_date(ui):
    from playwright.sync_api import expect
    ui['discovery_pages'] = {
        '': dict(panoramas=ui['panoramas'][:2], date_options=[],
                 continuation='mock-next', truncated=True),
        'mock-next': dict(panoramas=ui['panoramas'][2:], date_options=[],
                         continuation=None, truncated=False),
    }
    page = discover(ui)
    row(page, 0).locator('input').uncheck()
    with page.expect_response('**/api/discover?*'):
        page.locator('#rescan-date').click()
    expect(page.locator('#load-more')).to_be_enabled()
    page.locator('[data-mode="same_month"]').click()
    expect(page.locator('#load-more')).to_be_disabled()
    page.locator('[data-mode="same_day"]').click()
    expect(page.locator('#load-more')).to_be_enabled()
    ui['discovery_codes']['mock-next'] = 503
    with page.expect_response('**/api/discover?*'):
        page.locator('#load-more').click()
    expect(page.locator('#map-status')).to_have_text('조회 실패 · 다시 시도해 주세요')
    expect(page.locator('#selected-count')).to_have_text('1')
    expect(page.locator('#excluded-count')).to_have_text('1')
    expect(page.locator('#load-more')).to_be_enabled()
    ui['discovery_codes']['mock-next'] = 200
    with page.expect_response('**/api/discover?*'):
        page.locator('#load-more').click()
    expect(page.locator('#load-more')).to_be_hidden()
    expect(row(page, 0).locator('input')).not_to_be_checked()
    expect(page.locator('#selected-count')).to_have_text('1')
    queries = [parse_qs(urlsplit(url).query) for url in ui['gets']
               if url.startswith('/api/discover?')]
    assert [query['date'] for query in queries[1:]] == [['2026-08-20']] * 3
    assert [query['continuation'] for query in queries[2:]] == [['mock-next']] * 2
    assert not ui['posts']


def test_partial_response_without_continuation_does_not_offer_invented_next_page(ui):
    from playwright.sync_api import expect
    ui['discovery_pages'] = {'': dict(panoramas=ui['panoramas'], date_options=[], truncated=True)}
    page = discover(ui)
    expect(page.locator('#load-more')).to_be_hidden()
    expect(page.locator('#warning-banner')).to_contain_text('일부 자료만 확인했습니다.')
    expect(page.locator('#map-status')).to_contain_text('일부 결과')
    assert len([url for url in ui['gets'] if url.startswith('/api/discover?')]) == 1
    assert not ui['posts']
