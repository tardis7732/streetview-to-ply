"""Isolated UI actions; only a static preset image, never an intermediate comparison."""
import json
import os
import pytest
from tools.streetview_app.tests.test_selection_gui import browser, ui, load, discover, PNG

pytestmark = pytest.mark.skipif(os.environ.get('STREETVIEW_SELECTION_BROWSER') != '1', reason='Browser opt-in')


def mock_workflow(ui):
    page = ui['page']
    preset = dict(id='a'*32, name='L_91 테스트', executable=True,
        preview_url='/mock-face.png', settings=dict(generation_mode='multi_view', training_steps=6000,
        resolution=1280, max_splats=2000000, processing_options=dict(remove_sky=False, mask_dynamic=True),
        size_filter=dict(enabled=True,max_sigma_camera_radius_ratio=.5)))
    page.route('**/api/recipes', lambda route: route.fulfill(json=dict(recipes=[preset], default_recipe_id=preset['id'])))
    page.route('**/api/unreal-opens', lambda route: route.fulfill(json=dict(capability=dict(available=True),jobs=[])))
    return preset


def test_representative_image_locked_recipe_and_explicit_generation(ui):
    from playwright.sync_api import expect
    preset = mock_workflow(ui)
    page = load(ui)
    expect(page.locator('#recipe-select')).to_have_value(preset['id'])
    expect(page.locator('#recipe-image')).to_be_visible()
    expect(page.locator('#training-resolution')).to_have_value('1280')
    expect(page.locator('#training-resolution')).to_be_disabled()
    expect(page.locator('#remove-sky')).to_be_disabled()
    assert ui['posts'] == []
    assert page.locator('canvas[data-comparison], iframe[data-comparison]').count() == 0
    discover(ui)
    assert ui['posts'] == []
    page.locator('#generate').click()
    expect(page.locator('#generate')).to_be_enabled()
    jobs = [body for url, body in ui['posts'] if url == '/api/jobs']
    assert len(jobs) == 1
    assert jobs[0]['recipe_id'] == preset['id'] and jobs[0]['resolution'] == 1280


def test_custom_settings_unlock_and_crop_default_off(ui):
    from playwright.sync_api import expect
    mock_workflow(ui)
    page = load(ui)
    expect(page.locator('#recipe-select')).to_have_value('a'*32)
    page.locator('#recipe-select').select_option('')
    expect(page.locator('#training-resolution')).to_be_enabled()
    expect(page.locator('#remove-sky')).to_be_enabled()
    expect(page.locator('#ply-crop-enabled')).not_to_be_checked()
    expect(page.locator('#ply-crop-percent')).to_be_disabled()
    page.locator('#ply-crop-enabled').check()
    expect(page.locator('#ply-crop-percent')).to_be_enabled()
    assert ui['posts'] == []


def test_completed_job_actions_require_explicit_buttons(ui):
    from playwright.sync_api import expect
    mock_workflow(ui)
    page=ui['page']; actions=[]; job_id='b'*32
    page.route('**/api/status',lambda route:route.fulfill(json=dict(capabilities=dict(generate=True),
        jobs=[dict(id=job_id,status='completed',artifact=dict(path='export/scene.ply'),config={},stages=[])])))
    def reuse(route):
        actions.append((route.request.method,route.request.url,route.request.post_data_json))
        route.fulfill(json=dict(stages=[dict(from_stage='train',available=True)],job=dict(id='c'*32,status='queued')))
    page.route('**/api/jobs/*/reuse',reuse)
    def resolved(route):
        actions.append((route.request.method,route.request.url,route.request.post_data_json))
        route.fulfill(json=dict(source_ply='C:/verified/scene.ply',camera_json='C:/verified/cameras.json'))
    page.route('**/api/artifacts/resolve',resolved)
    page=load(ui)
    card=page.locator(f'[data-job-id="{job_id}"]')
    expect(card.get_by_text('단계 재사용',exact=True)).to_be_visible()
    assert not actions and not ui['posts']
    card.get_by_text('단계 재사용',exact=True).click()
    expect(page.locator('#reuse-stage')).to_have_value('train')
    assert [action[0] for action in actions]==['GET']
    page.locator('#reuse-start').click()
    expect(page.locator('#reuse-dialog')).not_to_be_visible()
    assert actions[-1][0]=='POST' and actions[-1][2]==dict(from_stage='train')
    card.get_by_text('범위·크기 정리',exact=True).click()
    expect(page.locator('#ply-filter-source')).to_have_value('C:/verified/scene.ply')
    expect(page.locator('#ply-size-enabled')).not_to_be_checked()
    assert all(url!='/api/ply-filters' for url,body in ui['posts'])
