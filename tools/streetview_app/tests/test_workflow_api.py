import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from tools.streetview_app.tests.test_server import app_server, post, payload, freeze_selection
from tools.streetview_app.tests.test_ply_filters import inputs, wait_job
from tools.streetview_app.artifacts import sha


class UnrealFixture:
    def __init__(self): self.calls = []
    def capability(self): return dict(available=True, reason=None)
    def list(self): return []
    def start(self, resolved):
        self.calls.append(resolved)
        return dict(id='c'*32, status='queued')


def test_static_preset_no_implicit_execution_and_clone_profile(app_server, tmp_path):
    app, base = app_server
    image = tmp_path/'representative.png'; image.write_bytes(b'reference-only')
    connected = app.recipes.import_connected(dict(name='Preset', config=freeze_selection(payload(), app.provider),
        operator_settings={'workflow':'panorama_brush_refine'},
        reference_thumbnail=dict(path=str(image), sha256=sha(image))))
    (app.data_dir/'ui_preferences.json').write_text(json.dumps({'default_recipe_id':connected['id']}))
    for route in ['/api/recipes', f'/api/recipes/{connected["id"]}']:
        with urlopen(base+route) as response:
            data = json.load(response)
        if route.endswith('recipes'):
            assert data['default_recipe_id'] == connected['id']
            assert data['recipes'][0]['preview_url'].endswith('/preview')
    with urlopen(base+f'/api/recipes/{connected["id"]}/preview') as response:
        assert response.read() == b'reference-only'
    assert not app.jobs.list() and not app.filters.list()
    _, saved = post(base+'/api/recipes', dict(name='Clone', config=payload(), base_recipe_id=connected['id']), base)
    assert saved['recipe']['operator_settings'] == {'workflow':'panorama_brush_refine'}
    for invalid in [dict(name='Unsafe', config=payload(), operator_settings={'workflow':'arbitrary'}),
                    dict(name='Unsafe', config=payload(), base_recipe_id='not-a-recipe')]:
        with pytest.raises(HTTPError) as error: post(base+'/api/recipes', invalid, base)
        assert error.value.code == 400
    image.write_bytes(b'changed')
    with pytest.raises(HTTPError) as error: urlopen(base+f'/api/recipes/{connected["id"]}/preview')
    assert error.value.code == 409


def test_bound_artifact_explicit_unreal_and_tamper_rejection(app_server, tmp_path):
    app, base = app_server
    app.unreal = UnrealFixture()
    _, created = post(base+'/api/ply-filters', inputs(tmp_path), base)
    job = wait_job(app.filters, created['job']['id'])
    assert job['status'] == 'completed'
    action = dict(kind='filter', job_id=job['id'])
    with urlopen(base+'/api/unreal-opens') as response: assert json.load(response)['capability']['available']
    assert app.unreal.calls == []
    _, resolved = post(base+'/api/artifacts/resolve', action, base)
    assert sha(resolved['source_ply']) == job['artifact']['sha256']
    assert app.unreal.calls == []
    output_camera = Path(resolved['camera_json'])
    assert json.loads(output_camera.read_text())['ply_binding']['sha256'] == resolved['source_sha256']
    for invalid in [dict(action, source_ply='arbitrary'), dict(kind='filter', job_id='../outside')]:
        with pytest.raises(HTTPError) as error: post(base+'/api/unreal-opens', invalid, base)
        assert error.value.code == 400
    with pytest.raises(HTTPError) as error: post(base+'/api/unreal-opens', action, 'https://foreign.example')
    assert error.value.code == 403 and not app.unreal.calls
    _, response = post(base+'/api/unreal-opens', action, base)
    assert response['job']['status'] == 'queued' and len(app.unreal.calls) == 1
    changed = json.loads(output_camera.read_text()); changed['units'] = 'centimetres'
    output_camera.write_text(json.dumps(changed))
    with pytest.raises(HTTPError) as error: post(base+'/api/unreal-opens', action, base)
    assert error.value.code == 400 and len(app.unreal.calls) == 1
    app.filters.close()


def test_recipe_numbers_survive_provider_freeze(app_server):
    app, base = app_server
    _, response = post(base+'/api/recipes', dict(name='Numbers', config=payload(
        training_steps=12000, resolution=1280, max_splats=2000000)), base)
    settings = response['recipe']['settings']
    assert [settings[k] for k in ('training_steps','resolution','max_splats')] == [12000,1280,2000000]
    assert response['recipe']['scene_selection']['panoramas'][0]['captured_at'] == '2025-04-06 10:30:00'
