"""Real tiny stage copies and preset round trips; no provider/model/GPU calls."""
import copy
import hashlib
import json
from pathlib import Path
import secrets

import pytest

from tools.streetview_app.jobs import JobManager
from tools.streetview_app.recipes import RecipeStore, file_hash
from tools.streetview_app.reuse import ReuseService, copy_files, record_stage_cache
from tools.streetview_engine.remote_worker import prepare_reuse
from tools.streetview_app.tests.test_jobs import backend, selection


@pytest.fixture(autouse=True)
def isolate_code_snapshot(tmp_path, monkeypatch):
    # The fixture's actual stage script remains hash-bound; unrelated agents
    # editing application code must not change a fixture's runtime identity.
    monkeypatch.setattr('tools.streetview_app.recipes.ROOT', tmp_path)


def test_named_recipe_separates_scene_and_reuses_settings_without_starting(tmp_path):
    manager = JobManager(tmp_path / 'jobs', backend(tmp_path))
    store = RecipeStore(tmp_path / 'recipes', manager.backend)
    config = selection(); config['training_steps'] = 321
    saved = store.save(dict(name='내 설정', config=config))
    assert saved['executable'] and manager.list() == []
    assert 'panorama_ids' not in saved['settings']
    assert saved['scene_selection']['panorama_ids'] == config['panorama_ids']
    other = selection(1)
    resolved = store.resolve(saved['id'], other)
    assert resolved['config']['panorama_ids'] == other['panorama_ids']
    assert resolved['config']['training_steps'] == 321
    job = manager.start(**resolved)
    completed = manager.wait(job['id'], 15)
    assert completed['status'] == 'completed', completed
    assert completed['execution_recipe']['content_sha256'] == saved['content_sha256']
    manager.close()


def test_recipe_rejects_changed_code_tampering_and_client_operator_commands(tmp_path):
    configured = backend(tmp_path)
    store = RecipeStore(tmp_path / 'recipes', configured)
    with pytest.raises(ValueError, match='only'):
        store.save(dict(name='invalid', config=selection(), operator_settings={'argv': ['bad']}))
    saved = store.save(dict(name='original', config=selection()))
    script = Path(configured.stages[0].argv[1])
    script.write_text(script.read_text(encoding='utf8') + '\n# code changed\n', encoding='utf8')
    assert not store.get(saved['id'])['executable']
    with pytest.raises(ValueError, match='changed'):
        store.resolve(saved['id'], selection())
    path = store.root / (saved['id'] + '.json')
    value = json.loads(path.read_text(encoding='utf8')); value['settings']['training_steps'] = 999
    path.write_text(json.dumps(value), encoding='utf8')
    with pytest.raises(ValueError, match='content changed'):
        store.get(saved['id'])


def test_verified_sfm_restart_skips_exact_prefix_and_preserves_source(tmp_path):
    manager = JobManager(tmp_path / 'jobs', backend(tmp_path))
    first = manager.wait(manager.start(selection())['id'], 15)
    assert first['status'] == 'completed', first
    service = ReuseService(manager)
    choices = service.inspect(first['id'])
    assert all(row['available'] for row in choices['stages'])
    second = manager.wait(service.start(first['id'], 'sfm')['id'], 15)
    assert second['status'] == 'completed', second
    assert [r['name'] for r in second['stages'] if r.get('reused')] == ['collect', 'preprocess']
    assert not (manager.root / second['id'] / 'logs/collect.log').exists()
    assert (manager.root / second['id'] / 'logs/sfm.log').is_file()
    assert second['artifact']['sha256'] == first['artifact']['sha256']
    original = manager.root / first['id'] / 'collect.json'
    original_bytes = original.read_bytes()
    (manager.root / second['id'] / 'collect.json').write_text('edited new copy')
    assert original.read_bytes() == original_bytes
    with pytest.raises(ValueError, match='identical settings'):
        service.start(first['id'], 'train', {'training_steps': 10})
    original.write_text('tampered source')
    assert not any(row['available'] for row in service.inspect(first['id'])['stages'])
    with pytest.raises(ValueError, match='artifact changed'):
        service.start(first['id'], 'sfm')
    manager.close()


def test_legacy_job_never_synthesizes_verified_cache(tmp_path):
    manager = JobManager(tmp_path / 'jobs', backend(tmp_path))
    first = manager.wait(manager.start(selection())['id'], 15)
    del manager._jobs[first['id']]['cache_manifest']
    result = ReuseService(manager).inspect(first['id'])
    assert all(not row['available'] and 'no complete-stage' in row['reason'] for row in result['stages'])
    manager.close()


def test_cloud_prefix_clone_verifies_owner_settings_code_and_full_bytes(tmp_path):
    source = tmp_path / secrets.token_hex(16); target = tmp_path / secrets.token_hex(16)
    source.mkdir(); target.mkdir()
    token = secrets.token_hex(32)
    write = lambda path, obj: path.write_text(json.dumps(obj), encoding='utf8')
    (source / 'collection').mkdir(); (source / 'collection/image.bin').write_bytes(b'pixel bytes')
    (source / 'collection/manifest.json').write_text('{"complete":true}', encoding='utf8')
    binding = record_stage_cache(source, 'collect', ['collection/manifest.json'])
    cache = json.loads((source / 'cache_manifest.json').read_text(encoding='utf8'))
    for directory in (source, target):
        write(directory / 'config.json', {'same': 'scene'})
        write(directory / 'settings.json', {'same': 'algorithm'})
        (directory / 'code').mkdir(); (directory / 'code/a.py').write_text('pass\n', encoding='utf8')
    codes = {'code/a.py': file_hash(source / 'code/a.py')}
    write(source / 'recipe.json', {'code_sha256': codes})
    write(source / 'owner.json', {'token_sha256': hashlib.sha256(token.encode()).hexdigest()})
    record = dict(name='collect', status='completed', outputs=[{'path': 'collection/manifest.json'}])
    write(source / 'remote_state.json', dict(status='completed', stages=[record], cache_manifest=binding))
    request = dict(source_job_id=source.name, source_owner_token=token, from_stage='preprocess',
        reused_stages=['collect'], cache_manifest_sha256=binding['sha256'], cache_stages=cache['stages'],
        files=cache['stages']['collect'])
    recipe = dict(code_sha256=codes, stages=[{'name': 'collect'}, {'name': 'preprocess'}], reuse=request)
    invalid = copy.deepcopy(recipe); invalid['reuse']['source_owner_token'] = '0' * 64
    with pytest.raises(PermissionError):
        prepare_reuse(target, invalid)
    (source / 'collection/image.bin').write_bytes(b'bad bytes!!')
    with pytest.raises(ValueError, match='artifact changed'):
        prepare_reuse(target, recipe)
    assert not (target / 'collection').exists()
    (source / 'collection/image.bin').write_bytes(b'pixel bytes')
    assert prepare_reuse(target, recipe) == {'collect': record}
    (target / 'collection/image.bin').write_bytes(b'new copy changed')
    assert (source / 'collection/image.bin').read_bytes() == b'pixel bytes'
