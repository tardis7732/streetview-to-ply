"""Real native-grid cache contract with small deterministic inference fixtures."""
import copy
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pytest

from tools.streetview_engine import panorama_input_cache as cache
from tools.streetview_engine import panorama_preprocess as panorama
from tools.streetview_engine.imaging import fingerprint, sha256, write_json
from tools.streetview_engine.native_collection import run as collect
from tools.streetview_engine.tests.test_native_collection import package
from tools.streetview_engine.tests.test_panorama_preprocess import deterministic_worker
from tools.streetview_app.reuse import stage_inventory


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf8')


def seal(source, *, status='failed'):
    records = []
    inventories = {}
    for stage, folder in cache.STAGES.items():
        inventories[stage] = stage_inventory(source, stage, [folder+'/manifest.json'])
        records.append(dict(name=stage, status='completed', exit_code=0,
            outputs=[dict(path=folder+'/manifest.json', **inventories[stage][folder+'/manifest.json'])]))
    save(source/'cache_manifest.json', dict(schema_version=1, stages=inventories))
    records.append(dict(name='sfm', status='failed', exit_code=1))
    save(source/'remote_state.json', dict(id=source.name, status=status, stages=records,
        cache_manifest=dict(path='cache_manifest.json', sha256=sha256(source/'cache_manifest.json'))))


@pytest.fixture
def source_fixture(tmp_path, monkeypatch):
    _, config, settings, _ = package(tmp_path)
    source = tmp_path/'failed_job'
    settings.update(workflow='panorama_brush_refine', panorama_preprocess=dict(
        sam_segmentation=dict(backend='sam3', instance_verifier={'fixture': True}, python_executable='fixture'),
        sky_segmentation={'model': 'fixture'}, flux=dict(model_receipt_path=str(tmp_path/'model.json'),
            model_revision='fixture-revision', python_executable='fixture', prompt='Remove fixtures', erp_width=128)))
    save(tmp_path/'model.json', {'fixture': True})
    settings['panorama_preprocess']['flux']['model_receipt_sha256'] = sha256(tmp_path/'model.json')
    provenance = dict(files_sha256='fixture-model-hash')
    monkeypatch.setattr(cache, 'model_provenance', lambda settings: provenance)
    def worker(operation, request_path, runtime):
        deterministic_worker(operation, request_path, runtime)
        request = json.loads(request_path.read_text())
        if operation == 'masks':
            path = source/'prepared/semantic_manifest.json'; report = json.loads(path.read_text())
            for row in report['rows']:
                with np.load(source/row['path'], allow_pickle=False) as evidence:
                    arrays = {key: evidence[key] for key in evidence.files}
                arrays['source_sha256'] = np.array(row['source_sha256'])
                stream = BytesIO(); np.savez_compressed(stream, **arrays)
                (source/row['path']).write_bytes(stream.getvalue()); row['sha256'] = sha256(source/row['path'])
            for key in ('sam', 'sky'):
                report[key] = dict(model_provenance=provenance)
                report[key+'_policy'] = cache.mask_policy(request[key])
            save(path, report)
        else:
            path=source/'prepared/generation_manifest.json'; report=json.loads(path.read_text())
            report.update(model_revision='fixture-revision', model_receipt_sha256=sha256(tmp_path/'model.json'))
            for row, requested in zip(report['rows'], request['rows']): row['original_sha256']=requested['original_sha256']
            save(path, report)
    collect(config, source, settings)
    panorama.run(config, source, settings, _worker=worker)
    save(source/'config.json', config); save(source/'settings.json', settings)
    code_name = 'code/tools/streetview_engine/panorama_preprocess.py'
    code = source/code_name; code.parent.mkdir(parents=True); code.write_bytes(Path(panorama.__file__).read_bytes())
    save(source/'recipe.json', dict(code_sha256={code_name: sha256(code)}))
    seal(source)
    settings['input_cache'] = dict(source_job_dir=str(source))
    return source, config, settings


def test_failed_job_completed_inputs_reused_without_inference_or_geometry(source_fixture, tmp_path):
    source, config, settings = source_fixture
    (source/'sfm').mkdir(); (source/'sfm/database.db').write_bytes(b'not copied')
    before = {p.relative_to(source).as_posix(): sha256(p) for p in source.rglob('*') if p.is_file()}
    destination = tmp_path/'fresh_job'
    receipt = cache.reuse_inputs(config, destination, settings)
    assert receipt['kind']==cache.KIND and receipt['source_job_status']=='failed'
    assert receipt['hardlinked_files']==0 and receipt['model_calls']==0 and receipt['copied_files']>30
    assert not (destination/'sfm').exists() and not (destination/'code').exists()
    assert cache.cached_stage(config, destination, settings, 'preprocess') == json.loads((source/'prepared/manifest.json').read_text())
    assert cache.reuse_inputs(config, destination, settings)==receipt
    assert before == {p.relative_to(source).as_posix(): sha256(p) for p in source.rglob('*') if p.is_file()}
    frame=json.loads((source/'prepared/manifest.json').read_text())['frames'][0]
    (destination/frame['file_path']).write_bytes(b'changed only new job')
    assert sha256(source/frame['file_path']) == frame['source_sha256']
    with pytest.raises(ValueError, match='size/hash'):
        cache.cached_stage(config, destination, settings, 'preprocess')


@pytest.mark.parametrize('change', ['config', 'collection_option', 'panorama_option', 'active', 'incomplete_stage', 'inventory_binding', 'file', 'extra_file', 'code'])
def test_bad_cache_rejected_before_destination_created(source_fixture, tmp_path, change):
    source, config, settings = source_fixture
    if change=='config': config=copy.deepcopy(config); config['radius_m']+=1
    elif change=='collection_option': settings['collection']['max_workers']=3
    elif change=='panorama_option': settings['panorama_preprocess']['feather_at_2048']=1
    elif change=='active': seal(source,status='running')
    elif change=='incomplete_stage':
        state=json.loads((source/'remote_state.json').read_text());state['stages'][1]['status']='failed';save(source/'remote_state.json',state)
    elif change=='inventory_binding':
        state=json.loads((source/'remote_state.json').read_text());state['cache_manifest']['sha256']='0'*64;save(source/'remote_state.json',state)
    elif change=='file': (source/'prepared/mask_request.json').write_bytes(b'changed')
    elif change=='extra_file': (source/'prepared/unbound.txt').write_bytes(b'unbound')
    elif change=='code': (source/'code/tools/streetview_engine/panorama_preprocess.py').write_bytes(b'changed code')
    destination=tmp_path/'fresh_job'
    with pytest.raises(ValueError):cache.reuse_inputs(config,destination,settings)
    assert not destination.exists()


@pytest.mark.parametrize('change', ['camera', 'source', 'mask_policy', 'model'])
def test_resealed_internally_inconsistent_cache_still_rejected(source_fixture, tmp_path, change, monkeypatch):
    source, config, settings=source_fixture
    path=source/'prepared/manifest.json';manifest=json.loads(path.read_text());frame=manifest['frames'][0]
    if change=='camera': frame['fl_x']+=1; save(path,manifest)
    elif change=='source': frame['semantic_source_sha256']='0'*64;save(path,manifest)
    elif change=='mask_policy':
        from PIL import Image
        Image.new('L',(frame['w'],frame['h']),255).save(source/frame['sfm_mask_path'])
        frame['sfm_mask_sha256']=sha256(source/frame['sfm_mask_path']);save(path,manifest)
    elif change=='model': monkeypatch.setattr(cache,'model_provenance',lambda settings:dict(files_sha256='different'))
    seal(source)
    with pytest.raises(ValueError):cache.validate_cache(config,source,settings)


def test_sfm_recovery_settings_do_not_change_preprocessing_identity(source_fixture, tmp_path):
    source, config, settings=source_fixture
    settings['sfm']=dict(recovery_rounds=4,recovery_seconds=600,recovery_match_neighbors=32)
    result=cache.validate_cache(config,source,settings)
    assert result['config_sha256']==fingerprint(config)
    existing=tmp_path/'existing';(existing/'prepared').mkdir(parents=True)
    (existing/'prepared/keep.txt').write_bytes(b'keep')
    with pytest.raises(ValueError,match='overwrite'):cache.reuse_inputs(config,existing,settings)
    assert (existing/'prepared/keep.txt').read_bytes()==b'keep'


def test_public_input_cache_routes_panorama_to_native_grid_import(source_fixture, tmp_path, monkeypatch):
    from tools.streetview_engine import input_cache
    source, config, settings=source_fixture
    monkeypatch.setattr(input_cache,'validate_cache',lambda *args:pytest.fail('Must not use legacy 1024 tile validation'))
    receipt=input_cache.reuse_inputs(config,tmp_path/'public_job',settings)
    assert receipt['kind']==cache.KIND and receipt['source_job_status']=='failed'
    assert receipt['collection_manifest_sha256']==sha256(source/'collection/manifest.json')
