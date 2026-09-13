"""Tiny cached-source fixtures; no network, model masks or invented scene data."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from tools.streetview_engine import collection
from tools.streetview_engine.imaging import FACES, sha256
from tools.streetview_engine.native_collection import run, validate_source
from tools.streetview_app.tests.test_jobs import selection


def package(tmp_path):
    source = tmp_path/'source'; source.mkdir()
    config = selection(1); key = config['panorama_ids'][0]
    raw = dict(id=key, latitude=10., longitude=20., camera_angle=[0, 12., 0], altitude=3.,
        proj_type='cubic', info=dict(photodate='2024-06-12T09:30:15'))
    (source/'metadata.json').write_text(json.dumps(raw), encoding='utf8')

    def record(name, **extras):
        path=source/name
        return dict(file_path=name, sha256=sha256(path), bytes=path.stat().st_size, **extras)

    faces={}
    for index,face in enumerate(FACES):
        name=face+'.jpg'; Image.new('RGB',(16,16),(index*30,54,91)).save(source/name)
        with Image.open(source/name) as image:
            decoded=hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
        faces[face]=record(name,w=16,h=16,decoded_rgb_sha256=decoded)
    Image.new('L',(16,16),255).save(source/'valid.png')
    faces['D'].update(valid_mask=record('valid.png',w=16,h=16),
        valid_mask_provenance='Declared original provider nadir coverage; not a dynamic-object mask')
    Image.new('RGB',(192,32),(12,34,56)).save(source/'strip.jpg')
    row=dict(pano_id=key, lat=10., lng=20., captured_at=raw['info']['photodate'],
        metadata_source=record('metadata.json'), faces=faces,
        raw_strip=record('strip.jpg',w=192,h=32,face_order=list(FACES)),
        image_history=['Fixture: earlier 32px strip face to 16px cube JPEG; importer must not resample it'])
    doc=dict(schema_version=1,kind='native_cube_source',status='complete',provider='naver',face_order=list(FACES),stations=[row])
    path=source/'manifest.json'; path.write_text(json.dumps(doc),encoding='utf8')
    settings=dict(collection=dict(native_source=dict(manifest_path=str(path),manifest_sha256=sha256(path))))
    return source, config, settings, doc


def update_manifest(source,settings,doc):
    path=source/'manifest.json';path.write_text(json.dumps(doc),encoding='utf8')
    settings['collection']['native_source']['manifest_sha256']=sha256(path)


def test_import_preserves_exact_encoded_pixels_history_coverage_and_resolves_standard_collection(tmp_path,monkeypatch):
    source, config, settings, doc=package(tmp_path); destination=tmp_path/'job'
    monkeypatch.setattr(collection,'_request',lambda *args,**kwargs:pytest.fail('Native import must not request imagery'))
    actual=collection.run(config,destination,settings)
    assert actual['native_face_size']==16 and actual['original_strip_face_sizes']==[32]
    assert actual['transport']['network_requests']==0 and actual['import_resampling'] is False
    for face,item in actual['stations'][0]['faces'].items():
        assert (destination/item['file_path']).read_bytes()==(source/doc['stations'][0]['faces'][face]['file_path']).read_bytes()
        assert item['image_history']==doc['stations'][0]['image_history']
    down=actual['stations'][0]['faces']['D']
    assert (destination/down['valid_mask_path']).read_bytes()==(source/'valid.png').read_bytes()
    assert run(config,destination,settings)==actual
    changed=dict(config,radius_m=110)
    with pytest.raises(ValueError,match='Existing collection differs'):
        run(changed,destination,settings)
    assert json.loads((destination/'collection/manifest.json').read_text())==actual


@pytest.mark.parametrize('mutation', ['order','face_hash','decoded','size','metadata_id','gps','capture','history','traversal','coverage'])
def test_invalid_package_fails_before_output_creation(tmp_path,mutation):
    source,config,settings,doc=package(tmp_path); row=doc['stations'][0]
    if mutation=='order': row['pano_id']='another-id'
    elif mutation=='face_hash': row['faces']['F']['sha256']='0'*64
    elif mutation=='decoded': row['faces']['F']['decoded_rgb_sha256']='0'*64
    elif mutation=='size': row['faces']['F']['w']=17
    elif mutation=='metadata_id':
        path=source/'metadata.json'; value=json.loads(path.read_text());value['id']='other'
        path.write_text(json.dumps(value));row['metadata_source'].update(sha256=sha256(path),bytes=path.stat().st_size)
    elif mutation=='gps': row['lat']+=.001
    elif mutation=='capture': row['captured_at']='2024-06-13'
    elif mutation=='history': row['image_history']=[]
    elif mutation=='traversal': row['faces']['F']['file_path']='../F.jpg'
    elif mutation=='coverage':
        path=source/'valid.png';Image.new('L',(16,16),128).save(path)
        row['faces']['D']['valid_mask'].update(sha256=sha256(path),bytes=path.stat().st_size)
    update_manifest(source,settings,doc)
    destination=tmp_path/'job'
    with pytest.raises(ValueError):run(config,destination,settings)
    assert not destination.exists()


def test_frozen_position_and_import_manifest_hash_are_bound(tmp_path):
    source,config,settings,doc=package(tmp_path)
    settings['collection']['native_source']['manifest_sha256']='0'*64
    with pytest.raises(ValueError,match='manifest hash'):validate_source(config,tmp_path/'job',settings)
    update_manifest(source,settings,doc)
    config['panoramas'][0]['lat']+=.0001
    with pytest.raises(ValueError,match='frozen capture position'):validate_source(config,tmp_path/'job',settings)


def test_completed_import_tamper_is_rejected_without_replacing_file(tmp_path):
    source,config,settings,doc=package(tmp_path);job=tmp_path/'job'
    actual=run(config,job,settings);target=job/actual['stations'][0]['faces']['F']['file_path']
    target.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Existing imported artifact differs'):run(config,job,settings)
    assert target.read_bytes()==b'changed'
