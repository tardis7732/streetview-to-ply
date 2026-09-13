"""Model-free fixtures verify cache bytes, geometry and actual model-file hashes."""
import copy
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from tools.streetview_engine import panorama_native_cache as module
from tools.streetview_engine.imaging import cube_camera_to_station_cv, sha256, write_json
from tools.streetview_engine.preprocess import model_provenance, mask_policy


def fixture(tmp_path,monkeypatch,*,verifier=False):
    monkeypatch.setattr(module,'_runtime_provenance',lambda options:dict(packages={'torch':'fixture-torch','transformers':'fixture-transformers'}))
    old=tmp_path/'old';new=tmp_path/'new';old.mkdir();new.mkdir()
    model=tmp_path/'model';model.mkdir();(model/'config.json').write_text('{}');(model/'model.safetensors').write_bytes(b'provenance-only fixture, never loaded')
    options=dict(backend='sam3',model_path=str(model));sky_options=dict(model_path=str(model))
    if verifier:
        directory=tmp_path/'verifier';directory.mkdir()
        labels={str(index):label for index,label in enumerate(['person','car','bus','truck','bicycle','motorcycle']+[f'fixture-{i}' for i in range(74)])}
        (directory/'config.json').write_text(json.dumps(dict(model_type='rt_detr_v2',id2label=labels)))
        (directory/'model.safetensors').write_bytes(b'fixture verifier weights')
        (directory/'preprocessor_config.json').write_text('{}')
        options['instance_verifier']=dict(backend='rtdetr_v2',model_path=str(directory))
    rows=[];evidence_rows=[]
    for index,face in enumerate(('F','R')):
        path=f'collection/{face}.png'
        for root in (old,new):
            (root/'collection').mkdir(exist_ok=True)
            Image.new('RGB',(16,16),(index*20,50,70)).save(root/path)
        row=dict(token='capture_'+face,pano_id='capture',station_id='station',face=face,w=16,h=16,
            fl_x=8.,fl_y=8.,cx=8.,cy=8.,camera_to_station_cv=cube_camera_to_station_cv(face).tolist(),
            original_file_path=path,original_sha256=sha256(old/path))
        rows.append(row)
        decoded=hashlib.sha256(np.asarray(Image.open(old/path)).tobytes()).hexdigest()
        meta=dict(source_rgb_sha256=decoded,image_size_hw=[16,16],instances=[])
        if verifier:meta['object_verification']=dict(source_rgb_sha256=decoded,image_size_hw=[16,16])
        stream=BytesIO();shape=(16,16)
        np.savez_compressed(stream,dynamic=np.zeros(shape,bool),sky_region=np.zeros(shape,bool),
            sky_high=np.zeros(shape,bool),ground=np.ones(shape,bool),group_dynamic=np.zeros(shape,np.float32),
            group_sky=np.zeros(shape,np.float32),instance_masks_shape=np.array([0,16,16],np.int32),
            instance_masks_packed=np.zeros((0,32),np.uint8),evidence_metadata_json=np.asarray(json.dumps(meta)),
            source_sha256=np.asarray(row['original_sha256']))
        rel=f'prepared/semantics/{row["token"]}.npz';(old/rel).parent.mkdir(parents=True,exist_ok=True);(old/rel).write_bytes(stream.getvalue())
        evidence_rows.append(dict(token=row['token'],path=rel,sha256=sha256(old/rel),source_sha256=row['original_sha256']))
    old_request=dict(root=str(old),rows=rows,sam=options,sky=sky_options,sky_instance_score_threshold=.5)
    request=dict(old_request,root=str(new),object_projection='native_cubes',stations=[],chunk_rows=32)
    write_json(old/'prepared/mask_request.json',old_request);write_json(new/'prepared/mask_request.json',request)
    sam=dict(model_provenance=model_provenance(options),code_sha256=sha256(module.ENGINE/'sam3_segmenter.py'),
        torch_version='fixture-torch',transformers_version='fixture-transformers')
    if verifier:
        sam['instance_verifier']=dict(model=dict(model_provenance=sam['model_provenance']['instance_verifier'],
            code_sha256=sha256(module.ENGINE/'object_verifier.py'),torch_version='fixture-torch',transformers_version='fixture-transformers'))
    manifest=dict(status='completed',request_sha256=sha256(old/'prepared/mask_request.json'),rows=evidence_rows,
        sam=sam,sky=dict(model_provenance=model_provenance(sky_options)),sam_policy=mask_policy(options),sky_policy=mask_policy(sky_options))
    write_json(old/'prepared/semantic_manifest.json',manifest)
    hashes={}
    for name in ('preprocess.py','sam3_segmenter.py','object_verifier.py','panorama_preprocess.py'):
        rel='code/tools/streetview_engine/'+name;(old/rel).parent.mkdir(parents=True,exist_ok=True)
        (old/rel).write_bytes((module.ENGINE/name).read_bytes());hashes[rel]=sha256(old/rel)
    write_json(old/'recipe.json',dict(code_sha256=hashes))
    cache=dict(source_job_dir=str(old),semantic_manifest_sha256=sha256(old/'prepared/semantic_manifest.json'),
        mask_request_sha256=sha256(old/'prepared/mask_request.json'))
    return old,new,cache,request,manifest


@pytest.mark.parametrize('verifier',[False,True])
def test_reuses_exact_npz_after_full_model_byte_identity_check(tmp_path,monkeypatch,verifier):
    old,new,cache,request,previous=fixture(tmp_path,monkeypatch,verifier=verifier)
    request_path=new/'prepared/mask_request.json'
    output=module.reuse_native_semantics(request_path,cache,new)
    assert output['request_sha256']==sha256(request_path) and output['request_sha256']!=previous['request_sha256']
    assert output['model_calls']==0 and output['inference_reused']
    for row in previous['rows']:
        assert (old/row['path']).read_bytes()==(new/row['path']).read_bytes()
    provenance=json.loads((new/output['native_semantic_cache']['path']).read_text())
    assert len(provenance['inputs'])==2 and not provenance['source_artifacts_modified']
    assert sha256(old/'prepared/semantic_manifest.json')==cache['semantic_manifest_sha256']
    assert module.reuse_native_semantics(request_path,cache,new)==output


@pytest.mark.parametrize('mutation',['whole_erp','row_order','rotation','original_hash','model_option','model_bytes','runtime','old_manifest','npz_hash'])
def test_rejects_changed_inputs_before_copying_semantics(tmp_path,monkeypatch,mutation):
    old,new,cache,request,manifest=fixture(tmp_path,monkeypatch)
    if mutation=='whole_erp':request['object_projection']='whole_erp'
    elif mutation=='row_order':request['rows'].reverse()
    elif mutation=='rotation':request['rows'][0]['camera_to_station_cv'][0][0]=2.
    elif mutation=='original_hash':request['rows'][0]['original_sha256']='0'*64
    elif mutation=='model_option':request['sam']['instance_score_threshold']=.2
    elif mutation=='model_bytes':(tmp_path/'model/model.safetensors').write_bytes(b'changed')
    elif mutation=='runtime':monkeypatch.setattr(module,'_runtime_provenance',lambda options:dict(packages={'torch':'different','transformers':'fixture-transformers'}))
    elif mutation=='old_manifest':cache['semantic_manifest_sha256']='0'*64
    elif mutation=='npz_hash':(old/manifest['rows'][0]['path']).write_bytes(b'changed')
    (new/'prepared/mask_request.json').write_text(json.dumps(request),encoding='utf8')
    with pytest.raises(ValueError):module.reuse_native_semantics(new/'prepared/mask_request.json',cache,new)
    assert not (new/'prepared/semantics').exists()


def test_npz_source_binding_checked_even_when_manifest_hash_is_updated(tmp_path,monkeypatch):
    old,new,cache,request,manifest=fixture(tmp_path,monkeypatch)
    path=old/manifest['rows'][0]['path']
    with np.load(path,allow_pickle=False) as source:values={key:source[key] for key in source.files}
    values['source_sha256']=np.asarray('0'*64)
    np.savez_compressed(path,**values)
    manifest['rows'][0]['sha256']=sha256(path)
    (old/'prepared/semantic_manifest.json').write_text(json.dumps(manifest),encoding='utf8')
    cache['semantic_manifest_sha256']=sha256(old/'prepared/semantic_manifest.json')
    with pytest.raises(ValueError,match='NPZ source'):module.reuse_native_semantics(new/'prepared/mask_request.json',cache,new)
    assert not (new/'prepared/semantics').exists()


def test_reuse_never_rewrites_source_job_manifest(tmp_path,monkeypatch):
    old,new,cache,request,manifest=fixture(tmp_path,monkeypatch)
    before=(old/'prepared/semantic_manifest.json').read_bytes()
    with pytest.raises(ValueError,match='separate destination job'):
        module.reuse_native_semantics(old/'prepared/mask_request.json',cache,old)
    assert (old/'prepared/semantic_manifest.json').read_bytes()==before
