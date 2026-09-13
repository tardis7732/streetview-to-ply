"""Verified reuse of complete native-cube semantic evidence, without inference."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import stat

import numpy as np
from PIL import Image

from .imaging import inside, sha256, write_json
from .native_collection import _ordinary, _copy, _hash
from .preprocess import model_provenance, mask_policy
from .preprocess_identity import _runtime_provenance

ENGINE=Path(__file__).parent
ROW_FIELDS=('token','pano_id','station_id','face','w','h','fl_x','fl_y','cx','cy','camera_to_station_cv','original_sha256')


def _same(a,b):
    # JSON model metadata can stringify numeric taxonomy keys during recording.
    def canonical(value):
        return json.dumps(json.loads(json.dumps(value,allow_nan=False)),sort_keys=True,separators=(',',':'))
    return canonical(a)==canonical(b)


def _file(root,name,digest=None):
    if not isinstance(name,str) or '\\' in name or ':' in name:
        raise ValueError('Cache artifacts require portable relative paths')
    inside(root,name)
    path=_ordinary(root/name)
    if digest is not None and sha256(path)!=_hash(digest):
        raise ValueError('Native semantic cache file hash differs: '+name)
    return path


def _directory(path):
    value=Path(path).absolute()
    for current in (value,*value.parents):
        info=current.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info,'st_file_attributes',0)&0x400:
            raise ValueError('Native semantic cache forbids linked directories')
    if not value.is_dir():raise ValueError('Native semantic cache source directory is missing')
    return value


def _functions(path):
    tree=ast.parse(path.read_text(encoding='utf8'))
    result={}
    guard=ast.dump(ast.parse("if request.get('object_projection', 'native_cubes') == 'whole_erp':\n return panorama_mask_worker(request_path)").body[0],include_attributes=False)
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in ('_rgb','gated_sky','mask_worker'):
            if node.name=='mask_worker':
                node.body=[statement for statement in node.body if ast.dump(statement,include_attributes=False)!=guard]
            result[node.name]=ast.dump(node,include_attributes=False)
    if set(result)!={'_rgb','gated_sky','mask_worker'}:raise ValueError('Cannot verify native semantic worker code')
    return result


def _validate_models(request,manifest,source):
    provenance={}; runtime=_runtime_provenance(request['sam'])
    recipe=_file(source,'recipe.json')
    codes=json.loads(recipe.read_text(encoding='utf8')).get('code_sha256',{})
    code_checks={}
    for name in ('preprocess.py','sam3_segmenter.py','object_verifier.py','panorama_preprocess.py'):
        relative='code/tools/streetview_engine/'+name
        old=_file(source,relative,codes.get(relative))
        if not codes.get(relative):raise ValueError('Native cache source code lacks its job hash binding')
        current=ENGINE/name
        if name=='panorama_preprocess.py':
            if _functions(old)!=_functions(current):raise ValueError('Native semantic inference/helper code changed')
            scope='native worker, RGB decoder and sky gate AST; only ERP dispatch branch excluded'
        else:
            if sha256(old)!=sha256(current):raise ValueError('Native semantic dependency code changed: '+name)
            scope='complete module bytes'
        code_checks[name]=dict(source_sha256=sha256(old),current_sha256=sha256(current),scope=scope)
    for key in ('sam','sky'):
        recorded=manifest.get(key)
        if not isinstance(recorded,dict):raise ValueError('Native cache lacks model metadata')
        fresh=model_provenance(request[key])
        if not _same(recorded.get('model_provenance'),fresh):raise ValueError('Native cached model/processor/verifier bytes changed: '+key)
        if not _same(manifest.get(key+'_policy'),mask_policy(request[key])):raise ValueError('Native cached mask policy differs')
        provenance[key]=fresh
    sam=manifest['sam']
    if sam.get('code_sha256')!=sha256(ENGINE/'sam3_segmenter.py'):raise ValueError('Recorded SAM code hash changed')
    records=[sam]
    if request['sam'].get('instance_verifier') is not None:
        verifier=sam.get('instance_verifier',{}).get('model',{})
        if verifier.get('code_sha256')!=sha256(ENGINE/'object_verifier.py'):raise ValueError('Recorded verifier code hash changed')
        if not _same(verifier.get('model_provenance'),provenance['sam'].get('instance_verifier')):
            raise ValueError('Recorded verifier nested provenance differs')
        records.append(verifier)
    for record in records:
        for package,field in [('torch','torch_version'),('transformers','transformers_version')]:
            if not record.get(field) or record[field]!=runtime['packages'].get(package):
                raise ValueError('Configured native inference runtime version changed: '+package)
    return dict(model_provenance=provenance,runtime=runtime,code_checks=code_checks,
        source_recipe_sha256=sha256(recipe),runtime_scope='torch/transformers recorded versions; other package versions are current observations only')


def validate_native_semantics(request_path,cache,destination_root):
    """Full read-only verification before copying any source artifact."""
    if not isinstance(cache,dict) or set(cache)!={'source_job_dir','semantic_manifest_sha256','mask_request_sha256'}:
        raise ValueError('Native cache requires explicit source job and two manifest hashes')
    if not isinstance(cache['source_job_dir'],str) or not Path(cache['source_job_dir']).is_absolute():
        raise ValueError('Native semantic source job must be an absolute directory')
    source=_directory(cache['source_job_dir']);destination=Path(destination_root).resolve()
    request_path=_ordinary(request_path);request=json.loads(request_path.read_text(encoding='utf8'))
    if Path(request.get('root','')).resolve()!=destination:raise ValueError('New mask request root differs from destination')
    old_request_path=_file(source,'prepared/mask_request.json',cache['mask_request_sha256'])
    old_manifest_path=_file(source,'prepared/semantic_manifest.json',cache['semantic_manifest_sha256'])
    old_request=json.loads(old_request_path.read_text(encoding='utf8'));manifest=json.loads(old_manifest_path.read_text(encoding='utf8'))
    if Path(old_request.get('root','')).resolve()!=source.resolve():raise ValueError('Cached mask request root differs from owned source job')
    if any(value.get('object_projection','native_cubes')!='native_cubes' for value in (request,old_request)) or manifest.get('object_inference_projection','native_cubes')!='native_cubes' or manifest.get('erp_rows'):
        raise ValueError('Native semantic cache cannot be used for whole-ERP object inference')
    if manifest.get('status')!='completed' or manifest.get('request_sha256')!=sha256(old_request_path):
        raise ValueError('Native semantic manifest did not complete its bound request')
    for key in ('sam','sky','sky_instance_score_threshold'):
        if key not in request or not _same(request[key],old_request.get(key)):
            raise ValueError('Native semantic request model/settings differ: '+key)
    rows=request.get('rows');old_rows=old_request.get('rows');records=manifest.get('rows')
    if any(not isinstance(value,list) or not value or any(not isinstance(row,dict) for row in value) for value in (rows,old_rows,records)):
        raise ValueError('Native semantic requests need complete row rosters')
    tokens=[row.get('token') for row in rows]
    if len(set(tokens))!=len(tokens) or [row.get('token') for row in old_rows]!=tokens or [row.get('token') for row in records]!=tokens:
        raise ValueError('Native semantic row roster/order differs')
    model_checks=_validate_models(request,manifest,source)
    files=[];inputs=[];seen=set()
    for row,previous,record in zip(rows,old_rows,records):
        if any(key not in row or key not in previous or not _same(row[key],previous[key]) for key in ROW_FIELDS):
            raise ValueError('Native semantic row calibration/identity differs')
        width,height=row['w'],row['h']
        if type(width) is not int or type(height) is not int or width<=0 or height<=0:raise ValueError('Invalid native semantic row size')
        original=_file(source,previous['original_file_path'],row['original_sha256'])
        current=_file(destination,row['original_file_path'],row['original_sha256'])
        with Image.open(original) as image:
            if image.size!=(width,height) or image.mode!='RGB':raise ValueError('Original native semantic image format changed')
            pixels=np.asarray(image);decoded=hashlib.sha256(pixels.tobytes()).hexdigest()
        if record.get('source_sha256')!=row['original_sha256']:raise ValueError('Native semantic manifest source binding differs')
        path=_file(source,record.get('path'),record.get('sha256'))
        if path.relative_to(source).parts[:2]!=('prepared','semantics') or path.suffix!='.npz' or record['path'] in seen:
            raise ValueError('Only distinct prepared/semantics NPZ files may be reused')
        seen.add(record['path'])
        with np.load(path,allow_pickle=False) as evidence:
            if str(evidence['source_sha256'].item())!=row['original_sha256']:raise ValueError('Native NPZ source binding differs')
            if 'object_inference_projection' in evidence and str(evidence['object_inference_projection'].item())!='native_cubes':
                raise ValueError('Whole-ERP evidence cannot be reused as native cube evidence')
            for key in ('dynamic','sky_region','sky_high','ground'):
                plane=evidence[key]
                if plane.shape!=(height,width) or plane.dtype!=bool:raise ValueError('Native semantic binary plane format differs')
            for key in ('group_dynamic','group_sky'):
                plane=evidence[key]
                if plane.shape!=(height,width) or not np.isfinite(plane).all() or np.any((plane<0)|(plane>1)):
                    raise ValueError('Native semantic score plane format differs')
            metadata=json.loads(evidence['evidence_metadata_json'].item())
            if metadata.get('source_rgb_sha256')!=decoded or metadata.get('image_size_hw')!=[height,width]:
                raise ValueError('Native evidence decoded RGB/size differs')
            dimensions=np.asarray(evidence['instance_masks_shape']);packed=evidence['instance_masks_packed']
            if dimensions.shape!=(3,) or not np.issubdtype(dimensions.dtype,np.integer) or list(dimensions[1:])!=[height,width] or dimensions[0]!=len(metadata.get('instances',[])) or packed.dtype!=np.uint8 or packed.shape!=(dimensions[0],(height*width+7)//8):
                raise ValueError('Native packed instance evidence format differs')
            if request['sam'].get('instance_verifier') is not None:
                verification=metadata.get('object_verification') or {}
                if verification.get('source_rgb_sha256')!=decoded or verification.get('image_size_hw')!=[height,width]:
                    raise ValueError('Native verifier evidence RGB binding differs')
        files.append(dict(source_path=path,destination_path=record['path'],sha256=record['sha256'],bytes=path.stat().st_size))
        inputs.append(dict(token=row['token'],source_file_path=previous['original_file_path'],current_file_path=row['original_file_path'],
            source_sha256=row['original_sha256'],decoded_rgb_sha256=decoded))
    return dict(source=source,request_path=request_path,source_request=old_request_path,source_manifest=old_manifest_path,
        manifest=manifest,files=files,inputs=inputs,model_checks=model_checks)


def reuse_native_semantics(request_path,cache,destination_root):
    checked=validate_native_semantics(request_path,cache,destination_root)
    root=Path(destination_root).resolve()
    source=checked['source'].resolve()
    if root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError('Native semantic reuse requires a separate destination job')
    for item in checked['files']:
        _copy(item['source_path'],inside(root,item['destination_path']),item['sha256'])
    source_request_rel='prepared/native_semantic_cache/source_mask_request.json'
    source_manifest_rel='prepared/native_semantic_cache/source_semantic_manifest.json'
    _copy(checked['source_request'],inside(root,source_request_rel),cache['mask_request_sha256'])
    _copy(checked['source_manifest'],inside(root,source_manifest_rel),cache['semantic_manifest_sha256'])
    provenance=dict(schema_version=1,status='verified',source_job_dir=str(checked['source']),
        source_mask_request_sha256=cache['mask_request_sha256'],source_semantic_manifest_sha256=cache['semantic_manifest_sha256'],
        source_request_path=source_request_rel,source_manifest_path=source_manifest_rel,
        current_request_sha256=sha256(checked['request_path']),native_model_calls=0,model_calls=0,
        inference_reused=True,object_inference_projection='native_cubes',rows=len(checked['files']),
        bytes=sum(item['bytes'] for item in checked['files']),inputs=checked['inputs'],verification=checked['model_checks'],
        semantics={item['destination_path']:item['sha256'] for item in checked['files']},source_artifacts_modified=False)
    provenance_rel='prepared/native_semantic_cache/provenance.json';write_json(inside(root,provenance_rel),provenance)
    result=dict(checked['manifest'],request_sha256=provenance['current_request_sha256'],native_model_calls=0,model_calls=0,
        inference_reused=True,object_inference_projection='native_cubes',
        native_semantic_cache=dict(path=provenance_rel,sha256=sha256(inside(root,provenance_rel))))
    write_json(inside(root,'prepared/semantic_manifest.json'),result)
    return result
