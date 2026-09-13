"""Actual camera reprojection/composition with deterministic model fixtures."""
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from tools.streetview_engine import panorama_preprocess as module
from tools.streetview_engine.imaging import FACES, cube_camera_to_station_cv, fingerprint, inside, png_bytes, sha256, write_bytes, write_json
from tools.streetview_engine.sfm import validate_prepared_selection, prepare_inputs


def fixture(root):
    ids = ['capture-A', 'capture-B']; stations = []
    for index, pid in enumerate(ids):
        faces = {}
        for face in FACES:
            y, x = np.mgrid[:32, :32]
            rgb = np.stack((x*6, y*6, (x+y)*3), -1).astype(np.uint8)
            path = f'collection/{pid}_{face}.png'; write_bytes(inside(root, path), png_bytes(rgb))
            faces[face] = dict(file_path=path, sha256=sha256(root/path), w=32, h=32)
        stations.append(dict(pano_id=pid, station_id=f'physical-{index}', lat=37., lng=127.+index*.001, faces=faces))
    write_json(root/'collection/manifest.json', dict(status='complete', face_order=list(FACES), stations=stations))
    settings = dict(panorama_preprocess=dict(sam_segmentation=dict(backend='sam3', instance_verifier={'explicit':'fixture'}, python_executable='fixture'),
        sky_segmentation={'model':'fixture'}, flux=dict(model_receipt_path='fixture', model_revision='fixture',
            python_executable='fixture', prompt='Fixture removal', erp_width=128)))
    config = dict(panorama_ids=ids, processing_options=dict(mask_dynamic=True, remove_sky=False))
    return config, settings


def deterministic_worker(operation, request_path, runtime):
    request = json.loads(request_path.read_text(encoding='utf8')); root = Path(request['root']); rows = []
    if operation == 'masks':
        for row in request['rows']:
            dynamic = np.zeros((row['h'], row['w']), bool)
            if row['face'] == 'F':
                dynamic[15:20, 12:17] = True
            sky = np.zeros_like(dynamic); sky[:2] = True
            out = BytesIO(); np.savez_compressed(out, dynamic=dynamic, sky_region=sky, sky_high=sky, ground=np.zeros_like(dynamic))
            path=f'prepared/semantics/{row["token"]}.npz'; write_bytes(root/path, out.getvalue())
            rows.append(dict(token=row['token'], path=path, sha256=sha256(root/path), source_sha256=row['original_sha256']))
        write_json(root/'prepared/semantic_manifest.json', dict(status='completed', request_sha256=sha256(request_path), rows=rows))
    else:
        for row in request['rows']:
            with Image.open(root/row['original_path']) as image:
                rgb=np.asarray(image).copy()
            rgb[:] = [20, 210, 75]
            write_bytes(root/row['output_path'], png_bytes(rgb))
            rows.append(dict(token=row['token'], path=row['output_path'], sha256=sha256(root/row['output_path'])))
        write_json(root/'prepared/generation_manifest.json', dict(status='completed', request_sha256=sha256(request_path), rows=rows))


def test_actual_full_erp_then_native_patch_preserves_pixels_calibration_and_evidence(tmp_path):
    config, settings = fixture(tmp_path)
    result = module.run(config, tmp_path, settings, _worker=deterministic_worker)
    assert result['status'] == 'complete' and len(result['frames']) == 12
    assert result['per_image_fits'] == 0 and result['native_training_resolution_preserved']
    assert result['sky_rgb_included'] and not result['generated_regions_are_geometry_evidence']
    validate_prepared_selection(config, result)
    changed = 0
    for row in result['frames']:
        with Image.open(tmp_path/row['original_file_path']) as image: original=np.asarray(image)
        with Image.open(tmp_path/row['file_path']) as image: composite=np.asarray(image)
        with Image.open(tmp_path/row['sfm_mask_path']) as image: sfm=np.asarray(image)>0
        with Image.open(tmp_path/row['mask_path']) as image: rgbvalid=np.asarray(image)>0
        alpha=np.load(tmp_path/row['edit_alpha_path'],allow_pickle=False)
        assert composite.shape == original.shape == (32,32,3)
        assert np.array_equal(composite[alpha==0], original[alpha==0])
        assert not sfm[alpha>0].any() and rgbvalid.all()
        assert np.array_equal(row['camera_to_station_cv'], cube_camera_to_station_cv(row['face']))
        assert row['semantic_source_sha256'] == sha256(tmp_path/row['original_file_path'])
        changed += np.count_nonzero(np.any(composite != original, axis=-1))
    assert changed > 0
    assert result['frames'][0]['source_sha256'] != result['frames'][0]['original_sha256']
    # Existing SfM adapter accepts the complete camera/image/mask contract.
    output=tmp_path/'sfm'; output.mkdir()
    prepare_inputs(tmp_path, output, result)


def test_mask_off_keeps_all_native_pixels_and_records_zero_edit_support(tmp_path):
    config, settings = fixture(tmp_path); config['processing_options']['mask_dynamic'] = False
    result = module.run(config, tmp_path, settings, _worker=deterministic_worker)
    for row in result['frames']:
        assert not np.load(tmp_path/row['edit_alpha_path'], allow_pickle=False).any()
        with Image.open(tmp_path/row['file_path']) as image: actual=np.asarray(image)
        with Image.open(tmp_path/row['original_file_path']) as image: original=np.asarray(image)
        assert np.array_equal(actual, original)


def test_generated_format_mismatch_fails_before_any_native_composite(tmp_path):
    config, settings = fixture(tmp_path)
    def bad_size(operation, path, runtime):
        deterministic_worker(operation, path, runtime)
        if operation == 'flux':
            report=json.loads((tmp_path/'prepared/generation_manifest.json').read_text(encoding='utf8'))
            row=report['rows'][0]; Image.new('RGB',(64,64)).save(tmp_path/row['path'])
            row['sha256']=sha256(tmp_path/row['path'])
            (tmp_path/'prepared/generation_manifest.json').write_text(json.dumps(report),encoding='utf8')
    with pytest.raises(ValueError, match='format differs'):
        module.run(config,tmp_path,settings,_worker=bad_size)
    assert not (tmp_path/'prepared/native').exists()


def test_seed_determinism_and_shared_transform_validation(tmp_path):
    assert module.seed_for('station-a') == module.seed_for('station-a')
    assert module.seed_for('station-a') != module.seed_for('station-b')
    _, settings=fixture(tmp_path)
    settings['panorama_preprocess']['common_transform']=[[1,0,0],[0,-1,0]]
    with pytest.raises(ValueError,match='orientation-preserving'):
        module.options_for(settings)


def test_sky_gate_is_instance_score_not_pixel_probability_and_ties_are_excluded():
    masks=np.zeros((3, 5, 7), bool); masks[0,:2]=True; masks[1,3,1:3]=True; masks[2,4,4:]=True
    metadata={'instances':[dict(group='sky',mask_index=i,score=score,accepted_for_group=True,
        verification={'accepted':True},mask_area_pixels=int(masks[i].sum())) for i,score in enumerate([.9,.35998,.5])]}
    evidence=dict(group_probability={'sky':np.any(masks,axis=0).astype(np.float32)*.7},
        instance_masks_packed=np.packbits(masks.reshape(3,-1),axis=1,bitorder='little'),
        instance_masks_shape=np.array(masks.shape),metadata=metadata)
    assert np.array_equal(module.gated_sky(evidence,.5),masks[0])
    evidence['group_probability']['sky'][:]=1
    with pytest.raises(ValueError,match='packed instance union'):
        module.gated_sky(evidence,.5)


def full_erp_worker(operation, request_path, runtime):
    deterministic_worker(operation, request_path, runtime)
    if operation != 'masks':
        return
    request=json.loads(request_path.read_text()); root=Path(request['root'])
    assert request['object_projection']=='whole_erp'
    manifest=json.loads((root/'prepared/semantic_manifest.json').read_text())
    # Deliberately empty native object masks prove that parent editing uses the
    # directly inferred ERP mask, not a native-mask-to-ERP reconstruction.
    for row in manifest['rows']:
        with np.load(root/row['path']) as data:
            arrays={key:data[key].copy() for key in data.files}
        arrays['dynamic'][:]=False
        stream=BytesIO();np.savez_compressed(stream,**arrays)
        write_bytes(root/row['path'],stream.getvalue(),immutable=False);row['sha256']=sha256(root/row['path'])
    erp_rows=[]
    for station in request['stations']:
        with Image.open(root/station['original_erp_path']) as image:
            assert image.size==(128,64)
        mask=np.zeros((64,128),bool);mask[29:35,61:67]=True;mask[29:35,:2]=True
        stream=BytesIO();np.savez_compressed(stream,dynamic=mask)
        path=f'prepared/erp_semantics/{station["token"]}.npz'
        write_bytes(root/path,stream.getvalue())
        erp_rows.append(dict(token=station['token'],path=path,sha256=sha256(root/path),
            source_sha256=station['original_erp_sha256']))
    manifest.update(erp_rows=erp_rows,object_inference_images=len(erp_rows),native_object_inference_images=0)
    write_json(root/'prepared/semantic_manifest.json',manifest,immutable=False)


def test_full_erp_objects_feed_direct_edit_core_and_preserve_original_native_pixels(tmp_path, monkeypatch):
    config,settings=fixture(tmp_path);settings['panorama_preprocess']['object_projection']='whole_erp'
    original_project=module.masks_to_erp;calls=[]
    def count_sky_only(masks,cameras,width):
        calls.append(len(masks));return original_project(masks,cameras,width)
    monkeypatch.setattr(module,'masks_to_erp',count_sky_only)
    result=module.run(config,tmp_path,settings,_worker=full_erp_worker)
    assert calls==[6,6]  # Only the independent broad sky guard is projected into ERP.
    assert result['object_inference_projection']=='whole_erp' and result['object_inference_images']==2
    assert not result['native_semantic_cache_used']
    changed=0
    for frame in result['frames']:
        with Image.open(tmp_path/frame['original_file_path']) as image:original=np.asarray(image)
        with Image.open(tmp_path/frame['file_path']) as image:actual=np.asarray(image)
        with Image.open(tmp_path/frame['sfm_mask_path']) as image:sfm=np.asarray(image)>0
        alpha=np.load(tmp_path/frame['edit_alpha_path'])
        assert np.array_equal(actual[alpha==0],original[alpha==0])
        assert not sfm[alpha>0].any()
        changed+=np.count_nonzero(np.any(actual!=original,axis=-1))
    assert changed>0


def test_full_erp_source_mismatch_fails_before_generation(tmp_path):
    config,settings=fixture(tmp_path);settings['panorama_preprocess']['object_projection']='whole_erp'
    def corrupt(operation,request_path,runtime):
        assert operation=='masks'
        full_erp_worker(operation,request_path,runtime)
        path=tmp_path/'prepared/semantic_manifest.json';manifest=json.loads(path.read_text())
        manifest['erp_rows'][0]['source_sha256']='0'*64;write_json(path,manifest,immutable=False)
    with pytest.raises(ValueError,match='object evidence/source hash'):
        module.run(config,tmp_path,settings,_worker=corrupt)
    assert not (tmp_path/'prepared/flux_request.json').exists()


def test_native_object_cache_cannot_be_used_for_whole_erp_inference(tmp_path):
    _,settings=fixture(tmp_path)
    settings['panorama_preprocess'].update(object_projection='whole_erp',native_semantic_cache={})
    with pytest.raises(ValueError,match='Native semantic cache'):
        module.options_for(settings)
