"""Import verified completed panorama inputs from a terminal job, never geometry.

This preserves historical output bytes and their original model/code provenance.
It does not claim that the current preprocessing implementation recomputed them.
Only collect/preprocess must have completed; a later SfM failure stays failed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import uuid

import numpy as np
from PIL import Image

from .imaging import FACES, cube_camera_to_station_cv, fingerprint, sha256, write_json
from .native_collection import validate_source
from .panorama_preprocess import options_for
from .processing_options import read_processing_options
from .preprocess import model_provenance, mask_policy

KIND = 'completed_panorama_inputs_v1'
STAGES = {'collect': 'collection', 'preprocess': 'prepared'}


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def _ordinary(path, *, directory=False):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('Panorama input cache forbids links/junctions')
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError('Panorama input cache requires ordinary files/directories')
    return path


def _relative(name):
    if not isinstance(name, str) or not name or '\\' in name or ':' in name:
        raise ValueError('Cache requires portable relative paths')
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or str(path) != name:
        raise ValueError('Cache artifact path escapes its tree')
    return path


def _tree(root):
    names = set()
    for folder in STAGES.values():
        directory = _ordinary(root/folder, directory=True)
        for current, dirs, files in os.walk(directory, followlinks=False):
            for name in dirs:
                _ordinary(Path(current)/name, directory=True)
            for name in files:
                names.add(_ordinary(Path(current)/name).relative_to(root).as_posix())
    return names


def _verify_files(root, files):
    if _tree(root) != set(files):
        raise ValueError('Completed panorama cache file roster differs')
    for name, record in files.items():
        _relative(name)
        path = _ordinary(root/name)
        if type(record.get('bytes')) is not int or path.stat().st_size != record['bytes'] or sha256(path) != record.get('sha256'):
            raise ValueError('Completed panorama cache file size/hash differs: ' + name)


def _settings_signature(settings):
    if settings.get('workflow') != 'panorama_brush_refine':
        raise ValueError('Panorama input cache requires the explicit panorama workflow')
    return fingerprint(dict(workflow=settings['workflow'], collection=settings.get('collection'), options=options_for(settings)))


def _image(path, size, *, binary=False):
    with Image.open(path) as image:
        if image.size != size or image.mode != ('L' if binary else 'RGB') or image.getexif().get(274, 1) != 1:
            raise ValueError('Cached image/mask native format differs')
        value = np.asarray(image).copy()
    if binary and not np.isin(value, [0, 255]).all():
        raise ValueError('Cached binary mask contains nonbinary pixels')
    return value > 0 if binary else value


def _validate_model_files(request, semantic, generated, options):
    """Hash local model/processor files without importing or running a model."""
    checked = {}
    for key in ('sam', 'sky'):
        recorded = semantic.get(key, {}).get('model_provenance')
        actual = model_provenance(request[key])
        canonical_actual = json.loads(json.dumps(actual, allow_nan=False))
        if not recorded or fingerprint(recorded) != fingerprint(canonical_actual) or fingerprint(semantic.get(key+'_policy')) != fingerprint(mask_policy(request[key])):
            raise ValueError('Cached semantic model/processor bytes or policy differ: ' + key)
        checked[key] = actual['files_sha256']
    receipt = _ordinary(options['flux']['model_receipt_path'])
    digest = sha256(receipt)
    if options['flux'].get('model_receipt_sha256') != digest or generated.get('model_receipt_sha256') != digest or generated.get('model_revision') != options['flux']['model_revision']:
        raise ValueError('Cached FLUX model receipt/revision differs')
    checked['flux_receipt_sha256'] = digest
    return checked


def validate_cache(config, source_job_dir, settings):
    """Read-only full manifest, provenance, native-grid and file verification."""
    source = _ordinary(source_job_dir, directory=True)
    controls = {name: _ordinary(source/name) for name in ('config.json', 'settings.json', 'recipe.json', 'remote_state.json', 'cache_manifest.json')}
    control_hashes = {name: sha256(path) for name, path in controls.items()}
    state, cache, old_settings, recipe = (_read(controls[name]) for name in ('remote_state.json', 'cache_manifest.json', 'settings.json', 'recipe.json'))
    if state.get('status') not in ('completed', 'failed', 'cancelled') or state.get('id') != source.name:
        raise ValueError('Panorama input cache source must be a terminal recorded job')
    if fingerprint(_read(controls['config.json'])) != fingerprint(config):
        raise ValueError('Panorama input cache frozen config differs')
    signature = _settings_signature(settings)
    if _settings_signature(old_settings) != signature:
        raise ValueError('Panorama input cache collection/preprocessing options differ')
    if state.get('cache_manifest') != dict(path='cache_manifest.json', sha256=control_hashes['cache_manifest.json']) or cache.get('schema_version') != 1:
        raise ValueError('Completed cache inventory is not bound to source state')
    stages = state.get('stages', [])
    if [row.get('name') for row in stages[:2]] != list(STAGES):
        raise ValueError('Completed source stages must be the ordered input prefix')
    files = {}
    for record in stages[:2]:
        stage = record['name']; folder = STAGES[stage]
        if record.get('status') != 'completed' or record.get('exit_code') != 0:
            raise ValueError('Source input stage did not complete successfully')
        rows = cache.get('stages', {}).get(stage)
        if not isinstance(rows, dict) or not rows:
            raise ValueError('Completed input stage has no bound file inventory')
        for name, item in rows.items():
            if _relative(name).parts[0] != folder or name in files:
                raise ValueError('Only distinct collection/prepared files may be reused')
            files[name] = item
        outputs = record.get('outputs', [])
        if not outputs or not any(row.get('path') == folder+'/manifest.json' for row in outputs):
            raise ValueError('Source input stage manifest output is missing')
        for row in outputs:
            if files.get(row.get('path')) != {key: row.get(key) for key in ('sha256', 'bytes')}:
                raise ValueError('Completed stage output differs from cache inventory')
    _verify_files(source, files)
    code = recipe.get('code_sha256')
    if not isinstance(code, dict) or not code:
        raise ValueError('Source job code provenance is missing')
    if 'code/tools/streetview_engine/panorama_preprocess.py' not in code:
        raise ValueError('Source panorama worker code provenance is missing')
    actual_code = set()
    for current, dirs, names in os.walk(_ordinary(source/'code', directory=True), followlinks=False):
        for name in dirs: _ordinary(Path(current)/name, directory=True)
        for name in names: actual_code.add(_ordinary(Path(current)/name).relative_to(source).as_posix())
    if actual_code != set(code):
        raise ValueError('Source job code roster differs from frozen recipe')
    for name, digest in code.items():
        if _relative(name).parts[0] != 'code' or Path(name).suffix != '.py' or sha256(_ordinary(source/name)) != digest:
            raise ValueError('Source job code hash differs from frozen recipe')

    def artifact(name, digest=None):
        _relative(name)
        if name not in files or digest is not None and files[name]['sha256'] != digest:
            raise ValueError('Panorama manifest artifact binding differs: ' + str(name))
        return source/name

    collected = _read(artifact('collection/manifest.json'))
    prepared = _read(artifact('prepared/manifest.json'))
    if collected.get('source_kind') != 'native_cube_import':
        raise ValueError('Completed panorama cache currently requires verified native source imports')
    # The current importer independently checks provider metadata, coordinates,
    # decoded RGB, coverage, processing history and all six native calibrations.
    import_settings = {key: value for key, value in settings.items() if key != 'input_cache'}
    native = validate_source(config, source, import_settings)
    if native['manifest'] != collected:
        raise ValueError('Cached collection differs from the currently verified native source')
    for name, record in native['files'].items():
        artifact(name, record['sha256'])
    collection_input = _read(artifact('collection/input.json'))
    if collection_input.get('config') != config or collection_input.get('input_fingerprint') != collected.get('input_fingerprint'):
        raise ValueError('Cached collection input identity differs')
    options = options_for(settings)
    expected = fingerprint(dict(config=config, collection_sha256=files['collection/manifest.json']['sha256'], options=options))
    if (prepared.get('stage') != 'preprocess' or prepared.get('status') != 'complete' or prepared.get('workflow') != 'panorama_brush_refine'
            or prepared.get('input_fingerprint') != expected or prepared.get('options') != options
            or prepared.get('collection_manifest_path') != 'collection/manifest.json' or prepared.get('collection_sha256') != files['collection/manifest.json']['sha256']
            or prepared.get('processing_options') != read_processing_options(config) or prepared.get('camera_convention') != 'opencv'):
        raise ValueError('Cached panorama preprocessing identity differs')
    for key in ('original_outside_alpha_exact', 'source_images_preserved', 'native_training_resolution_preserved'):
        if prepared.get(key) is not True: raise ValueError('Cached panorama does not preserve native source pixels')
    if prepared.get('generated_regions_are_geometry_evidence') is not False or prepared.get('per_image_fits') != 0:
        raise ValueError('Cached panorama geometry/composition policy differs')
    mask_request = _read(artifact('prepared/mask_request.json'))
    flux_request = _read(artifact('prepared/flux_request.json'))
    semantic = _read(artifact('prepared/semantic_manifest.json', prepared['semantic_manifest_sha256']))
    generated = _read(artifact('prepared/generation_manifest.json', prepared['generation_manifest_sha256']))
    for request, manifest, name in ((mask_request, semantic, 'mask_request'), (flux_request, generated, 'flux_request')):
        if Path(request.get('root', '')).absolute() != source or manifest.get('status') != 'completed' or manifest.get('request_sha256') != files['prepared/'+name+'.json']['sha256']:
            raise ValueError('Cached worker result is not bound to its original completed request')
    for field, key in [('sam', 'sam_segmentation'), ('sky', 'sky_segmentation'), ('object_projection', 'object_projection'), ('sky_instance_score_threshold', 'sky_instance_score_threshold')]:
        if mask_request.get(field) != options[key]: raise ValueError('Cached mask request settings differ')
    if flux_request.get('options') != options['flux']:
        raise ValueError('Cached generation request settings differ')
    model_files = _validate_model_files(mask_request, semantic, generated, options)
    faces = {(station['pano_id'], face): (station, station['faces'][face]) for station in collected['stations'] for face in FACES}
    frames = prepared.get('frames', [])
    if [(row.get('pano_id'), row.get('face')) for row in frames] != list(faces):
        raise ValueError('Cached native frame roster/order differs')
    request_rows = mask_request.get('rows', [])
    semantic_rows = semantic.get('rows', [])
    tokens = ['pano_'+fingerprint(row['pano_id'])[:20]+'_'+row['face'] for row in frames]
    if [row.get('token') for row in request_rows] != tokens or [row.get('token') for row in semantic_rows] != tokens:
        raise ValueError('Cached semantic native roster/order differs')
    for frame, request, record in zip(frames, request_rows, semantic_rows):
        station, original = faces[frame['pano_id'], frame['face']]
        size = (original['w'], original['h']); shape = size[::-1]
        expected_fields = dict(pano_id=station['pano_id'], station_id=station['station_id'], face=frame['face'],
            original_file_path=original['file_path'], original_sha256=original['sha256'], w=size[0], h=size[1],
            fl_x=size[0]/2, fl_y=size[1]/2, cx=size[0]/2, cy=size[1]/2,
            camera_to_station_cv=cube_camera_to_station_cv(frame['face']).tolist())
        if any(frame.get(key) != value or request.get(key) != value for key, value in expected_fields.items()):
            raise ValueError('Cached native image/camera/source binding differs')
        original_rgb = _image(artifact(frame['original_file_path'], frame['original_sha256']), size)
        rgb = _image(artifact(frame['file_path'], frame['source_sha256']), size)
        alpha = np.load(artifact(frame['edit_alpha_path'], frame['edit_alpha_sha256']), allow_pickle=False)
        if alpha.shape != shape or not np.isfinite(alpha).all() or np.any((alpha < 0) | (alpha > 1)):
            raise ValueError('Cached native alpha grid/range differs')
        if not np.array_equal(rgb[alpha == 0], original_rgb[alpha == 0]):
            raise ValueError('Cached composite changed original pixels outside alpha')
        masks = {}
        for name, prefix in [('valid', 'original_valid_mask'), ('photo', 'mask'), ('sfm', 'sfm_mask'), ('sky', 'sky_mask'), ('ground', 'ground_mask')]:
            masks[name] = _image(artifact(frame[prefix+'_path'], frame[prefix+'_sha256']), size, binary=True)
        native_valid = _image(artifact(original['valid_mask_path'], original['valid_mask_sha256']), size, binary=True) if original.get('valid_mask_path') else np.ones(shape, bool)
        if not np.array_equal(masks['valid'], native_valid): raise ValueError('Cached source coverage mask differs')
        if record.get('source_sha256') != frame['original_sha256'] or frame.get('semantic_source_sha256') != frame['original_sha256'] or record.get('path') != frame.get('semantic_path') or record.get('sha256') != frame.get('semantic_sha256'):
            raise ValueError('Cached semantic source binding differs')
        with np.load(artifact(record['path'], record['sha256']), allow_pickle=False) as evidence:
            if str(evidence['source_sha256'].item()) != frame['original_sha256']:
                raise ValueError('Cached NPZ source binding differs')
            planes = {name: evidence[name] for name in ('dynamic', 'sky_region', 'sky_high', 'ground')}
            if any(value.shape != shape or value.dtype != bool for value in planes.values()):
                raise ValueError('Cached semantic binary grid differs')
            expected_masks = dict(sfm=native_valid & ~planes['dynamic'] & ~planes['sky_region'] & (alpha == 0),
                photo=native_valid & ~planes['sky_region'] if read_processing_options(config)['remove_sky'] else native_valid,
                sky=planes['sky_high'], ground=planes['ground'])
            if any(not np.array_equal(masks[key], value) for key, value in expected_masks.items()):
                raise ValueError('Cached native masks do not match semantic/alpha policy')
    stations = mask_request.get('stations', [])
    station_tokens = ['pano_'+fingerprint(station['pano_id'])[:20] for station in collected['stations']]
    if [row.get('token') for row in stations] != station_tokens or [row.get('token') for row in flux_request.get('rows', [])] != station_tokens or [row.get('token') for row in generated.get('rows', [])] != station_tokens:
        raise ValueError('Cached whole-panorama roster/order differs')
    erp_size = (options['flux']['erp_width'], options['flux']['erp_width']//2)
    for station, request, record in zip(stations, flux_request['rows'], generated['rows']):
        if station.get('original_erp_path') != request.get('original_path') or station.get('original_erp_sha256') != request.get('original_sha256') or record.get('original_sha256') != request.get('original_sha256') or request.get('output_path') != record.get('path'):
            raise ValueError('Cached generation original/output binding differs')
        _image(artifact(request['original_path'], request['original_sha256']), erp_size)
        _image(artifact(record['path'], record['sha256']), erp_size)
    if options['object_projection'] == 'whole_erp':
        erp_rows = semantic.get('erp_rows', [])
        if [row.get('token') for row in erp_rows] != station_tokens:
            raise ValueError('Cached whole-ERP semantic roster differs')
        for station, row in zip(stations, erp_rows):
            if row.get('source_sha256') != station['original_erp_sha256']: raise ValueError('Cached whole-ERP source differs')
            with np.load(artifact(row['path'], row['sha256']), allow_pickle=False) as evidence:
                if evidence['dynamic'].dtype != bool or evidence['dynamic'].shape != erp_size[::-1]:
                    raise ValueError('Cached whole-ERP mask grid differs')
    if any(sha256(path) != control_hashes[name] for name, path in controls.items()):
        raise ValueError('Source control/provenance changed during validation')
    return dict(kind=KIND, source_job_dir=str(source), source_job_status=state['status'], source_stage_records=stages[:2],
        config_sha256=fingerprint(config), settings_signature=signature, source_control_sha256=control_hashes,
        source_code_sha256=code, collection_manifest_sha256=files['collection/manifest.json']['sha256'],
        model_files_verified=model_files,
        prepared_manifest_sha256=files['prepared/manifest.json']['sha256'], files=files,
        files_sha256={name: row['sha256'] for name, row in files.items()}, files_count=len(files), bytes=sum(row['bytes'] for row in files.values()),
        source_stages_only=list(STAGES.values()), model_calls=0, provenance_policy='Historical completed artifact import; original requests, model metadata and code hashes retained; no current-model equivalence or recomputation claimed')


def _cleanup(path, destination):
    if path.parent != destination or not (path.name.startswith('.panorama_input_cache_') or path.name in STAGES.values()):
        raise RuntimeError('Cache cleanup escaped the owned destination')
    if path.exists():
        _ordinary(path, directory=True); shutil.rmtree(path)


def reuse_inputs(config, job_dir, settings):
    cache = settings.get('input_cache')
    if not isinstance(cache, dict) or set(cache) != {'source_job_dir'} or not isinstance(cache['source_job_dir'], str) or not Path(cache['source_job_dir']).is_absolute():
        raise ValueError('input_cache requires only an absolute source_job_dir')
    source = _ordinary(cache['source_job_dir'], directory=True); destination = Path(job_dir).absolute()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Panorama cache source/destination must be separate non-nested jobs')
    verified = validate_cache(config, source, settings)
    if (destination/'input_cache.json').exists():
        cached_stage(config, destination, settings, 'preprocess')
        receipt = _read(destination/'input_cache.json')
        if receipt.get('signature') != fingerprint(verified): raise ValueError('Existing panorama import source changed')
        return receipt
    destination.mkdir(parents=True, exist_ok=True); _ordinary(destination, directory=True)
    if any((destination/folder).exists() or (destination/folder).is_symlink() for folder in STAGES.values()):
        raise ValueError('Refusing to overwrite destination collection/prepared trees')
    staging = destination/('.panorama_input_cache_'+uuid.uuid4().hex)
    created = []; lock = destination/'.panorama_input_cache_lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600); os.close(fd)
    try:
        staging.mkdir()
        for name, record in verified['files'].items():
            target = staging/name; target.parent.mkdir(parents=True, exist_ok=True)
            with _ordinary(source/name).open('rb') as left, target.open('xb') as right:
                shutil.copyfileobj(left, right, 1024*1024)
            if sha256(target) != record['sha256']: raise ValueError('Source changed during panorama cache copy')
        _verify_files(source, verified['files']); _verify_files(staging, verified['files'])
        for folder in STAGES.values():
            if (destination/folder).exists(): raise ValueError('Destination stage appeared during cache import')
            (staging/folder).rename(destination/folder); created.append(destination/folder)
        receipt = dict(status='reused', signature=fingerprint(verified), **verified, destination_job_dir=str(destination),
            hardlinked_files=0, copied_files=verified['files_count'], geometry_or_ownership_copied=False)
        write_json(destination/'input_cache.json', receipt)
        return receipt
    except BaseException:
        for path in reversed(created): _cleanup(path, destination)
        raise
    finally:
        _cleanup(staging, destination); lock.unlink(missing_ok=True)


def cached_stage(config, job_dir, settings, stage):
    """Return original manifest only after the destination receipt/bytes verify."""
    if stage not in STAGES: raise ValueError('Panorama input reuse is limited to collect/preprocess')
    root = _ordinary(job_dir, directory=True); receipt = _read(_ordinary(root/'input_cache.json'))
    cache = settings.get('input_cache', {})
    if (receipt.get('kind') != KIND or receipt.get('status') != 'reused' or receipt.get('config_sha256') != fingerprint(config)
            or receipt.get('settings_signature') != _settings_signature(settings) or receipt.get('destination_job_dir') != str(root)
            or receipt.get('source_job_dir') != cache.get('source_job_dir')):
        raise ValueError('Cached panorama stage receipt/config/settings differ')
    excluded = {'status', 'signature', 'destination_job_dir', 'hardlinked_files', 'copied_files', 'geometry_or_ownership_copied'}
    verified = {key: value for key, value in receipt.items() if key not in excluded}
    if receipt.get('signature') != fingerprint(verified): raise ValueError('Cached panorama receipt signature differs')
    _verify_files(root, receipt['files'])
    return _read(root/STAGES[stage]/'manifest.json')
