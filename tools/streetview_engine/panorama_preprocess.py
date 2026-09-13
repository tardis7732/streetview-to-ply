"""Whole-ERP object removal, then exact native-grid mask-only composition.

Segmentation and FLUX run in explicitly configured isolated runtimes. Source
photographs are never resized or overwritten; only generated pixels beneath
one spherical alpha and one shared panorama transform enter native targets.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from PIL import Image

from .imaging import FACES, cube_camera_to_station_cv, fingerprint, inside, png_bytes, sha256, write_bytes, write_json
from .processing_options import read_processing_options
from .panorama_cube import CubeSampler
from .panorama_feather import FeatherSettings, spherical_alpha
from .panorama_native_mapping import native_uv_alpha
from .panorama_native_composite import compose_erp, compose_on_target_grid


def options_for(settings):
    options = settings.get('panorama_preprocess')
    if not isinstance(options, dict):
        raise ValueError('Panorama workflow requires explicit panorama_preprocess operator settings')
    required = {'sam_segmentation', 'sky_segmentation', 'flux'}
    if not required <= options.keys() or set(options) - (required | {'feather_at_2048', 'dilation_at_2048', 'common_transform', 'chunk_rows', 'sky_instance_score_threshold', 'object_projection', 'object_dilation_at_2048', 'native_semantic_cache'}):
        raise ValueError('Invalid panorama preprocessing fields')
    options = json.loads(json.dumps(options, allow_nan=False))
    flux = options['flux']
    if not isinstance(flux, dict) or not {'model_receipt_path', 'model_revision', 'python_executable', 'prompt'} <= flux.keys():
        raise ValueError('FLUX requires an operator-owned model receipt, revision, runtime and prompt')
    if set(flux) - {'model_receipt_path', 'model_receipt_sha256', 'model_revision', 'python_executable',
                    'prompt', 'erp_width', 'steps', 'guidance_scale', 'seed'}:
        raise ValueError('Unknown FLUX processing option')
    if flux.get('model_receipt_sha256') is not None and not re.fullmatch(r'[0-9a-f]{64}', str(flux['model_receipt_sha256'])):
        raise ValueError('Invalid FLUX model receipt hash')
    flux.setdefault('erp_width', 2048); flux.setdefault('steps', 4); flux.setdefault('guidance_scale', 1.); flux.setdefault('seed', 42)
    width = flux['erp_width']
    if type(width) is not int or not 128 <= width <= 8192 or width % 32 or width*(width//2) > 2024*2024:
        raise ValueError('ERP width must be divisible by 32 and fit the pinned full-field processor area')
    if type(flux['steps']) is not int or not 1 <= flux['steps'] <= 100 or type(flux['seed']) is not int or not 0 <= flux['seed'] < 2**31:
        raise ValueError('Invalid FLUX inference steps or seed')
    if not isinstance(flux['prompt'], str) or not flux['prompt'].strip() or len(flux['prompt']) > 4096:
        raise ValueError('An explicit bounded FLUX prompt is required')
    if isinstance(flux['guidance_scale'], bool) or not isinstance(flux['guidance_scale'], (int, float)) or not math.isfinite(flux['guidance_scale']) or not 0 <= flux['guidance_scale'] <= 20:
        raise ValueError('Invalid FLUX guidance')
    options.setdefault('common_transform', [[1., 0., 0.], [0., 1., 0.]])
    matrix = np.asarray(options['common_transform'], np.float64)
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all() or np.linalg.det(matrix[:, :2]) <= 0:
        raise ValueError('One finite orientation-preserving panorama transform is required')
    options.setdefault('feather_at_2048', 4.); options.setdefault('dilation_at_2048', 2.)
    FeatherSettings(feather_ratio=options['feather_at_2048']/2048, dilation_ratio=options['dilation_at_2048']/2048)
    options.setdefault('chunk_rows', 32)
    if type(options['chunk_rows']) is not int or not 1 <= options['chunk_rows'] <= 256:
        raise ValueError('Invalid panorama chunk size')
    if options['sam_segmentation'].get('backend') != 'sam3' or not options['sam_segmentation'].get('instance_verifier'):
        raise ValueError('Panorama dynamics require SAM3 with independent instance verification')
    if options['sky_segmentation'].get('backend', 'hf_semantic') != 'hf_semantic':
        raise ValueError('Broad sky exclusion requires the configured semantic model')
    options.setdefault('sky_instance_score_threshold', .5)
    threshold = options['sky_instance_score_threshold']
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Invalid global sky instance score threshold')
    options.setdefault('object_projection', 'native_cubes')
    if options['object_projection'] not in ('native_cubes', 'whole_erp'):
        raise ValueError('Object inference projection must be native_cubes or whole_erp')
    options.setdefault('object_dilation_at_2048', 1.)
    dilation = options['object_dilation_at_2048']
    if isinstance(dilation, bool) or not isinstance(dilation, (int, float)) or not math.isfinite(dilation) or not 0 <= dilation <= 32:
        raise ValueError('Invalid full-panorama object dilation')
    if options.get('native_semantic_cache') is not None:
        cache = options['native_semantic_cache']
        if options['object_projection'] != 'native_cubes' or not isinstance(cache, dict) or set(cache) != {'source_job_dir','semantic_manifest_sha256','mask_request_sha256'}:
            raise ValueError('Native semantic cache requires an explicit native-cube source request and manifest')
    return options


def seed_for(station_id, base=42):
    import hashlib
    return (base + int(hashlib.sha256((str(station_id) + '/whole_erp').encode()).hexdigest()[:8], 16)) % (2**31)


def erp_rays(width, start=0, stop=None, *, dtype=np.float64):
    height = width // 2; stop = height if stop is None else stop
    lon = ((np.arange(width) + .5)/width - .5) * (2*np.pi)
    lat = ((np.arange(start, stop) + .5)/height - .5) * np.pi
    return np.stack((np.cos(lat[:, None])*np.sin(lon[None]),
        np.broadcast_to(np.sin(lat[:, None]), (stop-start, width)),
        np.cos(lat[:, None])*np.cos(lon[None])), -1).astype(dtype)


def cube_to_erp(images, cameras, width, chunk_rows=32):
    sampler = CubeSampler(images, [np.ones(image.shape[:2], bool) for image in images], cameras)
    result = np.empty((width//2, width, 3), np.uint8)
    for first in range(0, width//2, chunk_rows):
        last = min(first + chunk_rows, width//2)
        pixels, valid, _ = sampler.sample(erp_rays(width, first, last))
        if not valid.all():
            raise ValueError('Incomplete native cube RGB coverage')
        result[first:last] = pixels
    return result


def masks_to_erp(masks, cameras, width):
    """Original float32 ERP rays / max-Z owner / nearest native mask sampling."""
    import cv2
    rays = erp_rays(width, dtype=np.float32)
    owner = np.full(rays.shape[:2], -1, np.int8)
    best = np.full(owner.shape, -np.inf, np.float32)
    result = np.zeros(owner.shape, bool)
    for index, (mask, camera) in enumerate(zip(masks, cameras)):
        local = rays @ np.asarray(camera['camera_to_station_cv'])[:3, :3]
        z = local[..., 2]
        u = camera['fl_x']*local[..., 0]/np.maximum(z, 1e-12) + camera['cx']
        v = camera['fl_y']*local[..., 1]/np.maximum(z, 1e-12) + camera['cy']
        chosen = (z > 0) & (u >= 0) & (u < camera['w']) & (v >= 0) & (v < camera['h']) & (z > best)
        sampled = cv2.remap(np.asarray(mask, np.uint8), (u-.5).astype(np.float32), (v-.5).astype(np.float32),
            cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE) > 0
        result[chosen] = sampled[chosen]; best[chosen] = z[chosen]; owner[chosen] = index
    if (owner < 0).any() or len(np.unique(owner)) != 6:
        raise ValueError('ERP mask mapping does not cover all six native faces')
    return result


def worker_call(operation, request, runtime):
    executable = Path(runtime).expanduser()
    if not executable.is_absolute() or not executable.is_file():
        raise ValueError('Panorama model runtime must be an existing absolute executable')
    code_root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ, PYTHONUTF8='1', PYTHONUNBUFFERED='1')
    environment['PYTHONPATH'] = str(code_root) + os.pathsep + environment.get('PYTHONPATH', '')
    subprocess.run([str(executable), '-B', '-m', 'tools.streetview_engine.panorama_preprocess',
        operation, '--request', str(request)], cwd=code_root, env=environment,
        stdin=subprocess.DEVNULL, check=True, shell=False,
        **({'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}))


def _rgb(path, size=None):
    with Image.open(path) as image:
        if image.mode != 'RGB' or size is not None and image.size != tuple(size):
            raise ValueError('Original opaque native RGB format differs')
        return np.asarray(image).copy()


def _save_npy(path, value):
    stream = BytesIO(); np.save(stream, value, allow_pickle=False); write_bytes(path, stream.getvalue())


def gated_sky(evidence, threshold=.5):
    """Union accepted packed sky masks under one strict instance-score gate."""
    probabilities = np.asarray(evidence['group_probability']['sky'])
    shape = probabilities.shape
    packed = np.asarray(evidence['instance_masks_packed'])
    dimensions = np.asarray(evidence['instance_masks_shape'])
    instances = evidence['metadata']['instances']
    if probabilities.ndim != 2 or tuple(dimensions[1:]) != shape or len(packed) != len(instances):
        raise ValueError('Packed sky mask/metadata shape differs')
    all_sky = np.zeros(shape, bool); kept = np.zeros(shape, bool)
    for instance in instances:
        if instance['group'] != 'sky':
            continue
        score = float(instance['score'])
        if not math.isfinite(score) or not 0 <= score <= 1 or not instance['accepted_for_group'] or not instance['verification']['accepted']:
            raise ValueError('Unsupported sky instance or invalid score')
        mask = np.unpackbits(packed[instance['mask_index']], bitorder='little')[:shape[0]*shape[1]].reshape(shape).astype(bool)
        if int(mask.sum()) != instance['mask_area_pixels']:
            raise ValueError('Sky packed mask area differs')
        all_sky |= mask
        if score > threshold:
            kept |= mask
    if not np.array_equal(all_sky, probabilities > .5):
        raise ValueError('Original SAM sky group differs from its packed instance union')
    return kept


def run(config, job_dir, settings, *, _worker=worker_call):
    root = Path(job_dir).resolve(); options = options_for(settings)
    processing = read_processing_options(config)
    source_path = inside(root, 'collection/manifest.json')
    collection_hash = sha256(source_path)
    collection = json.loads(source_path.read_text(encoding='utf8'))
    if collection.get('status') != 'complete' or collection.get('face_order') != list(FACES):
        raise ValueError('Panorama preprocessing requires a complete native six-face collection')
    if [row['pano_id'] for row in collection['stations']] != config['panorama_ids']:
        raise ValueError('Collected panorama IDs/order differ from frozen selection')
    destination = inside(root, 'prepared/manifest.json')
    if destination.exists():
        raise FileExistsError('Use verified completed-stage reuse or a fresh preprocessing output')
    inputs = {}; stations = []; rows = []
    for station in collection['stations']:
        token = 'pano_' + fingerprint(station['pano_id'])[:20]
        cameras = []
        for face in FACES:
            source = station['faces'][face]
            path = inside(root, source['file_path'])
            if sha256(path) != source['sha256']:
                raise ValueError('Original cube photograph hash changed')
            image = _rgb(path, (source['w'], source['h']))
            if image.shape[0] != image.shape[1]:
                raise ValueError('Native cube faces must be square')
            inputs[source['file_path']] = source['sha256']
            width, height = source['w'], source['h']
            row = dict(token=token+'_'+face, pano_id=station['pano_id'], station_id=station['station_id'], face=face,
                original_file_path=source['file_path'], original_sha256=source['sha256'],
                w=width, h=height, fl_x=width/2, fl_y=height/2, cx=width/2, cy=height/2,
                camera_to_station_cv=cube_camera_to_station_cv(face).tolist())
            # Native collection is full RGB coverage. If a provider supplies an
            # invalid/nadir coverage mask, preserve that actual mask exactly.
            valid = np.ones((height, width), bool)
            if source.get('valid_mask_path'):
                valid_path = inside(root, source['valid_mask_path'])
                if sha256(valid_path) != source.get('valid_mask_sha256'):
                    raise ValueError('Provider coverage mask hash differs')
                with Image.open(valid_path) as mask:
                    pixels = np.asarray(mask)
                if pixels.shape != valid.shape or not np.isin(pixels, [0, 255]).all():
                    raise ValueError('Provider coverage mask format differs')
                valid = pixels > 0
            rel = f'prepared/masks/original_valid/{row["token"]}.png'
            write_bytes(inside(root, rel), png_bytes(valid.astype(np.uint8)*255))
            row.update(original_valid_mask_path=rel, original_valid_mask_sha256=sha256(inside(root, rel)))
            cameras.append(row); rows.append(row)
        stations.append(dict(token=token, station_id=station['station_id'], pano_id=station['pano_id'], frames=cameras))
    width = options['flux']['erp_width']
    # Prepare complete, unedited ERP images before any whole-panorama model call.
    # Native photographs remain the immutable final RGB target grid.
    for index, station in enumerate(stations):
        images = [_rgb(inside(root, row['original_file_path'])) for row in station['frames']]
        original = cube_to_erp(images, station['frames'], width, options['chunk_rows'])
        original_rel = 'prepared/panoramas/' + station['token'] + '/original.png'
        write_bytes(inside(root, original_rel), png_bytes(original))
        station.update(original_erp_path=original_rel, original_erp_sha256=sha256(inside(root, original_rel)),
            erp_width=width, erp_height=width//2)
        print(json.dumps(dict(stage='original_erp', completed=index+1, total=len(stations))), flush=True)
    mask_request = inside(root, 'prepared/mask_request.json')
    write_json(mask_request, dict(root=str(root), rows=rows, sam=options['sam_segmentation'], sky=options['sky_segmentation'],
        sky_instance_score_threshold=options['sky_instance_score_threshold'], object_projection=options['object_projection'],
        object_dilation_at_2048=options['object_dilation_at_2048'], stations=stations, chunk_rows=options['chunk_rows']))
    if options.get('native_semantic_cache') is not None:
        from .panorama_native_cache import reuse_native_semantics
        reuse_native_semantics(mask_request, options['native_semantic_cache'], root)
    else:
        _worker('masks', mask_request, options['sam_segmentation'].get('python_executable', sys.executable))
    mask_manifest = json.loads(inside(root, 'prepared/semantic_manifest.json').read_text(encoding='utf8'))
    if mask_manifest.get('status') != 'completed' or mask_manifest.get('request_sha256') != sha256(mask_request):
        raise ValueError('Native semantic worker did not complete the bound request')
    mask_rows = {row['token']: row for row in mask_manifest['rows']}
    if len(mask_rows) != len(mask_manifest['rows']) or set(mask_rows) != {row['token'] for row in rows}:
        raise ValueError('Native semantic output roster differs')
    erp_mask_rows = {row['token']: row for row in mask_manifest.get('erp_rows', [])}
    if options['object_projection'] == 'whole_erp' and (len(erp_mask_rows) != len(stations) or set(erp_mask_rows) != {row['token'] for row in stations}):
        raise ValueError('Whole-panorama object evidence roster differs')
    generated_requests = []
    for station in stations:
        dynamic = []; sky = []
        for row in station['frames']:
            record = mask_rows[row['token']]
            path = inside(root, record['path'])
            if sha256(path) != record['sha256'] or record['source_sha256'] != row['original_sha256']:
                raise ValueError('Semantic evidence source/hash differs')
            with np.load(path, allow_pickle=False) as evidence:
                dynamic.append(evidence['dynamic']); sky.append(evidence['sky_region'])
        if options['object_projection'] == 'whole_erp':
            erp_record = erp_mask_rows[station['token']]
            erp_path = inside(root, erp_record['path'])
            if sha256(erp_path) != erp_record['sha256'] or erp_record['source_sha256'] != station['original_erp_sha256']:
                raise ValueError('Whole-panorama object evidence/source hash differs')
            with np.load(erp_path, allow_pickle=False) as evidence:
                erp_dynamic = evidence['dynamic'].copy()
            if erp_dynamic.dtype != bool or erp_dynamic.shape != (width//2, width):
                raise ValueError('Whole-panorama dynamic mask grid differs')
        else:
            erp_dynamic = masks_to_erp(dynamic, station['frames'], width)
        core = erp_dynamic & ~masks_to_erp(sky, station['frames'], width)
        if not processing['mask_dynamic']:
            core[:] = False
        alpha, _ = spherical_alpha(core, settings=FeatherSettings(feather_ratio=options['feather_at_2048']/2048,
            dilation_ratio=options['dilation_at_2048']/2048, chunk_rows=options['chunk_rows']))
        prefix = 'prepared/panoramas/' + station['token']
        original_rel, alpha_rel = station['original_erp_path'], prefix+'/alpha.npy'
        if sha256(inside(root, original_rel)) != station['original_erp_sha256']:
            raise ValueError('Original full panorama changed after object inference')
        _save_npy(inside(root, alpha_rel), alpha)
        write_bytes(inside(root, prefix+'/edit_core.png'), png_bytes(core.astype(np.uint8)*255))
        station.update(original_erp_path=original_rel, alpha_erp_path=alpha_rel)
        generated_requests.append(dict(token=station['token'], station_id=station['station_id'],
            original_path=original_rel, original_sha256=sha256(inside(root, original_rel)),
            output_path=prefix+'/generated.png', seed=seed_for(station['station_id'], options['flux']['seed']), edit_requested=bool(core.any())))
    flux_request = inside(root, 'prepared/flux_request.json')
    write_json(flux_request, dict(root=str(root), rows=generated_requests, options=options['flux']))
    _worker('flux', flux_request, options['flux']['python_executable'])
    generated = json.loads(inside(root, 'prepared/generation_manifest.json').read_text(encoding='utf8'))
    if generated.get('status') != 'completed' or generated.get('request_sha256') != sha256(flux_request):
        raise ValueError('FLUX worker did not complete the bound whole-panorama request')
    generated_by_token = {row['token']: row for row in generated['rows']}
    if len(generated_by_token) != len(generated['rows']) or set(generated_by_token) != {row['token'] for row in stations}:
        raise ValueError('Generated panorama roster differs')
    frames = []
    matrix = np.asarray(options['common_transform'], np.float64)
    for station in stations:
        record = generated_by_token[station['token']]
        raw_path = inside(root, record['path'])
        if sha256(raw_path) != record['sha256']:
            raise ValueError('Generated panorama hash differs')
        raw = _rgb(raw_path, (width, width//2)); original_erp = _rgb(inside(root, station['original_erp_path']))
        alpha = np.load(inside(root, station['alpha_erp_path']), allow_pickle=False)
        complete, _ = compose_erp(original_erp, raw, alpha, matrix)
        write_bytes(inside(root, f'prepared/panoramas/{station["token"]}/composite.png'), png_bytes(complete))
        for source in station['frames']:
            native = _rgb(inside(root, source['original_file_path']))
            uv, native_alpha = native_uv_alpha(source, alpha, chunk_rows=options['chunk_rows'])
            pixels, detail = compose_on_target_grid(native, raw, native_alpha, uv, matrix, (width, width//2))
            prefix = f'prepared/native/{source["token"]}'
            photo_rel, alpha_rel = prefix+'.png', prefix+'_alpha.npy'
            write_bytes(inside(root, photo_rel), png_bytes(pixels)); _save_npy(inside(root, alpha_rel), native_alpha)
            with Image.open(inside(root, source['original_valid_mask_path'])) as valid_image:
                valid = np.asarray(valid_image) > 0
            semantic = mask_rows[source['token']]
            with np.load(inside(root, semantic['path']), allow_pickle=False) as evidence:
                masks = dict(photometric=valid & ~evidence['sky_region'] if processing['remove_sky'] else valid,
                    sfm=valid & ~evidence['dynamic'] & ~evidence['sky_region'] & (native_alpha == 0),
                    sky=evidence['sky_high'], ground=evidence['ground'])
            paths = {}
            for name, value in masks.items():
                rel = f'prepared/masks/{name}/{source["token"]}.png'
                write_bytes(inside(root, rel), png_bytes(value.astype(np.uint8)*255)); paths[name] = (rel, sha256(inside(root, rel)))
            row = {k: v for k, v in source.items() if k != 'token'}
            row.update(file_path=photo_rel, source_sha256=sha256(inside(root, photo_rel)),
                edit_alpha_path=alpha_rel, edit_alpha_sha256=sha256(inside(root, alpha_rel)),
                mask_path=paths['photometric'][0], mask_sha256=paths['photometric'][1],
                sfm_mask_path=paths['sfm'][0], sfm_mask_sha256=paths['sfm'][1],
                sky_mask_path=paths['sky'][0], sky_mask_sha256=paths['sky'][1],
                ground_mask_path=paths['ground'][0], ground_mask_sha256=paths['ground'][1],
                semantic_path=semantic['path'], semantic_sha256=semantic['sha256'],
                semantic_source_sha256=source['original_sha256'], original_outside_alpha_exact=True,
                composite_report=detail, mask_fractions={name: float(value.mean()) for name, value in masks.items()})
            frames.append(row)
        print(json.dumps(dict(stage='preprocess', completed_frames=len(frames), total_frames=len(rows))), flush=True)
    if sha256(source_path) != collection_hash or any(sha256(inside(root, name)) != value for name, value in inputs.items()):
        raise ValueError('Original collection changed during preprocessing')
    if any(sha256(inside(root, row['path'])) != row['sha256'] for row in [*mask_rows.values(), *erp_mask_rows.values()]) or any(
            sha256(inside(root, row['path'])) != row['sha256'] for row in generated_by_token.values()):
        raise ValueError('Bound semantic or generated inputs changed during composition')
    manifest = dict(schema_version=1, stage='preprocess', status='complete', workflow='panorama_brush_refine',
        input_fingerprint=fingerprint(dict(config=config, collection_sha256=collection_hash, options=options)),
        collection_manifest_path='collection/manifest.json', collection_sha256=collection_hash,
        frames=frames, stations=[{k:v for k,v in station.items() if k != 'faces'} for station in collection['stations']],
        processing_options=processing, camera_convention='opencv', options=options,
        semantic_manifest_sha256=sha256(inside(root, 'prepared/semantic_manifest.json')),
        generation_manifest_sha256=sha256(inside(root, 'prepared/generation_manifest.json')),
        original_outside_alpha_exact=True, source_images_preserved=True, per_image_fits=0,
        native_training_resolution_preserved=True, generated_regions_are_geometry_evidence=False,
        shadow_mask_added=False, sky_rgb_included=not processing['remove_sky'],
        object_inference_projection=options['object_projection'],
        object_inference_images=len(stations) if options['object_projection']=='whole_erp' else len(rows),
        native_semantic_cache_used=options.get('native_semantic_cache') is not None)
    write_json(destination, manifest)
    return manifest


def mask_worker(request_path):
    from .preprocess import model_provenance, mask_policy, build_group_masks, build_masks, HFSemanticSegmenter
    from .sam3_segmenter import Sam3EvidenceSegmenter
    request = json.loads(Path(request_path).read_text(encoding='utf8')); root = Path(request['root'])
    if request.get('object_projection', 'native_cubes') == 'whole_erp':
        return panorama_mask_worker(request_path)
    sam_options, sky_options = request['sam'], request['sky']
    sam = Sam3EvidenceSegmenter(sam_options, model_provenance(sam_options))
    sky = HFSemanticSegmenter(sky_options, model_provenance(sky_options))
    sam_policy, sky_policy = mask_policy(sam_options), mask_policy(sky_options)
    rows = []
    for row in request['rows']:
        image_path = inside(root, row['original_file_path'])
        if sha256(image_path) != row['original_sha256']:
            raise ValueError('Semantic source RGB hash changed')
        photo = Image.fromarray(_rgb(image_path, (row['w'], row['h'])))
        evidence = sam.predict_evidence(photo); probabilities = evidence['group_probability']
        sam_masks = build_group_masks(probabilities, sam_policy, {'mask_dynamic': True, 'remove_sky': True})
        labels, confidence = sky.predict(photo)
        sky_masks = build_masks(labels, confidence, sky.groups, sky_policy, {'mask_dynamic': True, 'remove_sky': True})
        stream = BytesIO()
        np.savez_compressed(stream, dynamic=sam_masks['dynamic'], sky_region=sky_masks['sky_region'],
            sky_high=gated_sky(evidence, request['sky_instance_score_threshold']), ground=sam_masks['ground'],
            group_dynamic=np.asarray(probabilities['dynamic'], np.float32), group_sky=np.asarray(probabilities['sky'], np.float32),
            instance_masks_packed=evidence['instance_masks_packed'], instance_masks_shape=evidence['instance_masks_shape'],
            evidence_metadata_json=np.asarray(json.dumps(evidence['metadata'], sort_keys=True, allow_nan=False)),
            source_sha256=np.asarray(row['original_sha256']))
        rel = f'prepared/semantics/{row["token"]}.npz'; write_bytes(inside(root, rel), stream.getvalue())
        rows.append(dict(token=row['token'], path=rel, sha256=sha256(inside(root, rel)), source_sha256=row['original_sha256']))
        print(json.dumps(dict(stage='native_semantics', completed=len(rows), total=len(request['rows']))), flush=True)
    write_json(inside(root, 'prepared/semantic_manifest.json'), dict(status='completed', request_sha256=sha256(request_path),
        rows=rows, sam=sam.metadata, sky=sky.metadata, sam_policy=sam_policy, sky_policy=sky_policy))


def panorama_mask_worker(request_path):
    """Infer objects on one complete ERP per station, then project masks only.

    SAM3 and its independent object verifier never see cube crops in this path.
    The separate broad-sky semantic guard keeps its original native camera input.
    SAM sky/ground evidence shares the single full-panorama object pass.
    """
    from time import perf_counter
    from .preprocess import model_provenance, mask_policy, build_group_masks, build_masks, HFSemanticSegmenter
    from .sam3_segmenter import Sam3EvidenceSegmenter
    from .panorama_objects import predict_panorama_objects
    from .panorama_mask_projection import project_erp_mask, projection_policy
    request = json.loads(Path(request_path).read_text(encoding='utf8')); root = Path(request['root'])
    sam_options, sky_options = request['sam'], request['sky']
    started = perf_counter()
    sam_provenance = model_provenance(sam_options)
    sky_provenance = model_provenance(sky_options)
    provenance_seconds = perf_counter()-started
    started = perf_counter(); sam = Sam3EvidenceSegmenter(sam_options, sam_provenance)
    sam_initialization_seconds = perf_counter()-started
    started = perf_counter(); sky = HFSemanticSegmenter(sky_options, sky_provenance)
    sky_initialization_seconds = perf_counter()-started
    base_policy, sky_policy = mask_policy(sam_options), mask_policy(sky_options)
    rows = []; erp_rows = []
    for station in request['stations']:
        station_started = perf_counter()
        width, height = station['erp_width'], station['erp_height']
        source = inside(root, station['original_erp_path'])
        if sha256(source) != station['original_erp_sha256']:
            raise ValueError('Whole-panorama object input changed')
        photo = Image.fromarray(_rgb(source, (width, height)))
        policy = dict(base_policy, dynamic_dilation_px=int(round(request['object_dilation_at_2048']*width/2048)))
        started = perf_counter()
        objects = predict_panorama_objects(photo, sam, policy, seam_roll=False)
        object_inference_seconds = perf_counter()-started
        evidence = objects['evidence']; probabilities = evidence['group_probability']
        masks = build_group_masks(probabilities, policy, {'mask_dynamic':True, 'remove_sky':True})
        dynamic = objects['dynamic']; high_sky = gated_sky(evidence, request['sky_instance_score_threshold'])
        ground = masks['ground'] & ~dynamic
        stream = BytesIO()
        np.savez_compressed(stream, dynamic=dynamic, dynamic_core=objects['dynamic_core'],
            sky_high=high_sky, ground=ground,
            group_dynamic=np.asarray(probabilities['dynamic'],np.float32),
            group_sky=np.asarray(probabilities['sky'],np.float32),
            group_ground=np.asarray(probabilities['ground'],np.float32),
            instance_masks_packed=evidence['instance_masks_packed'], instance_masks_shape=evidence['instance_masks_shape'],
            evidence_metadata_json=np.asarray(json.dumps(evidence['metadata'],sort_keys=True,allow_nan=False)),
            object_inference_metadata_json=np.asarray(json.dumps(objects['metadata'],sort_keys=True,allow_nan=False)),
            source_sha256=np.asarray(station['original_erp_sha256']))
        relative = f'prepared/erp_semantics/{station["token"]}.npz'
        write_bytes(inside(root,relative),stream.getvalue()); evidence_hash = sha256(inside(root,relative))
        erp_rows.append(dict(token=station['token'], path=relative, sha256=evidence_hash,
            source_path=station['original_erp_path'], source_sha256=station['original_erp_sha256'],
            w=width,h=height,object_inference_projection='whole_erp',object_inference=objects['metadata'],mask_policy=policy))
        broad_sky_seconds = 0.0
        for row in station['frames']:
            original_path = inside(root,row['original_file_path'])
            if sha256(original_path) != row['original_sha256']:
                raise ValueError('Native source for sky guard/projection changed')
            native = Image.fromarray(_rgb(original_path,(row['w'],row['h'])))
            started = perf_counter(); labels,confidence = sky.predict(native)
            broad_sky_seconds += perf_counter()-started
            broad = build_masks(labels,confidence,sky.groups,sky_policy,{'mask_dynamic':True,'remove_sky':True})
            native_dynamic = project_erp_mask(dynamic,row,chunk_rows=request['chunk_rows'])
            native_high_sky = project_erp_mask(high_sky,row,sampling='nearest',chunk_rows=request['chunk_rows'])
            native_ground = project_erp_mask(ground,row,sampling='nearest',chunk_rows=request['chunk_rows'])
            stream = BytesIO()
            np.savez_compressed(stream,dynamic=native_dynamic,sky_region=broad['sky_region'],
                sky_high=native_high_sky,ground=native_ground,
                source_sha256=np.asarray(row['original_sha256']),
                object_inference_projection=np.asarray('whole_erp'),
                erp_source_sha256=np.asarray(station['original_erp_sha256']),
                erp_evidence_path=np.asarray(relative),erp_evidence_sha256=np.asarray(evidence_hash),
                projection_policy_json=np.asarray(json.dumps(projection_policy(),sort_keys=True)),
                native_model_calls=np.asarray('broad sky semantic guard only; no native SAM3/RTDETR object inference'))
            rel = f'prepared/semantics/{row["token"]}.npz'
            write_bytes(inside(root,rel),stream.getvalue())
            rows.append(dict(token=row['token'],path=rel,sha256=sha256(inside(root,rel)),source_sha256=row['original_sha256'],
                object_inference_projection='whole_erp',erp_source_sha256=station['original_erp_sha256'],
                erp_evidence_path=relative,erp_evidence_sha256=evidence_hash))
        erp_rows[-1]['timing_seconds'] = dict(object_inference=object_inference_seconds,
            native_broad_sky_inference=broad_sky_seconds,total_station=perf_counter()-station_started)
        print(json.dumps(dict(stage='whole_erp_objects',completed=len(erp_rows),total=len(request['stations']),
            projected_native_faces=len(rows),timing_seconds=erp_rows[-1]['timing_seconds'])),flush=True)
    write_json(inside(root,'prepared/semantic_manifest.json'),dict(status='completed',request_sha256=sha256(request_path),
        rows=rows,erp_rows=erp_rows,sam=sam.metadata,sky=sky.metadata,sam_policy=base_policy,sky_policy=sky_policy,
        object_inference_projection='whole_erp',object_inference_images=len(erp_rows),
        native_object_inference_images=0,native_broad_sky_guard_images=len(rows),
        timing=dict(clock='perf_counter wall seconds; inference returns CPU mask evidence',
            object_scope='SAM3 plus RTDETR whole-ERP evidence and postprocessing, excluding model initialization and disk I/O',
            provenance_seconds=provenance_seconds,sam_and_verifier_initialization_seconds=sam_initialization_seconds,
            broad_sky_initialization_seconds=sky_initialization_seconds,
            object_inference_seconds=sum(row['timing_seconds']['object_inference'] for row in erp_rows),
            native_broad_sky_inference_seconds=sum(row['timing_seconds']['native_broad_sky_inference'] for row in erp_rows),
            station_total_seconds=sum(row['timing_seconds']['total_station'] for row in erp_rows)),
        sam_sky_ground_projection='same whole-ERP SAM pass, nearest native-center rays',
        dynamic_projection=projection_policy()))


def flux_worker(request_path):
    request = json.loads(Path(request_path).read_text(encoding='utf8')); root = Path(request['root']); options = request['options']
    width = options['erp_width']; height = width//2
    receipt_path = Path(options['model_receipt_path']); receipt = json.loads(receipt_path.read_text(encoding='utf8'))
    if options.get('model_receipt_sha256') is not None and sha256(receipt_path) != options['model_receipt_sha256']:
        raise ValueError('FLUX model receipt hash differs')
    if receipt.get('status') != 'downloaded_verified' or receipt.get('revision') != options['model_revision']:
        raise ValueError('FLUX model receipt/revision differs')
    model_path = Path(receipt['model_path']).resolve()
    for row in receipt['files']:
        path = inside(model_path, row['path'])
        if path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
            raise ValueError('FLUX pinned model cache changed')
    import torch
    from diffusers import Flux2KleinPipeline
    pipe = None; rows = []
    for row in request['rows']:
        source = inside(root, row['original_path'])
        if sha256(source) != row['original_sha256']:
            raise ValueError('FLUX original ERP changed')
        original = _rgb(source, (width, height))
        if row['edit_requested']:
            if pipe is None:
                pipe = Flux2KleinPipeline.from_pretrained(str(model_path), torch_dtype=torch.bfloat16, local_files_only=True).to('cuda')
                pipe.set_progress_bar_config(disable=True)
                if width % (pipe.vae_scale_factor*2) or height % (pipe.vae_scale_factor*2) or width*height > 2024*2024:
                    raise ValueError('FLUX conditioning size violates the pinned full-field processor contract')
                previous = pipe.image_processor.preprocess
                def no_auto_resize(image, target_area, *a, **kw):
                    if image.size != (width, height) or target_area != 1024*1024:
                        raise ValueError('Unexpected FLUX auto-resize request')
                    return image
                def exact_preprocess(image, *a, **kw):
                    if image.size != (width, height) or (kw.get('width'), kw.get('height')) != (width, height):
                        raise ValueError('FLUX changed full-field conditioning dimensions')
                    result = previous(image, *a, **kw)
                    if tuple(result.shape[-2:]) != (height, width):
                        raise ValueError('FLUX conditioning tensor shape differs')
                    return result
                pipe.image_processor._resize_to_target_area = no_auto_resize
                pipe.image_processor.preprocess = exact_preprocess
            generated = pipe(image=Image.fromarray(original), prompt=options['prompt'], height=height, width=width,
                num_inference_steps=options['steps'], guidance_scale=options['guidance_scale'],
                generator=torch.Generator(device='cuda').manual_seed(row['seed'])).images[0].convert('RGB')
            if generated.size != (width, height):
                raise ValueError('FLUX output aspect/size differs from whole input panorama')
            pixels = np.asarray(generated)
        else:
            pixels = original
        write_bytes(inside(root, row['output_path']), png_bytes(pixels))
        rows.append(dict(token=row['token'], path=row['output_path'], sha256=sha256(inside(root, row['output_path'])),
            seed=row['seed'], pipeline_calls=int(row['edit_requested']), original_sha256=row['original_sha256']))
        print(json.dumps(dict(stage='whole_erp_generation', completed=len(rows), total=len(request['rows']))), flush=True)
    write_json(inside(root, 'prepared/generation_manifest.json'), dict(status='completed', request_sha256=sha256(request_path),
        model_receipt_sha256=sha256(receipt_path), rows=rows, model_revision=options['model_revision'],
        input_output_size_wh=[width, height], conditioning_auto_resize=False, masking_during_inference=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('masks', 'flux')); parser.add_argument('--request', type=Path, required=True)
    arguments = parser.parse_args()
    (mask_worker if arguments.operation == 'masks' else flux_worker)(arguments.request)
