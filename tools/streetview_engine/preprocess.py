"""Real semantic inference with separate photometric, SfM, sky and road masks.

RGB photographs remain immutable. Missing model weights or unsupported device
fail the stage; no all-white replacement masks or hand-authored regions exist.
HF semantic inference follows the AutoImageProcessor/semantic-segmentation API:
https://huggingface.co/docs/transformers/tasks/semantic_segmentation
"""
from __future__ import annotations

from io import BytesIO
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageFilter

from .imaging import FACES, cube_camera_to_station_cv, fingerprint, inside, png_bytes, sha256, write_bytes, write_json
from .processing_options import read_processing_options
from .preprocess_identity import preprocess_input_fingerprint

DEFAULT_GROUPS = {
    'dynamic': ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'],
    'sky': ['sky'],
    'ground': ['road', 'sidewalk'],
}


def resolve_class_groups(id2label, requested=None):
    """Resolve explicit taxonomy names against this exact model, never pixels."""
    labels = {int(key): str(value).strip().casefold() for key, value in id2label.items()}
    if not labels or any(key < 0 for key in labels):
        raise ValueError('Model must expose its actual semantic id2label mapping')
    result = {}
    for group, names in (requested or DEFAULT_GROUPS).items():
        if group not in DEFAULT_GROUPS or not isinstance(names, (list, tuple)) or not names:
            raise ValueError('dynamic, sky and ground each require explicit model label names')
        matched = set()
        for name in names:
            if not isinstance(name, str) or not name.strip():
                raise ValueError('Configure semantic class names, not opaque class numbers')
            ids = [key for key, label in labels.items() if label == name.strip().casefold()]
            if not ids:
                raise ValueError(f'Semantic model does not contain configured class: {name}')
            matched.update(ids)
        result[group] = sorted(matched)
    if set(result) != set(DEFAULT_GROUPS):
        raise ValueError('All three semantic groups must be configured')
    if any(set(result[a]) & set(result[b]) for a, b in (('dynamic', 'sky'), ('dynamic', 'ground'), ('sky', 'ground'))):
        raise ValueError('Semantic groups must be disjoint')
    return result


def mask_policy(options):
    thresholds = {name: float(options.get(name, default)) for name, default in (
        ('dynamic_confidence', .6), ('sky_confidence', .8),
        ('sky_exclusion_confidence', .5), ('ground_confidence', .8))}
    if any(not math.isfinite(value) or not 0 < value <= 1 for value in thresholds.values()):
        raise ValueError('Semantic confidence thresholds must lie in (0,1]')
    if thresholds['sky_exclusion_confidence'] > thresholds['sky_confidence']:
        raise ValueError('Sky exclusion threshold cannot exceed sky-core threshold')
    for name, default in (('dynamic_dilation_px', 3), ('core_erosion_px', 2)):
        value = options.get(name, default)
        if type(value) is not int or not 0 <= value <= 32:
            raise ValueError('Morphology radii must be integer output pixels in 0..32')
        thresholds[name] = value
    return thresholds


def _morph(mask, radius, dilate):
    if radius == 0:
        return mask.copy()
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    kernel = ImageFilter.MaxFilter if dilate else ImageFilter.MinFilter
    return np.asarray(image.filter(kernel(2 * radius + 1))) > 0


def mask_conventions(processing_options):
    dynamic = 'excluded' if processing_options['mask_dynamic'] else 'included'
    sky = 'excluded' if processing_options['remove_sky'] else 'included'
    return dict(mask_path=f'255 photometric-valid; predicted dynamic {dynamic}; sky-region {sky}',
        sfm_mask_path=f'255 geometry-valid; predicted dynamic {dynamic}; sky-region always excluded',
        sky_mask_path='255 confident eroded sky evidence, independent of photometric toggles',
        ground_mask_path='255 confident eroded model-labelled ground; not a measured plane')


def build_masks(labels, confidence, groups, policy, processing_options=None):
    """Select loss/feature masks without changing RGB or semantic evidence."""
    processing = read_processing_options({'processing_options': processing_options} if processing_options is not None else {})
    labels = np.asarray(labels)
    confidence = np.asarray(confidence)
    if labels.ndim != 2 or labels.shape != confidence.shape or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('Semantic outputs must be integer labels and same-size confidence [H,W]')
    if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)) or np.any(labels < 0):
        raise ValueError('Invalid semantic probabilities or class IDs')
    if set(groups) != set(DEFAULT_GROUPS) or any(not values for values in groups.values()):
        raise ValueError('Three actual semantic class groups are required')
    dynamic = np.isin(labels, groups['dynamic']) & (confidence >= policy['dynamic_confidence'])
    dynamic = _morph(dynamic, policy['dynamic_dilation_px'], True)
    sky_region = np.isin(labels, groups['sky']) & (confidence >= policy['sky_exclusion_confidence'])
    sky = np.isin(labels, groups['sky']) & (confidence >= policy['sky_confidence']) & ~dynamic
    ground = np.isin(labels, groups['ground']) & (confidence >= policy['ground_confidence']) & ~dynamic & ~sky_region
    sky = _morph(sky, policy['core_erosion_px'], False)
    ground = _morph(ground, policy['core_erosion_px'], False)
    valid = ~dynamic if processing['mask_dynamic'] else np.ones(labels.shape, bool)
    photometric = valid & ~sky_region if processing['remove_sky'] else valid.copy()
    return dict(photometric=photometric, sfm=valid & ~sky_region, sky=sky,
                ground=ground, dynamic=dynamic, sky_region=sky_region)


def build_group_masks(probabilities, policy, processing_options=None):
    """Apply mask policy to separate, possibly overlapping concept evidence.

    These are model mask scores, not a categorical semantic softmax. Dynamic
    evidence takes precedence over sky/ground support; uncertainty never
    establishes a ground surface.
    """
    if not isinstance(probabilities, dict) or set(probabilities) != set(DEFAULT_GROUPS):
        raise ValueError('Require separate dynamic, sky and ground evidence')
    values = {key: np.asarray(value) for key, value in probabilities.items()}
    shapes = {value.shape for value in values.values()}
    if len(shapes) != 1 or any(value.ndim != 2 or not np.isfinite(value).all()
            or np.any((value < 0) | (value > 1)) for value in values.values()):
        raise ValueError('Concept evidence must be matching finite [H,W] scores in [0,1]')
    processing = read_processing_options({'processing_options': processing_options} if processing_options is not None else {})
    # SAM's official instance postprocessor uses a strict pixel threshold.
    # Keep exact ties consistent with the retained packed instance masks.
    dynamic = _morph(values['dynamic'] > policy['dynamic_confidence'], policy['dynamic_dilation_px'], True)
    sky_region = values['sky'] > policy['sky_exclusion_confidence']
    sky = _morph((values['sky'] > policy['sky_confidence']) & ~dynamic, policy['core_erosion_px'], False)
    ground = _morph((values['ground'] > policy['ground_confidence']) & ~dynamic & ~sky_region,
                    policy['core_erosion_px'], False)
    valid = ~dynamic if processing['mask_dynamic'] else np.ones(dynamic.shape, bool)
    return dict(photometric=valid & ~sky_region if processing['remove_sky'] else valid.copy(),
                sfm=valid & ~sky_region, sky=sky, ground=ground, dynamic=dynamic, sky_region=sky_region)


def segmentation_backend(options):
    backend = options.get('backend', 'hf_semantic')
    if backend not in ('hf_semantic', 'sam3'):
        raise ValueError('Unsupported segmentation backend; no model fallback is allowed')
    return backend


def run_in_configured_runtime(config, root, settings, semantic):
    """Run a whole stage in the configured isolated Python, loading SAM once.

    The existing job supervisor retains ownership of this descendant. No
    shell, detached process, token transport or per-image model restart is used.
    """
    executable = semantic.get('python_executable')
    if executable is None:
        return None
    target = Path(executable).expanduser()
    if not target.is_absolute() or not target.is_file():
        raise ValueError('Segmentation python_executable must be an existing absolute Python path')
    # Preserve venv spelling: resolving symlinks could equate unrelated venvs.
    if os.path.normcase(os.path.abspath(target)) == os.path.normcase(os.path.abspath(sys.executable)):
        return None
    code_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(code_root) + os.pathsep + environment.get('PYTHONPATH', '')
    environment.update(PYTHONUNBUFFERED='1', PYTHONUTF8='1')
    code = ('import json,sys; from pathlib import Path; '
            'from tools.streetview_engine.preprocess import run; '
            'p=json.load(sys.stdin); run(p["config"],Path(p["root"]),p["settings"])')
    payload = json.dumps(dict(config=config, root=str(root), settings=settings), allow_nan=False)
    subprocess.run([str(target), '-B', '-c', code], input=payload, text=True, encoding='utf8',
                   cwd=code_root, env=environment, check=True, shell=False)
    manifest = json.loads((root/'prepared/manifest.json').read_text(encoding='utf8'))
    if manifest.get('status') != 'complete':
        raise RuntimeError('Isolated segmentation runtime did not complete preprocessing')
    return manifest


def model_provenance(options):
    backend = segmentation_backend(options)
    model_path = Path(options.get('model_path', '')).expanduser()
    if not options.get('model_path') or not model_path.is_dir():
        raise RuntimeError('Preprocessing requires configured local semantic model weights: preprocess.segmentation.model_path')
    processor_path = Path(options.get('processor_path', model_path)).expanduser()
    if not processor_path.is_dir():
        raise RuntimeError('Configured local image-processor directory does not exist')
    sources = {}
    for name, directory in (('model', model_path), ('processor', processor_path)):
        rows = []
        for path in sorted(directory.rglob('*')):
            if path.is_file() and (path.suffix in ('.json', '.safetensors') or path.name.startswith('pytorch_model') and path.suffix == '.bin'
                    or backend == 'sam3' and path.suffix in ('.txt', '.model', '.tiktoken', '.gz')):
                rows.append(dict(file=path.relative_to(directory).as_posix(), sha256=sha256(path), bytes=path.stat().st_size))
        sources[name] = rows
    if not any(row['file'].endswith(('.safetensors', '.bin')) for row in sources['model']):
        raise RuntimeError('Configured semantic model directory contains no local model weights')
    result = dict(model_path=str(model_path.resolve()), processor_path=str(processor_path.resolve()), files=sources,
                  files_sha256=fingerprint(sources), local_files_only=True, trust_remote_code=False)
    if options.get('instance_verifier') is not None:
        if backend != 'sam3' or not isinstance(options['instance_verifier'], dict):
            raise ValueError('Independent instance verification requires SAM3 and explicit options')
        from .object_verifier import verifier_provenance
        result['instance_verifier'] = verifier_provenance(options['instance_verifier'])
        result['files_sha256'] = fingerprint(dict(segmentation_files=sources, instance_verifier=result['instance_verifier']))
    return result


class HFSemanticSegmenter:
    def __init__(self, options, provenance):
        if sys.platform == 'win32' and not options.get('allow_windows_inference', False):
            raise RuntimeError('Run semantic inference on the configured cloud/Linux host; local Windows inference is disabled')
        device = str(options.get('device', 'cuda'))
        if device == 'cpu' and not options.get('allow_cpu', False):
            raise RuntimeError('CPU semantic inference requires explicit operator allow_cpu; prefer configured cloud CUDA')
        if device != 'cpu' and not (device == 'cuda' or device.startswith('cuda:')):
            raise ValueError('Unsupported semantic inference device')
        size = options.get('inference_size', 512)
        if type(size) is not int or not 128 <= size <= 2048:
            raise ValueError('Inference size must be 128..2048 pixels')
        import torch
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('Configured CUDA semantic inference is unavailable; no CPU fallback was run')
        self.torch = torch
        self.device, self.size = device, size
        self.processor = AutoImageProcessor.from_pretrained(provenance['processor_path'], local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(provenance['model_path'], local_files_only=True, trust_remote_code=False).to(device).eval()
        self.groups = resolve_class_groups(self.model.config.id2label, options.get('class_groups'))
        self.metadata = dict(backend='transformers.AutoModelForSemanticSegmentation', architecture=self.model.__class__.__name__,
            device=device, inference_size=size, id2label={str(k): v for k, v in self.model.config.id2label.items()}, resolved_class_groups=self.groups,
            model_provenance=provenance, prediction='argmax of bilinearly upsampled logits; confidence is the corresponding softmax probability')

    def predict(self, image):
        torch = self.torch
        resized = image.convert('RGB').resize((self.size, self.size), Image.Resampling.BILINEAR)
        inputs = self.processor(images=resized, do_resize=False, return_tensors='pt')
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits
            if logits.ndim != 4 or logits.shape[0] != 1 or not torch.isfinite(logits).all():
                raise RuntimeError('Semantic model returned invalid logits')
            logits = torch.nn.functional.interpolate(logits.float(), size=(image.height, image.width), mode='bilinear', align_corners=False)
            confidence, labels = logits.softmax(1).max(1)
        return labels[0].cpu().numpy().astype(np.int32), confidence[0].cpu().numpy().astype(np.float32)


def run(config: dict, job_dir: Path, settings: dict) -> dict:
    processing = read_processing_options(config)
    options = dict(settings.get('preprocess', settings))
    semantic = dict(options.get('segmentation', options))
    backend = segmentation_backend(semantic)
    root = Path(job_dir).resolve()
    delegated = run_in_configured_runtime(config, root, settings, semantic)
    if delegated is not None:
        return delegated
    policy = mask_policy(semantic)
    provenance = model_provenance(semantic)
    source_path = inside(root, 'collection/manifest.json')
    collection = json.loads(source_path.read_text(encoding='utf8'))
    if collection.get('status') != 'complete' or collection.get('face_order') != list(FACES):
        raise ValueError('Preprocessing requires a complete six-face native collection')
    if set(config.get('panorama_ids', [])) != {station['pano_id'] for station in collection['stations']}:
        raise ValueError('Collection and current frozen panorama selection differ')
    signature = preprocess_input_fingerprint(config, sha256(source_path), semantic, policy, provenance)
    destination = inside(root, 'prepared/manifest.json')
    if destination.exists():
        prior = json.loads(destination.read_text(encoding='utf8'))
        if prior.get('input_fingerprint') != signature or prior.get('status') != 'complete' or read_processing_options(prior) != processing:
            raise ValueError('Prepared masks belong to different inputs/model/settings')
        for frame in prior['frames']:
            for path_key, hash_key in (('file_path', 'source_sha256'), ('mask_path', 'mask_sha256'), ('sfm_mask_path', 'sfm_mask_sha256'), ('sky_mask_path', 'sky_mask_sha256'), ('ground_mask_path', 'ground_mask_sha256'), ('semantic_path', 'semantic_sha256')):
                if sha256(inside(root, frame[path_key])) != frame[hash_key]:
                    raise ValueError('Prepared artifact changed after completion')
        return prior
    write_json(inside(root, 'prepared/input.json'), dict(input_fingerprint=signature, policy=policy, semantic_options=semantic, model_provenance=provenance, processing_options=processing))
    if backend == 'sam3':
        from .sam3_segmenter import Sam3EvidenceSegmenter
        segmenter = Sam3EvidenceSegmenter(semantic, provenance)
    else:
        segmenter = HFSemanticSegmenter(semantic, provenance)
    frames = []
    for station in collection['stations']:
        for face in FACES:
            source = station['faces'][face]
            source_file = inside(root, source['file_path'])
            if sha256(source_file) != source['sha256']:
                raise ValueError('Original collected face changed before preprocessing')
            with Image.open(source_file) as photo:
                photo.load()
                if photo.size != (source['w'], source['h']) or photo.width != photo.height:
                    raise ValueError('Native cube face dimensions differ from collection manifest')
                width, height = photo.size
                if backend == 'sam3':
                    evidence = segmenter.predict_evidence(photo)
                    probabilities = evidence['group_probability']
                    masks = build_group_masks(probabilities, policy, processing)
                    evidence_values = {f'group_{key}': np.asarray(value, np.float32) for key, value in probabilities.items()}
                    evidence_values['evidence_schema'] = np.asarray(segmenter.metadata.get('evidence_schema', 'sam3_group_evidence_v1'))
                    packed = np.asarray(evidence['instance_masks_packed'])
                    packed_shape = np.asarray(evidence['instance_masks_shape'])
                    if (packed.dtype != np.uint8 or packed_shape.shape != (3,) or not np.issubdtype(packed_shape.dtype, np.integer)
                            or tuple(packed_shape[1:]) != (height, width) or packed_shape[0] < 0
                            or packed.shape != (int(packed_shape[0]), (height*width+7)//8)):
                        raise ValueError('SAM3 packed instance evidence does not match source dimensions')
                    evidence_values.update(instance_masks_packed=packed, instance_masks_shape=packed_shape)
                    evidence_values['evidence_metadata_json'] = np.asarray(json.dumps(
                        evidence['metadata'],
                        sort_keys=True, allow_nan=False))
                else:
                    labels, confidence = segmenter.predict(photo)
                    masks = build_masks(labels, confidence, segmenter.groups, policy, processing)
                    evidence_values = dict(labels=labels.astype(np.int32), confidence=confidence.astype(np.float32))
            if masks['photometric'].shape != (height, width):
                raise ValueError('Semantic prediction does not cover the complete source photograph')
            token = 'pano_' + fingerprint(station['pano_id'])[:20] + '_' + face
            paths = {}
            for name in ('photometric', 'sfm', 'sky', 'ground'):
                relative = f'prepared/masks/{name}/{token}.png'
                target = inside(root, relative)
                write_bytes(target, png_bytes(masks[name].astype(np.uint8) * 255))
                paths[name] = (relative, sha256(target))
            semantic_relative = f'prepared/semantics/{token}.npz'
            output = BytesIO()
            evidence_values.update(source_sha256=np.asarray(source['sha256']), model_files_sha256=np.asarray(provenance['files_sha256']))
            np.savez_compressed(output, **evidence_values)
            semantic_file = inside(root, semantic_relative)
            # NPZ ZIP timestamps are not relied on for resumability: verify
            # semantic values before retaining a prior interrupted-stage file.
            if semantic_file.exists():
                with np.load(semantic_file, allow_pickle=False) as old:
                    if set(old.files) != set(evidence_values) or any(
                            not (np.allclose(old[key], value, rtol=0, atol=1e-6) if np.issubdtype(value.dtype, np.floating)
                                 else np.array_equal(old[key], value)) for key, value in evidence_values.items()):
                        raise ValueError('Existing semantic evidence differs; use a new prepared run')
            else:
                write_bytes(semantic_file, output.getvalue())
            frames.append(dict(file_path=source['file_path'], source_sha256=source['sha256'],
                mask_path=paths['photometric'][0], mask_sha256=paths['photometric'][1],
                sfm_mask_path=paths['sfm'][0], sfm_mask_sha256=paths['sfm'][1],
                sky_mask_path=paths['sky'][0], sky_mask_sha256=paths['sky'][1],
                ground_mask_path=paths['ground'][0], ground_mask_sha256=paths['ground'][1],
                semantic_path=semantic_relative, semantic_sha256=sha256(semantic_file),
                w=width, h=height, fl_x=width/2, fl_y=height/2, cx=width/2, cy=height/2,
                station_id=station['station_id'], pano_id=station['pano_id'], face=face,
                camera_to_station_cv=cube_camera_to_station_cv(face).tolist(),
                mask_fractions={name: float(mask.mean()) for name, mask in masks.items()}))
            if sha256(source_file) != source['sha256']:
                raise ValueError('Original photograph changed during preprocessing')
            print(json.dumps(dict(stage='preprocess', completed_frames=len(frames), total_frames=len(collection['stations'])*6)), flush=True)
    stations = [{key: value for key, value in station.items() if key != 'faces'} for station in collection['stations']]
    manifest = dict(schema_version=1, stage='preprocess', status='complete', input_fingerprint=signature,
        collection_manifest_path='collection/manifest.json', collection_sha256=sha256(source_path),
        frames=frames, stations=stations, camera_convention='opencv', station_frame='front_camera_opencv_X_right_Y_down_Z_forward',
        pose_note='camera_to_station_cv is a local cube rotation only; each pano_id has a separate SfM rig; station_id groups physical evidence',
        intrinsics_note='90 degree native cube; pixel-edge principal point w/2,h/2',
        mask_conventions=mask_conventions(processing), processing_options=processing,
        semantic_model=segmenter.metadata, mask_policy=policy, source_images_preserved=True,
        limitations=['Model labels/confidence are estimates; no hand-drawn floor or rig masks are inserted.',
            'Unknown semantic regions are not certified ground. Ground masks do not constrain ramps/stairs into a plane.',
            'No all-white replacement masks are emitted when inference cannot run.'])
    write_json(destination, manifest)
    return manifest
