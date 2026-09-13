"""Local-weight SAM3 concept evidence, with explicit instance-presence gating.

These planes are mask evidence scores, not a mutually exclusive class softmax
or calibrated semantic certainty. This module does not edit RGB, infer a plane,
remap labels by cube face, or read panorama IDs/locations.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import math
from pathlib import Path
import sys

import numpy as np


DEFAULT_CONCEPT_GROUPS = {
    'dynamic': ['person', 'car', 'bus', 'truck', 'van', 'motorcycle', 'bicycle'],
    'sky': ['sky'],
    'ground': ['road', 'sidewalk'],
}

DEFAULT_DETECTOR_LABELS = {
    'person': ['person'], 'car': ['car'], 'bus': ['bus'], 'truck': ['truck'],
    'van': ['car', 'truck'], 'motorcycle': ['motorcycle', 'motorbike'], 'bicycle': ['bicycle'],
}


def concept_groups(options):
    requested = options.get('concept_groups', DEFAULT_CONCEPT_GROUPS)
    if not isinstance(requested, dict) or set(requested) != set(DEFAULT_CONCEPT_GROUPS):
        raise ValueError('SAM3 requires explicit dynamic, sky and ground concept groups')
    result = {}
    seen = set()
    for group, prompts in requested.items():
        if not isinstance(prompts, (list, tuple)) or not 1 <= len(prompts) <= 32:
            raise ValueError('Each SAM3 concept group requires 1..32 text prompts')
        result[group] = []
        for prompt in prompts:
            if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 128 or '\x00' in prompt:
                raise ValueError('Invalid SAM3 concept text')
            prompt = prompt.strip()
            key = prompt.casefold()
            if key in seen:
                raise ValueError('SAM3 concepts must be distinct across the three groups')
            seen.add(key)
            result[group].append(prompt)
    return result


def probability_option(options, name, default):
    value = options.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
        raise ValueError(f'{name} must be a finite score threshold in (0,1)')
    return float(value)


def verification_policy(options, groups):
    if options.get('backend') != 'rtdetr_v2':
        raise ValueError('SAM3 instance verification requires the configured rtdetr_v2 backend')
    requested = options.get('concept_labels', DEFAULT_DETECTOR_LABELS)
    if not isinstance(requested, dict):
        raise ValueError('Verifier concept_labels must map every dynamic concept to detector labels')
    labels = {}
    for concept in groups['dynamic']:
        names = requested.get(concept)
        if not isinstance(names, (list, tuple)) or not names or any(
                not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError('Missing detector labels for SAM3 concept: ' + concept)
        labels[concept] = sorted(set(name.strip().casefold() for name in names))
    return dict(concept_labels=labels,
                minimum_box_iou=probability_option(options, 'minimum_box_iou', .25),
                minimum_mask_inside_box_fraction=probability_option(options, 'minimum_mask_inside_box_fraction', .5))


def verify_dynamic_instance(concept, box, binary_mask, detections, policy):
    """Require independent object evidence without cropping or painting a mask."""
    mask = np.asarray(binary_mask)
    if mask.ndim != 2 or mask.dtype != bool:
        raise ValueError('Object verification requires a binary HxW instance mask')
    height, width = mask.shape

    def clipped(value):
        value = np.asarray(value, np.float64)
        if value.shape != (4,) or not np.isfinite(value).all() or value[2] <= value[0] or value[3] <= value[1]:
            raise ValueError('Object verifier received an invalid XYXY box')
        return np.clip(value, [0, 0, 0, 0], [width, height, width, height])

    candidate = np.asarray(box, np.float64)
    if candidate.shape != (4,) or not np.isfinite(candidate).all():
        raise ValueError('SAM3 returned a malformed or nonfinite candidate box')
    if candidate[2] <= candidate[0] or candidate[3] <= candidate[1]:
        # A quantized model can produce a zero-width/height candidate. It has
        # no spatial support to verify; retain it as rejected evidence rather
        # than inventing a box or allowing it to erase the photograph.
        return dict(status='unsupported', accepted=False, reason='degenerate_candidate_box',
                    matched_detection_indices=[], best_match=None,
                    matching_class_family=policy['concept_labels'][concept])
    candidate = clipped(candidate)
    candidate_area = max(0., candidate[2]-candidate[0]) * max(0., candidate[3]-candidate[1])
    mask_area = int(mask.sum())
    matches = []
    for index, detection in enumerate(detections):
        target = clipped(detection['box_xyxy'])
        if not isinstance(detection.get('label'), str) or not math.isfinite(detection['score']) or not 0 <= detection['score'] <= 1:
            raise ValueError('Object verifier returned invalid label/score')
        if detection['label'].casefold() not in policy['concept_labels'][concept]:
            continue
        low = np.maximum(candidate[:2], target[:2])
        high = np.minimum(candidate[2:], target[2:])
        intersection = float(np.maximum(0., high-low).prod())
        target_area = max(0., target[2]-target[0]) * max(0., target[3]-target[1])
        union = candidate_area + target_area - intersection
        iou = intersection/union if union > 0 else 0.
        # Pixel centers use the same edge-coordinate convention as the boxes.
        x0, y0 = np.maximum(0, np.ceil(target[:2]-.5).astype(np.int64))
        x1, y1 = np.minimum([width, height], np.ceil(target[2:]-.5).astype(np.int64))
        inside = int(mask[y0:max(y0, y1), x0:max(x0, x1)].sum()) / mask_area if mask_area else 0.
        matches.append(dict(detection_index=index, box_iou=iou, mask_inside_box_fraction=inside))
    supported = [row for row in matches if row['box_iou'] >= policy['minimum_box_iou']
                 and row['mask_inside_box_fraction'] >= policy['minimum_mask_inside_box_fraction']]
    chosen = max(supported or matches, key=lambda row: (row['box_iou'], row['mask_inside_box_fraction']), default=None)
    return dict(status='supported' if supported else 'unsupported', accepted=bool(supported),
                reason='independent_detection_support' if supported else 'no_matching_detection' if not matches else 'insufficient_spatial_support',
                matched_detection_indices=[row['detection_index'] for row in supported],
                best_match=chosen, matching_class_family=policy['concept_labels'][concept])


class Sam3EvidenceSegmenter:
    """One in-process model per stage; image embeddings reused across concepts.

    Call ``predict_evidence(PIL.Image)`` for float32 HxW group planes and packed
    per-instance binary masks. Inputs are immutable; only the processor's
    internal image tensor is resized to the checkpoint's native resolution.
    """

    def __init__(self, options, provenance):
        if sys.platform == 'win32' and not options.get('allow_windows_inference', False):
            raise RuntimeError('Run SAM3 on the configured cloud/Linux host')
        model_path = Path(options.get('model_path', '')).expanduser()
        processor_path = Path(options.get('processor_path', model_path)).expanduser()
        if not options.get('model_path') or not model_path.is_dir() or not processor_path.is_dir():
            raise RuntimeError('SAM3 requires existing local model and processor directories')
        self.groups = concept_groups(options)
        self.score_threshold = probability_option(options, 'instance_score_threshold', .3)
        self.mask_threshold = probability_option(options, 'instance_mask_threshold', .5)
        device = str(options.get('device', 'cuda'))
        if device == 'cpu' and not options.get('allow_cpu', False):
            raise RuntimeError('CPU SAM3 inference requires explicit allow_cpu')
        if device != 'cpu' and not (device == 'cuda' or device.startswith('cuda:')):
            raise ValueError('Unsupported SAM3 device')
        dtype_name = str(options.get('dtype', 'bfloat16'))
        if dtype_name not in ('float32', 'bfloat16', 'float16'):
            raise ValueError('SAM3 dtype must be float32, bfloat16 or float16')
        if device == 'cpu' and dtype_name != 'float32':
            raise ValueError('CPU SAM3 requires float32')
        import torch
        from transformers import Sam3Model, Sam3Processor
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('Configured SAM3 CUDA is unavailable; no CPU fallback was run')
        self.torch, self.device = torch, device
        self.dtype = getattr(torch, dtype_name)
        self.verifier = None
        self.verification_policy = None
        verifier_options = options.get('instance_verifier')
        if verifier_options is not None:
            if not isinstance(verifier_options, dict) or not provenance.get('instance_verifier'):
                raise ValueError('Instance verification requires explicit options and bound model provenance')
            from .object_verifier import RtdetrObjectVerifier
            self.verification_policy = verification_policy(verifier_options, self.groups)
            self.verifier = RtdetrObjectVerifier(verifier_options, provenance['instance_verifier'])
            known_labels = set(self.verifier.metadata['id2label'].values())
            if any(not known_labels.intersection(names) for names in self.verification_policy['concept_labels'].values()):
                raise ValueError('A SAM3 concept has no matching class in the actual detector taxonomy')
        self.model = Sam3Model.from_pretrained(str(model_path), local_files_only=True,
            trust_remote_code=False, dtype=self.dtype, attn_implementation='sdpa').to(device).eval()
        self.processor = Sam3Processor.from_pretrained(str(processor_path),
            local_files_only=True, trust_remote_code=False)
        self.metadata = dict(backend='transformers.Sam3Model', architecture=self.model.__class__.__name__,
            processor=self.processor.__class__.__name__, device=device, dtype=dtype_name,
            transformers_version=importlib.metadata.version('transformers'), torch_version=torch.__version__,
            concept_groups=self.groups, instance_score_threshold=self.score_threshold,
            instance_mask_threshold=self.mask_threshold, model_provenance=provenance,
            model_repo=options.get('model_repo'), model_revision=options.get('model_revision'),
            code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            local_files_only=True, trust_remote_code=False,
            postprocessing_dtype='float32',
            postprocessing_order='sigmoid logits; class-times-presence instance gate; bilinear probability resize; pixel threshold',
            inference_size_policy='checkpoint-native processor resolution; no custom resizing override',
            instance_score_definition='sigmoid(class logit) times sigmoid(concept presence logit)',
            group_score_definition='maximum interpolated sigmoid mask probability over accepted instances of the group; absent concepts contribute zero',
            group_score_caveat='not a mutually exclusive class softmax or calibrated semantic certainty',
            semantic_head_used=False,
            instance_mask_encoding='numpy.packbits over row-major H*W pixels, bitorder=little; instance_masks_shape records [N,H,W]')
        if self.verifier is not None:
            self.metadata.update(instance_verifier=dict(model=self.verifier.metadata, policy=self.verification_policy),
                evidence_schema='sam3_group_evidence_v2',
                group_score_definition='maximum interpolated sigmoid mask probability over accepted instances; dynamic instances additionally require independent detector support',
                candidate_preservation='All SAM score-gated candidates retain packed masks and metadata, including verifier-rejected candidates')

    def predict_evidence(self, image):
        torch = self.torch
        rgb = image.convert('RGB')
        width, height = rgb.size
        if width < 1 or height < 1:
            raise ValueError('SAM3 requires a nonempty RGB photograph')
        pixel_hash = hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest()
        detections = self.verifier.predict(rgb) if self.verifier is not None else None
        image_inputs = self.processor(images=rgb, return_tensors='pt').to(self.device)
        pixels = image_inputs['pixel_values'].to(self.dtype)
        probability = {group: np.zeros((height, width), np.float32) for group in self.groups}
        instances, packed_masks, concepts = [], [], []
        with torch.inference_mode():
            vision = self.model.get_vision_features(pixel_values=pixels)
            for group, prompts in self.groups.items():
                for prompt in prompts:
                    inputs = self.processor(text=prompt, return_tensors='pt').to(self.device)
                    output = self.model(vision_embeds=vision, **inputs)
                    if output.presence_logits is None:
                        raise RuntimeError('SAM3 did not return required concept-presence evidence')
                    required = (output.pred_logits, output.presence_logits, output.pred_masks, output.pred_boxes)
                    if any(not bool(torch.isfinite(value).all()) for value in required):
                        raise RuntimeError('SAM3 returned nonfinite instance evidence')
                    presence = output.presence_logits[0].float().sigmoid()
                    if presence.numel() != 1:
                        raise RuntimeError('SAM3 concept-presence shape changed')
                    scores = output.pred_logits[0].float().sigmoid() * presence
                    accepted = scores > self.score_threshold
                    query_ids = torch.nonzero(accepted, as_tuple=False).flatten()
                    concept_start = len(instances)
                    # Resize one accepted mask at a time to bound memory when
                    # many small objects are present. No accepted objects are
                    # silently truncated by a count cap.
                    for query in query_ids.tolist():
                        raw = output.pred_masks[0, query].float().sigmoid()
                        plane = torch.nn.functional.interpolate(raw[None, None],
                            size=(height, width), mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
                        binary = plane > self.mask_threshold
                        box = output.pred_boxes[0, query].float().cpu().numpy() * np.array([width, height, width, height])
                        verification = (verify_dynamic_instance(prompt, box, binary, detections, self.verification_policy)
                                        if group == 'dynamic' and self.verifier is not None
                                        else dict(status='not_required', accepted=True))
                        if verification['accepted']:
                            np.maximum(probability[group], plane, out=probability[group])
                        mask_index = len(instances)
                        packed_masks.append(np.packbits(binary.reshape(-1), bitorder='little'))
                        instances.append(dict(mask_index=mask_index, group=group, concept=prompt,
                            query_id=query, score=float(scores[query].item()),
                            presence_score=float(presence.item()), box_xyxy=box.tolist(),
                            mask_area_pixels=int(binary.sum()), accepted_for_group=verification['accepted'],
                            verification=verification))
                    concepts.append(dict(group=group, concept=prompt,
                        presence_score=float(presence.item()), candidate_instances=len(instances)-concept_start,
                        accepted_instances=sum(row['accepted_for_group'] for row in instances[concept_start:]),
                        instance_indices=list(range(concept_start, len(instances)))))
                    del output
        count = len(instances)
        packed = np.stack(packed_masks) if packed_masks else np.empty((0, (height*width+7)//8), np.uint8)
        if any(v.shape != (height, width) or not np.isfinite(v).all() or (v < 0).any() or (v > 1).any() for v in probability.values()):
            raise RuntimeError('Invalid SAM3 group evidence')
        return dict(group_probability=probability, instance_masks_packed=packed,
            instance_masks_shape=np.asarray([count, height, width], np.int32),
            metadata=dict(source_rgb_sha256=pixel_hash, image_size_hw=[height, width],
                concepts=concepts, instances=instances, score_definition=self.metadata['group_score_definition'],
                score_caveat=self.metadata['group_score_caveat'], semantic_head_used=False,
                object_verification=(dict(detections=detections, policy=self.verification_policy,
                    source_rgb_sha256=pixel_hash, image_size_hw=[height, width]) if self.verifier is not None else None)))
