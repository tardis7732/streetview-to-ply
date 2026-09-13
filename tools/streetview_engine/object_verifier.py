"""Independent, local-weight RT-DETRv2 boxes; no segmentation or scene rules."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verifier_provenance(options):
    """Bind local safetensors/config bytes before loading, without network access."""
    if options.get('backend') != 'rtdetr_v2':
        raise ValueError('Object verifier requires explicit backend=rtdetr_v2')
    if not options.get('model_path'):
        raise ValueError('Object verifier requires an explicit local model_path')
    root = Path(options['model_path']).expanduser().resolve()
    processor = Path(options.get('processor_path', root)).expanduser().resolve()
    required = [(root, 'config.json'), (root, 'model.safetensors'),
                (processor, 'preprocessor_config.json')]
    files = {}
    for directory, name in required:
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError('Missing local verifier artifact: ' + str(path))
        files[str(path)] = dict(sha256=_sha(path), bytes=path.stat().st_size)
    config = json.loads((root/'config.json').read_text())
    if config.get('model_type') != 'rt_detr_v2':
        raise ValueError('Object verifier requires a native RT-DETRv2 checkpoint')
    labels = {int(k): str(v).strip().casefold() for k, v in config['id2label'].items()}
    required_labels = {'person', 'car', 'bus', 'truck', 'bicycle'}
    if (len(labels) != 80 or not required_labels.issubset(labels.values())
            or not {'motorcycle', 'motorbike'}.intersection(labels.values())):
        raise ValueError('Expected the declared COCO 80-class object taxonomy')
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    expected = options.get('expected_files_sha256')
    if expected is not None and digest != expected:
        raise ValueError('Object verifier local files changed')
    return dict(root=str(root), model_path=str(root), processor_path=str(processor),
                file_hashes=files, files_sha256=digest, model_type='rt_detr_v2',
                id2label=labels, local_files_only=True, trust_remote_code=False)


class RtdetrObjectVerifier:
    """An independent detector. Matching detected boxes to masks is the caller's job."""

    def __init__(self, options, provenance):
        if sys.platform == 'win32' and not options.get('allow_windows_inference', False):
            raise RuntimeError('Run the object verifier on the configured cloud/Linux host')
        if verifier_provenance(options) != provenance:
            raise ValueError('Object verifier weights/config changed after provenance capture')
        score = options.get('min_detection_score', .3)
        if isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 < score < 1:
            raise ValueError('min_detection_score must be finite and in (0,1)')
        device = str(options.get('device', 'cuda'))
        if device == 'cpu' and not options.get('allow_cpu', False):
            raise RuntimeError('CPU object verification requires explicit allow_cpu')
        if device != 'cpu' and not (device == 'cuda' or device.startswith('cuda:')):
            raise ValueError('Unsupported object verifier device')
        # Keep the small detector in fp32; no mixed-precision or resize overrides.
        if options.get('dtype', 'float32') != 'float32':
            raise ValueError('This verified RT-DETRv2 path requires float32')
        import torch
        from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('Configured verifier CUDA is unavailable; no CPU fallback')
        self.torch, self.device, self.min_score = torch, device, float(score)
        self.processor = RTDetrImageProcessor.from_pretrained(provenance['processor_path'],
            local_files_only=True, trust_remote_code=False)
        self.model = RTDetrV2ForObjectDetection.from_pretrained(provenance['model_path'],
            local_files_only=True, trust_remote_code=False, dtype=torch.float32).to(device).eval()
        self.labels = {int(k): str(v).strip().casefold() for k, v in self.model.config.id2label.items()}
        if self.labels != provenance['id2label']:
            raise ValueError('Loaded object detector taxonomy differs from its config')
        self.metadata = dict(backend='transformers.RTDetrV2ForObjectDetection',
            architecture=self.model.__class__.__name__, processor=self.processor.__class__.__name__,
            model_repo=options.get('model_repo'), model_revision=options.get('model_revision'),
            model_provenance=provenance, min_detection_score=self.min_score,
            taxonomy='COCO 80 checkpoint labels', id2label=self.labels, device=device, dtype='float32',
            processor_config=self.processor.to_dict(), torch_version=torch.__version__,
            transformers_version=importlib.metadata.version('transformers'),
            code_sha256=_sha(__file__), local_files_only=True, trust_remote_code=False,
            postprocessing='official post_process_object_detection, original pixel XYXY coordinates',
            inference_size_policy='checkpoint-native processor resolution',
            score_caveat='Object detection confidence, not independent calibrated certainty or ground truth')

    def predict(self, image):
        rgb = image.convert('RGB')
        width, height = rgb.size
        if min(width, height) < 1:
            raise ValueError('Object verifier requires nonempty RGB')
        original_bytes = rgb.tobytes()
        inputs = self.processor(images=rgb, return_tensors='pt').to(self.device)
        torch = self.torch
        with torch.inference_mode():
            output = self.model(**inputs)
            if not torch.isfinite(output.logits).all() or not torch.isfinite(output.pred_boxes).all():
                raise RuntimeError('Nonfinite object detector prediction')
            result = self.processor.post_process_object_detection(output,
                target_sizes=torch.tensor([(height, width)], device=self.device), threshold=self.min_score)[0]
        records = []
        for score, label_id, box in zip(result['scores'], result['labels'], result['boxes'], strict=True):
            values = [float(v) for v in box.detach().cpu().tolist()]
            confidence, label_id = float(score.item()), int(label_id.item())
            if (not all(math.isfinite(v) for v in values) or not math.isfinite(confidence)
                    or not 0 <= confidence <= 1 or label_id not in self.labels):
                raise RuntimeError('Invalid postprocessed detection')
            values = [min(max(values[0], 0.), float(width)), min(max(values[1], 0.), float(height)),
                      min(max(values[2], 0.), float(width)), min(max(values[3], 0.), float(height))]
            if values[2] <= values[0] or values[3] <= values[1]:
                continue
            records.append(dict(label=self.labels[label_id], label_id=label_id, score=confidence, box_xyxy=values))
        if rgb.tobytes() != original_bytes:
            raise RuntimeError('Object processor mutated RGB input')
        return records
