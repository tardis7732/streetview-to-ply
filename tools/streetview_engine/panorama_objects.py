"""Detect dynamic objects on the complete 2:1 ERP, never on cube crops.

SAM3 and its independent object verifier see the same complete RGB canvas.
Checkpoint-native whole-field resizing is inverted by their existing official
pixel/box postprocessing. No external letterbox, perspective crop, or RGB edit
is performed here. An optional half-turn is another complete ERP, not a crop.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
from PIL import Image


def _rgb_hash(image):
    return hashlib.sha256(np.asarray(image).tobytes()).hexdigest()


def _processor_geometry(processor, width, height):
    """Declare the actual loaded model's whole-field resize, not a guessed fit."""
    size = getattr(processor, 'size', None)
    # New transformers backends hold a SizeDict dataclass; older processors
    # retain a plain dict. Both explicitly describe the same fixed H/W resize.
    keys = ('height','width','shortest_edge','longest_edge','max_height','max_width')
    dimensions = ({key:value for key,value in size.items() if value is not None} if isinstance(size,dict)
                  else {key:getattr(size,key) for key in keys if getattr(size,key,None) is not None})
    if (set(dimensions) != {'height', 'width'}
            or any(type(dimensions[k]) is not int or dimensions[k] <= 0 for k in dimensions)):
        raise ValueError('A bound fixed height/width processor is required for ERP coordinate restoration')
    if (getattr(processor, 'do_resize', None) is not True
            or getattr(processor, 'do_pad', None) not in (None, False)
            or getattr(processor, 'do_center_crop', None) not in (None, False)):
        raise ValueError('Whole ERP inference requires native full-field resize without padding or center crop')
    return dict(processor_class=processor.__class__.__name__, input_size_hw=[height, width],
        model_size_hw=[dimensions['height'], dimensions['width']], whole_field=True,
        scale_xy=[dimensions['width']/width, dimensions['height']/height],
        restore_scale_xy=[width/dimensions['width'], height/dimensions['height']],
        crop=None, padding=None, external_letterbox=False,
        note='Checkpoint-native independent-axis tensor resize; original ERP RGB and output grid are retained')


def wrap_dilate(mask, radius):
    """Periodic longitude, bounded latitude. Poles do not wrap top-to-bottom."""
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != bool:
        raise ValueError('ERP morphology requires a boolean HxW mask')
    if type(radius) is not int or radius < 0 or radius > min(mask.shape):
        raise ValueError('ERP dilation radius must be a nonnegative integer within the image size')
    if radius == 0:
        return mask.copy()
    from scipy.ndimage import maximum_filter
    # Mode order follows array axes: latitude is bounded, longitude is periodic.
    return maximum_filter(mask, size=2*radius+1, mode=('constant', 'wrap'), cval=False)


def _validate_evidence(evidence, image):
    width, height = image.size
    expected_hash = _rgb_hash(image)
    if not isinstance(evidence, dict) or not isinstance(evidence.get('metadata'), dict):
        raise ValueError('Whole-ERP object evidence must include bound image metadata')
    metadata = evidence['metadata']
    if metadata.get('source_rgb_sha256') != expected_hash or metadata.get('image_size_hw') != [height, width]:
        raise ValueError('Object mask evidence belongs to another RGB image or pixel grid')
    verification = metadata.get('object_verification')
    if (not isinstance(verification, dict) or verification.get('source_rgb_sha256') != expected_hash
            or verification.get('image_size_hw') != [height, width]):
        raise ValueError('Independent object verification must use the same complete ERP image')
    groups = evidence.get('group_probability')
    if not isinstance(groups, dict) or 'dynamic' not in groups:
        raise ValueError('Missing original-grid SAM3 dynamic evidence')
    plane = np.asarray(groups['dynamic'])
    if (plane.shape != (height, width) or not np.issubdtype(plane.dtype, np.floating)
            or not np.isfinite(plane).all() or np.any(plane < 0) or np.any(plane > 1)):
        raise ValueError('SAM3 dynamic evidence must be finite original-ERP-grid probabilities in [0,1]')
    return np.asarray(plane, np.float32).copy(), metadata


def predict_panorama_objects(image, segmenter, policy, *, seam_roll=False):
    """Return a dynamic mask on the immutable complete ERP's original pixel grid.

    Default: one complete panorama inference. If explicitly enabled, a second
    complete panorama is shifted by exactly half its width, then its evidence
    is unshifted and unioned. Both passes retain all 360x180 pixels. This can
    supply context for an object split at the arbitrary longitude seam; it is
    optional because a second model pass can also change false positives.
    """
    if not isinstance(image, Image.Image) or image.mode != 'RGB':
        raise ValueError('A decoded RGB whole panorama is required')
    width, height = image.size
    if height < 1 or width != 2*height:
        raise ValueError('Object inference requires the complete 2:1 360x180 ERP')
    if type(seam_roll) is not bool:
        raise ValueError('seam_roll must be an explicit boolean')
    if not isinstance(policy, dict):
        raise ValueError('Explicit dynamic mask thresholds are required')
    threshold = policy.get('dynamic_confidence')
    radius = policy.get('dynamic_dilation_px')
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or not 0 < threshold < 1):
        raise ValueError('dynamic_confidence must be finite in (0,1)')
    if type(radius) is not int or not 0 <= radius <= min(height, width):
        raise ValueError('dynamic_dilation_px must be an explicit nonnegative ERP-pixel integer')
    verifier = getattr(segmenter, 'verifier', None)
    if verifier is None:
        raise ValueError('Whole-panorama dynamic removal requires an independent object verifier')
    sam_geometry = _processor_geometry(segmenter.processor.image_processor, width, height)
    verifier_geometry = _processor_geometry(verifier.processor, width, height)
    original_hash = _rgb_hash(image)
    shifts = [0, width//2] if seam_roll else [0]
    combined = np.zeros((height, width), np.float32)
    passes = []
    first_evidence = None
    for shift in shifts:
        # A copy prevents a third-party processor mutating the user's RGB canvas.
        current = image.copy() if shift == 0 else Image.fromarray(np.roll(np.asarray(image), shift, axis=1))
        input_hash = _rgb_hash(current)
        evidence = segmenter.predict_evidence(current)
        if _rgb_hash(current) != input_hash or _rgb_hash(image) != original_hash:
            raise RuntimeError('An object-inference processor mutated its input RGB')
        plane, metadata = _validate_evidence(evidence, current)
        if shift == 0:
            first_evidence = evidence
        if shift:
            plane = np.roll(plane, -shift, axis=1)
        np.maximum(combined, plane, out=combined)
        passes.append(dict(roll_pixels=shift, input_rgb_sha256=input_hash,
            image_size_hw=[height, width], original_pixel_count=height*width,
            context='complete_360x180_ERP', crop=None, evidence=metadata))
    core = combined > threshold
    dynamic = wrap_dilate(core, radius)
    metadata = dict(schema_version=1, policy='whole_erp_sam3_verified_dynamic_v1',
        source_rgb_sha256=original_hash, image_size_hw=[height, width],
        input_projection='equirectangular_360x180', inference_domain='whole_panorama',
        whole_panorama_inferences=len(shifts), cube_face_inferences=0,
        sam_processor=sam_geometry, verifier_processor=verifier_geometry,
        probability_restoration='Original H,W bilinear align_corners=False; normalized boxes scaled independently by original W,H',
        output_projection='same_original_ERP_grid', source_rgb_unchanged=True,
        dynamic_confidence=float(threshold), threshold_comparison='strict_greater_than',
        dynamic_dilation_px=radius, morphology_boundary=dict(longitude='periodic', latitude='constant_false'),
        seam_policy='explicit_half_turn_full_ERP_union' if seam_roll else 'single_complete_ERP_pass',
        seam_caveat='A single planar model pass may miss seam-split objects; wrap morphology alone does not infer missing object content',
        shadow_mask_inference=False, sky_ground_policy='Not changed by this dynamic-object helper',
        core_pixels=int(core.sum()), dynamic_pixels=int(dynamic.sum()), passes=passes,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return dict(dynamic=dynamic, dynamic_core=core, group_dynamic=combined, metadata=metadata,
                evidence=first_evidence)
