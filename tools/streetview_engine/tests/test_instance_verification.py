import copy

import numpy as np
import pytest

from tools.streetview_engine.preprocess import build_group_masks, mask_policy
from tools.streetview_engine.sam3_segmenter import (concept_groups,
    verification_policy, verify_dynamic_instance)


def policy():
    return verification_policy(dict(backend='rtdetr_v2'), concept_groups({}))


def test_unsupported_large_asphalt_is_not_accepted_as_person():
    mask = np.ones((100, 100), bool)
    detections = [dict(label='car', score=.95, box_xyxy=[0, 60, 25, 100])]
    result = verify_dynamic_instance('person', [0, 0, 100, 100], mask, detections, policy())
    assert not result['accepted']
    assert result['reason'] == 'no_matching_detection'


def test_supported_object_keeps_original_mask_and_detections():
    mask = np.zeros((100, 100), bool)
    mask[10:50, 20:60] = True
    before = mask.copy()
    detections = [dict(label='person', score=.8, box_xyxy=[18, 8, 63, 53])]
    source = copy.deepcopy(detections)
    result = verify_dynamic_instance('person', [20, 10, 60, 50], mask, detections, policy())
    assert result['accepted'] and result['matched_detection_indices'] == [0]
    assert np.array_equal(mask, before) and detections == source


def test_small_box_cannot_support_a_spilled_full_frame_mask():
    mask = np.ones((100, 100), bool)
    detections = [dict(label='person', score=.99, box_xyxy=[20, 10, 60, 50])]
    result = verify_dynamic_instance('person', [20, 10, 60, 50], mask, detections, policy())
    assert result['best_match']['box_iou'] == 1
    assert result['best_match']['mask_inside_box_fraction'] == .16
    assert not result['accepted']


def test_all_matching_detections_are_considered():
    mask = np.zeros((100, 100), bool)
    mask[10:50, 20:60] = True
    detections = [dict(label='person', score=.9, box_xyxy=[75, 75, 95, 95]),
                  dict(label='person', score=.7, box_xyxy=[20, 10, 60, 50])]
    result = verify_dynamic_instance('person', [20, 10, 60, 50], mask, detections, policy())
    assert result['accepted'] and result['matched_detection_indices'] == [1]


@pytest.mark.parametrize('bad_box', [[1, 1, 0, 2], [0, 0, float('nan'), 2], [1, 2]])
def test_invalid_detector_evidence_fails_without_unverified_fallback(bad_box):
    with pytest.raises(ValueError, match='invalid XYXY'):
        verify_dynamic_instance('person', [0, 0, 10, 10], np.ones((10, 10), bool),
            [dict(label='person', score=.9, box_xyxy=bad_box)], policy())


def test_object_support_is_invariant_to_uniform_image_scale():
    mask = np.zeros((100, 100), bool)
    mask[10:50, 20:60] = True
    detection = dict(label='car', score=.8, box_xyxy=[18, 8, 62, 52])
    original = verify_dynamic_instance('van', [20, 10, 60, 50], mask, [detection], policy())
    scaled = verify_dynamic_instance('van', [40, 20, 120, 100], mask.repeat(2, 0).repeat(2, 1),
        [dict(detection, box_xyxy=[v*2 for v in detection['box_xyxy']])], policy())
    assert original == scaled


@pytest.mark.parametrize('box', [[10, 2, 10, 8], [2, 5, 8, 5], [5, 2, 4, 8]])
def test_finite_degenerate_sam_candidate_is_explicitly_rejected(box):
    result = verify_dynamic_instance('person', box, np.ones((10, 10), bool), [], policy())
    assert not result['accepted'] and result['reason'] == 'degenerate_candidate_box'


def test_nonfinite_sam_candidate_still_fails_stage():
    with pytest.raises(ValueError, match='nonfinite candidate'):
        verify_dynamic_instance('person', [0, 0, float('nan'), 8], np.ones((10, 10), bool), [], policy())


def test_sam_pixel_threshold_matches_official_strict_mask_boundary():
    groups = {name: np.zeros((2, 3), np.float32) for name in ('dynamic', 'sky', 'ground')}
    groups['dynamic'][0, 0] = .5
    groups['dynamic'][0, 1] = np.nextafter(np.float32(.5), np.float32(1))
    groups['ground'][1, 0] = .5
    groups['ground'][1, 1] = np.nextafter(np.float32(.5), np.float32(1))
    settings = mask_policy(dict(dynamic_confidence=.5, ground_confidence=.5,
                                dynamic_dilation_px=0, core_erosion_px=0))
    masks = build_group_masks(groups, settings, dict(remove_sky=True, mask_dynamic=True))
    assert not masks['dynamic'][0, 0] and masks['dynamic'][0, 1]
    assert not masks['ground'][1, 0] and masks['ground'][1, 1]


def test_custom_concept_requires_explicit_detector_taxonomy_mapping():
    groups = concept_groups(dict(concept_groups=dict(dynamic=['scooter'], sky=['sky'], ground=['road'])))
    with pytest.raises(ValueError, match='Missing detector labels'):
        verification_policy(dict(backend='rtdetr_v2'), groups)
