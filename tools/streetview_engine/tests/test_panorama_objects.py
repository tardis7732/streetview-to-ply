import hashlib
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from tools.streetview_engine.panorama_objects import predict_panorama_objects, wrap_dilate


def processor(side):
    return SimpleNamespace(size={'height':side, 'width':side}, do_resize=True, do_pad=False, do_center_crop=None)


class FakeSegmenter:
    """CPU contract stub. Deliberately no model-quality or inference claim."""
    def __init__(self):
        self.processor = SimpleNamespace(image_processor=processor(1008))
        self.verifier = SimpleNamespace(processor=processor(640))
        self.images = []

    def predict_evidence(self, image):
        array = np.asarray(image).copy()
        self.images.append(array)
        height, width = array.shape[:2]
        digest = hashlib.sha256(array.tobytes()).hexdigest()
        plane = array[..., 0].astype(np.float32)/255
        return dict(group_probability={'dynamic':plane}, metadata=dict(
            source_rgb_sha256=digest, image_size_hw=[height, width], instances=[],
            object_verification=dict(source_rgb_sha256=digest, image_size_hw=[height, width], detections=[])))


def source():
    array = np.zeros((32, 64, 3), np.uint8)
    array[12:19, :3, 0] = 255
    array[0, 40, 1] = 117
    array[-1, 35, 2] = 217
    return Image.fromarray(array)


def test_complete_erp_one_pass_keeps_all_pixels_and_restores_original_mask_grid():
    image, segmenter = source(), FakeSegmenter()
    before = image.tobytes()
    result = predict_panorama_objects(image, segmenter, dict(dynamic_confidence=.5, dynamic_dilation_px=0))
    assert len(segmenter.images) == 1
    np.testing.assert_array_equal(segmenter.images[0], np.asarray(image))
    assert image.tobytes() == before
    np.testing.assert_array_equal(result['dynamic'], np.asarray(image)[..., 0] > 127)
    assert result['dynamic'].shape == (32, 64)
    assert result['metadata']['cube_face_inferences'] == 0
    assert result['metadata']['whole_panorama_inferences'] == 1
    assert result['metadata']['sam_processor']['scale_xy'] == [1008/64, 1008/32]
    assert result['metadata']['sam_processor']['padding'] is None
    assert result['evidence']['metadata']['source_rgb_sha256'] == hashlib.sha256(before).hexdigest()


def test_periodic_longitude_dilation_does_not_connect_top_and_bottom_poles():
    mask = np.zeros((12, 24), bool)
    mask[0, 0] = True
    dilated = wrap_dilate(mask, 1)
    assert dilated[0, -1] and dilated[1, -1] and dilated[0, 1]
    assert not dilated[-1].any()
    assert dilated.sum() == 6
    np.testing.assert_array_equal(wrap_dilate(mask, 0), mask)


def test_actual_backend_size_dataclass_is_accepted_without_geometry_override():
    segmenter = FakeSegmenter()
    segmenter.processor.image_processor.size = SimpleNamespace(height=1008, width=1008, longest_edge=None, shortest_edge=None)
    result = predict_panorama_objects(source(), segmenter, dict(dynamic_confidence=.5, dynamic_dilation_px=0))
    assert result['metadata']['sam_processor']['model_size_hw'] == [1008, 1008]


def test_optional_half_turn_keeps_full_sphere_and_unrolls_mask_exactly():
    image, segmenter = source(), FakeSegmenter()
    result = predict_panorama_objects(image, segmenter, dict(dynamic_confidence=.5, dynamic_dilation_px=0), seam_roll=True)
    assert len(segmenter.images) == 2
    np.testing.assert_array_equal(segmenter.images[1], np.roll(np.asarray(image), 32, axis=1))
    np.testing.assert_array_equal(result['dynamic'], np.asarray(image)[..., 0] > 127)
    assert result['metadata']['passes'][1]['original_pixel_count'] == 2048
    assert result['metadata']['passes'][1]['crop'] is None


def test_seam_half_turn_can_recover_a_model_mask_cut_by_image_boundary():
    class BoundarySegmenter(FakeSegmenter):
        def predict_evidence(self, image):
            evidence = super().predict_evidence(image)
            evidence['group_probability']['dynamic'][:, :4] = 0
            evidence['group_probability']['dynamic'][:, -4:] = 0
            return evidence
    image = source()
    policy = dict(dynamic_confidence=.5, dynamic_dilation_px=0)
    one = predict_panorama_objects(image, BoundarySegmenter(), policy)
    two = predict_panorama_objects(image, BoundarySegmenter(), policy, seam_roll=True)
    assert not one['dynamic'].any()
    np.testing.assert_array_equal(two['dynamic'], np.asarray(image)[..., 0] > 127)


@pytest.mark.parametrize('change', ['wrong_grid', 'wrong_verifier_source', 'padding', 'no_verifier', 'mutated_rgb'])
def test_reject_mismatched_or_mutated_inference_contract(change):
    segmenter = FakeSegmenter()
    if change == 'padding':
        segmenter.processor.image_processor.do_pad = True
    if change == 'no_verifier':
        segmenter.verifier = None
    original = segmenter.predict_evidence
    def altered(image):
        result = original(image)
        if change == 'wrong_grid':
            result['group_probability']['dynamic'] = result['group_probability']['dynamic'][::2]
        if change == 'wrong_verifier_source':
            result['metadata']['object_verification']['source_rgb_sha256'] = '0'*64
        if change == 'mutated_rgb':
            image.putpixel((0, 0), (1, 2, 3))
        return result
    segmenter.predict_evidence = altered
    with pytest.raises((ValueError, RuntimeError)):
        predict_panorama_objects(source(), segmenter, dict(dynamic_confidence=.5, dynamic_dilation_px=1))


@pytest.mark.parametrize('image', [Image.new('RGB',(32,32)), Image.new('L',(64,32))])
def test_reject_cube_face_or_non_rgb_input(image):
    with pytest.raises(ValueError):
        predict_panorama_objects(image, FakeSegmenter(), dict(dynamic_confidence=.5, dynamic_dilation_px=1))
