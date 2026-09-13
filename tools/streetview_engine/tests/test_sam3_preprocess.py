"""Model-free tests of new concept evidence and isolated-stage behavior."""
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from tools.streetview_engine import preprocess
from tools.streetview_engine.tests.test_preprocess import PreprocessStageTests


def test_overlapping_concepts_do_not_create_ground_or_invert_validity():
    planes = {key: np.zeros((12,12), np.float32) for key in ('dynamic','sky','ground')}
    planes['sky'][:4] = .9
    planes['ground'][4:8] = .9
    planes['dynamic'][3:6,3:6] = .9
    original = {key: value.copy() for key,value in planes.items()}
    policy = preprocess.mask_policy(dict(dynamic_dilation_px=0,core_erosion_px=0))
    for sky in (False, True):
        for dynamic in (False, True):
            masks = preprocess.build_group_masks(planes, policy, dict(remove_sky=sky,mask_dynamic=dynamic))
            assert bool(masks['photometric'][0,0]) == (not sky)
            assert bool(masks['photometric'][5,4]) == (not dynamic)
            assert not masks['sfm'][0,0]
            assert not masks['ground'][5,4]
            assert not masks['ground'][10,10]  # unknown is never certified ground
    for key in planes:
        np.testing.assert_array_equal(planes[key], original[key])
    planes['sky'][0,0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        preprocess.build_group_masks(planes, policy)


def test_unknown_backend_and_missing_runtime_do_not_fall_back():
    with pytest.raises(ValueError, match='Unsupported segmentation backend'):
        preprocess.segmentation_backend(dict(backend='sam3_typo'))
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder)
        with pytest.raises(ValueError, match='existing absolute Python path'):
            preprocess.run_in_configured_runtime({},root,{},dict(python_executable=str(root/'missing_python')))
        assert not (root/'prepared').exists()


class FakeConceptSegmenter:
    def __init__(self, options, provenance):
        self.metadata = dict(backend='synthetic_concept_test_only',model_provenance=provenance)
    def predict_evidence(self, image):
        planes = {key: np.zeros((image.height,image.width), np.float32) for key in ('dynamic','sky','ground')}
        planes['sky'][:3] = .95
        planes['ground'][4:] = .95
        planes['dynamic'][5,5] = .99
        return dict(group_probability=planes,
                    instance_masks_packed=np.packbits((planes['dynamic']>.5).reshape(1,-1),axis=1,bitorder='little'),
                    instance_masks_shape=np.array([1,image.height,image.width],np.int32),
                    metadata=dict(instances=[dict(group='dynamic',score=.8)],semantic_head_used=False))


def test_full_stage_saves_concept_evidence_keeps_six_faces_and_resumes_without_inference():
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        config,settings=PreprocessStageTests().fixture(root)
        settings['preprocess']['segmentation']['backend']='sam3'
        config['processing_options']=dict(remove_sky=True,mask_dynamic=True)
        with patch('tools.streetview_engine.sam3_segmenter.Sam3EvidenceSegmenter',FakeConceptSegmenter):
            result=preprocess.run(config,root,settings)
        assert len(result['frames'])==6
        for frame in result['frames']:
            with np.load(root/frame['semantic_path'],allow_pickle=False) as evidence:
                assert str(evidence['evidence_schema'])=='sam3_group_evidence_v1'
                assert 'labels' not in evidence.files  # no fake semantic class IDs
                assert evidence['instance_masks_shape'].tolist()==[1,12,12]
                assert evidence['group_ground'][10,10]>.8
        with patch('tools.streetview_engine.sam3_segmenter.Sam3EvidenceSegmenter',side_effect=AssertionError('must not reinfer')):
            assert preprocess.run(config,root,settings)==result
