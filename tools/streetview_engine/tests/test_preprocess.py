import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools.streetview_engine import preprocess
from tools.streetview_engine.imaging import FACES, png_bytes, sha256, write_bytes, write_json


class MaskTests(unittest.TestCase):
    def test_model_names_resolve_and_missing_taxonomy_refuses(self):
        mapping={0:'building',1:'person',2:'car',3:'sky',4:'road',5:'stairway'}
        groups=preprocess.resolve_class_groups(mapping,dict(dynamic=['person','car'],sky=['sky'],ground=['road']))
        self.assertEqual(groups,dict(dynamic=[1,2],sky=[3],ground=[4]))
        with self.assertRaisesRegex(ValueError,'does not contain'):
            preprocess.resolve_class_groups(mapping,dict(dynamic=['vehicle'],sky=['sky'],ground=['road']))
        with self.assertRaises(ValueError):
            preprocess.resolve_class_groups(mapping,dict(dynamic=[1],sky=['sky'],ground=['road']))

    def test_sky_retained_photometrically_and_stairs_not_ground(self):
        labels=np.array([[3,3,1],[4,4,5]],np.int32);confidence=np.ones_like(labels,np.float32)
        original=labels.copy();groups=dict(dynamic=[1,2],sky=[3],ground=[4])
        masks=preprocess.build_masks(labels,confidence,groups,preprocess.mask_policy(dict(dynamic_dilation_px=0,core_erosion_px=0)))
        self.assertTrue(masks['photometric'][0,0]);self.assertFalse(masks['sfm'][0,0]);self.assertTrue(masks['sky'][0,0])
        self.assertFalse(masks['photometric'][0,2]);self.assertTrue(masks['ground'][1,0]);self.assertFalse(masks['ground'][1,2])
        np.testing.assert_array_equal(labels,original)

    def test_confidence_and_morphology_are_explicit(self):
        labels=np.zeros((9,9),np.int32);labels[4,4]=1
        conf=np.ones_like(labels,np.float32)
        groups=dict(dynamic=[1],sky=[2],ground=[3]);policy=preprocess.mask_policy(dict(dynamic_dilation_px=1,core_erosion_px=0))
        self.assertEqual(preprocess.build_masks(labels,conf,groups,policy)['dynamic'].sum(),9)
        conf[4,4]=.4
        self.assertEqual(preprocess.build_masks(labels,conf,groups,policy)['dynamic'].sum(),0)
        conf[0,0]=np.nan
        with self.assertRaises(ValueError):preprocess.build_masks(labels,conf,groups,policy)
        with self.assertRaises(ValueError):preprocess.mask_policy(dict(core_erosion_px=1.5))


class FakeSegmenter:
    """Synthetic inference only inside temporary unit-test fixtures."""
    calls=0
    def __init__(self,options,provenance):
        self.groups=dict(dynamic=[1],sky=[2],ground=[3]);self.metadata=dict(backend='synthetic_unit_test_only')
    def predict(self,image):
        type(self).calls+=1
        labels=np.full((image.height,image.width),3,np.int32);labels[:3]=2;labels[5,5]=1
        return labels,np.ones_like(labels,np.float32)


class PreprocessStageTests(unittest.TestCase):
    def fixture(self,root):
        source=np.zeros((12,12,3),np.uint8);source[:,:,1]=100
        faces={}
        for face in FACES:
            path=root/f'collection/images/capture_{face}.png';write_bytes(path,png_bytes(source))
            faces[face]=dict(file_path=path.relative_to(root).as_posix(),sha256=sha256(path),w=12,h=12)
        write_json(root/'collection/manifest.json',dict(status='complete',face_order=list(FACES),stations=[dict(station_id='station_group',pano_id='capture',id='capture',lat=0.,lng=0.,faces=faces)]))
        model=root/'operator_model';model.mkdir();(model/'config.json').write_text('{}');(model/'model.safetensors').write_bytes(b'synthetic unit test model marker')
        settings=dict(preprocess=dict(segmentation=dict(model_path=str(model),dynamic_dilation_px=0,core_erosion_px=0)))
        return dict(panorama_ids=['capture']),settings

    def test_all_faces_real_stage_contract_source_preservation_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config,settings=self.fixture(root)
            before={path.name:sha256(path) for path in (root/'collection/images').glob('*.png')}
            FakeSegmenter.calls=0
            with patch.object(preprocess,'HFSemanticSegmenter',FakeSegmenter):
                manifest=preprocess.run(config,root,settings)
            self.assertEqual(FakeSegmenter.calls,6);self.assertEqual({f['face'] for f in manifest['frames']},set(FACES))
            for frame in manifest['frames']:
                self.assertEqual(frame['fl_x'],6);self.assertEqual(frame['station_id'],'station_group')
                self.assertEqual(np.array(frame['camera_to_station_cv']).shape,(4,4))
                self.assertGreater(frame['mask_fractions']['photometric'],frame['mask_fractions']['sfm'])
                self.assertEqual(frame['source_sha256'],before[Path(frame['file_path']).name])
            with patch.object(preprocess,'HFSemanticSegmenter',side_effect=AssertionError('resume must not infer')):
                self.assertEqual(preprocess.run(config,root,settings),manifest)
            (root/manifest['frames'][0]['mask_path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed after completion'):
                preprocess.run(config,root,settings)

    def test_missing_model_is_not_a_white_mask_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaisesRegex(RuntimeError,'local semantic model weights'):
                preprocess.run(dict(panorama_ids=['capture']),root,{})
            self.assertFalse((root/'prepared/manifest.json').exists())

    def test_changed_weights_refuse_prior_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config,settings=self.fixture(root)
            with patch.object(preprocess,'HFSemanticSegmenter',FakeSegmenter):preprocess.run(config,root,settings)
            (root/'operator_model/model.safetensors').write_bytes(b'different model')
            with self.assertRaisesRegex(ValueError,'different inputs/model/settings'):
                preprocess.run(config,root,settings)


if __name__=='__main__':unittest.main()
