import json
from pathlib import Path
import shutil
import pytest
from tools.streetview_engine import panorama_workflow as workflow, brush_refine, export
from tools.streetview_engine.__main__ import stage_module
from tools.streetview_engine.tests.test_sky_cleanup import fixture


CONFIG = dict(generation_mode='multi_view', training_steps=6000, resolution=1280, max_splats=2000000,
              size_filter=dict(enabled=True,max_sigma_camera_radius_ratio=.5))


def test_routes_remain_explicit_and_fixed_settings_not_silently_ignored():
    settings = dict(workflow='panorama_brush_refine')
    assert stage_module('preprocess', CONFIG, settings) == 'panorama_preprocess'
    assert stage_module('train', CONFIG, settings) == 'panorama_workflow'
    with pytest.raises(ValueError, match='Only multi_view'):
        stage_module('infer', {'generation_mode':'single_panorama'}, settings)
    assert stage_module('train', CONFIG) == 'experiment'
    with pytest.raises(ValueError, match='resolution'):
        workflow.workflow_options(dict(CONFIG,resolution=768),settings)


def test_completed_training_native_cleanup_final_export_lineage(tmp_path, monkeypatch):
    root = tmp_path/'job'; dataset = root/'sfm/dataset'
    source, camera, native = fixture(dataset)
    doc = json.loads(camera.read_text()); rows = json.loads(native.read_text())['frames']
    for frame, row in zip(doc['frames'], rows):
        for key, field in [('original_file_path','source_rgb'),('original_valid_mask_path','source_generic_valid'),
                           ('edit_alpha_path','source_edit_alpha'),('sky_mask_path','source_sky')]:
            frame[key] = row[field+'_path']
            frame[key.removesuffix('_path')+'_sha256'] = row[field+'_sha256']
    export.write_json(dataset/'transforms_train.json',doc)
    export.write_json(dataset/'transforms_heldout.json',dict(doc,frames=[]))
    export.write_json(dataset/'dataset_manifest.json',{'fixture':True})
    (dataset/'init.ply').write_bytes(b'seed fixture')
    monkeypatch.setattr(brush_refine,'_operator_settings',lambda settings: {})
    def training(config,job,settings):
        out=Path(job)/'training';out.mkdir()
        shutil.copyfile(source,out/'model.ply')
        state=dict(status='completed',artifact=export.validate_ply(out/'model.ply'),
            selection=dict(accepted_model='model.ply',accepted_model_sha256=export.sha256(out/'model.ply')),
            quality=dict(fixture=True))
        export.write_json(out/'manifest.json',state)
        return state
    monkeypatch.setattr(brush_refine,'run',training)
    settings=dict(workflow='panorama_brush_refine',sky_cleanup=dict(enabled=True,device='cpu',
        policy=dict(hard_size_ratio=None,protect_down=False)))
    result=workflow.run(CONFIG,root,settings)
    assert result['status']=='completed' and result['cleanup']['removed_rows']==1
    assert result['selection']['accepted_model']=='sky_cleanup/scene.ply'
    assert result['quality'] is None and result['source_quality']==dict(fixture=True)
    report=export.run(CONFIG,root,settings)
    assert report['artifact']['vertex_count']==3
    assert report['size_filter']['removed_rows']==1
    assert report['camera_reference']['sha256']==export.sha256(dataset/'transforms_train.json')
    assert export.sha256(root/'export/source.ply')==result['selection']['accepted_model_sha256']
    assert export.sha256(root/'training/model.ply')==export.sha256(source)
