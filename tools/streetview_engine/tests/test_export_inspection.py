"""Actual-file export/audit binding; synthetic geometry, no quality claim."""
import json
import shutil

import pytest

from tools.streetview_engine.export import run, sha256, validate_ply, write_json
from tools.streetview_engine.tests.test_inspection import fixture


def scene(tmp_path):
    dataset, ply = fixture(tmp_path / 'source', ground=True)
    job = tmp_path / 'job'
    shutil.copytree(dataset, job / 'sfm/dataset')
    (job / 'training').mkdir()
    shutil.copyfile(ply, job / 'training/model.ply')
    artifact = validate_ply(ply)
    write_json(job / 'training/manifest.json', dict(status='completed', artifact=artifact,
        selection=dict(accepted_model='model.ply', accepted_model_sha256=artifact['sha256']), quality={}))
    return job


def test_export_links_actual_readonly_inspection_and_preserves_ply(tmp_path):
    job = scene(tmp_path)
    before = sha256(job / 'training/model.ply')
    report = run({}, job, {'inspection': {'enabled': True, 'ground': True}})
    audit_path = job / report['inspection']['report']
    audit = json.loads(audit_path.read_text(encoding='utf8'))
    assert report['inspection']['sha256'] == sha256(audit_path)
    assert audit['input_ply']['sha256'] == before == sha256(job / 'export/scene.ply')
    assert audit['geometry_mutated'] is False and audit['quality_accepted'] is False
    assert report['inspection']['all_rows']['rows'] == 4
    assert report['inspection']['ground_support']['split'] == 'train'
    assert sha256(job / 'training/model.ply') == before


def test_corrupt_dataset_cannot_receive_completed_export_audit(tmp_path):
    job = scene(tmp_path)
    with (job / 'sfm/dataset/init_points.npz').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError):
        run({}, job, {'inspection': {'enabled': True, 'ground': True}})
    assert not (job / 'export/report.json').exists()
    assert not (job / 'export/scene.ply').exists()
