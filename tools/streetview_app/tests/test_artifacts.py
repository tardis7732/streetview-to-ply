import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from tools.streetview_app.artifacts import resolve_artifact,sha
from tools.streetview_app.ply_filters import PlyFilterManager
from tools.streetview_app.tests.test_ply_filters import inputs,wait_job
from tools.streetview_engine.tests.test_size_filter import export_fixture


@pytest.mark.parametrize('changed',[dict(world_up=[1,0,0]),dict(fl_x=1),dict(ply_binding={'sha256':'0'*64})])
def test_filter_output_binding_checks_document_camera_metadata(tmp_path,changed):
    manager=PlyFilterManager(tmp_path/'filters')
    try:
        job=wait_job(manager,manager.start(inputs(tmp_path))['id'])
        app=SimpleNamespace(filters=manager)
        payload=dict(kind='filter',job_id=job['id'])
        resolved=resolve_artifact(app,payload)
        output=Path(resolved['camera_json']);doc=json.loads(output.read_text())
        doc.update(changed);output.write_text(json.dumps(doc))
        with pytest.raises(ValueError,match='카메라 좌표 연결'):
            resolve_artifact(app,payload)
    finally:manager.close()


def generation(tmp_path,enabled):
    fixture,report=export_fixture(tmp_path/'fixture',enabled)
    root=tmp_path/'jobs';job_id='a'*32;directory=root/job_id
    shutil.copytree(fixture,directory)
    state=dict(id=job_id,status='completed',artifact=report['artifact'],config={'size_filter':dict(enabled=enabled,max_sigma_camera_radius_ratio=.5)})
    return SimpleNamespace(jobs=SimpleNamespace(root=root,get=lambda key:state)),dict(kind='generation',job_id=job_id),directory,state


@pytest.mark.parametrize('enabled',[False,True])
def test_generation_accepts_bound_training_and_export(tmp_path,enabled):
    app,payload,directory,state=generation(tmp_path,enabled)
    resolved=resolve_artifact(app,payload)
    assert sha(resolved['source_ply'])==state['artifact']['sha256']
    assert json.loads(Path(resolved['camera_json']).read_text())['frames']


@pytest.mark.parametrize('change',['training_manifest','selection','accepted_model','camera_metadata'])
def test_generation_rejects_changed_lineage_or_camera_defaults(tmp_path,change):
    app,payload,directory,state=generation(tmp_path,False)
    resolved=resolve_artifact(app,payload)
    report_path=directory/'export/report.json';training_path=directory/'training/manifest.json'
    report=json.loads(report_path.read_text());training=json.loads(training_path.read_text())
    if change=='training_manifest':
        training['quality']={'changed':True};training_path.write_text(json.dumps(training))
    elif change=='selection':
        report['selection']={'accepted_model':'other.ply'};report_path.write_text(json.dumps(report))
    elif change=='accepted_model':
        training['selection']['accepted_model_sha256']='0'*64;training_path.write_text(json.dumps(training))
        report['selection']=training['selection'];report['training_manifest_sha256']=sha(training_path);report_path.write_text(json.dumps(report))
    else:
        path=Path(resolved['camera_json']);doc=json.loads(path.read_text());doc['cx']=700;path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):resolve_artifact(app,payload)
