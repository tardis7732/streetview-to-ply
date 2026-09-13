"""Real-file CPU removal/lineage tests; no scene-quality assertion."""
import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

from tools.streetview_engine import export
from tools.streetview_engine.size_filter import read_size_filter_options, run_size_filter, verify_size_filter_report
from tools.streetview_app.jobs import Backend, JobManager, Stage, STAGES
from tools.streetview_app.remote_jobs import verify_result


OPTIONS = dict(enabled=True, max_sigma_camera_radius_ratio=.5)


def fixture(root, *, world_scale=1., rotation=None, shift=None, duplicates=0):
    root.mkdir(parents=True)
    rotation = np.eye(3) if rotation is None else rotation
    shift = np.zeros(3) if shift is None else shift
    positions = np.array([[-2., 0., 0.], [2., 0., 0.], [0., 0., 1.]])
    frames = []
    for index, position in enumerate(positions):
        pose = np.eye(4); pose[:3, :3] = rotation; pose[:3, 3] = world_scale * (rotation @ position) + shift
        frames.append(dict(station_id=str(index), transform_matrix=pose.tolist()))
    frames.extend(copy.deepcopy(frames[0]) for _ in range(duplicates))
    camera = root / 'cameras.json'
    export.write_json(camera, dict(coordinate_frame='EDN', units='metres', camera_convention='OpenGL_c2w', frames=frames))
    ply = root / 'source.ply'
    scales = np.array([[.1, .2, .3], [1.2, .2, .3], [100., .1, .1], [.001, .03, .3]])
    # Axis permutation is an exact rigid rotation; all channels have distinct data.
    export.write_model(ply, means=world_scale*(np.arange(12).reshape(4,3)@rotation.T)+shift,
        log_scales=np.log(scales*world_scale), quats=np.tile([1.,0.,0.,0.],(4,1)),
        opacity_logits=np.arange(4), sh0=np.arange(12).reshape(4,3)/20, shN=np.arange(96).reshape(4,8,3)/100)
    return ply, camera


def test_coordinate_and_scale_invariance_exact_row_preservation(tmp_path):
    reports = []
    for index, scale in enumerate((1e-4, 1., 1000.)):
        ply, camera = fixture(tmp_path / str(index), world_scale=scale,
            rotation=np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]]), shift=np.array([3.,-2.,1.])*scale)
        report = run_size_filter(ply, camera, tmp_path/f'out{index}', OPTIONS)
        assert report['removed_rows'] == 2 and report['remaining_rows'] == 2
        assert report['camera_reference']['physical_stations'] == 3
        assert verify_size_filter_report(report, ply, report['artifact']['path'], camera, report['selection']['path'])['exact_retained_row_bytes']
        with np.load(report['selection']['path']) as selection:
            np.testing.assert_array_equal(selection['removed_indices'], [1,2])
        reports.append(report)
    assert reports[0]['camera_reference']['scene_radius']*1e7 == pytest.approx(reports[2]['camera_reference']['scene_radius'])


def test_duplicate_cube_faces_do_not_weight_radius_and_disabled_is_exact(tmp_path):
    a, ac = fixture(tmp_path/'a')
    b, bc = fixture(tmp_path/'b', duplicates=37)
    first = run_size_filter(a, ac, tmp_path/'one', OPTIONS)
    second = run_size_filter(b, bc, tmp_path/'two', OPTIONS)
    assert first['camera_reference']['scene_radius'] == second['camera_reference']['scene_radius']
    assert first['artifact']['sha256'] == second['artifact']['sha256']
    disabled = run_size_filter(a, ac, tmp_path/'disabled', read_size_filter_options({}))
    assert disabled['removed_rows'] == 0 and Path(disabled['artifact']['path']).read_bytes() == a.read_bytes()


@pytest.mark.parametrize('value', [None, [], {'enabled':1}, {'max_sigma_camera_radius_ratio':True},
    {'max_sigma_camera_radius_ratio':float('nan')}, {'max_sigma_camera_radius_ratio':float('inf')},
    {'max_sigma_camera_radius_ratio':0}, {'max_sigma_camera_radius_ratio':11}, {'unknown':1}])
def test_bad_options_rejected(value):
    with pytest.raises(ValueError): read_size_filter_options({'size_filter':value})


@pytest.mark.parametrize('change', ['one_station','inconsistent_station','coincident_ids','nonrigid','unknown_frame','frame_mismatch'])
def test_invalid_camera_reference_rejected_without_output(tmp_path, change):
    ply, camera = fixture(tmp_path/'input')
    doc = json.loads(camera.read_text())
    if change == 'one_station': doc['frames'] = doc['frames'][:1]
    if change == 'inconsistent_station': doc['frames'][1]['station_id'] = '0'
    if change == 'coincident_ids': doc['frames'][1]['transform_matrix'] = copy.deepcopy(doc['frames'][0]['transform_matrix'])
    if change == 'nonrigid': doc['frames'][0]['transform_matrix'][0][0] = 2
    if change == 'unknown_frame': doc['camera_convention'] = 'unknown'
    if change == 'frame_mismatch': doc['coordinate_frame'] = 'ENU'
    export.write_json(camera, doc)
    with pytest.raises(ValueError): run_size_filter(ply, camera, tmp_path/'bad', OPTIONS)
    assert not (tmp_path/'bad').exists()


def test_nonfinite_ply_and_empty_result_rejected(tmp_path):
    ply, camera = fixture(tmp_path/'input')
    with pytest.raises(ValueError, match='all Gaussians'):
        run_size_filter(ply, camera, tmp_path/'empty', dict(enabled=True,max_sigma_camera_radius_ratio=1e-4))
    payload = bytearray(ply.read_bytes()); offset = payload.index(b'end_header\n')+len(b'end_header\n')
    payload[offset:offset+4] = np.array([np.nan],'<f4').tobytes(); ply.write_bytes(payload)
    with pytest.raises(ValueError, match='Nonfinite'):
        run_size_filter(ply, camera, tmp_path/'nan', OPTIONS)
    assert not (tmp_path/'empty').exists() and not (tmp_path/'nan').exists()


def test_nonstandard_comment_order_requires_explicit_source_frame_binding(tmp_path):
    ply, camera = fixture(tmp_path/'input')
    payload = ply.read_bytes().replace(b'comment coordinates EDN metres; world_up 0 -1 0\n',b'')
    payload = payload.replace(b'ply\n', b'ply\ncomment Vertical axis: y\n', 1); ply.write_bytes(payload)
    with pytest.raises(ValueError, match='ply_binding'):
        run_size_filter(ply,camera,tmp_path/'unknown',OPTIONS)
    doc=json.loads(camera.read_text()); doc['ply_binding']=dict(sha256=export.sha256(ply),coordinate_frame='EDN',units='metres')
    export.write_json(camera,doc)
    report=run_size_filter(ply,camera,tmp_path/'bound',OPTIONS)
    assert report['removed_rows']==2


def export_fixture(root, enabled):
    ply, camera = fixture(root/'input')
    job=root/'job';(job/'training').mkdir(parents=True);(job/'sfm/dataset').mkdir(parents=True)
    shutil.copyfile(ply,job/'training/model.ply');shutil.copyfile(camera,job/'sfm/dataset/transforms_train.json')
    export.write_json(job/'training/manifest.json',dict(status='completed',selection=dict(accepted_model='model.ply',accepted_model_sha256=export.sha256(ply)),
        provenance=dict(inputs={'transforms_train.json':export.sha256(camera)}),quality={'synthetic_source_metric':.7}))
    config={'size_filter':dict(OPTIONS,enabled=enabled)}
    export.write_json(job/'config.json',config)
    result=export.run(config,job,{})
    return job,result


def test_generation_export_opt_in_and_original_quality_scope(tmp_path):
    job, report=export_fixture(tmp_path/'enabled',True)
    assert report['size_filter']['removed_rows']==2
    assert export.sha256(job/'export/source.ply')==export.sha256(job/'training/model.ply')
    assert report['quality'] is None and report['source_quality']=={'synthetic_source_metric':.7}
    disabled, plain=export_fixture(tmp_path/'disabled',False)
    assert 'size_filter' not in plain and plain['quality']=={'synthetic_source_metric':.7}
    assert export.sha256(disabled/'export/scene.ply')==export.sha256(disabled/'training/model.ply')


def remote_fixture(root):
    job,report=export_fixture(root,True)
    outputs=dict(collect=('collection/manifest.json',),preprocess=('prepared/manifest.json',),sfm=('sfm/manifest.json',),
                 train=('training/manifest.json',),export=('export/scene.ply','export/report.json'))
    for stage in ('collect','preprocess','sfm'): export.write_json(job/outputs[stage][0],{'status':'completed'})
    stages=tuple(Stage(name,('python',),outputs[name]) for name in STAGES)
    state=dict(id='synthetic',status='completed',stages=[])
    for stage in stages:
        state['stages'].append(dict(name=stage.name,status='completed',exit_code=0,outputs=[dict(path=path,sha256=export.sha256(job/path),bytes=(job/path).stat().st_size) for path in stage.outputs]))
    export.write_json(job/'remote_state.json',state)
    return job,stages


def test_remote_replays_filter_and_rejects_selection_tamper(tmp_path):
    job,stages=remote_fixture(tmp_path)
    assert verify_result(job,'synthetic',stages,'export/scene.ply')['vertex_count']==2
    path=job/'export/size_filter/selection.npz'
    np.savez_compressed(path,removed_indices=np.array([0,2],dtype='<i8'),source_vertex_count=np.array(4,dtype='<i8'))
    with pytest.raises(ValueError,match='provenance'):
        verify_result(job,'synthetic',stages,'export/scene.ply')


def test_remote_rejects_undeclared_filter_and_wrong_camera_binding(tmp_path):
    job,stages=remote_fixture(tmp_path)
    export.write_json(job/'config.json',{})
    with pytest.raises(ValueError,match='configuration'): verify_result(job,'synthetic',stages,'export/scene.ply')


def test_payload_tamper_is_rejected_even_with_updated_output_hash(tmp_path):
    ply,camera=fixture(tmp_path/'input');report=run_size_filter(ply,camera,tmp_path/'out',OPTIONS)
    out=Path(report['artifact']['path']); data=bytearray(out.read_bytes()); data[-4:]=np.array([.25],'<f4').tobytes();out.write_bytes(data)
    digest=export.sha256(out);report['artifact']['sha256']=digest;report['selection']['artifact_sha256']=digest
    with pytest.raises(ValueError,match='exact retained-row'):
        verify_size_filter_report(report,ply,out,camera,report['selection']['path'])


@pytest.mark.parametrize('tamper',[False,True])
def test_local_job_completion_replays_enabled_export_lineage(tmp_path,tamper):
    from tools.streetview_app.tests.test_jobs import selection
    prepared,_=export_fixture(tmp_path/'prepared',True)
    script=tmp_path/'local_fixture.py'
    script.write_text('''import pathlib, shutil, sys
stage, job, prepared, tamper = sys.argv[1:]
job, prepared = pathlib.Path(job), pathlib.Path(prepared)
if stage == 'export':
    for folder in ('training', 'sfm', 'export'):
        shutil.copytree(prepared/folder, job/folder, dirs_exist_ok=True)
    if tamper == 'true':
        path=job/'export/source.ply'
        data=bytearray(path.read_bytes()); data[-4:]=b'\\x00\\x00\\x80\\x3e';path.write_bytes(data)
(job/(stage+'.done')).write_text('completed synthetic stage; no training')
''',encoding='utf8')
    stages=tuple(Stage(name,(sys.executable,str(script),name,'{job_dir}',str(prepared),str(tamper).lower()),
        (name+'.done',)+(('export/scene.ply','export/report.json') if name=='export' else ())) for name in STAGES)
    manager=JobManager(tmp_path/'jobs',Backend(name='synthetic-only',stages=stages))
    try:
        config=selection();config['size_filter']=OPTIONS.copy()
        started=manager.start(config);done=manager.wait(started['id'],20)
        if tamper:
            assert done['status']=='failed' and done['artifact'] is None
            assert 'binding differs' in done['error']
        else:
            assert done['status']=='completed',done
            assert done['artifact']['vertex_count']==2
            assert done['artifact']['size_filter_verified']['exact_retained_row_bytes']
    finally:
        manager.close()
