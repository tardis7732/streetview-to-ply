import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pytest

from tools.streetview_app import unreal_open as mod
from tools.streetview_engine.tests.test_size_filter import fixture


def inputs(tmp_path):
    ply, camera = fixture(tmp_path/'inputs')
    document = json.loads(camera.read_text())
    document.update(w=100, h=100, fl_x=50, fl_y=50)
    camera.write_text(json.dumps(document))
    return dict(source_ply=str(ply.resolve()), camera_json=str(camera.resolve()), source_sha256=mod._sha(ply),
                camera_sha256=mod._sha(camera), coordinate_frame='EDN', units='metres', camera_convention='OpenGL_c2w')


def profile(tmp_path):
    engine = tmp_path/'UnrealEditor.exe'
    engine.write_bytes(b'fixture; never execute')
    project = tmp_path/'Project.uproject'
    project.write_text(json.dumps(dict(Plugins=[dict(Name=p, Enabled=True) for p in
                       ['MLSLabsRenderer', 'PythonScriptPlugin', 'EditorScriptingUtilities']])))
    path = tmp_path/'profile.json'
    path.write_text(json.dumps(dict(executable=str(engine), project=str(project), map_root='/Game/Tool/Maps')))
    return path


def wait(manager, job_id):
    deadline = time.monotonic()+10
    while time.monotonic() < deadline:
        result = manager.get(job_id)
        if result['status'] not in ('queued', 'running'):
            return result
        time.sleep(.02)
    pytest.fail('fixture did not finish')


def test_descriptor_hash_coordinate_pinhole_guards_and_camera_convention(tmp_path):
    values = inputs(tmp_path)
    gl = mod.prepare_artifact(values)
    assert gl['cameras'] and len(gl['cameras']) <= 12
    camera = Path(values['camera_json'])
    doc = json.loads(camera.read_text())
    for frame in doc['frames']:
        m = np.array(frame['transform_matrix'])
        m[:3, 1:3] *= -1
        frame['transform_matrix'] = m.tolist()
    doc['camera_convention'] = 'OpenCV_c2w'
    camera.write_text(json.dumps(doc))
    cv = mod.prepare_artifact(dict(values, camera_convention='OpenCV_c2w', camera_sha256=mod._sha(camera)))
    assert gl['cameras'] == cv['cameras']
    with pytest.raises(ValueError, match='변경'):
        mod.prepare_artifact(values)
    with pytest.raises(ValueError, match='EDN'):
        mod.prepare_artifact(dict(values, coordinate_frame='ENU'))
    values.update(camera_convention='OpenCV_c2w', camera_sha256=mod._sha(camera))
    doc['fl_y'] = 70
    camera.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='원근'):
        mod.prepare_artifact(dict(values, camera_sha256=mod._sha(camera)))
    with pytest.raises(ValueError):
        mod.prepare_artifact(dict(values, executable='arbitrary.exe'))


def test_capability_get_never_launch_and_insufficient_gpu_fails_before_launch(tmp_path):
    calls = []
    manager = mod.UnrealOpenManager(tmp_path/'jobs', profile(tmp_path),
                                   launcher=lambda *a, **k: calls.append(a), gpu_probe=lambda: 100)
    try:
        assert manager.capability()['available']
        assert manager.list() == [] and calls == []
        job = manager.start(inputs(tmp_path))
        result = wait(manager, job['id'])
        assert result['status'] == 'failed' and '메모리' in result['error']
        assert calls == []
    finally:
        manager.close()


def test_manager_only_completes_after_bound_native_report_and_image(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, '_pid_alive', lambda pid: bool(pid))
    calls = []
    class Process:
        pid = 98765
        def poll(self):
            return None
    def launch(command, **kwargs):
        calls.append(command)
        assert isinstance(command, list) and kwargs.get('shell') is not True
        script = Path(next(a.split('=', 1)[1] for a in command if a.startswith('-ExecutePythonScript=')))
        compile(script.read_text(encoding='utf-8'), str(script), 'exec')
        config = json.loads(script.with_name('config.json').read_text(encoding='utf-8'))
        image = Path(config['screenshot_path'])
        image.write_bytes(b'\x89PNG\r\n\x1a\n'+b'fixture'*200)
        report = dict(id=config['id'], pid=Process.pid, status='completed', map=config['map'],
                      source_sha256=config['source_sha256'], camera_sha256=config['camera_sha256'],
                      serialized_verified=True, source_unchanged=True, screenshot_sha256=mod._sha(image))
        Path(config['report_path']).write_text(json.dumps(report))
        return Process()
    manager = mod.UnrealOpenManager(tmp_path/'jobs', profile(tmp_path), launcher=launch, gpu_probe=lambda: 4096)
    try:
        values = inputs(tmp_path)
        job = manager.start(values)
        result = wait(manager, job['id'])
        assert result['status'] == 'completed', result
        assert result['rendered'] and len(calls) == 1 and result['editor_running']
        assert result['map'].startswith('/Game/Tool/Maps/L_Streetview_')
        assert mod._sha(values['source_ply']) == values['source_sha256']
        with pytest.raises(ValueError, match='창'):
            manager.start(values)
        # Download-like proof cannot accept a report pointing to another Gaussian.
        config = json.loads((Path(result['runtime_dir'])/'config.json').read_text())
        report = json.loads(Path(result['result_path']).read_text())
        with pytest.raises(ValueError, match='일치'):
            manager._verify_result(dict(report, source_sha256='0'*64), config, Process.pid)
        Path(config['screenshot_path']).write_bytes(b'tampered')
        with pytest.raises(ValueError, match='이미지'):
            manager._verify_result(report, config, Process.pid)
    finally:
        manager.close()


def test_profile_rejects_missing_plugins_and_arbitrary_commands(tmp_path):
    path = profile(tmp_path)
    raw = json.loads(path.read_text())
    raw['arguments'] = ['-ExecCmds=arbitrary']
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='프로필'):
        mod._profile(path)
    raw.pop('arguments')
    path.write_text(json.dumps(raw))
    Path(raw['project']).write_text('{}')
    with pytest.raises(ValueError, match='MLSLabsRenderer'):
        mod._profile(path)


def test_native_pose_matches_expected_edn_to_unreal_axes():
    source = Path(mod.__file__).with_name('unreal_preview_script.py').read_text()
    # Extract the pure camera transform from the actual native template.
    import ast
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'pose')
    namespace = {'math': __import__('math')}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<pose>', 'exec'), namespace)
    matrix = np.eye(4)
    matrix[:3, 3] = [1, 2, 3]
    actual = namespace['pose'](dict(transform_matrix=matrix.tolist()))
    assert actual['position_cm'] == [300, 100, -200]
    assert actual['forward'] == [-1, 0, 0]
    assert actual['right'] == [0, 1, 0]
    assert actual['up'] == [0, 0, -1]
