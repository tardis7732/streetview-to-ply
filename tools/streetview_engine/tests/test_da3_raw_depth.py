import ast
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from tools.streetview_engine import da3_raw_depth as module
from tools.streetview_engine.export import sha256, write_json
from tools.streetview_engine.tests.test_brush_refine import dataset


def inputs(tmp_path):
    root, frames = dataset(tmp_path/'source')
    for i, frame in enumerate(frames):
        alpha = root/(str(i)+'_alpha.npy')
        np.save(alpha, np.zeros((32, 32), np.float32))
        frame.update(original_valid_mask_path=frame['mask_path'], original_valid_mask_sha256=sha256(root/frame['mask_path']),
                     image_sha256=sha256(root/frame['file_path']), edit_alpha_path=alpha.name, edit_alpha_sha256=sha256(alpha))
    camera = root/'cameras.json'
    write_json(camera, dict(coordinate_frame='EDN', units='metres', camera_convention='OpenGL_c2w', frames=frames))
    return root, camera, frames


def test_camera_baseline_scale_handles_collinear_stations_and_similarity():
    known, predicted, stations = [], [], []
    for index, station in enumerate(('a', 'b', 'c')):
        for face in range(6):
            view = np.eye(4)
            view[0, 3] = -index*7
            known.append(view.copy())
            view[0, 3] /= 3
            predicted.append(view)
            stations.append(station)
    scale, report = module.baseline_scale(predicted, known, stations)
    assert scale == 3 and report['known_camera_rank'] == 1
    expected = np.array(known)
    expected[:, :3, 3] *= 9
    scale2, _ = module.baseline_scale(predicted, expected, stations)
    assert np.isclose(scale2, 27)
    scale3, _ = module.baseline_scale(predicted+[predicted[0]]*5, known+[known[0]]*5, stations+['a']*5)
    assert np.isclose(scale3, scale)


def test_raw_evidence_rejects_metric_sky_edits_edges_and_invalid_masks():
    depth = np.full((14, 14), 10., np.float32)
    sky = np.zeros_like(depth)
    valid = np.full((14, 14), 255, np.uint8)
    alpha = np.zeros_like(depth)
    sky[2, 2] = .3
    alpha[5, 5] = 1e-6
    valid[8, 8] = 0
    depth[11, 11] = 30
    mask, _ = module.evidence_mask(depth, sky, valid, alpha, 14)
    assert not mask[2, 2] and not mask[5, 5] and not mask[8, 8]
    assert not mask[10:13, 10:13].any()
    assert mask[0, 0]
    alpha[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        module.evidence_mask(depth, sky, valid, alpha, 14)


def test_frame_roster_checks_bindings_and_deduplicates_identical_views(tmp_path):
    root, camera, frames = inputs(tmp_path)
    initial = module._frames(root, camera)
    document = json.loads(camera.read_text())
    document['frames'] += [copy.deepcopy(frames[0])]*4
    write_json(camera, document)
    assert module._frames(root, camera) == initial
    document['frames'][0]['w'] = 31
    write_json(camera, document)
    with pytest.raises(ValueError):
        module._frames(root, camera)


def test_two_model_worker_manifest_is_explicitly_uncalibrated(tmp_path, monkeypatch):
    root, camera, frames = inputs(tmp_path)
    options = dict(side=504, gpu_lock=str(tmp_path/'gpu.lock'), metric={}, pose={})
    monkeypatch.setattr(module.sys, 'platform', 'linux')
    monkeypatch.setattr(module, 'validate_assets', lambda settings: settings)
    calls = []
    class Process:
        def __init__(self, command, **kwargs):
            calls.append(command)
            config = json.loads(Path(command[command.index('--worker-config')+1]).read_text())
            out = Path(config['output'])
            mode = command[-1]
            if mode == 'metric':
                write_json(out/'metric_manifest.json', dict(status='completed', frames=[]))
            else:
                records = []
                for index, frame in enumerate(frames):
                    path = out/'raw'/(str(index)+'.npz')
                    side = 504
                    K = np.array([[frame['fl_x']*side/frame['w'], 0, frame['cx']*side/frame['w']],
                                  [0, frame['fl_y']*side/frame['h'], frame['cy']*side/frame['h']], [0, 0, 1]])
                    np.savez(path, metric_calibration_accepted=False, depth_z=np.ones((side, side), np.float32),
                             evidence_valid=np.ones((side, side), bool), K=K,
                             camera_from_world=np.linalg.inv(np.asarray(frame['transform_matrix'])@np.diag([1, -1, -1, 1])),
                             source_image_sha256=frame['image_sha256'], station_id=frame['station_id'],
                             image=frame['file_path'], world_frame='EDN', unit='metres', depth_convention='camera_z', pixel_center_offset=.5)
                    records.append(dict(image=frame['file_path'], station_id=frame['station_id'], face=frame['face'],
                                        depth_path=str(path), depth_sha256=sha256(path)))
                write_json(out/'pose_manifest.json', dict(status='completed', frames=records, groups=[],
                            auxiliary_ray_head_invariance_verified=True))
        def wait(self):
            return 0
        def poll(self):
            return 0
    monkeypatch.setattr(module.subprocess, 'Popen', Process)
    out = tmp_path/'result'
    result = module.infer_raw_depth(root, camera, out, options)
    assert [command[-1] for command in calls] == ['metric', 'pose']
    assert result['status'] == 'completed' and len(result['frames']) == 3
    assert result['metric_calibration_accepted'] is False
    assert result['role'] == 'heuristic_non_sky_abstention_only'
    assert result['cameras_sha256'] == sha256(camera)
    assert (out/'manifest.json').is_file()
    assert all('infer_gs' not in command for command in calls)


def test_exact_accepted_pose_and_baseline_math_is_preserved():
    expected = {'baseline_scale':'0e8bb77322a4ddd15ae60062ef2934d70ef3e875a5917d5aae02f6c9259c7404',
                'pose_group':'41b34ed64a61bcf8d6fc21b254291a8667582a59d1fb5ce030ea73be9aa4eba6'}
    text = Path(module.__file__).read_text(encoding='utf-8')
    tree = ast.parse(text)
    actual = {n.name:hashlib.sha256(ast.dump(n, include_attributes=False).encode()).hexdigest()
              for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in expected}
    assert actual == expected
    assert 'infer_gs=False' in text and 'metric_calibration_accepted=False' in text
