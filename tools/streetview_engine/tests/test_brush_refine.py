import ast
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from tools.streetview_engine import brush_refine as module
from tools.streetview_engine.export import sha256, write_json, write_model
from tools.streetview_engine.sfm_dataset import write_point_ply


def dataset(root):
    root.mkdir(parents=True)
    frames = []
    for index, station in enumerate(('a', 'b', 'c')):
        matrix = np.eye(4)
        matrix[0, 3] = index-1
        frame = dict(station_id=station, file_path=station+'.png', mask_path=station+'_mask.png',
                     foreground_mask_path=station+'_mask.png', w=32, h=32, fl_x=16., fl_y=16., cx=16., cy=16.,
                     transform_matrix=matrix.tolist(), face='F')
        Image.fromarray(np.full((32, 32, 3), 128, np.uint8)).save(root/frame['file_path'])
        Image.fromarray(np.full((32, 32), 255, np.uint8)).save(root/frame['mask_path'])
        frames.append(frame)
    xyz = np.array([[0, 0, -4], [0, 1, -4], [1, 0, -4], [1, 1, -4]], float)
    rgb = np.full((4, 3), 128, np.uint8)
    np.savez(root/'init_points.npz', xyz=xyz, rgb=rgb)
    write_point_ply(root/'init.ply', xyz, rgb)
    base = dict(coordinate_frame='EDN', units='metres', camera_convention='OpenGL_c2w')
    write_json(root/'transforms_train.json', dict(base, frames=frames[:2]))
    write_json(root/'transforms_heldout.json', dict(base, frames=frames[2:]))
    np.savez(root/'sparse_depth_observations.npz', frame_name=np.array(['a.png', 'b.png']),
             station_id=np.array(['a', 'b']), split=np.array(['train', 'train']), point_id=np.array([1, 1]),
             xy=np.array([[20., 16.], [16., 16.]]), depth_z=np.array([4., 4.]),
             support_station_count=np.array([2, 2]), triangulation_angle_degrees=np.array([10., 10.]),
             reprojection_error_px=np.array([0., 0.]))
    write_json(root/'sparse_depth_manifest.json', dict(base, status='supported_observations',
        depth_convention='camera_z', pixel_center_offset=.5, observation_kind='actual_sfm_tracks',
        geometry_scope='transductive_shared_sfm', npz='sparse_depth_observations.npz',
        sha256=sha256(root/'sparse_depth_observations.npz'),
        transforms_train_sha256=sha256(root/'transforms_train.json'),
        transforms_heldout_sha256=sha256(root/'transforms_heldout.json'), seed_npz_sha256=sha256(root/'init_points.npz'),
        train_station_ids=['a', 'b'], heldout_station_ids=['c']))
    files = ['transforms_train.json', 'transforms_heldout.json', 'init.ply', 'init_points.npz',
             'sparse_depth_observations.npz', 'sparse_depth_manifest.json']
    write_json(root/'dataset_manifest.json', dict(base, seed_colors_exclude_heldout=True,
                seed_color_station_ids=['a', 'b'], files={n:sha256(root/n) for n in files}))
    return root, frames


def test_bridge_keeps_rgb_seed_camera_and_real_depth_without_holdout_leakage(tmp_path):
    source, frames = dataset(tmp_path/'source')
    output, depth = tmp_path/'prepared/data', tmp_path/'prepared/depth'
    receipt = module.prepare_dataset(source, output, depth)
    assert receipt['training_stations'] == 2 and receipt['heldout_stations'] == 1
    assert sha256(source/'init.ply') == sha256(output/'init.ply')
    train = json.loads((output/'transforms_train.json').read_text())['frames']
    assert {f['station_id'] for f in train} == {'a', 'b'}
    assert sha256(source/'a.png') == sha256(output/train[0]['file_path'])
    assert train[0]['transform_matrix'] == frames[0]['transform_matrix']
    assert train[0]['mask_path'] == 'masks/train_000000.png'
    dmanifest = json.loads((depth/'depth_manifest.json').read_text())
    assert len(dmanifest['entries']) == 2
    for frame, entry in zip(train, dmanifest['entries']):
        with np.load(depth/entry['npz']) as values:
            assert values['valid'].sum() == 1
            assert set(values['source_count'][values['valid']]) == {2}
            assert set(values['depth_z'][values['valid']]) == {4.}
            A = np.asarray(json.loads((output/'dataset_manifest.json').read_text())['world_from_enu'])
            expected = np.linalg.inv(np.asarray(frame['transform_matrix'])@np.diag([1, -1, -1, 1]))[:3]
            np.testing.assert_allclose(values['camera_from_world']@np.linalg.inv(A), expected)
            assert values['depth_z'].shape == (32, 32)
    # A changed source binding is rejected before producing any adapted data.
    with (source/'init_points.npz').open('ab') as stream:
        stream.write(b'tamper')
    with pytest.raises(ValueError, match='binding'):
        module.prepare_dataset(source, tmp_path/'second', tmp_path/'second_depth')
    assert not (tmp_path/'second').exists()


def test_physical_normalization_is_face_duplicate_and_similarity_invariant(tmp_path):
    _, frames = dataset(tmp_path/'source')
    center, radius = module.physical_normalization(frames)
    center2, radius2 = module.physical_normalization(frames + [copy.deepcopy(frames[0])]*7)
    np.testing.assert_array_equal(center, center2)
    assert radius == radius2
    angle = .7
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    offset = np.array([30, -8, 12])
    changed = copy.deepcopy(frames)
    for frame in changed:
        matrix = np.asarray(frame['transform_matrix'])
        matrix[:3, 3] = rotation@matrix[:3, 3]*7 + offset
        frame['transform_matrix'] = matrix.tolist()
    c3, r3 = module.physical_normalization(changed)
    np.testing.assert_allclose(c3, rotation@center*7+offset)
    assert np.isclose(r3, radius*7)


def test_completed_stage_requires_brush_source_and_depth_gradient(tmp_path, monkeypatch):
    root = tmp_path/'job'
    dataset(root/'sfm/dataset')
    binary, renderer = tmp_path/'brush', tmp_path/'gsplat.so'
    binary.write_bytes(b'native fixture'); renderer.write_bytes(b'renderer fixture')
    settings = {'brush_refine':dict(brush_binary=str(binary), brush_sha256=sha256(binary),
                                  renderer_library=str(renderer), renderer_library_sha256=sha256(renderer))}
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **k: 'brush-cli 0.3.0\n')
    calls = []
    def execute(command, logfile, terminal=False):
        calls.append(command)
        if terminal:
            out = Path(command[command.index('--export-path')+1])/'brush_40000.ply'
        else:
            out = Path(command[command.index('--output')+1])/'final.ply'
            out.parent.mkdir()
        artifact = write_model(out, means=np.zeros((4, 3)), log_scales=np.zeros((4, 3)),
                    quats=np.array([[1., 0, 0, 0]]*4), opacity_logits=np.zeros(4),
                    sh0=np.zeros((4, 1, 3)), shN=np.zeros((4, 8, 3)))
        if not terminal:
            source = command[command.index('--init-gaussians-ply')+1]
            write_json(out.parent/'training_run.json', dict(status='process_completed', exit_code=0,
                       completed_steps=6000, optimization_steps=6000, fixed_count=True, point_order_preserved=True,
                       initial_gaussians=4, final_gaussians=4, source_initial_ply_sha256=sha256(source),
                       depth_gradient_verified=True, depth_supervised_steps=1, settings=module.REFINE_OPTIONS))
    monkeypatch.setattr(module, '_execute', execute)
    result = module.run({}, root, settings)
    assert len(calls) == 2 and result['status'] == 'completed'
    assert result['completed_steps'] == 46000
    assert result['selection']['accepted_model_sha256'] == sha256(root/'training/model.ply')
    assert calls[0][2:2+len(module.BRUSH_OPTIONS)] == module.BRUSH_OPTIONS
    assert calls[1][calls[1].index('--depth-weight')+1] == '0.002'
    assert result['train_station_ids'] == ['a', 'b'] and result['heldout_station_ids'] == ['c']
    assert result['quality']['quality_improved'] is False
    original_execute = execute
    def no_depth(command, logfile, terminal=False):
        original_execute(command, logfile, terminal)
        if not terminal:
            record = Path(command[command.index('--output')+1])/'training_run.json'
            value = json.loads(record.read_text())
            value['depth_gradient_verified'] = False
            write_json(record, value)
    monkeypatch.setattr(module, '_execute', no_depth)
    other = tmp_path/'failed_job'
    dataset(other/'sfm/dataset')
    with pytest.raises(ValueError, match='depth-gradient'):
        module.run({}, other, settings)
    rejected = json.loads((other/'training/manifest.json').read_text())
    assert rejected['status'] == 'failed' and rejected['selection']['accepted_model'] is None


def test_port_preserves_accepted_camera_loss_depth_and_render_math():
    expected = {
        'prepare_frame': 'b9a6e847e2143ee74752251aa160e4a2e28810429534b85cc9e4c5bd4effa72b',
        'load_depths': '2c270bce72f11cc47c4dd2d99ca3ccc35f214426ca16a17d2b3474da9dee4e45',
        'masked_losses': '93c8ed71b7ea132c953ff6a1ca45103f8655d5a843370efbb2be7f82676821a7',
        'render': '3fc62b82d9137fcdfa37bf929bec58062f2e87824896dddd33b48b0ed2f2ae7e',
        'evaluate': '9bfd5764c020333df13427d9f1ea0521fc312fd890237acc7f93dafd27ba6c9f',
    }
    trainer = Path(module.__file__).with_name('brush_refine_train.py')
    tree = ast.parse(trainer.read_text(encoding='utf-8'))
    actual = {node.name:hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
              for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in expected}
    assert actual == expected
    # Compilation/import discovery stays CPU-only; the actual trainer runs in the isolated GPU process.
    compile(trainer.read_text(encoding='utf-8'), str(trainer), 'exec')
    assert "training_physical_stations=len({str(f['meta']['station_id']) for f in frames})" in trainer.read_text(encoding='utf-8')
    assert 'station_index' not in trainer.read_text(encoding='utf-8')


def test_native_subprocess_keeps_its_package_root_outside_repository_cwd(tmp_path, monkeypatch):
    import sys
    monkeypatch.chdir(tmp_path)
    module._execute([sys.executable, '-c', 'from tools.streetview_engine.brush_refine import PORT_SOURCE_SHA256; print(PORT_SOURCE_SHA256)'],
                    tmp_path/'child.log')
    assert module.PORT_SOURCE_SHA256 in (tmp_path/'child.log').read_text()
