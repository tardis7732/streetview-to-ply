"""Production all-station input coverage without claiming heldout evaluation."""
import copy
import json

import numpy as np
from PIL import Image
import pytest

from tools.streetview_engine import brush_refine, panorama_workflow, sfm
from tools.streetview_engine.__main__ import stage_module
from tools.streetview_engine.export import sha256
from tools.streetview_engine.imaging import FACES, cube_camera_to_station_cv
from tools.streetview_engine.sfm_dataset import package_dataset
from tools.streetview_geometry.contracts import station_split


def _read(path):
    return json.loads(path.read_text(encoding='utf8'))


@pytest.mark.parametrize('quality', [None, {}, {'enabled': True}, {'enabled': 0}, {'enabled': 'false'}])
def test_zero_holdout_requires_explicit_boolean_quality_disabled(quality):
    settings = {'sfm': {'holdout_fraction': 0}}
    if quality is not None:
        settings['quality'] = quality
    with pytest.raises(ValueError, match='requires explicit quality.enabled=false'):
        sfm.validate_split_policy(settings)
    assert sfm.validate_split_policy(dict(settings, quality={'enabled': False})) == 'all_train'


def test_default_holdout_policy_and_station_grouping_remain_unchanged():
    assert sfm.SfMSettings().holdout_fraction == .2
    assert sfm.validate_split_policy({}) == 'physical_station_holdout'
    assert sfm.validate_split_policy({'sfm': {'holdout_fraction': .2},
                                     'quality': {'enabled': True}}) == 'physical_station_holdout'
    expected = (('a', 'b'), ('c',))
    assert station_split(['a', 'b', 'c'], holdout_fraction=.2, seed=0) == expected
    assert station_split(['a', 'b', 'c'] * 6, holdout_fraction=.2, seed=0) == expected


@pytest.mark.parametrize('stage', ['collect', 'preprocess', 'sfm', 'train', 'export'])
def test_dispatch_rejects_implicit_evaluation_before_any_stage(stage):
    settings = {'workflow': 'panorama_brush_refine', 'sfm': {'holdout_fraction': 0}}
    with pytest.raises(ValueError, match='requires explicit quality.enabled=false'):
        stage_module(stage, {'generation_mode': 'multi_view'}, settings)
    settings['quality'] = {'enabled': False}
    assert isinstance(stage_module(stage, {'generation_mode': 'multi_view'}, settings), str)


def test_direct_entry_guards_run_before_file_and_backend_work(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Invalid split policy reached file/backend work')

    monkeypatch.setattr(sfm, 'owned_file', forbidden)
    monkeypatch.setattr(brush_refine, '_operator_settings', forbidden)
    settings = {'workflow': 'panorama_brush_refine', 'sfm': {'holdout_fraction': 0}}
    missing_root = tmp_path/'must_not_be_created'
    for function in (sfm.run, brush_refine.run, panorama_workflow.run):
        with pytest.raises(ValueError, match='requires explicit quality.enabled=false'):
            function({}, missing_root, copy.deepcopy(settings))
        assert not missing_root.exists()


def _registered_cube_dataset(root):
    """Actual COLMAP poses and tracks: three stations, six calibrated faces each."""
    pc = pytest.importorskip('pycolmap')
    root.mkdir()
    (root/'prepared').mkdir()
    rec = pc.Reconstruction()
    rec.add_camera_with_trivial_rig(pc.Camera(camera_id=1, model='PINHOLE', width=96, height=96,
                                             params=[48, 48, 48, 48]))
    centers = [np.array(value, float) for value in [(-3, 0, -3), (0, 3, 0), (3, -1, 3)]]
    local_points = np.array([[-2, -2, 20], [-2, 2, 20], [2, -2, 20], [2, 2, 20]], float)
    rotations = {face: cube_camera_to_station_cv(face)[:3, :3] for face in FACES}
    face_points = {face: local_points @ rotations[face].T for face in FACES}
    mapping, captures, image_ids = {}, [], {}
    for station_index, (station, center) in enumerate(zip(('a', 'b', 'c'), centers)):
        captures.append({'station_id': station, 'pano_id': station})
        for face_index, face in enumerate(FACES):
            rotation = rotations[face]
            local = (face_points[face]-center) @ rotation
            xy = local[:, :2]/local[:, 2:]*48+48
            name = f'{station}_{face}.png'
            image_id = station_index*6+face_index+1
            image_ids[station, face] = image_id
            image = pc.Image(name=name, keypoints=xy, camera_id=1, image_id=image_id)
            pose = pc.Rigid3d(pc.Rotation3d(rotation.T), -rotation.T@center)
            rec.add_image_with_trivial_frame(image, pose)
            rgb, mask, sky = f'prepared/{name}', f'prepared/{station}_{face}_mask.png', f'prepared/{station}_{face}_sky.png'
            Image.new('RGB', (96, 96), (20+station_index*10, 40, 60)).save(root/rgb)
            Image.new('L', (96, 96), 255).save(root/mask)
            Image.new('L', (96, 96), 0).save(root/sky)
            mapping[name] = dict(file_path=rgb, mask_path=mask, sfm_mask_path=mask,
                                 sky_mask_path=sky, ground_mask_path=sky, pano_id=station,
                                 station_id=station, face=face, w=96, h=96, fl_x=48, fl_y=48,
                                 cx=48, cy=48, camera_to_station_cv=cube_camera_to_station_cv(face).tolist())
    for face in FACES:
        for point_index, point in enumerate(face_points[face]):
            track = pc.Track([pc.TrackElement(image_ids[station, face], point_index)
                              for station in ('a', 'b', 'c')])
            rec.add_point3D(point, track, np.array([255, 0, 255], np.uint8))
    rec.update_point_3d_errors()
    assert rec.compute_mean_reprojection_error() < 1e-10
    return rec, mapping, captures


@pytest.mark.parametrize('fraction,train_stations,heldout_stations', [
    (0, {'a', 'b', 'c'}, set()),
    (.2, {'a', 'b'}, {'c'}),
])
@pytest.mark.parametrize('with_original', [False, True])
def test_actual_track_package_and_brush_bridge_keep_complete_station_faces(
        tmp_path, fraction, train_stations, heldout_stations, with_original):
    source = tmp_path/'job'
    rec, mapping, captures = _registered_cube_dataset(source)
    if with_original:
        for index, row in enumerate(mapping.values()):
            alpha_path = f'prepared/original_alpha_{index}.npy'
            alpha = np.zeros((96, 96), np.float64)
            alpha[0, 0] = .25
            np.save(source/alpha_path, alpha, allow_pickle=False)
            row.update(original_file_path=row['file_path'], original_valid_mask_path=row['mask_path'],
                       edit_alpha_path=alpha_path)
    settings = sfm.SfMSettings(device='cpu', holdout_fraction=fraction)
    package = source/'dataset'
    package_dataset(rec, mapping, captures, source, package, settings, {'status': 'fixture_poses'})
    manifest = _read(package/'dataset_manifest.json')
    train = _read(package/'transforms_train.json')['frames']
    heldout = _read(package/'transforms_heldout.json')['frames']
    if with_original:
        originals = {(row['pano_id'], row['face']): row for row in mapping.values()}
        for frame in train + heldout:
            row = originals[frame['pano_id'], frame['face']]
            for field in ('original_file_path', 'original_valid_mask_path', 'edit_alpha_path'):
                assert sha256(package/frame[field]) == sha256(source/row[field])
    assert {f['station_id'] for f in train} == train_stations
    assert {f['station_id'] for f in heldout} == heldout_stations
    for frames, stations in ((train, train_stations), (heldout, heldout_stations)):
        assert len(frames) == len(stations)*6
        assert {(f['station_id'], f['face']) for f in frames} == {
            (station, face) for station in stations for face in FACES}
    assert set(manifest['seed_color_station_ids']) == train_stations
    with np.load(package/'init_points.npz', allow_pickle=False) as seeds:
        assert len(seeds['xyz']) == 24
        assert set(seeds['support_station_count']) == {len(train_stations)}
        expected_rgb = (30, 40, 60) if fraction == 0 else (25, 40, 60)
        np.testing.assert_array_equal(seeds['rgb'], np.tile(expected_rgb, (24, 1)))
    with np.load(package/'sparse_depth_observations.npz', allow_pickle=False) as depths:
        assert set(depths['station_id'][depths['split'] == 'train']) == train_stations
        assert set(depths['station_id'][depths['split'] == 'heldout']) == heldout_stations
        assert np.sum(depths['split'] == 'train') == 24*len(train_stations)
        assert np.sum(depths['split'] == 'heldout') == 24*len(heldout_stations)
        assert set(depths['support_station_count']) == {len(train_stations)}

    output, depth_dir = tmp_path/'brush/data', tmp_path/'brush/depth'
    receipt = brush_refine.prepare_dataset(package, output, depth_dir)
    brush_train = _read(output/'transforms_train.json')['frames']
    brush_val = _read(output/'transforms_val.json')['frames']
    assert len(brush_train) == len(train) and len(brush_val) == len(heldout)
    assert receipt['training_stations'] == len(train_stations)
    assert receipt['heldout_stations'] == len(heldout_stations)
    assert sha256(output/'init.ply') == sha256(package/'init.ply')
    for original, prepared in zip(train, brush_train):
        assert (original['station_id'], original['face']) == (prepared['station_id'], prepared['face'])
        assert original['transform_matrix'] == prepared['transform_matrix']
        assert sha256(package/original['file_path']) == sha256(output/prepared['file_path'])
    for entry in _read(depth_dir/'depth_manifest.json')['entries']:
        with np.load(depth_dir/entry['npz'], allow_pickle=False) as depth:
            assert depth['valid'].sum() == 4
            assert set(depth['source_count'][depth['valid']]) == {len(train_stations)}
    if fraction == 0:
        assert brush_val == []
        for record in (manifest, receipt):
            assert record['training_split'] == 'all_train'
            assert record['evaluation_status'] == 'not_run'
            assert record['quality_comparison_enabled'] is False
        assert manifest['geometry_uses_heldout_images'] is False
        assert all(f['valid_ground_truth'] is False for f in brush_train)
        assert all(f['evaluation_kind'] == 'not_run_all_train' for f in brush_train)
        # An empty split with no explicit declaration must not silently become production all-train.
        manifest.pop('training_split')
        (package/'dataset_manifest.json').write_text(json.dumps(manifest), encoding='utf8')
        with pytest.raises(ValueError, match='explicitly declare all_train'):
            brush_refine.prepare_dataset(package, tmp_path/'rejected', tmp_path/'rejected_depth')
        assert not (tmp_path/'rejected').exists()
    else:
        assert 'training_split' not in manifest
        assert len(brush_val) == 6
        assert all(f['valid_ground_truth'] is True for f in brush_val)
