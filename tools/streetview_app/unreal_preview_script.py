"""Unreal-native preview template. The launcher prepends CONFIG_PATH.

Only a fresh generated level is saved. Existing assets and render configuration
files are read-only. This uses Unreal's supported Python API, not UI automation.
"""
from pathlib import Path
import hashlib
import json
import math
import os
import re
import time
import traceback

import unreal as u


CONFIG = json.loads(Path(CONFIG_PATH).read_text(encoding='utf-8'))
OUT = Path(CONFIG['report_path'])
ROOT = Path(u.Paths.convert_relative_path_to_full(u.Paths.project_dir()))
LEVELS = u.get_editor_subsystem(u.LevelEditorSubsystem)
ACTORS = u.get_editor_subsystem(u.EditorActorSubsystem)
EDITOR = u.get_editor_subsystem(u.UnrealEditorSubsystem)
REPORT = dict(id=CONFIG['id'], pid=os.getpid(), status='running', map=CONFIG['map'],
              source_sha256=CONFIG['source_sha256'], camera_sha256=CONFIG['camera_sha256'])
STATE = dict(phase='create', next_at=time.monotonic()+5, busy=False)
HANDLE = None


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_report():
    temporary = OUT.with_suffix('.tmp')
    temporary.write_text(json.dumps(REPORT, indent=2), encoding='utf-8')
    temporary.replace(OUT)


def pose(frame):
    m = frame['transform_matrix']
    basis = lambda v: [v[2], v[0], -v[1]]
    forward = basis([-m[i][2] for i in range(3)])
    right = basis([m[i][0] for i in range(3)])
    up = basis([m[i][1] for i in range(3)])
    return dict(position_cm=[100*m[2][3], 100*m[0][3], -100*m[1][3]],
                forward=forward, right=right, up=up,
                angles=[math.degrees(math.atan2(forward[2], math.hypot(forward[0], forward[1]))),
                        math.degrees(math.atan2(forward[1], forward[0])),
                        math.degrees(math.atan2(-right[2], up[2]))])


def current():
    world = EDITOR.get_editor_world()
    if not world or world.get_path_name().split('.')[0] != CONFIG['map']:
        raise RuntimeError('Refusing to modify an unrelated level')


def focus(camera):
    LEVELS.pilot_level_actor(camera)
    LEVELS.editor_set_game_view(True)
    ACTORS.set_selected_level_actors([])


def verify_existing_files():
    for path, expected in STATE['protected'].items():
        if not Path(path).is_file() or sha(path) != expected:
            raise RuntimeError('An existing project map or configuration changed: '+path)


def create():
    if (not re.fullmatch(r'/Game/[A-Za-z0-9_/]+', CONFIG['map'])
            or not CONFIG['map'].endswith('_'+CONFIG['id'][:12])):
        raise RuntimeError('Invalid isolated preview level name')
    if ROOT.resolve() != Path(CONFIG['project']).resolve().parent:
        raise RuntimeError('Unexpected Unreal project')
    if u.EditorAssetLibrary.does_asset_exist(CONFIG['map']):
        raise RuntimeError('Preview level already exists; refusing to overwrite')
    if (sha(CONFIG['source_ply']) != CONFIG['source_sha256'] or sha(CONFIG['renderer_ply']) != CONFIG['source_sha256']
            or sha(CONFIG['camera_json']) != CONFIG['camera_sha256']):
        raise RuntimeError('Bound PLY or cameras changed before import')
    protected = list((ROOT/'Config').glob('*.ini')) + list((ROOT/'Content').rglob('*.umap'))
    STATE['protected'] = {str(path): sha(path) for path in protected}
    cls = u.load_class(None, '/Script/MLSLabsRenderer.GaussianSplattingActor')
    if cls is None:
        raise RuntimeError('MLSLabsRenderer is unavailable')
    LEVELS.eject_pilot_level_actor()
    if not LEVELS.new_level(CONFIG['map']):
        raise RuntimeError('Cannot create a new preview level')
    current()
    actor = ACTORS.spawn_actor_from_class(cls, u.Vector(0, 0, 0), u.Rotator(pitch=0, yaw=0, roll=0))
    actor.set_actor_label('GS_'+CONFIG['label'])
    actor.set_folder_path('GaussianModel')
    actor.get_editor_property('splatting_component').set_editor_property('splat_data_path', CONFIG['renderer_ply'])
    for frame in CONFIG['cameras']:
        expected = pose(frame)
        pitch, yaw, roll = expected['angles']
        camera = ACTORS.spawn_actor_from_class(u.CameraActor, u.Vector(*expected['position_cm']),
                                               u.Rotator(pitch=pitch, yaw=yaw, roll=roll))
        camera.set_actor_label(frame['label'])
        camera.set_folder_path('ComparisonCameras')
        component = camera.get_component_by_class(u.CameraComponent)
        component.set_editor_property('field_of_view', math.degrees(2*math.atan(frame['w']/(2*frame['fl_x']))))
        component.set_editor_property('aspect_ratio', frame['w']/frame['h'])
        component.set_editor_property('constrain_aspect_ratio', True)
        component.set_editor_property('post_process_blend_weight', 0.)
    if not LEVELS.save_current_level() or not LEVELS.load_level(CONFIG['map']):
        raise RuntimeError('Cannot save and reload preview level')
    current()
    actors = ACTORS.get_all_level_actors()
    splats = [a for a in actors if a.get_class().get_name() == 'GaussianSplattingActor']
    if len(splats) != 1 or splats[0].get_editor_property('splatting_component').get_editor_property('splat_data_path') != CONFIG['renderer_ply']:
        raise RuntimeError('Serialized PLY reference differs')
    splat = splats[0]
    for vector, target in [(splat.get_actor_location(), [0, 0, 0]), (splat.get_actor_scale3d(), [1, 1, 1])]:
        if any(abs(getattr(vector, axis)-value) > 1e-6 for axis, value in zip('xyz', target)):
            raise RuntimeError('Splat transform is not identity')
    if any(abs(getattr(splat.get_actor_rotation(), axis)) > 1e-6 for axis in ('pitch', 'yaw', 'roll')):
        raise RuntimeError('Splat rotation is not identity')
    by_label = {f['label']: f for f in CONFIG['cameras']}
    cameras = [a for a in actors if a.get_actor_label() in by_label]
    if len(cameras) != len(by_label):
        raise RuntimeError('Serialized camera count differs')
    maximum_error = 0.
    for camera in cameras:
        expected = pose(by_label[camera.get_actor_label()])
        for key, getter in [('position_cm', camera.get_actor_location), ('forward', camera.get_actor_forward_vector),
                            ('right', camera.get_actor_right_vector), ('up', camera.get_actor_up_vector)]:
            actual = getter()
            error = max(abs(getattr(actual, axis)-value) for axis, value in zip('xyz', expected[key]))
            maximum_error = max(maximum_error, error)
            if error > (.01 if key == 'position_cm' else 1e-5):
                raise RuntimeError('Serialized calibrated camera differs')
    settings = {key: u.SystemLibrary.get_console_variable_string_value(key) for key in
                ['r.RayTracing', 'r.AntiAliasingMethod', 'r.ScreenPercentage', 'r.DefaultFeature.AutoExposure',
                 'r.DefaultFeature.MotionBlur', 'r.DynamicGlobalIlluminationMethod', 'r.ReflectionMethod',
                 'r.PostProcessing.PropagateAlpha']}
    for key, expected in [('r.RayTracing', 0), ('r.AntiAliasingMethod', 0), ('r.ScreenPercentage', 100),
                          ('r.DefaultFeature.AutoExposure', 0), ('r.DefaultFeature.MotionBlur', 0),
                          ('r.DynamicGlobalIlluminationMethod', 0), ('r.ReflectionMethod', 0)]:
        if float(settings[key]) != expected:
            raise RuntimeError('Unreal render setting differs: '+key)
    if settings['r.PostProcessing.PropagateAlpha'].lower() not in ('true', '1', '2'):
        raise RuntimeError('MLSLabsRenderer alpha propagation is not enabled')
    initial = next(a for a in cameras if a.get_actor_label() == CONFIG['cameras'][0]['label'])
    focus(initial)
    REPORT.update(serialized_verified=True, cameras=len(cameras), maximum_camera_error=maximum_error,
                  render_settings=settings, initial_camera=initial.get_actor_label(), renderer='MLSLabsRenderer')
    STATE.update(initial=initial, phase='capture', next_at=time.monotonic()+18)


def tick(delta):
    if STATE['busy'] or time.monotonic() < STATE['next_at']:
        return
    STATE['busy'] = True
    try:
        if STATE['phase'] == 'create':
            create()
        elif STATE['phase'] == 'capture':
            current()
            frame = CONFIG['cameras'][0]
            width, height = 1280, max(128, min(2160, round(1280*frame['h']/frame['w'])))
            task = u.AutomationLibrary.take_high_res_screenshot(width, height, CONFIG['screenshot_path'],
                                                                camera=STATE['initial'], delay=2.)
            if not task or not task.is_valid_task():
                raise RuntimeError('Screenshot task was rejected')
            STATE.update(phase='await', task=task, deadline=time.monotonic()+90, next_at=time.monotonic()+3)
        elif STATE['phase'] == 'await':
            image = Path(CONFIG['screenshot_path'])
            if not STATE['task'].is_task_done() or not image.is_file() or image.stat().st_size < 1000:
                if time.monotonic() > STATE['deadline']:
                    raise RuntimeError('Screenshot timed out')
                STATE['next_at'] = time.monotonic()+3
                return
            current()
            focus(STATE['initial'])
            if not LEVELS.save_current_level():
                raise RuntimeError('Cannot save preview camera')
            verify_existing_files()
            if sha(CONFIG['source_ply']) != CONFIG['source_sha256'] or sha(CONFIG['camera_json']) != CONFIG['camera_sha256']:
                raise RuntimeError('Source files changed during preview')
            u.EditorAssetLibrary.sync_browser_to_objects([CONFIG['map']])
            REPORT.update(status='completed', screenshot_sha256=sha(image), source_unchanged=True,
                          previous_maps_unchanged=True, render_configs_unchanged=True)
            u.unregister_slate_post_tick_callback(HANDLE)
        save_report()
    except Exception:
        REPORT.update(status='failed', error=traceback.format_exc())
        save_report()
        u.log_error(REPORT['error'])
        u.unregister_slate_post_tick_callback(HANDLE)
    finally:
        STATE['busy'] = False


u.EditorPythonScripting.set_keep_python_script_alive(True)
HANDLE = u.register_slate_post_tick_callback(tick)
save_report()
