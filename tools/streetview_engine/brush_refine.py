"""Reusable SfM -> Brush 0.3 -> fixed-population gsplat refinement workflow.

The optimizer and inverse-depth loss are ported from the accepted depth-0.002
run. New scenes use their own registered cameras and actual SfM track depths.
Nothing here starts from learned Gaussian predictions or scene-specific seeds.
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from .export import sha256, write_json, validate_ply


BRUSH_OPTIONS = ['--total-steps', '40000', '--max-splats', '2000000', '--max-resolution', '1280',
                 '--sh-degree', '2', '--refine-every', '250', '--growth-stop-iter', '30000',
                 '--lpips-loss-weight', '0', '--seed', '42', '--eval-every', '10000',
                 '--eval-save-to-disk', '--export-every', '5000']
REFINE_OPTIONS = dict(strategy='fixed', regularization_scope='visible_sum', lr_multiplier=.1,
                      steps=6000, resolution=1280, max_splats=2000000, seed=42,
                      opacity_reg=.03, scale_reg=.05, depth_weight=.002,
                      eval_every=6000, export_every=6000, depth_alpha_min=.05,
                      depth_huber_beta=.1, sh_degree=2, cpu_workers=4)
PORT_SOURCE_SHA256 = '5cdbd7c104a67d0464e8a50f780f750d0098ea5cc0b96e180f100d759240a169'


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _file(root, name):
    if not isinstance(name, str) or Path(name).is_absolute():
        raise ValueError('Dataset members must be relative files')
    path = (root/name).resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('Dataset file escapes its root')
    return path


def physical_normalization(frames):
    """Equal physical-station weight; duplicate camera faces do not alter scale."""
    groups = {}
    for frame in frames:
        station = frame.get('station_id')
        if station is None or isinstance(station, bool) or not str(station):
            raise ValueError('Explicit physical station_id required')
        matrix = np.asarray(frame['transform_matrix'], float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError('Invalid camera matrix')
        groups.setdefault(str(station), set()).add(tuple(matrix[:3, 3]))
    if len(groups) < 2:
        raise ValueError('Multiple physical stations required')
    centers = np.asarray([np.mean(sorted(groups[k]), axis=0) for k in sorted(groups)])
    center = centers.mean(0)
    radius = float(np.linalg.norm(centers-center, axis=1).max()*1.1)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('Insufficient physical camera extent')
    return center, radius


def _operator_settings(settings):
    raw = settings.get('brush_refine', {})
    required = {'brush_binary', 'brush_sha256', 'renderer_library', 'renderer_library_sha256'}
    if not isinstance(raw, dict) or not required <= raw.keys() or raw.keys() - required - {'gpu_lock'}:
        raise ValueError('brush_refine requires pinned operator-owned Brush and gsplat renderer paths')
    for pathkey, hashkey in [('brush_binary', 'brush_sha256'), ('renderer_library', 'renderer_library_sha256')]:
        path = Path(raw[pathkey])
        if not path.is_absolute() or not path.is_file() or sha256(path) != raw[hashkey]:
            raise ValueError('Pinned backend is missing or changed: '+pathkey)
    return dict(raw)


def build_commands(options, dataset, output, depth_dir):
    brush = [options['brush_binary'], str(dataset), *BRUSH_OPTIONS,
             '--export-path', str(output/'brush'), '--export-name', 'brush_{iter}.ply']
    refine = [sys.executable, '-m', 'tools.streetview_engine.brush_refine_train', '--run-root', str(output),
              '--dataset', str(dataset), '--output', str(output/'refined'),
              '--init-gaussians-ply', str(output/'brush/brush_40000.ply'), '--depth-dir', str(depth_dir),
              '--renderer-library', options['renderer_library'],
              '--renderer-library-sha256', options['renderer_library_sha256']]
    for key, value in REFINE_OPTIONS.items():
        refine += ['--'+key.replace('_', '-'), str(value)]
    return brush, refine


def prepare_dataset(source, destination, depth_dir):
    """Adapt bound engine SfM data without recropping, resizing, or inventing depth."""
    from .training import TrainingSettings, load_sparse_depth, validate_splits, validate_dataset_manifest
    source, destination, depth_dir = Path(source).resolve(), Path(destination).resolve(), Path(depth_dir).resolve()
    if destination.exists() or depth_dir.exists() or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('Fresh independent Brush dataset and depth directories required')
    train_doc, held_doc = [_read(source/name) for name in ('transforms_train.json', 'transforms_heldout.json')]
    train, heldout = train_doc['frames'], held_doc['frames']
    station_train, station_heldout = validate_splits(train, heldout)
    manifest = _read(source/'dataset_manifest.json')
    validate_dataset_manifest(manifest, station_train, station_heldout)
    if manifest.get('training_split')=='all_train' and heldout:
        raise ValueError('An all_train dataset cannot retain heldout frames')
    if not heldout and (manifest.get('training_split')!='all_train' or manifest.get('evaluation_status')!='not_run'
                       or manifest.get('quality_comparison_enabled') is not False):
        raise ValueError('An empty heldout set must explicitly declare all_train without quality evaluation')
    for name, digest in manifest.get('files', {}).items():
        if sha256(_file(source, name)) != digest:
            raise ValueError('Source SfM dataset binding changed: '+name)
    with np.load(source/'init_points.npz', allow_pickle=False) as seed:
        xyz, rgb = seed['xyz'], seed['rgb']
        if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape or not np.isfinite(xyz).all() or len(xyz) < 4:
            raise ValueError('Actual finite SfM XYZRGB seeds required')
    # The native Brush loader sees only XYZRGB points, never a Gaussian PLY.
    seed_header = (source/'init.ply').read_bytes().split(b'end_header\n', 1)[0]
    if b'property float opacity' in seed_header or any(b'property uchar '+n not in seed_header for n in (b'red', b'green', b'blue')):
        raise ValueError('Brush initialization must be an ordinary SfM XYZRGB PLY')
    from .sfm_dataset import write_point_ply
    options = TrainingSettings(resolution=1280, depth_resolution=1280, depth_moment_weight=.002)
    sparse, sparse_receipt = load_sparse_depth(source, train, heldout, options, train_max_resolution=1280)
    if not any(row['split'] == 'train' and row['accepted_pixels'] for row in sparse.values()):
        raise ValueError('No verified actual-track depth supervision exists')
    observation_doc = _read(source/'sparse_depth_manifest.json')
    with np.load(_file(source, observation_doc['npz']), allow_pickle=False) as rows:
        point_support = {}
        for point, support in zip(rows['point_id'], rows['support_station_count']):
            if int(point) in point_support and point_support[int(point)] != int(support):
                raise ValueError('SfM point physical support count changed between observations')
            point_support[int(point)] = int(support)
    destination.mkdir(parents=True)
    depth_dir.mkdir(parents=True)
    write_point_ply(destination/'init.ply', xyz, rgb)
    if sha256(destination/'init.ply') != sha256(source/'init.ply'):
        raise ValueError('SfM XYZRGB PLY differs from its bound numerical seed arrays')
    # Recreated ordinary point serialization is proven against the numerical seed source.
    inputs = {str(source/'dataset_manifest.json'): sha256(source/'dataset_manifest.json')}
    for name in manifest['files']:
        inputs[str(_file(source, name))] = sha256(_file(source, name))
    world_from_enu = np.asarray([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
    output_frames, depth_entries = {'train': [], 'val': []}, []
    seen_stems = set()
    for split, frames in [('train', train), ('val', heldout)]:
        for index, frame in enumerate(frames):
            image_path = _file(source, frame['file_path'])
            name = f'{split}_{index:06d}'
            if name in seen_stems:
                raise ValueError('Duplicate Brush image stem')
            seen_stems.add(name)
            result = dict(frame, file_path='images/'+name+image_path.suffix.lower(),
                          mask_path='masks/'+name+'.png', render_name=name,
                          valid_ground_truth=split == 'val',
                          evaluation_kind='physical_station_appearance_holdout' if heldout else 'not_run_all_train')
            photometric = _file(source, frame['mask_path'])
            with Image.open(image_path) as image:
                if image.size != (frame['w'], frame['h']):
                    raise ValueError('Source image grid differs from its fixed camera')
            with Image.open(photometric) as mask:
                pixels = np.asarray(mask)
                if mask.mode != 'L' or mask.size != (frame['w'], frame['h']) or not np.isin(pixels, [0, 255]).all():
                    raise ValueError('Binary native-resolution training mask required')
                if split == 'train' and not pixels.any():
                    # A fully masked direction offers no RGB supervision; record the omission.
                    continue
            for src, relative in [(image_path, result['file_path']), (photometric, result['mask_path'])]:
                target = destination/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, target)
                inputs[str(src)] = sha256(src)
                if sha256(target) != inputs[str(src)]:
                    raise ValueError('Image or mask copy changed')
            output_frames[split].append(result)
            if split != 'train':
                continue
            mapped = sparse.get(frame['file_path'])
            factor = min(1., 1280/max(frame['w'], frame['h']))
            w, h = max(1, round(frame['w']*factor)), max(1, round(frame['h']*factor))
            z = np.zeros((h, w), np.float32) if mapped is None else mapped['depth_z']
            valid = np.zeros((h, w), bool) if mapped is None else mapped['valid'].copy()
            confidence = np.zeros((h, w), np.float32) if mapped is None else mapped['confidence'].copy()
            # Match the trainer's same-grid photometric validity; depth remains actual observed pixels.
            valid &= np.asarray(Image.fromarray(pixels).resize((w, h), Image.Resampling.NEAREST)) == 255
            confidence[~valid] = 0
            support = np.zeros((h, w), np.int32)
            if mapped is not None:
                support[valid] = [point_support[int(pid)] for pid in mapped['point_ids'][valid]]
            K = np.array([[frame['fl_x']*w/frame['w'], 0, frame['cx']*w/frame['w']],
                          [0, frame['fl_y']*h/frame['h'], frame['cy']*h/frame['h']], [0, 0, 1]], float)
            c2w_cv = np.asarray(frame['transform_matrix']) @ np.diag([1., -1., -1., 1.])
            path = depth_dir/(name+'.npz')
            np.savez_compressed(path, depth_z=z, valid=valid, confidence=confidence, source_count=support,
                                source_type=np.where(valid, 1, 0).astype('u1'), K=K, w=w, h=h,
                                camera_from_world=(np.linalg.inv(c2w_cv)@world_from_enu)[:3],
                                unit='metres', depth_convention='camera_z', world_frame='ENU', pixel_center_offset=.5)
            depth_entries.append(dict(image=Path(result['file_path']).name, npz=path.name,
                                      npz_sha256=sha256(path), valid_count=int(valid.sum())))
    if not output_frames['train'] or {f['station_id'] for f in output_frames['train']} != station_train:
        raise ValueError('Photometric masks removed every usable view from a physical station')
    physical_normalization(output_frames['train'])
    for split in ('train', 'val'):
        write_json(destination/('transforms_'+split+'.json'), dict(train_doc, frames=output_frames[split], ply_file_path='init.ply'))
    receipt = dict(status='prepared', initialization='actual_sfm_xyzrgb', input_bindings=inputs,
                   source_dataset=str(source), coordinate_frame='EDN', units='metres', camera_convention='OpenGL_c2w',
                   world_from_enu=world_from_enu.tolist(), processing_options=manifest.get('processing_options', {}),
                   training_stations=len(station_train), heldout_stations=len(station_heldout),
                   training_images=len(output_frames['train']), heldout_images=len(output_frames['val']),
                   initial_points=len(xyz), omitted_empty_training_views=len(train)-len(output_frames['train']),
                   evaluation='Physical station appearance holdout; shared SfM is not independent geometry ground truth',
                   source_sfm_manifest_sha256=sha256(source/'dataset_manifest.json'))
    if not heldout:
        receipt.update(training_split='all_train',evaluation_status='not_run',quality_comparison_enabled=False,
                       evaluation='All registered stations used for optimization; no heldout evaluation')
    write_json(destination/'dataset_manifest.json', receipt)
    write_json(depth_dir/'depth_manifest.json', dict(status='accepted', depth_convention='camera_z', entries=depth_entries,
               source='new actual SfM track observations', source_count_convention='physical stations including target',
               camera_frame='ENU is an axis conversion from bound EDN, not a newly inferred origin',
               processing=dict(training_max_resolution=1280, depth_resize='nearest-exact'), provenance=sparse_receipt))
    receipt['prepared_files'] = {str(p): sha256(p) for p in sorted(destination.rglob('*')) if p.is_file()}
    receipt['depth_files'] = {str(p): sha256(p) for p in sorted(depth_dir.iterdir()) if p.is_file()}
    write_json(destination.parent/'prepared_receipt.json', receipt)
    return receipt


def _verify_bindings(bindings):
    for path, digest in bindings.items():
        if sha256(path) != digest:
            raise ValueError('Bound pipeline input changed: '+path)


def _execute(command, logfile, *, terminal=False):
    """No shell; native Brush gets its expected terminal on Linux."""
    environment = dict(os.environ, RUST_LOG='info')
    package_root = str(Path(__file__).resolve().parents[2])
    environment['PYTHONPATH'] = package_root + (os.pathsep + environment['PYTHONPATH'] if environment.get('PYTHONPATH') else '')
    if terminal and sys.platform.startswith('linux'):
        import pty
        import select
        master, slave = pty.openpty()
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=slave, stderr=slave, env=environment)
        os.close(slave)
        try:
            with Path(logfile).open('wb') as log:
                while True:
                    ready, _, _ = select.select([master], [], [], 1.)
                    if ready:
                        try:
                            data = os.read(master, 65536)
                        except OSError:
                            data = b''
                        if data:
                            log.write(data); log.flush()
                        elif process.poll() is not None:
                            break
                    if process.poll() is not None and not ready:
                        break
            code = process.wait()
        finally:
            os.close(master)
            if process.poll() is None:
                process.terminate(); process.wait(timeout=30)
    else:
        with Path(logfile).open('wb') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
            try:
                code = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate(); process.wait(timeout=30)
    if code != 0:
        raise RuntimeError('Native training failed; inspect '+str(logfile))


def run(config, job_dir, settings):
    from .processing_options import read_processing_options
    if config.get('generation_mode', 'multi_view') != 'multi_view' or settings.get('initialization'):
        raise ValueError('Brush/refinement accepts only multi-view SfM initialization')
    from .sfm import validate_split_policy
    split_policy=validate_split_policy(settings)
    options = _operator_settings(settings)
    root = Path(job_dir).resolve()
    dataset = root/'sfm/dataset'
    output = root/'training'
    if output.exists():
        raise ValueError('Training output exists; use validated completed-stage reuse or a new job')
    processing = read_processing_options(config)
    if read_processing_options(_read(dataset/'dataset_manifest.json')) != processing:
        raise ValueError('Input mask options differ from the frozen generation configuration')
    dataset_manifest=_read(dataset/'dataset_manifest.json')
    if (dataset_manifest.get('training_split')=='all_train') != (split_policy=='all_train'):
        raise ValueError('Dataset training split differs from the explicit operator split policy')
    version = subprocess.check_output([options['brush_binary'], '--version'], text=True).strip()
    if version != 'brush-cli 0.3.0':
        raise ValueError('This workflow requires exactly Brush 0.3.0')
    output.mkdir()
    started = time.monotonic()
    state = dict(status='running', workflow='panorama_brush_refine', initialization='sfm_xyzrgb',
                 processing_options=processing, brush_steps=40000, refine_steps=6000,
                 brush_options=BRUSH_OPTIONS, refinement_options=REFINE_OPTIONS, selection={'accepted_model': None})
    write_json(output/'manifest.json', state)
    try:
        receipt = prepare_dataset(dataset, output/'dataset', output/'depth')
        brush, refine = build_commands(options, output/'dataset', output, output/'depth')
        write_json(output/'commands.json', dict(brush=brush, refine=refine, source_trainer_sha256=PORT_SOURCE_SHA256,
                   port_changes=['package imports', 'explicit PLY coordinate comment', 'equal physical-station normalization'],
                   sparse_depth='newly reconstructed actual SfM tracks; no depth prediction is substituted'))
        bindings = dict(receipt['input_bindings'], **receipt['prepared_files'], **receipt['depth_files'])
        lock = Path(options['gpu_lock']).open('a+') if options.get('gpu_lock') else nullcontext()
        with lock as handle:
            if handle is not None:
                if not sys.platform.startswith('linux'):
                    raise ValueError('Configured shared GPU lock requires Linux')
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX)
            _verify_bindings(bindings)
            _operator_settings(settings)
            (output/'brush').mkdir()
            _execute(brush, output/'brush.log', terminal=True)
            from .size_filter import _ply
            original, _, _ = _ply(output/'brush/brush_40000.ply')
            if original['vertex_count'] > 2000000 or original['sh_degree'] != 2:
                raise ValueError('Brush final Gaussian population or SH degree differs')
            state.update(stage='refine', brush_completed_steps=40000, brush_artifact=original)
            write_json(output/'manifest.json', state)
            _verify_bindings(bindings)
            _execute(refine, output/'refine.log')
            run_record = _read(output/'refined/training_run.json')
            if (run_record.get('status') != 'process_completed' or run_record.get('exit_code') != 0
                    or run_record.get('completed_steps') != 6000 or run_record.get('optimization_steps') != 6000
                    or not run_record.get('fixed_count') or not run_record.get('point_order_preserved')
                    or run_record.get('initial_gaussians') != original['vertex_count']
                    or run_record.get('final_gaussians') != original['vertex_count']
                    or run_record.get('source_initial_ply_sha256') != original['sha256']
                    or not run_record.get('depth_gradient_verified') or run_record.get('depth_supervised_steps', 0) <= 0):
                raise ValueError('Fixed-count refinement or active depth-gradient verification failed')
            for key, value in REFINE_OPTIONS.items():
                if run_record['settings'].get(key) != value:
                    raise ValueError('Actual refinement option differs: '+key)
            _verify_bindings(bindings)
            _operator_settings(settings)
        shutil.copyfile(output/'refined/final.ply', output/'model.ply')
        artifact = validate_ply(output/'model.ply')
        state.update(status='completed', completed_steps=46000, duration_s=time.monotonic()-started,
                     artifact=artifact, depth_gradient_verified=True, depth_supervised_steps=run_record['depth_supervised_steps'],
                     train_station_ids=sorted({str(f['station_id']) for f in _read(dataset/'transforms_train.json')['frames']}),
                     heldout_station_ids=sorted({str(f['station_id']) for f in _read(dataset/'transforms_heldout.json')['frames']}),
                     selection=dict(accepted_model='model.ply', accepted_model_sha256=artifact['sha256'],
                                    selected='user_selected_brush_fixed_depth_recipe',
                                    basis='User selected this recipe; no cross-location quality improvement is claimed'),
                     quality=dict(quality_improved=False, candidate_comparison='not_run',
                                  limitation='Same algorithm settings on new scene data; not identical geometry or independent ground truth'),
                     input_receipt_sha256=sha256(output/'prepared_receipt.json'),
                     training_run_sha256=sha256(output/'refined/training_run.json'))
        if split_policy=='all_train':
            state.update(training_split='all_train',evaluation_status='not_run')
            state['quality'].update(evaluation_status='not_run',heldout_physical_stations=0,
                                    limitation='All registered stations were used for optimization; no heldout quality evaluation was performed')
        write_json(output/'manifest.json', state)
        return state
    except BaseException as error:
        state.update(status='failed', error=str(error))
        write_json(output/'manifest.json', state)
        raise
