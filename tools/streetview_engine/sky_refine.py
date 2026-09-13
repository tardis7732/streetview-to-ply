"""Optional shared-sky appearance candidate with an immutable foreground PLY.

Only refine_sky()/run() launch CUDA work. Sky geometry is a finite angular
approximation; only its shared SH0 and opacity are optimized. This module never
promotes a candidate, changes a baseline, runs MCMC, or fits heldout RGB.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np

from . import training
from .export import sha256, validate_ply, write_json, write_model
from .imaging import inside
from .quality import paired_new_holes
from .sky_environment import SkyEnvironmentConfig, build_sky_environment
from .processing_options import read_processing_options


@dataclass(frozen=True)
class SkyRefineConfig:
    steps: int = 400
    resolution: int = 768
    seed: int = 42
    cpu_workers: int = 4
    log_every: int = 50
    color_lr: float = .01
    opacity_lr: float = .03
    coverage_weight: float = .02
    alpha_threshold: float = .5
    min_evaluation_stations: int = 2
    max_pooled_psnr_loss_db: float = .1
    max_single_view_psnr_loss_db: float = .5
    minimum_sky_psnr_gain_db: float = .05
    minimum_sky_coverage_gain: float = .001
    already_complete_sky_coverage: float = .995
    require_down_evaluation: bool = True

    def __post_init__(self):
        for key, lo, hi in [('steps', 1, 100000), ('resolution', 16, 8192), ('seed', 0, 2**32-1),
                            ('cpu_workers', 1, 128), ('log_every', 1, 100000), ('min_evaluation_stations', 2, 10000)]:
            value = getattr(self, key)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError('Invalid integer sky-refine setting: ' + key)
        for key in ['color_lr', 'opacity_lr', 'coverage_weight', 'alpha_threshold', 'max_pooled_psnr_loss_db',
                    'max_single_view_psnr_loss_db', 'minimum_sky_psnr_gain_db', 'minimum_sky_coverage_gain', 'already_complete_sky_coverage']:
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError('Invalid sky-refine setting: ' + key)
        if min(self.color_lr, self.opacity_lr, self.minimum_sky_psnr_gain_db, self.minimum_sky_coverage_gain) <= 0:
            raise ValueError('Learning rates and required sky improvement must be positive')
        if not 0 < self.alpha_threshold < 1 or not 0 < self.already_complete_sky_coverage <= 1 or self.minimum_sky_coverage_gain > 1 or type(self.require_down_evaluation) is not bool:
            raise ValueError('Invalid sky coverage or Down policy')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def _layout(path):
    info = validate_ply(path)
    lines = []
    with Path(path).open('rb') as stream:
        while True:
            line = stream.readline(); lines.append(line)
            if line.strip() == b'end_header':
                return info, lines, stream.tell()


def _hash_payload(path, offset, size=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        stream.seek(offset)
        while size is None or size > 0:
            data = stream.read(1024*1024 if size is None else min(size, 1024*1024))
            if not data:
                if size: raise ValueError('Truncated Gaussian payload')
                break
            digest.update(data)
            if size is not None: size -= len(data)
    return digest.hexdigest()


def read_gaussian_model(path):
    """Decode standard fields without changing quaternion or attribute bytes."""
    info, _, offset = _layout(path)
    if info['sh_degree'] > 3:
        raise ValueError('Sky refinement supports Gaussian SH degrees 0..3')
    fields = info['fields']; count = info['vertex_count']
    rows = np.memmap(path, mode='r', dtype='<f4', offset=offset, shape=(count, len(fields)))
    try:
        select = lambda names: np.asarray(rows[:, [fields.index(name) for name in names]]).copy()
        rest = (info['sh_degree']+1)**2-1
        result = dict(means=select(['x','y','z']), scales=select([f'scale_{i}' for i in range(3)]),
            quats=select([f'rot_{i}' for i in range(4)]), opacities=select(['opacity'])[:, 0],
            sh0=select([f'f_dc_{i}' for i in range(3)])[:, None, :],
            shN=select([f'f_rest_{i}' for i in range(rest*3)]).reshape(count, 3, rest).transpose(0, 2, 1))
    finally:
        rows._mmap.close()
    return result, info


def append_frozen_foreground(source_ply, sky_ply, destination):
    """Copy ALL original foreground payload bytes, then append mapped sky rows."""
    source, sky, destination = map(Path, (source_ply, sky_ply, destination))
    if destination.exists() or destination.resolve() in (source.resolve(), sky.resolve()):
        raise FileExistsError('Combined PLY must have a new destination')
    fg, header, fg_offset = _layout(source)
    background, _, sky_offset = _layout(sky)
    if set(fg['fields']) != set(background['fields']) or fg['sh_degree'] != background['sh_degree']:
        raise ValueError('Foreground and sky PLY field schemas must agree')
    count = fg['vertex_count'] + background['vertex_count']
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name+'.tmp')
    if temporary.exists(): raise FileExistsError('Combined PLY temporary output already exists')
    with temporary.open('xb') as output:
        for line in header:
            output.write(f'element vertex {count}\n'.encode() if line.startswith(b'element vertex ') else line)
        with source.open('rb') as stream:
            stream.seek(fg_offset); shutil.copyfileobj(stream, output, 1024*1024)
        rows = np.memmap(sky, mode='r', dtype='<f4', offset=sky_offset, shape=(background['vertex_count'], len(background['fields'])))
        try:
            order = [background['fields'].index(name) for name in fg['fields']]
            for start in range(0, len(rows), 65536):
                np.asarray(rows[start:start+65536, order], dtype='<f4').tofile(output)
        finally:
            rows._mmap.close()
    combined, _, output_offset = _layout(temporary)
    payload_size = fg['vertex_count']*len(fg['fields'])*4
    source_payload = _hash_payload(source, fg_offset)
    copied_payload = _hash_payload(temporary, output_offset, payload_size)
    if sha256(source) != fg['sha256'] or sha256(sky) != background['sha256'] or source_payload != copied_payload:
        raise ValueError('Foreground/sky changed during exact combined export')
    temporary.replace(destination)
    return dict(source_sha256=fg['sha256'], candidate_sha256=combined['sha256'],
        source_foreground_payload_sha256=source_payload, candidate_foreground_payload_sha256=copied_payload,
        foreground_rows=fg['vertex_count'], sky_rows=background['vertex_count'],
        identity='exact_original_payload_prefix_including_all_attributes', unchanged=True)


def write_sky_appearance(initial_ply, destination, sh0, opacity_logits):
    """Replace only four float fields; preserve every sky geometry byte."""
    initial_ply, destination = Path(initial_ply), Path(destination)
    info, header, offset = _layout(initial_ply)
    count = info['vertex_count']; fields = info['fields']
    colors = np.asarray(sh0)
    if colors.shape == (count, 1, 3): colors = colors[:, 0]
    opacity_logits = np.asarray(opacity_logits)
    if colors.shape != (count, 3) or opacity_logits.shape != (count,) or not np.isfinite(colors).all() or not np.isfinite(opacity_logits).all():
        raise ValueError('Invalid fitted sky appearance arrays')
    rows = np.memmap(initial_ply, mode='r', dtype='<f4', offset=offset, shape=(count, len(fields)))
    try: result = np.array(rows, copy=True)
    finally: rows._mmap.close()
    result[:, [fields.index(f'f_dc_{i}') for i in range(3)]] = colors
    result[:, fields.index('opacity')] = opacity_logits
    if not np.isfinite(result).all(): raise ValueError('Fitted sky appearance overflows float32')
    with destination.open('xb') as output:
        output.writelines(header); result.tofile(output)
    if sha256(initial_ply) != info['sha256']: raise ValueError('Initial sky changed during appearance serialization')
    return validate_ply(destination)


def _metric_valid(metric, count):
    return (isinstance(metric, dict) and type(metric.get('static_pixels')) is int and metric['static_pixels'] == count
            and type(metric.get('covered_pixels')) is int and 0 <= metric['covered_pixels'] <= count
            and type(metric.get('static_sse')) in (int, float) and math.isfinite(metric['static_sse']) and metric['static_sse'] >= 0)


def compare_sky_candidate(before, after, *, expected_manifest, new_holes, foreground_identity,
                          source_sha256, candidate_sha256, config=None):
    """No depth-improvement claim: exact foreground identity plus sky benefit."""
    options = config or SkyRefineConfig()
    reasons = []; guards = {}
    expected = expected_manifest.get('evaluation_frames', []) if isinstance(expected_manifest, dict) else []
    names = [x.get('frame') for x in expected]
    discovery = set(expected_manifest.get('discovery_station_ids', [])) if isinstance(expected_manifest, dict) else set()
    if not expected or any(not isinstance(x, str) or not x for x in names) or len(set(names)) != len(names) or not discovery:
        return dict(accepted=False, reasons=['invalid_trusted_evaluation_roster'])
    if not foreground_identity or foreground_identity.get('unchanged') is not True or foreground_identity.get('source_sha256') != source_sha256 or foreground_identity.get('candidate_sha256') != candidate_sha256 or not foreground_identity.get('source_foreground_payload_sha256') or foreground_identity.get('source_foreground_payload_sha256') != foreground_identity.get('candidate_foreground_payload_sha256'):
        reasons.append('foreground_not_proven_byte_identical')
    indexed = []
    for report, model_sha in [(before, source_sha256), (after, candidate_sha256)]:
        rows = report.get('views', [])
        if report.get('status') != 'measured' or report.get('model_sha256') != model_sha:
            reasons.append('render_artifact_binding_mismatch')
        if len(rows) != len(expected) or len({r.get('frame') for r in rows}) != len(rows) or {r.get('frame') for r in rows} != set(names):
            reasons.append('incomplete_or_duplicate_evaluation')
        lookup = {r.get('frame'): r for r in rows}; indexed.append(lookup)
        for trusted in expected:
            row = lookup.get(trusted['frame'], {})
            if any(row.get(key) != trusted.get(key) or not trusted.get(key) for key in ['station_id','reference_rgb_sha256','reference_mask_sha256','reference_sky_mask_sha256','reference_foreground_mask_sha256']):
                reasons.append('reference_or_station_binding_mismatch')
            if row.get('station_id') in discovery:
                reasons.append('physical_station_holdout_leakage')
    group_count = len({row['station_id'] for row in expected})
    if group_count < options.min_evaluation_stations:
        reasons.append('insufficient_heldout_stations')
    for category in ('all', 'foreground', 'down', 'sky'):
        source_sse = candidate_sse = pixels = source_covered = candidate_covered = 0
        worst_loss = -math.inf; groups = set(); category_invalid = False
        for trusted in expected:
            count = trusted.get('category_pixels', {}).get(category)
            if type(count) is not int or count < 0:
                reasons.append('invalid_trusted_category_counts'); category_invalid = True; continue
            left, right = (lookup.get(trusted['frame'], {}) for lookup in indexed)
            left, right = (left, right) if category == 'all' else (left.get(category, {}), right.get(category, {}))
            if count == 0:
                if left and not _metric_valid(left, 0) or right and not _metric_valid(right, 0):
                    reasons.append('changed_empty_category'); category_invalid = True
                continue
            holes = new_holes.get(trusted['frame'], {}).get(category)
            if not _metric_valid(left, count) or not _metric_valid(right, count) or left.get('alpha_threshold') != options.alpha_threshold or right.get('alpha_threshold') != options.alpha_threshold or type(holes) is not int or not 0 <= holes <= count:
                reasons.append('missing_or_incomparable_'+category+'_metrics'); category_invalid = True; continue
            if holes != 0:
                reasons.append('new_'+category+'_holes')
            source_sse += left['static_sse']; candidate_sse += right['static_sse']; pixels += count
            source_covered += left['covered_pixels']; candidate_covered += right['covered_pixels']
            groups.add(trusted['station_id'])
            loss = 10*math.log10(max(right['static_sse'], count*1e-12)/max(left['static_sse'], count*1e-12))
            worst_loss = max(worst_loss, loss)
        pooled_loss = 10*math.log10(max(candidate_sse, pixels*1e-12)/max(source_sse, pixels*1e-12)) if pixels else None
        required = category != 'down' or options.require_down_evaluation
        if required and (category_invalid or not pixels or len(groups) < options.min_evaluation_stations):
            reasons.append('insufficient_'+category+'_evaluation')
        if pixels and (pooled_loss > options.max_pooled_psnr_loss_db or worst_loss > options.max_single_view_psnr_loss_db):
            reasons.append(category+'_rgb_regression')
        guards[category] = dict(pixels=pixels, physical_stations=len(groups), pooled_psnr_loss_db=pooled_loss,
            worst_view_psnr_loss_db=worst_loss if pixels else None,
            before_coverage=source_covered/pixels if pixels else None, after_coverage=candidate_covered/pixels if pixels else None)
    sky = guards['sky']
    gain = -sky['pooled_psnr_loss_db'] if sky['pixels'] else None
    coverage_gain = sky['after_coverage']-sky['before_coverage'] if sky['pixels'] else None
    if gain is None or gain < options.minimum_sky_psnr_gain_db:
        reasons.append('no_measured_sky_rgb_improvement')
    coverage_ok = sky['pixels'] and (coverage_gain >= options.minimum_sky_coverage_gain or
        sky['before_coverage'] >= options.already_complete_sky_coverage and coverage_gain >= 0)
    if not coverage_ok:
        reasons.append('no_measured_sky_coverage_benefit')
    return dict(accepted=not reasons, reasons=sorted(set(reasons)), categories=guards,
        sky_psnr_gain_db=gain, sky_coverage_gain=coverage_gain, policy=asdict(options),
        scope='Complete known-pose heldout appearance/coverage; exact foreground payload preserved; no geometric or cross-location improvement claim')


def _source_inputs(training_root, dataset):
    manifest_path = training_root/'manifest.json'; manifest = _read(manifest_path)
    source = training_root/'model.ply'; arrays, artifact = read_gaussian_model(source)
    if manifest.get('status') != 'completed' or manifest.get('artifact', {}).get('sha256') != artifact['sha256']:
        raise ValueError('Sky refinement requires a completed hash-bound foreground model')
    dataset_manifest = _read(dataset/'dataset_manifest.json')
    train = _read(dataset/'transforms_train.json')['frames']; heldout = _read(dataset/'transforms_heldout.json')['frames']
    train_groups, heldout_groups = training.validate_splits(train, heldout)
    training.validate_dataset_manifest(dataset_manifest, train_groups, heldout_groups)
    if not heldout:
        raise ValueError('Sky refinement requires actual fixed heldout frames')
    expected_inputs = manifest.get('provenance', {}).get('inputs', {})
    sources = {str(manifest_path): sha256(manifest_path), str(source): artifact['sha256']}
    for name in ['dataset_manifest.json', 'transforms_train.json', 'transforms_heldout.json']:
        path = inside(dataset, name); actual = sha256(path)
        if expected_inputs.get(name) != actual:
            raise ValueError('Foreground training is bound to another dataset: '+name)
        sources[str(path)] = actual
    for frame in train + heldout:
        for key in ['file_path', 'mask_path', 'foreground_mask_path', 'sky_mask_path']:
            path = inside(dataset, frame[key]); actual = sha256(path)
            if expected_inputs.get('photos_and_masks', {}).get(frame[key]) != actual:
                raise ValueError('Frozen training photo/mask binding differs: '+frame[key])
            sources[str(path)] = actual
    return source, arrays, artifact, manifest, train, heldout, train_groups, heldout_groups, sources


def _expanded_sky(dataset, train, heldout, foreground, sky_config):
    groups = sorted({f['station_id'] for f in train})
    by_group = [np.unique(np.asarray([np.asarray(f['transform_matrix'])[:3, 3] for f in train if f['station_id'] == group]), axis=0) for group in groups]
    origin = np.asarray([centers.mean(axis=0) for centers in by_group]).mean(axis=0)
    train_radius = float(np.linalg.norm(np.concatenate(by_group)-origin, axis=1).max())
    all_centers = np.asarray([np.asarray(f['transform_matrix'])[:3, 3] for f in train+heldout])
    all_radius = float(np.linalg.norm(all_centers-origin, axis=1).max())
    if train_radius <= 0 or all_radius <= 0:
        raise ValueError('Sky needs a nondegenerate camera viewing region')
    # Only camera POSES from the heldout roster determine this radius. No
    # heldout photograph, mask, color or loss is passed to the builder.
    desired_radius = all_radius/math.sin(math.radians(sky_config.maximum_parallax_degrees))
    tighter_angle = math.degrees(math.asin(min(1., train_radius/desired_radius)))
    sigma = np.exp(foreground['scales'].astype(np.float64)).max(axis=1)
    foreground_extent = float((np.linalg.norm(foreground['means'].astype(np.float64)-origin, axis=1)+3*sigma).max())
    if not math.isfinite(foreground_extent):
        raise ValueError('Foreground 3-sigma extent is not finite')
    builder = replace(sky_config, maximum_parallax_degrees=tighter_angle,
                      far_plane_m=min(sky_config.far_plane_m or 1e7, 1e7))
    environment = build_sky_environment(dataset, train, config=builder, foreground_extent_m=foreground_extent)
    report = environment.provenance
    report['expanded_view_region'] = dict(center=origin.tolist(), radius_m=all_radius,
        camera_pose_frames=[f['file_path'] for f in train+heldout], heldout_appearance_used=False,
        maximum_parallax_degrees_bound=math.degrees(math.asin(all_radius/report['shell_radius_m'])))
    actual_far = report['required_renderer_far_plane_m'] + all_radius-train_radius
    if actual_far >= min(sky_config.far_plane_m or 1e7, 1e7):
        raise ValueError('Expanded heldout viewing region exceeds sky far-plane allowance')
    report['expanded_required_far_plane_m'] = actual_far
    return environment, report


def _tensor_params(arrays, center, radius, torch, device):
    result = {}
    for key, value in arrays.items():
        if key == 'means': value = (value.astype(np.float64)-center)/radius
        elif key == 'scales': value = value.astype(np.float64)-math.log(radius)
        result[key] = torch.tensor(value, dtype=torch.float32, device=device)
    return result


def _combined(foreground, sky, torch):
    return {key: torch.cat([foreground[key], sky[key]], dim=0) for key in foreground}


def _expected(frames, train_groups):
    rows = []
    for frame in frames:
        counts = dict(all=int(frame['mask'].sum()), foreground=int(frame['foreground'].sum()), sky=int(frame['sky'].sum()),
            down=int(frame['mask'].sum()) if str(frame['meta'].get('face','')).lower() in ('d','down') else 0)
        rows.append(dict(frame=frame['frame'], station_id=frame['station_id'],
            reference_rgb_sha256=frame['reference_rgb_sha256'], reference_mask_sha256=frame['reference_mask_sha256'],
            reference_sky_mask_sha256=frame['meta']['sky_mask_sha256'],
            reference_foreground_mask_sha256=frame['meta']['foreground_mask_sha256'], category_pixels=counts))
    return dict(discovery_station_ids=sorted(train_groups), evaluation_frames=rows)


def _new_holes(source_dir, candidate_dir, before, after, frames, threshold):
    whole = paired_new_holes(source_dir, candidate_dir, before, after, threshold=threshold)
    result = {}; old = {r['frame']:r for r in before['views']}; new = {r['frame']:r for r in after['views']}
    for frame in frames:
        name = frame['frame']
        with np.load(inside(source_dir, old[name]['alpha_npz']), allow_pickle=False) as data: a = data['alpha'].copy()
        with np.load(inside(candidate_dir, new[name]['alpha_npz']), allow_pickle=False) as data: b = data['alpha'].copy()
        lost = (a >= threshold) & (b < threshold)
        result[name] = dict(all=whole[name], foreground=int((lost & frame['foreground']).sum()), sky=int((lost & frame['sky']).sum()),
            down=whole[name] if str(frame['meta'].get('face','')).lower() in ('d','down') else 0)
    return result


def refine_sky(training_root, dataset_root, output_dir, *, config=None, sky_config=None):
    """Run explicit CUDA post-fit, retain all results, return acceptance only."""
    # This callable is also used directly outside the job dispatcher. Dataset
    # policy must prevent it from rebuilding deliberately excluded sky.
    dataset_policy = _read(Path(dataset_root) / 'dataset_manifest.json')
    if read_processing_options(dataset_policy)['remove_sky']:
        return dict(status='disabled',accepted=False,promoted=False,reason='remove_sky_requested')
    options = config or SkyRefineConfig(); builder_options = sky_config or SkyEnvironmentConfig()
    if not isinstance(options, SkyRefineConfig) or not isinstance(builder_options, SkyEnvironmentConfig):
        raise TypeError('Expected typed sky settings')
    training_root, dataset, output = map(lambda p: Path(p).resolve(), (training_root, dataset_root, output_dir))
    if output.is_relative_to(training_root) or output.is_relative_to(dataset) or output.exists():
        raise ValueError('Sky candidate requires a fresh separate output directory')
    source, foreground, artifact, source_manifest, train_raw, heldout_raw, train_groups, heldout_groups, sources = _source_inputs(training_root, dataset)
    builder_options = replace(builder_options, sh_degree=artifact['sh_degree'])
    environment, sky_report = _expanded_sky(dataset, train_raw, heldout_raw, foreground, builder_options)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'sky_builder.json', sky_report)
    signature = dict(sources=sources, config=asdict(options), sky_config=asdict(builder_options),
        code={name:sha256(Path(__file__).with_name(name)) for name in ['sky_refine.py','sky_environment.py','training.py','quality.py','export.py']})
    write_json(output/'inputs.json', signature)
    if not len(environment.means):
        report = dict(status='insufficient_evidence', accepted=False, reasons=['no_supported_shared_sky_cells'], source_sha256=artifact['sha256'], sky_builder='sky_builder.json')
        write_json(output/'manifest.json', report)
        return report
    # Imports happen only after immutable input checks and actual sky evidence.
    import torch
    import gsplat
    from gsplat import rasterization
    if not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA Torch and gsplat are required; no CPU renderer fallback')
    torch.cuda.set_device(0); torch.set_num_threads(options.cpu_workers); torch.manual_seed(options.seed)
    center, radius = training.normalization(train_raw)
    render_settings = training.TrainingSettings(resolution=options.resolution, sh_degree=artifact['sh_degree'], alpha_threshold=options.alpha_threshold)
    status = dict(status='running', accepted=False, source_sha256=artifact['sha256'],
        source_training_manifest_sha256=sources[str(training_root/'manifest.json')], config=asdict(options),
        runtime=dict(torch=torch.__version__,gsplat=gsplat.__version__,gpu=torch.cuda.get_device_name()),
        learning='Only shared sky SH0 and opacity; foreground and sky geometry frozen; TRAIN sky cores; full combined alpha*T occlusion',
        quality_scope='Known-pose heldout sky appearance/coverage only; no foreground geometry improvement claimed')
    write_json(output/'manifest.json', status); started = time.monotonic()
    try:
        initial_path = output/'sky_initial.ply'
        write_model(initial_path, means=environment.means,log_scales=environment.log_scales,quats=environment.quats,
                    opacity_logits=environment.opacity_logits,sh0=environment.sh0,shN=environment.shN)
        sky_arrays, _ = read_gaussian_model(initial_path)
        fg = _tensor_params(foreground,center,radius,torch,'cuda')
        sky = _tensor_params(sky_arrays,center,radius,torch,'cuda')
        sky['sh0'] = torch.nn.Parameter(sky['sh0']); sky['opacities'] = torch.nn.Parameter(sky['opacities'])
        optimizer = torch.optim.Adam([dict(params=[sky['sh0']],lr=options.color_lr), dict(params=[sky['opacities']],lr=options.opacity_lr)])
        fixed = {key:value.detach().clone() for key,value in fg.items()}
        fixed_sky = {key:value.detach().clone() for key,value in sky.items() if key not in ('sh0','opacities')}
        train = [training.prepare_frame(f,dataset,options.resolution,center,radius) for f in train_raw]
        eligible = [f for f in train if (f['sky'] & ~f['foreground']).any()]
        if len({f['station_id'] for f in eligible}) < builder_options.minimum_physical_stations:
            raise ValueError('Fewer than required TRAIN sky groups remain at fit resolution')
        for frame in eligible:
            frame['view_gpu']=torch.tensor(frame['view'],device='cuda');frame['K_gpu']=torch.tensor(frame['K'],device='cuda')
        schedule = training.station_uniform_schedule(eligible,options.steps,options.seed)
        write_json(output/'training_schedule.json', dict(seed=options.seed, frames=[dict(frame=f['frame'],station_id=f['station_id']) for f in eligible], indices=schedule.tolist(), heldout_used=False))
        np.savez_compressed(output/'sky_ancestry.npz',grid_indices=environment.grid_indices,support_counts=environment.support_counts,
            observation_row_indices=environment.observation_row_indices,observation_frame_indices=environment.observation_frame_indices,
            observation_pixels_xy=environment.observation_pixels_xy)
        nonzero_gradient = False
        with (output/'metrics_history.jsonl').open('x',encoding='utf8') as history:
            for step, index in enumerate(schedule,1):
                frame=eligible[int(index)]
                mask=torch.tensor(frame['sky'] & ~frame['foreground'],device='cuda')
                target=torch.tensor(frame['rgb'],dtype=torch.float32,device='cuda')/255
                optimizer.zero_grad(set_to_none=True)
                color,alpha,_=training._render(_combined(fg,sky,torch),frame,artifact['sh_degree'],rasterization,radius)
                rgb_loss=(color[0]-target).abs()[mask].mean()
                coverage=(1-alpha[0,...,0][mask]).mean()
                loss=rgb_loss+options.coverage_weight*coverage
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite sky loss')
                loss.backward()
                norms={key:float(sky[key].grad.norm()) if sky[key].grad is not None else 0. for key in ['sh0','opacities']}
                if not all(math.isfinite(x) for x in norms.values()):raise FloatingPointError('Nonfinite sky appearance gradient')
                nonzero_gradient |= any(x>0 for x in norms.values())
                optimizer.step()
                with torch.no_grad():
                    sky['sh0'].clamp_(-.5/.28209479177387814,.5/.28209479177387814)
                    sky['opacities'].clamp_(-12.,12.)
                if step==1 or step%options.log_every==0 or step==options.steps:
                    row=dict(stage='sky_refine',step=step,steps=options.steps,frame=frame['frame'],station_id=frame['station_id'],
                        loss=float(loss.detach()),rgb_l1=float(rgb_loss.detach()),coverage_loss=float(coverage.detach()),gradients=norms,elapsed_s=time.monotonic()-started)
                    history.write(json.dumps(row,allow_nan=False)+'\n');history.flush();print(json.dumps(row),flush=True)
        if not nonzero_gradient:raise RuntimeError('No nonzero shared sky appearance gradient was observed')
        if any(value.requires_grad or not torch.equal(value,fixed[key]) for key,value in fg.items()) or any(not torch.equal(sky[key],value) for key,value in fixed_sky.items()):
            raise RuntimeError('Frozen foreground or sky geometry changed during appearance fit')
        fitted_path=output/'sky_fitted.ply'
        write_sky_appearance(initial_path,fitted_path,sky['sh0'].detach().cpu().numpy(),sky['opacities'].detach().cpu().numpy())
        fitted_arrays,_=read_gaussian_model(fitted_path)
        if any(not np.array_equal(fitted_arrays[key],sky_arrays[key]) for key in ['means','scales','quats','shN']):
            raise RuntimeError('Serialized sky geometry changed')
        combined_path=output/'sky_combined.ply'
        identity=append_frozen_foreground(source,fitted_path,combined_path)
        combined_arrays,combined_artifact=read_gaussian_model(combined_path)
        combined_params=_tensor_params(combined_arrays,center,radius,torch,'cuda')
        # Evaluate reloaded PLY values, not unquantized training tensors. Both
        # sides use exactly the same implementation, poses, masks and roster.
        heldout=[training.prepare_frame(f,dataset,options.resolution,center,radius) for f in heldout_raw]
        for frame in heldout:
            frame['view_gpu']=torch.tensor(frame['view'],device='cuda');frame['K_gpu']=torch.tensor(frame['K'],device='cuda')
        expected=_expected(heldout,train_groups);write_json(output/'evaluation_manifest.json',expected)
        source_dir=output/'source_reference';candidate_dir=output/'candidate';source_dir.mkdir();candidate_dir.mkdir()
        before=training._evaluate(fg,heldout,source_dir,rasterization,torch,render_settings,radius,artifact['sha256'])
        after=training._evaluate(combined_params,heldout,candidate_dir,rasterization,torch,render_settings,radius,combined_artifact['sha256'])
        category_bindings={row['frame']:row for row in expected['evaluation_frames']}
        for report in [before,after]:
            for row in report['views']:
                for key in ['reference_sky_mask_sha256','reference_foreground_mask_sha256']:
                    row[key]=category_bindings[row['frame']][key]
        write_json(source_dir/'heldout_metrics.json',before);write_json(candidate_dir/'heldout_metrics.json',after)
        holes=_new_holes(source_dir,candidate_dir,before,after,heldout,options.alpha_threshold)
        comparison=compare_sky_candidate(before,after,expected_manifest=expected,new_holes=holes,foreground_identity=identity,
            source_sha256=artifact['sha256'],candidate_sha256=combined_artifact['sha256'],config=options)
        write_json(output/'comparison.json',dict(**comparison,new_holes=holes,foreground_identity=identity))
        if any(sha256(path)!=expected_hash for path,expected_hash in sources.items()):
            raise ValueError('Frozen baseline or dataset changed during sky refinement')
        status.update(status='completed',accepted=comparison['accepted'],completed_steps=options.steps,elapsed_s=time.monotonic()-started,
            candidate_ply='sky_combined.ply',candidate_sha256=combined_artifact['sha256'],sky_ply='sky_fitted.ply',
            source_reference_metrics='source_reference/heldout_metrics.json',candidate_metrics='candidate/heldout_metrics.json',
            comparison='comparison.json',comparison_sha256=sha256(output/'comparison.json'),foreground_identity=identity,
            nonzero_sky_gradient=nonzero_gradient,promoted=False,baseline_unchanged=True,
            reason='Passed complete sky appearance guard' if comparison['accepted'] else comparison['reasons'])
        write_json(output/'manifest.json',status)
        return status
    except BaseException as error:
        status.update(status='failed',accepted=False,error_type=type(error).__name__,error=str(error),elapsed_s=time.monotonic()-started)
        write_json(output/'manifest.json',status)
        raise


def run(config, job_dir, settings):
    """Optional explicit stage; never silently enable or promote a sky model."""
    if read_processing_options(config)['remove_sky']:
        return dict(status='disabled',accepted=False,promoted=False,reason='remove_sky_requested')
    options=settings.get('sky_refine',{})
    if not options or options.get('enabled') is False:
        return dict(status='disabled',accepted=False,promoted=False)
    if options.get('enabled') is not True or set(options)-{'enabled','fit','builder'}:
        raise ValueError('Sky refinement requires enabled=true and fit/builder settings')
    root=Path(job_dir).resolve()
    return refine_sky(root/'training',root/'sfm/dataset',root/'sky_candidate',
        config=SkyRefineConfig(**options.get('fit',{})),sky_config=SkyEnvironmentConfig(**options.get('builder',{})))
