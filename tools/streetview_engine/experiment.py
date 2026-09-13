"""Optional matched scene optimization arms and conservative result selection."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

from . import training
from .export import sha256, validate_ply, write_json
from .quality import compare_candidate, paired_new_holes


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def copy_immutable_dataset(source, destination):
    """Share immutable inputs within one host, never training outputs."""
    source = Path(source).resolve()
    if destination.exists():
        raise FileExistsError('Experiment dataset already exists')
    if any(p.is_symlink() for p in source.rglob('*')):
        raise ValueError('Experiment input dataset contains a symlink')
    def copy_file(a, b):
        try:
            os.link(a, b)
            return b
        except OSError:
            return shutil.copy2(a, b)
    shutil.copytree(source, destination, copy_function=copy_file)


def compare_arms(baseline_root, candidate_root):
    baseline_root, candidate_root = Path(baseline_root), Path(candidate_root)
    before = _read(baseline_root / 'heldout_metrics.json')
    after = _read(candidate_root / 'heldout_metrics.json')
    for root, report in ((baseline_root, before), (candidate_root, after)):
        if validate_ply(root / 'model.ply')['sha256'] != report['model_sha256']:
            raise ValueError('Evaluation is bound to another model')
    expected = _read(baseline_root / 'evaluation_manifest.json')
    if _read(candidate_root / 'evaluation_manifest.json') != expected:
        raise ValueError('Candidate evaluation split or references differ')
    bm = _read(baseline_root / 'manifest.json')
    cm = _read(candidate_root / 'manifest.json')
    for root, manifest in ((baseline_root, bm), (candidate_root, cm)):
        if manifest.get('status') != 'completed' or manifest.get('artifact', {}).get('sha256') != sha256(root / 'model.ply'):
            raise ValueError('Quality arm training is incomplete or bound to another artifact')
    for key in ('code_sha256', 'quality_code_sha256', 'export_code_sha256'):
        if not bm['provenance'].get(key) or bm['provenance'][key] != cm['provenance'].get(key):
            raise ValueError('Quality arms use different implementation code')
    bs, cs = bm['provenance']['settings'], cm['provenance']['settings']
    varying = {'depth_moment_weight', 'depth_coverage_weight'}
    if {k:v for k,v in bs.items() if k not in varying} != {k:v for k,v in cs.items() if k not in varying}:
        raise ValueError('Matched quality arms differ in unrelated training settings')
    if bm['provenance']['inputs'] != cm['provenance']['inputs']:
        raise ValueError('Quality arms use different scene input bytes')
    holes = paired_new_holes(baseline_root, candidate_root, before, after, threshold=bs['alpha_threshold'])
    return compare_candidate(before, after, expected_manifest=expected, new_hole_pixels=holes,
        depth_baseline=_read(baseline_root / 'heldout_depth_metrics.json'),
        depth_candidate=_read(candidate_root / 'heldout_depth_metrics.json'))


def prepare_depth_prior(config, root, settings):
    """Infer once when explicitly configured; a failed optional prior is evidence.

    A baseline can still be produced on failure. Positive learned-depth arms
    must pass the trainer's accepted-pixel checks and otherwise are rejected.
    """
    if not settings.get('training', {}).get('depth_use_validated_prior', False):
        return {}
    from . import depth_prior
    options = settings.get('depth_prior')
    if not isinstance(options, dict) or not options:
        raise ValueError('Validated learned supervision requires depth_prior model settings')
    if options.get('output_dir', 'depth_prior') != 'depth_prior':
        raise ValueError('Integrated learned supervision uses the portable depth_prior directory')
    prior = root / 'depth_prior'
    if not (prior / 'manifest.json').is_file() and not prior.exists():
        print(json.dumps(dict(event='validated_depth_prior_start')), flush=True)
        try:
            depth_prior.run(config, root, settings)
        except Exception as error:
            write_json(root / 'depth_prior_attempt.json', dict(status='failed',
                error_type=type(error).__name__, error=str(error),
                consequence='No learned supervision is authorized; baseline remains available'))
    paths = [root / 'depth_prior_attempt.json']
    if (prior / 'manifest.json').is_file():
        validate_prior_cache(config, root, settings)
        paths += prior_files(prior)
    return {p.relative_to(root).as_posix():sha256(p) for p in paths if p.is_file()}


def validate_prior_cache(config, root, settings):
    """Verify requested science settings/models and portable source bytes.

    Installation directories and operation-only resume flags may differ after a
    host/job copy. Model content hashes and every calibration setting must match.
    Reading cached weights/source hashes does not construct a model or use CUDA.
    """
    from . import depth_prior
    options = depth_prior.DepthPriorSettings(**settings['depth_prior'])
    prior = root / 'depth_prior'
    manifest = _read(prior / 'manifest.json')
    if manifest.get('status') != 'complete' or manifest.get('validation_status') not in {'multistation_consistent_subset', 'insufficient_supported_depth'}:
        raise ValueError('Cached learned prior is not a completed assessed result')
    artifacts = [('input_manifest.json', 'input_manifest_sha256'), ('calibration_report.json', 'calibration_report_sha256'), ('raw/provenance.json', 'provenance_sha256')]
    if 'observation_deduplication_sha256' in manifest:
        artifacts.append(('observation_deduplication.json', 'observation_deduplication_sha256'))
    # prior_files confines paths before any of their content is used.
    prior_files(prior)
    for name, key in artifacts:
        if sha256(prior / name) != manifest.get(key):
            raise ValueError('Cached learned prior provenance changed: ' + name)
    original = _read(prior / 'input_manifest.json')
    expected_config = hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if original.get('job_config_sha256') != expected_config:
        raise ValueError('Cached learned prior belongs to another frozen config')
    operation_paths = {'output_dir', 'resume_raw', 'repo_path', 'checkpoint_path', 'unik3d_snapshot_path', 'extra_python_path'}
    def scientific_options(value):
        normalized = asdict(depth_prior.DepthPriorSettings(**value))
        return {key:item for key,item in normalized.items() if key not in operation_paths}
    expected = scientific_options(settings['depth_prior'])
    if scientific_options(original.get('settings', {})) != expected or scientific_options(manifest.get('settings', {})) != expected:
        raise ValueError('Cached learned prior model/calibration settings differ from requested settings')
    # A path change can be harmless, a checkpoint/config/source-content change
    # cannot. model_assets verifies each explicitly trusted checkpoint SHA.
    model_paths = {'repo_path', 'checkpoint_path', 'unik3d_snapshot_path'}
    actual_models = {key:value for key,value in depth_prior.model_assets(options).items() if key not in model_paths}
    recorded_models = {key:value for key,value in original.get('models', {}).items() if key not in model_paths}
    if actual_models != recorded_models:
        raise ValueError('Cached learned prior model/source content changed')
    dataset = (root / 'sfm/dataset').resolve()
    sources = manifest.get('source_dataset_files_sha256')
    if not isinstance(sources, dict) or not sources:
        raise ValueError('Cached learned prior lacks portable dataset source bindings')
    for name, expected_sha in sources.items():
        path = dataset / name
        if Path(name).is_absolute() or '..' in Path(name).parts or not path.resolve().is_relative_to(dataset) or not path.is_file() or sha256(path) != expected_sha:
            raise ValueError('Cached learned prior source dataset changed: ' + name)
    for entry in manifest['entries']:
        if entry['valid_count'] > 0 and sha256(prior / entry['npz']) != entry.get('sha256'):
            raise ValueError('Cached accepted learned map changed')


def prior_files(prior):
    """Copy accepted map FILES and provenance; exclude raw/rejected map files.

    Accepted NPZs retain their exact hashes, including diagnostic candidate_*
    members. The trainer explicitly reads only accepted arrays, never those
    members. This is file-level inclusion, not a claim of NPZ sanitization.
    """
    manifest = _read(prior / 'manifest.json')
    names = ['manifest.json', 'input_manifest.json', 'calibration_report.json', 'raw/provenance.json']
    if 'observation_deduplication_sha256' in manifest:
        names.append('observation_deduplication.json')
    names += [entry['npz'] for entry in manifest['entries'] if entry['valid_count'] > 0]
    paths = []
    for name in names:
        path = prior / name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink() or not path.resolve().is_relative_to(prior.resolve()):
            raise ValueError('Learned prior artifact escapes its directory')
        if not path.is_file():
            raise ValueError('Learned prior artifact is missing')
        paths.append(path)
    return paths


def copy_prior(root, arm):
    prior = root / 'depth_prior'
    if not (prior / 'manifest.json').is_file():
        return
    for path in prior_files(prior):
        destination = arm / 'depth_prior' / path.relative_to(prior)
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected = sha256(path)
        if destination.is_symlink():
            raise ValueError('Learned prior copy target is a symlink')
        if destination.exists():
            if not destination.is_file() or sha256(destination) != expected:
                raise ValueError('Existing learned prior copy target differs')
            continue
        try:
            os.link(path, destination)
        except OSError:
            # Exclusive creation also rejects a differing target that appears
            # after the precheck; never silently overwrite an existing artifact.
            with path.open('rb') as source, destination.open('xb') as target:
                shutil.copyfileobj(source, target)
        if sha256(path) != expected or sha256(destination) != expected:
            raise ValueError('Learned prior changed during immutable copy')


def finalize(config, root, settings):
    # Loaded only after a foreground model is available.
    from .postprocess import finalize_sky
    return finalize_sky(config, root, settings)


def run(config, job_dir, settings):
    # Product generation starts from SfM; learned models may supply depth only.
    if any(settings.get(key) is not None for key in
           ('initialization', 'gaussian_initializer', 'spherical_initializer')):
        raise ValueError('Multi-view generation uses SfM initialization; '
                         'predicted Gaussian initialization is unavailable')
    root = Path(job_dir).resolve()
    options = settings.get('quality', {})
    if not options or options.get('enabled') is False:
        prepare_depth_prior(config, root, settings)
        # The foreground signature remains intact after sky publication, so
        # normal training reuse also verifies settings and immutable inputs.
        training.run(config, root, settings)
        return finalize(config, root, settings)
    if not isinstance(options, dict) or set(options) - {'enabled', 'depth_weights', 'coverage_weight'}:
        raise ValueError('Unknown quality experiment settings')
    if options.get('enabled') is not True:
        raise ValueError('Quality comparison requires explicit enabled=true')
    weights = options.get('depth_weights', [.01, .003])
    if not isinstance(weights, list) or not 1 <= len(weights) <= 4 or any(isinstance(w, bool) or not isinstance(w, (int,float)) or not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError('Configure one to four finite positive depth weights')
    coverage_weight = options.get('coverage_weight', .01)
    if isinstance(coverage_weight, bool) or not isinstance(coverage_weight, (int,float)) or not math.isfinite(coverage_weight) or coverage_weight <= 0:
        raise ValueError('Invalid quality coverage weight')
    prior_binding = prepare_depth_prior(config, root, settings)
    dataset = root / 'sfm/dataset'
    files = {p.relative_to(dataset).as_posix():sha256(p) for p in dataset.rglob('*') if p.is_file()}
    binding = dict(settings=settings, config=config, dataset=files, validated_depth_prior=prior_binding,
                   code={name:sha256(Path(__file__).with_name(name)) for name in ('experiment.py','training.py','quality.py','export.py','depth_prior.py','postprocess.py','sky_refine.py','sky_environment.py','processing_options.py')})
    signature = hashlib.sha256(json.dumps(binding, sort_keys=True, allow_nan=False).encode()).hexdigest()
    destination = root / 'training'
    if destination.exists():
        existing = _read(destination / 'manifest.json')
        if existing.get('experiment_signature') != signature or existing.get('status') != 'completed':
            raise ValueError('Existing experiment differs or is incomplete; use a new job')
        if validate_ply(destination / 'model.ply')['sha256'] != existing['selection']['accepted_model_sha256']:
            raise ValueError('Selected model changed')
        if sha256(destination / existing['selection']['comparison_report']) != existing['selection']['comparison_sha256']:
            raise ValueError('Selected comparison evidence changed')
        return finalize(config, root, settings)
    experiments = root / 'experiments'
    if experiments.exists():
        raise FileExistsError('Interrupted experiment is retained; use a fresh job')
    experiments.mkdir()
    write_json(experiments / 'input.json', dict(signature=signature, **binding))
    baseline_settings = deepcopy(settings)
    baseline_settings.setdefault('training', {}).update(depth_moment_weight=0., depth_coverage_weight=0.)
    baseline_root = experiments / 'baseline'
    copy_immutable_dataset(dataset, baseline_root / 'sfm/dataset')
    if prior_binding:
        copy_prior(root, baseline_root)
    print(json.dumps(dict(event='quality_arm_start', arm='baseline')), flush=True)
    baseline_manifest = training.run(config, baseline_root, baseline_settings)
    expected_model_sha = baseline_manifest['artifact']['sha256']
    chosen = baseline_root
    accepted = []
    comparisons = []
    for index, weight in enumerate(weights):
        name = f'depth_{index:02d}'
        candidate_root = experiments / name
        copy_immutable_dataset(dataset, candidate_root / 'sfm/dataset')
        if prior_binding:
            copy_prior(root, candidate_root)
        candidate_settings = deepcopy(baseline_settings)
        candidate_settings['training'].update(depth_moment_weight=float(weight), depth_coverage_weight=float(coverage_weight))
        print(json.dumps(dict(event='quality_arm_start', arm=name, depth_weight=weight)), flush=True)
        try:
            training.run(config, candidate_root, candidate_settings)
            result = compare_arms(baseline_root / 'training', candidate_root / 'training')
            record = dict(arm=name, depth_weight=weight, **result)
            if result['accepted']:
                score = result['geometry']['evidence']['after']['relative_expected_squared_error']
                accepted.append((score, name, candidate_root, result['geometry']['evidence']['candidate_sha256']))
        except Exception as error:
            record = dict(arm=name, depth_weight=weight, accepted=False, error_type=type(error).__name__, error=str(error))
        comparisons.append(record)
        write_json(experiments / 'comparisons.json', dict(baseline='baseline', candidates=comparisons))
    if accepted:
        selected = min(accepted)
        chosen, expected_model_sha = selected[2], selected[3]
    shutil.copytree(chosen / 'training', destination)
    comparison_report = dict(status='completed', experiment_signature=signature, baseline='baseline',
        selected_arm=chosen.name, candidates=comparisons,
        scope='Same-scene selection at heldout physical stations and shared SfM poses; no cross-location or independent sensor accuracy claim')
    write_json(destination / 'selection_comparison.json', comparison_report)
    manifest = _read(destination / 'manifest.json')
    artifact = validate_ply(destination / 'model.ply')
    if artifact['sha256'] != expected_model_sha or manifest.get('artifact', {}).get('sha256') != expected_model_sha or _read(destination / 'heldout_metrics.json').get('model_sha256') != expected_model_sha:
        raise ValueError('Selected model changed after comparison; publication rejected')
    manifest.update(experiment_signature=signature,
        selection=dict(accepted_model='model.ply', accepted_model_sha256=artifact['sha256'],
            selected='candidate' if accepted else 'baseline', source_arm=chosen.name,
            comparison_report='selection_comparison.json', comparison_sha256=sha256(destination / 'selection_comparison.json'),
            basis='Complete fixed-view appearance, coverage and heldout track-depth consistency comparison'))
    manifest['quality'].update(quality_improved=bool(accepted), candidate_comparison=comparison_report)
    write_json(destination / 'manifest.json', manifest)
    return finalize(config, root, settings)
