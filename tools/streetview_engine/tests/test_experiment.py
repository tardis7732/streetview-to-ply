"""Selection orchestration/integrity unit tests, not scene-quality results.

PLYs are real standard files; tiny analytic render statistics isolate selection
logic. No trainer or GPU is run in these unit tests.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from tools.streetview_engine import experiment
from tools.streetview_engine import depth_prior
from tools.streetview_engine.export import run as export_run, sha256, validate_ply, write_json, write_model
from tools.streetview_engine.quality import depth_metrics, masked_metrics
from tools.streetview_engine.training import TrainingSettings


def write_arm(directory, settings, inputs, *, candidate=False, improvement=True):
    directory.mkdir(parents=True, exist_ok=True)
    centers = np.array([[0, 0, 4.5 if candidate else 4.], [.1, 0, 5.5 if candidate else 6.]])
    artifact = write_model(directory / "model.ply", means=centers, log_scales=np.full((2, 3), -2),
        quats=np.array([[1, 0, 0, 0], [1, 0, 0, 0]]), opacity_logits=np.zeros(2),
        sh0=np.zeros((2, 1, 3)), shN=np.zeros((2, 8, 3)))
    mask, alpha = np.ones((4, 4), bool), np.ones((4, 4), np.float32)
    metric = masked_metrics(np.full((4, 4, 3), .09 if candidate and improvement else .1), np.zeros((4, 4, 3)), alpha, mask)
    depth = depth_metrics(np.array([5.]), np.array([25.5 if candidate and improvement else 27.]),
                          np.ones(1), np.array([5.]), np.ones(1, bool))
    rgb_rows, depth_rows = [], []
    for index in range(2):
        name, station = f"view{index}", f"holdout{index}"
        np.savez(directory / f"alpha{index}.npz", alpha=alpha, mask=mask)
        rgb_rows.append(dict(frame=name, station_id=station, reference_rgb_sha256=hashlib.sha256(f"rgb{index}".encode()).hexdigest(),
                            reference_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(), alpha_npz=f"alpha{index}.npz", **metric))
        depth_rows.append(dict(frame=name, station_id=station, target_roster_sha256=hashlib.sha256(f"targets{index}".encode()).hexdigest(), **depth))
    expected = dict(discovery_station_ids=["train"], evaluation_frames=[{key: row[key] for key in
        ("frame", "station_id", "reference_rgb_sha256", "reference_mask_sha256")} for row in rgb_rows],
        depth_evaluation=dict(observation_manifest_sha256="fixed-actual-observation-manifest", evaluation_frames=[{key: row[key]
            for key in ("frame", "station_id", "target_roster_sha256", "target_pixels")} for row in depth_rows]))
    write_json(directory / "evaluation_manifest.json", expected)
    write_json(directory / "heldout_metrics.json", dict(model_sha256=artifact["sha256"], views=rgb_rows))
    write_json(directory / "heldout_depth_metrics.json", dict(model_sha256=artifact["sha256"], split="heldout", used_for_optimization=False,
        geometry_scope="transductive_shared_sfm", optimization_station_ids=["train"], observation_manifest_sha256="fixed-actual-observation-manifest", views=depth_rows))
    module_root = Path(experiment.__file__).parent
    provenance = dict(settings=asdict(TrainingSettings.from_inputs({}, settings)), inputs=inputs,
        code_sha256=sha256(module_root / "training.py"), quality_code_sha256=sha256(module_root / "quality.py"), export_code_sha256=sha256(module_root / "export.py"))
    selection = dict(accepted_model=None if candidate else "model.ply", accepted_model_sha256=None if candidate else artifact["sha256"],
                     candidate_model="model.ply" if candidate else None, candidate_model_sha256=artifact["sha256"] if candidate else None,
                     selected="candidate_pending_comparison" if candidate else "baseline")
    manifest = dict(status="completed", artifact=artifact, provenance=provenance, selection=selection,
                    quality=dict(quality_improved=False, candidate_comparison="not_run"))
    write_json(directory / "manifest.json", manifest)
    return manifest


def paired_arms(tmp_path):
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    write_arm(baseline, dict(training=dict(depth_moment_weight=0., depth_coverage_weight=0.)), {"seed": "same"})
    write_arm(candidate, dict(training=dict(depth_moment_weight=.01, depth_coverage_weight=.01)), {"seed": "same"}, candidate=True)
    return baseline, candidate


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def test_compare_arms_uses_trusted_depth_roster_and_actual_alpha(tmp_path):
    before, after = paired_arms(tmp_path)
    result = experiment.compare_arms(before, after)
    assert result["accepted"] and result["geometry"]["evidence"]["relative_improvement"] == pytest.approx(.75)
    np.savez(after / "alpha0.npz", alpha=np.zeros((4, 4)), mask=np.ones((4, 4), bool))
    rejected = experiment.compare_arms(before, after)
    assert not rejected["accepted"] and "new_static_holes" in rejected["appearance"]["reasons"]


@pytest.mark.parametrize("mutation", ["model", "status", "artifact", "code", "budget", "inputs"])
def test_compare_arms_rejects_corrupt_or_unmatched_binding(tmp_path, mutation):
    before, after = paired_arms(tmp_path)
    manifest = read(after / "manifest.json")
    if mutation == "model":
        with (after / "model.ply").open("r+b") as stream:
            stream.seek(-4, 2); stream.write(np.float32(.1).tobytes())
    elif mutation == "status":
        manifest["status"] = "failed"
    elif mutation == "artifact":
        manifest["artifact"]["sha256"] = "0"*64
    elif mutation == "code":
        manifest["provenance"]["code_sha256"] = "different-trainer"
    elif mutation == "budget":
        manifest["provenance"]["settings"]["steps"] += 1
    else:
        manifest["provenance"]["inputs"] = {"seed": "different"}
    write_json(after / "manifest.json", manifest)
    with pytest.raises(ValueError):
        experiment.compare_arms(before, after)


def install_unit_trainer(monkeypatch, *, improvement=True, fail_candidate=False):
    calls = []
    def unit_trainer(config, job_dir, settings):
        job_dir = Path(job_dir)
        candidate = settings["training"]["depth_moment_weight"] > 0
        calls.append(job_dir.name)
        if candidate and fail_candidate:
            raise RuntimeError("Synthetic candidate failure")
        inputs = {p.relative_to(job_dir / "sfm/dataset").as_posix(): sha256(p) for p in (job_dir / "sfm/dataset").rglob("*") if p.is_file()}
        return write_arm(job_dir / "training", settings, inputs, candidate=candidate, improvement=improvement)
    monkeypatch.setattr(experiment.training, "run", unit_trainer)
    return calls


def job_fixture(tmp_path):
    dataset = tmp_path / "sfm/dataset"
    dataset.mkdir(parents=True)
    (dataset / "immutable_input.bin").write_bytes(b"actual immutable fixture bytes")
    return dict(quality=dict(enabled=True, depth_weights=[.01], coverage_weight=.01))


@pytest.mark.parametrize('quality_enabled', [False, True])
@pytest.mark.parametrize('initializer_key', ['initialization', 'gaussian_initializer', 'spherical_initializer'])
def test_product_rejects_predicted_gaussian_initialization_before_compute(tmp_path, monkeypatch, quality_enabled, initializer_key):
    settings = {initializer_key: dict(mode='predicted_gaussians'),
                'quality': dict(enabled=quality_enabled),
                'training': dict(depth_use_validated_prior=True)}
    def unexpected_compute(*args, **kwargs):
        pytest.fail('Rejected initializer must not start depth inference or training')
    monkeypatch.setattr(experiment, 'prepare_depth_prior', unexpected_compute)
    monkeypatch.setattr(experiment.training, 'run', unexpected_compute)
    with pytest.raises(ValueError, match='Multi-view generation uses SfM initialization'):
        experiment.run({}, tmp_path, settings)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("scenario", ["accept", "reject", "failure"])
def test_run_promotes_only_passing_candidate_and_exports_bound_selection(tmp_path, monkeypatch, scenario):
    settings = job_fixture(tmp_path)
    original = (tmp_path / "sfm/dataset/immutable_input.bin").read_bytes()
    calls = install_unit_trainer(monkeypatch, improvement=scenario=="accept", fail_candidate=scenario=="failure")
    result = experiment.run({}, tmp_path, settings)
    chosen = "candidate" if scenario == "accept" else "baseline"
    assert result["selection"]["selected"] == chosen
    assert result["quality"]["quality_improved"] == (scenario == "accept")
    exported = export_run({}, tmp_path, settings)
    assert exported["artifact"]["sha256"] == result["selection"]["accepted_model_sha256"]
    assert (tmp_path / "sfm/dataset/immutable_input.bin").read_bytes() == original
    assert calls == ["baseline", "depth_00"]
    assert experiment.run({}, tmp_path, settings) == result
    assert calls == ["baseline", "depth_00"]


def test_changed_reused_input_or_comparison_evidence_rejected(tmp_path, monkeypatch):
    settings = job_fixture(tmp_path)
    install_unit_trainer(monkeypatch)
    experiment.run({}, tmp_path, settings)
    comparison = tmp_path / "training/selection_comparison.json"
    original = comparison.read_bytes()
    comparison.write_bytes(original + b" ")
    with pytest.raises(ValueError):
        experiment.run({}, tmp_path, settings)
    comparison.write_bytes(original)
    (tmp_path / "sfm/dataset/immutable_input.bin").write_bytes(b"changed source")
    with pytest.raises(ValueError):
        experiment.run({}, tmp_path, settings)


def test_model_changed_after_approval_is_not_promoted(tmp_path, monkeypatch):
    settings = job_fixture(tmp_path)
    install_unit_trainer(monkeypatch)
    original_compare = experiment.compare_arms
    def changed_after_compare(before, after):
        result = original_compare(before, after)
        assert result["accepted"]
        with (Path(after) / "model.ply").open("r+b") as stream:
            stream.seek(-4, 2); stream.write(np.float32(.1).tobytes())
        return result
    monkeypatch.setattr(experiment, "compare_arms", changed_after_compare)
    with pytest.raises(ValueError):
        experiment.run({}, tmp_path, settings)


def test_zero_coverage_weight_is_rejected_before_any_training(tmp_path, monkeypatch):
    settings = job_fixture(tmp_path)
    settings["quality"]["coverage_weight"] = 0.
    calls = install_unit_trainer(monkeypatch)
    with pytest.raises(ValueError):
        experiment.run({}, tmp_path, settings)
    assert calls == []


def write_prior_fixture(root, config, settings, *, accepted=True):
    """Synthetic artifact roster for orchestration, never model/quality evidence."""
    prior = root / 'depth_prior'
    prior.mkdir(parents=True, exist_ok=True)
    options = asdict(depth_prior.DepthPriorSettings(**settings['depth_prior']))
    write_json(prior / 'input_manifest.json', dict(
        job_config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        settings=options, inputs_sha256={}, models={}))
    write_json(prior / 'calibration_report.json', dict(test_double='orchestration only'))
    write_json(prior / 'raw/provenance.json', dict(test_double='no actual model inference'))
    write_json(prior / 'observation_deduplication.json', dict(removed_rows=0))
    (prior / 'raw/radial.npz').write_bytes(b'unvalidated raw range must not enter arms')
    (prior / 'validated').mkdir(exist_ok=True)
    valid = np.zeros((4,4), bool); valid[1,1] = accepted
    for name in ['accepted.npz', 'rejected.npz']:
        mask = valid if name == 'accepted.npz' else np.zeros_like(valid)
        np.savez(prior / 'validated' / name, valid=mask, depth_z=mask.astype(np.float32)*4,
                 confidence=mask.astype(np.float32)*.4, source_count=mask.astype(np.uint16)*2,
                 candidate_depth_z=np.full((4,4), 9000., np.float32), candidate_valid=np.ones((4,4), bool))
    manifest = dict(schema_version=1, status='complete', stage='depth_prior', settings=options,
        validation_status='multistation_consistent_subset' if accepted else 'insufficient_supported_depth',
        accepted_pixels=int(accepted), training_integration_status='not_attached_to_dataset_or_trainer',
        input_manifest_sha256=sha256(prior / 'input_manifest.json'),
        calibration_report_sha256=sha256(prior / 'calibration_report.json'),
        provenance_sha256=sha256(prior / 'raw/provenance.json'),
        observation_deduplication_sha256=sha256(prior / 'observation_deduplication.json'),
        source_dataset_files_sha256={p.relative_to(root/'sfm/dataset').as_posix():sha256(p) for p in (root/'sfm/dataset').rglob('*') if p.is_file()},
        entries=[dict(frame_name=name, npz='validated/'+name, split='train', valid_count=int(accepted and name=='accepted.npz'),
                      status='accepted' if accepted and name=='accepted.npz' else 'insufficient_support',
                      sha256=sha256(prior/'validated'/name)) for name in ['accepted.npz','rejected.npz']])
    write_json(prior / 'manifest.json', manifest)
    return manifest


def install_prior_orchestration(monkeypatch, events, *, fail_prior=False, accepted=True):
    def infer(config, root, settings):
        events.append(('infer', Path(root).name))
        if fail_prior:
            (Path(root)/'depth_prior/raw').mkdir(parents=True)
            (Path(root)/'depth_prior/raw/unfinished.npz').write_bytes(b'partial raw inference')
            raise RuntimeError('Synthetic prior failure')
        return write_prior_fixture(Path(root), config, settings, accepted=accepted)
    def train(config, root, settings):
        root = Path(root); candidate = settings.get('training', {}).get('depth_moment_weight', 0.) > 0
        prior = root/'depth_prior'; paths = {p.relative_to(prior).as_posix():sha256(p) for p in prior.rglob('*') if p.is_file()} if prior.exists() else {}
        events.append(('train', root.name, candidate, paths))
        valid_prior = bool((prior/'manifest.json').exists() and read(prior/'manifest.json')['accepted_pixels'])
        if candidate and settings.get('training', {}).get('depth_use_validated_prior') and not valid_prior:
            raise RuntimeError('No accepted learned TRAIN pixels; cannot claim UniSHARP supervision')
        inputs = {p.relative_to(root/'sfm/dataset').as_posix():sha256(p) for p in (root/'sfm/dataset').rglob('*') if p.is_file()}
        inputs['validated_depth_prior'] = paths
        result = write_arm(root/'training', settings, inputs, candidate=candidate)
        result['provenance']['validated_depth_prior'] = dict(used_for_optimization=bool(candidate and valid_prior))
        write_json(root/'training/manifest.json', result)
        return result
    def finish(config, root, settings):
        root = Path(root); result = read(root/'training/manifest.json')
        assert validate_ply(root/'training/model.ply')['sha256'] == result['selection']['accepted_model_sha256']
        if settings.get('quality', {}).get('enabled'):
            assert result.get('experiment_signature')
            assert (root/'training/selection_comparison.json').is_file()
        events.append(('finalize', result['selection']['selected']))
        write_json(root/'training/sky_publication.json', dict(test_double='finalization call only'))
        return result
    monkeypatch.setattr(depth_prior, 'run', infer)
    monkeypatch.setattr(depth_prior, 'model_assets', lambda options: {})
    monkeypatch.setattr(experiment.training, 'run', train)
    monkeypatch.setattr(experiment, 'finalize', finish)


def prior_settings(tmp_path):
    settings = job_fixture(tmp_path)
    settings['training'] = dict(depth_use_validated_prior=True)
    settings['depth_prior'] = dict(erp_width=64, output_side=16)
    return settings


def test_prior_once_identical_accepted_files_all_arms_then_finalize_and_reuse(tmp_path, monkeypatch):
    settings = prior_settings(tmp_path); settings['quality']['depth_weights'] = [.01, .003]
    events = []; install_prior_orchestration(monkeypatch, events)
    original = (tmp_path/'sfm/dataset/immutable_input.bin').read_bytes()
    result = experiment.run({}, tmp_path, settings)
    assert result['selection']['selected'] == 'candidate'
    assert [event[0] for event in events] == ['infer','train','train','train','finalize']
    copied = [event[3] for event in events if event[0]=='train']
    assert copied[0] == copied[1] == copied[2]
    assert 'validated/accepted.npz' in copied[0]
    assert 'validated/rejected.npz' not in copied[0] and 'raw/radial.npz' not in copied[0]
    assert 'raw/provenance.json' in copied[0]  # provenance is needed, raw prediction is not
    with np.load(tmp_path/'experiments/baseline/depth_prior/validated/accepted.npz') as data:
        assert 'candidate_depth_z' in data.files  # preserve accepted file bytes, not a sanitized NPZ
    assert (tmp_path/'sfm/dataset/immutable_input.bin').read_bytes() == original
    assert not read(tmp_path/'experiments/baseline/training/manifest.json')['provenance']['validated_depth_prior']['used_for_optimization']
    assert experiment.run({}, tmp_path, settings) == result
    assert [event[0] for event in events].count('infer') == 1
    assert [event[0] for event in events].count('train') == 3
    assert [event[0] for event in events].count('finalize') == 2


@pytest.mark.parametrize('quality_enabled', [False, True])
def test_required_prior_missing_operator_config_rejected_before_training(tmp_path, monkeypatch, quality_enabled):
    settings = prior_settings(tmp_path); del settings['depth_prior']; settings['quality']['enabled'] = quality_enabled
    events = []; install_prior_orchestration(monkeypatch, events)
    with pytest.raises(ValueError, match='depth_prior model settings'):
        experiment.run({}, tmp_path, settings)
    assert events == [] and not (tmp_path/'training').exists()


@pytest.mark.parametrize('scenario', ['failed', 'empty'])
def test_failed_or_empty_prior_keeps_baseline_without_claiming_use(tmp_path, monkeypatch, scenario):
    settings = prior_settings(tmp_path); events = []
    install_prior_orchestration(monkeypatch, events, fail_prior=scenario=='failed', accepted=False)
    result = experiment.run({}, tmp_path, settings)
    assert result['selection']['selected'] == 'baseline' and not result['quality']['quality_improved']
    assert not result['provenance']['validated_depth_prior']['used_for_optimization']
    record = result['quality']['candidate_comparison']['candidates'][0]
    assert not record['accepted'] and 'No accepted learned TRAIN' in record['error']
    for event in events:
        if event[0]=='train':
            assert not any(name.endswith('.npz') for name in event[3])
    if scenario=='failed':
        assert read(tmp_path/'depth_prior_attempt.json')['status']=='failed'
        assert not (tmp_path/'experiments/baseline/depth_prior').exists()
    assert events[-1] == ('finalize','baseline')
    experiment.run({}, tmp_path, settings)
    assert [event[0] for event in events].count('infer') == 1
    assert events[-1] == ('finalize','baseline')


@pytest.mark.parametrize('mutation', ['accepted_map', 'calibration', 'source_dataset'])
def test_prior_or_source_mutation_rejects_reuse_before_finalization(tmp_path, monkeypatch, mutation):
    settings = prior_settings(tmp_path); events=[]; install_prior_orchestration(monkeypatch, events)
    experiment.run({}, tmp_path, settings)
    paths = dict(accepted_map='depth_prior/validated/accepted.npz', calibration='depth_prior/calibration_report.json', source_dataset='sfm/dataset/immutable_input.bin')
    path = tmp_path/paths[mutation]; path.write_bytes(path.read_bytes()+b'changed')
    finalized = sum(event[0]=='finalize' for event in events)
    with pytest.raises(ValueError):experiment.run({}, tmp_path, settings)
    assert sum(event[0]=='finalize' for event in events)==finalized
    assert sum(event[0]=='infer' for event in events)==1


def test_no_quality_branch_finalizes_first_result_and_existing_publication(tmp_path, monkeypatch):
    settings=prior_settings(tmp_path); settings['quality']['enabled']=False
    events=[]; install_prior_orchestration(monkeypatch, events)
    first=experiment.run({},tmp_path,settings);again=experiment.run({},tmp_path,settings)
    assert first==again
    assert [event[0] for event in events]==['infer','train','finalize','train','finalize']


@pytest.mark.parametrize('mutation', ['calibration', 'model_setting', 'model_bytes', 'frozen_config'])
def test_cached_prior_rejects_changed_request_before_training(tmp_path, monkeypatch, mutation):
    settings=prior_settings(tmp_path); events=[]; install_prior_orchestration(monkeypatch, events)
    write_prior_fixture(tmp_path, {}, settings)
    requested=deepcopy(settings); config={}
    if mutation=='calibration':
        requested['depth_prior']['maximum_relative_depth_error']=.04
    elif mutation=='model_setting':
        requested['depth_prior']['checkpoint_sha256']='a'*64
    elif mutation=='model_bytes':
        monkeypatch.setattr(depth_prior, 'model_assets', lambda options: dict(source_tree_sha256='changed'))
    else:
        config={'selection': 'different frozen selection'}
    with pytest.raises(ValueError, match='Cached learned prior'):
        experiment.run(config,tmp_path,requested)
    assert events==[] and not (tmp_path/'experiments').exists()


def test_portable_prior_clone_allows_installation_paths_but_binds_source_bytes(tmp_path, monkeypatch):
    source=tmp_path/'source';settings=prior_settings(source)
    settings['depth_prior'].update(repo_path='/old/repo',checkpoint_path='/old/model.pt',
                                   unik3d_snapshot_path='/old/unik3d',extra_python_path='/old/site')
    manifest=write_prior_fixture(source, {}, settings)
    input_path=source/'depth_prior/input_manifest.json';original=read(input_path)
    # Historical absolute paths remain in the signed input manifest. The clone
    # must verify its portable dataset roster instead of reopening these paths.
    original['inputs_sha256']={'/old/job/sfm/dataset/immutable_input.bin':'unused-old-path'}
    original['models']=dict(repo_path='/old/repo',checkpoint_path='/old/model.pt',unik3d_snapshot_path='/old/unik3d',
                            checkpoint_sha256='a'*64,unik3d_sha256='b'*64,source_files={'unisharp/model.py':'c'*64})
    write_json(input_path, original);manifest['input_manifest_sha256']=sha256(input_path)
    write_json(source/'depth_prior/manifest.json',manifest)
    clone=tmp_path/'clone';shutil.copytree(source,clone)
    requested=deepcopy(settings)
    requested['depth_prior'].update(repo_path='/new/repo',checkpoint_path='/new/model.pt',
                                    unik3d_snapshot_path='/new/unik3d',extra_python_path='/new/site',resume_raw=True)
    actual=deepcopy(original['models']);actual.update(repo_path='/new/repo',checkpoint_path='/new/model.pt',unik3d_snapshot_path='/new/unik3d')
    monkeypatch.setattr(depth_prior,'model_assets',lambda options:actual)
    experiment.validate_prior_cache({},clone,requested)
    (clone/'sfm/dataset/immutable_input.bin').write_bytes(b'changed clone input')
    with pytest.raises(ValueError,match='source dataset changed'):
        experiment.validate_prior_cache({},clone,requested)
    assert (source/'sfm/dataset/immutable_input.bin').read_bytes()==b'actual immutable fixture bytes'


def test_prior_copy_is_idempotent_and_never_overwrites_a_different_target(tmp_path):
    source=tmp_path/'source';settings=prior_settings(source);write_prior_fixture(source,{},settings)
    arm=tmp_path/'arm';experiment.copy_prior(source,arm)
    before={p.relative_to(arm).as_posix():sha256(p) for p in arm.rglob('*') if p.is_file()}
    experiment.copy_prior(source,arm)
    assert {p.relative_to(arm).as_posix():sha256(p) for p in arm.rglob('*') if p.is_file()}==before
    # A separate target avoids mutating a deliberately hardlinked valid copy.
    conflicting=tmp_path/'conflicting/depth_prior/manifest.json';conflicting.parent.mkdir(parents=True)
    conflicting.write_bytes(b'existing unrelated prior')
    source_bytes=(source/'depth_prior/manifest.json').read_bytes()
    with pytest.raises(ValueError,match='copy target differs'):
        experiment.copy_prior(source,tmp_path/'conflicting')
    assert conflicting.read_bytes()==b'existing unrelated prior'
    assert (source/'depth_prior/manifest.json').read_bytes()==source_bytes


@pytest.mark.parametrize('scenario', ['failed', 'empty'])
def test_no_quality_required_positive_prior_fails_without_fake_fallback(tmp_path,monkeypatch,scenario):
    settings=prior_settings(tmp_path);settings['quality']['enabled']=False
    settings['training']['depth_moment_weight']=.01
    events=[];install_prior_orchestration(monkeypatch,events,fail_prior=scenario=='failed',accepted=False)
    with pytest.raises(RuntimeError,match='No accepted learned TRAIN'):
        experiment.run({},tmp_path,settings)
    assert [event[0] for event in events]==['infer','train']
    assert not (tmp_path/'training').exists()
