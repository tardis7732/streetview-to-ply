"""No-model cache tests: actual weight bytes, isolated Python, and cache paths."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import venv

import pytest

from tools.streetview_engine import collection, input_cache, preprocess
from tools.streetview_engine import preprocess_identity as identity
from tools.streetview_engine.imaging import fingerprint, sha256, write_json
from tools.streetview_engine.tests.test_collection import selection, source_response
from tools.streetview_engine.tests.test_sam3_preprocess import FakeConceptSegmenter


RUNTIME = dict(python_implementation='CPython', python_version='3.12.3',
               packages={name: 'fixture-1' for name in identity._PACKAGES})


def model_options(root):
    sam = root/'sam'
    sam.mkdir(parents=True)
    (sam/'config.json').write_text('{}')
    (sam/'model.safetensors').write_bytes(b'synthetic SAM weights')
    detector = root/'detector'
    detector.mkdir()
    names = ['person', 'car', 'bus', 'truck', 'motorcycle', 'bicycle']
    names += [f'synthetic_{i}' for i in range(74)]
    (detector/'config.json').write_text(json.dumps(dict(
        model_type='rt_detr_v2', id2label={str(i): n for i,n in enumerate(names)})))
    (detector/'model.safetensors').write_bytes(b'synthetic detector weights')
    (detector/'preprocessor_config.json').write_text('{}')
    return dict(backend='sam3', model_path=str(sam), instance_score_threshold=.3,
                dynamic_dilation_px=0, core_erosion_px=0,
                instance_verifier=dict(backend='rtdetr_v2', model_path=str(detector),
                                       minimum_box_iou=.25, min_detection_score=.3))


@pytest.fixture
def bound(tmp_path, monkeypatch):
    options = model_options(tmp_path/'models')
    # Hash actual files in an isolated source-tree fixture, never mutate code.
    code = tmp_path/'code'
    code.mkdir()
    for name in identity._code_provenance():
        (code/name).write_text('# synthetic inference code: '+name)
    monkeypatch.setattr(identity, '_ENGINE_DIR', code)
    monkeypatch.setattr(identity, '_runtime_provenance', lambda _: copy.deepcopy(RUNTIME))
    return options, code


def signed(options):
    return identity.preprocess_input_fingerprint(
        dict(panorama_ids=['synthetic']), 'collection-sha', options,
        preprocess.mask_policy(options), preprocess.model_provenance(options))


def test_legacy_hf_payload_is_identical_without_runtime_or_code_queries(monkeypatch):
    monkeypatch.setattr(identity, '_runtime_provenance', lambda _: pytest.fail('legacy runtime query'))
    monkeypatch.setattr(identity, '_code_provenance', lambda: pytest.fail('legacy code query'))
    config, semantic, policy, provenance = {'x': 1}, {'model_path': 'legacy'}, {'threshold': .5}, {'files_sha256': 'hash'}
    expected = dict(config=config, collection_sha256='collection', semantic_options=semantic,
                    policy=policy, model_files_sha256='hash')
    assert identity.preprocess_input_identity(config, 'collection', semantic, policy, provenance) == expected
    assert identity.preprocess_input_fingerprint(config, 'collection', semantic, policy, provenance) == fingerprint(expected)


@pytest.mark.parametrize('mutation', ['verifier_weight', 'verifier_processor', 'sam_weight', 'mapping', 'threshold', 'mask_policy'])
def test_current_model_bytes_and_verifier_configuration_change_identity(bound, mutation):
    options, _ = bound
    before = signed(options)
    if mutation in ('verifier_weight', 'verifier_processor'):
        name = 'model.safetensors' if mutation == 'verifier_weight' else 'preprocessor_config.json'
        path = Path(options['instance_verifier']['model_path'])/name
        path.write_bytes(path.read_bytes()+b' ')
    elif mutation == 'sam_weight':
        Path(options['model_path'], 'model.safetensors').write_bytes(b'changed weights, same path')
    elif mutation == 'mapping':
        options['instance_verifier']['concept_labels'] = dict(car=['truck'])
    elif mutation == 'threshold':
        options['instance_verifier']['minimum_box_iou'] = .4
    else:
        options['ground_confidence'] = .5
    assert signed(options) != before


@pytest.mark.parametrize('name', ['preprocess.py', 'sam3_segmenter.py', 'object_verifier.py', 'preprocess_identity.py'])
def test_inference_code_changes_invalidate_sam3(bound, name):
    options, code = bound
    before = signed(options)
    (code/name).write_text('# updated implementation at same path')
    assert signed(options) != before


@pytest.mark.parametrize('package', ['transformers', 'torch', 'torchvision', 'Pillow'])
def test_runtime_package_changes_invalidate_sam3(bound, monkeypatch, package):
    options, _ = bound
    before = signed(options)
    changed = copy.deepcopy(RUNTIME)
    changed['packages'][package] = 'fixture-2'
    monkeypatch.setattr(identity, '_runtime_provenance', lambda _: changed)
    assert signed(options) != before


def test_sam3_v1_gets_stronger_identity_without_requiring_v2_metadata(bound):
    options, _ = bound
    options.pop('instance_verifier')
    provenance = preprocess.model_provenance(options)
    value = identity.preprocess_input_identity({}, 'collection', options, {}, provenance)
    assert value['sam3_inference_identity']['evidence_schema'] == 'sam3_group_evidence_v1'
    assert value['sam3_inference_identity']['instance_verifier_provenance'] is None
    old = dict(config={}, collection_sha256='collection', semantic_options=options,
               policy={}, model_files_sha256=provenance['files_sha256'])
    assert fingerprint(value) != fingerprint(old)


@pytest.mark.parametrize('bad', [{}, True, {'backend': 'rtdetr_v2'}])
def test_empty_verifier_fails_before_runtime_probe(bound, monkeypatch, bad):
    options, _ = bound
    provenance = preprocess.model_provenance(options)
    options['instance_verifier'] = bad
    monkeypatch.setattr(identity, '_runtime_provenance', lambda _: pytest.fail('invalid options reached runtime'))
    with pytest.raises(ValueError, match='nonempty explicit'):
        identity.preprocess_input_identity({}, 'collection', options, {}, provenance)


def test_missing_nested_weight_binding_cannot_masquerade_as_verified(bound):
    options, _ = bound
    provenance = preprocess.model_provenance(options)
    provenance['instance_verifier'] = {}
    with pytest.raises(ValueError, match='nested weight'):
        identity.preprocess_input_identity({}, 'collection', options, {}, provenance)


def test_configured_interpreter_metadata_uses_its_venv_without_importing_models(tmp_path, monkeypatch):
    isolated = tmp_path/'isolated runtime'
    venv.EnvBuilder(with_pip=False).create(isolated)
    executable = isolated/('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    query = subprocess.run([str(executable), '-c', 'import sysconfig;print(sysconfig.get_path("purelib"))'],
                           check=True, capture_output=True, text=True)
    site = Path(query.stdout.strip())
    distribution = site/'transformers-99.0.dist-info'
    distribution.mkdir()
    (distribution/'METADATA').write_text('Metadata-Version: 2.1\nName: transformers\nVersion: 99.0\n')
    poison = site/'transformers'
    poison.mkdir()
    (poison/'__init__.py').write_text('raise AssertionError("Identity probe must not import inference code")\n')
    monkeypatch.setattr(identity, '_local_runtime', lambda: pytest.fail('Queried caller environment instead of configured venv'))
    observed = identity._runtime_provenance(dict(python_executable=str(executable)))
    assert observed['packages']['transformers'] == '99.0'
    assert observed['packages']['torch'] is None


def test_current_interpreter_does_not_spawn_another_process(monkeypatch):
    monkeypatch.setattr(identity, '_local_runtime', lambda: RUNTIME)
    monkeypatch.setattr(identity.subprocess, 'run', lambda *a, **k: pytest.fail('Unnecessary runtime subprocess'))
    assert identity._runtime_provenance(dict(python_executable=sys.executable)) == RUNTIME


def test_invalid_runtime_has_no_fallback(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='existing absolute Python'):
        identity._runtime_provenance(dict(python_executable=str(tmp_path/'missing-python')))
    target = tmp_path/'pretend-python'
    target.write_bytes(b'not launched')
    monkeypatch.setattr(identity.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0, 'not-json', ''))
    monkeypatch.setattr(identity, '_local_runtime', lambda: pytest.fail('Unexpected global fallback'))
    with pytest.raises(RuntimeError, match='Cannot query configured'):
        identity._runtime_provenance(dict(python_executable=str(target)))


class FakeVerifiedSegmenter(FakeConceptSegmenter):
    def __init__(self, options, provenance):
        super().__init__(options, provenance)
        self.metadata['evidence_schema'] = 'sam3_group_evidence_v2'


@pytest.fixture
def completed_sam3_source(tmp_path, monkeypatch):
    root = tmp_path/'source'
    options = model_options(tmp_path/'cache_models')
    settings = dict(collection=dict(max_workers=1), preprocess=dict(segmentation=options))
    config = selection()
    config['capture_policy'] = dict(mode='same_day', value='2024-03-02', time_start=None, time_end=None)
    monkeypatch.setattr(identity, '_runtime_provenance', lambda _: copy.deepcopy(RUNTIME))
    monkeypatch.setattr(collection, '_request', source_response)
    from tools.streetview_engine import sam3_segmenter
    monkeypatch.setattr(sam3_segmenter, 'Sam3EvidenceSegmenter', FakeVerifiedSegmenter)
    collection.run(config, root, settings)
    preprocess.run(config, root, settings)
    write_json(root/'config.json', config)
    return root, config, settings


def test_stage_and_cache_share_identity_without_inference_on_reuse(completed_sam3_source, monkeypatch):
    root, config, settings = completed_sam3_source
    from tools.streetview_engine import sam3_segmenter
    monkeypatch.setattr(sam3_segmenter, 'Sam3EvidenceSegmenter', lambda *a: pytest.fail('Cache reinferred'))
    verified = input_cache.validate_cache(config, root, settings)
    assert verified['prepared_fingerprint'] == preprocess.run(config, root, settings)['input_fingerprint']


@pytest.mark.parametrize('mutation', ['verifier_weight', 'mapping', 'threshold', 'runtime', 'code', 'saved_fingerprint'])
def test_changed_sam3_identity_rejects_cache_before_copy(completed_sam3_source, monkeypatch, mutation):
    root, config, settings = completed_sam3_source
    original_rgb = {p: sha256(p) for p in (root/'collection/images').glob('*.png')}
    options = settings['preprocess']['segmentation']
    if mutation == 'verifier_weight':
        Path(options['instance_verifier']['model_path'], 'model.safetensors').write_bytes(b'new detector weights')
    elif mutation == 'mapping':
        options['instance_verifier']['concept_labels'] = dict(car=['truck'])
    elif mutation == 'threshold':
        options['instance_verifier']['minimum_box_iou'] = .45
    elif mutation == 'runtime':
        changed = copy.deepcopy(RUNTIME)
        changed['packages']['transformers'] = 'changed'
        monkeypatch.setattr(identity, '_runtime_provenance', lambda _: changed)
    elif mutation == 'code':
        code = identity._code_provenance()
        code['object_verifier.py'] = 'f'*64
        monkeypatch.setattr(identity, '_code_provenance', lambda: code)
    else:
        path = root/'prepared/manifest.json'
        manifest = json.loads(path.read_text())
        manifest['input_fingerprint'] = 'f'*64
        write_json(path, manifest, immutable=False)
    destination = root.parent/'cache_destination'
    settings['input_cache'] = dict(source_job_dir=str(root))
    with pytest.raises(ValueError, match='fingerprint/model/settings'):
        input_cache.reuse_inputs(config, destination, settings)
    assert not destination.exists() or not list(destination.iterdir())
    assert all(sha256(path) == digest for path,digest in original_rgb.items())
