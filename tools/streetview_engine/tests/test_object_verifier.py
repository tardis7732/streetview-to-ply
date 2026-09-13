"""Verifier loading contract tests require neither Transformers nor CUDA."""
import json

import pytest

from tools.streetview_engine.object_verifier import verifier_provenance


def model(tmp_path):
    names = ['person', 'car', 'bus', 'truck', 'motorbike', 'bicycle']
    names += ['unused_' + str(i) for i in range(74)]
    (tmp_path/'config.json').write_text(json.dumps(dict(model_type='rt_detr_v2', id2label=dict(enumerate(names)))))
    (tmp_path/'preprocessor_config.json').write_text('{}')
    (tmp_path/'model.safetensors').write_bytes(b'contract fixture only, not loadable weights')
    return dict(backend='rtdetr_v2', model_path=str(tmp_path))


def test_provenance_preserves_actual_motorbike_taxonomy(tmp_path):
    options = model(tmp_path)
    result = verifier_provenance(options)
    assert result['id2label'][4] == 'motorbike'
    assert result['local_files_only'] and not result['trust_remote_code']
    assert len(result['file_hashes']) == 3


def test_bound_artifact_hash_rejects_changed_model(tmp_path):
    options = model(tmp_path)
    options['expected_files_sha256'] = verifier_provenance(options)['files_sha256']
    (tmp_path/'model.safetensors').write_bytes(b'different weights')
    with pytest.raises(ValueError, match='local files changed'):
        verifier_provenance(options)


@pytest.mark.parametrize('fault', ['backend', 'model_type', 'taxonomy', 'missing_weights'])
def test_invalid_model_contract_fails_before_any_torch_import(tmp_path, fault):
    options = model(tmp_path)
    if fault == 'backend':
        options['backend'] = 'typo'
    elif fault == 'missing_weights':
        (tmp_path/'model.safetensors').unlink()
    else:
        path = tmp_path/'config.json'
        config = json.loads(path.read_text())
        if fault == 'model_type':
            config['model_type'] = 'other'
        else:
            config['id2label']['0'] = 'unmapped'
        path.write_text(json.dumps(config))
    with pytest.raises((ValueError, FileNotFoundError)):
        verifier_provenance(options)
