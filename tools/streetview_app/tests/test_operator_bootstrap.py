"""Scene-free operator registration must not fabricate or query provider data."""
import copy
import json
from pathlib import Path

import pytest

from tools.streetview_app.jobs import Backend
from tools.streetview_app.recipes import RecipeStore

ROOT = Path(__file__).resolve().parents[3]


def template():
    operator = json.loads((ROOT/'examples/operator.example.json').read_text(encoding='utf8'))
    # Synthetic configuration only; no model files or successful runtime are claimed.
    operator['panorama_preprocess']['flux']['model_receipt_sha256'] = 'a' * 64
    for mode in ('metric', 'pose'):
        operator['sky_depth'][mode]['model_sha256'] = 'b' * 64
        operator['sky_depth'][mode]['config_sha256'] = 'c' * 64
    preset = json.loads((ROOT/'examples/preset.example.json').read_text(encoding='utf8'))
    return dict(preset, operator_settings=operator)


def test_scene_free_registration(tmp_path):
    store = RecipeStore(tmp_path/'recipes', Backend())
    recipe = store.import_operator(template())
    assert recipe['scene_selection'] == {}
    assert recipe['automatic_start'] is False
    assert recipe['settings']['depth_cleanup']['enabled'] is True
    assert recipe['settings']['training_steps'] == 6000
    assert not (tmp_path/'jobs').exists()


@pytest.mark.parametrize('extra', ['center', 'panorama_ids', 'selection_verified'])
def test_registration_rejects_scene_or_fabricated_verification(tmp_path, extra):
    payload = template()
    payload['settings'][extra] = True
    with pytest.raises(ValueError, match='portable'):
        RecipeStore(tmp_path/'recipes', Backend()).import_operator(payload)


def test_registration_rejects_retired_mode_and_mismatched_fixed_steps(tmp_path):
    payload = template()
    for key, value in [('generation_mode', 'single_panorama'), ('training_steps', 40000)]:
        changed = copy.deepcopy(payload)
        changed['settings'][key] = value
        with pytest.raises(ValueError):
            RecipeStore(tmp_path/'recipes', Backend()).import_operator(changed)
