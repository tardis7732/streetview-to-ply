"""Portable named settings with separate scene selection and executable provenance."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import uuid

PORTABLE_FIELDS = ('generation_mode', 'training_steps', 'resolution', 'max_splats',
                   'processing_options', 'size_filter', 'depth_cleanup')
SCENE_FIELDS = ('provider', 'center', 'radius_m', 'capture_policy', 'panorama_ids',
                'excluded_panorama_ids', 'panoramas', 'timestamp_policy')
ROOT = Path(__file__).resolve().parents[2]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(',', ':')).encode('utf8')).hexdigest()


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def backend_snapshot(backend):
    """No network or execution. Hash trusted code/settings, never copy secrets."""
    declared = asdict(backend)
    sources = {}
    for package in ('streetview_app', 'streetview_engine', 'streetview_geometry'):
        for path in sorted((ROOT / 'tools' / package).glob('*.py')):
            sources[path.relative_to(ROOT).as_posix()] = file_hash(path)
    inputs = {}
    for stages in (backend.stages, *backend.mode_stages.values()):
        for stage in stages:
            for name in stage.required_paths:
                path = Path(name)
                if path.is_file():
                    inputs[str(path.resolve())] = file_hash(path)
            for name in stage.argv:
                if '{' not in name and Path(name).is_file() and Path(name).suffix.lower() in ('.py', '.json'):
                    inputs[str(Path(name).resolve())] = file_hash(name)
    settings_path = backend.remote.get('settings_path')
    if settings_path and Path(settings_path).is_file():
        inputs[str(Path(settings_path).resolve())] = file_hash(settings_path)
    result = dict(schema_version=1, backend=backend.name, compute=backend.compute,
        backend_sha256=digest(declared), source_sha256=sources, operator_files_sha256=inputs)
    result['identity_sha256'] = digest(result)
    return result


def operator_settings(backend):
    path = backend.remote.get('settings_path')
    if not path:
        return {}
    value = json.loads(Path(path).read_text(encoding='utf8'))
    if not isinstance(value, dict):
        raise ValueError('Operator settings must be an object')
    return value


def depth_cleanup_capability(settings, config=None):
    """Describe configured support without loading models or probing the cloud."""
    from ..streetview_engine.depth_cleanup_options import (
        read_depth_cleanup_options, validate_depth_cleanup_support)
    config = config if isinstance(config, dict) else {}
    try:
        explicit = read_depth_cleanup_options(config)
        validate_depth_cleanup_support(dict(config, depth_cleanup={'enabled': True}), settings)
    except (ValueError, TypeError, KeyError):
        return dict(available=False, default_enabled=False)
    enabled = explicit['enabled'] if explicit is not None else settings['sky_cleanup']['enabled']
    return dict(available=True, default_enabled=enabled)


class RecipeStore:
    def __init__(self, root, backend):
        self.root = Path(root).resolve(); self.root.mkdir(parents=True, exist_ok=True)
        self.backend = backend

    def save(self, payload):
        return self._save(payload)

    def _save(self, payload, *, operator=None):
        if not isinstance(payload, dict) or set(payload) - {'name', 'config', 'base_recipe_id'}:
            raise ValueError('Recipe accepts only name and configuration')
        name = payload.get('name')
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 100 or any(ord(c) < 32 for c in name):
            raise ValueError('Recipe name must contain 1–100 visible characters')
        config = payload.get('config')
        if not isinstance(config, dict):
            raise ValueError('Recipe configuration must be an object')
        from .jobs import generation_mode, validate_config
        config = validate_config(config)
        if generation_mode(config) != 'multi_view':
            raise ValueError('Only multi_view recipes can be created')
        base = self.get(payload['base_recipe_id']) if payload.get('base_recipe_id') else None
        if base and not base['executable']:
            raise ValueError('; '.join(base['unavailable_reasons']))
        configured_operator = (operator if operator is not None else
            base['operator_settings'] if base else operator_settings(self.backend))
        from ..streetview_engine.depth_cleanup_options import validate_depth_cleanup_support
        validate_depth_cleanup_support(config, configured_operator)
        recipe = dict(schema_version=1, id=uuid.uuid4().hex, name=name.strip(),
            created_utc=datetime.now(timezone.utc).isoformat(), kind='connected_pipeline',
            settings={k: config[k] for k in PORTABLE_FIELDS if k in config},
            scene_selection={k: config[k] for k in SCENE_FIELDS if k in config},
            operator_settings=configured_operator, execution_snapshot=backend_snapshot(self.backend),
            automatic_start=False, reproduces_reference=False)
        if base:
            for key in ('reference_thumbnail', 'description', 'algorithm_reference'):
                if key in base:
                    recipe[key] = base[key]
        recipe['content_sha256'] = digest(recipe)
        (self.root / (recipe['id'] + '.json')).write_text(json.dumps(recipe, ensure_ascii=False,
            indent=2, allow_nan=False), encoding='utf8')
        return self.get(recipe['id'])

    def import_operator(self, payload):
        """Bootstrap a scene-free preset from trusted local operator configuration.

        This is not an HTTP operation. Registration checks configuration only;
        the execution host verifies actual binaries and model assets at runtime.
        No fabricated provider selection is needed to register an algorithm.
        """
        if not isinstance(payload, dict) or set(payload) - {'name', 'settings', 'operator_settings', 'description'}:
            raise ValueError('Unsupported operator preset fields')
        value = json.loads(json.dumps(payload, allow_nan=False))
        name = value.get('name')
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 100 or any(ord(c) < 32 for c in name):
            raise ValueError('Preset name must contain 1-100 visible characters')
        config, operator = value.get('settings'), value.get('operator_settings')
        if not isinstance(config, dict) or set(config) != set(PORTABLE_FIELDS):
            raise ValueError('Operator preset requires exactly the portable settings fields')
        if config['generation_mode'] != 'multi_view':
            raise ValueError('Only multi-view generation is supported')
        for key in ('training_steps', 'resolution', 'max_splats'):
            if type(config[key]) is not int:
                raise ValueError(key + ' must be an integer')
        if not isinstance(operator, dict):
            raise ValueError('Operator settings must be an object')
        from ..streetview_engine.processing_options import read_processing_options
        from ..streetview_engine.size_filter import read_size_filter_options
        from ..streetview_engine.panorama_workflow import workflow_options
        from ..streetview_engine.panorama_preprocess import options_for
        from ..streetview_engine.sfm import SfMSettings
        config['processing_options'] = read_processing_options(config)
        config['size_filter'] = read_size_filter_options(config)
        workflow_options(config, operator)
        options_for(operator)
        SfMSettings(**operator.get('sfm', {}))
        description = value.get('description', '')
        if not isinstance(description, str) or len(description) > 2000:
            raise ValueError('Preset description must be a bounded string')
        recipe = dict(schema_version=1, id=uuid.uuid4().hex, name=name.strip(),
            created_utc=datetime.now(timezone.utc).isoformat(), kind='connected_pipeline',
            settings=config, scene_selection={}, operator_settings=operator,
            execution_snapshot=backend_snapshot(self.backend), automatic_start=False,
            reproduces_reference=False, description=description)
        recipe['content_sha256'] = digest(recipe)
        (self.root / (recipe['id'] + '.json')).write_text(json.dumps(recipe,
            ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
        return self.get(recipe['id'])

    def import_connected(self, payload):
        """Register a trusted operator profile; HTTP clients cannot set executable paths."""
        if not isinstance(payload, dict) or set(payload) - {'name', 'config', 'operator_settings',
                'reference_thumbnail', 'description', 'algorithm_reference'}:
            raise ValueError('Unsupported connected recipe fields')
        operator = payload.get('operator_settings')
        if not isinstance(operator, dict):
            raise ValueError('Connected recipe needs explicit operator settings')
        # Validate the requested option against this trusted imported profile,
        # rather than an unrelated current backend/default profile.
        operator = json.loads(json.dumps(operator, allow_nan=False))
        saved = self._save({key: payload[key] for key in ('name', 'config')}, operator=operator)
        path = self.root / (saved['id'] + '.json')
        recipe = json.loads(path.read_text(encoding='utf8'))
        recipe['operator_settings'] = json.loads(json.dumps(operator, allow_nan=False))
        # An operator preset describes an algorithm; the user selects a scene separately.
        recipe['scene_selection'] = {}
        for key in ('reference_thumbnail', 'description', 'algorithm_reference'):
            if key in payload:
                recipe[key] = payload[key]
        recipe.pop('content_sha256')
        recipe['content_sha256'] = digest(recipe)
        path.write_text(json.dumps(recipe, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
        return self.get(saved['id'])

    def import_reference(self, payload):
        """Trusted operator data only; never exposed as an HTTP body endpoint."""
        recipe = json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))
        recipe.update(schema_version=1, kind='reference', automatic_start=False)
        recipe.setdefault('id', uuid.uuid4().hex)
        if not re.fullmatch(r'[0-9a-f]{32}', recipe['id']):
            raise ValueError('Invalid reference identity')
        recipe.setdefault('created_utc', datetime.now(timezone.utc).isoformat())
        recipe.pop('content_sha256', None)
        recipe['content_sha256'] = digest(recipe)
        destination = self.root / (recipe['id'] + '.json')
        if destination.exists():
            existing = json.loads(destination.read_text(encoding='utf8'))
            if existing != recipe:
                raise FileExistsError('A saved reference is immutable')
        else:
            destination.write_text(json.dumps(recipe, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
        return self.get(recipe['id'])

    def get(self, recipe_id):
        if not isinstance(recipe_id, str) or not re.fullmatch(r'[0-9a-f]{32}', recipe_id):
            raise KeyError('Unknown recipe')
        path = self.root / (recipe_id + '.json')
        if not path.is_file():
            raise KeyError('Unknown recipe')
        recipe = json.loads(path.read_text(encoding='utf8'))
        original_hash = recipe.pop('content_sha256', None)
        if recipe.get('id') != recipe_id or original_hash != digest(recipe):
            raise ValueError('Saved recipe content changed')
        recipe['content_sha256'] = original_hash
        reasons = list(recipe.get('unavailable_reasons', []))
        if recipe.get('kind') != 'connected_pipeline':
            reasons.append('Reference recipe has not been connected to the current generation pipeline')
        elif recipe.get('execution_snapshot') != backend_snapshot(self.backend):
            reasons.append('Backend code or operator settings changed since this recipe was saved')
        recipe.update(executable=not reasons, unavailable_reasons=reasons)
        return recipe

    def list(self):
        rows = []
        for path in self.root.glob('*.json'):
            try:
                rows.append(self.get(path.stem))
            except (OSError, ValueError, KeyError):
                continue
        return sorted(rows, key=lambda r: r.get('created_utc', ''), reverse=True)

    def resolve(self, recipe_id, config):
        """Called only by explicit generation. Scene IDs stay caller selected."""
        from .jobs import validate_config
        recipe = self.get(recipe_id)
        if not recipe['executable']:
            raise ValueError('; '.join(recipe['unavailable_reasons']))
        merged = {**config, **recipe['settings']}
        # This switch remains editable for each generation even with a preset.
        if 'depth_cleanup' in config:
            merged['depth_cleanup'] = config['depth_cleanup']
        configured = validate_config(merged)
        from ..streetview_engine.depth_cleanup_options import validate_depth_cleanup_support
        validate_depth_cleanup_support(configured, recipe['operator_settings'])
        execution = dict(id=recipe_id, content_sha256=recipe['content_sha256'],
            execution_snapshot=recipe['execution_snapshot'], operator_settings=recipe['operator_settings'])
        return dict(config=configured, execution_recipe=execution)
