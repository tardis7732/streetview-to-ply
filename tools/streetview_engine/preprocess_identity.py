"""One input identity for preprocessing and same-host cache verification.

SAM3 binds the configured interpreter's installed package metadata and current
inference code as well as model bytes/settings. This module never imports an
inference framework or loads weights. The established HF semantic identity is
kept byte-for-byte compatible with completed legacy caches.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from .imaging import fingerprint, sha256


_ENGINE_DIR = Path(__file__).parent
_PACKAGES = ('torch', 'torchvision', 'transformers', 'tokenizers', 'safetensors',
             'huggingface-hub', 'numpy', 'Pillow')


def _local_runtime():
    versions = {}
    for name in _PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            # Model-free contract tests may lack the inference packages. An
            # absent distribution is bound explicitly; inference still fails
            # normally if a required package is unavailable.
            versions[name] = None
    return dict(python_implementation=platform.python_implementation(),
                python_version=platform.python_version(), packages=versions)


def _runtime_provenance(options):
    configured = options.get('python_executable')
    if configured is None:
        return _local_runtime()
    if not isinstance(configured, str):
        raise ValueError('Segmentation python_executable must be an existing absolute Python path')
    target = Path(configured).expanduser()
    if not target.is_absolute() or not target.is_file():
        raise ValueError('Segmentation python_executable must be an existing absolute Python path')
    # Do not resolve venv symlinks: distinct environments may share one binary.
    if os.path.normcase(os.path.abspath(target)) == os.path.normcase(os.path.abspath(sys.executable)):
        return _local_runtime()
    code = (
        'import importlib.metadata,json,platform\n'
        f'names={_PACKAGES!r}\n'
        'versions={}\n'
        'for name in names:\n'
        ' try: versions[name]=importlib.metadata.version(name)\n'
        ' except importlib.metadata.PackageNotFoundError: versions[name]=None\n'
        'print(json.dumps(dict(python_implementation=platform.python_implementation(),'
        'python_version=platform.python_version(),packages=versions),sort_keys=True))\n'
    )
    try:
        result = subprocess.run([str(target), '-B', '-c', code], check=True,
                                capture_output=True, text=True, encoding='utf8',
                                timeout=30, shell=False)
        value = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        # Neither arbitrary child output nor environment values belong in logs.
        raise RuntimeError('Cannot query configured segmentation runtime metadata') from exc
    if (not isinstance(value, dict) or set(value) != {'python_implementation', 'python_version', 'packages'}
            or not all(isinstance(value[k], str) and value[k] for k in ('python_implementation', 'python_version'))
            or not isinstance(value['packages'], dict) or set(value['packages']) != set(_PACKAGES)
            or any(v is not None and (not isinstance(v, str) or not v) for v in value['packages'].values())):
        raise RuntimeError('Configured segmentation runtime returned invalid package metadata')
    return value


def _code_provenance():
    names = ('preprocess_identity.py', 'preprocess.py', 'sam3_segmenter.py',
             'imaging.py', 'processing_options.py')
    paths = [_ENGINE_DIR/name for name in names]
    # The verifier is optional for SAM3 v1; adding/changing it invalidates a
    # current-code cache without altering any frozen snapshot or old output.
    verifier = _ENGINE_DIR/'object_verifier.py'
    if verifier.exists():
        paths.append(verifier)
    if any(not path.is_file() for path in paths):
        raise RuntimeError('Missing preprocessing inference code for cache identity')
    return {path.name: sha256(path) for path in paths}


def _verifier_binding(options, provenance):
    configured = options.get('instance_verifier')
    recorded = provenance.get('instance_verifier')
    if configured is None:
        if recorded is not None:
            raise ValueError('Verifier provenance exists without configured instance_verifier')
        return None
    if not isinstance(configured, dict) or not configured.get('model_path'):
        raise ValueError('Instance verifier requires nonempty explicit local model options')
    if (not isinstance(recorded, dict) or not recorded.get('files_sha256')
            or not isinstance(recorded.get('file_hashes'), dict) or not recorded['file_hashes']):
        raise ValueError('Instance verifier requires current nested weight provenance')
    return recorded


def preprocess_input_identity(config, collection_sha256, semantic_options, policy, provenance):
    """Return the canonical payload; caller obtains provenance from local files.

    Both stage creation and cache validation must pass freshly calculated model
    provenance, never a manifest's previously recorded model file hashes.
    """
    backend = semantic_options.get('backend', 'hf_semantic')
    if backend not in ('hf_semantic', 'sam3'):
        raise ValueError('Unsupported segmentation backend for input identity')
    identity = dict(config=config, collection_sha256=collection_sha256,
                    semantic_options=semantic_options, policy=policy,
                    model_files_sha256=provenance['files_sha256'])
    if backend == 'hf_semantic':
        return identity
    verifier = _verifier_binding(semantic_options, provenance)
    identity['sam3_inference_identity'] = dict(
        schema_version=1,
        evidence_schema='sam3_group_evidence_v2' if verifier is not None else 'sam3_group_evidence_v1',
        inference_code_sha256=_code_provenance(),
        runtime=_runtime_provenance(semantic_options),
        instance_verifier_provenance=verifier)
    return identity


def preprocess_input_fingerprint(config, collection_sha256, semantic_options, policy, provenance):
    return fingerprint(preprocess_input_identity(config, collection_sha256,
                                                semantic_options, policy, provenance))
