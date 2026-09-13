"""Configure the existing local job runner without SSH or a remote worker."""
from pathlib import Path
import json
import sys

from .jobs import Backend, STAGES


def backend_profile(settings_path, *, python=None):
    settings = Path(settings_path).resolve(strict=True)
    if not isinstance(json.loads(settings.read_text(encoding='utf-8-sig')), dict):
        raise ValueError('Operator settings must be a JSON object')
    executable = Path(python or sys.executable).resolve(strict=True)
    if not executable.is_file():
        raise ValueError('Python executable must be a file')
    root = Path(__file__).resolve().parents[2]
    outputs = {'collect': ['collection/manifest.json'], 'preprocess': ['prepared/manifest.json'],
               'sfm': ['sfm/manifest.json'], 'train': ['training/manifest.json'],
               'export': ['export/scene.ply', 'export/report.json']}
    def literal(value):
        return str(value).replace('{', '{{').replace('}', '}}')
    value = dict(name='Streetview engine / local GPU', compute='local',
        environment={'PYTHONPATH': str(root), 'STREETVIEW_OPERATOR_SETTINGS': str(settings)},
        final_ply='export/scene.ply', stages=[dict(name=name,
            argv=[literal(executable), '-B', '-m', 'tools.streetview_engine', name,
                  '--job-config', '{config}', '--job-dir', '{job_dir}', '--settings', literal(settings)],
            outputs=outputs[name], required_paths=[str(settings), str(executable)],
            description=name) for name in STAGES])
    Backend.from_dict(value)
    return value
