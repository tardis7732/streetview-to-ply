"""Register an operator-owned remote backend and a scene-free GUI preset.

Configuration only: no SSH connection, model download, or GPU job is started.
Run from the checkout after installing its basic dependencies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    value = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(value, dict):
        raise ValueError('Configuration must be a JSON object')
    return value


def unresolved(value, prefix=''):
    if isinstance(value, dict):
        return [item for key, child in value.items()
                for item in unresolved(child, prefix + '.' + key)]
    if isinstance(value, list):
        return [item for index, child in enumerate(value)
                for item in unresolved(child, prefix + '[' + str(index) + ']')]
    if isinstance(value, str) and re.search(r'<[A-Z][A-Z0-9_]*>', value):
        return [prefix.lstrip('.')]
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', required=True, type=Path,
                        help='Local JSON declaring paths on the Linux GPU host')
    parser.add_argument('--data-dir', required=True, type=Path,
                        help='The same data directory passed to the GUI server')
    parser.add_argument('--host', required=True, help='Your configured SSH alias')
    parser.add_argument('--remote-root', default='/srv/streetview-to-ply')
    parser.add_argument('--preset', type=Path, default=ROOT/'examples/preset.example.json')
    parser.add_argument('--replace', action='store_true',
                        help='Replace backend/default preferences; existing recipes and jobs remain')
    args = parser.parse_args()
    try:
        settings_path = args.settings.resolve(strict=True)
        operator, preset = read_json(settings_path), read_json(args.preset)
        missing = unresolved(operator) + unresolved(preset)
        if missing:
            raise ValueError('Fill operator placeholders before registration: ' + ', '.join(missing))
        if 'single_panorama' in operator:
            raise ValueError('Remove the retired single_panorama configuration')
        if set(preset) - {'name', 'settings', 'description'}:
            raise ValueError('Preset must contain only name, settings and optional description')
        sys.path.insert(0, str(ROOT))
        from tools.streetview_app.cloud_setup import backend_profile
        from tools.streetview_app.jobs import Backend
        from tools.streetview_app.recipes import RecipeStore
        value = backend_profile(settings_path, host=args.host, remote_root=args.remote_root)
        directory = args.data_dir.resolve()
        backend_path, preferences_path = directory/'backend.json', directory/'ui_preferences.json'
        if not args.replace and (backend_path.exists() or preferences_path.exists()):
            raise FileExistsError('Configuration exists; use --replace after reviewing it')
        if args.replace and (directory/'jobs').is_dir():
            for state_path in (directory/'jobs').glob('*/state.json'):
                if read_json(state_path).get('status') in {'queued', 'running', 'cancelling'}:
                    raise ValueError('Wait for active jobs to finish before replacing the backend')
        backend = Backend.from_dict(value)
        store = RecipeStore(directory/'recipes', backend)
        saved = store.import_operator(dict(preset, operator_settings=operator))
        # Recipe hashes bind this exact backend and operator file. No provider data
        # or scene coordinates are created as part of registration.
        for path, document in ((backend_path, value),
                               (preferences_path, {'default_recipe_id': saved['id']})):
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2,
                                            allow_nan=False) + '\n', encoding='utf8')
            temporary.replace(path)
        print(json.dumps({'status': 'registered', 'recipe_id': saved['id'],
                          'data_dir': str(directory), 'scene_selection': {},
                          'remote_assets_checked': False, 'job_started': False}, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
