"""Create an explicit operator profile for the already configured SSH host."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .jobs import Backend, STAGES
from .remote_jobs import CloudProfile


def backend_profile(settings_path, *, host, remote_root='/srv/streetview-to-ply'):
    settings_path = Path(settings_path).resolve()
    if not settings_path.is_file():
        raise ValueError('Engine settings file does not exist')
    json.loads(settings_path.read_text(encoding='utf8'))
    remote = dict(host=host, launcher=remote_root + '/run_python.sh',
                  worker=remote_root + '/code/tools/streetview_engine/remote_worker.py',
                  jobs_root=remote_root + '/runs/streetview_app', settings_path=str(settings_path),
                  lease_seconds=90, timeout_seconds=21600)
    CloudProfile(remote)
    outputs = {'collect':['collection/manifest.json'], 'preprocess':['prepared/manifest.json'],
               'sfm':['sfm/manifest.json'], 'train':['training/manifest.json'],
               'export':['export/scene.ply', 'export/report.json']}
    value = dict(name='Streetview engine / ' + host, compute='remote_adapter', remote=remote,
                 final_ply='export/scene.ply', stages=[dict(name=name, argv=['ssh'],
                 outputs=outputs[name], required_paths=[str(settings_path)], description=name) for name in STAGES])
    Backend.from_dict(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--host', required=True, help='Your configured SSH alias')
    parser.add_argument('--remote-root', default='/srv/streetview-to-ply')
    args = parser.parse_args()
    value = backend_profile(args.settings, host=args.host, remote_root=args.remote_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError('Profile already exists; review it before replacing')
    args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
