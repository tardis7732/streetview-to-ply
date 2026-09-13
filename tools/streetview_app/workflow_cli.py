"""Explicit CLI for the same loopback GUI recipe, reuse and Unreal actions."""
import argparse
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def request(base, route, payload=None):
    parsed = urlsplit(base)
    if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1') or parsed.username or parsed.password:
        raise ValueError('The running local GUI server is required')
    data = None if payload is None else json.dumps(payload, allow_nan=False).encode('utf8')
    query = Request(base.rstrip('/') + route, data=data,
                    headers={'Content-Type': 'application/json', 'Origin': base.rstrip('/')})
    try:
        with urlopen(query, timeout=60) as response:
            return json.load(response)
    except HTTPError as error:
        detail = json.loads(error.read()).get('error', str(error))
        raise ValueError(detail) from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', default='http://127.0.0.1:8765')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('presets')
    save = commands.add_parser('save-preset')
    save.add_argument('--name', required=True)
    save.add_argument('--config', type=Path, required=True)
    save.add_argument('--base-recipe-id')
    run = commands.add_parser('generate')
    run.add_argument('--config', type=Path, required=True, help='Verified selection JSON; provider metadata is rechecked')
    run.add_argument('--recipe-id')
    reuse = commands.add_parser('reuse')
    reuse.add_argument('job_id')
    reuse.add_argument('--from-stage', choices=['collect', 'preprocess', 'sfm', 'train', 'export'])
    unreal = commands.add_parser('open-unreal')
    unreal.add_argument('job_id')
    unreal.add_argument('--kind', choices=['generation', 'filter'], default='generation')
    status = commands.add_parser('status')
    status.add_argument('job_id', nargs='?')
    status.add_argument('--kind', choices=['generation', 'filter', 'unreal'], default='generation')
    args = parser.parse_args()
    payload = None
    if args.command == 'presets':
        route = '/api/recipes'
    elif args.command in ('save-preset', 'generate'):
        try:
            config = json.loads(args.config.read_text(encoding='utf-8-sig'))
            if not isinstance(config, dict):
                raise ValueError('Configuration must be a JSON object')
        except (ValueError, OSError) as error:
            parser.exit(1, str(error) + '\n')
        if args.command == 'save-preset':
            route, payload = '/api/recipes', dict(name=args.name, config=config)
            if args.base_recipe_id:
                payload['base_recipe_id'] = args.base_recipe_id
        else:
            route, payload = '/api/jobs', config
            if args.recipe_id:
                payload['recipe_id'] = args.recipe_id
    elif args.command == 'reuse':
        route = f'/api/jobs/{args.job_id}/reuse'
        if args.from_stage:
            payload = dict(from_stage=args.from_stage)
    elif args.command == 'open-unreal':
        route, payload = '/api/unreal-opens', dict(kind=args.kind, job_id=args.job_id)
    else:
        route = {'generation': '/api/jobs', 'filter': '/api/ply-filters', 'unreal': '/api/unreal-opens'}[args.kind]
        if args.job_id:
            route += '/' + args.job_id
    try:
        print(json.dumps(request(args.server, route, payload), ensure_ascii=False, indent=2))
    except (ValueError, OSError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
