"""Hash operator-installed GPU assets; create the required local FLUX receipt.

Run on the Linux execution host. This reads existing files and writes only the
specified receipt/settings outputs. It downloads no files and loads no models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def existing(value, directory=False):
    if not isinstance(value, str) or '<' in value or not Path(value).is_absolute():
        raise ValueError('Replace asset path placeholders with absolute execution-host paths')
    result = Path(value).resolve(strict=True)
    valid = result.is_dir() if directory else result.is_file()
    if not valid:
        raise ValueError('Expected ' + ('directory: ' if directory else 'file: ') + str(result))
    return result


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Output exists; choose a fresh file: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', required=True, type=Path,
                        help='Example edited to point at installed assets; hash placeholders are allowed')
    parser.add_argument('--flux-model', required=True, type=Path,
                        help='Complete, ordinary-file Flux2KleinPipeline snapshot directory')
    parser.add_argument('--flux-receipt', required=True, type=Path,
                        help='Fresh receipt path on the execution host')
    parser.add_argument('--output', required=True, type=Path, help='Fresh pinned settings JSON')
    args = parser.parse_args()
    try:
        if args.flux_receipt.resolve() == args.output.resolve():
            raise ValueError('Receipt and settings outputs must differ')
        if args.flux_receipt.exists() or args.output.exists():
            raise FileExistsError('Choose fresh receipt and settings output files')
        options = read(args.settings)
        flux = options['panorama_preprocess']['flux']
        root = existing(str(args.flux_model.resolve()), directory=True)
        if read(root/'model_index.json').get('_class_name') != 'Flux2KleinPipeline':
            raise ValueError('Expected the declared Flux2KleinPipeline model snapshot')
        files = []
        for path in sorted(root.rglob('*')):
            relative = path.relative_to(root)
            if any(part.startswith('.') for part in relative.parts):
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError('FLUX snapshot must contain ordinary files within its directory')
            if path.is_file():
                files.append({'path': relative.as_posix(), 'bytes': path.stat().st_size,
                              'sha256': sha(path)})
        if not any(row['path'].endswith('.safetensors') for row in files):
            raise ValueError('FLUX snapshot has no safetensors weights')
        revision = flux['model_revision']
        if not isinstance(revision, str) or not revision or '<' in revision:
            raise ValueError('Record the model revision you actually installed')
        # The engine's receipt schema calls this status downloaded_verified.
        # Hashes establish local file identity; the supplied revision is operator declared.
        receipt = {'status': 'downloaded_verified', 'revision': revision,
                   'model_path': str(root), 'files': files,
                   'verification': 'local file hashes; upstream revision declared by operator'}
        brush = options['brush_refine']
        for path_key, hash_key in (('brush_binary', 'brush_sha256'),
                                   ('renderer_library', 'renderer_library_sha256')):
            brush[hash_key] = sha(existing(brush[path_key]))
        sys.path.insert(0, str(ROOT))
        from tools.streetview_engine.object_verifier import verifier_provenance
        verifier = options['panorama_preprocess']['sam_segmentation']['instance_verifier']
        verifier.pop('expected_files_sha256', None)
        verifier['expected_files_sha256'] = verifier_provenance(verifier)['files_sha256']
        for mode in ('pose', 'metric'):
            spec = options['sky_depth'][mode]
            model = existing(spec['model_path'], directory=True)
            repo = existing(spec['repo_path'], directory=True)
            commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
            if commit != spec['repo_revision']:
                raise ValueError('Configured DA3 ' + mode + ' revision differs from checkout')
            if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--',
                                        'src/depth_anything_3'], text=True).strip():
                raise ValueError('DA3 source tree must be unmodified')
            if read(model/'config.json').get('model_name') != spec['model_name']:
                raise ValueError('DA3 model configuration differs: ' + mode)
            spec['model_sha256'] = sha(model/'model.safetensors')
            spec['config_sha256'] = sha(model/'config.json')
        write_new(args.flux_receipt, receipt)
        flux['model_receipt_path'] = str(args.flux_receipt.resolve())
        flux['model_receipt_sha256'] = sha(args.flux_receipt)
        write_new(args.output, options)
        print(json.dumps({'status': 'pinned_local_assets', 'settings': str(args.output.resolve()),
                          'flux_files': len(files), 'models_loaded': False,
                          'network_used': False, 'runtime_compatibility_checked': False}, indent=2))
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
