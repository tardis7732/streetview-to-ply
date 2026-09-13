"""Check tracked source files before publishing; never print a secret value."""
from __future__ import annotations
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BLOCKED_SUFFIXES = {'.ply', '.splat', '.pt', '.pth', '.ckpt', '.safetensors', '.npz',
    '.npy', '.mp4', '.webm', '.mov', '.tar', '.zip', '.pem', '.key', '.log'}
BLOCKED_PARTS = {'Reference', 'data', '.local', 'weights', 'models', 'checkpoints', '.venv',
    '__pycache__', '.ssh', '.codex', '.agents'}
SECRET_PATTERNS = {
    'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
    'github_token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})'),
    'model_hub_token': re.compile(r'\bhf_[A-Za-z0-9]{25,}'),
    'api_secret': re.compile(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{30,}'),
    'aws_access_key': re.compile(r'\bAKIA[0-9A-Z]{16}\b'),
}


def main():
    raw = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT)
    files = [Path(p) for p in raw.decode('utf-8').split('\0') if p]
    if not files:
        raise SystemExit('No tracked files to check; stage the intended source files first.')
    findings = []
    total = 0
    for relative in files:
        path = ROOT / relative
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            findings.append(dict(file=relative.as_posix(), reason='external_or_symlink'))
            continue
        if relative.suffix.lower() in BLOCKED_SUFFIXES or BLOCKED_PARTS.intersection(relative.parts):
            findings.append(dict(file=relative.as_posix(), reason='non_source_or_local_data'))
        if path.stat().st_size > 2_000_000:
            findings.append(dict(file=relative.as_posix(), reason='unexpected_large_file'))
        total += path.stat().st_size
        try:
            source = path.read_text(encoding='utf-8-sig')
        except UnicodeError:
            findings.append(dict(file=relative.as_posix(), reason='unexpected_binary'))
            continue
        for name, pattern in SECRET_PATTERNS.items():
            for match in pattern.finditer(source):
                findings.append(dict(file=relative.as_posix(), line=source.count('\n', 0, match.start()) + 1,
                                     reason=name))
    print(json.dumps(dict(files=len(files), bytes=total, findings=findings,
                         status='passed' if not findings else 'failed'), indent=2))
    return bool(findings)


if __name__ == '__main__':
    raise SystemExit(main())
