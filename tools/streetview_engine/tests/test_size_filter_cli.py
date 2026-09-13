"""Exercise the public CPU CLI with real PLY/camera files, without training."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tools.streetview_engine import export
from tools.streetview_engine.tests.test_size_filter import fixture


WORKSPACE = Path(__file__).resolve().parents[3]


def invoke(*arguments):
    return subprocess.run(
        [sys.executable, '-B', '-m', 'tools.streetview_engine.size_filter', *map(str, arguments)],
        cwd=WORKSPACE, capture_output=True, text=True, encoding='utf8', timeout=30,
    )


def completed(result):
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert result.stderr == ''
    lines = result.stdout.splitlines()
    assert len(lines) == 1
    report = json.loads(lines[0])
    assert report['status'] == 'completed'
    return report


def failed(result, *, structured=True):
    assert result.returncode == 2, (result.stdout, result.stderr)
    assert result.stdout == ''
    if not structured:
        assert 'usage:' in result.stderr and 'error:' in result.stderr
        assert 'Traceback' not in result.stderr
        return
    lines = result.stderr.splitlines()
    assert len(lines) == 1
    error = json.loads(lines[0])
    assert error['status'] == 'failed'
    assert error['error']
    return error


def test_flags_and_relative_config_produce_identical_exact_rows(tmp_path):
    ply, camera = fixture(tmp_path / 'input')
    original = ply.read_bytes()
    direct = completed(invoke('--input', ply, '--cameras', camera, '--output-dir', tmp_path / 'flags'))
    config = tmp_path / 'filter.json'
    config.write_text(json.dumps(dict(
        source_ply='input/source.ply', camera_json='input/cameras.json',
        max_sigma_camera_radius_ratio=.5,
    )), encoding='utf8')
    configured = completed(invoke('--config', config, '--output-dir', tmp_path / 'config'))
    assert direct['removed_rows'] == configured['removed_rows'] == 2
    assert direct['remaining_rows'] == configured['remaining_rows'] == 2
    assert direct['artifact']['sha256'] == configured['artifact']['sha256']
    output = Path(direct['artifact']['path']).read_bytes()
    assert output == Path(configured['artifact']['path']).read_bytes()
    start = original.index(b'end_header\n') + len(b'end_header\n')
    out_start = output.index(b'end_header\n') + len(b'end_header\n')
    rows = np.frombuffer(original[start:], dtype='u1').reshape(4, -1)
    assert output[out_start:] == rows[[0, 3]].tobytes()
    assert ply.read_bytes() == original


def test_ratio_flag_overrides_config(tmp_path):
    ply, camera = fixture(tmp_path / 'input')
    config = tmp_path / 'filter.json'
    config.write_text(json.dumps(dict(
        source_ply=str(ply), camera_json=str(camera), max_sigma_camera_radius_ratio=.5,
    )), encoding='utf8')
    report = completed(invoke('--config', config, '--output-dir', tmp_path / 'out', '--ratio', '.8'))
    assert report['removed_rows'] == 1 and report['remaining_rows'] == 3
    assert report['options']['max_sigma_camera_radius_ratio'] == .8


@pytest.mark.parametrize('ratio', ['0', 'nan', '11', 'invalid'])
def test_invalid_ratio_fails_without_output(tmp_path, ratio):
    ply, camera = fixture(tmp_path / 'input')
    out = tmp_path / 'out'
    failed(invoke('--input', ply, '--cameras', camera, '--output-dir', out, '--ratio', ratio),
           structured=ratio != 'invalid')
    assert not out.exists()


def test_existing_output_is_never_overwritten(tmp_path):
    ply, camera = fixture(tmp_path / 'input')
    out = tmp_path / 'out'
    first = completed(invoke('--input', ply, '--cameras', camera, '--output-dir', out))
    before = {path.relative_to(out): path.read_bytes() for path in out.rglob('*') if path.is_file()}
    failed(invoke('--input', ply, '--cameras', camera, '--output-dir', out, '--ratio', '.8'))
    after = {path.relative_to(out): path.read_bytes() for path in out.rglob('*') if path.is_file()}
    assert before == after
    assert export.sha256(Path(first['artifact']['path'])) == first['artifact']['sha256']


@pytest.mark.parametrize('extra_flag', ['--input', '--cameras'])
def test_config_cannot_be_mixed_with_input_flags(tmp_path, extra_flag):
    ply, camera = fixture(tmp_path / 'input')
    config = tmp_path / 'filter.json'
    config.write_text(json.dumps(dict(source_ply=str(ply), camera_json=str(camera),
                                     max_sigma_camera_radius_ratio=.5)), encoding='utf8')
    out = tmp_path / 'out'
    failed(invoke('--config', config, '--output-dir', out,
                  extra_flag, ply if extra_flag == '--input' else camera), structured=False)
    assert not out.exists()


def test_missing_input_and_malformed_config_return_machine_readable_failure(tmp_path):
    out = tmp_path / 'missing-out'
    failed(invoke('--input', tmp_path / 'missing.ply', '--cameras', tmp_path / 'missing.json',
                  '--output-dir', out))
    assert not out.exists()
    config = tmp_path / 'broken.json'
    config.write_text('{broken', encoding='utf8')
    failed(invoke('--config', config, '--output-dir', tmp_path / 'malformed-out'))
    assert not (tmp_path / 'malformed-out').exists()
