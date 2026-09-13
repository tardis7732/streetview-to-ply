"""CPU-only, scale-invariant Gaussian size cap with exact row provenance.

This removes complete PLY rows; it neither retrains nor changes opacity. Camera
coordinates and log standard deviations must share the declared world frame.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import shutil

import numpy as np


DEFAULT_RATIO = 0.5
POLICY = 'max_sigma_ge_physical_camera_radius_ratio_v1'


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_size_filter_options(config):
    if not isinstance(config, dict):
        raise ValueError('Job configuration must be an object')
    value = config.get('size_filter', {})
    if not isinstance(value, dict) or set(value) - {'enabled', 'max_sigma_camera_radius_ratio'}:
        raise ValueError('Invalid size_filter fields')
    enabled = value.get('enabled', False)
    ratio = value.get('max_sigma_camera_radius_ratio', DEFAULT_RATIO)
    if type(enabled) is not bool:
        raise ValueError('size_filter.enabled must be a boolean')
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or not 1e-4 <= ratio <= 10:
        raise ValueError('Size ratio must be finite and between 0.0001 and 10')
    return dict(enabled=enabled, max_sigma_camera_radius_ratio=float(ratio))


def _ply(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError('PLY must be an ordinary existing file')
    path = path.resolve()
    lines, fields, count, format_seen = [], [], None, False
    with path.open('rb') as stream:
        if stream.readline() != b'ply\n':
            raise ValueError('Expected binary Gaussian PLY with LF header')
        lines.append(b'ply\n')
        while True:
            raw = stream.readline(8192)
            if not raw or stream.tell() > 65536 or not raw.endswith(b'\n'):
                raise ValueError('Invalid or oversized PLY header')
            lines.append(raw)
            items = raw.decode('ascii').split()
            if not items or items[0] in ('comment', 'obj_info'):
                continue
            if items == ['end_header']:
                break
            if items == ['format', 'binary_little_endian', '1.0'] and not format_seen and count is None:
                format_seen = True
            elif len(items) == 3 and items[:2] == ['element', 'vertex'] and format_seen and count is None:
                count = int(items[2])
            elif len(items) == 3 and items[0] == 'property' and items[1] in ('float', 'float32') and count is not None and items[2] not in fields:
                fields.append(items[2])
            else:
                raise ValueError('Unsupported Gaussian PLY declaration')
        offset = stream.tell()
    required = {'x', 'y', 'z', 'opacity', *(f'f_dc_{i}' for i in range(3)), *(f'scale_{i}' for i in range(3)), *(f'rot_{i}' for i in range(4))}
    rest = [name for name in fields if name.startswith('f_rest_')]
    basis = len(rest) // 3 + 1
    if not count or count < 0 or not required <= set(fields) or len(rest) % 3 or set(rest) != {f'f_rest_{i}' for i in range(len(rest))} or math.isqrt(basis) ** 2 != basis:
        raise ValueError('Missing or invalid Gaussian PLY fields')
    if path.stat().st_size != offset + count * len(fields) * 4:
        raise ValueError('PLY payload length differs from header')
    data = np.memmap(path, mode='r', dtype='<f4', offset=offset, shape=(count, len(fields)))
    try:
        for start in range(0, count, 65536):
            chunk = data[start:start + 65536]
            if not np.isfinite(chunk).all():
                raise ValueError('Nonfinite Gaussian PLY parameters')
            quat = chunk[:, [fields.index(f'rot_{i}') for i in range(4)]].astype(np.float64)
            if np.any(np.linalg.norm(quat, axis=1) < 1e-8):
                raise ValueError('Zero Gaussian quaternion')
    finally:
        data._mmap.close()
    return dict(path=str(path), sha256=_sha(path), vertex_count=count, bytes=path.stat().st_size,
                fields=fields, sh_degree=math.isqrt(basis)-1, format='binary_little_endian_gaussian_ply'), b''.join(lines), offset


def _units(value):
    return 'metres' if value in ('metres', 'meters', 'm') else value


def _camera_reference(path, source, header):
    path = Path(path).resolve()
    doc = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(doc, dict) or not isinstance(doc.get('frames'), list) or not doc['frames']:
        raise ValueError('Camera JSON requires frames')
    metadata = doc
    metadata_path = None
    if not doc.get('coordinate_frame') or not doc.get('units'):
        metadata_path = path.with_name('dataset_manifest.json')
        metadata = json.loads(metadata_path.read_text(encoding='utf-8-sig'))
        files = metadata.get('files', {})
        if _sha(path) not in files.values():
            raise ValueError('Dataset coordinate metadata is not bound to camera JSON')
    frame, units = metadata.get('coordinate_frame'), _units(metadata.get('units'))
    if not isinstance(frame, str) or not frame or units != 'metres':
        raise ValueError('Camera JSON must explicitly declare world coordinate frame and metres')
    convention = doc.get('camera_convention', metadata.get('camera_convention'))
    if convention not in ('OpenGL_c2w', 'OpenCV_c2w'):
        raise ValueError('Camera transforms must explicitly declare a c2w convention')
    declaration = re.search(rb'^comment coordinates ([A-Za-z0-9_]+) (metres|meters|m)(?:;|\s|$)', header, re.MULTILINE)
    if declaration:
        if declaration.group(1).decode() != frame or _units(declaration.group(2).decode()) != units:
            raise ValueError('PLY and camera coordinate frames differ')
        frame_evidence = 'PLY coordinate comment matches camera metadata'
    else:
        binding = doc.get('ply_binding', {})
        if binding.get('sha256') != source['sha256'] or binding.get('coordinate_frame') != frame or _units(binding.get('units')) != units:
            raise ValueError('PLY lacks coordinate metadata: camera JSON needs an explicit hash-bound ply_binding')
        frame_evidence = 'Caller-supplied hash-bound PLY coordinate declaration; not inferred from geometry'
    groups = {}
    for item in doc['frames']:
        station = item.get('station_id')
        if isinstance(station, bool) or not isinstance(station, (str, int)) or not str(station):
            raise ValueError('Camera frame lacks explicit physical station_id')
        pose = np.asarray(item.get('transform_matrix'), dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1], rtol=0, atol=1e-8):
            raise ValueError('Invalid camera-to-world transform')
        rot = pose[:3, :3]
        if not np.allclose(rot.T @ rot, np.eye(3), rtol=0, atol=1e-5) or not np.isclose(np.linalg.det(rot), 1, rtol=0, atol=1e-5):
            raise ValueError('Camera-to-world transform must be rigid')
        groups.setdefault(str(station), set()).add(tuple(pose[:3, 3]))
    if len(groups) < 2:
        raise ValueError('Size filtering needs multiple spatially separated physical stations')
    # Deduplicate face/capture copies before computing an equally weighted station center.
    centers = np.array([np.asarray(sorted(groups[key])).mean(axis=0) for key in sorted(groups)])
    center = centers.mean(axis=0)
    radius = float(np.linalg.norm(centers - center, axis=1).max())
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('Degenerate physical camera radius')
    tolerance = radius * 1e-6
    for index, key in enumerate(sorted(groups)):
        if np.linalg.norm(np.asarray(list(groups[key])) - centers[index], axis=1).max() > tolerance:
            raise ValueError('Inconsistent translations within a physical station')
    for index in range(len(centers)):
        if np.any(np.linalg.norm(centers[index+1:] - centers[index], axis=1) <= tolerance):
            raise ValueError('Coincident physical stations must share a station_id')
    result = dict(path=str(path), sha256=_sha(path), physical_stations=len(centers), frames=len(doc['frames']),
                  scene_radius=radius, center=center.tolist(), coordinate_frame=frame, units=units,
                  camera_convention=convention, coordinate_evidence=frame_evidence,
                  grouping='deduplicate identical centers per explicit station_id, equal weight per physical station')
    if metadata_path is not None:
        result['metadata'] = dict(path=str(metadata_path), sha256=_sha(metadata_path))
    return result


def _selection(source, offset, threshold, enabled):
    count, fields = source['vertex_count'], source['fields']
    data = np.memmap(source['path'], mode='r', dtype='<f4', offset=offset, shape=(count, len(fields)))
    keep = np.ones(count, dtype=bool)
    try:
        if enabled:
            log_threshold = math.log(threshold)
            columns = [fields.index(f'scale_{i}') for i in range(3)]
            for start in range(0, count, 65536):
                stop = min(count, start + 65536)
                keep[start:stop] = data[start:stop, columns].astype(np.float64).max(axis=1) < log_threshold
    finally:
        data._mmap.close()
    if not keep.any():
        raise ValueError('Size filter would remove all Gaussians; no empty PLY published')
    return keep


def _filtered_header(header, count):
    result, substitutions = re.subn(rb'(?m)^element vertex [0-9]+\r?$', f'element vertex {count}'.encode(), header)
    if substitutions != 1:
        raise ValueError('Expected one vertex declaration')
    return result


def run_size_filter(source_ply, camera_json, output_dir, options):
    """Write a fresh exact-row subset and replayable source/selection binding."""
    options = read_size_filter_options({'size_filter': options})
    source, header, offset = _ply(source_ply)
    camera = _camera_reference(camera_json, source, header)
    threshold = camera['scene_radius'] * options['max_sigma_camera_radius_ratio']
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError('Nonfinite size threshold')
    keep = _selection(source, offset, threshold, options['enabled'])
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError('Size-filter output directory must be new')
    output.mkdir(parents=True)
    target, temporary = output / 'scene.ply', output / 'scene.ply.tmp'
    count = int(keep.sum())
    if count == source['vertex_count']:
        shutil.copyfile(source['path'], temporary)
    else:
        data = np.memmap(source['path'], mode='r', dtype='u1', offset=offset,
                         shape=(source['vertex_count'], len(source['fields']) * 4))
        try:
            with temporary.open('wb') as stream:
                stream.write(_filtered_header(header, count))
                for start in range(0, len(keep), 65536):
                    data[start:start+65536][keep[start:start+65536]].tofile(stream)
        finally:
            data._mmap.close()
    artifact, _, _ = _ply(temporary)
    temporary.replace(target)
    artifact['path'] = str(target)
    if _sha(source['path']) != source['sha256'] or _sha(camera_json) != camera['sha256']:
        raise ValueError('Input changed during filtering')
    copied_camera = output / 'cameras.json'
    shutil.copyfile(camera_json, copied_camera)
    if _sha(copied_camera) != camera['sha256']:
        raise ValueError('Copied camera reference changed')
    if 'metadata' in camera:
        shutil.copyfile(camera['metadata']['path'], output / 'dataset_manifest.json')
        if _sha(output / 'dataset_manifest.json') != camera['metadata']['sha256']:
            raise ValueError('Copied camera coordinate metadata changed')
    selection_path = output / 'selection.npz'
    np.savez_compressed(selection_path, removed_indices=np.flatnonzero(~keep).astype('<i8'),
                        source_vertex_count=np.array(source['vertex_count'], dtype='<i8'))
    selection = dict(path=str(selection_path), sha256=_sha(selection_path), source_sha256=source['sha256'],
                     artifact_sha256=artifact['sha256'], camera_sha256=camera['sha256'], options_sha256=_json_sha(options),
                     encoding='sorted zero-based int64 removed_indices; exact complement retained in original order')
    camera['copied_path'] = str(copied_camera)
    report = dict(schema_version=1, policy=POLICY, status='completed', source=source, artifact=artifact,
                  options=options, camera_reference=camera, threshold_sigma_scene_units=threshold,
                  removed_rows=source['vertex_count']-count, remaining_rows=count, selection=selection,
                  attributes_unchanged=True, preservation='All retained row bytes and order exact; header differs only in vertex count',
                  exemptions='None: size cap includes rows previously protected by other filters',
                  quality='Serialization and removal rule verified; visual quality not evaluated by this filter')
    verify_size_filter_report(report, source['path'], target, camera_json, selection_path)
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    return report


def verify_size_filter_report(report, source_ply, filtered_ply, camera_json, selection_npz):
    """Independently replay the policy and compare every retained source byte."""
    if report.get('schema_version') != 1 or report.get('status') != 'completed' or report.get('policy') != POLICY:
        raise ValueError('Unsupported size-filter report')
    options = read_size_filter_options({'size_filter': report.get('options')})
    source, header, offset = _ply(source_ply)
    artifact, out_header, out_offset = _ply(filtered_ply)
    camera = _camera_reference(camera_json, source, header)
    threshold = camera['scene_radius'] * options['max_sigma_camera_radius_ratio']
    if report.get('threshold_sigma_scene_units') != threshold:
        raise ValueError('Size-filter threshold differs from bound cameras')
    for field in ('sha256', 'vertex_count', 'bytes', 'fields', 'sh_degree', 'format'):
        if report.get('source', {}).get(field) != source[field] or report.get('artifact', {}).get(field) != artifact[field]:
            raise ValueError('Size-filter source or artifact binding differs')
    for field in ('sha256', 'physical_stations', 'frames', 'scene_radius', 'center', 'coordinate_frame', 'units', 'camera_convention'):
        if report.get('camera_reference', {}).get(field) != camera[field]:
            raise ValueError('Size-filter camera reference differs')
    selection = report.get('selection', {})
    expected = dict(sha256=_sha(selection_npz), source_sha256=source['sha256'], artifact_sha256=artifact['sha256'],
                    camera_sha256=camera['sha256'], options_sha256=_json_sha(options))
    if any(selection.get(k) != v for k, v in expected.items()):
        raise ValueError('Size-filter selection provenance differs')
    keep = _selection(source, offset, threshold, options['enabled'])
    removed = np.flatnonzero(~keep).astype('<i8')
    with np.load(selection_npz, allow_pickle=False) as arrays:
        if set(arrays.files) != {'removed_indices', 'source_vertex_count'} or arrays['removed_indices'].dtype != np.dtype('<i8') or arrays['source_vertex_count'].shape != () or int(arrays['source_vertex_count']) != source['vertex_count'] or not np.array_equal(arrays['removed_indices'], removed):
            raise ValueError('Size-filter row selection differs from recomputed rule')
    if report.get('removed_rows') != len(removed) or report.get('remaining_rows') != int(keep.sum()) or artifact['vertex_count'] != int(keep.sum()):
        raise ValueError('Size-filter counts differ')
    expected_header = header if not len(removed) else _filtered_header(header, int(keep.sum()))
    if out_header != expected_header:
        raise ValueError('Size-filter changed PLY header beyond vertex count')
    stride = len(source['fields']) * 4
    before = np.memmap(source_ply, mode='r', dtype='u1', offset=offset, shape=(source['vertex_count'], stride))
    after = np.memmap(filtered_ply, mode='r', dtype='u1', offset=out_offset, shape=(artifact['vertex_count'], stride))
    cursor = 0
    try:
        for start in range(0, len(keep), 65536):
            expected_rows = before[start:start+65536][keep[start:start+65536]]
            if not np.array_equal(expected_rows, after[cursor:cursor+len(expected_rows)]):
                raise ValueError('Filtered PLY is not the exact retained-row subset')
            cursor += len(expected_rows)
    finally:
        before._mmap.close(); after._mmap.close()
    return dict(status='verified', exact_retained_row_bytes=True, removed_rows=len(removed), artifact_sha256=artifact['sha256'])


def verify_filtered_export_lineage(directory, final_ply, config):
    """Shared local/cloud gate for an explicitly requested filtered export."""
    root = Path(directory).resolve()

    def inside(relative):
        target = (root / relative).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError('Missing or escaped filtered export provenance')
        return target

    def read(relative):
        return json.loads(inside(relative).read_text(encoding='utf8'))

    options = read_size_filter_options(config)
    if not options['enabled']:
        raise ValueError('Exported size filter differs from requested configuration')
    exported, training = read('export/report.json'), read('training/manifest.json')
    if exported.get('status') != 'completed' or training.get('status') != 'completed':
        raise ValueError('Training or filtered export manifest is incomplete')
    record = exported.get('size_filter')
    if not isinstance(record, dict) or record.get('status') != 'completed' or record.get('options') != options:
        raise ValueError('Exported size filter differs from requested configuration')
    required_paths = dict(report='export/size_filter/report.json', source_path='export/source.ply',
                          camera_path='export/size_filter/cameras.json', selection_path='export/size_filter/selection.npz')
    if any(record.get(key) != value for key, value in required_paths.items()):
        raise ValueError('Unexpected size-filter provenance paths')
    if _sha(inside(record['report'])) != record.get('sha256'):
        raise ValueError('Size-filter report changed')
    report = read(record['report'])
    accepted_sha = training.get('selection', {}).get('accepted_model_sha256')
    if report.get('source', {}).get('sha256') != accepted_sha or exported.get('source', {}).get('sha256') != accepted_sha or exported.get('source', {}).get('path') != 'export/source.ply':
        raise ValueError('Size-filter source differs from evaluated accepted model')
    if report.get('options') != options or report.get('artifact', {}).get('sha256') != exported.get('artifact', {}).get('sha256') or exported.get('artifact', {}).get('path') != final_ply:
        raise ValueError('Size-filter artifact or options differ')
    if exported.get('selection') != training.get('selection') or exported.get('training_manifest_sha256') != _sha(inside('training/manifest.json')):
        raise ValueError('Filtered export is bound to another accepted training selection')
    if exported.get('quality') is not None or not exported.get('quality_scope') or exported.get('source_quality') != training.get('quality'):
        raise ValueError('Training quality must be scoped to the unfiltered source')
    verified = verify_size_filter_report(report, inside(record['source_path']), inside(final_ply),
                                        inside(record['camera_path']), inside(record['selection_path']))
    if any(record.get(key) != report.get(key) for key in ('removed_rows', 'remaining_rows')):
        raise ValueError('Size-filter summary counts differ')
    if training.get('provenance', {}).get('inputs', {}).get('transforms_train.json') != report['camera_reference']['sha256']:
        raise ValueError('Size filter camera radius is not bound to training cameras')
    return verified


def main(argv=None):
    """Explicit standalone CLI; uses exactly the GUI/export filtering function."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Remove oversized Gaussians from an existing PLY without retraining.')
    parser.add_argument('--input', type=Path, help='Source Gaussian PLY (preserved)')
    parser.add_argument('--cameras', type=Path, help='Registered camera JSON in the same world coordinate frame')
    parser.add_argument('--config', type=Path, help='JSON with source_ply, camera_json and optional max_sigma_camera_radius_ratio')
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory for scene.ply and verification records')
    parser.add_argument('--ratio', type=float, help='Maximum sigma / physical camera radius; default 0.5 (GUI 50%%)')
    args = parser.parse_args(argv)
    if args.config and (args.input is not None or args.cameras is not None):
        parser.error('--config cannot be combined with --input or --cameras')
    if not args.config and (args.input is None or args.cameras is None):
        parser.error('supply --input and --cameras, or --config')
    try:
        if args.config:
            config_path = args.config.expanduser().resolve()
            config = json.loads(config_path.read_text(encoding='utf-8-sig'))
            if not isinstance(config, dict) or set(config) - {'source_ply', 'camera_json', 'max_sigma_camera_radius_ratio'}:
                raise ValueError('Invalid filter config fields')
            paths = []
            for key in ('source_ply', 'camera_json'):
                value = config.get(key)
                if not isinstance(value, str) or not value.strip() or '\x00' in value:
                    raise ValueError(f'Filter config requires a nonempty {key} path')
                path = Path(value).expanduser()
                paths.append(path if path.is_absolute() else config_path.parent / path)
            source, cameras = paths
            ratio = config.get('max_sigma_camera_radius_ratio', DEFAULT_RATIO)
        else:
            source, cameras = args.input.expanduser(), args.cameras.expanduser()
            ratio = DEFAULT_RATIO
        if args.ratio is not None:
            ratio = args.ratio
        options = read_size_filter_options({'size_filter': dict(enabled=True, max_sigma_camera_radius_ratio=ratio)})
        output = args.output_dir.expanduser().resolve()
        if output.exists():
            raise ValueError('Size-filter output directory must be new')
        report = run_size_filter(source, cameras, output, options)
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
        print(json.dumps(dict(status='failed', error=str(error)), ensure_ascii=True), file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        print(json.dumps(dict(status='interrupted', error='Interrupted by user')), file=sys.stderr, flush=True)
        return 130
    print(json.dumps(dict(status='completed', artifact=report['artifact'],
        source_ply=report['source']['path'], source_sha256=report['source']['sha256'],
        removed_rows=report['removed_rows'], remaining_rows=report['remaining_rows'], options=report['options'],
        threshold_sigma_scene_units=report['threshold_sigma_scene_units'], report=str(output / 'report.json'),
        exact_retained_row_bytes=True, training_started=False), ensure_ascii=True, allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
