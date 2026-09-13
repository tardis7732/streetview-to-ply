"""Exact-row output cropping, without training.

The optional cylinder is centered on equally weighted physical camera stations.
Its height is unlimited; it clips Gaussian centers, not their covariance support.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import shutil
import sys

import numpy as np

from .size_filter import (_camera_reference, _filtered_header, _json_sha, _ply,
                         _selection, _sha, read_size_filter_options,
                         run_size_filter, verify_size_filter_report)

POLICY = 'size_cap_and_camera_centered_cylinder_v1'


def read_crop_options(config):
    if not isinstance(config, dict):
        raise ValueError('Cleanup configuration must be an object')
    value = config.get('crop', {})
    if not isinstance(value, dict) or set(value) - {'enabled', 'radius_camera_radius_ratio'}:
        raise ValueError('Invalid crop fields')
    enabled, ratio = value.get('enabled', False), value.get('radius_camera_radius_ratio', 1.)
    if type(enabled) is not bool:
        raise ValueError('crop.enabled must be a boolean')
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or not 1e-4 <= ratio <= 100:
        raise ValueError('Crop radius ratio must be finite and between 0.0001 and 100')
    return dict(enabled=enabled, radius_camera_radius_ratio=float(ratio))


def _unit(value):
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all() or np.linalg.norm(vector) <= 1e-10:
        raise ValueError('Invalid world-up vector')
    return vector / np.linalg.norm(vector)


def _world_up(doc, header, convention):
    # Prefer declared geometry. Never infer a ground plane or assume an axis.
    declared = []
    if 'world_up' in doc:
        declared.append(_unit(doc['world_up']))
    match = re.search(rb'world_up\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)', header)
    if match:
        declared.append(_unit([float(part) for part in match.groups()]))
    if declared:
        if any(float(vector @ declared[0]) < math.cos(math.radians(1)) for vector in declared):
            raise ValueError('PLY and camera world-up declarations differ')
        return declared[0], 'Explicit world_up declaration'
    grouped = {}
    for frame in doc['frames']:
        face = str(frame.get('face', '')).lower()
        if face not in ('u', 'up', 'd', 'down'):
            continue
        forward = np.asarray(frame['transform_matrix'], dtype=np.float64)[:3, 2]
        if convention == 'OpenGL_c2w':
            forward = -forward
        up = forward if face in ('u', 'up') else -forward
        grouped.setdefault(str(frame['station_id']), set()).add(tuple(up))
    if not grouped:
        raise ValueError('Cylinder crop needs declared world_up or labeled Up/Down camera faces')
    votes = np.array([_unit(np.asarray(sorted(grouped[key])).mean(axis=0)) for key in sorted(grouped)])
    consensus = _unit(votes.mean(axis=0))
    if np.any(votes @ consensus < math.cos(math.radians(5))):
        raise ValueError('Up/Down camera directions lack a consistent vertical axis')
    return consensus, 'Equal-weight physical-station Up/Down face direction consensus (within 5 degrees)'


def _crop_geometry(camera_json, camera, header, options):
    if not options['enabled']:
        return None
    doc = json.loads(Path(camera_json).read_text(encoding='utf-8-sig'))
    up, evidence = _world_up(doc, header, camera['camera_convention'])
    radius = camera['scene_radius'] * options['radius_camera_radius_ratio']
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('Invalid crop radius')
    return dict(center=camera['center'], world_up=up.tolist(), radius_scene_units=radius,
                up_evidence=evidence, height='unlimited', boundary='keep center at or inside radius',
                support='Gaussian centers only; Gaussian covariance can extend beyond the boundary')


def _keep(source, offset, camera, size_options, crop_geometry):
    threshold = camera['scene_radius'] * size_options['max_sigma_camera_radius_ratio']
    keep = _selection(source, offset, threshold, size_options['enabled'])
    size_removed = int((~keep).sum())
    if crop_geometry:
        fields = source['fields']
        data = np.memmap(source['path'], mode='r', dtype='<f4', offset=offset,
                         shape=(source['vertex_count'], len(fields)))
        center, up = np.array(crop_geometry['center']), np.array(crop_geometry['world_up'])
        radius = crop_geometry['radius_scene_units']
        try:
            for start in range(0, len(keep), 65536):
                stop = min(len(keep), start + 65536)
                delta = data[start:stop, [fields.index(axis) for axis in ('x','y','z')]].astype(np.float64) - center
                horizontal = delta - np.outer(delta @ up, up)
                keep[start:stop] &= np.linalg.norm(horizontal, axis=1) <= radius
        finally:
            data._mmap.close()
    if not keep.any():
        raise ValueError('Cleanup would remove all Gaussians; no empty result PLY published')
    return keep, size_removed


def _write_subset(source, header, offset, keep, target):
    """Empty comparison sets are represented by None, never an invalid PLY."""
    count = int(keep.sum())
    if not count:
        return None
    target = Path(target)
    temporary = target.with_suffix(target.suffix + '.tmp')
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
    artifact['path'] = str(target.resolve())
    return artifact


def run_ply_cleanup(source_ply, camera_json, output_dir, size_options, crop_options=None):
    size_options = read_size_filter_options({'size_filter': size_options})
    crop = read_crop_options({'crop': {} if crop_options is None else crop_options})
    if not crop['enabled']:
        return run_size_filter(source_ply, camera_json, output_dir, size_options)
    source, header, offset = _ply(source_ply)
    camera = _camera_reference(camera_json, source, header)
    geometry = _crop_geometry(camera_json, camera, header, crop)
    keep, size_removed = _keep(source, offset, camera, size_options, geometry)
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError('Cleanup output directory must be new')
    output.mkdir(parents=True)
    artifact = _write_subset(source, header, offset, keep, output/'scene.ply')
    selection_path = output/'selection.npz'
    np.savez_compressed(selection_path, removed_indices=np.flatnonzero(~keep).astype('<i8'),
                        source_vertex_count=np.array(source['vertex_count'], dtype='<i8'))
    shutil.copyfile(camera_json, output/'cameras.json')
    if 'metadata' in camera:
        shutil.copyfile(camera['metadata']['path'], output/'dataset_manifest.json')
    camera['copied_path'] = str(output/'cameras.json')
    if _sha(source_ply) != source['sha256'] or _sha(output/'cameras.json') != camera['sha256']:
        raise ValueError('Cleanup inputs changed')
    report = dict(schema_version=1, policy=POLICY, status='completed', source=source, artifact=artifact,
                  options=size_options, crop=crop, crop_geometry=geometry, camera_reference=camera,
                  threshold_sigma_scene_units=camera['scene_radius']*size_options['max_sigma_camera_radius_ratio'],
                  size_removed_rows=size_removed, crop_additional_removed_rows=int((~keep).sum())-size_removed,
                  removed_rows=int((~keep).sum()), remaining_rows=int(keep.sum()), attributes_unchanged=True,
                  selection=dict(path=str(selection_path), sha256=_sha(selection_path), source_sha256=source['sha256'],
                    artifact_sha256=artifact['sha256'], camera_sha256=camera['sha256'],
                    options_sha256=_json_sha(dict(size_filter=size_options,crop=crop))))
    verify_cleanup_report(report, source_ply, artifact['path'], camera_json, selection_path)
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    return report


def _verify_exact_rows(source, header, offset, artifact_path, keep):
    artifact, out_header, out_offset = _ply(artifact_path)
    if artifact['vertex_count'] != int(keep.sum()) or out_header != (header if keep.all() else _filtered_header(header,int(keep.sum()))):
        raise ValueError('Cleanup count or header differs')
    stride = len(source['fields'])*4
    before = np.memmap(source['path'],mode='r',dtype='u1',offset=offset,shape=(len(keep),stride))
    after = np.memmap(artifact_path,mode='r',dtype='u1',offset=out_offset,shape=(int(keep.sum()),stride))
    cursor = 0
    try:
        for start in range(0,len(keep),65536):
            rows = before[start:start+65536][keep[start:start+65536]]
            if not np.array_equal(rows,after[cursor:cursor+len(rows)]):
                raise ValueError('Cleanup changed retained Gaussian row bytes or order')
            cursor += len(rows)
    finally:
        before._mmap.close(); after._mmap.close()
    return artifact


def verify_cleanup_report(report, source_ply, filtered_ply, camera_json, selection_npz):
    if report.get('policy') != POLICY:
        return verify_size_filter_report(report, source_ply, filtered_ply, camera_json, selection_npz)
    if report.get('schema_version') != 1 or report.get('status') != 'completed':
        raise ValueError('Unsupported cleanup report')
    options = read_size_filter_options({'size_filter':report['options']})
    crop = read_crop_options({'crop':report['crop']})
    source, header, offset = _ply(source_ply)
    camera = _camera_reference(camera_json,source,header)
    geometry = _crop_geometry(camera_json,camera,header,crop)
    keep, size_removed = _keep(source,offset,camera,options,geometry)
    artifact = _verify_exact_rows(source,header,offset,filtered_ply,keep)
    for key in ('sha256','vertex_count','bytes','fields','sh_degree','format'):
        if report['source'].get(key) != source[key] or report['artifact'].get(key) != artifact[key]:
            raise ValueError('Cleanup source/artifact binding changed')
    for key in ('sha256','physical_stations','frames','scene_radius','center','coordinate_frame','units','camera_convention'):
        if report['camera_reference'].get(key) != camera[key]:
            raise ValueError('Cleanup camera reference changed')
    if report.get('crop_geometry') != geometry or report.get('threshold_sigma_scene_units') != camera['scene_radius']*options['max_sigma_camera_radius_ratio']:
        raise ValueError('Cleanup geometry changed')
    if (report.get('removed_rows') != int((~keep).sum()) or report.get('remaining_rows') != int(keep.sum())
            or report.get('size_removed_rows') != size_removed or report.get('crop_additional_removed_rows') != int((~keep).sum())-size_removed):
        raise ValueError('Cleanup counts changed')
    expected = dict(sha256=_sha(selection_npz),source_sha256=source['sha256'],artifact_sha256=artifact['sha256'],
                    camera_sha256=camera['sha256'],options_sha256=_json_sha(dict(size_filter=options,crop=crop)))
    if any(report['selection'].get(key) != value for key,value in expected.items()):
        raise ValueError('Cleanup selection binding changed')
    with np.load(selection_npz,allow_pickle=False) as arrays:
        if (set(arrays.files) != {'removed_indices','source_vertex_count'} or arrays['removed_indices'].dtype != np.dtype('<i8')
                or arrays['source_vertex_count'].shape != () or int(arrays['source_vertex_count']) != len(keep)
                or not np.array_equal(arrays['removed_indices'],np.flatnonzero(~keep))):
            raise ValueError('Cleanup row selection changed')
    return dict(status='verified',exact_retained_row_bytes=True,removed_rows=int((~keep).sum()),artifact_sha256=artifact['sha256'])


def write_output_camera_reference(report, source_camera, target):
    """Bind copied poses to unchanged-frame output, allowing later crop-only jobs."""
    if _sha(source_camera) != report['camera_reference']['sha256']:
        raise ValueError('Output camera reference source changed')
    if _sha(report['artifact']['path']) != report['artifact']['sha256']:
        raise ValueError('Output PLY changed before camera binding')
    doc = json.loads(Path(source_camera).read_text(encoding='utf-8-sig'))
    reference = report['camera_reference']
    doc.update(coordinate_frame=reference['coordinate_frame'], units=reference['units'],
               camera_convention=reference['camera_convention'],
               ply_binding=dict(sha256=report['artifact']['sha256'], coordinate_frame=reference['coordinate_frame'],units=reference['units']),
               cleanup_provenance=dict(source_camera_sha256=reference['sha256'],source_ply_sha256=report['source']['sha256'],
                 artifact_ply_sha256=report['artifact']['sha256'],unchanged_coordinate_frame=True,exact_retained_row_bytes=True))
    target = Path(target).resolve()
    if target.exists():
        raise ValueError('Output camera reference must be new')
    target.write_text(json.dumps(doc,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description='Filter/crop an existing Gaussian PLY without retraining; original preserved.')
    parser.add_argument('--config',type=Path,help='JSON with source_ply, camera_json, optional ratio, size_filter_enabled and crop')
    parser.add_argument('--input',type=Path)
    parser.add_argument('--cameras',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--ratio',type=float)
    parser.add_argument('--no-size-filter',action='store_true')
    parser.add_argument('--crop-radius-ratio',type=float,help='Enable camera-centered, unlimited-height cylinder crop')
    args = parser.parse_args(argv)
    if args.config and (args.input or args.cameras): parser.error('--config cannot be combined with --input or --cameras')
    if not args.config and not (args.input and args.cameras): parser.error('supply --input and --cameras, or --config')
    try:
        config = json.loads(args.config.read_text(encoding='utf-8-sig')) if args.config else dict(source_ply=str(args.input),camera_json=str(args.cameras))
        if not isinstance(config,dict) or set(config)-{'source_ply','camera_json','max_sigma_camera_radius_ratio','size_filter_enabled','crop'}:
            raise ValueError('Invalid cleanup config fields')
        paths = []
        for key in ('source_ply','camera_json'):
            value=config.get(key)
            if not isinstance(value,str) or not value.strip() or '\x00' in value: raise ValueError(f'Missing {key}')
            path=Path(value).expanduser()
            paths.append(args.config.resolve().parent/path if args.config and not path.is_absolute() else path)
        options=read_size_filter_options({'size_filter':dict(enabled=False if args.no_size_filter else config.get('size_filter_enabled',True),
                 max_sigma_camera_radius_ratio=args.ratio if args.ratio is not None else config.get('max_sigma_camera_radius_ratio',.5))})
        crop=read_crop_options({'crop':dict(enabled=True,radius_camera_radius_ratio=args.crop_radius_ratio) if args.crop_radius_ratio is not None else config.get('crop',{})})
        report=run_ply_cleanup(*paths,args.output_dir,options,crop)
        output_camera=write_output_camera_reference(report,args.output_dir/'cameras.json',args.output_dir/'artifact_cameras.json')
        result=dict(status='completed',artifact=report['artifact'],removed_rows=report['removed_rows'],remaining_rows=report['remaining_rows'],
                    options=options,crop=crop,training_started=False,output_camera_json=str(output_camera),report=str(args.output_dir.resolve()/'report.json'))
        print(json.dumps(result,ensure_ascii=True,allow_nan=False))
        return 0
    except (OSError,ValueError,TypeError,KeyError,OverflowError) as error:
        print(json.dumps(dict(status='failed',error=str(error)),ensure_ascii=True),file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
