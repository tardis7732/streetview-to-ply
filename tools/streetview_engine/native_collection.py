"""Import an explicit, hash-bound cube source package without network or resampling.

This is an operator-owned, per-job source. It is not a portable recipe default.
Earlier crop/resize/encoding history is retained rather than called a new native
provider download. Model masks are not imported; only declared coverage validity.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import uuid

import numpy as np
from PIL import Image

from .imaging import FACES, distance_m, fingerprint, group_physical_stations, inside, sha256, write_json


def _ordinary(path):
    path = Path(path).absolute()
    for current in (path, *path.parents):
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('Native source forbids symlinks and junctions')
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError('Native source must be an ordinary file')
    return path


def _hash(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
        raise ValueError('Native source requires a lowercase SHA256')
    return value


def _image(path, width, height, *, binary=False, decoded=None):
    if any(type(n) is not int or not 1 <= n <= 65536 for n in (width, height)):
        raise ValueError('Invalid declared source image size')
    with Image.open(path) as image:
        if image.format not in ('JPEG', 'PNG') or image.size != (width, height) or image.getexif().get(274, 1) != 1:
            raise ValueError('Source image format, size or EXIF orientation differs')
        image.load()
        if binary:
            pixels = np.asarray(image)
            if pixels.shape != (height, width) or not np.isin(pixels, [0, 255]).all():
                raise ValueError('Coverage validity must be one-channel 0/255 pixels')
        else:
            if image.mode != 'RGB':
                raise ValueError('Source cube/strip must already be RGB')
            pixels = np.asarray(image)
            if decoded is not None and hashlib.sha256(pixels.tobytes()).hexdigest() != _hash(decoded):
                raise ValueError('Decoded source RGB hash differs')


def validate_source(config, job_dir, settings):
    """Read-only preflight of every source byte and its frozen capture identity."""
    from .collection import _validate_selection
    ids, frozen = _validate_selection(config)
    if settings.get('input_cache') is not None:
        raise ValueError('Choose either native source import or completed input-cache reuse')
    options = settings.get('collection', {})
    source = options.get('native_source')
    if not isinstance(source, dict) or set(source) != {'manifest_path', 'manifest_sha256'}:
        raise ValueError('Native source needs only manifest_path and manifest_sha256')
    name = source['manifest_path']
    if not isinstance(name, str) or not name or any(c in name for c in '\x00\r\n'):
        raise ValueError('Invalid native source manifest path')
    manifest_path = Path(name)
    if not manifest_path.is_absolute():
        inside(job_dir, name)  # Verify confinement without hiding a link's original path.
        manifest_path = Path(job_dir).resolve()/name
    manifest_path = _ordinary(manifest_path)
    expected_hash = _hash(source['manifest_sha256'])
    if sha256(manifest_path) != expected_hash:
        raise ValueError('Native source manifest hash differs')
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    if (manifest.get('schema_version') != 1 or isinstance(manifest.get('schema_version'), bool)
            or manifest.get('kind') != 'native_cube_source' or manifest.get('status') != 'complete'
            or manifest.get('provider') != 'naver' or manifest.get('face_order') != list(FACES)):
        raise ValueError('Unsupported native source package contract')
    stations = manifest.get('stations')
    if not isinstance(stations, list) or any(not isinstance(row, dict) for row in stations) or [row.get('pano_id') for row in stations] != ids:
        raise ValueError('Native source panorama IDs/order differ from frozen selection')
    tolerance = float(options.get('colocation_tolerance_m', .25))
    files = {}; converted = []; face_paths = set(); native_sizes = set(); acquisition_sizes = set()

    def artifact(record, target):
        if not isinstance(record, dict):
            raise ValueError('Source file descriptor must be an object')
        relative = record.get('file_path')
        if not isinstance(relative, str) or not relative or '\\' in relative or ':' in relative:
            raise ValueError('Source files require portable relative paths')
        parsed = PurePosixPath(relative)
        if parsed.is_absolute() or '..' in parsed.parts:
            raise ValueError('Source file escapes its manifest directory')
        inside(root, relative)
        path = _ordinary(root/relative)
        digest = _hash(record.get('sha256'))
        size = record.get('bytes')
        if type(size) is not int or size <= 0 or path.stat().st_size != size or sha256(path) != digest:
            raise ValueError('Native source file size/hash differs: ' + relative)
        if target in files:
            raise ValueError('Duplicate native collection destination')
        files[target] = dict(path=path, sha256=digest, bytes=size, source_file_path=relative)
        return path, dict(file_path=target, sha256=digest, bytes=size)

    for row in stations:
        key = row['pano_id']; token = 'pano_' + fingerprint(key)[:20]
        metadata_path, metadata_record = artifact(row.get('metadata_source'), f'collection/sources/{token}/metadata.json')
        raw = json.loads(metadata_path.read_text(encoding='utf8'))
        if raw.get('id') != key or raw.get('proj_type') not in ('cubic', 'equirect'):
            raise ValueError('Source provider metadata ID/projection differs')
        captured = (raw.get('info') or {}).get('photodate')
        if not isinstance(captured, str) or not captured or row.get('captured_at') != captured:
            raise ValueError('Source capture text differs from provider metadata')
        selected_date = frozen[key].get('captured_at') or frozen[key].get('capture_date')
        if selected_date and not captured.startswith(str(selected_date)):
            raise ValueError('Source capture text differs from frozen selection')
        coordinates = dict(lat=raw.get('latitude'), lng=raw.get('longitude'))
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in coordinates.values()):
            raise ValueError('Provider metadata coordinates must be finite numbers')
        distance_m(coordinates, coordinates)
        if any(isinstance(row.get(k), bool) or not isinstance(row.get(k), (int, float))
               or not math.isfinite(row[k]) or abs(row[k]-coordinates[k]) > 1e-9 for k in ('lat', 'lng')):
            raise ValueError('Source coordinates differ from provider metadata')
        if distance_m(coordinates, frozen[key]) > .5:
            raise ValueError('Source position differs from frozen capture position')
        angles = raw.get('camera_angle')
        if not isinstance(angles, list) or len(angles) != 3 or any(type(a) not in (int, float) or not math.isfinite(a) for a in angles):
            raise ValueError('Provider camera angles must be finite')
        history = row.get('image_history')
        if not isinstance(history, list) or not history or any(not isinstance(item, str) or not item.strip() for item in history):
            raise ValueError('Explicit source image processing history is required')
        station = dict(frozen[key], id=key, pano_id=key, **coordinates,
            camera_angle=angles, projection=raw['proj_type'], provider_altitude=raw.get('altitude'),
            captured_at=captured, metadata_source=metadata_record, faces={}, image_history=list(history))
        if row.get('raw_strip') is not None:
            strip = row['raw_strip']
            strip_path, strip_record = artifact(strip, f'collection/sources/{token}/raw_strip' + Path(strip['file_path']).suffix.lower())
            if strip.get('w') != 6*strip.get('h', 0) or len(strip.get('face_order', [])) != 6 or set(strip['face_order']) != set(FACES):
                raise ValueError('Declared raw strip must have six square faces and a complete face order')
            _image(strip_path, strip.get('w'), strip.get('h'))
            acquisition_sizes.add(strip['h'])
            station['raw_strip_source'] = dict(strip_record, w=strip['w'], h=strip['h'], face_order=strip['face_order'])
        faces = row.get('faces')
        if not isinstance(faces, dict) or set(faces) != set(FACES):
            raise ValueError('Every source station requires all six cube faces')
        sizes = set()
        for face in FACES:
            item = faces[face]
            if not isinstance(item, dict) or item.get('w') != item.get('h'):
                raise ValueError('Source cube faces must be square')
            relative = item.get('file_path')
            if not isinstance(relative, str) or relative in face_paths:
                raise ValueError('Each source cube face needs a distinct file')
            face_paths.add(relative)
            path, record = artifact(item, f'collection/images/{token}_{face}' + Path(relative).suffix.lower())
            _image(path, item.get('w'), item.get('h'), decoded=item.get('decoded_rgb_sha256'))
            _hash(item.get('decoded_rgb_sha256'))
            sizes.add(item['w']); native_sizes.add(item['w'])
            record.update(w=item['w'], h=item['h'], decoded_rgb_sha256=item['decoded_rgb_sha256'], tiles=[],
                color_processing='byte-identical cached source copy; no import resampling or color edits', image_history=list(history))
            if item.get('valid_mask') is not None:
                validity = item['valid_mask']
                valid_path, valid_record = artifact(validity, f'collection/coverage/{token}_{face}.png')
                if (validity.get('w'), validity.get('h')) != (item['w'], item['h']):
                    raise ValueError('Coverage validity size differs from source cube')
                _image(valid_path, item['w'], item['h'], binary=True)
                provenance = item.get('valid_mask_provenance')
                if not isinstance(provenance, str) or not provenance.strip():
                    raise ValueError('Coverage validity needs explicit provenance; semantic masks are not source coverage')
                record.update(valid_mask_path=valid_record['file_path'], valid_mask_sha256=valid_record['sha256'],
                    valid_mask_provenance=provenance)
            station['faces'][face] = record
        if len(sizes) != 1:
            raise ValueError('All six source faces in a panorama must have the same resolution')
        converted.append(station)
    assignment = group_physical_stations(converted, tolerance)
    for station in converted:
        station['station_id'] = assignment[station['pano_id']]
    signature = fingerprint(dict(schema_version=1, kind='native_cube_import', config=config,
        source_manifest_sha256=expected_hash, colocation_tolerance_m=tolerance))
    files['collection/import/source_manifest.json'] = dict(path=manifest_path, sha256=expected_hash,
        bytes=manifest_path.stat().st_size, source_file_path=manifest_path.name)
    output = dict(schema_version=1, status='complete', stage='collect', provider='naver', input_fingerprint=signature,
        face_order=list(FACES), native_face_size=next(iter(native_sizes)) if len(native_sizes)==1 else None,
        native_face_sizes=sorted(native_sizes), original_strip_face_sizes=sorted(acquisition_sizes),
        source_projection='cached_cube_faces_with_declared_history', source_kind='native_cube_import',
        station_frame='front_camera_opencv_X_right_Y_down_Z_forward', stations=converted,
        physical_station_count=len(set(assignment.values())), panorama_count=len(converted), colocation_tolerance_m=tolerance,
        physical_grouping='complete-link horizontal provider-GPS tolerance; evidence groups only; separate pano_id rigs retain separate camera poses',
        source_images_preserved=True, import_resampling=False, import_color_processing=False, approximate_gps_not_surveyed=True,
        native_source_manifest=dict(file_path='collection/import/source_manifest.json', sha256=expected_hash),
        input_files={name: value['sha256'] for name,value in files.items()},
        transport=dict(network_requests=0, mode='explicit verified cached source copy'))
    return dict(manifest=output, files=files)


def _copy(source, target, digest):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        _ordinary(target)
        if sha256(target) != digest:
            raise ValueError('Existing imported artifact differs; use a new job directory')
        return
    temporary = target.with_name(target.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with _ordinary(source).open('rb') as left, temporary.open('xb') as right:
            shutil.copyfileobj(left, right, 1024*1024)
        if sha256(temporary) != digest:
            raise ValueError('Native source changed during import')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def run(config, job_dir, settings):
    checked = validate_source(config, job_dir, settings)
    root = Path(job_dir).resolve(); manifest = checked['manifest']
    destination = inside(root, 'collection/manifest.json')
    if destination.exists():
        prior = json.loads(destination.read_text(encoding='utf8'))
        if prior != manifest:
            raise ValueError('Existing collection differs from verified native source/configuration')
    for name, record in checked['files'].items():
        _copy(record['path'], inside(root, name), record['sha256'])
    write_json(inside(root, 'collection/input.json'), dict(input_fingerprint=manifest['input_fingerprint'], config=config,
        colocation_tolerance_m=manifest['colocation_tolerance_m'], native_source_manifest_sha256=manifest['native_source_manifest']['sha256']))
    write_json(destination, manifest)
    return manifest
