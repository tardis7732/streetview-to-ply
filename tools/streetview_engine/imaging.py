"""Reusable image provenance, native cube camera frames and capture grouping."""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import uuid

import numpy as np
from PIL import Image

FACES = ('F', 'R', 'B', 'L', 'U', 'D')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')).hexdigest()


def inside(root, relative):
    root = Path(root).resolve()
    name = Path(relative)
    if name.is_absolute() or '..' in name.parts:
        raise ValueError('Artifact paths must be relative to the job directory')
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Artifact path escapes the job directory')
    return path


def write_bytes(path, content, *, immutable=True):
    """Atomic, idempotent writes; an existing different artifact is an error."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if immutable and path.read_bytes() != content:
            raise ValueError(f'Existing artifact differs: {path.name}')
        if immutable:
            return
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_bytes(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value, *, immutable=True):
    write_bytes(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode('utf8'), immutable=immutable)


def png_bytes(pixels):
    output = BytesIO()
    Image.fromarray(np.asarray(pixels)).save(output, format='PNG')
    return output.getvalue()


def cube_camera_to_station_cv(face):
    """Camera-to-local matrix: station axes equal the native front CV camera.

    Local station X=front-image right, Y=front-image down, Z=front forward.
    CV rays use (u-cx)/fx, (v-cy)/fy, +1. Up/down preserve native tile roll;
    no provider heading, GPS alignment or learned camera pose is applied here.
    """
    rotations = {
        'F': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        'R': [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
        'B': [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
        'L': [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
        'U': [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        'D': [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
    }
    if face not in rotations:
        raise ValueError('Unknown native cube face')
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotations[face]
    return matrix


def distance_m(a, b):
    lat1, lng1 = float(a['lat']), float(a.get('lng', a.get('lon')))
    lat2, lng2 = float(b['lat']), float(b.get('lng', b.get('lon')))
    if not all(math.isfinite(x) for x in (lat1, lng1, lat2, lng2)) or max(abs(lat1), abs(lat2)) > 90 or max(abs(lng1), abs(lng2)) > 180:
        raise ValueError('Invalid geographic coordinates')
    p1, p2 = math.radians(lat1), math.radians(lat2)
    h = math.sin((p2-p1)/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(math.radians(lng2-lng1)/2)**2
    return 6371008.8 * 2 * math.asin(math.sqrt(min(1., max(0., h))))


def group_physical_stations(stations, tolerance_m=.25):
    """Deterministic complete-link co-location groups, never cube-face votes.

    Every pair inside a group is within tolerance. This avoids long chains of
    adjacent route captures collapsing into one station. The groups are for
    evidence/holdout accounting; each panorama retains its own SfM rig/pose.
    Provider GPS is an approximate co-location cue, not surveyed truth.
    """
    if isinstance(tolerance_m, bool) or not math.isfinite(tolerance_m) or tolerance_m < 0:
        raise ValueError('Co-location tolerance must be finite and nonnegative')
    records = [dict(row) for row in stations]
    ids = [row.get('pano_id', row.get('id')) for row in records]
    if any(not isinstance(key, str) or not key for key in ids) or len(set(ids)) != len(ids):
        raise ValueError('Each capture requires a unique panorama ID')
    groups = []
    for record in sorted(records, key=lambda row: row.get('pano_id', row.get('id'))):
        distance_m(record, record)
        group = next((items for items in groups if all(distance_m(record, other) <= tolerance_m for other in items)), None)
        if group is None:
            groups.append([record])
        else:
            group.append(record)
    assignment = {}
    for group in groups:
        members = sorted(row.get('pano_id', row.get('id')) for row in group)
        key = 'station_' + fingerprint(members)[:20]
        assignment.update({pano: key for pano in members})
    return assignment
