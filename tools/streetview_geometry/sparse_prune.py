"""Opt-in sparse-depth free-space proposals and exact-row diagnostic PLY export.

No opacity editing, sparse depth filling, percentile pruning or scene exceptions.
Finite projected samples are evidence, not a proof of a whole empty volume.
"""
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import time

import numpy as np
import torch

from .contracts import CameraSet
from .free_space import FreeSpaceConfig, select_free_space


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf8')


def project(means, covariance, view, K):
    xyz = means @ view[:3, :3].T + view[:3, 3]
    z = xyz[:, 2]
    safe = torch.where(z > 0, z, torch.ones_like(z))
    uv = xyz[:, :2] / safe[:, None] * K.diag()[:2] + K[:2, 2]
    sigma = torch.einsum('i,nij,j->n', view[2, :3], covariance, view[2, :3]).clamp_min(0).sqrt()
    return xyz, uv, sigma


def footprint_uv(xyz, covariance, view, K, uv):
    z = torch.where(xyz[:, 2] > 0, xyz[:, 2], torch.ones_like(xyz[:, 2]))
    zero = torch.zeros_like(z)
    jx = torch.stack([K[0, 0]/z, zero, -K[0, 0]*xyz[:, 0]/z.square()], -1) @ view[:3, :3]
    jy = torch.stack([zero, K[1, 1]/z, -K[1, 1]*xyz[:, 1]/z.square()], -1) @ view[:3, :3]
    J = torch.stack([jx, jy], 1)
    values, vectors = torch.linalg.eigh(J @ covariance @ J.transpose(-1, -2))
    # The symmetric square root is unique even at repeated eigenvalues. Its
    # columns tie sample directions to image axes, not arbitrary eigenvectors.
    axes = (vectors * values.clamp_min(0).sqrt()[:, None, :]) @ vectors.transpose(-1, -2)
    offsets = [torch.zeros_like(uv)]
    for axis in range(2):
        for multiplier in [-2., -1., 1., 2.]:
            offsets.append(axes[:, :, axis] * multiplier)
    return uv[:, None, :] + torch.stack(offsets, 1)


def sample_uv(uv, depth, valid):
    """Native pixel centres are i+.5; floor selects its containing pixel.

    Invalid samples carry -1 rather than a clamped valid border pixel ID.
    """
    h, w = depth.shape
    finite = torch.isfinite(uv).all(-1)
    xy = torch.where(torch.isfinite(uv), uv, torch.zeros_like(uv)).floor().long()
    inside = finite & (xy[..., 0] >= 0) & (xy[..., 0] < w) & (xy[..., 1] >= 0) & (xy[..., 1] < h)
    indices = xy[..., 1].clamp(0, h-1)*w + xy[..., 0].clamp(0, w-1)
    values = depth.reshape(-1)[indices]
    keep = inside & valid.reshape(-1)[indices] & torch.isfinite(values) & (values > 0)
    return values, keep, torch.where(keep, indices, -torch.ones_like(indices))


def unique_pixels(ids, valid):
    unique = valid.clone()
    for i in range(1, ids.shape[1]):
        unique[:, i] &= ~((ids[:, i, None] == ids[:, :i]) & valid[:, :i]).any(-1)
    return unique


def reduce_station(free_per_view, blocked_per_view, support_per_view, groups):
    """Any nonfree evidence cancels that station's free vote across all faces."""
    groups = np.asarray(list(map(str, groups)))
    arrays = [np.asarray(a) for a in (free_per_view, blocked_per_view, support_per_view)]
    if any(a.dtype != np.bool_ or a.ndim != 2 or a.shape != arrays[0].shape for a in arrays):
        raise ValueError('Expected matching boolean row/view matrices')
    if len(groups) != arrays[0].shape[1]:
        raise ValueError('Missing physical station identity')
    names = np.unique(groups)
    free, blocked, support = arrays
    ff = np.zeros((len(free), len(names)), bool); ss = ff.copy()
    for i, group in enumerate(names):
        cols = groups == group
        ss[:, i] = support[:, cols].any(1)
        ff[:, i] = free[:, cols].any(1) & ~blocked[:, cols].any(1) & ~ss[:, i]
    return dict(station_ids=names, free=ff, support=ss)


def classify(z, sigma, depth, valid, config):
    # No calibrated sigma_depth was supplied. The explicit relative tolerance
    # is a policy margin, NOT a claimed statistical uncertainty estimate.
    lower, upper = z-config.gaussian_sigma*sigma, z+config.gaussian_sigma*sigma
    usable = valid & (z > 0) & torch.isfinite(z) & torch.isfinite(sigma)
    free = usable & (lower > 0) & (upper < depth*(1-config.relative_depth_margin))
    behind = usable & (lower > depth*(1+config.relative_depth_margin))
    support = usable & ~free & ~behind
    return free, support


def quaternion_covariance(quats, scales):
    q = quats / torch.linalg.vector_norm(quats, dim=1, keepdim=True)
    w, x, y, z = q.unbind(1)
    R = torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                     2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                     2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], 1).reshape(-1, 3, 3)
    basis = R * scales[:, None, :]
    return basis @ basis.transpose(-1, -2)


def write_subset(source, destination, keep, expected_sha):
    """Preserve header except count and every retained attribute byte/order."""
    from plyfile import PlyData
    if destination.exists() or source.resolve() == destination.resolve() or sha(source) != expected_sha:
        raise ValueError('Fresh destination and unchanged source required')
    ply = PlyData.read(source, mmap='r')
    if len(ply.elements) != 1 or ply.elements[0].name != 'vertex' or ply.text:
        raise ValueError('Only one binary vertex element supported')
    rows = ply['vertex'].data
    if keep.dtype != np.bool_ or keep.shape != (len(rows),) or not keep.any():
        raise ValueError('Invalid nonempty keep mask')
    with source.open('rb') as f:
        lines = []
        for _ in range(2048):
            line = f.readline(8192); lines.append(line)
            if line.strip() == b'end_header':
                break
        else:
            raise ValueError('Invalid header')
        offset = f.tell()
    if source.stat().st_size != offset+len(rows)*rows.dtype.itemsize or rows.dtype.hasobject:
        raise ValueError('Unexpected PLY payload layout')
    digest = hashlib.sha256()
    with destination.open('xb') as f:
        for line in lines:
            if line.split()[:2] == [b'element', b'vertex']:
                line = f'element vertex {int(keep.sum())}'.encode() + (b'\r\n' if line.endswith(b'\r\n') else b'\n')
            f.write(line)
        for start in range(0, len(rows), 65536):
            block = rows[start:start+65536][keep[start:start+65536]].tobytes()
            digest.update(block); f.write(block)
    reread = PlyData.read(destination, mmap='r')['vertex'].data
    if reread.dtype != rows.dtype or len(reread) != keep.sum() or hashlib.sha256(reread.tobytes()).hexdigest() != digest.hexdigest() or sha(source) != expected_sha:
        raise ValueError('Kept PLY bytes or source identity changed')
    return dict(path=str(destination), sha256=sha(destination), rows=len(reread),
                retained_payload_sha256=digest.hexdigest(), retained_attributes='exact_bytes_original_order')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['model', 'depth-dir', 'dataset', 'validated-code', 'output', 'gpu-lock']:
        p.add_argument('--'+name, required=True, type=Path)
    for name in ['model-sha256', 'depth-manifest-sha256', 'transforms-sha256', 'validated-contracts-sha256']:
        p.add_argument('--'+name, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--export-candidate', action='store_true')
    p.add_argument('--full-observed-footprint', action='store_true',
                   help='Scan every observed native depth pixel inside the projected ellipsoid instead of nine point samples')
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Use a fresh output directory')
    config = FreeSpaceConfig()
    torch.set_num_threads(4)
    start = time.monotonic()
    paths = {str(a.model): a.model_sha256,
             str(a.depth_dir/'depth_manifest.json'): a.depth_manifest_sha256,
             str(a.dataset/'transforms_train.json'): a.transforms_sha256,
             str(a.validated_code/'contracts.py'): a.validated_contracts_sha256,
             str(Path(__file__).resolve()): sha(__file__)}
    for name in ['contracts.py', 'free_space.py']:
        path = Path(__file__).with_name(name).resolve()
        paths[str(path)] = sha(path)
    if a.full_observed_footprint:
        from .observed_footprint import observe
        path = Path(__file__).with_name('observed_footprint.py').resolve()
        paths[str(path)] = sha(path)
    for path, expected in paths.items():
        if sha(path) != expected:
            raise ValueError('Input hash differs: '+path)
    spec = importlib.util.spec_from_file_location('validated_depth_contracts', a.validated_code/'contracts.py')
    verified = importlib.util.module_from_spec(spec); spec.loader.exec_module(verified)
    manifest = read(a.depth_dir/'depth_manifest.json')
    verified.validate_depth_bindings(a.depth_dir, a.dataset, manifest['processing']['training_max_resolution'])
    paths.update(manifest['input_bindings'])
    transforms = read(a.dataset/'transforms_train.json')
    frames = transforms['frames']
    by_name = {Path(f['file_path']).name: f for f in frames}
    world = np.asarray(manifest['world_from_enu'], np.float64)
    if world.shape != (4, 4) or not np.allclose(world[3], [0, 0, 0, 1]) or not np.allclose(world[:3, :3].T@world[:3, :3], np.eye(3)) or not np.isclose(np.linalg.det(world[:3, :3]), 1):
        raise ValueError('Explicit proper rigid depth-world conversion required')
    cameras = CameraSet.from_frames(frames, convention='opengl', station_key='station_index', world_up=world[:3, :3]@np.array([0., 0., 1.]))
    station_names = sorted(set(cameras.station_ids))
    entries = sorted(manifest['entries'], key=lambda e: (str(e['station']), e['image']))
    if 'other physical' not in manifest['source_count_convention'].lower():
        raise ValueError('This adapter requires declared other-physical-station counts')
    cache = []
    depth_pixels = 0
    for entry in entries:
        path = (a.depth_dir/entry['npz']).resolve(strict=True)
        if not path.is_relative_to(a.depth_dir.resolve()) or sha(path) != entry['npz_sha256']:
            raise ValueError('Depth binding differs')
        paths[str(path)] = entry['npz_sha256']
        with np.load(path, allow_pickle=False) as d:
            z, valid, confidence = d['depth_z'].copy(), d['valid'].copy(), d['confidence'].copy()
            f = by_name[entry['image']]; h, w = z.shape
            if valid.dtype != np.bool_ or int(d['h']) != h or int(d['w']) != w or str(d['world_frame']) != 'ENU' or str(d['depth_convention']) != 'camera_z' or str(d['unit']) not in ('m','metre','metres') or float(d['pixel_center_offset']) != .5:
                raise ValueError('Native depth calibration metadata differs')
            if int(d['station']) != f['station_index'] or entry['station'] != f['station_index']:
                raise ValueError('Physical station identity differs')
            K = d['K'].copy(); view = d['camera_from_world']@np.linalg.inv(world)
            expected_K = np.array([[f['fl_x']*w/f['w'], 0, f['cx']*w/f['w']], [0,f['fl_y']*h/f['h'],f['cy']*h/f['h']],[0,0,1]])
            expected_view = np.linalg.inv(np.asarray(f['transform_matrix'])@np.diag([1.,-1.,-1.,1.]))[:3]
            if not np.allclose(K,expected_K,rtol=0,atol=1e-5) or not np.allclose(view,expected_view,rtol=0,atol=1e-6):
                raise ValueError('Depth projection differs from photographic camera')
            if np.any(valid & (d['source_type'] != 1)):
                raise ValueError('Sparse-only audit refuses dense/unknown evidence')
            valid &= np.isfinite(z) & (z > 0) & np.isfinite(confidence) & (confidence > 0) & (d['source_count'] >= 1)
            strict = valid & (confidence >= .25)
        depth_pixels += int(valid.sum())
        if valid.any():
            cache.append(dict(image=entry['image'], station=str(entry['station']), z=z, valid=valid, strict=strict, view=view, K=K))
    from plyfile import PlyData
    rows = PlyData.read(a.model, mmap='r')['vertex'].data
    def columns(names):
        values = np.column_stack([rows[n] for n in names]).astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite Gaussian')
        return torch.tensor(values, device=a.device)
    # The same shared lock as training/rendering prevents competing GPU work.
    import fcntl
    with a.gpu_lock.open('a+') as lock, torch.no_grad():
        fcntl.flock(lock, fcntl.LOCK_EX)
        means = columns(['x','y','z'])
        scales = columns([f'scale_{i}' for i in range(3)]).exp()
        quats = columns([f'rot_{i}' for i in range(4)])
        if not torch.isfinite(scales).all() or (scales <= 0).any() or (torch.linalg.vector_norm(quats, dim=1) <= 1e-8).any():
            raise ValueError('Invalid Gaussian extents/rotations')
        covariance = quaternion_covariance(quats, scales)
        del quats, scales
        def gpu(frame):
            return {k: torch.as_tensor(frame[k], device=a.device) for k in ['z','valid','strict','view','K']}
        n = len(rows)
        cf = np.zeros((n, len(station_names)), bool); cs = cf.copy(); cb = cf.copy()
        for i, frame in enumerate(cache):
            d = gpu(frame); si = station_names.index(frame['station'])
            xyz, uv, sigma = project(means, covariance, d['view'], d['K'])
            z, valid, _ = sample_uv(uv, d['z'], d['valid'])
            _, strict, _ = sample_uv(uv, d['z'], d['strict'])
            free, _ = classify(xyz[:, 2], sigma, z, strict, config)
            loose_free, support = classify(xyz[:, 2], sigma, z, valid, config)
            cf[:, si] |= free.cpu().numpy(); cs[:, si] |= support.cpu().numpy()
            cb[:, si] |= (valid & (xyz[:, 2]>0) & ~loose_free).cpu().numpy()
            if (i+1)%25 == 0:
                print(json.dumps(dict(stage='center', frames=i+1, elapsed_s=time.monotonic()-start)), flush=True)
        cf &= ~cb & ~cs
        pool_ids = np.flatnonzero((cf.sum(1)>=config.min_free_stations) & ~cs.any(1))
        print(json.dumps(dict(stage='center_complete', pool=len(pool_ids), elapsed_s=time.monotonic()-start)), flush=True)
        mm, cov = means[pool_ids], covariance[pool_ids]
        del means, covariance
        pf = np.zeros((len(pool_ids), len(cache)), bool); pb = pf.copy(); ps = pf.copy()
        observed_free_count = np.zeros(pf.shape, np.int32)
        observed_valid_count = np.zeros(pf.shape, np.int32)
        witnesses = []
        for i, frame in enumerate(cache):
            if not len(pool_ids):
                break
            d = gpu(frame); xyz, uv, sigma = project(mm, cov, d['view'], d['K'])
            if a.full_observed_footprint:
                evidence = observe(mm, cov, d['view'], d['K'], d['z'], d['valid'], d['strict'], config)
                pf[:, i] = evidence['free_view'].cpu().numpy()
                pb[:, i] = evidence['blocked_view'].cpu().numpy()
                ps[:, i] = evidence['support_view'].cpu().numpy()
                observed_free_count[:, i] = evidence['strict_free_pixels'].cpu().numpy()
                observed_valid_count[:, i] = evidence['valid_pixels'].cpu().numpy()
                continue
            samples = footprint_uv(xyz, cov, d['view'], d['K'], uv)
            z, valid, pixels = sample_uv(samples, d['z'], d['valid'])
            _, strict, _ = sample_uv(samples, d['z'], d['strict'])
            free, _ = classify(xyz[:, 2, None], sigma[:, None], z, strict, config)
            loose_free, support = classify(xyz[:, 2, None], sigma[:, None], z, valid, config)
            distinct = unique_pixels(pixels, strict)
            count = (free & distinct).sum(1)
            blocked = (valid & (xyz[:, 2, None]>0) & ~loose_free).any(1)
            accepted = (count >= config.min_footprint_samples) & ~blocked
            pf[:, i] = accepted.cpu().numpy(); pb[:, i] = blocked.cpu().numpy(); ps[:, i] = support.any(1).cpu().numpy()
            ids = accepted.nonzero().flatten()
            if len(ids):
                witnesses.append(dict(frame=i, ids=ids.cpu().numpy(), pixels=torch.where(free & distinct,pixels,-1)[ids].cpu().numpy(), depth=z[ids].cpu().numpy(), upper=(xyz[:,2]+3*sigma)[ids].cpu().numpy()))
        flags = reduce_station(pf, pb, ps, [f['station'] for f in cache])
        selected = select_free_space(pool_ids, flags['station_ids'], flags['free'], flags['support'], flags['free'], config=config)
    for path, expected in paths.items():
        if sha(path) != expected:
            raise ValueError('Inputs changed during audit: '+path)
    a.output.mkdir(parents=True)
    ids = selected['indices']; keep = np.ones(n, bool); keep[ids] = False
    np.savez_compressed(a.output/'selection.npz', source_sha256=np.array(a.model_sha256), keep=keep,
                        center_free_stations=cf.sum(1), center_supported_stations=cs.sum(1),
                        pool_ids=pool_ids, candidate_ids=ids, station_ids=flags['station_ids'],
                        view_strict_free_pixels=observed_free_count, view_valid_pixels=observed_valid_count,
                        footprint_free=flags['free'], footprint_support=flags['support'])
    if witnesses:
        np.savez_compressed(a.output/'witnesses.npz',
                            frame_index=np.concatenate([np.full(len(w['ids']),w['frame']) for w in witnesses]),
                            **{k:np.concatenate([w[k] for w in witnesses]) for k in ['ids','pixels','depth','upper']})
    artifact = write_subset(a.model, a.output/'candidate.ply', keep, a.model_sha256) if len(ids) and a.export_candidate else None
    report = dict(status='completed', source=str(a.model), source_sha256=a.model_sha256,
                  source_rows=n, center_pool=len(pool_ids), proposed_rows=len(ids), candidate=artifact,
                  selection_sha256=sha(a.output/'selection.npz'), policy=asdict(config),
                  native_depth_pixels=depth_pixels, nonempty_depth_views=len(cache), physical_stations=len(station_names),
                  evidence_pixel_resolution='Original NPZ grid; no filling/dilation/resizing',
                  confidence_gate=.25, surface_uncertainty='not_supplied; explicit 10% relative policy margin only',
                  footprint_mode='all_native_observations_core_2sigma_veto_3sigma' if a.full_observed_footprint else 'nine_symmetric_root_samples',
                  source_count_convention=manifest['source_count_convention'],
                  input_bindings=paths, frame_roster=[{k:f[k] for k in ['image','station']} for f in cache],
                  source_unchanged=sha(a.model)==a.model_sha256, product_default_changed=False,
                  outcome='diagnostic_candidate_requires_render_review' if artifact else 'insufficient_supported_candidates_no_ply_exported',
                  limitations=['Sparse SfM shares training poses; it is not surveyed ground truth.',
                               'Finite observed pixels and 3-sigma truncation are heuristic evidence, not full-volume emptiness proof.',
                               'Center screening is conservative and may miss removable Gaussians.',
                               'No reliable depth is unknown, not empty space. Behind-surface observations never cause deletion.',
                               'Confidence is not calibrated depth sigma. One scene does not establish cross-location quality.'],
                  elapsed_s=time.monotonic()-start)
    write(a.output/'report.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['input_bindings','frame_roster']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
