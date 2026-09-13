"""Strict standard 3DGS SH2 PLY reader preserving point and coefficient order."""
from pathlib import Path
import numpy as np
from plyfile import PlyData


def load_gaussian_ply(path,center,radius):
    path=Path(path)
    vertex=PlyData.read(path,mmap='r')['vertex'].data
    names=vertex.dtype.names
    required=['x','y','z','opacity']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]+[f'f_dc_{i}' for i in range(3)]+[f'f_rest_{i}' for i in range(24)]
    if any(key not in names for key in required):raise ValueError('Input PLY must contain complete standard SH2 Gaussian fields')
    extra=[key for key in names if key.startswith('f_rest_') and key not in required]
    if extra:raise ValueError('Input SH degree is not exactly 2; no silent truncation is allowed')
    def columns(keys):return np.column_stack([vertex[key] for key in keys]).astype(np.float32)
    xyz=columns(['x','y','z'])
    logscales=columns([f'scale_{i}' for i in range(3)])
    quats=columns([f'rot_{i}' for i in range(4)])
    sh0=columns([f'f_dc_{i}' for i in range(3)])[:,None,:]
    # Standard PLY stores channel-major rest coefficients. Brush orders field
    # names lexicographically in its header, so explicitly use numeric suffixes.
    shN=columns([f'f_rest_{i}' for i in range(24)]).reshape(-1,3,8).transpose(0,2,1).copy()
    opacity=np.array(vertex['opacity'],np.float32)
    for array in [xyz,logscales,quats,sh0,shN,opacity]:
        if not np.isfinite(array).all():raise ValueError('Nonfinite input Gaussian; refusing to alter the original point population')
    quat_norm=np.linalg.norm(quats,axis=1)
    if (quat_norm<1e-8).any():raise ValueError('Zero quaternion in Gaussian PLY')
    arrays=dict(means=((xyz.astype(np.float64)-center)/radius).astype(np.float32),
        scales=(logscales.astype(np.float64)-np.log(radius)).astype(np.float32),quats=quats,
        opacities=opacity,sh0=sh0,shN=shN)
    scale_values=np.exp(logscales.astype(np.float64))
    diagnostics=dict(source_vertex_count=len(xyz),sh_degree=2,point_order_preserved=True,
        property_lookup='Numeric suffix lookup; channel-major f_rest reshaped to N,8,3',
        quaternion_order='w,x,y,z; orientation unchanged by translation/uniform normalization',
        quaternion_norm_min=float(quat_norm.min()),quaternion_norm_max=float(quat_norm.max()),
        sh_transform='No SH rotation: external and normalized internal world axes are identical',
        opacity_transform='None: preserve raw logit',scale_transform='log_sigma_internal=log_sigma_metres-log(scene_radius_metres)',
        scales_m=dict(mean=float(scale_values.mean()),median=float(np.median(scale_values)),p95=float(np.percentile(scale_values,95)),
            p99=float(np.percentile(scale_values,99)),max=float(scale_values.max())))
    return arrays,diagnostics
