"""Numerical, scene-independent helpers for visually reconstructed camera rigs."""
from __future__ import annotations
import math
import numpy as np

EDN_FROM_ENU = np.array([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])
CV_FROM_GL = np.diag([1.,-1.,-1.,1.])


def proper_rotation(matrix):
    value=np.asarray(matrix,dtype=np.float64)
    if value.shape!=(3,3) or not np.isfinite(value).all() or not np.allclose(value.T@value,np.eye(3),atol=1e-6,rtol=0) or not np.isclose(np.linalg.det(value),1.,atol=1e-6):
        raise ValueError('Expected a finite proper rotation')
    return value


def camera_to_station(value):
    pose=np.asarray(value,dtype=np.float64)
    if pose.shape!=(4,4) or not np.isfinite(pose).all() or not np.allclose(pose[3],[0,0,0,1],atol=1e-9,rtol=0):
        raise ValueError('camera_to_station_cv must be a finite affine 4x4 matrix')
    proper_rotation(pose[:3,:3])
    if not np.allclose(pose[:3,3],0.,atol=1e-9,rtol=0):
        raise ValueError('Cube faces must share their own capture rig center')
    return pose


def gps_local_enu(stations):
    """WGS84 ECEF to local ENU; absent altitude is not a height observation."""
    ll=np.asarray([[s['lat'],s.get('lng',s.get('lon'))] for s in stations],np.float64)
    if ll.ndim!=2 or ll.shape[1]!=2 or not np.isfinite(ll).all() or np.any(np.abs(ll[:,0])>90) or np.any(np.abs(ll[:,1])>180):
        raise ValueError('Station GPS must be finite latitude/longitude')
    lat0=float(np.median(ll[:,0]));lon0=float(np.rad2deg(np.arctan2(np.sin(np.deg2rad(ll[:,1])).mean(),np.cos(np.deg2rad(ll[:,1])).mean())))
    altitude=[];known=[]
    for station in stations:
        value=station.get('altitude_m',station.get('altitude'))
        ok=value is not None and not isinstance(value,bool) and np.isfinite(float(value))
        known.append(ok);altitude.append(float(value) if ok else np.nan)
    origin_alt=float(np.nanmedian(altitude)) if any(known) else 0.
    altitude=np.where(known,altitude,origin_alt)
    def ecef(lat,lon,height):
        lat,lon=np.deg2rad([lat,lon]);a=6378137.;e2=6.6943799901413165e-3
        n=a/np.sqrt(1-e2*np.sin(lat)**2)
        return np.array([(n+height)*np.cos(lat)*np.cos(lon),(n+height)*np.cos(lat)*np.sin(lon),(n*(1-e2)+height)*np.sin(lat)])
    la,lo=np.deg2rad([lat0,lon0]);basis=np.array([[-np.sin(lo),np.cos(lo),0],[-np.sin(la)*np.cos(lo),-np.sin(la)*np.sin(lo),np.cos(la)],[np.cos(la)*np.cos(lo),np.cos(la)*np.sin(lo),np.sin(la)]])
    origin=ecef(lat0,lon0,origin_alt)
    coordinates=np.asarray([basis@(ecef(lat,lon,h)-origin) for (lat,lon),h in zip(ll,altitude)])
    coordinates[~np.asarray(known),2]=0.
    return coordinates,np.asarray(known,bool),dict(lat=lat0,lng=lon0,altitude_m=origin_alt,altitude_origin_is_arbitrary=not any(known),coordinate_convention='ENU metres')


def upright_from_rig_rotations(world_to_station, headings_degrees):
    """Estimate one global geographic orientation from calibrated upright rigs.

    Heading supplies compass orientation; local -Y supplies the cube upright
    convention. This is an orientation gauge, not a per-camera GPS pose.
    """
    candidates=[]
    for matrix,heading in zip(world_to_station,headings_degrees):
        if heading is None or not np.isfinite(float(heading)):continue
        h=np.deg2rad(float(heading));enu_from_station=np.array([[np.cos(h),0,np.sin(h)],[-np.sin(h),0,np.cos(h)],[0,-1,0]])
        candidates.append(enu_from_station@proper_rotation(matrix))
    if len(candidates)<3:raise ValueError('Insufficient upright/heading metadata: need at least three visually registered physical groups')
    u,_,vt=np.linalg.svd(np.sum(candidates,axis=0));rotation=u@np.diag([1.,1.,np.linalg.det(u@vt)])@vt
    angles=np.rad2deg(np.arccos(np.clip([(np.trace(c@rotation.T)-1)/2 for c in candidates],-1,1)))
    return rotation,dict(orientation_samples=len(candidates),orientation_residual_median_degrees=float(np.median(angles)),orientation_residual_max_degrees=float(np.max(angles)),orientation_policy='Common upright cube gauge plus provider headings; no individual camera positions pinned')


def fit_horizontal_similarity(source,target,*,upright=None,huber_m=3.,iterations=50):
    """Positive Sim3 from a visual trajectory to metric horizontal GPS.

    Vertical geometry is preserved by one shared scale. A vertical translation
    merely sets an origin; absent GPS heights never flatten camera elevations.
    """
    source=np.asarray(source,np.float64);target=np.asarray(target,np.float64)
    if source.ndim!=2 or source.shape[1]!=3 or target.shape!=source.shape or len(source)<3 or not np.isfinite(source).all() or not np.isfinite(target).all():raise ValueError('Need at least three finite paired 3D positions')
    if not np.isfinite(huber_m) or huber_m<=0:raise ValueError('GPS robust scale must be positive')
    upright=np.eye(3) if upright is None else proper_rotation(upright)
    aligned=source@upright.T;weights=np.ones(len(source))
    for _ in range(iterations):
        w=weights/weights.sum();xm=np.sum(aligned[:,:2]*w[:,None],axis=0);ym=np.sum(target[:,:2]*w[:,None],axis=0)
        x=aligned[:,:2]-xm;y=target[:,:2]-ym
        denominator=np.sum(w*np.sum(x*x,axis=1))
        if not np.isfinite(denominator) or denominator<=np.finfo(float).tiny:raise ValueError('Visual camera baseline is degenerate')
        u,_,vt=np.linalg.svd((y*w[:,None]).T@x);r=u@np.diag([1.,np.linalg.det(u@vt)])@vt
        scale=float(np.sum(w*np.sum(y*(x@r.T),axis=1))/denominator)
        if not np.isfinite(scale) or scale<=0:raise ValueError('GPS scale is nonpositive or degenerate')
        shift=ym-scale*r@xm;residual=np.linalg.norm(scale*(aligned[:,:2]@r.T)+shift-target[:,:2],axis=1)
        updated=np.minimum(1.,huber_m/np.maximum(residual,np.finfo(float).eps))
        if np.max(np.abs(updated-weights))<1e-10:break
        weights=updated
    r3=np.eye(3);r3[:2,:2]=r;rotation=r3@upright
    translation=np.array([*shift,float(np.median(target[:,2]-scale*(source@rotation.T)[:,2]))])
    return scale,rotation,translation,dict(scale=scale,horizontal_residuals_m=residual.tolist(),horizontal_rmse_m=float(np.sqrt(np.mean(residual**2))),horizontal_median_m=float(np.median(residual)),robust_weights=weights.tolist(),gps_horizontal_extent_m=float(np.max(np.linalg.norm(target[:,:2]-np.mean(target[:,:2],axis=0),axis=1))),visual_horizontal_extent=float(np.max(np.linalg.norm(aligned[:,:2]-np.mean(aligned[:,:2],axis=0),axis=1))))


def candidate_capture_pairs(stations,neighbors=8):
    """GPS proposes match candidates only. Same physical groups are excluded."""
    if neighbors<1:raise ValueError('neighbors must be positive')
    coordinates,_,_=gps_local_enu(stations);distance=np.linalg.norm(coordinates[:,None,:2]-coordinates[None,:,:2],axis=-1)
    ids=[str(p.get('pano_id',p.get('id'))) for p in stations];groups=[str(p['station_id']) for p in stations]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate capture IDs')
    pairs=set()
    for i in range(len(stations)):
        candidates=sorted((j for j in range(len(stations)) if groups[j]!=groups[i]),key=lambda j:(distance[i,j],ids[j]))
        pairs.update(tuple(sorted((i,j))) for j in candidates[:neighbors])
    # Include explicit provider graph edges while retaining physical grouping.
    lookup={pid:i for i,pid in enumerate(ids)}
    for i,station in enumerate(stations):
        for link in station.get('links',[]) or []:
            other=link.get('id') if isinstance(link,dict) else None;j=lookup.get(str(other))
            if j is not None and groups[j]!=groups[i]:pairs.add(tuple(sorted((i,j))))
    return sorted(pairs)


def maximum_ray_angle_degrees(point,centers):
    rays=np.asarray(centers,np.float64)-np.asarray(point,np.float64);length=np.linalg.norm(rays,axis=1)
    if len(rays)<2 or np.any(length<=0):return 0.
    rays/=length[:,None];return float(np.rad2deg(np.arccos(np.clip(np.min(rays@rays.T),-1,1))))


def training_observation_color(observations,train_groups,*,minimum_groups=2):
    """Median by physical group, then across groups; held-out RGB never enters."""
    groups={}
    for group,rgb in observations:
        if str(group) not in train_groups:continue
        color=np.asarray(rgb,np.float64)
        if color.shape!=(3,) or not np.isfinite(color).all() or np.any((color<0)|(color>255)):raise ValueError('Observation RGB must be finite 0..255')
        groups.setdefault(str(group),[]).append(color)
    if len(groups)<minimum_groups:return None
    color=np.median([np.median(values,axis=0) for values in groups.values()],axis=0)
    return np.rint(color).astype(np.uint8),len(groups)
