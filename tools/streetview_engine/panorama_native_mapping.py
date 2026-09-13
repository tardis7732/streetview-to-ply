"""Frozen calibrated fixed-alpha mapping, independent of registration estimation."""
import numpy as np
def require(v,m):
 if not v:raise ValueError(m)

def native_uv_alpha(camera, erp_alpha, chunk_rows=32):
    """Original calibrated pixel-center rays; fixed-alpha four-term interpolation.

    The generated-image registration matrix is deliberately absent here.
    """
    w,h = int(camera['w']),int(camera['h'])
    require(w > 0 and h > 0, 'Invalid native camera size')
    fx,fy,cx,cy = (float(camera[k]) for k in ('fl_x','fl_y','cx','cy'))
    require(np.isfinite([fx,fy,cx,cy]).all() and min(fx,fy)>0, 'Invalid calibration')
    pose = np.asarray(camera['camera_to_station_cv'], dtype=np.float64)
    require(pose.shape in ((3,3),(4,4)) and np.isfinite(pose).all(), 'Invalid station rotation')
    if pose.shape == (4,4):
        require(np.allclose(pose[3],[0,0,0,1],rtol=0,atol=1e-10) and
                np.allclose(pose[:3,3],0,rtol=0,atol=1e-10), 'Non-centered camera-to-station pose')
    R=pose[:3,:3]
    require(np.allclose(R.T@R,np.eye(3),rtol=0,atol=1e-7) and
            np.isclose(np.linalg.det(R),1,rtol=0,atol=1e-7), 'Improper camera rotation')
    eh,ew=erp_alpha.shape
    require(ew==2*eh and np.isfinite(erp_alpha).all() and np.all((erp_alpha>=0)&(erp_alpha<=1)), 'Invalid ERP alpha')
    uv=np.zeros((h,w,2),np.float64); result=np.zeros((h,w),np.float64)
    xx=(np.arange(w,dtype=np.float64)+.5-cx)/fx
    for start in range(0,h,chunk_rows):
        stop=min(start+chunk_rows,h)
        x,y=np.meshgrid(xx,(np.arange(start,stop,dtype=np.float64)+.5-cy)/fy)
        rays=np.stack((x,y,np.ones_like(x)),-1)@R.T
        rays/=np.linalg.norm(rays,axis=-1,keepdims=True)
        u=np.arctan2(rays[...,0],rays[...,2])/(2*np.pi)+.5
        v=np.arcsin(np.clip(rays[...,1],-1,1))/np.pi+.5
        uv[start:stop]=np.stack((u,v),-1)
        px=u*ew-.5; py=np.clip(v*eh-.5,0,eh-1)
        x0=np.floor(px).astype(np.int64);y0=np.floor(py).astype(np.int64)
        dx=px-x0;dy=py-y0
        a=np.zeros_like(px)
        for iy,ix,weight in ((y0,x0%ew,(1-dx)*(1-dy)),(y0,(x0+1)%ew,dx*(1-dy)),
                             (np.minimum(y0+1,eh-1),x0%ew,(1-dx)*dy),
                             (np.minimum(y0+1,eh-1),(x0+1)%ew,dx*dy)):
            a += erp_alpha[iy,ix]*weight
        result[start:stop]=np.clip(a,0,1)
    return uv,result
