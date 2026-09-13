"""Common pinhole view rigs and four-tap seam-aware cubemap resampling.

Coordinates use pixel-edge intrinsics and centers x+.5,y+.5. Sampling decodes
sRGB to linear RGB. A bilinear tap beyond its primary cube face is projected
by that tap's ray into the adjacent face and reads its nearest actual texel.
Valid masks require every positive-weight actual contributor to be valid.
This conservatism applies to these four taps, not the full target-pixel footprint.
It cannot remove discontinuities already present in source cube RGB.
"""
import hashlib
import numpy as np
FACE_ORDER='FRBLUD'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def srgb_to_linear(rgb):
 a=np.asarray(rgb).astype(np.float32)/255.;return np.where(a<=.04045,a/12.92,((a+.055)/1.055)**2.4)
def linear_to_srgb(value):
 v=np.clip(value,0,1);v=np.where(v<=.0031308,v*12.92,1.055*v**(1/2.4)-.055);return np.clip(np.rint(v*255),0,255).astype(np.uint8)
def validate_cameras(cameras):
 require(len(cameras)==6,'A complete six-face cube is required')
 labels=[c['face'] for c in cameras];require(len(set(labels))==6,'Duplicate face names')
 rots=[]
 for c in cameras:
  w,h=c['w'],c['h'];require(type(w)is int and type(h)is int and w==h and w>0,'Square native faces required')
  K=np.array([c['fl_x'],c['fl_y'],c['cx'],c['cy']],np.float64);require(np.isfinite(K).all() and np.allclose(K,[w/2,h/2,w/2,h/2],rtol=0,atol=1e-9),'Source cube faces require centered90-degree calibration')
  R=np.array(c['camera_to_station_cv'],np.float64);require(R.shape==(4,4) and np.isfinite(R).all() and np.allclose(R[3],[0,0,0,1],rtol=0,atol=1e-12) and np.allclose(R[:3,3],0,rtol=0,atol=1e-12),'Cube camera must share station center')
  R=R[:3,:3];require(np.allclose(R.T@R,np.eye(3),rtol=0,atol=1e-9) and abs(np.linalg.det(R)-1)<1e-9,'Improper cube camera rotation');rots.append(R)
 normals=np.stack([R[:,2] for R in rots]);dots=normals@normals.T
 require(np.allclose(np.sort(dots,axis=1),np.tile([-1,0,0,0,0,1],(6,1)),rtol=0,atol=1e-9),'Camera optical axes do not describe a cube')
 require(all(np.allclose(np.max(np.abs(normals@R[:,:2]),axis=0),[1,1],rtol=0,atol=1e-9) for R in rots),'Cube face tangent axes must align with the shared cube edges')
 return rots

def build_rig(cameras,*,fov_degrees=70.,size=1024,offset_degrees=22.5):
 rots=validate_cameras(cameras);require(0<fov_degrees<179 and 0<offset_degrees<45 and type(size)is int and size>0,'Invalid rig settings')
 a=np.tan(np.deg2rad(offset_degrees));f=size/2/np.tan(np.deg2rad(fov_degrees/2));views=[]
 for c,R in zip(cameras,rots):
  for sy in (-1,1):
   for sx in (-1,1):
    forward=np.array([sx*a,sy*a,1.]);forward/=np.linalg.norm(forward);right=np.cross([0.,1.,0.],forward);right/=np.linalg.norm(right);down=np.cross(forward,right)
    pose=np.eye(4);pose[:3,:3]=R@np.column_stack([right,down,forward])
    views.append(dict(name=c['face']+('_left' if sx<0 else '_right')+('_up' if sy<0 else '_down'),anchor_face=c['face'],offset_sign_xy=[sx,sy],w=size,h=size,fl_x=float(f),fl_y=float(f),cx=size/2,cy=size/2,camera_to_station_cv=pose.tolist(),fov_x_degrees=float(fov_degrees),fov_y_degrees=float(fov_degrees),optical_direction_xy_over_z=[sx*a,sy*a]))
 return views

def view_rays(view,start=0,stop=None):
 stop=view['h'] if stop is None else stop;require(0<=start<stop<=view['h'],'Invalid row range')
 x,y=np.meshgrid((np.arange(view['w'])+.5-view['cx'])/view['fl_x'],(np.arange(start,stop)+.5-view['cy'])/view['fl_y']);R=np.asarray(view['camera_to_station_cv'],np.float64)[:3,:3]
 rays=np.stack([x,y,np.ones_like(x)],-1)@R.T;return rays/np.linalg.norm(rays,axis=-1,keepdims=True)

def coverage_report(views,*,erp_width=2048,erp_height=1024):
 require(erp_width==2*erp_height and erp_height>0,'Coverage ERP must be2:1')
 u,v=np.meshgrid((np.arange(erp_width)+.5)/erp_width*2*np.pi-np.pi,(np.arange(erp_height)+.5)/erp_height*np.pi-np.pi/2)
 rays=np.stack([np.cos(v)*np.sin(u),np.sin(v),np.cos(v)*np.cos(u)],-1).reshape(-1,3)
 special=np.array([[x,y,z] for x in (-1.,0.,1.) for y in (-1.,0.,1.) for z in (-1.,0.,1.) if x or y or z]);special/=np.linalg.norm(special,axis=1,keepdims=True);rays=np.concatenate([rays,special])
 count=np.zeros(len(rays),np.uint8);best=np.full(len(rays),np.inf)
 for view in views:
  q=rays@np.array(view['camera_to_station_cv'])[:3,:3];z=q[:,2];scale=np.maximum(np.abs(q[:,0])*2*view['fl_x']/view['w'],np.abs(q[:,1])*2*view['fl_y']/view['h'])/np.maximum(z,1e-30);scale[z<=0]=np.inf;count+=(scale<=1);best=np.minimum(best,scale)
 return dict(status='passed' if np.all(count>0) else 'failed',erp_rays=erp_width*erp_height,axis_edge_corner_rays=len(special),uncovered_rays=int(np.count_nonzero(count==0)),minimum_view_coverage=int(count.min()),maximum_view_coverage=int(count.max()),mean_view_coverage=float(count.mean()),worst_required_fraction_of_configured_half_fov_tangent=float(best.max()),coverage_is_sampled_not_continuous_proof=True)

class CubeSampler:
 def __init__(self,images,valid_masks,cameras):
  self.rotations=validate_cameras(cameras);require(len(images)==len(valid_masks)==len(cameras),'Incomplete cube arrays');self.cameras=cameras;self.linear=[];self.valid=[]
  for image,mask,c in zip(images,valid_masks,cameras):
   image=np.asarray(image);mask=np.asarray(mask);require(image.dtype==np.uint8 and image.shape==(c['h'],c['w'],3),'Cube RGB shape/dtype mismatch');require(mask.dtype==bool and mask.shape==image.shape[:2],'Cube valid mask shape/type mismatch')
   self.linear.append(srgb_to_linear(image));self.valid.append(mask)
  self.normals=np.stack([R[:,2] for R in self.rotations])
 def _nearest_taps(self,primary,tx,ty):
  c=self.cameras[primary];R=self.rotations[primary];ray=np.stack([(tx+.5-c['cx'])/c['fl_x'],(ty+.5-c['cy'])/c['fl_y'],np.ones_like(tx)],-1)@R.T
  owner=np.argmax(ray@self.normals.T,axis=-1);values=np.empty((len(tx),3),np.float64);valid=np.empty(len(tx),bool)
  for j,target in enumerate(self.cameras):
   choose=owner==j
   if not choose.any():continue
   q=ray[choose]@self.rotations[j];xx=np.rint(q[:,0]/q[:,2]*target['fl_x']+target['cx']-.5).astype(np.int64);yy=np.rint(q[:,1]/q[:,2]*target['fl_y']+target['cy']-.5).astype(np.int64)
   xx=np.clip(xx,0,target['w']-1);yy=np.clip(yy,0,target['h']-1)
   values[choose]=self.linear[j][yy,xx];valid[choose]=self.valid[j][yy,xx]
  return values,valid
 def sample(self,rays):
  rays=np.asarray(rays,np.float64);require(rays.ndim>=2 and rays.shape[-1]==3 and np.isfinite(rays).all() and np.all(np.linalg.norm(rays,axis=-1)>0),'Invalid target rays')
  shape=rays.shape[:-1];flat=rays.reshape(-1,3);owner=np.argmax(flat@self.normals.T,axis=-1);linear=np.zeros((len(flat),3),np.float64);valid=np.ones(len(flat),bool);cross_count=0;cross_pixels=0
  for j,c in enumerate(self.cameras):
   indices=np.flatnonzero(owner==j)
   if not len(indices):continue
   q=flat[indices]@self.rotations[j];x=q[:,0]/q[:,2]*c['fl_x']+c['cx']-.5;y=q[:,1]/q[:,2]*c['fl_y']+c['cy']-.5
   # Snap only floating-point near-integer ties, preserving exact identity masks.
   tol=128*np.finfo(np.float64).eps*max(c['w'],c['h']);x=np.where(np.abs(x-np.rint(x))<=tol,np.rint(x),x);y=np.where(np.abs(y-np.rint(y))<=tol,np.rint(y),y)
   x0=np.floor(x).astype(np.int64);y0=np.floor(y).astype(np.int64);dx=x-x0;dy=y-y0;had_cross=np.zeros(len(indices),bool)
   for tx,ty,wgt in ((x0,y0,(1-dx)*(1-dy)),(x0+1,y0,dx*(1-dy)),(x0,y0+1,(1-dx)*dy),(x0+1,y0+1,dx*dy)):
    active=wgt>0;outside=active&((tx<0)|(tx>=c['w'])|(ty<0)|(ty>=c['h']));direct=active&~outside
    if direct.any():
     linear[indices[direct]]+=self.linear[j][ty[direct],tx[direct]]*wgt[direct,None];valid[indices[direct]]&=self.valid[j][ty[direct],tx[direct]]
    if outside.any():
     value,tapvalid=self._nearest_taps(j,tx[outside],ty[outside]);linear[indices[outside]]+=value*wgt[outside,None];valid[indices[outside]]&=tapvalid;cross_count+=int(outside.sum());had_cross|=outside
   cross_pixels+=int(had_cross.sum())
  return linear_to_srgb(linear.reshape(*shape,3)),valid.reshape(shape),dict(output_pixels=len(flat),cross_face_positive_weight_taps=cross_count,pixels_with_cross_face_taps=cross_pixels,positive_weight_mask_contributors_only=True)
 def render(self,view,*,chunk_rows=32):
  require(type(chunk_rows)is int and chunk_rows>0,'chunk_rows must be positive');rgb=np.empty((view['h'],view['w'],3),np.uint8);valid=np.empty((view['h'],view['w']),bool);stats=dict(output_pixels=0,cross_face_positive_weight_taps=0,pixels_with_cross_face_taps=0)
  for start in range(0,view['h'],chunk_rows):
   stop=min(start+chunk_rows,view['h']);image,mask,counts=self.sample(view_rays(view,start,stop));rgb[start:stop]=image;valid[start:stop]=mask
   for key in stats:stats[key]+=counts[key]
  return rgb,valid,dict(stats,interpolation='linear RGB four bilinear taps; out-of-face taps ray-map to adjacent face nearest actual texel',mask_policy='Every positive-weight actual contributor must be valid; no claim of whole-pixel-footprint coverage',pixel_center_offset=.5,source_rgb_changed=False)
