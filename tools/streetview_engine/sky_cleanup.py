"""Reusable semantic-center cleanup with explicit heuristic depth abstention.

Inputs are native masks and registered cameras. Positive sky votes never come
from depth; below-camera preservation is a height rule, not a ground surface.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil

import numpy as np

from .size_filter import _camera_reference, _json_sha, _ply, _sha
from .ply_cleanup import _verify_exact_rows, _world_up, _write_subset, write_output_camera_reference
from ..streetview_geometry.camera_below import below_camera_centers

POLICY = 'semantic_center_down_camera_below_and_hard_size_v1'
DEFAULTS = dict(erosion_radius_px=2,minimum_sky_stations=3,sky_to_non_sky_ratio=.25,
    depth_non_sky_abstention=False,depth_factor=1.5,soft_size_ratio=.1,
    protect_down=True,minimum_down_stations=1,protect_camera_below=True,
    nearest_camera_stations=3,hard_size_ratio=.5)


def build_mask_manifest(dataset_dir,camera_json,destination,depth_manifest=None):
    """Join registered camera extras to native semantic inputs; writes a new JSON.

    Original valid coverage is mandatory. A sky-excluding training mask is not
    a substitute. All prepared paths must stay inside the supplied dataset.
    """
    dataset=Path(dataset_dir).resolve();camera_path=Path(camera_json).resolve();target=Path(destination).resolve()
    if target.exists(): raise ValueError('Native cleanup input manifest must be new')
    doc=json.loads(camera_path.read_text(encoding='utf-8-sig'))
    convention=doc.get('camera_convention')
    if convention not in ('OpenGL_c2w','OpenCV_c2w'): raise ValueError('Explicit registered camera convention required')
    camera_sha=_sha(camera_path);depth_rows={};depth_root=None;bindings={str(camera_path):camera_sha}
    if depth_manifest is not None:
        depth_path=Path(depth_manifest).resolve();depth_doc=json.loads(depth_path.read_text(encoding='utf-8-sig'))
        if depth_doc.get('status')!='completed' or depth_doc.get('cameras_sha256')!=camera_sha:
            raise ValueError('Depth manifest is not completed or belongs to other cameras')
        depth_root=depth_path.parent;bindings[str(depth_path)]=_sha(depth_path)
        for row in depth_doc['frames']:
            key=(str(row['station_id']),row['face'],row['image'])
            if key in depth_rows: raise ValueError('Duplicate depth frame identity')
            depth_rows[key]=row
    records=[];seen={}
    fields=[('source_rgb','file_path',('image_sha256','source_sha256')),
        ('source_original','original_file_path',('original_file_sha256','original_sha256')),
        ('source_sky','sky_mask_path',('sky_mask_sha256',)),
        ('source_generic_valid','original_valid_mask_path',('original_valid_mask_sha256',)),
        ('source_edit_alpha','edit_alpha_path',('edit_alpha_sha256',))]
    for raw in doc['frames']:
        frame={**{key:doc[key] for key in ('w','h','fl_x','fl_y','cx','cy') if key in doc},**raw}
        key=(str(frame['station_id']),frame['face'],frame['file_path'])
        if key in seen:
            if seen[key]!=frame: raise ValueError('Conflicting duplicate registered camera')
            continue
        seen[key]=frame
        pose=np.asarray(frame['transform_matrix'],np.float64)
        if pose.shape!=(4,4) or not np.isfinite(pose).all(): raise ValueError('Invalid registered camera pose')
        cv=pose@np.diag([1.,-1.,-1.,1.]) if convention=='OpenGL_c2w' else pose
        row=dict(image=frame['file_path'],station_id=str(frame['station_id']),face=frame['face'],width=frame['w'],height=frame['h'],
            K=[[frame['fl_x'],0,frame['cx']],[0,frame['fl_y'],frame['cy']],[0,0,1.]],camera_from_world=np.linalg.inv(cv).tolist())
        for output_name,path_key,hash_keys in fields:
            value=frame.get(path_key)
            digest=next((frame.get(key) for key in hash_keys if frame.get(key)),None)
            if not isinstance(value,str) or not value or not isinstance(digest,str):
                raise ValueError(f'Registered frame lacks original cleanup input: {path_key}')
            path=(dataset/Path(value)).resolve()
            if not path.is_relative_to(dataset) or not path.is_file() or _sha(path)!=digest:
                raise ValueError(f'Registered cleanup input escaped dataset or changed: {path_key}')
            row[output_name+'_path']=str(path);row[output_name+'_sha256']=digest;bindings[str(path)]=digest
        if depth_root is not None:
            if key not in depth_rows: raise ValueError('Depth manifest is missing a registered frame')
            depth=depth_rows[key];path=(depth_root/Path(depth['depth_path'])).resolve()
            if not path.is_file() or _sha(path)!=depth['depth_sha256']: raise ValueError('Bound image depth changed')
            row.update(depth_path=str(path),depth_sha256=depth['depth_sha256']);bindings[str(path)]=depth['depth_sha256']
        records.append(row)
    if not records or (depth_root is not None and set(seen)!=set(depth_rows)):
        raise ValueError('Registered native input/depth roster differs')
    for path,digest in bindings.items():
        if _sha(path)!=digest: raise ValueError('Native cleanup input changed while preparing manifest')
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(dict(schema_version=1,cameras_sha256=camera_sha,frames=records,
        source_bindings=bindings),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    return target


def read_sky_cleanup_options(value=None):
    if value is None: value={}
    if not isinstance(value,dict) or set(value)-set(DEFAULTS):
        raise ValueError('Invalid sky cleanup options')
    options=dict(DEFAULTS,**value)
    for key in ('depth_non_sky_abstention','protect_down','protect_camera_below'):
        if type(options[key]) is not bool: raise ValueError(f'{key} must be a boolean')
    for key,low,high in [('erosion_radius_px',0,64),('minimum_sky_stations',2,10000),
                         ('minimum_down_stations',1,10000),('nearest_camera_stations',2,10000)]:
        if type(options[key]) is not int or not low<=options[key]<=high:
            raise ValueError(f'Invalid {key}')
    for key,low,high in [('sky_to_non_sky_ratio',.0001,100),('depth_factor',1.,100),
                         ('soft_size_ratio',.0001,10),('hard_size_ratio',.0001,10)]:
        value=options[key]
        if key=='hard_size_ratio' and value is None: continue
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not low<=value<=high:
            raise ValueError(f'Invalid {key}')
        options[key]=float(value)
    if options['hard_size_ratio'] is not None and options['hard_size_ratio']<options['soft_size_ratio']:
        raise ValueError('Hard size cap must not be smaller than the protected soft size threshold')
    return options


def select_cleanup_rows(means, log_scales, frames, up, counts, options):
    """Pure policy from physical-station counts; returns original-order mask."""
    options=read_sky_cleanup_options(options)
    means,scales=np.asarray(means,np.float64),np.asarray(log_scales,np.float64)
    if means.ndim!=2 or means.shape[1]!=3 or scales.shape!=means.shape or not np.isfinite(means).all() or not np.isfinite(scales).all():
        raise ValueError('Finite Gaussian centers and log scales required')
    groups={}
    for frame in frames:
        if isinstance(frame.get('station_id'),bool) or not isinstance(frame.get('station_id'),(str,int)) or not str(frame['station_id']):
            raise ValueError('Explicit physical station IDs required')
        groups.setdefault(str(frame['station_id']),set()).add(tuple(np.asarray(frame['transform_matrix'],np.float64)[:3,3]))
    centers=np.array([np.asarray(sorted(groups[key])).mean(0) for key in sorted(groups)])
    radius=float(np.linalg.norm(centers-centers.mean(0),axis=1).max())
    if not math.isfinite(radius) or radius<=0 or len(groups)<options['minimum_sky_stations']:
        raise ValueError('Insufficient physical camera stations for cleanup evidence')
    for index,key in enumerate(sorted(groups)):
        if np.linalg.norm(np.asarray(list(groups[key]))-centers[index],axis=1).max()>radius*1e-8:
            raise ValueError('Physical station camera centers differ')
        if np.any(np.linalg.norm(centers[index+1:]-centers[index],axis=1)<=radius*1e-8):
            raise ValueError('Co-located cameras need physical station grouping')
    counts={key:np.asarray(value) for key,value in counts.items()}
    for name in ('sky','non_sky','down'):
        values=np.asarray(counts[name])
        if values.shape!=(len(means),) or values.dtype.kind not in 'iu' or (values<0).any() or (values>len(groups)).any():
            raise ValueError('Invalid physical-station evidence counts')
    semantic=(counts['sky']>=options['minimum_sky_stations'])&(counts['sky']>=options['sky_to_non_sky_ratio']*counts['non_sky'])
    soft=scales.max(1)>=math.log(radius*options['soft_size_ratio'])
    down=soft&(counts['down']>=options['minimum_down_stations']) if options['protect_down'] else np.zeros(len(means),bool)
    previous=semantic|(soft&~down)
    below=np.zeros(len(means),bool)
    if options['protect_camera_below']:
        candidate=np.flatnonzero(previous)
        flag,_=below_camera_centers(means[candidate],np.stack([np.asarray(frame['transform_matrix'])[:3,3] for frame in frames]),
            [frame['station_id'] for frame in frames],up,options['nearest_camera_stations'])
        below[candidate]=flag
    hard=scales.max(1)>=math.log(radius*options['hard_size_ratio']) if options['hard_size_ratio'] is not None else np.zeros(len(means),bool)
    remove=(previous&~below)|hard
    return remove,dict(semantic_remove=semantic,soft_size_remove=soft,down_protected_size=down&~semantic,
                      camera_below_protected=previous&below,hard_size_remove=hard,
                      restored_semantic=semantic&below,scene_radius=radius)


def _load_inputs(camera_json, source, header, mask_manifest, options):
    """Validate all identities/grids before projection, using no implicit resize."""
    from PIL import Image
    camera=_camera_reference(camera_json,source,header)
    doc=json.loads(Path(camera_json).read_text(encoding='utf-8-sig'))
    manifest_path=Path(mask_manifest).resolve()
    manifest=json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if manifest.get('schema_version')!=1 or manifest.get('cameras_sha256')!=camera['sha256'] or not isinstance(manifest.get('frames'),list):
        raise ValueError('Native cleanup manifest must be bound to the camera JSON')
    bindings={str(manifest_path):_sha(manifest_path),str(Path(camera_json).resolve()):camera['sha256']}
    for value,digest in manifest.get('source_bindings',{}).items():
        path=(manifest_path.parent/Path(value)).resolve()
        if not isinstance(digest,str) or not path.is_file() or _sha(path)!=digest:
            raise ValueError('Native cleanup source manifest binding changed')
        if str(path) in bindings and bindings[str(path)]!=digest:
            raise ValueError('Native cleanup source bindings conflict')
        bindings[str(path)]=digest
    rows={}
    for row in manifest['frames']:
        key=(str(row['station_id']),row['face'],row['image'])
        if key in rows: raise ValueError('Duplicate native cleanup input identity')
        rows[key]=row
    camera_rows={}
    for item in doc['frames']:
        frame={**{key:doc[key] for key in ('w','h','fl_x','fl_y','cx','cy') if key in doc},**item}
        key=(str(frame['station_id']),frame['face'],frame['file_path'])
        if key in camera_rows:
            if camera_rows[key]!=frame: raise ValueError('Conflicting duplicate camera frame')
            continue
        camera_rows[key]=frame
    if set(rows)!=set(camera_rows): raise ValueError('Native mask and camera rosters differ')
    up,evidence=_world_up(doc,header,camera['camera_convention'])
    records=[]

    def bind(row,name):
        value=row.get(name+'_path');digest=row.get(name+'_sha256')
        if not isinstance(value,str) or not value or not isinstance(digest,str) or len(digest)!=64:
            raise ValueError(f'Missing bound {name} input')
        path=(manifest_path.parent/Path(value)).resolve()
        if not path.is_file() or _sha(path)!=digest: raise ValueError(f'Cleanup input changed: {name}')
        bindings[str(path)]=digest
        return path

    for key in sorted(camera_rows):
        frame,row=camera_rows[key],rows[key]
        width,height=frame['w'],frame['h']
        if type(width) is not int or type(height) is not int or min(width,height)<1:
            raise ValueError('Native image dimensions must be positive integers')
        K=np.array([[frame['fl_x'],0,frame['cx']],[0,frame['fl_y'],frame['cy']],[0,0,1.]],np.float64)
        if not np.isfinite(K).all() or min(K[0,0],K[1,1])<=0: raise ValueError('Invalid camera intrinsics')
        c2w=np.asarray(frame['transform_matrix'],np.float64)
        cv=c2w@np.diag([1.,-1.,-1.,1.]) if camera['camera_convention']=='OpenGL_c2w' else c2w
        view=np.linalg.inv(cv)
        if (row.get('width')!=width or row.get('height')!=height or not np.allclose(row.get('K'),K,rtol=0,atol=1e-8)
                or not np.allclose(row.get('camera_from_world'),view,rtol=0,atol=1e-8)):
            raise ValueError('Native semantic mask pose/intrinsics/grid differs')
        image=bind(row,'source_rgb');sky_path=bind(row,'source_sky');valid_path=bind(row,'source_generic_valid');alpha_path=bind(row,'source_edit_alpha')
        known_image=frame.get('image_sha256',frame.get('source_sha256'))
        if known_image and known_image!=row['source_rgb_sha256']: raise ValueError('Cleanup RGB differs from registered frame')
        with Image.open(image) as image_data:
            if image_data.size!=(width,height): raise ValueError('Native source RGB dimensions differ')
        if 'source_original_path' in row:
            original=bind(row,'source_original')
            known_original=frame.get('original_file_sha256',frame.get('original_sha256'))
            if known_original and known_original!=row['source_original_sha256']:
                raise ValueError('Cleanup original RGB differs from registered source')
            with Image.open(original) as image_data:
                if image_data.size!=(width,height): raise ValueError('Native original RGB dimensions differ')
        with Image.open(sky_path) as data: sky_u8=np.asarray(data.convert('L'))
        with Image.open(valid_path) as data: valid_u8=np.asarray(data.convert('L'))
        alpha=np.load(alpha_path,allow_pickle=False)
        if sky_u8.shape!=(height,width) or valid_u8.shape!=sky_u8.shape or alpha.shape!=sky_u8.shape:
            raise ValueError('Native semantic mask/alpha grid differs')
        if not np.isin(sky_u8,[0,255]).all() or not np.isin(valid_u8,[0,255]).all():
            raise ValueError('Native semantic and validity masks must be binary')
        record=dict(frame=frame,sky=sky_u8==255,valid=(valid_u8==255)&np.isfinite(alpha)&(alpha<=0),K=K,view=view)
        if options['depth_non_sky_abstention']:
            depth_path=bind(row,'depth')
            with np.load(depth_path,allow_pickle=False) as arrays:
                depth=arrays['depth_z'].copy();valid=arrays['evidence_valid'].copy();depth_k=arrays['K'].copy()
                expected=dict(source_image_sha256=row['source_rgb_sha256'],station_id=str(frame['station_id']),image=frame['file_path'],
                    world_frame=camera['coordinate_frame'],depth_convention='camera_z')
                if any(str(arrays[name])!=value for name,value in expected.items()) or str(arrays['unit']) not in ('metres','meters','m') or float(arrays['pixel_center_offset'])!=.5:
                    raise ValueError('Depth identity/frame/units/convention differs')
                if not np.allclose(arrays['camera_from_world'],view,rtol=0,atol=1e-8): raise ValueError('Depth pose differs')
                accepted=bool(arrays['metric_calibration_accepted'])
            if depth.ndim!=2 or valid.shape!=depth.shape or valid.dtype!=bool:
                raise ValueError('Depth grid or validity differs')
            expected_k=K.copy();expected_k[0]*=depth.shape[1]/width;expected_k[1]*=depth.shape[0]/height
            if not np.allclose(depth_k,expected_k,rtol=0,atol=1e-8): raise ValueError('Depth camera grid differs')
            record.update(depth=depth,depth_valid=valid,depth_K=depth_k,metric_calibration_accepted=accepted)
        records.append(record)
    if options['protect_down'] and not any(str(record['frame']['face']).lower() in ('d','down') for record in records):
        raise ValueError('Down-camera protection requested but no Down face exists')
    return camera,records,up,evidence,bindings


def _observe(means, records, options, device):
    import torch
    from ..streetview_geometry.semantic_center import SemanticCenterConfig,prepare_semantic_center,observe_semantic_center
    from ..streetview_geometry.center_occlusion import prepare_center_occlusion,observe_center_occlusion
    if device not in ('cpu','cuda') or (device=='cuda' and not torch.cuda.is_available()):
        raise ValueError('Requested cleanup compute device is unavailable')
    data=torch.as_tensor(means,dtype=torch.float64,device=device)
    n=len(means);packed={key:[] for key in ('sky','non_sky','down')};stations=[]
    current=None;flags=None

    def flush():
        if current is not None:
            stations.append(current)
            for key in packed: packed[key].append(np.packbits(flags[key].cpu().numpy(),bitorder='big'))

    for record in records:
        station=str(record['frame']['station_id'])
        if current!=station:
            flush();current=station;flags={key:torch.zeros(n,dtype=torch.bool,device=device) for key in packed}
        prepared=prepare_semantic_center(torch.as_tensor(record['sky'],device=device),torch.as_tensor(record['valid'],device=device),
                                         config=SemanticCenterConfig(options['erosion_radius_px']))
        view=torch.as_tensor(record['view'],device=device);K=torch.as_tensor(record['K'],device=device)
        observed=observe_semantic_center(data,view,K,prepared)
        flags['sky']|=observed['sky_view']
        non_sky=observed['non_sky_view']
        if str(record['frame']['face']).lower() in ('d','down'): flags['down']|=non_sky
        if options['depth_non_sky_abstention']:
            depth=prepare_center_occlusion(torch.as_tensor(record['depth']*options['depth_factor'],device=device),
                                           torch.as_tensor(record['depth_valid'],device=device))
            occluded=observe_center_occlusion(data,view,torch.as_tensor(record['depth_K'],device=device),depth)
            non_sky=non_sky&~occluded['occluded_view']
        flags['non_sky']|=non_sky
    flush()
    packed={key:np.stack(value) for key,value in packed.items()}
    counts={key:np.unpackbits(value,axis=1,count=n,bitorder='big').sum(0,dtype=np.int32) for key,value in packed.items()}
    return counts,packed,stations


def run_sky_cleanup(source_ply,camera_json,mask_manifest,output_dir,options=None,*,device='cpu'):
    options=read_sky_cleanup_options(options)
    source,header,offset=_ply(source_ply)
    output=Path(output_dir).resolve()
    if output.exists(): raise ValueError('Sky cleanup output directory must be new')
    camera,records,up,up_evidence,bindings=_load_inputs(camera_json,source,header,mask_manifest,options)
    rows=np.memmap(source['path'],mode='r',dtype='<f4',offset=offset,shape=(source['vertex_count'],len(source['fields'])))
    try:
        means=np.column_stack([rows[:,source['fields'].index(key)] for key in ('x','y','z')]).astype(np.float64)
        scales=np.column_stack([rows[:,source['fields'].index(f'scale_{axis}')] for axis in range(3)]).astype(np.float64)
    finally: rows._mmap.close()
    counts,packed,stations=_observe(means,records,options,device)
    remove,details=select_cleanup_rows(means,scales,[record['frame'] for record in records],up,counts,options)
    if remove.all(): raise ValueError('Sky cleanup would remove all Gaussians')
    bindings[source['path']]=source['sha256']
    for path,digest in bindings.items():
        if _sha(path)!=digest: raise ValueError('Sky cleanup input changed while processing')
    output.mkdir(parents=True)
    artifact=_write_subset(source,header,offset,~remove,output/'scene.ply')
    shutil.copyfile(camera_json,output/'cameras.json')
    if 'metadata' in camera: shutil.copyfile(camera['metadata']['path'],output/'dataset_manifest.json')
    evidence_path=output/'evidence.npz'
    np.savez_compressed(evidence_path,**{key+'_counts':value for key,value in counts.items()},
        **{key+'_packed':value for key,value in packed.items()},station_ids=np.asarray(stations),source_rows=np.int64(len(means)))
    selection_path=output/'selection.npz'
    np.savez_compressed(selection_path,removed_indices=np.flatnonzero(remove).astype('<i8'),source_vertex_count=np.array(len(means),dtype='<i8'))
    report=dict(schema_version=1,policy=POLICY,status='completed',source=source,artifact=artifact,options=options,
        camera_reference=camera,world_up=up.tolist(),up_evidence=up_evidence,input_bindings=bindings,
        evidence=dict(path=str(evidence_path),sha256=_sha(evidence_path),bitorder='big'),
        selection=dict(path=str(selection_path),sha256=_sha(selection_path),options_sha256=_json_sha(options)),
        removed_rows=int(remove.sum()),remaining_rows=int((~remove).sum()),
        policy_counts={key:int(value.sum()) for key,value in details.items() if isinstance(value,np.ndarray)},
        scene_radius=details['scene_radius'],attributes_unchanged=True,ground_plane_inferred=False,
        depth_scope='Explicit heuristic: depth only abstains raw non-sky center votes; never creates positive sky votes' if options['depth_non_sky_abstention'] else 'No depth evidence used',
        all_depth_metric_calibration_accepted=all(record.get('metric_calibration_accepted',False) for record in records) if options['depth_non_sky_abstention'] else None,
        rendering_quality='No cross-location quality claim; center evidence is not Gaussian footprint or visibility proof')
    verify_sky_cleanup_report(report,source_ply,artifact['path'],camera_json,evidence_path,selection_path)
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    write_output_camera_reference(report,output/'cameras.json',output/'artifact_cameras.json')
    return report


def verify_sky_cleanup_report(report,source_ply,filtered_ply,camera_json,evidence_npz,selection_npz):
    if report.get('schema_version')!=1 or report.get('policy')!=POLICY or report.get('status')!='completed':
        raise ValueError('Unsupported sky cleanup report')
    options=read_sky_cleanup_options(report['options'])
    source,header,offset=_ply(source_ply)
    camera=_camera_reference(camera_json,source,header)
    for path,digest in report['input_bindings'].items():
        if _sha(path)!=digest: raise ValueError('Sky cleanup bound input changed')
    if _sha(evidence_npz)!=report['evidence']['sha256'] or _sha(selection_npz)!=report['selection']['sha256'] or _json_sha(options)!=report['selection']['options_sha256']:
        raise ValueError('Sky cleanup evidence/selection binding changed')
    doc=json.loads(Path(camera_json).read_text(encoding='utf-8-sig'))
    up,_=_world_up(doc,header,camera['camera_convention'])
    with np.load(evidence_npz,allow_pickle=False) as data:
        stations=data['station_ids'].astype(str).tolist()
        if stations!=sorted({str(frame['station_id']) for frame in doc['frames']}) or int(data['source_rows'])!=source['vertex_count']:
            raise ValueError('Sky evidence station/source roster differs')
        counts={}
        for key in ('sky','non_sky','down'):
            packed=data[key+'_packed']
            if packed.dtype!=np.uint8 or packed.shape!=(len(stations),(source['vertex_count']+7)//8):
                raise ValueError('Sky evidence packed shape differs')
            counts[key]=np.unpackbits(packed,axis=1,count=source['vertex_count'],bitorder='big').sum(0,dtype=np.int32)
            if not np.array_equal(counts[key],data[key+'_counts']): raise ValueError('Physical-station vote counts differ')
    rows=np.memmap(source['path'],mode='r',dtype='<f4',offset=offset,shape=(source['vertex_count'],len(source['fields'])))
    try:
        means=np.column_stack([rows[:,source['fields'].index(key)] for key in ('x','y','z')]).astype(np.float64)
        scales=np.column_stack([rows[:,source['fields'].index(f'scale_{axis}')] for axis in range(3)]).astype(np.float64)
    finally: rows._mmap.close()
    remove,details=select_cleanup_rows(means,scales,doc['frames'],up,counts,options)
    if remove.all(): raise ValueError('Sky cleanup empty result')
    with np.load(selection_npz,allow_pickle=False) as data:
        if (set(data.files)!={'removed_indices','source_vertex_count'} or data['removed_indices'].dtype!=np.dtype('<i8')
                or data['source_vertex_count'].shape!=() or int(data['source_vertex_count'])!=len(remove)
                or not np.array_equal(data['removed_indices'],np.flatnonzero(remove))):
            raise ValueError('Sky cleanup policy row selection differs')
    artifact=_verify_exact_rows(source,header,offset,filtered_ply,~remove)
    for key in ('sha256','vertex_count','bytes','fields','sh_degree','format'):
        if source[key]!=report['source'].get(key) or artifact[key]!=report['artifact'].get(key):
            raise ValueError('Sky cleanup source/output binding differs')
    for key in ('sha256','physical_stations','frames','scene_radius','center','coordinate_frame','units','camera_convention'):
        if camera[key]!=report['camera_reference'].get(key): raise ValueError('Sky cleanup camera binding differs')
    if (report['removed_rows']!=int(remove.sum()) or report['remaining_rows']!=int((~remove).sum())
            or report['world_up']!=up.tolist() or report['scene_radius']!=details['scene_radius']
            or report['policy_counts']!={key:int(value.sum()) for key,value in details.items() if isinstance(value,np.ndarray)}):
        raise ValueError('Sky cleanup summary differs')
    return dict(status='verified',exact_retained_row_bytes=True,removed_rows=int(remove.sum()),artifact_sha256=artifact['sha256'])
