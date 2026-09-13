"""Verified optional sky publication after foreground selection, before export.

Sky refinement remains nonmutating. This finalizer preserves the original
foreground artifacts and publishes only a fully bound, independently checked
candidate. A saved boolean is never sufficient authorization for publication.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
from PIL import Image

from . import sky_refine, training
from .export import sha256, validate_ply, write_json
from .imaging import inside
from .quality import summarize_views
from .sky_environment import SkyEnvironmentConfig
from .processing_options import read_processing_options


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def _copy_exact(source, target):
    source, target=Path(source),Path(target)
    expected=sha256(source)
    if target.is_symlink():raise ValueError('Publication target is a symlink')
    if target.exists():
        if sha256(target)!=expected:raise ValueError('Existing preserved artifact differs: '+str(target))
        return expected
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_name(target.name+'.'+uuid.uuid4().hex+'.sky_tmp')
    shutil.copyfile(source,temporary)
    if sha256(temporary)!=expected or sha256(source)!=expected:
        raise ValueError('Source changed during immutable copy')
    temporary.replace(target)
    return expected


def _options(settings):
    options=settings.get('sky_refine',{})
    if not options or isinstance(options,dict) and options.get('enabled') is False:return None
    if not isinstance(options,dict) or options.get('enabled') is not True or set(options)-{'enabled','fit','builder'}:
        raise ValueError('Sky finalization requires enabled=true and fit/builder settings')
    return sky_refine.SkyRefineConfig(**options.get('fit',{})),SkyEnvironmentConfig(**options.get('builder',{}))


def _input_roles(root, source_model, source_manifest):
    """Bind current bytes to the completed foreground's original dataset."""
    baseline=_read(source_manifest);artifact=validate_ply(source_model)
    if baseline.get('status')!='completed' or baseline.get('artifact',{}).get('sha256')!=artifact['sha256']:
        raise ValueError('Foreground source is not completed or its model hash changed')
    dataset=root/'sfm/dataset';dm=_read(dataset/'dataset_manifest.json')
    train=_read(dataset/'transforms_train.json')['frames'];heldout=_read(dataset/'transforms_heldout.json')['frames']
    groups,heldout_groups=training.validate_splits(train,heldout)
    training.validate_dataset_manifest(dm,groups,heldout_groups)
    if not heldout:raise ValueError('Sky publication needs a complete heldout roster')
    original=baseline.get('provenance',{}).get('inputs',{})
    roles={'training/model.ply':artifact['sha256'],'training/manifest.json':sha256(source_manifest)}
    for name in ['dataset_manifest.json','transforms_train.json','transforms_heldout.json']:
        value=sha256(inside(dataset,name))
        if original.get(name)!=value:raise ValueError('Foreground dataset binding changed: '+name)
        roles['sfm/dataset/'+name]=value
    for frame in train+heldout:
        for key in ['file_path','mask_path','foreground_mask_path','sky_mask_path']:
            name=frame[key];value=sha256(inside(dataset,name))
            if original.get('photos_and_masks',{}).get(name)!=value:
                raise ValueError('Foreground reference binding changed: '+name)
            roles['sfm/dataset/'+name]=value
    return baseline,artifact,train,heldout,groups,roles


def _canonical_roles(sources):
    """Allow content-identical review-root clones, never arbitrary references."""
    result={}
    if not isinstance(sources,dict):raise ValueError('Missing candidate input sources')
    for name,value in sources.items():
        normalized=str(name).replace('\\','/')
        if '/sfm/dataset/' in normalized:
            role='sfm/dataset/'+normalized.rsplit('/sfm/dataset/',1)[1]
        elif normalized.endswith('/training/model.ply'):role='training/model.ply'
        elif normalized.endswith('/training/manifest.json'):role='training/manifest.json'
        else:raise ValueError('Unrecognized candidate source role')
        if role in result or '..' in Path(role).parts:raise ValueError('Duplicate/escaping candidate source role')
        result[role]=value
    return result


def _verify_saved_render(directory, report, frames, options):
    """Bind float-alpha metrics to saved images, allowing PNG quantization only."""
    rows={row['frame']:row for row in report.get('views',[])}
    if len(rows)!=len(report.get('views',[])) or set(rows)!={f['frame'] for f in frames}:
        raise ValueError('Saved render roster is incomplete or duplicated')
    delta=.5/255+1e-7
    for frame in frames:
        row=rows[frame['frame']]
        with Image.open(inside(directory,row['rgb_png'])) as image:
            if image.size!=(frame['w'],frame['h']):raise ValueError('Saved RGB dimensions differ')
            prediction=np.asarray(image.convert('RGB'),np.float64)/255
        with np.load(inside(directory,row['alpha_npz']),allow_pickle=False) as values:
            alpha=values['alpha'].copy();mask=values['mask'].copy()
        if alpha.shape!=frame['mask'].shape or not np.isfinite(alpha).all() or np.any((alpha<0)|(alpha>1)) or not np.array_equal(mask,frame['mask']):
            raise ValueError('Saved alpha/photometric reference differs')
        reference=frame['rgb'].astype(np.float64)/255
        difference=np.abs(prediction-reference)
        categories=dict(all=frame['mask'],foreground=frame['foreground'],sky=frame['sky'])
        if str(frame['meta'].get('face','')).lower() in ('d','down'):categories['down']=frame['mask']
        for category,region in categories.items():
            metric=row if category=='all' else row.get(category,{})
            count=int(region.sum())
            if not sky_refine._metric_valid(metric,count) or metric.get('alpha_threshold')!=options.alpha_threshold:
                raise ValueError('Missing/invalid saved '+category+' render metric')
            if metric['covered_pixels']!=int((region&(alpha>=options.alpha_threshold)).sum()):
                raise ValueError('Reported coverage differs from actual alpha')
            sse=float(np.square(difference[region]).sum())
            tolerance=float((2*difference[region]*delta+delta**2).sum())+1e-7
            if abs(sse-metric['static_sse'])>tolerance:
                raise ValueError('Reported RGB error is inconsistent with saved PNG pixels')
            if count and metric.get('masked_l1') is not None and abs(float(difference[region].mean())-metric['masked_l1'])>delta+1e-6:
                raise ValueError('Reported L1 is inconsistent with saved PNG pixels')
            if count:
                psnr=-10*math.log10(max(metric['static_sse']/(3*count),1e-12))
                if metric.get('masked_psnr_db') is None or abs(metric['masked_psnr_db']-psnr)>1e-9:
                    raise ValueError('Reported PSNR differs from saved error sum')
                if metric.get('coverage_fraction')!=metric['covered_pixels']/count or metric.get('alpha_mean') is None or abs(metric['alpha_mean']-float(alpha[region].mean()))>1e-7:
                    raise ValueError('Reported alpha summary differs from saved alpha')
    if report.get('summary')!=summarize_views(report['views']):
        raise ValueError('Saved render summary differs from complete view metrics')


def _candidate_identity(source, candidate, sky):
    foreground,_,source_offset=sky_refine._layout(source)
    combined,_,candidate_offset=sky_refine._layout(candidate)
    background,_,sky_offset=sky_refine._layout(sky)
    if foreground['fields']!=combined['fields'] or foreground['fields']!=background['fields'] or combined['vertex_count']!=foreground['vertex_count']+background['vertex_count']:
        raise ValueError('Combined PLY component schema/count differs')
    size=foreground['vertex_count']*len(foreground['fields'])*4
    left=sky_refine._hash_payload(source,source_offset)
    right=sky_refine._hash_payload(candidate,candidate_offset,size)
    if left!=right:raise ValueError('Combined PLY changed original foreground attribute bytes')
    if sky_refine._hash_payload(candidate,candidate_offset+size)!=sky_refine._hash_payload(sky,sky_offset):
        raise ValueError('Combined PLY sky suffix differs from fitted sky artifact')
    return dict(source_sha256=foreground['sha256'],candidate_sha256=combined['sha256'],
        source_foreground_payload_sha256=left,candidate_foreground_payload_sha256=right,
        foreground_rows=foreground['vertex_count'],sky_rows=background['vertex_count'],
        identity='exact_original_payload_prefix_including_all_attributes',unchanged=True)


def _file_hashes(root,directory):
    result={}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():raise ValueError('Candidate/publication evidence contains a symlink')
        if path.is_file():result[path.relative_to(root).as_posix()]=sha256(path)
    return result


def _verify_candidate(root,source_model,source_manifest,options,builder_options):
    baseline,artifact,train,heldout,groups,roles=_input_roles(root,source_model,source_manifest)
    candidate=root/'sky_candidate';record=_read(candidate/'manifest.json');inputs=_read(candidate/'inputs.json')
    builder_options=replace(builder_options,sh_degree=artifact['sh_degree'])
    expected_code={name:sha256(Path(__file__).with_name(name)) for name in ['sky_refine.py','sky_environment.py','training.py','quality.py','export.py']}
    if _canonical_roles(inputs.get('sources'))!=roles or inputs.get('config')!=asdict(options) or inputs.get('sky_config')!=asdict(builder_options) or inputs.get('code')!=expected_code:
        raise ValueError('Candidate inputs/settings/implementation are not bound to this foreground')
    if record.get('source_sha256')!=artifact['sha256']:
        raise ValueError('Candidate belongs to another foreground model')
    if record.get('status')!='completed':
        return dict(accepted=False,status=record.get('status','incomplete'),reasons=['sky_candidate_not_completed'],record=record,
            input_roles=roles,candidate_hashes=_file_hashes(root,candidate))
    if record.get('source_training_manifest_sha256')!=sha256(source_manifest) or record.get('nonzero_sky_gradient') is not True or record.get('baseline_unchanged') is not True or record.get('promoted') is not False:
        raise ValueError('Completed sky candidate lacks fixed-source/actual-fit provenance')
    model=inside(candidate,record['candidate_ply']);sky=inside(candidate,record['sky_ply'])
    identity=_candidate_identity(source_model,model,sky)
    if identity['candidate_sha256']!=record.get('candidate_sha256') or record.get('foreground_identity')!=identity:
        raise ValueError('Candidate model/payload identity record differs')
    initial,_=sky_refine.read_gaussian_model(candidate/'sky_initial.ply');fitted,_=sky_refine.read_gaussian_model(sky)
    if any(not np.array_equal(initial[key],fitted[key]) for key in ['means','scales','quats','shN']):
        raise ValueError('Sky geometry changed during appearance-only fit')
    sky_builder=_read(candidate/'sky_builder.json')
    region=sky_builder.get('expanded_view_region',{})
    origin=np.asarray(region.get('center'),np.float64);shell_radius=sky_builder.get('shell_radius_m')
    if origin.shape!=(3,) or not np.isfinite(origin).all() or type(shell_radius) not in (int,float) or not math.isfinite(shell_radius) or shell_radius<=0:
        raise ValueError('Sky component has no finite bound viewing region')
    cameras=np.asarray([np.asarray(f['transform_matrix'])[:3,3] for f in train+heldout])
    extent=float(np.linalg.norm(cameras-origin,axis=1).max())
    radial=np.linalg.norm(fitted['means'].astype(np.float64)-origin,axis=1)
    if not np.allclose(radial,shell_radius,rtol=2e-6,atol=0) or extent>=shell_radius or extent>region.get('radius_m',-1)*(1+1e-9):
        raise ValueError('Published sky radius/view ball differs from actual cameras or PLY means')
    if math.degrees(math.asin(extent/shell_radius))>builder_options.maximum_parallax_degrees+1e-8:
        raise ValueError('Finite sky exceeds the declared angular parallax tolerance')
    before=_read(inside(candidate,record['source_reference_metrics']));after=_read(inside(candidate,record['candidate_metrics']))
    center,radius=training.normalization(train)
    frames=[training.prepare_frame(f,root/'sfm/dataset',options.resolution,center,radius) for f in heldout]
    expected=sky_refine._expected(frames,groups)
    if _read(candidate/'evaluation_manifest.json')!=expected:raise ValueError('Sky heldout roster or reference masks changed')
    source_directory=inside(candidate,record['source_reference_metrics']).parent
    after_directory=inside(candidate,record['candidate_metrics']).parent
    _verify_saved_render(source_directory,before,frames,options);_verify_saved_render(after_directory,after,frames,options)
    holes=sky_refine._new_holes(source_directory,after_directory,before,after,frames,options.alpha_threshold)
    comparison=sky_refine.compare_sky_candidate(before,after,expected_manifest=expected,new_holes=holes,foreground_identity=identity,
        source_sha256=artifact['sha256'],candidate_sha256=identity['candidate_sha256'],config=options)
    comparison_path=inside(candidate,record['comparison'])
    stored=_read(comparison_path)
    if sha256(comparison_path)!=record.get('comparison_sha256') or stored!=dict(**comparison,new_holes=holes,foreground_identity=identity):
        raise ValueError('Stored acceptance/comparison differs from independently recomputed bound gate')
    if record.get('accepted') is not comparison['accepted']:
        raise ValueError('Standalone accepted flag disagrees with verified render evidence')
    return dict(accepted=comparison['accepted'],status='verified',reasons=comparison['reasons'],record=record,
        model_path=model,sky_path=sky,identity=identity,comparison=comparison,after=after,expected=expected,
        sky_builder=sky_builder,input_roles=roles,candidate_hashes=_file_hashes(root,candidate))


def _check_hashes(root,bindings):
    for relative,expected in bindings.items():
        path=inside(root,relative)
        if not path.is_file() or sha256(path)!=expected:raise ValueError('Bound sky publication artifact changed: '+relative)


def _resume_publication(root,journal,signature):
    """Recover only our declared before/after hashes; never guess old files."""
    if journal.get('signature')!=signature:raise ValueError('Existing sky finalization belongs to another configuration/source')
    _check_hashes(root,journal['evidence'])
    for entry in journal['writes']:
        source=inside(root,entry['staged']);target=inside(root,entry['target'])
        if sha256(source)!=entry['after']:raise ValueError('Staged publication changed')
        if target.is_symlink():raise ValueError('Publication target is a symlink')
        actual=sha256(target) if target.exists() else None
        if actual==entry['after']:continue
        if actual!=entry['before']:raise ValueError('Publication target is neither the verified old nor new artifact')
        temporary=target.with_name(target.name+'.'+uuid.uuid4().hex+'.sky_commit')
        shutil.copyfile(source,temporary)
        if sha256(temporary)!=entry['after']:raise ValueError('Publication copy hash mismatch')
        os.replace(temporary,target)
    _check_hashes(root,{entry['target']:entry['after'] for entry in journal['writes']})
    journal=dict(journal,status='completed')
    write_json(root/'training/sky_publication.json',journal)
    return _read(root/'training/manifest.json')


def finalize_sky(config,job_dir,settings):
    """Optional finalization; rejected sky retains the foreground artifact."""
    root=Path(job_dir).resolve();directory=root/'training'
    processing=read_processing_options(config)
    baseline=_read(directory/'manifest.json')
    if read_processing_options(baseline)!=processing:
        raise ValueError('Foreground processing options differ from frozen sky finalization configuration')
    if processing['remove_sky']:
        if (directory/'sky_publication.json').exists():
            raise ValueError('Sky removal cannot reuse a prior sky publication; use a new job')
        return baseline
    options=_options(settings)
    if options is None:return _read(directory/'manifest.json')
    fit,builder=options
    original_model=directory/'foreground_model.ply';original_manifest=directory/'foreground_training_manifest.json'
    journal_path=directory/'sky_publication.json'
    if original_model.exists()!=original_manifest.exists():raise ValueError('Incomplete preserved foreground snapshot')
    source_model=original_model if original_model.exists() else directory/'model.ply'
    source_manifest=original_manifest if original_manifest.exists() else directory/'manifest.json'
    baseline,artifact,train,heldout,groups,roles=_input_roles(root,source_model,source_manifest)
    signature=_digest(dict(config=config,settings=settings,inputs=roles,fit=asdict(fit),builder=asdict(replace(builder,sh_degree=artifact['sh_degree'])),
        implementation={name:sha256(Path(__file__).with_name(name)) for name in ['postprocess.py','sky_refine.py','sky_environment.py','training.py','quality.py','export.py']}))
    if journal_path.exists():return _resume_publication(root,_read(journal_path),signature)
    # Preserve exact source files before modifying any publication target.
    _copy_exact(source_model,original_model);_copy_exact(source_manifest,original_manifest)
    source_model,source_manifest=original_model,original_manifest
    snapshot_hashes={original_model.relative_to(root).as_posix():sha256(original_model),original_manifest.relative_to(root).as_posix():sha256(original_manifest)}
    for name in ['heldout_metrics.json','evaluation_manifest.json']:
        if (directory/name).is_file():
            preserved=directory/('foreground_'+name);_copy_exact(directory/name,preserved)
            snapshot_hashes[preserved.relative_to(root).as_posix()]=sha256(preserved)
    candidate=root/'sky_candidate'
    if not candidate.exists():
        # This is the only call that can launch CUDA. The caller already chose
        # foreground training and explicitly enabled this optional stage.
        sky_refine.refine_sky(directory,root/'sfm/dataset',candidate,config=fit,sky_config=builder)
    verified=_verify_candidate(root,source_model,source_manifest,fit,builder)
    if _input_roles(root,source_model,source_manifest)[-1]!=roles:raise ValueError('Source inputs changed during sky verification')
    publication=directory/'.sky_publication'
    if publication.exists():raise ValueError('Unjournaled prior publication is retained; use a fresh review root')
    publication.mkdir()
    result=deepcopy(baseline)
    component=None
    writes=[]
    evidence=dict(snapshot_hashes,**verified['candidate_hashes'])
    def stage_file(source,target_name):
        destination=publication/target_name
        _copy_exact(source,destination)
        target=directory/target_name
        writes.append(dict(staged=destination.relative_to(root).as_posix(),target=target.relative_to(root).as_posix(),
            before=sha256(target) if target.exists() else None,after=sha256(destination)))
    if verified['accepted']:
        record=verified['record'];identity=verified['identity'];candidate_artifact=validate_ply(verified['model_path'])
        stage_file(verified['model_path'],'model.ply')
        _copy_exact(candidate/record['comparison'],directory/'sky_comparison.json')
        evidence['training/sky_comparison.json']=sha256(directory/'sky_comparison.json')
        rendered=deepcopy(verified['after'])
        metric_source=inside(candidate,record['candidate_metrics']).parent
        for row in rendered['views']:
            for key in ['rgb_png','alpha_npz']:
                original=inside(metric_source,row[key]);relative='sky_evaluation/'+Path(row[key]).name
                target=directory/relative;_copy_exact(original,target)
                evidence[target.relative_to(root).as_posix()]=sha256(target);row[key]=relative
        write_json(publication/'heldout_metrics.json',rendered)
        write_json(publication/'evaluation_manifest.json',verified['expected'])
        for name in ['heldout_metrics.json','evaluation_manifest.json']:
            target=directory/name
            writes.append(dict(staged=(publication/name).relative_to(root).as_posix(),target=target.relative_to(root).as_posix(),
                before=sha256(target) if target.exists() else None,after=sha256(publication/name)))
        sky=verified['sky_builder']
        component=dict(kind='shared_angular_sky_fixed_geometry',rows=identity['sky_rows'],
            foreground_rows=identity['foreground_rows'],combined_rows=candidate_artifact['vertex_count'],
            row_start=identity['foreground_rows'],row_end_exclusive=candidate_artifact['vertex_count'],
            shell_radius_m=sky['shell_radius_m'],valid_view_region=sky.get('expanded_view_region',sky['valid_view_region']),
            finite_angular_approximation=True,geometry_is_measured=False,foreground_statistics_exclude_sky=True,
            model_sha256=sha256(verified['sky_path']),provenance='sky_candidate/sky_builder.json')
        result['foreground_selection']=deepcopy(baseline.get('selection'))
        result['foreground_artifact']=artifact
        result['foreground_quality']=deepcopy(baseline.get('quality',{}))
        result['artifact']=candidate_artifact
        result['selection']=dict(accepted_model='model.ply',accepted_model_sha256=candidate_artifact['sha256'],
            selected='foreground_plus_shared_sky',foreground_model='foreground_model.ply',
            comparison_report='sky_comparison.json',comparison_sha256=sha256(directory/'sky_comparison.json'),
            basis='Exact foreground bytes; independently verified full heldout appearance/zero-new-holes guard and measured sky RGB/coverage benefit')
        quality=deepcopy(baseline.get('quality',{}))
        quality.update(heldout=rendered['summary'],sky_coverage=rendered['summary']['sky'],sky_component=component,
            quality_improved=True,sky_appearance_improved=True,
            foreground_candidate_comparison=deepcopy(quality.get('candidate_comparison')),
            candidate_comparison=verified['comparison'])
        if isinstance(quality.get('geometry'),dict):
            quality['geometry']['component_scope']='foreground_model.ply only; distant sky excluded'
            quality['geometry']['model_sha256']=artifact['sha256']
        result['quality']=quality
        result['checkpoint_scope']=dict(component='foreground_only',model_sha256=artifact['sha256'],
            contains_sky=False,reason='Original foreground checkpoint retained; combined PLY has separately fitted sky appearance')
    result['sky_postprocess']=dict(status='accepted' if verified['accepted'] else 'rejected',accepted=verified['accepted'],
        signature=signature,candidate_status=verified['status'],reasons=verified['reasons'],
        candidate_manifest='sky_candidate/manifest.json',candidate_manifest_sha256=verified['candidate_hashes']['sky_candidate/manifest.json'],
        foreground_model_sha256=artifact['sha256'],sky_component=component,
        source_training_manifest_sha256=sha256(source_manifest),publication_journal='sky_publication.json')
    write_json(publication/'manifest.json',result)
    writes.append(dict(staged=(publication/'manifest.json').relative_to(root).as_posix(),target='training/manifest.json',
        before=sha256(directory/'manifest.json'),after=sha256(publication/'manifest.json')))
    journal=dict(schema_version=1,status='prepared',signature=signature,accepted=verified['accepted'],
        evidence=evidence,writes=writes,model_commit='Atomic per-file replace, manifest last; verified journal resumes interrupted commits')
    write_json(journal_path,journal)
    return _resume_publication(root,journal,signature)
