"""Publication contract fixtures; synthetic saved renders, no CUDA or network."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest

from tools.streetview_engine import postprocess,sky_refine,training
from tools.streetview_engine.export import sha256,validate_ply,write_json,write_model
from tools.streetview_engine.quality import masked_metrics,summarize_views
from tools.streetview_engine.sky_environment import SkyEnvironmentConfig
from tools.streetview_engine.tests.test_sky_refine import cuda_fixture


def candidate_fixture(root,*,accepted=True):
    """A declared fake renderer receipt only for testing publication checks."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    foreground,dataset=cuda_fixture(root)
    fit=sky_refine.SkyRefineConfig(steps=2,resolution=32,log_every=1)
    builder=SkyEnvironmentConfig(grid_size=8,tangent_sigma_cells=.35)
    settings=dict(sky_refine=dict(enabled=True,fit=asdict(fit),builder=asdict(builder)))
    source,arrays,artifact,manifest,train,heldout,groups,heldout_groups,sources=sky_refine._source_inputs(foreground,dataset)
    env,sky_report=sky_refine._expanded_sky(dataset,train,heldout,arrays,builder)
    candidate=root/'sky_candidate';candidate.mkdir()
    write_json(candidate/'sky_builder.json',sky_report)
    for name in ['sky_initial.ply','sky_fitted.ply']:
        write_model(candidate/name,means=env.means,log_scales=env.log_scales,quats=env.quats,
            opacity_logits=env.opacity_logits,sh0=env.sh0,shN=env.shN)
    identity=sky_refine.append_frozen_foreground(source,candidate/'sky_fitted.ply',candidate/'sky_combined.ply')
    center,radius=training.normalization(train)
    frames=[training.prepare_frame(frame,dataset,fit.resolution,center,radius) for frame in heldout]
    expected=sky_refine._expected(frames,groups);write_json(candidate/'evaluation_manifest.json',expected)
    def render_report(name,model_sha,is_candidate):
        directory=candidate/name;directory.mkdir();(directory/'evaluation').mkdir()
        rows=[]
        for index,frame in enumerate(frames):
            reference=frame['rgb'].astype(np.float32)/255
            is_sky=frame['meta']['face']=='F'
            prediction=reference.copy() if not is_sky or is_candidate and accepted else np.zeros_like(reference)
            alpha=np.full((frame['h'],frame['w']),1. if not is_sky or is_candidate else 0.,np.float32)
            row=dict(frame=frame['frame'],station_id=frame['station_id'],reference_rgb_sha256=frame['reference_rgb_sha256'],
                reference_mask_sha256=frame['reference_mask_sha256'],reference_sky_mask_sha256=frame['meta']['sky_mask_sha256'],
                reference_foreground_mask_sha256=frame['meta']['foreground_mask_sha256'],
                **masked_metrics(prediction,reference,alpha,frame['mask'],alpha_threshold=fit.alpha_threshold))
            for category in ['foreground','sky']:
                row[category]=masked_metrics(prediction,reference,alpha,frame[category],alpha_threshold=fit.alpha_threshold)
            if frame['meta']['face']=='D':row['down']=masked_metrics(prediction,reference,alpha,frame['mask'],alpha_threshold=fit.alpha_threshold)
            png=f'evaluation/{index:06d}.png';npz=f'evaluation/{index:06d}.npz'
            Image.fromarray(np.rint(prediction*255).astype(np.uint8)).save(directory/png)
            np.savez_compressed(directory/npz,alpha=alpha,mask=frame['mask'])
            row.update(rgb_png=png,alpha_npz=npz,width=frame['w'],height=frame['h']);rows.append(row)
        result=dict(status='measured',model_sha256=model_sha,views=rows,summary=summarize_views(rows),
            scope='synthetic publication-contract test only; not real gsplat render evidence')
        write_json(directory/'heldout_metrics.json',result)
        return result
    before=render_report('source_reference',artifact['sha256'],False)
    after=render_report('candidate',identity['candidate_sha256'],True)
    write_json(foreground/'heldout_metrics.json',before);write_json(foreground/'evaluation_manifest.json',expected)
    manifest['selection']=dict(accepted_model='model.ply',accepted_model_sha256=artifact['sha256'],selected='baseline')
    manifest['quality']=dict(heldout=before['summary'],geometry=dict(gaussians=artifact['vertex_count']),quality_improved=False)
    write_json(foreground/'manifest.json',manifest)
    sources=sky_refine._source_inputs(foreground,dataset)[-1]
    write_json(candidate/'inputs.json',dict(sources=sources,config=asdict(fit),sky_config=asdict(builder),
        code={name:sha256(Path(sky_refine.__file__).with_name(name)) for name in ['sky_refine.py','sky_environment.py','training.py','quality.py','export.py']}))
    holes=sky_refine._new_holes(candidate/'source_reference',candidate/'candidate',before,after,frames,fit.alpha_threshold)
    comparison=sky_refine.compare_sky_candidate(before,after,expected_manifest=expected,new_holes=holes,foreground_identity=identity,
        source_sha256=artifact['sha256'],candidate_sha256=identity['candidate_sha256'],config=fit)
    assert comparison['accepted'] is accepted
    write_json(candidate/'comparison.json',dict(**comparison,new_holes=holes,foreground_identity=identity))
    write_json(candidate/'manifest.json',dict(status='completed',accepted=accepted,source_sha256=artifact['sha256'],
        source_training_manifest_sha256=sha256(foreground/'manifest.json'),nonzero_sky_gradient=True,baseline_unchanged=True,promoted=False,
        candidate_ply='sky_combined.ply',candidate_sha256=identity['candidate_sha256'],sky_ply='sky_fitted.ply',foreground_identity=identity,
        source_reference_metrics='source_reference/heldout_metrics.json',candidate_metrics='candidate/heldout_metrics.json',
        comparison='comparison.json',comparison_sha256=sha256(candidate/'comparison.json'),
        fixture='synthetic contract fixture; does not assert actual CUDA fit'))
    return settings,artifact['sha256'],sha256(foreground/'manifest.json')


def test_accepted_candidate_publishes_exact_model_and_actual_evaluation(tmp_path):
    settings,source,manifest_hash=candidate_fixture(tmp_path)
    result=postprocess.finalize_sky({},tmp_path,settings)
    root=tmp_path/'training';candidate=tmp_path/'sky_candidate'
    assert result['sky_postprocess']['accepted'] and result['selection']['selected']=='foreground_plus_shared_sky'
    assert result['selection']['accepted_model_sha256']==sha256(root/'model.ply')==sha256(candidate/'sky_combined.ply')
    assert result['artifact']['sha256']==result['selection']['accepted_model_sha256']
    assert sha256(root/'foreground_model.ply')==source
    assert sha256(root/'foreground_training_manifest.json')==manifest_hash
    assert result['foreground_selection']['accepted_model_sha256']==source
    assert result['checkpoint_scope']['component']=='foreground_only' and result['checkpoint_scope']['contains_sky'] is False
    assert result['quality']['geometry']['gaussians']==4
    assert result['quality']['sky_component']['combined_rows']>4
    assert result['quality']['sky_component']['geometry_is_measured'] is False
    metrics=json.loads((root/'heldout_metrics.json').read_text())
    assert metrics['model_sha256']==result['artifact']['sha256']
    assert all(row['rgb_png'].startswith('sky_evaluation/') and (root/row['alpha_npz']).is_file() for row in metrics['views'])
    assert result['selection']['comparison_sha256']==sha256(root/result['selection']['comparison_report'])


def test_rejected_candidate_keeps_foreground_and_report(tmp_path):
    settings,source,_=candidate_fixture(tmp_path,accepted=False)
    result=postprocess.finalize_sky({},tmp_path,settings)
    assert result['sky_postprocess']['status']=='rejected'
    assert sha256(tmp_path/'training/model.ply')==source
    assert result['selection']['selected']=='baseline' and result['artifact']['sha256']==source
    assert result['quality']['quality_improved'] is False
    assert (tmp_path/'sky_candidate/comparison.json').is_file()


def test_idempotence_checks_config_and_all_published_bytes(tmp_path):
    settings,_,_=candidate_fixture(tmp_path)
    first=postprocess.finalize_sky({'training_steps':6000},tmp_path,settings)
    assert postprocess.finalize_sky({'training_steps':6000},tmp_path,settings)==first
    with pytest.raises(ValueError,match='another configuration'):
        postprocess.finalize_sky({'training_steps':7000},tmp_path,settings)
    image=next((tmp_path/'training/sky_evaluation').glob('*.png'));image.write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact changed'):
        postprocess.finalize_sky({'training_steps':6000},tmp_path,settings)


@pytest.mark.parametrize('fault',['accepted_flag','comparison_flag','render_pixels','foreground_payload','reference_mask','candidate_metrics','radius'])
def test_manual_flags_and_tampered_evidence_never_publish(tmp_path,fault):
    settings,source,_=candidate_fixture(tmp_path,accepted=fault not in ['accepted_flag','comparison_flag'])
    candidate=tmp_path/'sky_candidate';manifest=json.loads((candidate/'manifest.json').read_text())
    if fault=='accepted_flag':manifest['accepted']=True;write_json(candidate/'manifest.json',manifest)
    elif fault=='comparison_flag':
        comparison=json.loads((candidate/'comparison.json').read_text());comparison['accepted']=True
        write_json(candidate/'comparison.json',comparison);manifest['accepted']=True
        manifest['comparison_sha256']=sha256(candidate/'comparison.json');write_json(candidate/'manifest.json',manifest)
    elif fault=='render_pixels':
        image=next((candidate/'candidate/evaluation').glob('*.png'));Image.new('RGB',(32,32),'red').save(image)
    elif fault=='foreground_payload':
        info,_,offset=sky_refine._layout(candidate/'sky_combined.ply')
        data=np.memmap(candidate/'sky_combined.ply',mode='r+',dtype='<f4',offset=offset,shape=(info['vertex_count'],len(info['fields'])))
        data[0,0]+=1;data.flush();data._mmap.close()
    elif fault=='reference_mask':
        path=next((tmp_path/'sfm/dataset').glob('*sky_mask_path.png'));Image.new('L',(64,64),127).save(path)
    elif fault=='candidate_metrics':
        path=candidate/'candidate/heldout_metrics.json';report=json.loads(path.read_text());report['views'].pop()
        write_json(path,report)
    elif fault=='radius':
        path=candidate/'sky_builder.json';report=json.loads(path.read_text());report['shell_radius_m']=1.
        write_json(path,report)
    with pytest.raises(ValueError):postprocess.finalize_sky({},tmp_path,settings)
    assert sha256(tmp_path/'training/model.ply')==source


def test_identical_candidate_can_be_reused_in_a_new_review_root(tmp_path):
    old=tmp_path/'old';settings,source,_=candidate_fixture(old)
    review=tmp_path/'review';shutil.copytree(old,review)
    before={p.relative_to(old):sha256(p) for p in old.rglob('*') if p.is_file()}
    result=postprocess.finalize_sky({},review,settings)
    assert result['sky_postprocess']['accepted'] and sha256(review/'training/foreground_model.ply')==source
    assert before=={p.relative_to(old):sha256(p) for p in old.rglob('*') if p.is_file()}


def test_commit_interruption_recovers_only_bound_before_after_files(tmp_path,monkeypatch):
    settings,source,_=candidate_fixture(tmp_path)
    original=postprocess.os.replace
    failed=[False]
    def stop_manifest(source_path,target):
        if Path(target)==tmp_path/'training/manifest.json' and '.sky_commit' in str(source_path) and not failed[0]:
            failed[0]=True;raise OSError('injected final manifest interruption')
        return original(source_path,target)
    monkeypatch.setattr(postprocess.os,'replace',stop_manifest)
    with pytest.raises(OSError,match='injected'):
        postprocess.finalize_sky({},tmp_path,settings)
    assert sha256(tmp_path/'training/model.ply')!=source
    assert json.loads((tmp_path/'training/manifest.json').read_text())['artifact']['sha256']==source
    result=postprocess.finalize_sky({},tmp_path,settings)
    assert result['sky_postprocess']['accepted'] and result['artifact']['sha256']==sha256(tmp_path/'training/model.ply')


def test_disabled_finalizer_does_not_create_artifacts(tmp_path):
    foreground,_=cuda_fixture(tmp_path)
    before={p.relative_to(tmp_path):sha256(p) for p in tmp_path.rglob('*') if p.is_file()}
    result=postprocess.finalize_sky({},tmp_path,{})
    assert result['status']=='completed'
    assert before=={p.relative_to(tmp_path):sha256(p) for p in tmp_path.rglob('*') if p.is_file()}
