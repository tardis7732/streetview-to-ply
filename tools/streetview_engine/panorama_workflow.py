"""Explicit panorama edit -> SfM-seeded Brush/refinement -> native sky cleanup.

No image-comparison candidate is generated or selected by this workflow.
The accepted recipe is applied once to the user's registered scene data.
"""
from pathlib import Path
import json

from .export import sha256, write_json, validate_ply


def workflow_options(config, settings):
    if settings.get('workflow') != 'panorama_brush_refine' or config.get('generation_mode', 'multi_view') != 'multi_view':
        raise ValueError('Explicit multi-view panorama workflow required')
    from .sfm import validate_split_policy
    validate_split_policy(settings)
    from .depth_cleanup_options import validate_depth_cleanup_support
    depth_cleanup = validate_depth_cleanup_support(config, settings)
    # This named recipe has two fixed optimization stages, rather than a hidden generic trainer.
    for key, expected in [('training_steps',6000), ('resolution',1280), ('max_splats',2000000)]:
        if config.get(key) != expected:
            raise ValueError(f'This fixed Brush/refinement recipe requires {key}={expected}')
    raw = settings.get('sky_cleanup', {'enabled':False})
    if not isinstance(raw, dict) or set(raw)-{'enabled','device','policy'} or type(raw.get('enabled')) is not bool:
        raise ValueError('Sky cleanup requires explicit operator options')
    from .sky_cleanup import read_sky_cleanup_options
    policy = read_sky_cleanup_options(raw.get('policy', {}))
    enabled = raw['enabled'] if depth_cleanup is None else depth_cleanup['enabled']
    if enabled and policy['hard_size_ratio'] is not None:
        raise ValueError('The final size cap belongs to the separate export stage; use hard_size_ratio=null here')
    device = raw.get('device', 'cuda')
    if device not in ('cuda','cpu'):
        raise ValueError('Unknown cleanup device')
    return dict(raw, enabled=enabled, policy=policy, device=device)


def run(config, job_dir, settings):
    from . import brush_refine
    from .sky_cleanup import build_mask_manifest, run_sky_cleanup
    options = workflow_options(config, settings)
    root = Path(job_dir).resolve()
    dataset, output = root/'sfm/dataset', root/'training'
    cameras = dataset/'transforms_train.json'
    inputs = {name:sha256(dataset/name) for name in ('transforms_train.json','transforms_heldout.json','dataset_manifest.json','init.ply')}
    brush_refine._operator_settings(settings)
    native_manifest = None
    if options['enabled']:
        depth_manifest = None
        if options['policy']['depth_non_sky_abstention']:
            from .da3_raw_depth import validate_assets, infer_raw_depth
            if Path(settings.get('sky_depth', {}).get('gpu_lock', '')).resolve() == root.parent/'compute.lock':
                raise ValueError('The remote supervisor owns compute.lock; the depth subprocess needs its separate lock')
            validate_assets(settings.get('sky_depth', {}))
            infer_raw_depth(dataset, cameras, root/'cleanup/depth', settings['sky_depth'])
            depth_manifest = root/'cleanup/depth/manifest.json'
        native_manifest = build_mask_manifest(dataset, cameras, root/'cleanup/native_inputs.json', depth_manifest)
    state = brush_refine.run(config, root, settings)
    try:
        if any(sha256(dataset/name) != digest for name,digest in inputs.items()):
            raise ValueError('Bound SfM camera or initialization changed during training')
        state['provenance'] = dict(inputs=inputs, initialization='actual_sfm_xyzrgb',
            workflow='panorama_brush_refine', scene_specific_exceptions=False)
        if options['enabled']:
            state.update(status='running', stage='sky_cleanup')
            write_json(output/'manifest.json',state)
            raw_artifact = validate_ply(output/'model.ply')
            report = run_sky_cleanup(output/'model.ply', cameras, native_manifest,
                output/'sky_cleanup', options['policy'], device=options['device'])
            state['source_artifact'] = raw_artifact
            state['artifact'] = validate_ply(output/'sky_cleanup/scene.ply')
            state['selection'] = dict(accepted_model='sky_cleanup/scene.ply', accepted_model_sha256=state['artifact']['sha256'],
                selected='user_selected_recipe_cleanup', basis='Native semantic evidence, camera-below protection, exact row deletion')
            state['cleanup'] = dict(report='sky_cleanup/report.json', sha256=sha256(output/'sky_cleanup/report.json'),
                input_manifest_sha256=sha256(native_manifest), source_sha256=raw_artifact['sha256'],
                removed_rows=report['removed_rows'], remaining_rows=report['remaining_rows'],
                ground_plane_inferred=False, depth_scope=report['depth_scope'])
            state['source_quality'] = state.pop('quality', None)
            state['quality'] = None
            state['quality_scope'] = 'Cleanup is an exact row subset; source evaluation does not establish filtered visual quality'
            state['sky_policy'] = 'Native sky votes with Down and camera-below preservation; hard size cap is applied at export'
        if 'depth_cleanup' in config:
            state['depth_cleanup'] = dict(enabled=options['enabled'],
                status='applied' if options['enabled'] else 'skipped',
                scope='Combined image-depth inference and semantic/depth Gaussian cleanup',
                training_depth_loss_unchanged=True, export_size_filter_unchanged=True)
        state['status'] = 'completed'
        write_json(output/'manifest.json',state)
        return state
    except BaseException as error:
        state.update(status='failed', error=str(error))
        write_json(output/'manifest.json',state)
        raise
