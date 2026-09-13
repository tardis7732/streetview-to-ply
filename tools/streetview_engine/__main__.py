"""Explicit stage entry point; browsing a map never invokes this module."""
import argparse
import importlib
import json
from pathlib import Path

STAGES = {'collect': 'collection', 'preprocess': 'preprocess', 'sfm': 'sfm', 'train': 'experiment', 'export': 'export'}


def stage_module(stage, config, settings=None):
    """Explicit generation routes; a single input never implies a new mode."""
    mode = config.get('generation_mode', 'multi_view')
    if mode != 'multi_view':
        raise ValueError('Only multi_view generation is available')
    if stage not in STAGES:
        raise ValueError('Stage is unavailable for the configured generation mode')
    workflow = (settings or {}).get('workflow')
    if workflow not in (None, 'panorama_brush_refine'):
        raise ValueError('Unknown operator workflow')
    from .depth_cleanup_options import validate_depth_cleanup_support
    validate_depth_cleanup_support(config, settings or {})
    if mode == 'multi_view':
        from .sfm import validate_split_policy
        validate_split_policy(settings or {})
    if mode == 'multi_view' and workflow == 'panorama_brush_refine':
        if stage == 'preprocess':
            return 'panorama_preprocess'
        if stage == 'train':
            return 'panorama_workflow'
    return STAGES[stage]


def main():
    parser = argparse.ArgumentParser(description='Run one configured streetview reconstruction stage')
    parser.add_argument('stage', choices=STAGES)
    parser.add_argument('--job-config', type=Path, required=True)
    parser.add_argument('--job-dir', type=Path, required=True)
    parser.add_argument('--settings', type=Path, required=True)
    args = parser.parse_args()
    from tools.streetview_app.jobs import validate_config
    config = validate_config(json.loads(args.job_config.read_text(encoding='utf8')))
    settings = json.loads(args.settings.read_text(encoding='utf8'))
    if not isinstance(settings, dict):
        raise ValueError('Engine settings must be an operator-owned JSON object')
    selected_module = stage_module(args.stage, config, settings)
    job_dir = args.job_dir.resolve()
    job_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == 'collect' and settings.get('input_cache') is not None:
        from .input_cache import reuse_inputs
        cached = reuse_inputs(config, job_dir, settings)
        summary = {key: cached[key] for key in ('status', 'signature', 'source_job_dir', 'files_count', 'bytes', 'hardlinked_files', 'copied_files') if key in cached}
        print(json.dumps(dict(event='input_cache', result=summary), ensure_ascii=False), flush=True)
    module = importlib.import_module('tools.streetview_engine.' + selected_module)
    print(json.dumps(dict(event='stage_start', stage=args.stage)), flush=True)
    if args.stage in ('collect','preprocess') and settings.get('workflow')=='panorama_brush_refine' and settings.get('input_cache') is not None:
        from .panorama_input_cache import cached_stage
        result = cached_stage(config, job_dir, settings, args.stage)
    else:
        result = module.run(config, job_dir, settings)
    print(json.dumps(dict(event='stage_complete', stage=args.stage, result=result), ensure_ascii=False, default=str), flush=True)


if __name__ == '__main__':
    main()
