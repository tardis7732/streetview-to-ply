"""Explicit per-job switch for the combined depth/semantic Gaussian cleanup.

Omission inherits the operator's historical sky_cleanup setting. This switch
does not change training depth losses, input image masks, or the export size
filter. Configuration inspection never loads a model or probes remote files.
"""
import re


def read_depth_cleanup_options(config):
    """Return None for a legacy config, or a fresh explicit boolean option."""
    if not isinstance(config, dict):
        raise ValueError('Job configuration must be an object')
    if 'depth_cleanup' not in config:
        return None
    raw = config['depth_cleanup']
    if not isinstance(raw, dict) or set(raw) != {'enabled'} or type(raw['enabled']) is not bool:
        raise ValueError('depth_cleanup must contain only an explicit enabled boolean')
    if raw['enabled'] and config.get('generation_mode', 'multi_view') != 'multi_view':
        raise ValueError('Depth cleanup is available only for multi-view generation')
    return dict(enabled=raw['enabled'])


def validate_depth_cleanup_support(config, settings):
    """Reject unsupported explicit enablement before collecting/training.

    Only configured asset declarations are read here: Windows GUI processes can
    hold Linux cloud paths. The existing DA3 validate_assets path still checks
    actual pinned files and revisions on the execution host before inference.
    """
    option = read_depth_cleanup_options(config)
    if option is None or not option['enabled']:
        return option
    if not isinstance(settings, dict) or settings.get('workflow') != 'panorama_brush_refine':
        raise ValueError('Depth cleanup requires the panorama_brush_refine operator workflow')
    cleanup = settings.get('sky_cleanup')
    if (not isinstance(cleanup, dict) or set(cleanup)-{'enabled', 'device', 'policy'}
            or type(cleanup.get('enabled')) is not bool):
        raise ValueError('Depth cleanup requires configured operator sky_cleanup options')
    policy = cleanup.get('policy')
    if not isinstance(policy, dict) or policy.get('depth_non_sky_abstention') is not True:
        raise ValueError('Depth cleanup requires policy.depth_non_sky_abstention=true')
    from .sky_cleanup import read_sky_cleanup_options
    policy = read_sky_cleanup_options(policy)
    if policy['hard_size_ratio'] is not None:
        raise ValueError('Depth cleanup requires hard_size_ratio=null; the size filter is a separate export option')
    if cleanup.get('device', 'cuda') not in ('cuda', 'cpu'):
        raise ValueError('Unknown cleanup device')
    depth = settings.get('sky_depth')
    if not isinstance(depth, dict):
        raise ValueError('Depth cleanup requires configured sky_depth metric and pose assets')
    assets = depth.get('da3_raw_depth', depth)
    if (not isinstance(assets, dict) or set(assets)-{'pose', 'metric', 'side', 'gpu_lock', 'extra_python'}
            or type(assets.get('side', 504)) is not int or assets.get('side', 504) != 504):
        raise ValueError('Depth cleanup requires the configured 504-pixel raw DA3 recipe')
    required = {'repo_path', 'model_path', 'repo_revision', 'model_sha256', 'config_sha256', 'model_name'}
    for name in ('metric', 'pose'):
        spec = assets.get(name)
        if (not isinstance(spec, dict) or set(spec) != required
                or any(not isinstance(value, str) or not value.strip() for value in spec.values())
                or any(not re.fullmatch('[0-9a-f]{64}', spec[key]) for key in ('model_sha256', 'config_sha256'))):
            raise ValueError('Depth cleanup requires complete pinned DA3 '+name+' asset settings')
    return option
