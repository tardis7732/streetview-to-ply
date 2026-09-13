"""Scene-independent, per-job masking options shared by the GUI and engine."""


def read_processing_options(config):
    """Validate options and return a fresh, complete dictionary.

    Missing fields retain the legacy engine behavior: preserve sky appearance
    and mask people/vehicles. Merely reading an older config must not change
    its serialized contents or the cache hashes derived from those contents.
    """
    if not isinstance(config, dict):
        raise ValueError('Job configuration must be an object')
    options = config.get('processing_options', {})
    if not isinstance(options, dict):
        raise ValueError('processing_options must be an object')
    defaults = dict(remove_sky=False, mask_dynamic=True)
    if set(options) - set(defaults):
        raise ValueError('Unknown processing_options fields')
    result = defaults.copy()
    for key, value in options.items():
        if type(value) is not bool:
            raise ValueError(f'processing_options.{key} must be a boolean')
        result[key] = value
    return result
