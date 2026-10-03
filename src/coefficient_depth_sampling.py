"""Depth-specific coefficient filters around an unchanged native pair sampler.

This adapter is for serial inference in an isolated evaluation process. It
retains native cached event order and commits each coefficient before the next
pair. It changes categorical filtering, never physical bins or coefficients.
The temporary native-helper override is restored even if sampling fails.
"""
from collections.abc import Sequence
from inspect import unwrap
import math
from numbers import Real

_ACTIVE = '__laser_coefficient_depth_sampling_active__'


def depth_settings(value, depth, *, name, probability=False):
    """Validate a scalar or exactly one setting per generation depth."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != depth:
            raise ValueError(f'{name} needs exactly {depth} depth settings')
        values = tuple(value)
    else:
        values = (value,) * depth
    result = []
    for item in values:
        if probability and item is None:
            result.append(None)
            continue
        if (isinstance(item, bool) or not isinstance(item, Real)
                or not math.isfinite(float(item)) or item <= 0
                or (probability and item > 1)):
            interval = '(0, 1] or None' if probability else 'finite and positive'
            raise ValueError(f'{name} entries must be {interval}')
        result.append(float(item))
    return tuple(result)


def _native_namespace(model):
    # The shared decoder obtains its helper from its inherited compound method.
    # Inspect class definitions rather than importing a second runtime module.
    for cls in type(model).__mro__:
        method = cls.__dict__.get('sample_compound')
        if method is not None:
            namespace = unwrap(method).__globals__
            if 'sample_from_logits' in namespace:
                return namespace
    raise ValueError('Native categorical sampling helper was not found')


def sample_compound_depthwise(model, batch_size, model_aux, *, coeff_top_p=.92,
                             coeff_temperature=None, temperature=1., **native_kwargs):
    """Use scalar or depth-list coefficient temperature/nucleus settings.

    Atom settings are forwarded unchanged. Uniform depth lists use the native
    sampler directly, preserving outputs and random-number consumption exactly.
    Nonuniform lists intercept only coefficient draws, in the native raster and
    depth order. Do not invoke this adapter concurrently in the same process.
    """
    shape = tuple(model.block_size)
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError('Expected a positive spatial/depth block_size')
    depth = shape[-1]
    physical_vector_prior = (callable(getattr(model, 'spatial_context_from_vectors', None))
                             and callable(getattr(model, '_row_log_probabilities', None))
                             and getattr(model, 'depth_specific_coeff_heads', False))
    if not (getattr(model, 'interleaved_vector_decoder', False)
            or physical_vector_prior
            or (getattr(model, 'pair_autoregressive', False)
                and not getattr(model, 'causal_prefix_state', False))):
        raise ValueError('A complete-pair-autoregressive compound sampler is required')
    if any(module.training for module in model.modules()):
        raise ValueError('Call model.eval() before depth-specific sampling')
    if tuple(model_aux.coeff_scales.shape) != (depth,):
        raise ValueError('Physical coefficient scale count does not match depth')
    probabilities = depth_settings(coeff_top_p, depth, name='coeff_top_p', probability=True)
    temperatures = depth_settings(temperature if coeff_temperature is None else coeff_temperature,
                                  depth, name='coeff_temperature')
    namespace = _native_namespace(model)
    if namespace.get(_ACTIVE, False):
        raise RuntimeError('Depth-specific sampler cannot be nested or concurrent')
    arguments = dict(native_kwargs, temperature=temperature,
                     coeff_temperature=temperatures[0], coeff_top_p=probabilities[0])
    if len(set(probabilities)) == len(set(temperatures)) == 1:
        try:
            return model.sample_compound(batch_size, model_aux, **arguments)
        finally:
            # Historical compound samplers did not clear caches on exceptions.
            model.init_cache()
    original = namespace['sample_from_logits']
    calls = 0

    def draw(logits, **settings):
        nonlocal calls
        is_coefficient = calls % 2 == 1
        expected = model.coeff_vocab_size if is_coefficient else model.num_atoms
        if logits.ndim != 2 or logits.shape[-1] != expected:
            raise RuntimeError('Native pair sampling order or vocabulary changed')
        if is_coefficient:
            d = (calls // 2) % depth
            settings = dict(settings, temperature=temperatures[d], top_p=probabilities[d])
        calls += 1
        return original(logits, **settings)

    namespace[_ACTIVE] = True
    namespace['sample_from_logits'] = draw
    try:
        result = model.sample_compound(batch_size, model_aux, **arguments)
        if calls != 2 * math.prod(shape):
            raise RuntimeError('Native sampler did not visit every pair exactly once')
        return result
    finally:
        namespace['sample_from_logits'] = original
        namespace.pop(_ACTIVE, None)
        model.init_cache()
