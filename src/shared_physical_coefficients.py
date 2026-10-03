"""Convert legacy depth-normalized coefficients to one shared physical grid.

Only the existing signed coefficients are rescaled, exactly once in FP32:
``physical = stored_normalized * legacy_depth_scale``. Atom IDs, order, sparse
support and dictionary remain unchanged; there is no OMP solve or fitting.

Continuous physical reconstruction is preserved by conversion. Replacing four
depth-specific quantized grids with one uniform grid introduces ordinary scalar
quantization error and changes coefficient-token meaning. Existing stage-two
weights therefore cannot silently be used with the converted cache.
"""
from copy import deepcopy
import hashlib
import json
import math
from numbers import Integral, Real

import torch


SHARED_PHYSICAL_REPRESENTATION = 'shared_physical_uniform_v1'
LEGACY_CACHE_FORMATS = frozenset({'laser_compound_pairs_v1', 'laser_stochastic_compound_bank_v1'})
_STALE_PAYLOAD_KEYS = frozenset({'bank_log_normalizers', 'coefficient_probs', 'coefficient_ids', 'packed', 'tokens'})


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _centers_digest(centers):
    return hashlib.sha256(centers.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _legacy_scales(meta, depth, device='cpu'):
    if not isinstance(meta, dict) or meta.get('format') not in LEGACY_CACHE_FORMATS:
        raise ValueError('recognized legacy compound-cache metadata is required')
    if (meta.get('coefficient_representation') not in (None, 'depth_normalized', 'legacy_depth_normalized_v1')
            or str(meta.get('coefficient_units', '')).startswith('physical')
            or meta.get('coefficient_normalization') == 'none'):
        raise ValueError('cache is already physical or has an unknown representation; refusing double conversion')
    if meta.get('clip_coefficients') is not False:
        raise ValueError('conversion requires an explicitly unclipped legacy cache')
    if str(meta.get('coefficient_storage', '')).lower() not in {'fp32', 'float32'}:
        raise ValueError('conversion requires legacy FP32 coefficient storage')
    if meta.get('causal_prefix_coeffs', False):
        raise ValueError('prefix-coefficient caches require a separate explicit conversion')
    if 'coeff_scales' not in meta:
        raise ValueError('legacy per-depth coefficient scales are required')
    scales = torch.as_tensor(meta['coeff_scales'], device=device, dtype=torch.float32)
    if scales.ndim != 1 or scales.numel() != depth or not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any()):
        raise ValueError('legacy scales must be finite, positive and match the last tensor dimension')
    return scales


def legacy_coefficient_centers(meta, *, device='cpu'):
    """Reconstruct the unchanged legacy normalized coefficient grid."""
    count = meta.get('coeff_vocab_size')
    if isinstance(count, bool) or not isinstance(count, Integral) or count < 2:
        raise ValueError('legacy coefficient vocabulary size must be an integer >=2')
    if meta.get('coeff_bin_centers') is not None:
        centers = torch.as_tensor(meta['coeff_bin_centers'], dtype=torch.float32, device=device)
    else:
        bound = meta.get('coeff_max')
        if isinstance(bound, bool) or not isinstance(bound, Real) or not math.isfinite(float(bound)) or bound <= 0:
            raise ValueError('legacy coeff_max must be finite and positive')
        centers = torch.linspace(-float(bound), float(bound), int(count), dtype=torch.float32).to(device)
    if (centers.ndim != 1 or centers.numel() != count or not bool(torch.isfinite(centers).all())
            or not bool((centers[1:] > centers[:-1]).all())):
        raise ValueError('legacy centers must be finite and strictly increasing')
    return centers


@torch.no_grad()
def legacy_physical_coefficients(coefficients, meta):
    """Recover physical FP32 coefficients from any ``[..., depth]`` tensor."""
    if not torch.is_tensor(coefficients) or coefficients.ndim < 1 or coefficients.numel() == 0:
        raise ValueError('coefficients must be a nonempty tensor with a final depth dimension')
    if coefficients.dtype != torch.float32:
        raise ValueError('legacy coefficients must be FP32; no silent lossy-storage conversion')
    scales = _legacy_scales(meta, coefficients.shape[-1], coefficients.device)
    if not bool(torch.isfinite(coefficients).all()):
        raise ValueError('legacy coefficients must be finite')
    with torch.autocast(device_type=coefficients.device.type, enabled=False):
        physical = coefficients.detach() * scales
    if not bool(torch.isfinite(physical).all()):
        raise ValueError('physical coefficient conversion overflowed')
    return physical


def _outward_float32_bound(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) or value <= 0:
        raise ValueError('shared physical bound must be finite and positive')
    bound = torch.tensor(float(value), dtype=torch.float32)
    if float(bound) < float(value):
        bound = torch.nextafter(bound, torch.tensor(float('inf'), dtype=torch.float32))
    if not bool(torch.isfinite(bound)):
        raise ValueError('shared bound cannot be represented in FP32')
    return float(bound)


def _validate_uniform_centers(centers):
    if (not torch.is_tensor(centers) or centers.ndim != 1 or centers.numel() < 2
            or centers.dtype != torch.float32 or not bool(torch.isfinite(centers).all())
            or not bool((centers[1:] > centers[:-1]).all())):
        raise ValueError('shared centers must be finite, ordered FP32 values')
    bound = float(centers[-1])
    if bound <= 0 or float(centers[0]) != -bound:
        raise ValueError('shared grid must have symmetric physical endpoints')
    # CPU FP32 serialization is the canonical vocabulary. Re-running linspace
    # on a different device can round interior centers differently; a device
    # transfer must preserve the serialized centers, never redefine them.
    expected = torch.linspace(-bound, bound, len(centers), dtype=torch.float32)
    if not torch.equal(centers.detach().cpu(), expected):
        raise ValueError('shared grid must be the specified uniform grid, without fitted centers')
    return bound


@torch.no_grad()
def build_shared_physical_grid(physical, meta, *, num_bins=2048,
                               additional_bound=None, cover_legacy_grid=True):
    """Uniform symmetric physical centers covering observed and old-grid values.

    ``additional_bound`` can include extrema audited from another cache so a
    deterministic cache and stochastic bank share exactly one grid. By default
    every value in every old depth grid is also covered, including values absent
    from the cache. This permits later explicit migration without clipping; it
    does not make old coefficient-token IDs compatible with the new grid.
    """
    if (not torch.is_tensor(physical) or physical.ndim < 1 or not physical.numel()
            or physical.dtype != torch.float32 or not bool(torch.isfinite(physical).all())):
        raise ValueError('physical coefficients must be finite, nonempty FP32 values')
    if isinstance(num_bins, bool) or not isinstance(num_bins, Integral) or num_bins < 2:
        raise ValueError('num_bins must be an integer >=2')
    if not isinstance(cover_legacy_grid, bool):
        raise ValueError('cover_legacy_grid must be a boolean')
    scales = _legacy_scales(meta, physical.shape[-1], physical.device)
    bound = float(physical.abs().max())
    if cover_legacy_grid:
        old = legacy_coefficient_centers(meta, device=physical.device)
        bound = max(bound, float((old[None] * scales[:, None]).abs().max()))
    if additional_bound is not None:
        bound = max(bound, _outward_float32_bound(additional_bound))
    bound = _outward_float32_bound(bound)
    centers = torch.linspace(-bound, bound, int(num_bins), dtype=torch.float32).to(physical.device)
    _validate_uniform_centers(centers)
    return centers


@torch.no_grad()
def nearest_coefficient_ids(values, centers):
    """Nearest shared physical bin, lower-ID ties, with explicit range rejection."""
    _validate_uniform_centers(centers)
    if (not torch.is_tensor(values) or not values.is_floating_point()
            or values.device != centers.device or not bool(torch.isfinite(values).all())):
        raise ValueError('coefficient values must be finite and share the grid device')
    if bool(((values < centers[0]) | (values > centers[-1])).any()):
        raise ValueError('physical coefficient outside the shared grid; clipping is forbidden')
    values = values.float().contiguous()
    right = torch.searchsorted(centers, values).clamp(1, len(centers) - 1)
    left = right - 1
    return torch.where((values - centers[left]).abs() <= (values - centers[right]).abs(), left, right).long()


def require_shared_physical_cache(meta, centers=None):
    """Validate explicit shared-physical semantics before an aux/cache is used."""
    if (not isinstance(meta, dict) or meta.get('format') not in LEGACY_CACHE_FORMATS
            or meta.get('coefficient_representation') != SHARED_PHYSICAL_REPRESENTATION
            or meta.get('coefficient_units') != 'physical'
            or meta.get('coefficient_normalization') != 'none'
            or meta.get('coefficient_storage') != 'fp32'
            or meta.get('coeff_scale') != 1.
            or meta.get('clip_coefficients') is not False):
        raise ValueError('explicit unclipped shared-physical cache metadata is required')
    scales = meta.get('coeff_scales')
    if not isinstance(scales, (list, tuple)) or not scales or any(float(x) != 1. for x in scales):
        raise ValueError('shared physical coefficients require every depth scale to equal one')
    stored = torch.as_tensor(meta.get('coeff_bin_centers', []), dtype=torch.float32)
    bound = _validate_uniform_centers(stored)
    if len(stored) != meta.get('coeff_vocab_size') or _centers_digest(stored) != meta.get('coeff_bin_centers_sha256'):
        raise ValueError('shared coefficient-grid identity mismatch')
    if meta.get('coeff_max') != bound:
        raise ValueError('runtime coefficient bound disagrees with the stored centers')
    if centers is not None and not torch.equal(stored, centers.detach().cpu()):
        raise ValueError('runtime centers disagree with cache physical-token meanings')
    fields = dict(representation=SHARED_PHYSICAL_REPRESENTATION,
        centers_sha256=_centers_digest(stored), coeff_vocab_size=len(stored),
        coeff_scales=[1.] * len(scales), stage1_checkpoint_sha256=meta.get('stage1_checkpoint_sha256'),
        num_atoms=meta.get('num_atoms'), depth=len(scales))
    if _digest(fields) != meta.get('coefficient_codec_identity'):
        raise ValueError('shared coefficient codec identity does not match its semantics')
    return stored


def require_compatible_checkpoint_codec(checkpoint_config, cache_meta):
    """Refuse an unchanged old normalized-vocabulary checkpoint for this grid."""
    centers = require_shared_physical_cache(cache_meta)
    if (checkpoint_config.get('coefficient_representation') != SHARED_PHYSICAL_REPRESENTATION
            or checkpoint_config.get('coefficient_codec_identity') != cache_meta['coefficient_codec_identity']
            or checkpoint_config.get('coeff_scales') != cache_meta['coeff_scales']
            or checkpoint_config.get('coeff_vocab_size') != len(centers)):
        raise ValueError('checkpoint coefficient vocabulary requires explicit migration or fresh training')
    saved_centers = checkpoint_config.get('coeff_bin_centers')
    if saved_centers is None or not torch.equal(torch.as_tensor(saved_centers, dtype=torch.float32), centers):
        raise ValueError('checkpoint coefficient centers do not match the shared physical grid')
    return True


def guard_shared_physical_checkpoint(checkpoint_config, cache_meta):
    """Guard both directions of a shared-grid transition at checkpoint load.

    Existing unrelated legacy checkpoint/cache pairs are unaffected. If either
    side declares the shared physical representation, both must explicitly
    agree before any model or optimizer tensors are loaded.
    """
    checkpoint_shared = (checkpoint_config.get('coefficient_representation') == SHARED_PHYSICAL_REPRESENTATION
                         or checkpoint_config.get('coefficient_codec_identity') is not None)
    cache_shared = cache_meta is not None and (
        cache_meta.get('coefficient_representation') == SHARED_PHYSICAL_REPRESENTATION
        or cache_meta.get('coefficient_codec_identity') is not None
        or cache_meta.get('coefficient_quantizer') == 'shared_uniform_physical')
    if checkpoint_shared or cache_shared:
        if cache_meta is None:
            raise ValueError('shared-physical checkpoints require an explicit compatible cache')
        return require_compatible_checkpoint_codec(checkpoint_config, cache_meta)
    return True


@torch.no_grad()
def convert_cache_payload(payload, *, source_sha256, centers=None, additional_bound=None):
    """Return a new cache; retain format/variants and invalidate old bin tables.

    Supports deterministic ``[N,H,W,D]`` and stochastic ``[N,H,W,V,D]`` layouts.
    Original tensors/metadata are not mutated. Coefficient-bin-dependent tables
    are deliberately removed and recorded, so an old soft-label loader cannot
    accidentally treat them as belonging to the new vocabulary.
    """
    if not isinstance(source_sha256, str) or len(source_sha256) != 64 or any(x not in '0123456789abcdef' for x in source_sha256):
        raise ValueError('verified source SHA256 is required')
    if not isinstance(payload, dict) or not {'atoms', 'coeffs', 'labels', 'meta'}.issubset(payload):
        raise ValueError('expected a complete compound cache payload')
    unknown = set(payload) - {'atoms', 'coeffs', 'labels', 'meta'} - _STALE_PAYLOAD_KEYS
    if unknown:
        raise ValueError(f'additional cache tensors require an explicit conversion policy: {sorted(unknown)}')
    atoms, coefficients, labels, original_meta = [payload[k] for k in ('atoms', 'coeffs', 'labels', 'meta')]
    if not isinstance(original_meta, dict):
        raise ValueError('cache metadata must be a dictionary')
    expected_ndim = 5 if original_meta.get('format') == 'laser_stochastic_compound_bank_v1' else 4
    if (not torch.is_tensor(atoms) or not torch.is_tensor(coefficients)
            or atoms.dtype not in (torch.int16, torch.int32, torch.int64)
            or atoms.shape != coefficients.shape or atoms.ndim != expected_ndim
            or not torch.is_tensor(labels) or labels.shape != (len(atoms),)):
        raise ValueError('paired cache atoms, coefficients and labels do not align')
    physical = legacy_physical_coefficients(coefficients, original_meta)
    count = original_meta['coeff_vocab_size']
    if centers is None:
        centers = build_shared_physical_grid(physical, original_meta, num_bins=count,
                                            additional_bound=additional_bound)
    else:
        centers = centers.detach().to(device=physical.device).clone()
        _validate_uniform_centers(centers)
        if len(centers) != count:
            raise ValueError('conversion must preserve the coefficient vocabulary size')
        minimum_grid = build_shared_physical_grid(physical, original_meta, num_bins=count,
                                                 additional_bound=additional_bound)
        if float(centers[-1]) < float(minimum_grid[-1]):
            raise ValueError('provided shared grid does not cover observed and full legacy physical ranges')
    depth = atoms.shape[-1]
    identity_fields = dict(representation=SHARED_PHYSICAL_REPRESENTATION,
        centers_sha256=_centers_digest(centers), coeff_vocab_size=len(centers),
        coeff_scales=[1.] * depth, stage1_checkpoint_sha256=original_meta.get('stage1_checkpoint_sha256'),
        num_atoms=original_meta.get('num_atoms'), depth=depth)
    codec_identity = _digest(identity_fields)
    invalidated = sorted(set(payload) & _STALE_PAYLOAD_KEYS)
    meta = deepcopy(original_meta)
    old_partition_temperature = meta.pop('coefficient_log_partition_temperature', None)
    old_auto_scale = meta.pop('auto_coeff_scales_percentile', None)
    old_targets = meta.pop('atom_targets', None)
    meta.update(coefficient_representation=SHARED_PHYSICAL_REPRESENTATION,
        coefficient_units='physical', coefficient_normalization='none', coefficient_storage='fp32',
        coefficient_quantizer='shared_uniform_physical', coeff_scales=[1.] * depth,
        coeff_scale=1., coeff_max=float(centers[-1]), coeff_vocab_size=len(centers),
        coeff_bin_centers=centers.cpu().tolist(), coeff_bin_centers_sha256=_centers_digest(centers),
        coefficient_codec_identity=codec_identity, coefficient_codec_identity_fields=identity_fields,
        source_cache_sha256=source_sha256, source_coeff_scales=list(original_meta['coeff_scales']),
        source_bank_identity=original_meta.get('bank_identity'), source_coefficient_metadata=deepcopy(original_meta),
        source_auto_coeff_scales_percentile=old_auto_scale, source_atom_targets=old_targets,
        source_coefficient_log_partition_temperature=old_partition_temperature,
        invalidated_coefficient_payload_keys=invalidated,
        bank_identity=_digest(dict(source_cache_sha256=source_sha256, coefficient_codec_identity=codec_identity)),
        clip_coefficients=False, soft_atom_targets=False,
        coefficient_bins_changed=True, requires_stage2_vocabulary_migration=True,
        legacy_stage2_checkpoint_compatible=False, refitting=False,
        conversion='FP32 stored normalized coefficient times legacy depth scale exactly once; shared uniform physical bins')
    require_shared_physical_cache(meta, centers)
    return dict(atoms=atoms.detach().clone(), coeffs=physical, labels=labels.detach().clone(), meta=meta)
