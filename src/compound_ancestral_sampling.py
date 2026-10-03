"""Explicit conditional ancestral sampling for the existing compound RQ model.

Nash et al.'s DCTransformer samples categorical tuple fields in sequence. For
LASER's fixed-position representation the corresponding factorization is
``p(atom | completed pairs) * p(coefficient | atom, completed pairs)``. This is
already the native compound model's factorization, not a new learned model or
a claim to reproduce undocumented DCTransformer sampling hyperparameters.

The default policy draws from both full categorical vocabularies at temperature
one. Optional filters reproduce the existing RQ probability arithmetic. The
trained raster/depth order, dictionary, coefficient grids/scales, embeddings,
and cached body/depth transformers are unchanged. No coefficient is refitted.
"""
import math
from numbers import Integral, Real

import torch
from torch.nn import functional as F


def _validate_filter(temperature, top_k, top_p):
    if (isinstance(temperature, bool) or not isinstance(temperature, Real)
            or not math.isfinite(float(temperature)) or temperature <= 0):
        raise ValueError('temperature must be a finite positive number')
    if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k < 0):
        raise ValueError('top_k must be a nonnegative integer or None')
    if top_p is not None and (isinstance(top_p, bool) or not isinstance(top_p, Real)
            or not math.isfinite(float(top_p)) or not 0 < top_p <= 1):
        raise ValueError('top_p must be in (0, 1] or None')


def categorical_probabilities(logits, *, temperature=1., top_k=None, top_p=None):
    """Native RQ FP32 probabilities, without redundant full-vocabulary top-k sorting.

    ``top_k=None`` or zero disables top-k. Values at least the vocabulary size
    retain every entry without a redundant sort. ``top_p=None`` disables
    nucleus filtering; explicit ``top_p=1`` retains the native cumulative-rounding
    behavior and therefore must not be silently treated as None. Ties at the
    top-k threshold retain the same extra entries as the native implementation.
    """
    _validate_filter(temperature, top_k, top_p)
    if logits.ndim != 2 or logits.shape[-1] < 1 or not logits.is_floating_point():
        raise ValueError('logits must be a floating [batch, vocabulary] tensor')
    values = logits.float() / float(temperature)
    if top_k and top_k < values.shape[-1]:
        threshold = torch.topk(values, int(top_k)).values[..., [-1]]
        values = values.masked_fill(values < threshold, -float('inf'))
    # Preserve the native helper exactly, including its default +/-inf mapping.
    values = torch.nan_to_num(values, nan=-float('inf'))
    probabilities = F.softmax(values, dim=-1)
    if top_p is not None:
        ordered, indices = torch.sort(probabilities, descending=True)
        remove = torch.cumsum(ordered, dim=-1) >= float(top_p)
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        remove = remove.scatter(-1, indices, remove)
        probabilities = probabilities.masked_fill(remove, 0.)
        probabilities = probabilities / probabilities.sum(-1, keepdim=True).clamp_min(1e-12)
    return probabilities


def _draw(logits, temperature, top_k, top_p, generator):
    probabilities = categorical_probabilities(logits, temperature=temperature,
                                               top_k=top_k, top_p=top_p)
    return torch.multinomial(probabilities, num_samples=1, generator=generator).view(-1)


@torch.no_grad()
def sample_dc_ancestral(model, batch_size, model_aux, cond=None, amp=True,
                        atom_temperature=1., coeff_temperature=1.,
                        atom_top_k=None, coeff_top_k=None,
                        atom_top_p=None, coeff_top_p=None, generator=None):
    """Sample complete atom/coefficient pairs using the trained native RQ caches.

    Require ``model.eval()``: this function never changes module training flags,
    model parameters, auxiliary parameters, or precision flags. ``amp`` is
    forwarded only to the original ``cached_head_output``; no new outer autocast
    context changes coefficient-head precision. The native body/head caches are
    reset before sampling and in ``finally`` on success or failure.

    Only the existing full-pair-autoregressive, non-causal-prefix compound mode
    is supported. Other factorizations must use their own native samplers.
    Returned tensors are atom IDs and coefficient IDs, both ``[B,H,W,D]``.
    """
    _validate_filter(atom_temperature, atom_top_k, atom_top_p)
    _validate_filter(coeff_temperature, coeff_top_k, coeff_top_p)
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    if generator is not None and not isinstance(generator, torch.Generator):
        raise ValueError('generator must be a torch.Generator or None')
    if not isinstance(amp, bool):
        raise ValueError('amp must be a boolean')
    if not bool(getattr(model, 'pair_autoregressive', False)):
        raise ValueError('sampling requires the trained full-pair-autoregressive compound model')
    if bool(getattr(model, 'causal_prefix_state', False)):
        raise ValueError('causal-prefix-state models require their own native sampler')
    if any(module.training for module in model.modules()):
        raise ValueError('call model.eval() before conditional ancestral sampling')
    shape = tuple(model.block_size)
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError('model block_size must contain positive height, width and depth')
    height, width, depth_count = shape
    if model.num_atoms < depth_count or model.coeff_vocab_size < 1:
        raise ValueError('vocabulary cannot represent the distinct-atom sparse support')
    device = next(model.parameters()).device
    if (model_aux.dictionary.ndim != 2 or model_aux.dictionary.shape[1] != model.num_atoms
            or tuple(model_aux.coeff_bins.shape) != (model.coeff_vocab_size,)
            or tuple(model_aux.coeff_scales.shape) != (depth_count,)):
        raise ValueError('auxiliary dictionary, coefficient bins or depth scales mismatch')
    if any(value.device != device for value in (model_aux.dictionary, model_aux.coeff_bins, model_aux.coeff_scales)):
        raise ValueError('model and frozen auxiliary tensors must share a device')
    atoms = torch.zeros(batch_size, height, width, depth_count, device=device, dtype=torch.long)
    coefficient_ids = torch.full_like(atoms, model.coeff_vocab_size // 2)
    packed = atoms * model.coeff_vocab_size + coefficient_ids
    try:
        model.init_cache()
        for h in range(height):
            for w in range(width):
                for depth in range(depth_count):
                    hidden = model.cached_head_output(packed, model_aux, cond, (h, w, depth), amp=amp)
                    atom_logits = model.classifier(hidden)
                    if depth:
                        atom_logits = atom_logits.clone()
                        atom_logits.scatter_(1, atoms[:, h, w, :depth], -float('inf'))
                    atom = _draw(atom_logits, atom_temperature, atom_top_k, atom_top_p, generator)
                    selected_atom_vector = model_aux.dictionary.T[atom.long()]
                    refined = model.refine_coefficient_hidden(hidden, selected_atom_vector)
                    coefficient_logits = model.classify_coefficients(refined, depth_index=depth)
                    coefficient = _draw(coefficient_logits, coeff_temperature,
                                        coeff_top_k, coeff_top_p, generator)
                    # Commit both fields together. The next cached event sees
                    # the exact trained signed/scaled physical pair embedding.
                    atoms[:, h, w, depth] = atom
                    coefficient_ids[:, h, w, depth] = coefficient
                    packed[:, h, w, depth] = atom * model.coeff_vocab_size + coefficient
    finally:
        model.init_cache()
    return atoms, coefficient_ids
