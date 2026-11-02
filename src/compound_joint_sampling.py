"""Joint decoding of the existing compound atom/coefficient distribution.

The transformer state is evaluated once per pair. Candidate coefficients use
only the trained atom-conditioned coefficient head, without branching caches.
Prior proposals give a Monte Carlo approximation; enumerating every atom gives
the exact joint law before optional pair nucleus filtering.
"""
from dataclasses import dataclass
import math
from numbers import Integral

import torch

from src.compound_ancestral_sampling import categorical_probabilities


@dataclass(frozen=True)
class CompoundJointPolicy:
    candidate_atoms: int = 32
    atom_proposal: str = 'prior'
    atom_temperature: float = 1.
    atom_top_k: int = 256
    atom_top_p: float | None = .95
    coefficient_temperatures: tuple = (1., 1., 1., 1.)
    joint_top_p: float | None = .95


def atom_proposals(probabilities, count, *, mode='prior', generator=None):
    """Combine repeated prior draws without multiplying by p(atom) twice."""
    if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
        raise ValueError('candidate count must be positive')
    if mode == 'top':
        weights, atoms = probabilities.topk(min(count, probabilities.shape[-1]), dim=-1)
        return atoms, weights
    if mode != 'prior':
        raise ValueError('atom proposal must be prior or top')
    atoms = torch.multinomial(probabilities, count, replacement=True, generator=generator)
    equal = atoms[:, :, None] == atoms[:, None, :]
    duplicate = (equal & torch.ones(count, count, device=atoms.device,
                                    dtype=torch.bool).tril(-1)).any(-1)
    weights = (equal.sum(-1).float() / count).masked_fill(duplicate, 0.)
    return atoms, weights


@torch.no_grad()
def candidate_coefficient_logits(model, aux, hidden, atoms, *, depth):
    """Score candidate atoms through the coefficient head, preserving caches."""
    if hidden.ndim != 2 or atoms.ndim != 2 or len(hidden) != len(atoms):
        raise ValueError('expected hidden [batch,width] and atoms [batch,candidates]')
    batch, count = atoms.shape
    contexts = hidden[:, None].expand(batch, count, hidden.shape[-1]).reshape(batch * count, -1)
    vectors = aux.dictionary.T[atoms.long()].reshape(batch * count, -1)
    logits = model.coefficient_logits(contexts, vectors, depth_index=depth)
    return logits.reshape(batch, count, model.coeff_vocab_size)


def joint_probabilities(weights, coefficient_logits, *, temperature=1., top_p=None):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('coefficient temperature must be positive and finite')
    if weights.shape != coefficient_logits.shape[:-1] or weights.ndim != 2:
        raise ValueError('expected weights [batch,candidates] and logits [batch,candidates,bins]')
    if not torch.isfinite(weights).all() or (weights < 0).any() or not (weights.sum(-1) > 0).all():
        raise ValueError('candidate weights must be nonnegative with positive row mass')
    if torch.isnan(coefficient_logits).any() or not torch.isfinite(coefficient_logits).any(-1).all():
        raise ValueError('each candidate requires a valid conditional distribution')
    log_coefficients = (coefficient_logits.float() / temperature).log_softmax(-1)
    logits = weights.float().log()[..., None] + log_coefficients
    # A mass of one means no joint truncation. Avoid dropping tiny tail
    # probabilities when FP32 cumulative sums round to one early.
    mass = None if top_p == 1. else top_p
    return categorical_probabilities(logits.flatten(1), top_p=mass).reshape_as(logits)


@torch.no_grad()
def sample_compound_joint(model, batch_size, aux, cond=None, *, policy=None,
                          amp=True, generator=None):
    policy = policy or CompoundJointPolicy()
    if not getattr(model, 'pair_autoregressive', False) or getattr(model, 'causal_prefix_state', False):
        raise ValueError('full-pair compound autoregression is required')
    if any(module.training for module in model.modules()):
        raise ValueError('call model.eval() before joint sampling')
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError('batch size must be positive')
    height, width, depths = model.block_size
    if model.num_atoms < depths or len(policy.coefficient_temperatures) != depths:
        raise ValueError('depth temperatures or distinct-atom vocabulary mismatch')
    if (aux.dictionary.ndim != 2 or aux.dictionary.shape[1] != model.num_atoms
            or tuple(aux.coeff_bins.shape) != (model.coeff_vocab_size,)
            or tuple(aux.coeff_scales.shape) != (depths,)):
        raise ValueError('frozen auxiliary shape mismatch')
    device = next(model.parameters()).device
    atoms = torch.zeros(batch_size, height, width, depths, device=device, dtype=torch.long)
    coefficient_ids = torch.full_like(atoms, model.coeff_vocab_size // 2)
    packed = atoms * model.coeff_vocab_size + coefficient_ids
    model.init_cache()
    try:
        for h in range(height):
            for w in range(width):
                for depth in range(depths):
                    hidden = model.cached_head_output(packed, aux, cond, (h, w, depth), amp=amp)
                    atom_logits = model.classifier(hidden)
                    if depth:
                        atom_logits = atom_logits.clone()
                        atom_logits.scatter_(1, atoms[:, h, w, :depth], -float('inf'))
                    prior = categorical_probabilities(atom_logits,
                        temperature=policy.atom_temperature,
                        top_k=policy.atom_top_k, top_p=policy.atom_top_p)
                    candidates, weights = atom_proposals(prior, policy.candidate_atoms,
                        mode=policy.atom_proposal, generator=generator)
                    logits = candidate_coefficient_logits(model, aux, hidden, candidates, depth=depth)
                    joint = joint_probabilities(weights, logits,
                        temperature=policy.coefficient_temperatures[depth], top_p=policy.joint_top_p)
                    selected = torch.multinomial(joint.flatten(1), 1, generator=generator).squeeze(-1)
                    atom = candidates.gather(1, (selected // model.coeff_vocab_size)[:, None]).squeeze(-1)
                    coefficient = selected % model.coeff_vocab_size
                    atoms[:, h, w, depth] = atom
                    coefficient_ids[:, h, w, depth] = coefficient
                    packed[:, h, w, depth] = atom * model.coeff_vocab_size + coefficient
    finally:
        model.init_cache()
    return atoms, coefficient_ids
