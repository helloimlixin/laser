"""Experimental RQ-style stochastic teacher over LASER atom/coefficient pairs.

This is a causal residual coder, NOT full-refit OMP. It is deliberately not
wired into the production trainer. Validate reconstruction and throughput before
launching a run: summing over the full pair vocabulary is expensive.
"""
from dataclasses import dataclass
import math

import torch


@dataclass
class ResidualPairTargets:
    atoms: torch.Tensor
    coefficient_ids: torch.Tensor
    atom_probabilities: torch.Tensor
    coefficient_probabilities: torch.Tensor
    reconstruction: torch.Tensor


@torch.no_grad()
def residual_pair_targets(signals, dictionary, coefficient_values, *, temperature=.5,
                          site_chunk_size=16, atom_chunk_size=256, generator=None):
    """Sample q(a,c|r) proportional to exp(-||r-c*d_a||² / temperature).

    ``signals``: [..., channels]; ``dictionary``: [channels, atoms];
    ``coefficient_values``: [depth, bins], in physical latent units. Coefficients
    may use different grids at each depth. Repeated atoms are allowed, as in RQ.
    Returns full atom soft targets and coefficient soft targets conditioned on
    the sampled current atom. Both draws drive the next residual; earlier pairs
    are never refit. There is no support bank or training top-k/top-p filter.

    Site and atom chunks bound temporary pair-score memory. The full atom and
    coefficient target tensors are still retained. Chunking may change RNG draw
    order, so record chunk sizes along with the generator state for replay.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
           for v in (site_chunk_size, atom_chunk_size)):
        raise ValueError('chunk sizes must be positive integers')
    if (signals.ndim < 2 or dictionary.ndim != 2 or coefficient_values.ndim != 2
            or signals.shape[-1] != dictionary.shape[0]
            or any(v.numel() == 0 for v in (signals, dictionary, coefficient_values))):
        raise ValueError('invalid signal, dictionary, or coefficient grid shape')
    if (any(v.device != signals.device or v.dtype != signals.dtype
            for v in (dictionary, coefficient_values))
            or signals.dtype not in (torch.float32, torch.float64)):
        raise ValueError('all inputs must share an FP32/FP64 dtype and device')
    if not all(torch.isfinite(v).all() for v in (signals, dictionary, coefficient_values)):
        raise ValueError('teacher inputs must be finite')
    with torch.autocast(device_type=signals.device.type, enabled=False):
        shape = signals.shape[:-1]
        residual = signals.reshape(-1, signals.shape[-1]).clone()
        n, vocab = len(residual), dictionary.shape[1]
        depth, bins = coefficient_values.shape
        norms = dictionary.square().sum(0)
        atoms = torch.empty((n, depth), dtype=torch.long, device=signals.device)
        coefficients = torch.empty_like(atoms)
        atom_targets = signals.new_empty(n, depth, vocab)
        coefficient_targets = signals.new_empty(n, depth, bins)
        for d, values in enumerate(coefficient_values):
            for start in range(0, n, site_chunk_size):
                r = residual[start:start + site_chunk_size]
                correlations = r @ dictionary
                log_marginal = signals.new_empty(len(r), vocab)
                for a in range(0, vocab, atom_chunk_size):
                    # The common -||r||² term cancels from both softmaxes.
                    scores = (2 * correlations[:, a:a + atom_chunk_size, None]
                              * values - norms[None, a:a + atom_chunk_size, None]
                              * values.square()) / temperature
                    log_marginal[:, a:a + atom_chunk_size] = scores.logsumexp(-1)
                qa = log_marginal.softmax(-1)
                atom = torch.multinomial(qa, 1, generator=generator).squeeze(-1)
                selected = correlations.gather(1, atom[:, None])
                qc = ((2 * selected * values - norms[atom, None] * values.square())
                      / temperature).softmax(-1)
                coefficient = torch.multinomial(qc, 1, generator=generator).squeeze(-1)
                stop = start + len(r)
                atoms[start:stop, d] = atom
                coefficients[start:stop, d] = coefficient
                atom_targets[start:stop, d] = qa
                coefficient_targets[start:stop, d] = qc
                r.sub_(dictionary.T[atom] * values[coefficient, None])
        return ResidualPairTargets(
            atoms.reshape(*shape, depth), coefficients.reshape(*shape, depth),
            atom_targets.reshape(*shape, depth, vocab),
            coefficient_targets.reshape(*shape, depth, bins),
            (signals.reshape_as(residual) - residual).reshape_as(signals),
        )


def residual_pair_cross_entropy(atom_logits, coefficient_logits, targets):
    """Unweighted joint pair CE, averaged over sites/depth (not over heads).

    The coefficient logits must condition on ``targets.atoms`` at this depth.
    Averaging visits samples the atom expectation in the joint CE. The atom
    term itself uses its full marginal soft target on every visit.
    """
    if (atom_logits.shape != targets.atom_probabilities.shape
            or coefficient_logits.shape != targets.coefficient_probabilities.shape):
        raise ValueError('logit and teacher target shapes differ')
    dtype = torch.float64 if atom_logits.dtype == torch.float64 else torch.float32
    atom_ce = -(targets.atom_probabilities.to(dtype) *
                atom_logits.to(dtype).log_softmax(-1)).sum(-1)
    coefficient_ce = -(targets.coefficient_probabilities.to(dtype) *
                       coefficient_logits.to(dtype).log_softmax(-1)).sum(-1)
    return (atom_ce + coefficient_ce).mean()
