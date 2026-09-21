"""Stochastic OMP trajectories and paired selection from a prebuilt cache.

The encoder samples atom supports proportional to exp(correlation**2 / tau)
at each OMP iteration and refits all coefficients on the sampled support.
Training draws one whole trajectory per spatial site. Coefficients must never
be selected independently from a different trajectory's atoms.
"""
from __future__ import annotations

import math
import torch


BANK_FORMAT = "laser_stochastic_compound_bank_v1"


@torch.no_grad()
def stochastic_omp(signals, dictionary, *, depth=4, temperature=0.0, gram=None, generator=None):
    """Return sampled atoms and physical least-squares coefficients.

    ``signals`` is [..., channels], dictionary is [channels, atoms] with unit
    columns. Temperature is in squared physical latent-distance units.
    Zero temperature is ordinary greedy OMP. Inputs must be FP32 or FP64.
    """
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    if not 0 < depth <= dictionary.shape[1]:
        raise ValueError("depth must be between one and the vocabulary size")
    if signals.dtype not in (torch.float32, torch.float64) or dictionary.dtype != signals.dtype:
        raise ValueError("OMP requires matching FP32 or FP64 inputs")
    if signals.shape[-1] != dictionary.shape[0]:
        raise ValueError("signal and dictionary channels differ")
    shape = signals.shape[:-1]
    x = signals.reshape(-1, signals.shape[-1])
    gram = dictionary.T @ dictionary if gram is None else gram
    correlation = x @ dictionary
    residual_correlation = correlation
    available = torch.ones_like(correlation, dtype=torch.bool)
    rows = torch.arange(len(x), device=x.device)
    support = torch.empty(len(x), 0, dtype=torch.long, device=x.device)
    chol = None
    entropy = []
    for step in range(depth):
        scores = residual_correlation.square().masked_fill(~available, -torch.inf)
        if temperature:
            probabilities = ((scores - scores.amax(-1, keepdim=True)) / temperature).softmax(-1)
            entropy.append(-(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1))
            atom = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
        else:
            entropy.append(torch.zeros(len(x), device=x.device, dtype=x.dtype))
            atom = scores.argmax(-1)
        available[rows, atom] = False
        if step == 0:
            chol = gram[atom, atom].sqrt()[:, None, None]
        else:
            cross = gram[support, atom[:, None]].unsqueeze(-1)
            solved = torch.linalg.solve_triangular(chol, cross, upper=False).transpose(1, 2)
            diagonal = gram[atom, atom][:, None, None] - solved.square().sum(-1, keepdim=True)
            if (diagonal <= 1e-7).any():
                raise ValueError("sampled support is numerically dependent; reject this trajectory")
            chol = torch.cat((torch.cat((chol, x.new_zeros(len(x), step, 1)), dim=-1),
                              torch.cat((solved, diagonal.sqrt()), dim=-1)), dim=-2)
        support = torch.cat((support, atom[:, None]), dim=-1)
        rhs = correlation.gather(1, support)
        coefficients = torch.cholesky_solve(rhs.unsqueeze(-1), chol).squeeze(-1)
        residual_correlation = correlation - coefficients.unsqueeze(1).bmm(gram[support]).squeeze(1)
    quantized = (dictionary.T[support] * coefficients[..., None]).sum(-2)
    return dict(atoms=support.reshape(*shape, depth),
                coefficients=coefficients.reshape(*shape, depth),
                quantized=quantized.reshape_as(signals),
                selection_entropy=torch.stack(entropy, -1).reshape(*shape, depth))


def sample_compound_bank(atoms, coefficients, *, generator=None, choices=None):
    """Choose a complete K-pair trajectory independently at each spatial site.

    Both banks have shape [B,H,W,variants,K]. Draws happen on their device,
    using the same torch RNG checkpointed by the training loop. No DataLoader
    worker RNG is involved. Returning choices makes provenance checks cheap.
    """
    if atoms.shape != coefficients.shape or atoms.ndim != 5:
        raise ValueError("paired banks must have equal [B,H,W,variants,K] shapes")
    if atoms.device != coefficients.device:
        raise ValueError("paired banks must share a device")
    count = atoms.shape[-2]
    if count < 2:
        raise ValueError("a stochastic compound bank needs at least two variants")
    if choices is None:
        choices = torch.randint(count, atoms.shape[:-2], device=atoms.device, generator=generator)
    if choices.shape != atoms.shape[:-2]:
        raise ValueError("variant choices must match [B,H,W]")
    index = choices[..., None, None].expand(*choices.shape, 1, atoms.shape[-1])
    return atoms.gather(-2, index).squeeze(-2), coefficients.gather(-2, index).squeeze(-2), choices
