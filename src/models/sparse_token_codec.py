"""Shared hard atom/coefficient codec for tokenizer learning and prior targets."""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F
from src.stochastic_compound import stochastic_omp


TOKEN_POLICY_VERSION = 'compound-hard-token-v1'


def token_temperatures(q):
    """Physical squared-distance temperatures tied to the learned coefficient grid."""
    policy = q.tokenized_sparse_policy
    ranges = q.coefficient_max.detach().float().cpu().square().tolist()
    return ([r * policy['atom_temperature_ratio'] for r in ranges],
            [r * policy['coefficient_temperature_ratio'] for r in ranges])


def validate_tokenized_checkpoint(q, checkpoint):
    expected = getattr(q, 'tokenized_sparse_policy', None)
    recorded = checkpoint.get('initialization', {}).get('tokenized_sparse_policy')
    if recorded != expected:
        raise ValueError('Tokenizer checkpoint and configuration use different sparse-token training policies')


def decompose_sparse_tokens(q, latent, *, atom_temperatures=None, coefficient_temperatures=None,
                       stochastic=False, generator=None, update_ranges=False, calibrate=False,
                       include_probabilities=True):
    """Sample an internally consistent multiscale OMP trajectory.

Every later scale is re-encoded against the *actually sampled* earlier scales.
Mixing independently cached per-site variants across scales is invalid here.
Soft targets use squared distances in physical coefficient units, not bin IDs.
"""
    stages = len(q.v_patch_nums)
    policy = getattr(q, 'tokenized_sparse_policy', None)
    relative = policy is not None and atom_temperatures is None and coefficient_temperatures is None
    if relative:
        atom_temperatures, coefficient_temperatures = token_temperatures(q)
    atom_temperatures = atom_temperatures or [0.] * stages
    coefficient_temperatures = coefficient_temperatures or [0.] * stages
    if len(atom_temperatures) != stages or len(coefficient_temperatures) != stages:
        raise ValueError('One temperature per spatial scale is required')
    if any(not math.isfinite(t) or t < 0 for t in (*atom_temperatures, *coefficient_temperatures)):
        raise ValueError('Temperatures must be nonnegative')
    with torch.autocast(device_type=latent.device.type, enabled=False):
        z = latent.float()
        if z.shape[1:] != (q.Cvae, q.v_patch_nums[-1], q.v_patch_nums[-1]):
            raise ValueError(f'Unexpected latent shape {tuple(z.shape)}')
        if not torch.isfinite(z).all():
            raise FloatingPointError('Nonfinite encoder output')
        accumulated = torch.zeros_like(z)
        dictionary = q.normalized_dictionary().detach()
        gram = dictionary.T @ dictionary if stochastic and any(atom_temperatures) else None
        atoms_all, ids_all, values_all, probs_all, inputs, clipping, losses = [], [], [], [], [], [], []
        for scale, pn in enumerate(q.v_patch_nums):
            if scale:
                inputs.append(q.next_input(accumulated.detach(), scale))
            signals = F.interpolate(z.detach() - accumulated.detach(), (pn, pn), mode='area')
            signals = signals.permute(0, 2, 3, 1).reshape(-1, q.Cvae)
            with torch.no_grad():
                atoms, values = [], []
                for chunk in signals.split(q.omp_chunk_size):
                    if stochastic and atom_temperatures[scale] > 0:
                        result = stochastic_omp(chunk, dictionary, depth=q.sparsity,
                                                temperature=atom_temperatures[scale], gram=gram,
                                                generator=generator)
                        a, c = result['atoms'], result['coefficients']
                    else:
                        a, c = q.dictionary.batch_omp_with_support(chunk.T, dictionary)
                    atoms.append(a)
                    values.append(c)
                atoms = torch.cat(atoms).reshape(len(z), pn * pn, q.sparsity)
                values = torch.cat(values).reshape_as(atoms)
                if update_ranges or calibrate:
                    q.update_coefficient_range(values, scale, calibrate=calibrate)
                    if relative:
                        coefficient_temperatures[scale] = float(q.coefficient_max[scale].square()) * policy['coefficient_temperature_ratio']
                clipping.append((values.abs() > q.coefficient_max[scale]).float().mean())
                grid = q.coefficient_values(torch.arange(q.coefficient_bins, device=z.device), scale)
                temperature = coefficient_temperatures[scale]
                if stochastic and temperature > 0:
                    probabilities = (-(values[..., None] - grid).square() / temperature).softmax(-1)
                    ids = torch.multinomial(probabilities.reshape(-1, q.coefficient_bins), 1,
                                            generator=generator).reshape_as(atoms)
                else:
                    ids = q.coefficient_ids(values, scale)
                    probabilities = F.one_hot(ids, q.coefficient_bins).float() if include_probabilities else None
            accumulated = accumulated + q.contribution(q.embed(atoms, ids, scale), scale)
            losses.append(F.mse_loss(accumulated, z.detach()) + q.beta * F.mse_loss(accumulated.detach(), z))
            atoms_all.append(atoms)
            ids_all.append(ids)
            values_all.append(values)
            if include_probabilities:
                probs_all.append(probabilities)
        return dict(atoms=torch.cat(atoms_all, 1), coefficients=torch.cat(ids_all, 1),
                    physical_coefficients=torch.cat(values_all, 1),
                    coefficient_probabilities=torch.cat(probs_all, 1) if include_probabilities else None,
                    loss=torch.stack(losses).mean(),
                    inputs=torch.cat(inputs, 1), latent=accumulated,
                    clip_fraction=torch.stack(clipping).mean())

