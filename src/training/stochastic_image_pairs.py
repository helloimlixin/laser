"""Fresh stochastic OMP supports with their matching coefficient targets.

The teacher draws a complete support, refits its physical coefficients, and
then uses LaserAux.sparse_targets to draw coefficient bins. Hard sampled atom
labels are Monte Carlo supervision for this joint trajectory law. OMP's
support-selection probabilities are deliberately not used as interleaved soft
labels: they do not condition on previously observed noisy coefficients.
"""
from contextlib import contextmanager
import math

import torch

from src.stochastic_compound import stochastic_omp


POLICY_VERSION = 'online-stochastic-omp-pairs-v1'
SOFT_POLICY_VERSION = 'online-stochastic-omp-conditional-soft-pairs-v2'


def validate_policy(temperature, site_chunk_size):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('stochastic atom temperature must be finite and positive')
    if (isinstance(site_chunk_size, bool) or not isinstance(site_chunk_size, int)
            or site_chunk_size < 1):
        raise ValueError('stochastic atom site chunk must be a positive integer')


@contextmanager
def omp_precision(device):
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.autocast(device_type=device.type, enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@torch.no_grad()
def stochastic_components(signals, dictionary, scales, *, temperature,
                          site_chunk_size=128, gram=None, generator=None):
    """Sample full-vocabulary supports online; never mix trajectory components.

    Chunk size is part of the policy because it changes RNG draw order. The
    default generator is the training CUDA/CPU stream saved by the trainer.
    No cache of alternative codes and no top-k or nucleus filter is used.
    """
    validate_policy(temperature, site_chunk_size)
    if (signals.ndim < 2 or dictionary.ndim != 2 or scales.ndim != 1
            or signals.shape[-1] != dictionary.shape[0]
            or signals.numel() == 0 or not 0 < scales.numel() <= dictionary.shape[1]
            or signals.device != dictionary.device or signals.device != scales.device
            or not torch.isfinite(scales).all() or not (scales > 0).all()):
        raise ValueError('invalid stochastic signal, dictionary, or coefficient scales')
    with omp_precision(signals.device):
        x = signals.reshape(-1, signals.shape[-1]).float()
        book = dictionary.float()
        gram = book.T @ book if gram is None else gram.float()
        depth = scales.numel()
        atoms = torch.empty(len(x), depth, dtype=torch.long, device=x.device)
        physical = x.new_empty(len(x), depth)
        entropy = torch.empty_like(physical)
        for start in range(0, len(x), site_chunk_size):
            stop = min(start + site_chunk_size, len(x))
            sampled = stochastic_omp(x[start:stop], book, depth=depth,
                temperature=temperature, gram=gram, generator=generator)
            atoms[start:stop] = sampled['atoms']
            physical[start:stop] = sampled['coefficients']
            entropy[start:stop] = sampled['selection_entropy']
        shape = (*signals.shape[:-1], depth)
        return dict(atoms=atoms.reshape(shape),
            coefficients=(physical / scales.float()).reshape(shape),
            physical_coefficients=physical.reshape(shape),
            support_selection_entropy=entropy.reshape(shape))


@torch.no_grad()
def stochastic_image_components(aux, images, *, temperature, site_chunk_size=128):
    """Use fresh image latents and a derived Gram matrix of the frozen dictionary."""
    validate_policy(temperature, site_chunk_size)
    if aux.clamp_coeffs:
        raise ValueError('stochastic pair targets require unclipped OMP coefficients')
    signals = aux.quant_conv(aux.encoder(images)).permute(0, 2, 3, 1).float()
    with omp_precision(signals.device):
        if not hasattr(aux, '_stochastic_pair_gram'):
            aux._stochastic_pair_gram = aux.dictionary.float().T @ aux.dictionary.float()
        result = stochastic_components(signals, aux.dictionary, aux.coeff_scales,
            temperature=temperature, site_chunk_size=site_chunk_size,
            gram=aux._stochastic_pair_gram)
    return result['atoms'], result['coefficients']


@torch.no_grad()
def stochastic_soft_image_targets(aux, images, *, atom_temperature,
                                  coefficient_temperature, variants=16, site_chunk_size=128,
                                  backend='eager'):
    """Draw both components and Rao–Blackwellize their interleaved soft labels.

    Every visit draws fresh full-vocabulary OMP alternatives. A whole matching
    trajectory is selected at each site and its coefficient bins are sampled.
    Exact conditional labels within this fresh mixture use previous observed
    atom/coefficient pairs, plus the current atom for the coefficient head.
    Averaging these conditional labels preserves the expected joint teacher
    objective; this is not a persistent finite-support training cache.
    """
    from src.stochastic_compound import sample_compound_bank
    from src.training.omp_joint_targets import omp_bank_joint_targets
    validate_policy(atom_temperature, site_chunk_size)
    if backend not in ('eager','cudagraph'):
        raise ValueError('unknown stochastic teacher backend')
    if isinstance(variants, bool) or not isinstance(variants, int) or variants < 2:
        raise ValueError('soft stochastic pairs require at least two fresh variants')
    if not math.isfinite(coefficient_temperature) or coefficient_temperature <= 0:
        raise ValueError('coefficient temperature must be finite and positive')
    if aux.clamp_coeffs or getattr(aux, 'soft_target_physical', False):
        raise ValueError('soft image pair policy requires unclipped, normalized coefficient targets')
    signals = aux.quant_conv(aux.encoder(images)).permute(0, 2, 3, 1).float()
    with omp_precision(signals.device):
        if not hasattr(aux, '_stochastic_pair_gram'):
            aux._stochastic_pair_gram = aux.dictionary.float().T @ aux.dictionary.float()
        if backend == 'cudagraph':
            from src.training.stochastic_omp_graph import stochastic_bank_graph
            bank_atoms,bank_coefficients = stochastic_bank_graph(aux,signals,
                temperature=atom_temperature,variants=variants,site_chunk_size=site_chunk_size)
        else:
            alternatives = [stochastic_components(signals, aux.dictionary, aux.coeff_scales,
                temperature=atom_temperature, site_chunk_size=site_chunk_size,
                gram=aux._stochastic_pair_gram) for _ in range(variants)]
            bank_atoms = torch.stack([r['atoms'] for r in alternatives], -2)
            bank_coefficients = torch.stack([r['coefficients'] for r in alternatives], -2)
        atoms, coefficients, choices = sample_compound_bank(bank_atoms, bank_coefficients)
        tokens, _ = aux.sparse_targets(atoms, coefficients, temp=coefficient_temperature,
            stochastic=True, compact=True, hard=False)
        coefficient_ids = tokens[..., 1::2] - aux.num_atoms
        labels = omp_bank_joint_targets(bank_atoms, bank_coefficients, atoms, coefficient_ids,
            aux.coeff_bins.float(), temperature=coefficient_temperature, site_chunk_size=site_chunk_size)
    return dict(tokens=tokens, atoms=atoms, coefficients=coefficients,
        atom_ids=labels.atom_ids, atom_weights=labels.atom_weights,
        coefficient_probabilities=labels.coefficient_probabilities, trajectory_choices=choices)
