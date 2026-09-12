"""Church dimensions/settings around the unmodified successful FFHQ-v4 model."""
import math
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src import ffhq_v4_archived as archived
from src.church_epoch50_loop import GatedDepthLoop
from src.church_original_scratch import FullBatchEpochStream, initialization_audit
from src.training.rqtransformer import LaserAux as ChurchTokenizer

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SHA256 = '9ba1b49b4e5e339f0076bebee6fbac5629f6c391601de467019a3723c9d3e33f'
UPSTREAM_CONFIG = ROOT / 'archive/configs/church_ffhq_archived_upstream.yaml'


class ChurchAux(ChurchTokenizer):
    # Use the actual archived normalized soft-target sampler. The tokenizer
    # loader/decoder adapter supports Church's four depths and frozen weights.
    compound_coeff_ids = archived.LaserAux.compound_coeff_ids


def church_config():
    upstream = OmegaConf.load(UPSTREAM_CONFIG)
    config = OmegaConf.to_container(upstream.arch, resolve=True)
    config['vocab_size'] = upstream.dataset.vocab_size
    return archived.RQTransformerConfig.create(OmegaConf.create(config))


def make_prior(variant='control', config=None, num_atoms=16384, coeff_vocab_size=2048):
    if variant not in ('control', 'looped'):
        raise ValueError(variant)
    model = archived.CompoundLaserRQTransformer(
        church_config() if config is None else config,
        num_atoms, coeff_vocab_size, micro_transformer_layers=2,
        depth_specific_coeff_heads=True,
    )
    if variant == 'looped':
        model.head_transformer = GatedDepthLoop(list(model.head_transformer.blocks))
    return model


def cosine_lr(completed_steps, total_steps, peak=5e-4):
    if total_steps < 1 or completed_steps < 0:
        raise ValueError('Invalid schedule progress')
    fraction = min(completed_steps / total_steps, 1.)
    return peak * .5 * (1 + math.cos(math.pi * fraction))


@torch.no_grad()
def targets(aux, atoms, physical, stochastic=True):
    normalized = physical.float() / aux.coeff_scales
    ids, probabilities = aux.compound_coeff_ids(normalized, stochastic=stochastic, temp=.5)
    return atoms.long() * aux.coeff_vocab_size + ids, probabilities


def objective(model, aux, atoms, physical, geometry_weight, stochastic=True):
    # Preserve the archived training call: outer BF16, model amp=False. The
    # model's nested scope therefore computes teacher forcing in FP32/TF32.
    with torch.autocast(atoms.device.type, dtype=torch.bfloat16, enabled=atoms.is_cuda):
        packed, probabilities = targets(aux, atoms, physical, stochastic)
        out = model(packed, model_aux=aux, amp=False)
        loss, values = archived.compound_objective(
            out['atom_logits'], out['coeff_logits'], None, atoms, probabilities,
            aux.dictionary.t()[atoms.long()] * physical[..., None],
            atom_weight=1.5, geometry_weight=geometry_weight, accumulation=1,
            distribution_geometry=True, geometry_dictionary=aux.dictionary,
            geometry_coeff_bins=aux.coeff_bins, geometry_coeff_scales=aux.coeff_scales,
            geometry_top_k=4,
        )
    with torch.no_grad():
        entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)
        predicted = out['coeff_logits'].float().softmax(-1)
        mean = (predicted * aux.coeff_bins).sum(-1) * aux.coeff_scales
        sign = predicted[..., aux.coeff_vocab_size // 2:].sum(-1) >= .5
        metrics = {'loss': loss.detach(), 'atom_nll': values['atom_nll'].mean(),
            'coefficient_cross_entropy': values['coeff_cross_entropy'].mean(),
            'coefficient_target_entropy': entropy.mean(),
            'coefficient_kl': (values['coeff_cross_entropy'] - entropy).mean(),
            'geometry': values['geometry'],
            'coefficient_mean_mae': (mean - physical).abs().mean(),
            'sign_accuracy': (sign == (physical >= 0)).float().mean()}
    return loss, {k: float(v) for k, v in metrics.items()}


def full_training_cache(raw):
    keys = [set(raw[s]['keys']) for s in ('train', 'holdout', 'validation')]
    if any(keys[i] & keys[j] for i in range(3) for j in range(i)):
        raise ValueError('Cache partitions overlap')
    data = {'train': {name: torch.cat([raw[s][name] for s in ('train', 'holdout')])
                      for name in ('atoms', 'coefficients')},
            'train_probe': {name: raw['holdout'][name] for name in ('atoms', 'coefficients')},
            'validation': {name: raw['validation'][name] for name in ('atoms', 'coefficients')}}
    # As in the FFHQ cache, normalize each depth's training maximum to 3.
    scales = data['train']['coefficients'].abs().flatten(0, -2).amax(0) / 3
    return data, scales.tolist()
