"""Explicit FFHQ-v4 compound recipe and Church capacity comparison."""
import math

from omegaconf import OmegaConf
import torch

from scripts.train_official_rqtransformer_laser_stage2 import CompoundLaserRQTransformer
from src.models.rqtransformer.configs import RQTransformerConfig


ARCHITECTURES = {
    'reference': {'width': 1024, 'body_layers': 24, 'depth_layers': 4, 'heads': 16},
    'balanced': {'width': 768, 'body_layers': 20, 'depth_layers': 6, 'heads': 12},
}


def recipe_config(architecture):
    a = ARCHITECTURES[architecture]
    return RQTransformerConfig.create(OmegaConf.create({
        'type': 'rq-transformer', 'block_size': [8, 8, 4],
        'embed_dim': a['width'], 'input_embed_dim': 256,
        'shared_tok_emb': True, 'shared_cls_emb': True,
        'input_emb_vqvae': True, 'head_emb_vqvae': True, 'cumsum_depth_ctx': True,
        'vocab_size': 16384, 'vocab_size_cond': 1, 'block_size_cond': 1,
        'body': {'n_layer': a['body_layers'], 'block': {'n_head': a['heads'], 'resid_pdrop': .1}},
        'head': {'n_layer': a['depth_layers'], 'block': {'n_head': a['heads'], 'resid_pdrop': .1}},
    }))


def make_prior(architecture):
    # The archived FFHQ-v4 code used complete pairs in BOTH shifted streams.
    # In the maintained implementation this requires pair_autoregressive=True.
    return CompoundLaserRQTransformer(
        recipe_config(architecture), num_atoms=16384, coeff_vocab_size=2048,
        micro_transformer_layers=2, depth_specific_coeff_heads=True,
        pair_autoregressive=True,
    )


def early_decay_lr(progress, total_epochs=200, peak=2e-4, knee_lr=5e-5,
                   minimum=2e-6, warmup=1., knee_epoch=30.):
    """Short warmup, immediate exponential decay, then a low-rate cosine tail.

    Absolute image-epoch progress determines the rate, including after resume;
    the decay does not restart and cannot oscillate beyond the planned end.
    """
    if not (0 < warmup < knee_epoch < total_epochs):
        raise ValueError('Expected 0 < warmup < knee epoch < total epochs')
    if not (0 < minimum <= knee_lr <= peak):
        raise ValueError('Expected 0 < minimum <= knee LR <= peak LR')
    progress = max(0., min(float(progress), total_epochs))
    if progress <= warmup:
        return peak * (.005 + .995 * progress / warmup)
    if progress <= knee_epoch:
        return peak * (knee_lr / peak) ** ((progress - warmup) / (knee_epoch - warmup))
    fraction = (progress - knee_epoch) / (total_epochs - knee_epoch)
    return minimum + (knee_lr - minimum) * .5 * (1 + math.cos(math.pi * fraction))


@torch.no_grad()
def recipe_targets(aux, atoms, physical_coefficients, stochastic=True):
    if aux.soft_target_physical:
        raise ValueError('FFHQ v4 uses normalized-distance soft targets')
    normalized = physical_coefficients.float() / aux.coeff_scales
    ids, probabilities = aux.compound_coeff_ids(normalized, stochastic=stochastic, temp=.5)
    return atoms.long() * aux.coeff_vocab_size + ids, probabilities
