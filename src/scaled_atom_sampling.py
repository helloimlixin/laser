"""Inference samplers for packed atom/coefficient IDs; no tokenizer updates."""
from itertools import product
import math

import torch


SAMPLER_SETTINGS = {
    'original': dict(mode='joint',temperature=1.,top_k=16384,top_p=.92),
    'full_p92': dict(mode='joint',temperature=1.,top_k=None,top_p=.92),
    'full_t09_p92': dict(mode='joint',temperature=.9,top_k=None,top_p=.92),
    'full_t09_p95': dict(mode='joint',temperature=.9,top_k=None,top_p=.95),
    'original_t09': dict(mode='joint',temperature=.9,top_k=16384,top_p=.92),
    'atom_p92_coeff1': dict(mode='factorized',atom_temperature=1.,atom_top_p=.92,coefficient_temperature=1.),
    'atom_p92_coeff07': dict(mode='factorized',atom_temperature=1.,atom_top_p=.92,coefficient_temperature=.7),
    'atom_t09_p95_coeff07': dict(mode='factorized',atom_temperature=.9,atom_top_p=.95,coefficient_temperature=.7),
}


def validate_sampler(settings):
    if settings['mode']=='joint':
        temperatures=[settings['temperature']];probabilities=[settings['top_p']]
        if settings['top_k'] is not None and settings['top_k']<1:raise ValueError('top_k must be positive or null')
    elif settings['mode']=='factorized':
        temperatures=[settings['atom_temperature'],settings['coefficient_temperature']]
        probabilities=[settings['atom_top_p']]
    else:raise ValueError('Unknown sampler mode')
    if not all(math.isfinite(t) and t>0 for t in temperatures):raise ValueError('Positive finite temperatures required')
    if not all(math.isfinite(p) and 0<p<=1 for p in probabilities):raise ValueError('Nucleus probability must be in (0,1]')
    return dict(settings)


def factorized_logits(logits,levels):
    """Exact atom marginal logits, including one zero candidate, plus bin logits."""
    if logits.ndim!=2 or levels<1 or (logits.shape[-1]-1)%levels:
        raise ValueError('Expected one zero ID followed by atom-major coefficient IDs')
    bins=logits[:,1:].float().reshape(len(logits),-1,levels)
    atoms=torch.cat([logits[:,:1].float(),bins.logsumexp(-1)],-1)
    return atoms,bins


@torch.no_grad()
def draw_token(logits,settings,levels):
    from rqvae.utils.utils import sample_from_logits
    if settings['mode']=='joint':
        k=settings['top_k']
        if k is not None and k>=logits.shape[-1]:k=None
        return sample_from_logits(logits,temperature=settings['temperature'],top_k=k,top_p=settings['top_p'])
    atoms,bins=factorized_logits(logits,levels)
    selected=sample_from_logits(atoms,temperature=settings['atom_temperature'],top_k=None,top_p=settings['atom_top_p'])
    row=torch.arange(len(logits),device=logits.device)
    coefficients=sample_from_logits(bins[row,(selected-1).clamp_min(0)],
        temperature=settings['coefficient_temperature'],top_k=None,top_p=None)
    return torch.where(selected==0,0,1+(selected-1)*levels+coefficients)


@torch.no_grad()
def sample_codes(model,tokenizer,labels,settings,*,amp=True):
    """Use the released cached-forward order with an explicitly selected filter."""
    settings=validate_sampler(settings)
    shape=tuple(model.block_size)
    codes=torch.zeros(len(labels),*shape,device=labels.device,dtype=torch.long)
    model.init_cache()
    try:
        for h,w,d in product(*(range(n) for n in shape)):
            logits=model.cached_forward(codes[:,:h+1],tokenizer,cond=labels,amp=amp,sample_loc=(h,w,d))
            codes[:,h,w,d]=draw_token(logits,settings,len(tokenizer.quantizer.levels))
    finally:
        model.init_cache()
    return codes
