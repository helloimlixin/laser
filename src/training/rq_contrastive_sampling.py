"""Experimental cached contrastive decoding for two compatible RQ priors."""
import math

import torch


def contrast_logits(expert, amateur, strength, top_k):
    if (not math.isfinite(strength) or strength < 0 or expert.shape != amateur.shape
            or expert.ndim != 2 or not 1 <= top_k <= expert.shape[-1]):
        raise ValueError('Compatible logits, nonnegative strength, and a valid expert top-k are required')
    if strength == 0:
        return expert
    good, bad = expert.float(), amateur.float()
    cutoff = good.topk(top_k,dim=-1).values[:,-1:]
    guided = (1+strength)*good-strength*bad
    return guided.masked_fill(good < cutoff, -torch.inf)


class ContrastiveCachedPrior:
    """Keep both priors on the exact same sampled causal prefix.

    No parameter is replaced. The expert's existing native sampler applies
    top-p and performs its ordinary single categorical draw at each event.
    """
    def __init__(self, expert, amateur, *, top_k=2048):
        if tuple(expert.block_size) != tuple(amateur.block_size) or expert.vocab_size != amateur.vocab_size:
            raise ValueError('Expert and amateur must use the same code shape and vocabulary')
        self.expert, self.amateur = expert, amateur
        self.top_k, self.strength = int(top_k), 0.
        self.expert_cached = expert.cached_forward
        self.expert_init = expert.init_cache
        expert.cached_forward = self.cached_forward
        expert.init_cache = self.init_cache

    def init_cache(self):
        self.expert_init()
        self.amateur.init_cache()

    @torch.no_grad()
    def cached_forward(self,*args,**kwargs):
        good = self.expert_cached(*args,**kwargs)
        if self.strength == 0:
            return good
        bad = self.amateur.cached_forward(*args,**kwargs)
        return contrast_logits(good,bad,self.strength,self.top_k)

    def restore(self):
        self.expert.cached_forward = self.expert_cached
        self.expert.init_cache = self.expert_init
