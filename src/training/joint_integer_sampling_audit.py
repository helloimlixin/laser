"""Probability mass retained by native top-k followed by top-p sampling."""
import torch


@torch.no_grad()
def retained_probability(logits, *, top_k, top_p=.95):
    """Return retained original probability mass and support size per row.

    Temperature is one: this audit measures the current model distribution.
    Ties and the nucleus boundary follow the upstream RQ implementation.
    """
    logits = logits.float()
    if logits.ndim != 2 or not 0 < top_p <= 1:
        raise ValueError('Expected row logits and top-p in (0,1]')
    probabilities = logits.softmax(-1)
    if top_k is None:
        support = torch.ones_like(logits, dtype=torch.bool)
    else:
        if not isinstance(top_k, int) or not 1 <= top_k <= logits.shape[-1]:
            raise ValueError('Invalid top-k')
        cutoff = logits.topk(top_k, dim=-1).values[:, -1:]
        support = logits >= cutoff
    kept = probabilities * support
    conditioned = kept / kept.sum(-1, keepdim=True)
    ordered, indices = conditioned.sort(dim=-1, descending=True)
    remove = ordered.cumsum(-1) >= top_p
    remove[:, 1:] = remove[:, :-1].clone()
    remove[:, 0] = False
    nucleus = (~remove).scatter(-1, indices, ~remove)
    support &= nucleus
    return (probabilities * support).sum(-1), support.sum(-1)
