"""Ordinary hard-token autoregression over one deterministic sparse encoding.

Sparse coding belongs to the frozen stage-1 tokenizer. There is no trajectory
bank, teacher distribution, coefficient noise, or marginal target construction.
The prior predicts four ordered support/coefficient pairs with categorical CE.
"""
import torch
from torch.nn import functional as F

@torch.no_grad()
def sparse_encoding(signals, dictionary, depth, *, gram=None):
    """Extract the tokenizer's greedy support and final least-squares values."""
    if (signals.dtype != torch.float32 or dictionary.dtype != torch.float32
            or signals.shape[-1] != dictionary.shape[0]
            or not 0 < depth <= dictionary.shape[1]):
        raise ValueError('Sparse encoding requires matching FP32 signals/dictionary and valid depth')
    x = signals.reshape(-1, signals.shape[-1])
    gram = dictionary.T @ dictionary if gram is None else gram
    # The stage-1 dictionary has unit columns; retain its unit-diagonal
    # Cholesky convention and final refit, using ordinary PyTorch operations.
    correlation = (dictionary.T @ x.T).T
    residual = correlation
    support = torch.empty(len(x), 0, dtype=torch.long, device=x.device)
    lower = x.new_ones(len(x), 1, 1)
    for index in range(depth):
        scores = residual.abs()
        scores.scatter_(1, support, -torch.inf)
        atom = scores.argmax(-1)
        if index:
            cross = gram[support, atom[:, None]].unsqueeze(-1)
            solved = torch.linalg.solve_triangular(lower, cross, upper=False).transpose(1, 2)
            diagonal = (1. - solved.square().sum(-1, keepdim=True)).clamp_min(1.e-10).sqrt()
            lower = torch.cat((torch.cat((lower, x.new_zeros(len(x), index, 1)), -1),
                               torch.cat((solved, diagonal), -1)), -2)
        support = torch.cat((support, atom[:, None]), -1)
        values = torch.cholesky_solve(correlation.gather(1, support).unsqueeze(-1), lower).squeeze(-1)
        if index + 1 < depth:
            residual = correlation - values.unsqueeze(1).bmm(gram[support]).squeeze(1)
    if not torch.isfinite(values).all().item():
        raise ValueError('Invalid deterministic sparse encoding')
    shape = (*signals.shape[:-1], depth)
    return support.reshape(shape), values.reshape(shape)


def nearest_coefficient_ids(coefficients, bins):
    """Nearest bin with endpoint clipping and lower-bin tie breaking."""
    upper = torch.searchsorted(bins, coefficients.contiguous()).clamp_max(len(bins) - 1)
    lower = (upper - 1).clamp_min(0)
    return torch.where((coefficients - bins[lower]).abs() <= (coefficients - bins[upper]).abs(),
                       lower, upper)


@torch.no_grad()
def image_tokens(aux, images, options):
    """Encode each augmented image once and emit integer targets without RNG."""
    chunks = []
    with torch.autocast(device_type=images.device.type, enabled=False):
        if not hasattr(aux, '_hard_ar_gram'):
            aux._hard_ar_gram = (aux.dictionary.float().T @ aux.dictionary.float()).contiguous()
        for chunk in images.split(int(options.encoder_batch_size)):
            signals = aux.quant_conv(aux.encoder(chunk.float())).permute(0, 2, 3, 1).float()
            atoms, coefficients = sparse_encoding(signals, aux.dictionary.float(),
                aux.sparsity_level, gram=aux._hard_ar_gram)
            ids = nearest_coefficient_ids(coefficients / aux.coeff_scales, aux.coeff_bins)
            tokens = torch.empty(*atoms.shape[:-1], 2 * aux.sparsity_level,
                                 dtype=torch.long, device=images.device)
            tokens[..., 0::2] = atoms
            tokens[..., 1::2] = ids + aux.num_atoms
            chunks.append(tokens)
    return torch.cat(chunks)


def token_cross_entropy(logits, tokens):
    """Mean next-token negative log likelihood across the eight scalar events."""
    atoms, coefficients = logits['atom_logits'], logits['coeff_logits']
    atom_ce = F.cross_entropy(atoms.float().reshape(-1, atoms.shape[-1]),
                              tokens[..., 0::2].reshape(-1))
    coefficient_ce = F.cross_entropy(coefficients.float().reshape(-1, coefficients.shape[-1]),
                                     (tokens[..., 1::2] - atoms.shape[-1]).reshape(-1))
    return (atom_ce + coefficient_ce) / 2


@torch.no_grad()
def verify_tokens(aux, tokens):
    assert tokens.dtype == torch.long and not tokens.requires_grad
    assert tokens.shape[-1] == 2 * aux.sparsity_level
    atoms = tokens[..., 0::2]
    coefficients = tokens[..., 1::2] - aux.num_atoms
    assert ((atoms >= 0) & (atoms < aux.num_atoms)).all()
    assert ((coefficients >= 0) & (coefficients < aux.coeff_vocab_size)).all()
    assert (atoms.sort(-1).values.diff(dim=-1) > 0).all()
