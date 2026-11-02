"""Fit one physical sparse codeword table shared by every residual depth."""
import torch
import torch.distributed as dist

from src.training.laser_compact_joint_vocabulary import reconstruct_fixed_codes


def assert_shared_codewords(quantizer):
    if not (torch.equal(quantizer.atom_ids, quantizer.atom_ids[:1].expand_as(quantizer.atom_ids))
            and torch.equal(quantizer.coefficients, quantizer.coefficients[:1].expand_as(quantizer.coefficients))):
        raise ValueError('Shared integer IDs must identify exactly the same physical vector at every depth')


def _sum(value):
    if dist.is_initialized():
        dist.all_reduce(value)
    return value


@torch.no_grad()
def refit_shared_book(quantizer, signals, codes, *, ridge_fraction=.01, prior_weight=4.):
    """Global dictionary LS and a monotone pooled scalar update at fixed codes.

    Repeated IDs at several depths contribute multiplicity squared to the
    scalar diagonal. A line search includes all cross terms in the actual
    final residual, while retaining one identical table at every depth.
    Optional distributed fitting partitions signals, never the codebook.
    """
    assert_shared_codewords(quantizer)
    if ridge_fraction <= 0 or prior_weight <= 0:
        raise ValueError('Positive regularization required')
    x = signals.reshape(-1, signals.shape[-1]).float()
    codes = codes.reshape(-1, quantizer.depth)
    atoms = quantizer.atom_ids[0, codes]
    coefficients = quantizer.coefficients[0, codes]
    n_atoms, channels = quantizer.dictionary.shape[1], x.shape[1]
    size = _sum(x.new_tensor(float(x.numel()), dtype=torch.float64))

    def mse(error):
        return float(_sum(error.double().square().sum()) / size)

    before = mse(x - reconstruct_fixed_codes(quantizer, codes))
    gram = x.new_zeros(n_atoms, n_atoms)
    rhs = x.new_zeros(n_atoms, channels)
    for depth in range(quantizer.depth):
        rhs.index_add_(0, atoms[:, depth], x * coefficients[:, depth, None])
        for other in range(quantizer.depth):
            index = atoms[:, depth] * n_atoms + atoms[:, other]
            gram.view(-1).scatter_add_(0, index, coefficients[:, depth] * coefficients[:, other])
    _sum(gram); _sum(rhs)
    gram = .5 * (gram + gram.T)
    ridge = ridge_fraction * gram.diagonal().clamp_min(1.)
    gram.diagonal().add_(ridge)
    rhs.add_(quantizer.dictionary.T * ridge[:, None])
    updated = torch.cholesky_solve(rhs, torch.linalg.cholesky(gram)).T
    if dist.is_initialized():
        dist.broadcast(updated, 0)
    norm = updated.norm(dim=0)
    assert torch.isfinite(updated).all() and (norm > 1.e-8).all()
    quantizer.dictionary.copy_(updated / norm)
    shared = quantizer.coefficients[0].clone()
    shared[1:].mul_(norm[quantizer.atom_ids[0, 1:]])
    quantizer.coefficients.copy_(shared.expand_as(quantizer.coefficients))
    quantizer.norms.copy_(quantizer.dictionary.square().sum(0))
    del gram, rhs, updated
    error = x - reconstruct_fixed_codes(quantizer, codes)
    after_dictionary = mse(error)
    numerator = x.new_zeros(quantizer.vocab_size)
    diagonal = torch.zeros_like(numerator)
    for depth in range(quantizer.depth):
        chosen = codes[:, depth]
        vectors = quantizer.dictionary.T[atoms[:, depth]]
        numerator.scatter_add_(0, chosen, (error * vectors).sum(-1))
        multiplicity = (codes == chosen[:, None]).sum(-1).float()
        diagonal.scatter_add_(0, chosen, multiplicity * quantizer.norms[atoms[:, depth]])
    _sum(numerator); _sum(diagonal)
    proposal = shared + numerator / (diagonal + prior_weight)
    old, values = shared[1:], proposal[1:]
    layout = quantizer.atom_ids[0, 1:]
    midpoint = (old[:-1] + old[1:]) * .5
    same_atom = layout[:-1] == layout[1:]
    values[:-1] = torch.minimum(values[:-1], torch.where(same_atom, midpoint, torch.inf))
    values[1:] = torch.maximum(values[1:], torch.where(same_atom, midpoint, -torch.inf))
    values.copy_(torch.where(old < 0, values.clamp(max=-1.e-6), values.clamp(min=1.e-6)))
    proposal[0] = 0
    direction = proposal - shared
    change = sum(quantizer.dictionary.T[atoms[:, depth]] * direction[codes[:, depth], None]
                 for depth in range(quantizer.depth))
    dot = _sum(torch.stack([(error.double() * change).sum(), change.double().square().sum()]))
    step = float((dot[0] / dot[1].clamp_min(1.e-30)).clamp(0, 1))
    shared.add_(direction, alpha=step)
    quantizer.coefficients.copy_(shared.expand_as(quantizer.coefficients))
    after = mse(x - reconstruct_fixed_codes(quantizer, codes))
    assert after_dictionary <= before + 2.e-6 * max(1., before)
    assert after <= after_dictionary + 2.e-6 * max(1., after_dictionary)
    assert_shared_codewords(quantizer)
    return dict(fixed_assignment_mse_before=before,
        fixed_assignment_mse_after_dictionary=after_dictionary,
        fixed_assignment_mse_after=after, shared_scalar_step=step)
