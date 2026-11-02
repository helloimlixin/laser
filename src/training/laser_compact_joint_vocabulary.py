"""Offline dictionary fitting for greedy, finite LASER compound vocabularies."""
import torch


@torch.no_grad()
def reconstruct_fixed_codes(quantizer, codes):
    result = quantizer.embed_depth(codes[..., 0], 0)
    for depth in range(1, quantizer.depth):
        result.add_(quantizer.embed_depth(codes[..., depth], depth))
    return result


@torch.no_grad()
def refit_compact_book(quantizer, signals, codes, *, ridge_fraction=.01, prior_weight=4.):
    """Minimize final latent error at fixed greedy assignments.

    The dictionary update solves regularized least squares including all
    cross-atom terms. Per-depth scalar tables then take coordinate LS updates.
    No image-specific coefficient is refitted during greedy encoding.
    """
    if ridge_fraction <= 0 or prior_weight <= 0:
        raise ValueError('Positive regularization required')
    x = signals.reshape(-1, signals.shape[-1]).float()
    codes = codes.reshape(-1, quantizer.depth)
    count, channels = x.shape
    atoms = torch.stack([quantizer.atom_ids[d, codes[:, d]] for d in range(quantizer.depth)], -1)
    coefficients = torch.stack([quantizer.coefficients[d, codes[:, d]] for d in range(quantizer.depth)], -1)
    num_atoms = quantizer.dictionary.shape[1]
    layout = quantizer.atom_ids[:, 1:]
    if not torch.equal(layout, layout[:1].expand_as(layout)) or (layout[:, 1:] < layout[:, :-1]).any():
        raise ValueError('Expected a shared, contiguous atom/level layout')
    frequency_by_atom = torch.bincount(layout[0], minlength=num_atoms)
    if (frequency_by_atom < 2).any() or (frequency_by_atom % 2).any():
        raise ValueError('Expected an even number of signed levels for every atom')
    before = float((x-reconstruct_fixed_codes(quantizer, codes)).square().mean())
    gram = x.new_zeros(num_atoms, num_atoms)
    rhs = x.new_zeros(num_atoms, channels)
    for d in range(quantizer.depth):
        rhs.index_add_(0, atoms[:, d], x*coefficients[:, d, None])
        for other in range(quantizer.depth):
            index = atoms[:, d]*num_atoms+atoms[:, other]
            gram.view(-1).scatter_add_(0, index, coefficients[:, d]*coefficients[:, other])
    gram = .5*(gram+gram.T)
    ridge = ridge_fraction*gram.diagonal().clamp_min(1.)
    gram.diagonal().add_(ridge)
    rhs.add_(quantizer.dictionary.T*ridge[:, None])
    factor = torch.linalg.cholesky(gram)
    updated = torch.cholesky_solve(rhs, factor).T
    norm = updated.norm(dim=0)
    assert torch.isfinite(updated).all() and (norm>1.e-8).all()
    quantizer.dictionary.copy_(updated/norm)
    quantizer.coefficients[:, 1:].mul_(norm[layout])
    quantizer.norms.copy_(quantizer.dictionary.square().sum(0))
    del gram, factor, rhs, updated
    error = x-reconstruct_fixed_codes(quantizer, codes)
    after_dictionary = float(error.square().mean())
    for depth in range(quantizer.depth):
        chosen = codes[:, depth]
        atom = quantizer.atom_ids[depth, chosen]
        vectors = quantizer.dictionary.T[atom]
        old = quantizer.coefficients[depth].clone()
        optimum = old[chosen]+(error*vectors).sum(-1)/quantizer.norms[atom]
        frequency = torch.bincount(chosen, minlength=quantizer.vocab_size)
        total = torch.bincount(chosen, weights=optimum, minlength=quantizer.vocab_size)
        proposal = (total+prior_weight*old)/(frequency+prior_weight)
        prior = old[1:]
        values = proposal[1:]
        # Keep ordinal IDs and signs stable. Projection onto intervals defined
        # by the old midpoints also retains the fixed-assignment LS guarantee.
        same_atom = layout[depth, :-1] == layout[depth, 1:]
        midpoint = (prior[:-1]+prior[1:])*.5
        upper = torch.where(same_atom, midpoint, torch.inf)
        lower = torch.where(same_atom, midpoint, -torch.inf)
        values[:-1] = torch.minimum(values[:-1], upper)
        values[1:] = torch.maximum(values[1:], lower)
        values.copy_(torch.where(prior < 0, values.clamp(max=-1.e-6), values.clamp(min=1.e-6)))
        proposal[0] = 0
        difference = proposal[chosen]-old[chosen]
        error.sub_(vectors*difference[:, None])
        quantizer.coefficients[depth].copy_(proposal)
    after = float(error.square().mean())
    assert torch.isfinite(quantizer.dictionary).all() and torch.isfinite(quantizer.coefficients).all()
    assert after_dictionary <= before+2.e-6*max(1., before)
    assert after <= after_dictionary+2.e-6*max(1., after_dictionary)
    return dict(fixed_assignment_mse_before=before,
        fixed_assignment_mse_after_dictionary=after_dictionary, fixed_assignment_mse_after=after)
