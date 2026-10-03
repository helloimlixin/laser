"""Soft targets scored in complete sparse-reconstruction geometry.

Fresh candidate supports are proposed using full least-squares reconstruction
improvement. Complete supports are refitted and weighted by their total squared
error. This is a finite-candidate teacher, not enumeration of all supports.
Coefficient noise follows the inverse support Gram matrix; sequential discrete
Gaussian kernels approximate the continuous full-combination Gaussian law.
Both target heads exactly marginalize the specified finite-candidate law given
the observed pair prefix. They never condition on future emitted pairs.
"""
from dataclasses import dataclass
import math

import torch


POLICY_VERSION = 'full_combination_neighbor_candidates_correlated_coefficients_v1'
EXPANDED_POLICY_VERSION = 'full_combination_cartesian_candidates_correlated_coefficients_v2'


@dataclass
class CombinationBank:
    atoms: torch.Tensor
    centers: torch.Tensor
    covariance_cholesky: torch.Tensor
    errors: torch.Tensor
    unique: torch.Tensor


@dataclass
class CombinationTargets:
    atoms: torch.Tensor
    coefficient_ids: torch.Tensor
    atom_target_ids: torch.Tensor
    atom_target_weights: torch.Tensor
    coefficient_probabilities: torch.Tensor
    physical_coefficients: torch.Tensor
    atom_entropy: torch.Tensor
    coefficient_entropy: torch.Tensor
    selected_variant: torch.Tensor


def _unique_ordered_supports(atoms, vocabulary):
    """Exact, linear-memory deduplication; preserve the earliest occurrence."""
    if vocabulary ** atoms.shape[-1] > torch.iinfo(torch.int64).max:
        raise ValueError('Ordered-support keys exceed int64')
    key = torch.zeros_like(atoms[..., 0])
    for component in atoms.unbind(-1):
        key = key * vocabulary + component
    ordered, index = key.sort(dim=-1, stable=True)
    first = torch.ones_like(ordered, dtype=torch.bool)
    first[..., 1:] = ordered[..., 1:] != ordered[..., :-1]
    return torch.zeros_like(first).scatter_(-1, index, first)


def _small_spd_solve(matrix, rhs):
    """Unrolled small Cholesky solve; compilable into one elementwise kernel."""
    depth = rhs.shape[-1]
    lower = {}; forward = []; positive = torch.ones_like(rhs[..., 0], dtype=torch.bool)
    for i in range(depth):
        for j in range(i):
            lower[i,j] = (matrix[..., i,j] - sum(lower[i,k]*lower[j,k] for k in range(j))) / lower[j,j]
        pivot = matrix[..., i,i] - sum(lower[i,k].square() for k in range(i))
        positive = positive & (pivot > 1e-6)
        lower[i,i] = pivot.clamp_min(1e-6).sqrt()
        forward.append((rhs[..., i]-sum(lower[i,k]*forward[k] for k in range(i)))/lower[i,i])
    result = {}
    for i in reversed(range(depth)):
        result[i] = (forward[i]-sum(lower[k,i]*result[k] for k in range(i+1,depth)))/lower[i,i]
    return torch.stack([result[i] for i in range(depth)], -1), positive


_compiled_small_spd_solve = torch.compile(_small_spd_solve, fullgraph=True, dynamic=True)


def _triangular_spd_solve(rhs, chol):
    y = torch.linalg.solve_triangular(chol, rhs, upper=False)
    return torch.linalg.solve_triangular(chol.transpose(-1,-2), y, upper=True)


@torch.no_grad()
def propose_expanded_bank(signals, dictionary, *, depth=4, swaps_per_atom=32,
                          cartesian_per_atom=4, max_variants=128, gram=None,
                          return_diagnostics=False):
    """Full-vocabulary replacements plus simultaneous multi-position changes.

    Each slot has its greedy atom and its best full-refit replacements. Score
    the Cartesian product of these slot choices, as well as a wider set of
    single replacements. Invalid/repeated atoms and duplicate ordered supports
    receive no mass. Retain the anchor plus the best complete combinations.
    This is a deterministic, finite search, not exhaustive support enumeration.
    Diagnostics describe the entire searched pool before retention.
    """
    if not 1 <= cartesian_per_atom <= swaps_per_atom or max_variants < 2:
        raise ValueError('Invalid expanded support search sizes')
    with torch.autocast(device_type=signals.device.type, enabled=False):
        gram = dictionary.T @ dictionary if gram is None else gram
        single_atoms = propose_swap_bank(signals, dictionary, depth=depth,
                                    swaps_per_atom=swaps_per_atom, gram=gram, return_atoms_only=True)
        n, _, depth = single_atoms.shape
        choices = torch.stack([torch.cat([
            single_atoms[:, :1, d],
            single_atoms[:, 1+d*swaps_per_atom:1+d*swaps_per_atom+cartesian_per_atom, d]
        ], -1) for d in range(depth)], -2)
        grid = torch.cartesian_prod(*[
            torch.arange(cartesian_per_atom+1, device=signals.device) for _ in range(depth)])
        combinations = choices[:, torch.arange(depth, device=signals.device), grid]
        atoms = torch.cat([single_atoms, combinations], 1)
        valid = _unique_ordered_supports(atoms, dictionary.shape[1])
        valid &= (atoms.sort(-1).values.diff(dim=-1) != 0).all(-1)
        matrices = gram[atoms[..., :, None], atoms[..., None, :]]
        correlation = signals @ dictionary
        rhs = correlation[:, None].expand(-1, atoms.shape[1], -1).gather(-1, atoms)
        solve = _compiled_small_spd_solve if signals.is_cuda else _small_spd_solve
        centers, positive = solve(matrices, rhs)
        valid &= positive
        # Quadratic identity avoids materializing [sites, candidates, dim].
        errors = (signals.square().sum(-1, keepdim=True) - (rhs*centers).sum(-1)).masked_fill(~valid, torch.inf)
        remaining = errors.clone(); remaining[:, 0] = torch.inf
        count = min(max_variants-1, atoms.shape[1]-1)
        selected = torch.cat([torch.zeros(n, 1, device=signals.device, dtype=torch.long),
                              remaining.topk(count, dim=-1, largest=False).indices], -1)
        rows = torch.arange(n, device=signals.device)[:, None]
        kept_atoms = atoms[rows, selected]
        kept_centers = centers[rows, selected]
        kept_matrices = matrices[rows, selected]
        identity = torch.eye(depth, dtype=signals.dtype, device=signals.device)
        kept_chol = torch.linalg.cholesky_ex(torch.where(valid[rows, selected, None, None], kept_matrices, identity))[0]
        inverse_factor = torch.linalg.solve_triangular(kept_chol, identity.expand_as(kept_chol), upper=False)
        covariance = inverse_factor.transpose(-1,-2) @ inverse_factor
        # Explicit error is used for the actual teacher distribution.
        reconstruction = torch.zeros(*kept_atoms.shape[:-1], dictionary.shape[0], device=signals.device, dtype=signals.dtype)
        for d in range(depth):
            reconstruction.add_(dictionary.T[kept_atoms[..., d]] * kept_centers[..., d, None])
        kept_errors = (signals[:, None] - reconstruction).square().sum(-1)
        bank = CombinationBank(kept_atoms, kept_centers,
            torch.linalg.cholesky_ex(covariance)[0], kept_errors, valid[rows, selected])
        if return_diagnostics:
            return bank, dict(errors=errors, valid=valid, selected=selected,
                              changed_slots=(atoms != single_atoms[:, :1]).sum(-1),
                              original_nine_indices=[0]+[1+d*swaps_per_atom+j for d in range(depth) for j in range(min(2, swaps_per_atom))])
        return bank


@torch.no_grad()
def propose_swap_bank(signals, dictionary, *, depth=4, swaps_per_atom=2,
                      proposal_temperature=.125, gram=None, generator=None, return_atoms_only=False):
    """Full-vocabulary single-coordinate alternatives to a greedy OMP support.

For each position, keep the OTHER three atoms and score every replacement by
the error after jointly refitting all four coefficients. This reuses one
signal/dictionary product, avoiding eight separate OMP trajectories per site.
    The candidate set is deterministic given the image: the anchor plus the best
    distinct replacement atoms for each position. Stochastic support draws occur
    AFTER complete-combination scoring. This avoids hidden random proposal noise
    missing from the reported conditional target entropy. proposal_temperature
    and generator are accepted only for compatibility with the audited prototype.
"""
    if (signals.ndim != 2 or dictionary.ndim != 2 or signals.shape[-1] != dictionary.shape[0]
            or signals.dtype not in (torch.float32, torch.float64)
            or dictionary.dtype != signals.dtype or signals.device != dictionary.device
            or not 2 <= depth <= min(dictionary.shape) or swaps_per_atom < 1
            or not math.isfinite(proposal_temperature) or proposal_temperature <= 0):
        raise ValueError('Invalid swap bank inputs')
    with torch.autocast(device_type=signals.device.type, enabled=False):
        gram = dictionary.T @ dictionary if gram is None else gram
        correlation = signals @ dictionary
        residual_correlation = correlation
        n, vocab = correlation.shape
        rows = torch.arange(n, device=signals.device)
        available = torch.ones_like(correlation, dtype=torch.bool)
        anchor = torch.empty(n, 0, device=signals.device, dtype=torch.long)
        for d in range(depth):
            atom = residual_correlation.abs().masked_fill(~available, -torch.inf).argmax(-1)
            anchor = torch.cat([anchor, atom[:, None]], -1)
            available[rows, atom] = False
            g = gram[anchor[..., :, None], anchor[..., None, :]]
            chol = torch.linalg.cholesky(g)
            fitted = _triangular_spd_solve(correlation.gather(1, anchor)[..., None], chol).squeeze(-1)
            residual_correlation = correlation - (fitted[:, None] @ gram[anchor]).squeeze(1)
        proposals = [anchor[:, None]]
        for d in range(depth):
            other = torch.cat([anchor[:, :d], anchor[:, d+1:]], -1)
            cross = gram[other]
            chol = torch.linalg.cholesky(gram[other[..., :, None], other[..., None, :]])
            projection = torch.linalg.solve_triangular(chol, cross, upper=False)
            norm = gram.diagonal()[None] - projection.square().sum(-2)
            fitted = _triangular_spd_solve(correlation.gather(1, other)[..., None], chol).squeeze(-1)
            residual_corr = correlation - (fitted[:, None] @ cross).squeeze(1)
            valid = norm > 1e-6
            valid.scatter_(1, anchor, False)
            score = (residual_corr.square()/norm.clamp_min(1e-6)).masked_fill(~valid, -torch.inf)
            candidates = score.topk(swaps_per_atom, dim=-1).indices
            support = anchor[:, None].expand(-1, swaps_per_atom, -1).clone()
            support[..., d] = candidates
            proposals.append(support)
        atoms = torch.cat(proposals, 1)
        if return_atoms_only:
            return atoms
        g = gram[atoms[..., :, None], atoms[..., None, :]]
        chol = torch.linalg.cholesky(g)
        rhs = correlation[:, None].expand(-1, atoms.shape[1], -1).gather(-1, atoms)
        centers = torch.cholesky_solve(rhs[..., None], chol).squeeze(-1)
        error = (signals[:, None] - (dictionary.T[atoms]*centers[..., None]).sum(-2)).square().sum(-1)
        covariance_cholesky = torch.linalg.cholesky(torch.cholesky_inverse(chol))
        unique = _unique_ordered_supports(atoms, vocab)
        return CombinationBank(atoms, centers, covariance_cholesky, error, unique)


@torch.no_grad()
def propose_combination_bank(signals, dictionary, *, depth=4, variants=8,
                             proposal_temperature=.0625, gram=None, generator=None):
    """Include greedy OMP plus fresh stochastic least-squares support proposals.

The improvement from adding atom a is <r,d_a>² / ||P_perp d_a||²,
which equals the reduction in error after refitting the ENTIRE active support.
The first trajectory preserves the existing deterministic OMP anchor.
"""
    if (signals.ndim != 2 or dictionary.ndim != 2 or signals.shape[-1] != dictionary.shape[0]
            or signals.dtype not in (torch.float32, torch.float64)
            or dictionary.dtype != signals.dtype or signals.device != dictionary.device
            or not 1 <= depth <= min(dictionary.shape) or variants < 2
            or not math.isfinite(proposal_temperature) or proposal_temperature <= 0):
        raise ValueError('Invalid combination bank inputs')
    with torch.autocast(device_type=signals.device.type, enabled=False):
        gram = dictionary.T @ dictionary if gram is None else gram
        n, vocab = len(signals), dictionary.shape[1]
        x = signals[:, None, :].expand(-1, variants, -1).reshape(-1, signals.shape[-1])
        correlation = (signals @ dictionary)[:, None, :].expand(-1, variants, -1).reshape(-1, vocab)
        residual_correlation = correlation
        available = torch.ones_like(correlation, dtype=torch.bool)
        support = torch.empty(len(x), 0, device=x.device, dtype=torch.long)
        rows = torch.arange(len(x), device=x.device)
        diagonal = gram.diagonal()
        for d in range(depth):
            if d:
                cross = gram[support]
                projected = torch.linalg.solve_triangular(chol, cross, upper=False)
                perpendicular_norm = diagonal[None] - projected.square().sum(-2)
            else:
                perpendicular_norm = diagonal[None].expand(len(x), -1)
            valid = available & (perpendicular_norm > 1e-6)
            improvement = residual_correlation.square() / perpendicular_norm.clamp_min(1e-6)
            score = improvement.masked_fill(~valid, -torch.inf)
            probabilities = ((score-score.amax(-1, keepdim=True))/proposal_temperature).softmax(-1)
            atom = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
            # Original greedy OMP anchor, including its original atom ordering.
            anchor = residual_correlation[::variants].abs().masked_fill(~valid[::variants], -torch.inf).argmax(-1)
            atom[::variants] = anchor
            available[rows, atom] = False
            support = torch.cat([support, atom[:, None]], -1)
            selected_gram = gram[support[..., :, None], support[..., None, :]]
            chol = torch.linalg.cholesky(selected_gram)
            centers = torch.cholesky_solve(correlation.gather(1, support)[..., None], chol).squeeze(-1)
            residual_correlation = correlation - (centers[:, None] @ gram[support]).squeeze(1)
        active = dictionary.T[support]
        residual = x - (active * centers[..., None]).sum(-2)
        errors = residual.square().sum(-1).reshape(n, variants)
        covariance_cholesky = torch.linalg.cholesky(torch.cholesky_inverse(chol))
        atoms = support.reshape(n, variants, depth)
        # Identical ordered supports count once, rather than gaining mass from
        # proposal duplication. Different orderings are different AR sequences.
        same = (atoms[:, :, None] == atoms[:, None, :]).all(-1)
        prior = torch.tril(torch.ones(variants, variants, device=x.device, dtype=torch.bool), diagonal=-1)
        unique = ~(same & prior).any(-1)
        return CombinationBank(atoms, centers.reshape(n, variants, depth),
                               covariance_cholesky.reshape(n, variants, depth, depth), errors, unique)


@torch.no_grad()
def combination_targets(bank, coefficient_values, *, support_temperature,
                        coefficient_temperature, generator=None, coefficient_kernel_radius=None):
    """Exact prefix-conditional labels for the explicitly defined bank law.

Support mass is proportional to exp(-||z-D_S c*_S||² / support_temperature).
Given S, continuous covariance is coefficient_temperature/2 * (D_S.T D_S)^-1.
Discretization normalizes each conditional on the actual finite bin grid.
"""
    depth = bank.atoms.shape[-1]
    temperatures = ([float(coefficient_temperature)] * depth if isinstance(coefficient_temperature, (int, float))
                    else list(coefficient_temperature))
    if (len(temperatures) != depth or not all(math.isfinite(t) and t > 0 for t in [support_temperature, *temperatures])
            or coefficient_values.ndim != 2 or coefficient_values.shape[0] != bank.atoms.shape[-1]
            or coefficient_values.device != bank.centers.device or coefficient_values.dtype != bank.centers.dtype):
        raise ValueError('Invalid temperatures or physical coefficient bins')
    with torch.autocast(device_type=bank.centers.device.type, enabled=False):
        n, variants, depth = bank.atoms.shape
        logits = (-(bank.errors-bank.errors.amin(-1, keepdim=True))/support_temperature).masked_fill(~bank.unique, -torch.inf)
        chosen = torch.multinomial(logits.softmax(-1), 1, generator=generator).squeeze(-1)
        row = torch.arange(n, device=chosen.device)
        observed_atoms = bank.atoms[row, chosen]
        log_posterior = logits.log_softmax(-1)
        whitened = torch.zeros_like(bank.centers)
        weight_list, coefficient_list, id_list, values_list, ah_list, ch_list = [], [], [], [], [], []
        for d, bins in enumerate(coefficient_values):
            weights = log_posterior.softmax(-1)
            weight_list.append(weights)
            # Aggregate equal IDs in linear candidate memory, with no host sync.
            ordered_ids, order = bank.atoms[..., d].sort(-1)
            starts = torch.ones_like(ordered_ids)
            starts[:, 1:] = ordered_ids[:, 1:] != ordered_ids[:, :-1]
            groups = starts.cumsum(-1)-1
            mass = torch.zeros_like(weights).scatter_add_(-1, groups, weights.gather(-1, order))
            ah_list.append(-(mass * mass.clamp_min(1e-30).log()).sum(-1))
            consistent = bank.atoms[..., d] == observed_atoms[:, d, None]
            conditioned = log_posterior.masked_fill(~consistent, -torch.inf)
            means = bank.centers[..., d] + (bank.covariance_cholesky[..., d, :d] * whitened[..., :d]).sum(-1)
            width = bank.covariance_cholesky[..., d, d]
            kernel_ids, log_kernel = _coefficient_kernel(bins, means, width, temperatures[d], coefficient_kernel_radius)
            mixed = conditioned.softmax(-1)[..., None] * log_kernel.exp()
            target = torch.zeros(n, len(bins), device=bins.device, dtype=bins.dtype)
            target.scatter_add_(-1, kernel_ids.reshape(n, -1), mixed.reshape(n, -1))
            kernel_index = torch.multinomial(log_kernel[row, chosen].exp(), 1, generator=generator)
            coefficient = kernel_ids[row, chosen].gather(-1, kernel_index).squeeze(-1)
            value = bins[coefficient]
            whitened[..., d] = (value[:, None] - means) / width
            coefficient_list.append(target)
            id_list.append(coefficient)
            values_list.append(value)
            ch_list.append(-(target * target.clamp_min(1e-30).log()).sum(-1))
            if d+1 < depth:
                likelihood = log_kernel.masked_fill(kernel_ids != coefficient[:, None, None], -torch.inf).logsumexp(-1)
                log_posterior = conditioned + likelihood
                log_posterior -= log_posterior.logsumexp(-1, keepdim=True)
        return CombinationTargets(observed_atoms, torch.stack(id_list, -1), bank.atoms.transpose(-1, -2),
                                  torch.stack(weight_list, -2), torch.stack(coefficient_list, -2),
                                  torch.stack(values_list, -1), torch.stack(ah_list, -1),
                                  torch.stack(ch_list, -1), chosen)


def _coefficient_kernel(bins, means, width, temperature, radius):
    """Compact finite-bin Gaussians with a conservative omitted-mass bound.

    At most 1e-30 relative mass is omitted; otherwise use the complete grid.
    This keeps the same Gaussian teacher to far below floating-point accuracy,
    including means outside the grid. Full-grid behavior is the default.
    """
    if radius is not None:
        if not isinstance(radius, int) or radius < 1:
            raise ValueError('Coefficient kernel radius must be a positive integer')
        spacing = bins[1]-bins[0]
        # Compact evaluation requires a uniform ascending grid.
        if not bool((spacing > 0) & torch.allclose(bins.diff(), spacing.expand(len(bins)-1), rtol=1e-3, atol=1e-7)):
            raise ValueError('Compact coefficient kernels require uniform ascending bins')
        nearest = ((means-bins[0])/spacing).round().long().clamp(0, len(bins)-1)
        raw = nearest[..., None] + torch.arange(-radius, radius+1, device=bins.device)
        valid = (raw >= 0) & (raw < len(bins))
        ids = raw.clamp(0, len(bins)-1)
        logits = (-((bins[ids]-means[..., None])/width[..., None]).square()/temperature).masked_fill(~valid, -torch.inf)
        log_z = logits.logsumexp(-1)
        left = (nearest-radius-1).clamp(0, len(bins)-1)
        right = (nearest+radius+1).clamp(0, len(bins)-1)
        omitted_left = (nearest-radius).clamp_min(0)
        omitted_right = (len(bins)-1-nearest-radius).clamp_min(0)
        tail = omitted_left * (-((bins[left]-means)/width).square()/temperature-log_z).exp()
        tail += omitted_right * (-((bins[right]-means)/width).square()/temperature-log_z).exp()
        if bool((tail <= 1e-30).all()):
            return ids, logits-log_z[..., None]
    ids = torch.arange(len(bins), device=bins.device).expand(*means.shape, -1)
    return ids, (-((bins-means[..., None])/width[..., None]).square()/temperature).log_softmax(-1)


def combination_soft_cross_entropy(atom_logits, coeff_logits, ids, weights, coefficient_targets):
    """Preserve the parent's equal-head loss scale; report pair CE separately."""
    log_atoms = atom_logits.float().log_softmax(-1).gather(-1, ids)
    atom_ce = -(weights * torch.where(weights > 0, log_atoms, 0.)).sum(-1)
    coeff_ce = -(coefficient_targets * coeff_logits.float().log_softmax(-1)).sum(-1)
    return (atom_ce + coeff_ce).mean()/2, atom_ce, coeff_ce


@torch.no_grad()
def _encode_combination_images(aux, images, policy):
    """Fresh images and proposals each visit, with bounded teacher memory."""
    fields = {}
    with torch.autocast(device_type=images.device.type, enabled=False):
        if not hasattr(aux, '_combination_gram'):
            aux._combination_gram = aux.dictionary.float().T @ aux.dictionary.float()
        z = torch.cat([aux.quant_conv(aux.encoder(x.float())).permute(0,2,3,1)
                       for x in images.split(int(policy.get('encoder_chunk_size',32)))])
        values = aux.coeff_bins.float()[None] * aux.coeff_scales.float()[:,None]
        shape = z.shape[:-1]
        for chunk in z.reshape(-1,z.shape[-1]).split(int(policy.get('site_chunk_size',256))):
            if policy.get('version') == EXPANDED_POLICY_VERSION:
                bank = propose_expanded_bank(chunk, aux.dictionary.float(), depth=aux.sparsity_level,
                    swaps_per_atom=policy['swaps_per_atom'], cartesian_per_atom=policy['cartesian_per_atom'],
                    max_variants=policy['max_variants'], gram=aux._combination_gram)
            else:
                bank = propose_swap_bank(chunk,aux.dictionary.float(),depth=aux.sparsity_level,
                                         swaps_per_atom=policy['swaps_per_atom'],
                                         proposal_temperature=policy['proposal_temperature'],gram=aux._combination_gram)
            result = combination_targets(bank,values,support_temperature=policy['support_temperature'],
                                         coefficient_temperature=policy['coefficient_temperature'],
                                         coefficient_kernel_radius=policy.get('coefficient_kernel_radius'))
            for key,value in vars(result).items():fields.setdefault(key,[]).append(value)
        merged={k:torch.cat(v) for k,v in fields.items()}
        for key,value in merged.items():merged[key]=value.reshape(*shape,*value.shape[1:])
        return CombinationTargets(**merged)


def encode_combination_images(aux, images, policy):
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        return _encode_combination_images(aux, images, policy)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
