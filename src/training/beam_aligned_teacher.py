"""A coherent soft AR teacher over complete, high-quality beam trajectories.

The sequence is sampled once from the beam posterior. Every next-token label
is its exact conditional given the sampled prefix. No local greedy teacher is
substituted after lookahead, and every sequence uses the frozen physical book.
"""
import torch


@torch.no_grad()
def beam_pool(book, latent, beam_width=16, chunk_size=256):
    flat = latent.reshape(-1, latent.shape[-1]).float()
    greedy = book._quantize_greedy(latent, chunk_size=chunk_size)
    anchor = greedy['codes'].reshape(-1, book.depth)
    anchor_values = greedy['quantized'].reshape_as(flat)
    all_codes, all_costs, all_canonical, all_valid = [], [], [], []
    for first in range(0, len(flat), chunk_size):
        x = flat[first:first + chunk_size]
        n = len(x)
        residual = x[:, None].clone()
        history = torch.empty(n, 1, 0, device=x.device, dtype=torch.long)
        for depth in range(book.depth):
            branches = residual.shape[1]
            score = book.scores(residual.reshape(-1, x.shape[-1]), depth).reshape(n, branches, -1)
            cost = residual.square().sum(-1, keepdim=True) - score
            width = min(beam_width, cost.shape[1] * cost.shape[2])
            chosen = cost.flatten(1).topk(width, dim=-1, largest=False).indices
            parent, token = chosen // book.vocab_size, chosen % book.vocab_size
            residual = residual.gather(1, parent[..., None].expand(-1, -1, x.shape[-1]))
            residual = residual - book.embed_depth(token, depth)
            history = torch.cat([history.gather(1, parent[..., None].expand(-1, -1, depth)), token[..., None]], -1)
        # Use the exact canonical selection rule of the measured beam encoder.
        best = residual.square().sum(-1).argmin(-1)
        row = torch.arange(n, device=x.device)
        candidate_values = book.embed(history).sum(-2)
        costs = (x[:, None] - candidate_values).square().sum(-1)
        anchor_cost = (x - anchor_values[first:first+n]).square().sum(-1)
        canonical = torch.where(costs[row, best] < anchor_cost, best, torch.full_like(best, width))
        anchor_codes = anchor[first:first+n]
        # Keep the anchor and remove its duplicate from the terminal beam.
        valid = ~(history == anchor_codes[:, None]).all(-1)
        valid = torch.cat([valid, torch.ones(n, 1, dtype=torch.bool, device=x.device)], -1)
        all_codes.append(torch.cat([history, anchor_codes[:, None]], 1))
        all_costs.append(torch.cat([costs, anchor_cost[:, None]], 1))
        all_canonical.append(canonical)
        all_valid.append(valid)
    return dict(candidates=torch.cat(all_codes), costs=torch.cat(all_costs),
                canonical_index=torch.cat(all_canonical), valid=torch.cat(all_valid),
                latent_shape=tuple(latent.shape[:-1]))


def posterior(pool, temperature, max_relative_error=.01):
    costs, canonical = pool['costs'], pool['canonical_index']
    best_cost = costs.gather(1, canonical[:, None])
    delta = (costs - best_cost).clamp_min(0)
    valid = pool['valid'] & (delta <= max_relative_error * best_cost.clamp_min(1e-8) + 1e-6)
    valid.scatter_(1, canonical[:, None], True)
    if temperature == 0:
        weights = torch.zeros_like(costs)
        weights.scatter_(1, canonical[:, None], 1.)
        return weights
    if temperature <= 0: raise ValueError('Nonnegative finite sequence temperature required')
    return (-delta / float(temperature)).masked_fill(~valid, -torch.inf).softmax(-1)


def conditional_targets(candidates, weights, codes, vocabulary):
    n, _, depth = candidates.shape
    targets = weights.new_zeros(n, depth, vocabulary)
    matching = torch.ones_like(weights, dtype=torch.bool)
    for d in range(depth):
        mass = weights * matching
        mass = mass / mass.sum(-1, keepdim=True).clamp_min(1e-30)
        targets[:, d].scatter_add_(1, candidates[:, :, d], mass)
        matching &= candidates[:, :, d] == codes[:, d, None]
    return targets


@torch.no_grad()
def aligned_targets(book, latent, temperature, *, stochastic=True, return_details=False, pool_fn=None):
    pool = (pool_fn or beam_pool)(book, latent)
    weights = posterior(pool, temperature)
    chosen = torch.multinomial(weights, 1).squeeze(-1) if stochastic else pool['canonical_index']
    row = torch.arange(len(chosen), device=chosen.device)
    codes = pool['candidates'][row, chosen]
    targets = conditional_targets(pool['candidates'], weights, codes, book.vocab_size)
    shape = pool['latent_shape']
    result = (targets.reshape(*shape, book.depth, book.vocab_size), codes.reshape(*shape, book.depth))
    if return_details: return (*result, pool, weights, chosen)
    return result


def verify_conditionals():
    # Different prefixes share a first token, so testing only normalization
    # would miss the distinction between marginals and conditional labels.
    candidates = torch.tensor([[[1,2,3],[1,2,4],[1,5,6],[7,8,9]]])
    weights = torch.tensor([[.1,.2,.3,.4]], dtype=torch.float64)
    for k in range(4):
        codes = candidates[:, k]
        targets = conditional_targets(candidates, weights, codes, 10)
        selected = targets.gather(-1, codes[..., None]).squeeze(-1)
        torch.testing.assert_close(selected.prod(-1), weights[:, k], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(targets.sum(-1), torch.ones(1,3,dtype=weights.dtype))
    return dict(passed=True, exact_joint_probability_from_ar_conditionals=True,
                common_prefixes_tested=True)
