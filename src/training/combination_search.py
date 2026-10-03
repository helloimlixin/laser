"""Bounded search over complete sparse vectors with frozen discrete coefficients.

This module generates experimental training candidates. It never changes an
emitted generation prefix or solves for continuous coefficients. Selection uses
full-vector error, including cross terms. The final Gibbs law is exact over the
retained finite set, not over the full dictionary/coefficient vocabulary.
"""
from dataclasses import dataclass
import itertools
import math

import torch

from .cartesian_pair_candidates import nearest_physical_coefficient_ids

POLICY_VERSION = 'bounded_two_slot_support_stratified_complete_vector_search_v2'


@dataclass
class PairPool:
    atoms: torch.Tensor
    coefficient_ids: torch.Tensor
    target_vectors: torch.Tensor


@dataclass
class CombinationBank:
    atoms: torch.Tensor
    coefficient_ids: torch.Tensor
    valid: torch.Tensor
    errors: torch.Tensor
    evaluated_candidates: torch.Tensor

    def probabilities(self, temperature):
        _temperature(temperature)
        return (-self.errors / temperature).masked_fill(~self.valid, -torch.inf).softmax(-1)

    def sample(self, temperature, *, draws=4, generator=None):
        if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
            raise ValueError('draws must be a positive integer')
        probability = self.probabilities(temperature)
        leading = probability.shape[:-1]
        selected = torch.multinomial(probability.reshape(-1, probability.shape[-1]),
                                     draws, replacement=True, generator=generator)
        depth = self.atoms.shape[-1]
        indices = selected[..., None].expand(-1, -1, depth)
        atoms = self.atoms.reshape(-1, self.atoms.shape[-2], depth).gather(1, indices)
        bins = self.coefficient_ids.reshape_as(self.atoms).reshape(-1, self.atoms.shape[-2], depth).gather(1, indices)
        return atoms.reshape(*leading, draws, depth), bins.reshape(*leading, draws, depth)

    def sample_novel_supports(self, anchor_atoms, temperature, *, draws=4, generator=None):
        """Distance-weighted loss proposals from different atom supports.

Keep the lowest-error observed tuple for each novel unordered support and draw
without replacement by Gumbel top-k. This samples finite proposal events; it is
not the context teacher's Gibbs law. Missing proposals use the anchor and are
marked false, so callers can deduplicate them without assigning extra mass.
        """
        _temperature(temperature)
        if (isinstance(draws, bool) or not isinstance(draws, int) or draws < 1
                or anchor_atoms.shape != self.atoms.shape[:-2] + self.atoms.shape[-1:]):
            raise ValueError('invalid proposal count or anchor shape')
        leading, count, depth = self.atoms.shape[:-2], self.atoms.shape[-2], self.atoms.shape[-1]
        a = self.atoms.reshape(-1, count, depth)
        b = self.coefficient_ids.reshape_as(a)
        error = self.errors.reshape(-1, count)
        support = a.sort(-1).values
        eligible = self.valid.reshape(-1, count) & (support != anchor_atoms.reshape(-1, 1, depth).sort(-1).values).any(-1)
        base = int(a.max()) + 1
        if base ** depth >= torch.iinfo(torch.int64).max:
            raise ValueError('support key exceeds supported bounds')
        key = torch.zeros_like(error, dtype=torch.long)
        for d in range(depth):
            key = key * base + support[..., d]
        order = error.argsort(dim=-1, stable=True)
        first = _first_unique(key.gather(1, order), eligible.gather(1, order))
        representatives = torch.zeros_like(eligible).scatter(1, order, first)
        uniform = torch.rand(error.shape, device=error.device, dtype=error.dtype, generator=generator)
        epsilon = torch.finfo(error.dtype).eps
        gumbel = -torch.log(-torch.log(uniform.clamp(epsilon, 1 - epsilon)))
        score = (-error / temperature + gumbel).masked_fill(~representatives, -torch.inf)
        selected = score.argsort(dim=-1, descending=True, stable=True)[:, :min(draws, count)]
        usable = representatives.gather(1, selected)
        selected = selected.masked_fill(~usable, 0)
        if selected.shape[-1] < draws:
            padding = draws - selected.shape[-1]
            selected = torch.cat((selected, torch.zeros(len(a), padding, dtype=torch.long, device=a.device)), -1)
            usable = torch.cat((usable, torch.zeros(len(a), padding, dtype=torch.bool, device=a.device)), -1)
        indices = selected[..., None].expand(-1, -1, depth)
        return (a.gather(1, indices).reshape(*leading, draws, depth),
                b.gather(1, indices).reshape(*leading, draws, depth),
                usable.reshape(*leading, draws))


def _temperature(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError('temperature must be finite and positive')


def _mixed(logits, mixture, excluded=None):
    if excluded is None:
        return (1 - mixture) * logits.softmax(-1) + mixture / logits.shape[-1]
    logits = logits.scatter(-1, excluded[..., None], -torch.inf)
    uniform = torch.full_like(logits, mixture / (logits.shape[-1] - 1))
    uniform.scatter_(-1, excluded[..., None], 0.)
    return (1 - mixture) * logits.softmax(-1) + uniform


@torch.no_grad()
def wide_pair_pool(anchor_atoms, physical_coefficients, dictionary, coefficient_values,
                   *, temperature, alternatives_per_depth=4, bins_per_atom=3,
                   uniform_mixture=.001, site_chunk_size=128, generator=None):
    """Several full-vocabulary alternative atoms and discrete coefficient draws.

Distance-aware atom proposals marginalize both signs of the cached binned
amplitude. Coefficients condition on their proposed atom and are drawn from the
entire fixed physical grid. The first branch is always the original atom with
its nearest bin. These proposals are exploratory; the teacher weights complete
vectors after search, rather than using independent pair weights as targets.
    """
    _temperature(temperature)
    if (any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in
            (alternatives_per_depth, bins_per_atom, site_chunk_size))
            or not isinstance(uniform_mixture, (int, float))
            or isinstance(uniform_mixture, bool) or not math.isfinite(uniform_mixture)
            or not 0 < uniform_mixture <= 1):
        raise ValueError('invalid proposal sizes or uniform mixture')
    if (anchor_atoms.shape != physical_coefficients.shape or anchor_atoms.ndim < 1
            or dictionary.ndim != 2 or dictionary.shape[1] < 2
            or coefficient_values.ndim != 2
            or coefficient_values.shape[0] != anchor_atoms.shape[-1]
            or dictionary.dtype not in (torch.float32, torch.float64)
            or any(t.dtype != dictionary.dtype or t.device != dictionary.device
                   for t in (physical_coefficients, coefficient_values))
            or anchor_atoms.device != dictionary.device
            or anchor_atoms.dtype not in (torch.int16, torch.int32, torch.int64)
            or anchor_atoms.numel() == 0 or not torch.isfinite(dictionary).all()
            or (anchor_atoms < 0).any() or (anchor_atoms >= dictionary.shape[1]).any()):
        raise ValueError('invalid immutable anchors, dictionary or coefficient grids')
    centers = nearest_physical_coefficient_ids(physical_coefficients, coefficient_values)
    shape, depth = anchor_atoms.shape[:-1], anchor_atoms.shape[-1]
    flat_a = anchor_atoms.reshape(-1, depth).long()
    flat_c = physical_coefficients.reshape(-1, depth)
    flat_b = centers.reshape(-1, depth)
    atom_chunks, bin_chunks, signals = [], [], []
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.autocast(device_type=dictionary.device.type, enabled=False):
            norms = dictionary.square().sum(0)
            for start in range(0, len(flat_a), site_chunk_size):
                a, c, b = flat_a[start:start+site_chunk_size], flat_c[start:start+site_chunk_size], flat_b[start:start+site_chunk_size]
                atom_vectors = dictionary.T[a]
                z = (atom_vectors * c[..., None]).sum(-2)
                binned = coefficient_values[torch.arange(depth, device=dictionary.device), b]
                contributions = atom_vectors * binned[..., None]
                residual = (z - contributions.sum(-2))[:, None] + contributions
                pool_a, pool_b = [], []
                for d in range(depth):
                    projection = residual[:, d] @ dictionary
                    signed = 2 * binned[:, d, None] * projection / temperature
                    penalty = binned[:, d, None].square() * norms / temperature
                    atom_p = _mixed(torch.logaddexp(signed - penalty, -signed - penalty),
                                    uniform_mixture, a[:, d])
                    alternate = torch.multinomial(atom_p, alternatives_per_depth,
                                                   replacement=True, generator=generator)
                    selected_atoms = torch.cat((a[:, d, None], alternate), -1)
                    selected_vectors = dictionary.T[selected_atoms]
                    dot = (residual[:, d, None] * selected_vectors).sum(-1)
                    selected_norms = norms[selected_atoms]
                    scores = (2 * dot[..., None] * coefficient_values[d]
                              - selected_norms[..., None] * coefficient_values[d].square()) / temperature
                    coefficient_p = _mixed(scores, uniform_mixture)
                    drawn = torch.multinomial(coefficient_p.reshape(-1, coefficient_values.shape[-1]),
                                              bins_per_atom, replacement=True, generator=generator)
                    drawn = drawn.reshape(len(a), alternatives_per_depth + 1, bins_per_atom)
                    drawn[:, 0, 0] = b[:, d]
                    pool_a.append(selected_atoms.repeat_interleave(bins_per_atom, -1))
                    pool_b.append(drawn.flatten(-2))
                atom_chunks.append(torch.stack(pool_a, 1))
                bin_chunks.append(torch.stack(pool_b, 1))
                signals.append(z)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    return PairPool(torch.cat(atom_chunks).reshape(*shape, depth, -1),
                    torch.cat(bin_chunks).reshape(*shape, depth, -1),
                    torch.cat(signals).reshape(*shape, dictionary.shape[0]))


def _first_unique(key, valid):
    """Keep the first valid occurrence of each token sequence, in original order."""
    sentinel = torch.iinfo(torch.int64).max
    order = key.masked_fill(~valid, sentinel).argsort(dim=-1, stable=True)
    ordered = key.gather(-1, order)
    ordered_valid = valid.gather(-1, order)
    keep = ordered_valid.clone()
    keep[:, 1:] &= (ordered[:, 1:] != ordered[:, :-1]) | ~ordered_valid[:, :-1]
    return torch.zeros_like(keep).scatter(-1, order, keep)


def _search_chunk(signals, dictionary, atoms, ids, grids, beam_width, sweeps, support_quota):
    sites, depth, choices = atoms.shape
    values = grids[torch.arange(depth, device=signals.device)[None, :, None], ids]
    contributions = dictionary.T[atoms] * values[..., None]
    differences = contributions - contributions[:, :, :1]
    residual = signals - contributions[:, :, 0].sum(1)
    constant = residual.square().sum(-1)
    unary = differences.square().sum(-1) - 2 * (residual[:, None, None] * differences).sum(-1)
    crosses = {(i, j): 2 * torch.bmm(differences[:, i], differences[:, j].transpose(1, 2))
               for i, j in itertools.combinations(range(depth), 2)}
    same = (atoms[..., :, None] == atoms[..., None, :]) & (ids[..., :, None] == ids[..., None, :])
    canonical = same.to(torch.int64).argmax(-1)
    rows = torch.arange(sites, device=signals.device)[:, None]
    beam = torch.zeros(sites, 1, depth, dtype=torch.long, device=signals.device)
    valid = torch.ones(sites, 1, dtype=torch.bool, device=signals.device)
    pairs = list(itertools.combinations(range(depth), 2)) if depth > 1 else [(0,)]
    evaluated = torch.zeros(sites, dtype=torch.long, device=signals.device)
    powers = torch.tensor([choices ** d for d in reversed(range(depth))], device=signals.device)
    for _ in range(sweeps):
        for block in pairs:
            states = torch.tensor(list(itertools.product(range(choices), repeat=len(block))),
                                  dtype=torch.long, device=signals.device)
            trial = beam[:, :, None].expand(-1, -1, len(states), -1).clone()
            for k, d in enumerate(block):
                trial[..., d] = canonical[:, d, states[:, k]][:, None]
            trial = trial.flatten(1, 2)
            trial_valid = valid[:, :, None].expand(-1, -1, len(states)).reshape(sites, -1).clone()
            # Keep anchor as a permanent reference and as a valid finite event.
            trial = torch.cat((torch.zeros_like(trial[:, :1]), trial), 1)
            trial_valid = torch.cat((torch.ones_like(trial_valid[:, :1]), trial_valid), 1)
            selected_atoms = torch.stack([atoms[:, d].gather(1, trial[..., d]) for d in range(depth)], -1)
            if depth > 1:
                trial_valid &= selected_atoms.sort(-1).values.diff(dim=-1).ne(0).all(-1)
            key = (trial * powers).sum(-1)
            trial_valid = _first_unique(key, trial_valid)
            energy = constant[:, None].expand_as(trial_valid).clone()
            for d in range(depth):
                energy += unary[:, d].gather(1, trial[..., d])
            for (i, j), cross in crosses.items():
                energy += cross[rows, trial[..., i], trial[..., j]]
            evaluated += trial_valid.sum(-1)
            energy = energy.masked_fill(~trial_valid, torch.inf)
            # Anchor is always retained; other states compete by COMPLETE error.
            energy[:, 0] = torch.inf
            count = min(beam_width - 1, trial.shape[1] - 1)
            order = energy.argsort(dim=-1, stable=True)
            if support_quota:
                sorted_atoms = selected_atoms.sort(-1).values
                support_key = torch.zeros_like(key)
                for d in range(depth):
                    support_key = support_key * dictionary.shape[1] + sorted_atoms[..., d]
                # Reserve diverse SUPPORTS, not merely different coefficient bins.
                eligible = trial_valid & torch.isfinite(energy) & support_key.ne(support_key[:, :1])
                representative_sorted = _first_unique(support_key.gather(1, order), eligible.gather(1, order))
                representatives = torch.zeros_like(eligible).scatter(1, order, representative_sorted)
                representative_energy = energy.masked_fill(~representatives, torch.inf)
                reserved_indices = representative_energy.argsort(dim=-1, stable=True)[:, :support_quota]
                reserved_valid = torch.isfinite(representative_energy.gather(1, reserved_indices))
                reserved = torch.zeros_like(eligible).scatter(1, reserved_indices, reserved_valid)
                # Stable ordering keeps lowest energy within each priority group.
                priority = (~reserved).gather(1, order).to(torch.int64).argsort(dim=-1, stable=True)
                order = order.gather(1, priority)
            selected = order[:, :count]
            selected = torch.cat((torch.zeros_like(selected[:, :1]), selected), -1)
            if count == 0:
                selected = torch.zeros(sites, 1, dtype=torch.long, device=signals.device)
            beam = trial.gather(1, selected[..., None].expand(-1, -1, depth))
            valid = trial_valid.gather(1, selected)
            valid[:, 1:] &= selected[:, 1:].ne(0)
    final_atoms = torch.stack([atoms[:, d].gather(1, beam[..., d]) for d in range(depth)], -1)
    final_ids = torch.stack([ids[:, d].gather(1, beam[..., d]) for d in range(depth)], -1)
    physical = grids[torch.arange(depth, device=signals.device), final_ids]
    reconstruction = (dictionary.T[final_atoms] * physical[..., None]).sum(-2)
    # Recompute the final energy directly; avoid quadratic cancellation in q.
    errors = (signals[:, None] - reconstruction).square().sum(-1)
    return final_atoms, final_ids, valid, errors, evaluated


@torch.no_grad()
def search_combinations(signals, dictionary, candidate_atoms, candidate_coefficient_ids,
                        coefficient_values, *, beam_width=32, sweeps=1, site_chunk_size=128,
                        support_quota=None):
    """Search two-slot changes and retain bounded unique complete token tuples.

Every expansion changes two unobserved training-candidate pairs together, scores
the complete vector, and retains the anchor plus low-error states. By default,
half the beam is reserved for distinct alternative supports, so coefficient
variants of one support cannot evict every new atom combination. Wider
beams can preserve alternatives for later blocks. This is a deterministic finite
search with pruning, not exact enumeration of all pool combinations. Caller
samples the retained bank's normalized Gibbs law using ``bank.sample``.
    """
    if (any(isinstance(v, bool) or not isinstance(v, int) or v < 1
            for v in (beam_width, sweeps, site_chunk_size))
            or signals.ndim < 2 or dictionary.ndim != 2 or coefficient_values.ndim != 2
            or candidate_atoms.ndim != signals.ndim + 1
            or candidate_atoms.shape != candidate_coefficient_ids.shape
            or candidate_atoms.shape[:-2] != signals.shape[:-1]
            or signals.shape[-1] != dictionary.shape[0]
            or candidate_atoms.shape[-2] != coefficient_values.shape[0]
            or candidate_atoms.shape[-2] < 1 or candidate_atoms.shape[-1] < 1
            or signals.numel() == 0 or dictionary.dtype not in (torch.float32, torch.float64)
            or any(t.dtype != dictionary.dtype or t.device != dictionary.device
                   for t in (signals, coefficient_values))
            or any(t.dtype not in (torch.int16, torch.int32, torch.int64)
                   or t.device != dictionary.device for t in (candidate_atoms, candidate_coefficient_ids))
            or not all(torch.isfinite(t).all() for t in (signals, dictionary, coefficient_values))
            or (candidate_atoms < 0).any() or (candidate_atoms >= dictionary.shape[1]).any()
            or (candidate_coefficient_ids < 0).any()
            or (candidate_coefficient_ids >= coefficient_values.shape[1]).any()):
        raise ValueError('invalid complete-vector search inputs or search budget')
    depth, choices = candidate_atoms.shape[-2:]
    if choices ** depth >= torch.iinfo(torch.int64).max or beam_width > 65536:
        raise ValueError('search budget or token key exceeds supported bounds')
    if support_quota is None:
        support_quota = min(beam_width - 1, beam_width // 2)
    if (isinstance(support_quota, bool) or not isinstance(support_quota, int)
            or not 0 <= support_quota < beam_width
            or (support_quota and dictionary.shape[1] ** depth >= torch.iinfo(torch.int64).max)):
        raise ValueError('invalid support quota or support key exceeds supported bounds')
    anchor = candidate_atoms[..., 0]
    if depth > 1 and (anchor.sort(-1).values.diff(dim=-1) == 0).any():
        raise ValueError('anchor atoms must be distinct')
    leading = signals.shape[:-1]
    flat_signals = signals.reshape(-1, dictionary.shape[0])
    flat_atoms = candidate_atoms.reshape(-1, depth, choices).long()
    flat_ids = candidate_coefficient_ids.reshape(-1, depth, choices).long()
    parts = []
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.autocast(device_type=dictionary.device.type, enabled=False):
            for start in range(0, len(flat_signals), site_chunk_size):
                parts.append(_search_chunk(flat_signals[start:start+site_chunk_size], dictionary,
                    flat_atoms[start:start+site_chunk_size], flat_ids[start:start+site_chunk_size],
                    coefficient_values, beam_width, sweeps, support_quota))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    combined = [torch.cat([p[k] for p in parts]).reshape(*leading, *parts[0][k].shape[1:])
                for k in range(5)]
    return CombinationBank(*combined)
