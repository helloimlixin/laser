"""Exact finite-Cartesian soft targets in complete sparse-vector geometry.

Each slot provides a fixed list of atom/coefficient token pairs.  The teacher
scores the vector SUM of a complete tuple, including all cross terms, and has
probability proportional to the product base mass times exp(-error / tau).
Sampling and both head targets use this same distribution.  Conditioning is on
observed token pairs, never on an unobserved candidate-list index.  This matters
when different candidate branches alias the same pair.

This is exact on the supplied finite candidate grid, not an approximation claim
about the unrestricted vocabulary.  No coefficient is fitted or revised here.
If signals are cached reconstructions, error measures reconstruction preservation
rather than distortion to the original encoder output; callers must record this.
"""
from dataclasses import dataclass
import math

import torch


POLICY_VERSION = "finite_cartesian_full_vector_no_refit_v1"


@dataclass
class CartesianCombinationTargets:
    atoms: torch.Tensor
    coefficient_ids: torch.Tensor
    physical_coefficients: torch.Tensor
    atom_target_ids: torch.Tensor
    atom_target_weights: torch.Tensor
    coefficient_target_ids: torch.Tensor
    coefficient_target_weights: torch.Tensor
    reconstruction: torch.Tensor
    expected_distortion: torch.Tensor
    sampled_distortion: torch.Tensor
    minimum_distortion: torch.Tensor
    joint_entropy: torch.Tensor
    atom_entropy: torch.Tensor
    coefficient_entropy: torch.Tensor


def _token_entropy(ids, weights):
    """Entropy after adding sparse entries that name the same token."""
    order = ids.argsort(dim=-1, stable=True)
    ordered = ids.gather(-1, order)
    starts = torch.ones_like(ordered)
    starts[..., 1:] = ordered[..., 1:] != ordered[..., :-1]
    groups = starts.cumsum(-1) - 1
    mass = torch.zeros_like(weights).scatter_add_(-1, groups, weights.gather(-1, order))
    return -(mass * mass.clamp_min(torch.finfo(mass.dtype).tiny).log()).sum(-1)


def _deduplicate_log_mass(atoms, coefficient_ids, log_mass):
    """Keep the first candidate for each exact token pair at each site/slot."""
    choices = atoms.shape[-1]
    same = ((atoms[..., :, None] == atoms[..., None, :])
            & (coefficient_ids[..., :, None] == coefficient_ids[..., None, :]))
    earlier = torch.tril(torch.ones(choices, choices, dtype=torch.bool,
                                    device=atoms.device), diagonal=-1)
    # A masked earlier branch must not discard a later usable branch.
    same = same & torch.isfinite(log_mass)[..., None, :]
    return log_mass.masked_fill((same & earlier).any(-1), -torch.inf)


@torch.no_grad()
def cartesian_combination_targets(signals, dictionary, candidate_atoms,
                                  candidate_coefficient_ids, coefficient_values,
                                  *, temperature, candidate_log_base_mass=None,
                                  forbid_repeated_atoms=True, site_chunk_size=128,
                                  duplicate_policy="deduplicate", generator=None):
    """Return exact sampled-prefix marginals of a bounded combination teacher.

    Shapes: signals [..., C], dictionary [C, A], candidate atom/coefficient IDs
    [..., K, L], physical coefficient_values [K, B].  Optional log base masses
    have shape [..., K, L]; -inf masks a branch.  Their sum is the log mass of a
    complete tuple.  Inputs share a device and physical arrays use FP32/FP64.

    ``duplicate_policy='deduplicate'`` retains the first finite-base branch for
    an identical token pair in each slot (including that branch's base mass).
    ``'sum'`` retains branch multiplicities, which are explicitly part of the
    base measure. Both policies condition correctly on token identity.

    Sparse target IDs may repeat: their weights add to the probability of that
    token. Coefficient targets condition on the returned current atom; future
    tuples are marginalized. The sampled complete tuple is never changed.
    Chunk size is part of RNG replay because draws occur once per site chunk.
    """
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("temperature must be finite and positive")
    if (isinstance(site_chunk_size, bool) or not isinstance(site_chunk_size, int)
            or site_chunk_size < 1):
        raise ValueError("site_chunk_size must be a positive integer")
    if duplicate_policy not in ("deduplicate", "sum"):
        raise ValueError("duplicate_policy must be 'deduplicate' or 'sum'")
    if (signals.ndim < 2 or dictionary.ndim != 2 or coefficient_values.ndim != 2
            or candidate_atoms.ndim != signals.ndim + 1
            or candidate_coefficient_ids.shape != candidate_atoms.shape
            or candidate_atoms.shape[:-2] != signals.shape[:-1]
            or signals.shape[-1] != dictionary.shape[0]
            or candidate_atoms.shape[-2] != coefficient_values.shape[0]
            or any(t.numel() == 0 for t in (signals, dictionary, candidate_atoms,
                                          coefficient_values))):
        raise ValueError("invalid signal, dictionary, candidate, or grid shapes")
    floats = (signals, dictionary, coefficient_values)
    if (signals.dtype not in (torch.float32, torch.float64)
            or any(t.device != signals.device or t.dtype != signals.dtype for t in floats)
            or any(t.device != signals.device for t in (candidate_atoms, candidate_coefficient_ids))
            or any(t.dtype not in (torch.int32, torch.int64)
                   for t in (candidate_atoms, candidate_coefficient_ids))):
        raise ValueError("inputs must share a device and compatible FP32/FP64 or integer dtypes")
    if (not all(bool(torch.isfinite(t).all()) for t in floats)
            or bool((candidate_atoms < 0).any())
            or bool((candidate_atoms >= dictionary.shape[1]).any())
            or bool((candidate_coefficient_ids < 0).any())
            or bool((candidate_coefficient_ids >= coefficient_values.shape[1]).any())):
        raise ValueError("inputs must be finite with valid atom and coefficient IDs")
    if candidate_log_base_mass is None:
        log_base = torch.zeros_like(candidate_atoms, dtype=signals.dtype)
    else:
        if (candidate_log_base_mass.shape != candidate_atoms.shape
                or candidate_log_base_mass.device != signals.device
                or candidate_log_base_mass.dtype != signals.dtype
                or bool(torch.isnan(candidate_log_base_mass).any())
                or bool(torch.isposinf(candidate_log_base_mass).any())):
            raise ValueError("log base masses must match candidates and may only mask with -inf")
        log_base = candidate_log_base_mass
    leading = signals.shape[:-1]
    depth, choices = candidate_atoms.shape[-2:]
    count = choices ** depth
    if count > 2_000_000:
        raise ValueError("Cartesian candidate grid exceeds 2,000,000 tuples per site")
    with torch.autocast(device_type=signals.device.type, enabled=False):
        flat_signals = signals.reshape(-1, signals.shape[-1])
        flat_atoms = candidate_atoms.reshape(-1, depth, choices).long()
        flat_ids = candidate_coefficient_ids.reshape(-1, depth, choices).long()
        flat_base = log_base.reshape(-1, depth, choices)
        powers = torch.tensor([choices ** d for d in reversed(range(depth))],
                              device=signals.device, dtype=torch.long)
        states = torch.arange(count, device=signals.device)[:, None].div(
            powers, rounding_mode="floor").remainder(choices)
        fields = {}
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            # Geometry is evaluated in full FP32/FP64 even when model matmuls
            # use TF32. Restore the caller's setting after failures as well.
            torch.backends.cuda.matmul.allow_tf32 = False
            for start in range(0, len(flat_signals), site_chunk_size):
                stop = start + site_chunk_size
                result = _chunk_targets(flat_signals[start:stop], dictionary,
                                        flat_atoms[start:stop], flat_ids[start:stop],
                                        coefficient_values, flat_base[start:stop], states,
                                        temperature, forbid_repeated_atoms,
                                        duplicate_policy, generator)
                for name, value in result.items():
                    fields.setdefault(name, []).append(value)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        combined = {name: torch.cat(parts).reshape(*leading, *parts[0].shape[1:])
                    for name, parts in fields.items()}
        return CartesianCombinationTargets(**combined)


def _chunk_targets(signals, dictionary, atoms, ids, grids, log_base, states,
                   temperature, forbid_repeated_atoms, duplicate_policy, generator):
    sites, depth, choices = atoms.shape
    if duplicate_policy == "deduplicate":
        log_base = _deduplicate_log_mass(atoms, ids, log_base)
    values = grids[torch.arange(depth, device=signals.device)[None, :, None], ids]
    vectors = dictionary.T[atoms] * values[..., None]
    # Expand around one fixed reconstruction to reduce cancellation between
    # large signal/atom norms. This is the same full-sum energy, not a refit.
    residual = signals - vectors[:, :, 0].sum(1)
    differences = vectors - vectors[:, :, :1]
    unary = differences.square().sum(-1) - 2 * (residual[:, None, None] * differences).sum(-1)
    energy = residual.square().sum(-1, keepdim=True).expand(-1, len(states)).clone()
    log_mass = torch.zeros_like(energy)
    state_atoms = []
    for d in range(depth):
        index = states[:, d]
        energy += unary[:, d, index]
        log_mass += log_base[:, d, index]
        state_atoms.append(atoms[:, d, index])
        for earlier in range(d):
            cross = 2 * torch.bmm(differences[:, earlier], differences[:, d].transpose(1, 2))
            energy += cross[:, states[:, earlier], index]
            if forbid_repeated_atoms:
                log_mass.masked_fill_(state_atoms[earlier] == state_atoms[d], -torch.inf)
    if not bool(torch.isfinite(energy).all()):
        raise ValueError("non-finite complete-vector energies")
    # Roundoff may make squared norms slightly negative; the clamp is only for
    # reporting. Distribution logits retain the quadratic calculation verbatim.
    logits = log_mass - energy / temperature
    normalizer = logits.logsumexp(-1, keepdim=True)
    if not bool(torch.isfinite(normalizer).all()):
        raise ValueError("a site has no finite-mass valid complete combination")
    log_joint = logits - normalizer
    joint = log_joint.exp()
    selected = torch.multinomial(joint, 1, generator=generator).squeeze(-1)
    selected_states = states[selected]
    sampled_atoms = atoms.gather(-1, selected_states[..., None]).squeeze(-1)
    sampled_ids = ids.gather(-1, selected_states[..., None]).squeeze(-1)
    sampled_values = values.gather(-1, selected_states[..., None]).squeeze(-1)
    reconstruction = (dictionary.T[sampled_atoms] * sampled_values[..., None]).sum(-2)
    posterior = log_joint
    atom_weights, coefficient_weights, atom_entropies, coefficient_entropies = [], [], [], []
    for d in range(depth):
        probability = posterior.softmax(-1)
        branch = torch.zeros(sites, choices, device=signals.device, dtype=signals.dtype)
        branch.scatter_add_(-1, states[:, d].expand(sites, -1), probability)
        atom_weights.append(branch)
        atom_entropies.append(_token_entropy(atoms[:, d], branch))
        # Atom conditioning and coefficient labels aggregate all branch aliases.
        atom_mask = atoms[:, d] == sampled_atoms[:, d, None]
        coefficients = branch * atom_mask
        coefficients = coefficients / coefficients.sum(-1, keepdim=True)
        coefficient_weights.append(coefficients)
        coefficient_entropies.append(_token_entropy(ids[:, d], coefficients))
        if d + 1 < depth:
            matching_pair = atom_mask & (ids[:, d] == sampled_ids[:, d, None])
            posterior = posterior.masked_fill(~matching_pair[:, states[:, d]], -torch.inf)
            posterior = posterior - posterior.logsumexp(-1, keepdim=True)
    finite_log_joint = torch.where(torch.isfinite(log_joint), log_joint, 0.)
    # Branch entropy equals sequence entropy after deduplication; under 'sum',
    # explicitly aggregate complete token-sequence aliases for a correct metric.
    if duplicate_policy == "sum":
        pair_keys = atoms * grids.shape[-1] + ids
        canonical = []
        for d in range(depth):
            same = pair_keys[:, d, :, None] == pair_keys[:, d, None, :]
            first = same.to(torch.int64).argmax(-1)
            canonical.append(first[:, states[:, d]])
        key = torch.zeros(sites, len(states), device=signals.device, dtype=torch.long)
        for canonical_d in canonical:
            key = key * choices + canonical_d
        joint_entropy = _token_entropy(key, joint)
    else:
        joint_entropy = -(joint * finite_log_joint).sum(-1)
    valid_energy = energy.masked_fill(~torch.isfinite(log_mass), torch.inf)
    return dict(atoms=sampled_atoms, coefficient_ids=sampled_ids,
                physical_coefficients=sampled_values,
                atom_target_ids=atoms, atom_target_weights=torch.stack(atom_weights, 1),
                coefficient_target_ids=ids,
                coefficient_target_weights=torch.stack(coefficient_weights, 1),
                reconstruction=reconstruction,
                expected_distortion=(joint * energy).sum(-1).clamp_min(0),
                sampled_distortion=(signals - reconstruction).square().sum(-1),
                minimum_distortion=valid_energy.amin(-1).clamp_min(0),
                joint_entropy=joint_entropy,
                atom_entropy=torch.stack(atom_entropies, 1),
                coefficient_entropy=torch.stack(coefficient_entropies, 1))


def cartesian_combination_cross_entropy(atom_logits, coefficient_logits, targets):
    """Equal-head loss plus per-site/depth CEs; coefficient head sees sampled atom.

    Returned scalar is (atom_CE + coefficient_CE).mean() / 2.  This common
    half-scale preserves the relative weighting of a joint likelihood. It does
    not apply a different atom weight. Repeated sparse IDs add their masses.
    """
    if (atom_logits.shape[:-1] != targets.atoms.shape
            or coefficient_logits.shape[:-1] != targets.atoms.shape):
        raise ValueError("logit leading shapes must match sampled target pairs")
    dtype = torch.float64 if atom_logits.dtype == torch.float64 else torch.float32
    atom_log_probability = atom_logits.to(dtype).log_softmax(-1).gather(
        -1, targets.atom_target_ids)
    coefficient_log_probability = coefficient_logits.to(dtype).log_softmax(-1).gather(
        -1, targets.coefficient_target_ids)
    atom_ce = -(targets.atom_target_weights.to(dtype) * atom_log_probability).sum(-1)
    coefficient_ce = -(targets.coefficient_target_weights.to(dtype)
                       * coefficient_log_probability).sum(-1)
    return (atom_ce + coefficient_ce).mean() / 2, atom_ce, coefficient_ce
