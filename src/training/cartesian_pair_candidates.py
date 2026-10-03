"""No-refit pair proposals for a finite whole-combination teacher.

The proposal pool is not itself the teacher. Score its Cartesian product using
the complete reconstructed vector and derive samples/labels from that same law.
No least-squares solve, coefficient reoptimization, or prefix revision occurs.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class CartesianPairCandidates:
    atoms: torch.Tensor
    coefficient_ids: torch.Tensor
    anchor_vectors: torch.Tensor
    anchor_coefficient_ids: torch.Tensor
    neighbor_ranks: torch.Tensor
    bin_offsets: torch.Tensor


@torch.no_grad()
def dictionary_neighbor_ids(dictionary, *, count=32, chunk_size=256):
    """Return highest-positive-cosine nonself neighbors and their cosines.

    Build once for the frozen dictionary, preferably on an otherwise idle GPU.
    The returned rank list is deterministic apart from implementation-specific
    ordering of exact cosine ties. Store the list and its hash for exact replay.
    """
    if (dictionary.ndim != 2 or dictionary.dtype not in (torch.float32, torch.float64)
            or not 1 <= count < dictionary.shape[1] or chunk_size < 1
            or not torch.isfinite(dictionary).all()
            or not (dictionary.square().sum(0) > 0).all()):
        raise ValueError('Invalid dictionary or neighbor count/chunk size')
    with torch.autocast(device_type=dictionary.device.type, enabled=False):
        normalized = F.normalize(dictionary, dim=0)
        ids, similarities = [], []
        for start in range(0, dictionary.shape[1], chunk_size):
            stop = min(start + chunk_size, dictionary.shape[1])
            scores = normalized[:, start:stop].T @ normalized
            scores[torch.arange(stop-start, device=dictionary.device),
                   torch.arange(start, stop, device=dictionary.device)] = -torch.inf
            values, indices = scores.topk(count, dim=-1)
            if not (values > 0).all():
                raise ValueError('Dictionary does not have enough positive-cosine neighbors')
            ids.append(indices)
            similarities.append(values)
        return torch.cat(ids), torch.cat(similarities)


@torch.no_grad()
def nearest_physical_coefficient_ids(coefficients, coefficient_values):
    """Nearest actual grid entry at each depth; exact ties choose lower ID."""
    if (coefficients.ndim < 1 or coefficient_values.ndim != 2
            or coefficients.shape[-1] != coefficient_values.shape[0]
            or coefficient_values.shape[1] < 2
            or coefficients.device != coefficient_values.device
            or coefficients.dtype not in (torch.float32, torch.float64)
            or coefficient_values.dtype != coefficients.dtype
            or not torch.isfinite(coefficients).all()
            or not torch.isfinite(coefficient_values).all()
            or not (coefficient_values.diff(dim=-1) > 0).all()):
        raise ValueError('Invalid physical coefficients or increasing per-depth grids')
    values = coefficients.reshape(-1, coefficients.shape[-1]).T.contiguous()
    upper = torch.searchsorted(coefficient_values.contiguous(), values)
    upper = upper.clamp(max=coefficient_values.shape[-1]-1)
    lower = (upper-1).clamp_min(0)
    lower_distance = (coefficient_values.gather(1, lower)-values).abs()
    upper_distance = (coefficient_values.gather(1, upper)-values).abs()
    ids = torch.where(lower_distance <= upper_distance, lower, upper)
    return ids.T.reshape_as(coefficients)


@torch.no_grad()
def cartesian_pair_candidates(anchor_atoms, anchor_coefficients, dictionary,
                              coefficient_values, neighbor_ids, *,
                              max_bin_offset=32, generator=None):
    """Build six pair branches per depth without changing cached centers.

    Inputs:
      anchor_atoms / anchor_coefficients: [..., K], physical coefficients.
      dictionary: [channels, atoms], actual decoding vectors.
      coefficient_values: [K, bins], actual physical scalar grids.
      neighbor_ids: [atoms, R], positive-cosine nonself neighbors.
      max_bin_offset: positive integer scalar or K-vector, in bin-ID units.

    For each site/depth draw one neighbor rank uniformly and one radius
    uniformly in [1, max_bin_offset]. Use the same nearest cached center and
    symmetric bin radius for the anchor and alternative atom. Branch order is
    anchor(center, lower, upper), neighbor(center, lower, upper). All inputs
    remain unchanged. The anchor combination is always present exactly.

    Clamping at bin edges can duplicate branches. The combination teacher MUST
    deduplicate identical atom/coefficient pairs before assigning uniform local
    base mass (its default duplicate_policy='deduplicate' does this). Uniform
    mass refers to distinct local pairs conditional on this random pool. This
    is not importance sampling of a full-vocabulary global Gibbs law.

    anchor_vectors are a reconstruction baseline, NOT prequantized encoder z.
    Pass actual frozen encoder signals separately to the combination teacher.
    """
    integers = (torch.int16, torch.int32, torch.int64, torch.uint8)
    if (anchor_atoms.shape != anchor_coefficients.shape or anchor_atoms.ndim < 1
            or anchor_atoms.dtype not in integers or neighbor_ids.dtype not in integers
            or dictionary.ndim != 2 or neighbor_ids.ndim != 2
            or neighbor_ids.shape[0] != dictionary.shape[1] or neighbor_ids.shape[1] < 1
            or dictionary.dtype != anchor_coefficients.dtype
            or dictionary.dtype != coefficient_values.dtype
            or dictionary.dtype not in (torch.float32, torch.float64)
            or any(x.device != dictionary.device for x in
                   (anchor_atoms, anchor_coefficients, coefficient_values, neighbor_ids))
            or not torch.isfinite(dictionary).all()
            or (anchor_atoms < 0).any() or (anchor_atoms >= dictionary.shape[1]).any()
            or (neighbor_ids < 0).any() or (neighbor_ids >= dictionary.shape[1]).any()):
        raise ValueError('Invalid anchor pairs, dictionary, or neighbor table')
    depth = anchor_atoms.shape[-1]
    maximum = torch.as_tensor(max_bin_offset, device=dictionary.device)
    if (maximum.dtype not in integers or maximum.ndim > 1
            or maximum.numel() not in (1, depth) or not (maximum >= 1).all()):
        raise ValueError('max_bin_offset must be positive integer scalar or depth vector')
    with torch.autocast(device_type=dictionary.device.type, enabled=False):
        center = nearest_physical_coefficient_ids(anchor_coefficients, coefficient_values)
        ranks = torch.randint(neighbor_ids.shape[-1], anchor_atoms.shape,
                              generator=generator, device=dictionary.device)
        alternative = neighbor_ids[anchor_atoms.long(), ranks]
        if (alternative == anchor_atoms).any():
            raise ValueError('Neighbor table includes a sampled self neighbor')
        # Draw by inverse CDF to support a different radius range at each depth.
        offsets = (torch.rand(anchor_atoms.shape, generator=generator,
                              device=dictionary.device) * maximum).floor().long()+1
        bins = coefficient_values.shape[-1]
        local_ids = torch.stack((center, (center-offsets).clamp_min(0),
                                 (center+offsets).clamp_max(bins-1)), dim=-1)
        atoms = torch.stack((anchor_atoms.long(), alternative.long()), dim=-1)
        atoms = atoms[..., :, None].expand(*atoms.shape, 3).reshape(*anchor_atoms.shape, 6)
        ids = local_ids[..., None, :].expand(*local_ids.shape[:-1], 2, 3).reshape_as(atoms)
        physical = coefficient_values[torch.arange(depth, device=dictionary.device), center]
        anchor_vectors = (dictionary.T[anchor_atoms.long()] * physical[..., None]).sum(-2)
        return CartesianPairCandidates(atoms, ids, anchor_vectors, center, ranks, offsets)
