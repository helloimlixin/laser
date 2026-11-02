# Exact historical objective from W&B run ffhqcmp0804205803.
import torch
from torch.nn import functional as F

def compound_objective(
    atom_logits,
    coeff_logits,
    physical_prediction,
    target_atoms,
    target_coeff_probs,
    target_physical,
    *,
    atom_weight: float,
    geometry_weight: float,
    accumulation: int,
    distribution_geometry: bool = False,
    geometry_dictionary=None,
    geometry_coeff_bins=None,
    geometry_coeff_scales=None,
    geometry_top_k: int = 4,
):
    """Weighted token objective plus normalized physical latent geometry.

    V3 learns geometry through an auxiliary contribution head.  V4 instead
    constructs the physical contribution from the atom/coeff distributions
    used at sampling time, so the auxiliary objective cannot be solved by an
    inference-time-unused shortcut.
    """
    atom_log_probs = F.log_softmax(atom_logits.float(), dim=-1)
    coeff_log_probs = F.log_softmax(coeff_logits.float(), dim=-1)
    atom_nll = -atom_log_probs.gather(
        -1, target_atoms.long().unsqueeze(-1)
    ).squeeze(-1)
    coeff_cross_entropy = -(target_coeff_probs * coeff_log_probs).sum(dim=-1)
    depth = target_atoms.shape[-1]
    classification = (
        atom_weight * atom_nll.sum(dim=-1) + coeff_cross_entropy.sum(dim=-1)
    ).mean() / ((atom_weight + 1.0) * depth)

    geometry = atom_logits.new_zeros((), dtype=torch.float32)
    pair_mse = atom_logits.new_zeros((), dtype=torch.float32)
    spatial_mse = atom_logits.new_zeros((), dtype=torch.float32)
    if geometry_weight > 0:
        if target_physical is None:
            raise ValueError("physical contribution target required for geometry loss")
        if distribution_geometry:
            if geometry_dictionary is None or geometry_coeff_bins is None or geometry_coeff_scales is None:
                raise ValueError("distribution geometry requires dictionary, bins, and depth scales")
            candidate_count = min(max(int(geometry_top_k), 1), atom_logits.shape[-1])
            candidate_logits, candidate_atoms = atom_logits.float().topk(candidate_count, dim=-1)
            target_atom_logits = atom_logits.float().gather(
                -1, target_atoms.long().unsqueeze(-1)
            )
            target_is_candidate = (candidate_atoms == target_atoms.long().unsqueeze(-1)).any(
                dim=-1, keepdim=True
            )
            target_atom_logits = target_atom_logits.masked_fill(
                target_is_candidate, -float("inf")
            )
            candidate_logits = torch.cat((candidate_logits, target_atom_logits), dim=-1)
            candidate_atoms = torch.cat(
                (candidate_atoms, target_atoms.long().unsqueeze(-1)), dim=-1
            )
            candidate_weights = candidate_logits.softmax(dim=-1)
            dictionary = geometry_dictionary.float().t()
            candidate_vectors = dictionary[candidate_atoms]
            expected_atom = (
                candidate_weights.unsqueeze(-1) * candidate_vectors
            ).sum(dim=-2)
            coeff_bins = geometry_coeff_bins.float()
            expected_coeff = (coeff_log_probs.exp() * coeff_bins).sum(dim=-1)
            scales = geometry_coeff_scales.float().view(
                *([1] * (expected_coeff.ndim - 1)), -1
            )
            prediction = expected_atom * (expected_coeff * scales).unsqueeze(-1)
        else:
            if physical_prediction is None:
                raise ValueError("auxiliary-head geometry requires physical prediction")
            prediction = physical_prediction.float()
        target = target_physical.float()
        pair_mse = F.mse_loss(prediction, target)
        predicted_spatial = prediction.sum(dim=-2)
        target_spatial = target.sum(dim=-2)
        spatial_mse = F.mse_loss(predicted_spatial, target_spatial)
        pair_energy = target.square().mean().detach().clamp_min(1e-6)
        spatial_energy = target_spatial.square().mean().detach().clamp_min(1e-6)
        geometry = 0.5 * (pair_mse / pair_energy + spatial_mse / spatial_energy)

    total = (classification + geometry_weight * geometry) / accumulation
    return total, {
        "atom_nll": atom_nll,
        "coeff_cross_entropy": coeff_cross_entropy,
        "classification": classification,
        "geometry": geometry,
        "geometry_pair_mse": pair_mse,
        "geometry_spatial_mse": spatial_mse,
    }
