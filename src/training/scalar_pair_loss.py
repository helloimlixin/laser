"""Balanced field losses for interleaved atom and offset coefficient tokens."""
import torch


def scalar_pair_objective(atom_logits, coefficient_logits, atoms, probabilities,
                          coefficient_bins, *, atom_weight=1., atom_depth_weights=None,
                          coefficient_depth_weights=None, crps_weight=0., accumulation=1,
                          target_atom_ids=None, target_atom_weights=None,
                          geometry_weight=0., target_physical=None,
                          geometry_dictionary=None, geometry_coeff_scales=None,
                          geometry_top_k=4, geometry_estimator="candidate_conditional",
                          geometry_candidate_coeff_logits=None):
    # Reuse the field likelihood math, not the compound-token representation.
    from src.training.rqtransformer import compound_objective
    span = coefficient_bins[-1] - coefficient_bins[0]
    loss, details = compound_objective(
        atom_logits, coefficient_logits, None, atoms, probabilities, target_physical,
        atom_weight=atom_weight, atom_depth_weights=atom_depth_weights,
        coefficient_depth_weights=coefficient_depth_weights,
        geometry_weight=geometry_weight, accumulation=accumulation,
        distribution_geometry=geometry_weight > 0,
        geometry_dictionary=geometry_dictionary, geometry_coeff_bins=coefficient_bins,
        geometry_coeff_scales=geometry_coeff_scales, geometry_top_k=geometry_top_k,
        geometry_estimator=geometry_estimator,
        geometry_candidate_coeff_logits=geometry_candidate_coeff_logits,
        coefficient_bins=coefficient_bins,
        coeff_crps_weight=crps_weight / float(span) if crps_weight else 0.,
        target_atom_ids=target_atom_ids, target_atom_weights=target_atom_weights,
    )
    # Preserve the existing scalar path's dimensionless ordered-loss metric.
    details['coefficient_crps'] = details['coefficient_crps'] / span
    return loss, details


@torch.no_grad()
def scalar_pair_metrics(atom_logits, coefficient_logits, atoms, probabilities,
                        coefficient_bins, coefficient_scales, details, *, atom_weight=1., geometry_weight=0.):
    atom_ce, coeff_ce = details['atom_cross_entropy'], details['coeff_cross_entropy']
    centers = coefficient_bins.float()
    prediction = (coefficient_logits.float().softmax(-1) * centers).sum(-1)
    target = (probabilities.float() * centers).sum(-1)
    error = (prediction - target).abs()
    correct = (atom_logits.argmax(-1) == atoms).float()
    entropy = -(probabilities.float() * probabilities.float().clamp_min(1e-30).log()).sum(-1)
    metrics = {
        'train/atom_nll': float(details['atom_nll'].mean()),
        'train/atom_soft_cross_entropy': float(atom_ce.mean()),
        'train/coeff_cross_entropy': float(coeff_ce.mean()),
        'train/coeff_target_entropy': float(entropy.mean()),
        'train/coeff_kl': float((coeff_ce - entropy).mean()),
        'train/atom_top1': float(correct.mean()),
        'train/coeff_expected_mae': float(error.mean()),
        'train/coeff_expected_physical_mae': float((error * coefficient_scales).mean()),
        'train/coeff_crps': float(details['coefficient_crps']),
        'train/classification_loss': float(details['classification']),
        'train/geometry_loss': float(details['geometry']),
        'train/geometry_pair_mse': float(details['geometry_pair_mse']),
        'train/geometry_spatial_mse': float(details['geometry_spatial_mse']),
        'train/geometry_weight': geometry_weight,
        'train/weighted_geometry_loss': geometry_weight*float(details['geometry']),
        'train/image_loss': float(details['classification'])+geometry_weight*float(details['geometry']),
        'train/atom_loss_weight': atom_weight,
        'train/weighted_atom_ce': float(details['depth_weighted_atom_cross_entropy'].mean()) * atom_weight/(atom_weight+1.),
        'train/weighted_coefficient_ce': float(details['depth_weighted_coeff_cross_entropy'].mean())/(atom_weight+1.),
    }
    for depth in range(atoms.shape[-1]):
        metrics.update({
            f'train/atom_nll_depth{depth}': float(details['atom_nll'][..., depth].mean()),
            f'train/atom_soft_cross_entropy_depth{depth}': float(atom_ce[..., depth].mean()),
            f'train/coeff_cross_entropy_depth{depth}': float(coeff_ce[..., depth].mean()),
            f'train/atom_top1_depth{depth}': float(correct[..., depth].mean()),
            f'train/coeff_expected_mae_depth{depth}': float(error[..., depth].mean()),
            f'train/coeff_expected_physical_mae_depth{depth}': float(error[..., depth].mean() * coefficient_scales[depth]),
        })
    return metrics
