"""Measure coefficient prediction accuracy in bin units, separately from CE.

Errors use the original continuous coefficients, including quantizer overflow.
The returned sums can be accumulated over microbatches and reduced across DDP
ranks before reporting. These diagnostics do not modify the training loss.
"""
import torch
from torch.nn import functional as F


@torch.no_grad()
def coefficient_diagnostic_sums(logits, coefficients, bins, targets):
    width = (bins[-1] - bins[0]).float() / (bins.numel() - 1)
    coefficients = coefficients.detach().float()
    log_probs = F.log_softmax(logits.detach().float(), dim=-1)
    mode = bins[logits.detach().argmax(dim=-1)].float()
    mean = (log_probs.exp() * bins.float()).sum(dim=-1)
    mode_error = (mode - coefficients).abs() / width
    mean_error = (mean - coefficients).abs() / width
    nearest = (coefficients[..., None] - bins).abs().argmin(dim=-1)
    rounding_error = (bins[nearest] - coefficients).abs() / width
    outside = (coefficients < bins[0]) | (coefficients > bins[-1])
    ce = -(targets.float() * log_probs).sum(dim=-1)
    entropy = -(targets.float() * targets.float().clamp_min(1e-30).log()).sum(-1)
    return torch.stack((mode_error.new_tensor(mode_error.numel()),
        mode_error.sum(), mean_error.sum(), mode_error.square().sum(),
        (mode_error < 1).float().sum(), rounding_error.sum(),
        outside.float().sum(), ce.sum(), entropy.sum(),
        *mode_error.reshape(-1, mode_error.shape[-1]).sum(0).unbind())).double()


def coefficient_diagnostic_metrics(sums, bins, scales):
    values = sums.detach().cpu().tolist()
    count, mode, mean, squared, within, rounding, outside, ce, entropy, *depths = values
    width = float((bins[-1] - bins[0]) / (bins.numel() - 1))
    metrics = dict(coeff_mode_mae_bins=mode/count, coeff_mean_mae_bins=mean/count,
        coeff_mode_rmse_bins=(squared/count)**.5,
        coeff_within_one_bin_fraction=within/count,
        coeff_quantization_mae_bins=rounding/count, coeff_out_of_range_fraction=outside/count,
        coeff_cross_entropy_nats=ce/count, coeff_target_entropy_nats=entropy/count,
        coeff_mode_mae_below_one_bin=mode/count < 1., coeff_bin_width=width)
    for depth, (error, scale) in enumerate(zip(depths, scales)):
        mae = error / (count / len(depths))
        metrics[f'coeff_mode_mae_bins_depth{depth}'] = mae
        metrics[f'coeff_mode_physical_mae_depth{depth}'] = mae * width * float(scale)
    return {f'train/{name}': value for name, value in metrics.items()}
