"""Measure and match stochastic residual-quantization noise in physical units.

Quantizers provide scores (negative squared distance, up to a row constant),
embed_depth, and depth. All measurements follow sampled causal prefixes.
"""
import math

import numpy as np
import torch


def align_noise_budgets(reference_energies, reference_budget, target_energies):
    """Transfer a cumulative budget along normalized hard-error reduction.

    Log residual-error reduction locates coarse-to-fine progress independently
    of latent units and depth count. This is a calibration choice, not an
    assertion that matching corruption optimizes a learned prior.
    """
    def progress(values):
        values = np.asarray(values, dtype=np.float64)
        if (values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all()
                or (values <= 0).any() or (np.diff(values) >= 0).any()):
            raise ValueError('Hard residual energies must decrease and remain positive')
        return np.log(values[0] / values) / np.log(values[0] / values[-1])

    ref = progress(reference_energies)
    target = progress(target_energies)
    budget = np.asarray(reference_budget, dtype=np.float64)
    if len(budget) != len(ref)-1 or (budget <= 0).any() or not np.isfinite(budget).all():
        raise ValueError('One positive noise budget is required per reference depth')
    cumulative = np.interp(target, ref, np.r_[0., budget.cumsum()])
    transferred = np.diff(cumulative)
    return dict(reference_progress=ref.tolist(), target_progress=target.tolist(),
                reference_cumulative_budget=np.r_[0., budget.cumsum()].tolist(),
                target_cumulative_budget=cumulative.tolist(), budgets=transferred.tolist())


@torch.no_grad()
def fit_gap_temperature(gaps, target_extra_error, iterations=22):
    """Fit mean extra squared distance for a fixed set of residuals."""
    if not math.isfinite(target_extra_error) or target_extra_error <= 0:
        raise ValueError('Expected extra error must be positive and finite')
    if (not torch.isfinite(gaps).all() or (gaps < 0).any()
            or gaps.ndim != 2 or not (gaps.min(-1).values == 0).all()):
        raise ValueError('Distance gaps must be finite, nonnegative, with a zero per row')
    def excess(temperature):
        return float(((-gaps / temperature).softmax(-1) * gaps).sum() / len(gaps))

    low, high = 1.e-8, 1.
    while excess(high) < target_extra_error and high < 1.e8:
        high *= 2
    if not excess(low) <= target_extra_error <= excess(high):
        raise ValueError('Extra-error target is outside the attainable range')
    for _ in range(iterations):
        midpoint = (low+high)/2
        if excess(midpoint) < target_extra_error:
            low = midpoint
        else:
            high = midpoint
    result = (low+high)/2
    return result, excess(result)


@torch.no_grad()
def hard_trajectory(quantizer, latent, chunk_size=512):
    residual = latent.reshape(-1, latent.shape[-1]).float().clone()
    codes = torch.empty(len(residual), quantizer.depth, dtype=torch.long, device=residual.device)
    energies = [float(residual.square().sum(-1).mean())]
    for depth in range(quantizer.depth):
        for start in range(0, len(residual), chunk_size):
            r = residual[start:start+chunk_size]
            chosen = quantizer.scores(r, depth).argmax(-1)
            codes[start:start+len(r), depth] = chosen
            r.sub_(quantizer.embed_depth(chosen, depth))
        energies.append(float(residual.square().sum(-1).mean()))
    return dict(energies=energies, codes=codes.reshape(*latent.shape[:-1], quantizer.depth).cpu())


@torch.no_grad()
def measure_noise(quantizer, latent, temperatures, seeds, hard=None, chunk_size=512):
    """Exact conditional expected excess; Monte Carlo cumulative latent error."""
    if len(temperatures) != quantizer.depth or any(t <= 0 for t in temperatures):
        raise ValueError('One positive temperature is required per depth')
    hard = hard or hard_trajectory(quantizer, latent, chunk_size)
    x = latent.reshape(-1, latent.shape[-1]).float()
    denominator = hard['energies'][-1]
    traces, sampled_codes, final_ratios = [], [], []
    for seed in seeds:
        generator = torch.Generator(device=x.device).manual_seed(seed)
        residual = x.clone()
        codes = torch.empty(len(x), quantizer.depth, dtype=torch.long, device=x.device)
        rows = []
        for depth, temperature in enumerate(temperatures):
            total = torch.zeros(4, dtype=torch.float64, device=x.device)
            for start in range(0, len(x), chunk_size):
                r = residual[start:start+chunk_size]
                scores = quantizer.scores(r, depth)
                gap = scores.max(-1).values[:, None]-scores
                probability = (scores / temperature).softmax(-1)
                chosen = torch.multinomial(probability, 1, generator=generator).squeeze(-1)
                total += torch.stack(((probability*gap).sum(),
                    -(probability*probability.clamp_min(1.e-30).log()).sum(),
                    probability.max(-1).values.sum(),
                    (probability > 1.e-6).float().sum())).double()
                codes[start:start+len(r), depth] = chosen
                r.sub_(quantizer.embed_depth(chosen, depth))
            total = total.cpu().tolist()
            energy = float(residual.square().sum(-1).mean())
            rows.append(dict(depth=depth+1, temperature=temperature,
                extra_squared_error_per_site=total[0]/len(x),
                extra_error_over_hard_final_error=total[0]/len(x)/denominator,
                mean_entropy=total[1]/len(x), mean_maximum_probability=total[2]/len(x),
                mean_support_above_1e6=total[3]/len(x), residual_energy=energy,
                residual_energy_over_hard=energy/hard['energies'][depth+1]))
        traces.append(rows)
        final_ratios.append(rows[-1]['residual_energy']/denominator)
        sampled_codes.append(codes.reshape(*latent.shape[:-1], quantizer.depth).cpu())
    profiles = []
    for depth in range(quantizer.depth):
        profiles.append({key: (depth+1 if key == 'depth' else float(np.mean([r[depth][key] for r in traces])))
                        for key in traces[0][depth]})
    return dict(temperatures=list(temperatures), seeds=list(seeds), images=len(latent),
        hard_residual_energies=hard['energies'], profiles=profiles,
        final_sampled_to_hard_latent_error_ratio=float(np.mean(final_ratios)),
        ratio_by_seed=final_ratios, ratio_seed_std=float(np.std(final_ratios, ddof=1)) if len(seeds)>1 else 0.,
        sum_expected_extra_error_over_hard_final_error=sum(r['extra_error_over_hard_final_error'] for r in profiles),
        hard_codes=hard['codes'], sampled_codes=sampled_codes)


@torch.no_grad()
def fit_depth_temperatures(quantizer, latent, budgets, hard_final_error, seed):
    """Sequential fit on sampled prefixes; retain each full paired codeword."""
    if len(budgets) != quantizer.depth:
        raise ValueError('Budget count must match residual depth')
    residual = latent.reshape(-1, latent.shape[-1]).float().clone()
    generator = torch.Generator(device=residual.device).manual_seed(seed)
    temperatures, trace = [], []
    for depth, budget in enumerate(budgets):
        scores = quantizer.scores(residual, depth)
        gaps = scores.max(-1).values[:, None]-scores
        del scores
        temperature, actual = fit_gap_temperature(gaps, budget*hard_final_error)
        probability = (-gaps / temperature).softmax(-1)
        chosen = torch.multinomial(probability, 1, generator=generator).squeeze(-1)
        residual.sub_(quantizer.embed_depth(chosen, depth))
        temperatures.append(temperature)
        trace.append(dict(depth=depth+1, budget=budget,
                          measured_extra_error_over_hard_final_error=actual/hard_final_error))
        del gaps, probability
    return temperatures, trace


def public_profile(profile):
    return {key:value for key, value in profile.items() if key not in ('hard_codes', 'sampled_codes')}
