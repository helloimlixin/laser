"""Calibrate K4 coefficient targets below the historical K2 bin-unit noise."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F


def target_moments(coefficients, bins, temperature, hard_max_bins=None):
    delta = coefficients[..., None] - bins
    logits = -delta.square() / temperature
    if hard_max_bins is not None:
        width = float((bins[-1] - bins[0]) / (len(bins) - 1))
        allowed = delta.abs() < width * hard_max_bins
        if not allowed.any(-1).all():
            raise ValueError('A coefficient has no bin within the requested noise bound')
        logits = logits.masked_fill(~allowed, -torch.inf)
    probabilities = logits.softmax(-1)
    mean = (probabilities * bins).sum(-1)
    variance = (probabilities * (bins - mean[..., None]).square()).sum(-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)
    return probabilities, mean, variance, entropy


def measure(atoms, coefficients, dictionary, scales, bins, temperature, hard_max_bins=None):
    energy = clean_energy = 0.0
    squared_delta = torch.zeros(4, dtype=torch.float64)
    clean_coefficients = torch.zeros(4, dtype=torch.float64)
    entropies = torch.zeros(4, dtype=torch.float64)
    maximum_supported = maximum_conditional_rms = 0.0
    nominal_width = float((bins[-1] - bins[0]) / (len(bins) - 1))
    for start in range(0, len(atoms), 128):
        c = coefficients[start:start + 128]
        vectors = dictionary[atoms[start:start + 128]]
        probabilities, mean, variance, entropy = target_moments(c, bins, temperature, hard_max_bins)
        clean = (vectors * (c * scales)[..., None]).sum(1)
        bias = (vectors * ((mean - c) * scales)[..., None]).sum(1)
        expected = (variance * scales.square() * vectors.square().sum(-1)).sum(1)
        expected = expected + bias.square().sum(1)
        energy += float(expected.double().sum())
        clean_energy += float(clean.square().double().sum())
        squared_delta += (variance + (mean - c).square()).double().sum(0)
        clean_coefficients += (c * scales).double().square().sum(0)
        entropies += entropy.double().sum(0)
        if hard_max_bins is not None:
            displacement = (c[..., None] - bins).abs() / nominal_width
            maximum_supported = max(maximum_supported, float(displacement.masked_fill(probabilities == 0, 0).max()))
            maximum_conditional_rms = max(maximum_conditional_rms,
                float((variance + (mean-c).square()).sqrt().max()) / nominal_width)
    normalized_rms = (squared_delta / len(atoms)).sqrt()
    result = dict(temperature=temperature, target_space='normalized',
        expected_added_latent_energy_fraction=energy / clean_energy,
        noise_rms_bins_per_depth=(normalized_rms / float(bins[1] - bins[0])).tolist(),
        noise_rms_physical_per_depth=(normalized_rms * scales.double()).tolist(),
        noise_relative_to_coefficient_rms_per_depth=(
            normalized_rms * scales.double() / (clean_coefficients / len(atoms)).sqrt()).tolist(),
        target_entropy_nats_per_depth=(entropies / len(atoms)).tolist())
    if hard_max_bins is not None:
        result.update(hard_max_noise_bins=hard_max_bins,
            maximum_supported_error_bins=maximum_supported,
            maximum_conditional_rms_error_bins=maximum_conditional_rms)
    return result


def calibrate_subbin(assets, output, samples=16384):
    torch.set_num_threads(4)
    cache = torch.load(assets / 'compound-cache.pt', mmap=True, map_location='cpu', weights_only=False)
    stage1 = torch.load(assets / 'stage1-tokenizer.pt', mmap=True, map_location='cpu', weights_only=False)
    dictionary = F.normalize(stage1['state_dict']['quantizer.dictionary'].float(), dim=0).t()
    scales = torch.tensor(cache['meta']['coeff_scales'])
    bins = torch.linspace(-3., 3., 2048)
    width = float((bins[-1] - bins[0]) / (len(bins) - 1))
    temperature = 2 * (0.5 * width) ** 2
    cap = 0.999
    minimum = cache['coeffs'].amin((0, 1, 2))
    maximum = cache['coeffs'].amax((0, 1, 2))
    assert torch.isfinite(minimum).all() and torch.isfinite(maximum).all()
    assert minimum.min() >= -3 - 1e-6 and maximum.max() <= 3 + 1e-6
    rng = torch.Generator().manual_seed(261009)
    images = torch.randperm(len(cache['atoms']), generator=rng)[:2 * samples]
    sites = torch.randint(64, (2 * samples,), generator=rng)
    atoms = cache['atoms'][images, sites // 8, sites % 8].long()
    coefficients = cache['coeffs'][images, sites // 8, sites % 8].float()
    assert len(images.unique()) == 2 * samples
    selected = measure(atoms[:samples], coefficients[:samples], dictionary,
                       scales, bins, temperature, cap)
    verification = measure(atoms[samples:], coefficients[samples:], dictionary,
                           scales, bins, temperature, cap)
    for result in (selected, verification):
        assert result['maximum_supported_error_bins'] < cap
        assert result['maximum_conditional_rms_error_bins'] <= 0.501
        assert max(result['noise_rms_bins_per_depth']) < 0.5
    c, d = coefficients[:16], dictionary[atoms[:16]]
    p, _, _, _ = target_moments(c, bins, temperature, cap)
    ids = torch.multinomial(p.reshape(-1, 2048), 4096, replacement=True,
                           generator=rng).reshape(16, 4, 4096)
    delta = (bins[ids] - c[..., None]) * scales[None, :, None]
    assert float(((bins[ids] - c[..., None]).abs() / width).max()) < cap
    perturbation = torch.einsum('ndm,ndc->nmc', delta, d)
    clean = (d * (c * scales)[..., None]).sum(1)
    observed = float(perturbation.square().sum(-1).mean()) / float(clean.square().sum(-1).mean())
    exact = measure(atoms[:16], c, dictionary, scales, bins,
                    temperature, cap)['expected_added_latent_energy_fraction']
    assert abs(observed / exact - 1) < 0.04
    report = dict(passed=True, seed=261009, calibration_sites=samples,
        verification_sites=samples, disjoint_training_images_verified=True,
        stage1_sha256=cache['meta']['stage1_checkpoint_sha256'], coeff_scales=scales.tolist(),
        noise_distribution='Gaussian categorical probabilities truncated to less than one bin from clean OMP coefficients',
        hard_max_noise_bins=cap, bin_width_normalized=width,
        historical_k2=dict(source='helloimlixin-rutgers/laser/swgbasnb',
            temperature=0.5, bins=2048, normalized_range=[-20,20],
            interior_noise_sd_bins=0.5/(40/2047), interior_noise_sd_physical=3.2),
        selection=dict(untruncated_gaussian_sd_bins=0.5, strict_supported_error_bins=cap,
            maximum_measured_rms_bins=0.5), selected=selected,
        independent_verification=verification,
        full_cache_coefficient_range=dict(minimum=minimum.tolist(), maximum=maximum.tolist(),
            images=len(cache['atoms'])),
        categorical_sampling_check=dict(passed=True, draws_per_site=4096,
            observed_fraction=observed, exact_fraction=exact, all_draws_strictly_subbin=True),
        limits=['Calibration and verification use disjoint training images, not held-out ImageNet.',
                'The hard bound applies to training perturbations around known clean coefficients, not generated coefficient predictions.',
                'This calibration does not establish the optimal FID.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    return report


def calibrate(assets, output, samples=16384):
    torch.set_num_threads(4)
    cache = torch.load(assets / 'compound-cache.pt', mmap=True, map_location='cpu', weights_only=False)
    stage1 = torch.load(assets / 'stage1-tokenizer.pt', mmap=True, map_location='cpu', weights_only=False)
    dictionary = F.normalize(stage1['state_dict']['quantizer.dictionary'].float(), dim=0).t()
    scales = torch.tensor(cache['meta']['coeff_scales'])
    bins = torch.linspace(-3., 3., 2048)
    rng = torch.Generator().manual_seed(261008)
    images = torch.randperm(len(cache['atoms']), generator=rng)[:2 * samples]
    sites = torch.randint(64, (2 * samples,), generator=rng)
    atoms = cache['atoms'][images, sites // 8, sites % 8].long()
    coefficients = cache['coeffs'][images, sites // 8, sites % 8].float()
    assert len(images.unique()) == 2 * samples
    # Historical K2 p(c) is proportional to exp(-(c-center)^2/0.5),
    # with 2048 centers in [-20,20]. Its interior Gaussian SD is 0.5.
    k2_bin_sd = 0.5 / (40. / 2047)
    bin_matched_temperature = 0.5 * (6. / 40.) ** 2
    candidates = [bin_matched_temperature / factor ** 2 for factor in (2., 4., 8.)]
    measured = [measure(atoms[:samples], coefficients[:samples], dictionary,
                        scales, bins, temperature) for temperature in candidates]
    eligible = [x for x in measured
        if max(x['noise_rms_bins_per_depth']) <= k2_bin_sd / 4 * 1.001
        and x['expected_added_latent_energy_fraction'] <= 0.0005]
    assert eligible, 'No candidate meets both the bin-unit and latent-energy limits'
    selected = max(eligible, key=lambda x: x['temperature'])
    verification = measure(atoms[samples:], coefficients[samples:], dictionary,
                          scales, bins, selected['temperature'])
    assert verification['expected_added_latent_energy_fraction'] <= 0.0005
    assert max(verification['noise_rms_bins_per_depth']) <= k2_bin_sd / 4 * 1.001
    baseline = measure(atoms[samples:], coefficients[samples:], dictionary, scales, bins, 0.5)
    # Independently check the moment calculation with categorical samples.
    c, d = coefficients[:16], dictionary[atoms[:16]]
    p, _, _, _ = target_moments(c, bins, selected['temperature'])
    ids = torch.multinomial(p.reshape(-1, 2048), 2048, replacement=True,
                           generator=rng).reshape(16, 4, 2048)
    delta = (bins[ids] - c[..., None]) * scales[None, :, None]
    noisy_latent = torch.einsum('ndm,ndc->nmc', delta, d)
    clean = (d * (c * scales)[..., None]).sum(1)
    observed = float(noisy_latent.square().sum(-1).mean()) / float(clean.square().sum(-1).mean())
    exact = measure(atoms[:16], c, dictionary, scales, bins,
                    selected['temperature'])['expected_added_latent_energy_fraction']
    assert abs(observed / exact - 1) < 0.04
    report = dict(passed=True, seed=261008, calibration_sites=samples,
        verification_sites=samples, disjoint_training_images_verified=True,
        stage1_sha256=cache['meta']['stage1_checkpoint_sha256'], coeff_scales=scales.tolist(),
        historical_k2=dict(source='helloimlixin-rutgers/laser/swgbasnb',
            temperature=0.5, bins=2048, normalized_range=[-20, 20],
            interior_noise_sd_bins=k2_bin_sd, interior_noise_sd_physical=3.2),
        selection=dict(max_latent_noise_energy_fraction=0.0005,
            maximum_bin_noise_fraction_of_k2=0.25, rule='Highest candidate temperature meeting both limits'),
        candidates=measured, selected=selected, independent_verification=verification,
        original_setting_verification=baseline,
        categorical_sampling_check=dict(passed=True, draws_per_site=2048,
            observed_fraction=observed, exact_fraction=exact),
        limits=['Calibration and verification use disjoint training images, not held-out ImageNet.',
                'Historical K2 reference is coefficient noise in bin units, not its measured latent-energy ratio.',
                'These measurements do not select the optimal FID temperature.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', type=Path, default=Path('/tmp/laser-imagenet-classcond-20261007/inputs'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sub-bin', action='store_true')
    args = parser.parse_args()
    result = (calibrate_subbin if args.sub_bin else calibrate)(args.assets, args.output)
    print(json.dumps({key: result[key] for key in ('passed', 'selected', 'independent_verification')}), flush=True)
