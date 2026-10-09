"""Measure bin changes from tiny pre-quantization noise on real OMP codes."""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.training.coefficient_jitter import jitter_probabilities, jitter_moments


def measure(atoms, coefficients, dictionary, scales, bins, sigma, cap):
    totals = torch.zeros(4, dtype=torch.float64)
    entropy = torch.zeros_like(totals)
    rounding = torch.zeros_like(totals)
    quantized_change = torch.zeros_like(totals)
    extra_energy = total_energy = clean_energy = 0.0
    width = float((bins[-1]-bins[0]) / (len(bins)-1))
    for start in range(0, len(atoms), 256):
        c = coefficients[start:start+256].double()
        d = dictionary[atoms[start:start+256]].double()
        nearest, p = jitter_probabilities(c, bins, sigma_bins=sigma, cap_bins=cap)
        p = p.double()
        centers = bins.double()
        q = centers[nearest]
        mean = (p*centers).sum(-1)
        var = (p*(centers-mean[..., None]).square()).sum(-1)
        totals += (1-p.gather(-1, nearest[..., None]).squeeze(-1)).sum(0)
        entropy += -(p*p.clamp_min(1e-30).log()).sum((0, 2))
        rounding += (q-c).square().sum(0)
        quantized_change += (var+(mean-q).square()).sum(0)
        clean = (d*(c*scales)[..., None]).sum(1)
        trace = (var*scales.square()*d.square().sum(-1)).sum(1)
        extra_bias = (d*((mean-q)*scales)[..., None]).sum(1)
        total_bias = (d*((mean-c)*scales)[..., None]).sum(1)
        extra_energy += float((trace+extra_bias.square().sum(1)).sum())
        total_energy += float((trace+total_bias.square().sum(1)).sum())
        clean_energy += float(clean.square().sum())
    return dict(sites=len(atoms), expected_bin_flip_fraction_per_depth=(totals/len(atoms)).tolist(),
        expected_bin_flip_fraction=float(totals.mean()/len(atoms)),
        target_entropy_nats_per_depth=(entropy/len(atoms)).tolist(),
        deterministic_rounding_rms_bins_per_depth=(rounding/len(atoms)).sqrt().div(width).tolist(),
        additional_quantized_change_rms_bins_per_depth=(quantized_change/len(atoms)).sqrt().div(width).tolist(),
        additional_quantized_latent_energy_fraction=extra_energy/clean_energy,
        total_quantized_latent_error_fraction=total_energy/clean_energy)


def calibrate(assets, output, samples=16384, sigma=0.01, cap=0.025):
    torch.set_num_threads(4)
    cache = torch.load(assets/'compound-cache.pt', mmap=True, map_location='cpu', weights_only=False)
    stage1 = torch.load(assets/'stage1-tokenizer.pt', mmap=True, map_location='cpu', weights_only=False)
    dictionary = F.normalize(stage1['state_dict']['quantizer.dictionary'].float(), dim=0).t()
    scales = torch.tensor(cache['meta']['coeff_scales']).double()
    bins = torch.linspace(-3., 3., 2048)
    width = float((bins[-1]-bins[0]).double()/(len(bins)-1))
    assert torch.allclose(bins.diff(), torch.full_like(bins[1:], width), atol=3e-7, rtol=0)
    rng = torch.Generator().manual_seed(261010)
    images = torch.randperm(len(cache['atoms']), generator=rng)[:2*samples]
    sites = torch.randint(64, (2*samples,), generator=rng)
    atoms = cache['atoms'][images, sites//8, sites%8].long()
    c = cache['coeffs'][images, sites//8, sites%8].float()
    assert len(images.unique()) == 2*samples
    selected = measure(atoms[:samples], c[:samples], dictionary, scales, bins, sigma, cap)
    verification = measure(atoms[samples:], c[samples:], dictionary, scales, bins, sigma, cap)
    for item in (selected, verification):
        assert max(item['expected_bin_flip_fraction_per_depth']) < 0.01
    # Independent simulation: add actual continuous truncated-Gaussian draws
    # and run nearest-center quantization, rather than sampling categorical p.
    test = c[:128].double()
    z = torch.randn((128, 4, 4096), generator=rng, dtype=torch.float64)
    a = cap/sigma
    while True:
        rejected = z.abs() > a
        count = int(rejected.sum())
        if count == 0:
            break
        z[rejected] = torch.randn(count, generator=rng, dtype=torch.float64)
    jitter = z*sigma*width
    assert float(jitter.abs().max()/width) <= cap
    midpoints = (bins.double()[:-1]+bins.double()[1:])/2
    sampled = torch.searchsorted(midpoints, (test[..., None]+jitter).contiguous())
    nearest, p = jitter_probabilities(test, bins, sigma_bins=sigma, cap_bins=cap)
    observed = float((sampled != nearest[..., None]).double().mean())
    expected = float((1-p.gather(-1, nearest[..., None]).squeeze(-1)).double().mean())
    assert abs(observed-expected) < 0.0003
    histogram = torch.zeros_like(p).double()
    histogram.scatter_add_(-1, sampled, torch.ones_like(sampled, dtype=torch.float64))
    histogram /= sampled.shape[-1]
    assert float((histogram-p).abs().max()) < 0.025
    moments = jitter_moments(sigma, cap)
    assert abs(float(jitter.square().mean().sqrt()/width)/moments['actual_rms_bins']-1) < 0.005
    report = dict(passed=True, seed=261010, calibration_sites=samples, verification_sites=samples,
        disjoint_training_images_verified=True, stage1_sha256=cache['meta']['stage1_checkpoint_sha256'],
        distribution='continuous Gaussian truncated before nearest-bin quantization; exact cell-integrated categorical probabilities',
        selected=dict(**moments, sigma_normalized=sigma*width, cap_normalized=cap*width,
            sigma_physical_per_depth=(scales*sigma*width).tolist(),
            cap_physical_per_depth=(scales*cap*width).tolist()),
        bin_width_normalized=width, bins=2048, normalized_range=[-3,3],
        calibration=selected, independent_verification=verification,
        empirical_check=dict(passed=True, draws=sampled.numel(),
            observed_bin_flip_fraction=observed, exact_bin_flip_fraction=expected,
            maximum_added_noise_bins=float(jitter.abs().max()/width),
            observed_noise_rms_bins=float(jitter.square().mean().sqrt()/width)),
        native_temperature_floor_bypassed=True,
        limits=['Nonzero jitter can change IDs for coefficients extremely close to a bin boundary.',
            'The 0.025-bin bound applies to added continuous noise, not ordinary quantization rounding.',
            'Calibration uses disjoint training images and does not establish optimal FID.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', type=Path, default=Path('/tmp/laser-imagenet-classcond-20261007/inputs'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = calibrate(args.assets, args.output)
    print(json.dumps({k: report[k] for k in ('passed', 'selected', 'independent_verification', 'empirical_check')}))
