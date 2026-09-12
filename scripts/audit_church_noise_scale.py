#!/usr/bin/env python3
"""Audit existing coefficient noise against ranges, active supports and sparsity.

Read-only with respect to training. Uses exact discrete target moments and the
actual frozen dictionary; no stage-2 model, GPU, or new calibration selection.
"""
import argparse
import codecs
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.church_coefficient_noise import physical_coefficient_distribution
from src.church_ffhq_archived import full_training_cache
from scripts.tools.build_sign_probe_cache import sha256_file


def expected_latent_error(gram, error_mean, error_variance):
    """E||D_S delta||² for conditionally independent coefficient draws.

    Includes nonzero means from discrete binning and finite-range truncation.
    Correlated draws would require their full covariance, not just variances.
    """
    return ((gram.diagonal(dim1=-2, dim2=-1) * error_variance).sum(-1)
            + torch.einsum('...i,...ij,...j->...', error_mean, gram, error_mean))


def summary(values):
    values = values.double().flatten()
    q = torch.quantile(values, torch.tensor([.01, .05, .5, .95, .99], dtype=torch.float64))
    return dict(zip(('p01', 'p05', 'median', 'p95', 'p99'), q.tolist()),
                mean=float(values.mean()), maximum=float(values.max()))


@torch.no_grad()
def audit_distribution(dictionary, atoms, coefficients, scales, sigma):
    bins = torch.linspace(-3, 3, 2048)
    centers = scales[:, None] * bins
    records = {name: [] for name in ('error_energy', 'signal_energy', 'coefficient_mse',
        'flip_probability', 'absolute_error_over_magnitude', 'entropy', 'boundary_mass')}
    for first in range(0, len(atoms), 8):
        c = coefficients[first:first+8]
        a = atoms[first:first+8].long()
        normalized = c / scales
        if sigma == 'legacy':
            probs = (-(normalized[..., None]-bins).square()/.5).softmax(-1)
        else:
            probs = physical_coefficient_distribution(normalized, bins, scales, sigma)
        delta = centers - c[..., None]
        mean = (probs*delta).sum(-1)
        mse = (probs*delta.square()).sum(-1)
        variance = (mse-mean.square()).clamp_min(0)
        vectors = dictionary.t()[a]
        gram = vectors @ vectors.transpose(-1, -2)
        error = expected_latent_error(gram, mean, variance)
        signal = (vectors*c[..., None]).sum(-2).square().sum(-1)
        records['error_energy'].append(error)
        records['signal_energy'].append(signal)
        records['coefficient_mse'].append(mse)
        records['flip_probability'].append((probs*((centers >= 0) != (c[..., None] >= 0))).sum(-1))
        # Zero magnitudes have undefined relative error; report their count
        # separately, and use only nonzero entries for this diagnostic.
        records['absolute_error_over_magnitude'].append(mse.sqrt()/c.abs().clamp_min(1e-12))
        records['entropy'].append(-(probs*probs.clamp_min(1e-30).log()).sum(-1))
        records['boundary_mass'].append(probs[..., 0]+probs[..., -1])
    r = {k:torch.cat(v) for k,v in records.items()}
    image_ratio = r['error_energy'].mean((1, 2))/r['signal_energy'].mean((1, 2))
    site_ratio = r['error_energy']/r['signal_energy'].clamp_min(1e-12)
    return {
        'expected_image_relative_latent_mse': summary(image_ratio),
        'expected_site_relative_latent_mse': summary(site_ratio),
        'site_fraction_expected_relative_mse_above_0.005': float((site_ratio > .005).float().mean()),
        'aggregate_latent_rms_error_over_signal_rms': float((r['error_energy'].sum()/r['signal_energy'].sum()).sqrt()),
        'passes_existing_mean_latent_mse_limit': float(image_ratio.mean()) <= .005,
        'per_depth': [{
            'realized_noise_rms': float(r['coefficient_mse'][..., d].mean().sqrt()),
            'noise_rms_over_coefficient_rms': float((r['coefficient_mse'][..., d].mean()/coefficients[..., d].square().mean()).sqrt()),
            'relative_noise_for_nonzero_coefficients': summary(r['absolute_error_over_magnitude'][..., d][coefficients[..., d] != 0]),
            'probability_of_sign_flip': float(r['flip_probability'][..., d].mean()),
            'mean_target_entropy_nats': float(r['entropy'][..., d].mean()),
            'mean_probability_at_endpoint_bins': float(r['boundary_mass'][..., d].mean()),
        } for d in range(atoms.shape[-1])],
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, default=ROOT/'outputs/church-ffhq-noise-20260911/calibration/calibration.json')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.serialization.add_safe_globals([codecs.encode])
    cache = ROOT/'outputs/church-ffhq-recipe-20260911/continuous-cache.pt'
    checkpoint = ROOT/'outputs/lsun-church-sign-pattern-20260910/assets/best_rfid_slot1_model.pt'
    config_path = ROOT/'outputs/church-ffhq-recipe-20260910/ffhq-success-files/config.yaml'
    calibration = json.loads(args.calibration.read_text())
    source_hashes = {'cache_sha256':sha256_file(cache), 'checkpoint_sha256':sha256_file(checkpoint),
                     'target_code_sha256':sha256_file(ROOT/'src/church_coefficient_noise.py')}
    assert all(calibration[k] == value for k,value in source_hashes.items())
    data, scales_list = full_training_cache(torch.load(cache, map_location='cpu', weights_only=False))
    scales = torch.tensor(scales_list)
    assert scales_list == calibration['coefficient_scales']
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    dictionary = F.normalize(payload['state_dict']['quantizer.dictionary'].float(), dim=0)
    del payload
    assert dictionary.shape == (256, 16384)
    train = data['train']
    depth = train['atoms'].shape[-1]
    assert depth == 4
    sigma = calibration['selected_sigma']
    # Fresh training-image audit, disjoint from both selection and confirmation.
    excluded = set(calibration['protocol']['fit_indices']+calibration['protocol']['confirmation_indices'])
    indices = torch.tensor([i for i in torch.randperm(len(train['atoms']), generator=torch.Generator().manual_seed(62011)).tolist()
                            if i not in excluded][:1024])
    assert len(indices) == 1024 and not set(indices.tolist()) & excluded
    protocol = {'images': len(indices), 'indices': indices.tolist(), 'seed':62011,
        'population':'training images disjoint from earlier noise selection and confirmation',
        'mode':'exact moments of full untruncated production discrete targets; no Monte Carlo or nucleus filtering',
        'selection':'none; audit the existing selected sigma and original normalized T=0.5',
        'limits':'reuse mean per-image relative latent MSE <= 0.005; tails are diagnostics, not a pre-existing acceptance criterion',
        'scope':'correct fixed supports; does not establish tolerance to predicted wrong support or generation quality'}
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    flat = train['coefficients'].flatten(0, -2)
    per_depth = []
    for d in range(depth):
        coeff = flat[:, d]
        counts = torch.bincount(train['atoms'][..., d].flatten().long(), minlength=dictionary.shape[1]).double()
        probabilities = counts/counts.sum()
        entropy = -(probabilities*probabilities.clamp_min(1e-30).log()).sum()
        spacing = 6*float(scales[d])/2047
        per_depth.append({'depth':d+1, 'signed_min':float(coeff.min()), 'signed_max':float(coeff.max()),
            'absolute_coefficient':summary(coeff.abs()), 'coefficient_rms':float(coeff.square().mean().sqrt()),
            'zero_fraction':float((coeff == 0).float().mean()),
            'fraction_magnitude_below_sigma':float((coeff.abs() < sigma).float().mean()),
            'fraction_magnitude_below_3sigma':float((coeff.abs() < 3*sigma).float().mean()),
            'physical_bin_spacing':spacing, 'sigma_in_bins':sigma/spacing,
            'configured_range':[-3*float(scales[d]), 3*float(scales[d])],
            'outside_configured_range_fraction':float((coeff.abs() > 3*scales[d]).float().mean()),
            'used_atoms':int((counts > 0).sum()), 'marginal_atom_entropy_nats':float(entropy),
            'marginal_effective_atom_count':float(entropy.exp())})
        print(json.dumps({'phase':'full_training_coefficients', **per_depth[-1]}), flush=True)
    atoms = train['atoms'][indices].long()
    coeff = train['coefficients'][indices]
    # Small active-support Gram matrices account for the actual selected atoms.
    gram_parts = []
    for first in range(0, len(atoms), 16):
        vectors = dictionary.t()[atoms[first:first+16]]
        gram_parts.append(vectors @ vectors.transpose(-1, -2))
    gram = torch.cat(gram_parts)
    eig = torch.linalg.eigvalsh(gram)
    off_diagonal = gram[..., ~torch.eye(depth, dtype=torch.bool)].abs()
    anchors = torch.randperm(dictionary.shape[1], generator=torch.Generator().manual_seed(62012))[:256]
    correlations = dictionary[:, anchors].t() @ dictionary
    correlations[torch.arange(len(anchors)), anchors] = 0
    norms = dictionary.square().sum(0).sqrt()
    geometry = {'dictionary_shape':list(dictionary.shape), 'unit_norm_min':float(norms.min()), 'unit_norm_max':float(norms.max()),
        'active_support_abs_inner_product':summary(off_diagonal),
        'active_support_gram_largest_eigenvalue':summary(eig[..., -1]),
        'active_support_gram_smallest_eigenvalue':summary(eig[..., 0]),
        'nearest_abs_correlation_for_256_sampled_atoms':summary(correlations.abs().amax(-1)),
        'duplicate_support_site_fraction':float((atoms.sort(-1).values.diff(dim=-1) == 0).any(-1).float().mean()),
        'independent_zero_mean_unit_atom_noise_energy':depth*sigma**2,
        'independent_zero_mean_unit_atom_noise_norm_rms':depth**.5*sigma,
        'same_absolute_noise_energy_sigma_if_depth_doubles':sigma/2**.5,
        'note':'last scaling holds fixed absolute error budget, unit atom norms and independent zero-mean errors; real finite-bin moments used below'}
    print(json.dumps({'phase':'dictionary_geometry', **geometry}), flush=True)
    distributions = {}
    for name in ('legacy', sigma):
        distributions[str(name)] = audit_distribution(dictionary, atoms, coeff, scales, name)
        print(json.dumps({'phase':'exact_distribution_audit', 'sigma':name, **distributions[str(name)]}), flush=True)
    ffhq = {k:v['value'] for k,v in yaml.safe_load(config_path.read_text()).items() if isinstance(v, dict) and 'value' in v}
    ffhq_scales = ffhq['coeff_scales']
    comparison = {'ffhq':{'dictionary_size':ffhq['num_atoms'], 'active_coefficients':len(ffhq_scales),
        'coefficient_bins':ffhq['coeff_vocab_size'], 'scales':ffhq_scales,
        'configured_physical_ranges':[[-3*s, 3*s] for s in ffhq_scales],
        'original_nominal_physical_sigma':[.5*s for s in ffhq_scales],
        'geometry_and_coefficient_population':'not available locally; config-only comparison, no measured FFHQ noise-to-signal claim'},
        'church':{'dictionary_size':dictionary.shape[1], 'active_coefficients':depth, 'coefficient_bins':2048,
        'scales':scales_list, 'selected_physical_sigma':sigma}}
    result = {'protocol':protocol, **source_hashes, 'calibration_sha256':sha256_file(args.calibration),
        'audit_code_sha256':sha256_file(Path(__file__)), 'ffhq_config_sha256':sha256_file(config_path),
        'comparison':comparison, 'full_training_images':len(train['atoms']), 'per_depth':per_depth,
        'geometry':geometry, 'distributions':distributions,
        'existing_decoder_confirmation':calibration['confirmation'][str(sigma)],
        'training_modified':False}
    (args.output/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    assert distributions[str(sigma)]['passes_existing_mean_latent_mse_limit']


if __name__ == '__main__':
    main()
