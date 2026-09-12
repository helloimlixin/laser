#!/usr/bin/env python3
"""Confirm bounded, magnitude-relative noise using the frozen Church decoder."""
import argparse
import codecs
import json
from pathlib import Path
import sys

import torch
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.church_ffhq_archived import full_training_cache
from src.church_relative_noise import RelativeChurchAux, relative_coefficient_distribution
from src.church_coefficient_noise import physical_coefficient_distribution
from src.models.lpips import LPIPS
from scripts.audit_church_noise_scale import expected_latent_error, summary
from scripts.tools.build_sign_probe_cache import sha256_file


@torch.no_grad()
def assess(aux, perceptual, data, indices, relative, seed, path):
    records = {k:[] for k in ('lpips', 'relative_latent_mse', 'psnr', 'expected_image_mse',
        'expected_site_mse', 'relative_coefficient_rms', 'coefficient_mse', 'sign_flip_probability', 'nearest_fallback')}
    uniforms = torch.rand(len(indices), 8, 8, 4, generator=torch.Generator().manual_seed(seed))
    violations = 0
    for first in range(0, len(indices), 16):
        selected = indices[first:first+16]
        atoms = data['atoms'][selected].cuda().long()
        c = data['coefficients'][selected].cuda()
        normalized = c/aux.coeff_scales
        vectors = aux.dictionary.t()[atoms]
        latent = (vectors*c[..., None]).sum(-2)
        clean = aux.decoder(aux.post_quant_conv(latent.permute(0, 3, 1, 2).contiguous())).clamp(-1, 1)
        if relative == 'fixed':
            probs = physical_coefficient_distribution(normalized, aux.coeff_bins, aux.coeff_scales, .1875)
        else:
            probs = relative_coefficient_distribution(normalized, aux.coeff_bins, aux.coeff_scales, .1875, relative, 3.)
        centers = aux.coeff_scales[:, None]*aux.coeff_bins
        # Use the same FP32 physical round trip as the production distribution
        # for the strict support bound; all distortion is measured vs cache c.
        target_c = normalized*aux.coeff_scales
        delta = centers-c[..., None]
        nearest = (centers-target_c[..., None]).abs().amin(-1)
        if relative != 'fixed':
            radius = (relative*target_c.abs()).clamp_max(.1875)*3
            bound = torch.maximum(radius, nearest)
            violations += int(((probs > 0) & ((centers-target_c[..., None]).abs() > bound[..., None])).sum())
        mean = (probs*delta).sum(-1)
        mse = (probs*delta.square()).sum(-1)
        gram = vectors @ vectors.transpose(-1, -2)
        expected = expected_latent_error(gram, mean, (mse-mean.square()).clamp_min(0))
        signal = latent.square().sum(-1)
        cumulative = probs.flatten(0, -2).cumsum(-1).contiguous()
        ids = torch.searchsorted(cumulative, uniforms[first:first+len(atoms)].cuda().reshape(-1, 1).contiguous()).clamp_max(2047).reshape_as(atoms)
        image = aux.decode_compound(atoms, ids)
        sampled_latent = aux.compound_embeddings(atoms, ids).sum(-2)
        rows = {'lpips':perceptual(image, clean).flatten(),
            'relative_latent_mse':(sampled_latent-latent).square().mean((1, 2, 3))/latent.square().mean((1, 2, 3)),
            'psnr':-10*((image-clean)/2).square().mean((1, 2, 3)).clamp_min(1e-12).log10(),
            'expected_image_mse':expected.mean((1, 2))/signal.mean((1, 2)),
            'expected_site_mse':expected/signal.clamp_min(1e-12),
            'relative_coefficient_rms':mse.sqrt()/c.abs().clamp_min(1e-12),
            'coefficient_mse':mse,
            'sign_flip_probability':(probs*((centers >= 0) != (c[..., None] >= 0))).sum(-1),
            'nearest_fallback':((probs > 0).sum(-1) == 1).float()}
        for k,v in rows.items(): records[k].append(v.cpu())
        if first == 0:
            save_image((torch.stack([clean[:8], image[:8]], 1).flatten(0, 1)+1)/2, path, nrow=2)
    r = {k:torch.cat(v) for k,v in records.items()}
    result = {k:float(r[k].mean()) for k in ('lpips', 'relative_latent_mse', 'psnr')}
    result['lpips_upper_2se'] = float(r['lpips'].mean()+2*r['lpips'].std()/len(indices)**.5)
    result['expected_image_mse'] = summary(r['expected_image_mse'])
    result['expected_site_mse'] = summary(r['expected_site_mse'])
    result['per_depth'] = [{'relative_noise_rms':summary(r['relative_coefficient_rms'][..., d]),
        'physical_noise_rms':float(r['coefficient_mse'][..., d].mean().sqrt()),
        'sign_flip_probability':float(r['sign_flip_probability'][..., d].mean()),
        'nearest_fallback_fraction':float(r['nearest_fallback'][..., d].mean())} for d in range(4)]
    result['support_bound_violations'] = violations if relative != 'fixed' else None
    result['passes'] = (relative != 'fixed' and violations == 0 and result['lpips_upper_2se'] <= .01
        and result['relative_latent_mse'] <= .005 and result['expected_image_mse']['mean'] <= .005
        and max(x['relative_noise_rms']['p99'] for x in result['per_depth']) <= .052)
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.serialization.add_safe_globals([codecs.encode])
    cache = ROOT/'outputs/church-ffhq-recipe-20260911/continuous-cache.pt'
    stage1 = ROOT/'outputs/lsun-church-sign-pattern-20260910/assets/best_rfid_slot1_model.pt'
    raw = torch.load(cache, map_location='cpu', weights_only=False)
    data, scales = full_training_cache(raw)
    assert sha256_file(stage1) == raw['meta']['checkpoint_sha256']
    aux = RelativeChurchAux(stage1, 16384, 2048, 3., coeff_scales=scales, sparsity_level=4,
        sigma_cap=.1875, relative_sigma=.05, truncate=3.).cuda().eval().requires_grad_(False)
    frozen = [(v, v._version) for v in (*aux.parameters(), *aux.buffers())]
    perceptual = LPIPS().cuda().eval().requires_grad_(False)
    indices = torch.randperm(len(data['train']['atoms']), generator=torch.Generator().manual_seed(63021))
    fit, confirm = indices[:512], indices[512:1024]
    candidates = [.025, .0375, .05]
    protocol = {'relative_sigma_candidates':candidates, 'sigma_cap':.1875, 'truncate':3.,
        'fit_indices':fit.tolist(), 'confirmation_indices':confirm.tolist(), 'fit_images':512, 'confirmation_images':512,
        'selection':'largest relative sigma passing fit and disjoint confirmation: LPIPS upper 2SE <= .01, sampled and expected mean relative latent MSE <= .005, p99 per-depth RMS/magnitude <= .052, zero hard-bound violations',
        'fixed_comparison': 'sigma=.1875 untruncated, coupled inverse-CDF draws, same images',
        'quantization_exception':'nearest-bin fallback when relative interval contains no bin; no positive sigma floor; all existing bins retained',
        'validation':'300 official images reported after selection only',
        'scope':'correct supports; frozen-tokenizer reconstruction distortion, not generation FID'}
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    results = {}
    for relative in ['fixed']+candidates:
        results[str(relative)] = assess(aux, perceptual, data['train'], fit, relative, 63022, args.output/f'fit-{relative}.png')
        print(json.dumps({'phase':'fit', 'relative':relative, **results[str(relative)]}), flush=True)
        (args.output/'fit-progress.json').write_text(json.dumps(results, indent=2)+'\n')
    confirmation, selected = {}, None
    for relative in reversed(candidates):
        if results[str(relative)]['passes']:
            row = assess(aux, perceptual, data['train'], confirm, relative, 63023, args.output/f'confirm-{relative}.png')
            confirmation[str(relative)] = row
            print(json.dumps({'phase':'confirmation', 'relative':relative, **row}), flush=True)
            if row['passes']:
                selected = relative
                break
    if selected is None: raise RuntimeError('No relative width passed calibration')
    validation = assess(aux, perceptual, data['validation'], torch.arange(300), selected, 63024, args.output/'validation.png')
    assert all(v._version == version and v.grad is None for v,version in frozen)
    result = {'selected_relative_sigma':selected, 'sigma_cap':.1875, 'truncate':3.,
        'coefficient_scales':scales, 'dictionary_size':16384, 'sparsity':4, 'coefficient_bins':2048,
        'checkpoint_sha256':sha256_file(stage1), 'cache_sha256':sha256_file(cache),
        'target_code_sha256':sha256_file(ROOT/'src/church_relative_noise.py'),
        'calibration_code_sha256':sha256_file(Path(__file__)), 'protocol':protocol, 'fit':results,
        'confirmation':confirmation, 'validation':validation, 'frozen_tokenizer_verified':True}
    (args.output/'calibration.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'selected_relative_sigma':selected, 'validation':validation}), flush=True)


if __name__ == '__main__': main()
