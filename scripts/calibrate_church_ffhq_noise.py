#!/usr/bin/env python3
"""Select a small physical noise width using frozen Church reconstructions."""
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
from src.church_coefficient_noise import CalibratedChurchAux, physical_coefficient_distribution
from src.ffhq_v4_archived import LaserAux as ArchiveAux
from src.models.lpips import LPIPS
from src.models.rqtransformer.transformers import _top_p_probs
from scripts.tools.build_sign_probe_cache import sha256_file


@torch.no_grad()
def assess(aux, perceptual, data, indices, sigma, seed, path, top_p=1.):
    records = {k: [] for k in ('lpips', 'psnr', 'relative_latent_mse', 'coefficient_mae', 'sign_flip_fraction')}
    error_squares = torch.zeros(4, dtype=torch.float64)
    values = 0
    uniforms = torch.rand(len(indices), 8, 8, 4, generator=torch.Generator().manual_seed(seed))
    for first in range(0, len(indices), 16):
        selected = indices[first:first+16]
        atoms = data['atoms'][selected].cuda().long()
        physical = data['coefficients'][selected].cuda()
        normalized = physical / aux.coeff_scales
        latent = (aux.dictionary.t()[atoms] * physical[..., None]).sum(-2)
        clean = aux.decoder(aux.post_quant_conv(latent.permute(0, 3, 1, 2).contiguous())).clamp(-1, 1)
        if sigma == 'nearest':
            ids = ((normalized+3) * (2047/6)).round().long().clamp(0, 2047)
        else:
            if sigma == 'legacy':
                _, probabilities = ArchiveAux.compound_coeff_ids(aux, normalized, stochastic=False, temp=.5)
            else:
                probabilities = physical_coefficient_distribution(normalized, aux.coeff_bins, aux.coeff_scales, sigma)
            probabilities = probabilities.flatten(0, -2)
            if top_p < 1:
                probabilities = _top_p_probs(probabilities, top_p)
            cumulative = (probabilities / probabilities.sum(-1, keepdim=True)).cumsum(-1).contiguous()
            ids = torch.searchsorted(cumulative, uniforms[first:first+len(atoms)].cuda().reshape(-1, 1).contiguous())
            ids = ids.clamp_max(2047).reshape_as(atoms)
        coefficients = aux.coeff_bins[ids] * aux.coeff_scales
        image = aux.decode_compound(atoms, ids)
        delta = coefficients-physical
        error_squares += delta.double().square().sum((0, 1, 2)).cpu()
        values += len(atoms)*64
        image_mse = ((image-clean)/2).square().mean((1, 2, 3))
        metrics = {'lpips': perceptual(image, clean).flatten(),
            'psnr': -10*image_mse.clamp_min(1e-12).log10(),
            'relative_latent_mse': (aux.compound_embeddings(atoms, ids).sum(-2)-latent).square().mean((1, 2, 3)) / latent.square().mean((1, 2, 3)),
            'coefficient_mae': delta.abs().mean((1, 2, 3)),
            'sign_flip_fraction': ((coefficients >= 0) != (physical >= 0)).float().mean((1, 2, 3))}
        for k, value in metrics.items():
            records[k].append(value.cpu())
        if first == 0:
            save_image((torch.stack([clean[:8], image[:8]], 1).flatten(0, 1)+1)/2, path, nrow=2)
    records = {k:torch.cat(v) for k, v in records.items()}
    result = {k:float(v.mean()) for k, v in records.items()}
    result['lpips_upper_2se'] = float(records['lpips'].mean()+2*records['lpips'].std()/len(indices)**.5)
    result['realized_noise_rms_per_depth'] = (error_squares/values).sqrt().tolist()
    result['passes'] = result['lpips_upper_2se'] <= .01 and result['relative_latent_mse'] <= .005
    return result


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.serialization.add_safe_globals([codecs.encode])
    cache = ROOT/'outputs/church-ffhq-recipe-20260911/continuous-cache.pt'
    stage1 = ROOT/'outputs/lsun-church-sign-pattern-20260910/assets/best_rfid_slot1_model.pt'
    raw = torch.load(cache, weights_only=False, map_location='cpu')
    data, scales = full_training_cache(raw)
    assert sha256_file(stage1) == raw['meta']['checkpoint_sha256']
    aux = CalibratedChurchAux(stage1, 16384, 2048, 3., coeff_scales=scales,
        sparsity_level=4, coefficient_sigma=.125).cuda().eval().requires_grad_(False)
    frozen = [(v, v._version) for v in (*aux.parameters(), *aux.buffers())]
    perceptual = LPIPS().cuda().eval().requires_grad_(False)
    indices = torch.randperm(len(data['train']['atoms']), generator=torch.Generator().manual_seed(61021))
    fit, confirm = indices[:512], indices[512:1024]
    candidates = [.0625, .125, .1875, .25, .5]
    protocol = {'candidates_physical_sigma': candidates, 'fit_images': 512, 'confirmation_images': 512,
        'fit_indices': fit.tolist(), 'confirmation_indices': confirm.tolist(), 'index_seed': 61021,
        'selection': 'largest sigma passing fit and disjoint training-image confirmation; LPIPS mean + 2SE <= .01 and relative latent MSE <= .005',
        'validation': '300 official validation images, reporting only after selection',
        'sampling': 'one coupled inverse-CDF draw per coefficient; full training distribution for selection; p=.85 additionally checked after selection',
        'reference': 'clean continuous frozen-tokenizer reconstruction; not generation FID'}
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    results = {}
    for sigma in ['nearest', 'legacy'] + candidates:
        row = assess(aux, perceptual, data['train'], fit, sigma, 61022, args.output/f'fit-{sigma}.png')
        results[str(sigma)] = row
        print(json.dumps({'phase': 'fit', 'sigma': sigma, **row}), flush=True)
        (args.output/'fit-progress.json').write_text(json.dumps(results, indent=2)+'\n')
    confirmation = {}
    selected = None
    for sigma in reversed(candidates):
        if results[str(sigma)]['passes']:
            row = assess(aux, perceptual, data['train'], confirm, sigma, 61023, args.output/f'confirm-{sigma}.png')
            confirmation[str(sigma)] = row
            print(json.dumps({'phase': 'confirmation', 'sigma': sigma, **row}), flush=True)
            if row['passes']:
                selected = sigma
                break
    if selected is None:
        raise RuntimeError('No candidate passed both training-image checks')
    validation = {}
    for top_p in [1., .85]:
        validation[str(top_p)] = assess(aux, perceptual, data['validation'], torch.arange(300),
            selected, 61024, args.output/f'validation-p{top_p}.png', top_p=top_p)
        print(json.dumps({'phase': 'validation', 'sigma': selected, 'top_p': top_p, **validation[str(top_p)]}), flush=True)
    assert all(v._version == version and v.grad is None for v, version in frozen)
    rms = data['train']['coefficients'].square().mean((0, 1, 2)).sqrt()
    result = {'selected_sigma': selected, 'physical_temperature': 2*selected**2,
        'normalized_temperatures_per_depth': (2*(selected/torch.tensor(scales)).square()).tolist(),
        'coefficient_scales': scales, 'physical_coefficient_rms': rms.tolist(),
        'noise_std_over_coefficient_rms': (selected/rms).tolist(),
        'checkpoint_sha256': sha256_file(stage1), 'cache_sha256': sha256_file(cache),
        'target_code_sha256': sha256_file(ROOT/'src/church_coefficient_noise.py'),
        'protocol': protocol, 'fit': results, 'confirmation': confirmation,
        'validation': validation, 'frozen_tokenizer_verified': True}
    (args.output/'calibration.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'selected_sigma': selected, 'physical_temperature': 2*selected**2}), flush=True)


if __name__ == '__main__':
    main()
