#!/usr/bin/env python3
"""Fit one-token COMPLETE sparse-code vocabularies and measure decoder distortion."""
import argparse
import codecs
import json
from pathlib import Path
import sys
import time

import torch
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_official_rqtransformer_laser_stage2 import LaserAux, atomic_torch_save
from scripts.tools.build_sign_probe_cache import sha256_file
from src.coefficient_pattern_codec import assign_coefficient_patterns
from src.complete_sparse_codec import (
    pack_exact, unpack_exact, sparse_latents, decode_site_ids, fit_complete_codebook,
)
from src.models.lpips import LPIPS


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache', type=Path, default=ROOT / 'outputs/church-ffhq-recipe-20260911/continuous-cache.pt')
    p.add_argument('--stage1', type=Path, default=ROOT / 'outputs/lsun-church-sign-pattern-20260910/assets/best_rfid_slot1_model.pt')
    p.add_argument('--sizes', type=int, nargs='+', default=[4096, 16384])
    p.add_argument('--fit-sites', type=int, default=262144)
    p.add_argument('--iterations', type=int, default=8)
    p.add_argument('--assignment-batch', type=int, default=1024)
    p.add_argument('--image-batch', type=int, default=8)
    p.add_argument('--calibration-images', type=int, default=128)
    p.add_argument('--holdout-images', type=int, default=256)
    p.add_argument('--validation-images', type=int, default=300)
    p.add_argument('--seed', type=int, default=9701)
    args = p.parse_args()
    if min(args.sizes + [args.fit_sites, args.iterations, args.assignment_batch, args.image_batch,
                         args.calibration_images, args.holdout_images, args.validation_images]) < 1:
        raise ValueError('Sizes and iteration counts must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    torch.cuda.set_per_process_memory_fraction(.25)
    torch.serialization.add_safe_globals([codecs.encode])
    start = time.monotonic()

    def log(row):
        row = {'elapsed_seconds': time.monotonic() - start, **row}
        print(json.dumps(row, allow_nan=False), flush=True)
        with (args.output / 'history.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')

    raw = torch.load(args.cache, weights_only=False, map_location='cpu')
    checkpoint_hash = sha256_file(args.stage1)
    assert checkpoint_hash == raw['meta']['checkpoint_sha256']
    keys = {s: set(raw[s]['keys']) for s in ('train', 'holdout', 'validation')}
    assert not keys['train'] & keys['holdout'] and not keys['train'] & keys['validation']
    assert not keys['holdout'] & keys['validation']
    train_order = torch.randperm(len(raw['train']['atoms']), generator=torch.Generator().manual_seed(args.seed))
    calibration_indices = train_order[:args.calibration_images]
    fit_images = train_order[args.calibration_images:]
    count = min(args.fit_sites, len(fit_images) * 64)
    if max(args.sizes) > count:
        raise ValueError('Each codebook must fit within the fitting population')
    selected = torch.randperm(len(fit_images) * 64, generator=torch.Generator().manual_seed(args.seed + 1))[:count]
    fit_indices = fit_images[selected // 64] * 64 + selected % 64
    assert not torch.isin(fit_indices // 64, calibration_indices).any()

    aux = LaserAux(args.stage1, 16384, 2048, 3., coeff_scales=raw['meta']['coeff_scales'],
                   sparsity_level=4, soft_target_physical=True, clamp_coeffs=False).cuda().eval()
    perceptual = LPIPS().cuda().eval()
    dictionary, bins, scales = aux.dictionary.t(), aux.coeff_bins, aux.coeff_scales
    boundaries = (bins[:-1] + bins[1:]) / 2

    def quantize(physical):
        return torch.bucketize((physical / scales).contiguous(), boundaries)

    fit_atoms = raw['train']['atoms'].reshape(-1, 4)[fit_indices].cuda().long()
    fit_coefficients = raw['train']['coefficients'].reshape(-1, 4)[fit_indices].cuda()
    fit_coefficient_ids = quantize(fit_coefficients)
    del fit_coefficients
    probe_atoms, probe_coefficients = fit_atoms[:1024].cpu(), fit_coefficient_ids[:1024].cpu()
    exact_values = [pack_exact(a.tolist(), c.tolist()) for a, c in zip(probe_atoms, probe_coefficients)]
    for value, a, c in zip(exact_values, probe_atoms, probe_coefficients):
        assert unpack_exact(value) == (a.tolist(), c.tolist())
    manifest = {'format': 'church_complete_sparse_site_v1', 'scope': 'all four atom IDs AND four signed coefficient IDs',
        'checkpoint_sha256': checkpoint_hash, 'cache_sha256': sha256_file(args.cache),
        'source_quantization': {'atoms': 16384, 'coefficient_bins': 2048, 'depth': 4, 'bits_per_site': 100,
            'nominal_packed_integer_states': str(2 ** 100), 'exact_round_trip_sites_checked': len(exact_values)},
        'fit_split': 'train_only', 'fit_sites': count, 'calibration_train_images': calibration_indices.tolist(),
        'calibration_excluded_from_fit': True, 'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'selection': 'Smallest vocabulary with calibration LPIPS mean + 2 SE <= .01 and relative latent MSE <= .005',
        'reference': 'continuous frozen LASER reconstruction; oracle compression, not unconditional generation',
        'candidates': {}}
    write_json(args.output / 'results.json', manifest)
    atomic_torch_save({'fit_site_indices': fit_indices, 'calibration_image_indices': calibration_indices}, args.output / 'fit-indices.pt')
    log({'phase': 'ready', 'fit_sites': count, 'codebook_sizes': args.sizes})

    def decode(z):
        return aux.decoder(aux.post_quant_conv(z.permute(0, 3, 1, 2).contiguous())).clamp(-1, 1)

    evaluations = [('calibration', 'train', calibration_indices),
                   ('holdout', 'holdout', torch.randperm(len(raw['holdout']['atoms']), generator=torch.Generator().manual_seed(args.seed + 2))[:args.holdout_images]),
                   ('validation', 'validation', torch.arange(min(args.validation_images, len(raw['validation']['atoms']))))]

    for size in args.sizes:
        output = args.output / f'vocab-{size}'
        output.mkdir()
        log({'phase': 'fit_start', 'vocabulary': size})
        book = fit_complete_codebook(fit_atoms, fit_coefficient_ids, dictionary, bins, scales,
            num_codes=size, iterations=args.iterations, seed=args.seed, chunk_size=args.assignment_batch,
            progress=lambda iteration, error, empty: log({'phase': 'lloyd', 'vocabulary': size,
                'iteration': iteration, 'mean_squared_latent_error': error, 'empty_clusters': empty}))
        saved = {key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in book.items()}
        saved.update({'format': manifest['format'], 'checkpoint_sha256': checkpoint_hash,
                      'coefficient_bins': bins.cpu(), 'coefficient_scales': scales.cpu(),
                      'fit_site_indices': fit_indices[book['fit_representative_indices'].cpu()]})
        saved['atoms'], saved['coefficient_ids'] = saved['atoms'].short(), saved['coefficient_ids'].short()
        atomic_torch_save(saved, output / 'codebook.pt')
        candidate = {'bits_per_token': (size - 1).bit_length(), 'prototype': 'nearest assigned observed training code to each Lloyd center',
                     'unique_representatives': int(torch.unique(book['fit_representative_indices']).numel()),
                     'empty_clusters': book['empty_clusters']}
        for label, split, indices in evaluations:
            records, all_ids = {}, []
            for first in range(0, len(indices), args.image_batch):
                selected = indices[first:first + args.image_batch]
                atoms = raw[split]['atoms'][selected].cuda().long()
                coefficients = raw[split]['coefficients'][selected].cuda()
                true_latents = (dictionary[atoms] * coefficients[..., None]).sum(-2)
                target = decode(true_latents)
                ids, _ = assign_coefficient_patterns(true_latents.flatten(0, -2), book['latents'], chunk_size=args.assignment_batch)
                ids = ids.reshape(atoms.shape[:-1])
                recovered_atoms, recovered_coefficients = decode_site_ids(ids, book)
                complete_z = sparse_latents(recovered_atoms, recovered_coefficients, dictionary, bins, scales)
                # This assertion verifies that ID lookup alone fully reconstructs each site.
                torch.testing.assert_close(complete_z, book['latents'][ids], atol=1e-6, rtol=1e-6)
                hard_ids = quantize(coefficients)
                nearest_z = sparse_latents(atoms, hard_ids, dictionary, bins, scales)
                dense_ids, _ = assign_coefficient_patterns(true_latents.flatten(0, -2), book['dense_centers'], chunk_size=args.assignment_batch)
                dense_z = book['dense_centers'][dense_ids].reshape_as(true_latents)
                grid = [target[:8]]
                for name, z in [('nearest_scalar', nearest_z), ('complete_sparse', complete_z), ('dense_center_diagnostic', dense_z)]:
                    image = decode(z)
                    grid.append(image[:8])
                    values = {'lpips': perceptual(image, target).flatten(),
                              'psnr': -10 * ((image - target) / 2).square().mean((1, 2, 3)).clamp_min(1e-12).log10(),
                              'relative_latent_mse': (z - true_latents).square().mean((1, 2, 3)) / true_latents.square().mean((1, 2, 3))}
                    records.setdefault(name, []).append({k: v.cpu() for k, v in values.items()})
                all_ids.append(ids.cpu().to(torch.int32))
                if first == 0:
                    save_image((torch.stack(grid, 1).flatten(0, 1) + 1) / 2,
                               output / f'{label}-reconstructions.png', nrow=len(grid))
            per_image = {name: {k: torch.cat([row[k] for row in rows]) for k in rows[0]} for name, rows in records.items()}
            token_ids = torch.cat(all_ids)
            assert tuple(token_ids.shape[1:]) == (8, 8)
            counts = torch.bincount(token_ids.flatten().long(), minlength=size).double()
            probabilities = counts[counts > 0] / counts.sum()
            metrics = {name: {k: float(v.mean()) for k, v in rows.items()} for name, rows in per_image.items()}
            for name, rows in per_image.items():
                metrics[name]['lpips_upper_2se'] = float(rows['lpips'].mean() + 2 * rows['lpips'].std() / len(indices) ** .5)
            metrics['site_tokens'] = {'used_vocabulary': int((counts > 0).sum()),
                                     'marginal_entropy_bits': float(-(probabilities * probabilities.log2()).sum()),
                                     'images': len(indices), 'tokens_per_image': 64}
            candidate[label] = metrics
            atomic_torch_save({'metrics': per_image, 'image_indices': indices, 'site_ids': token_ids}, output / f'{label}.pt')
            write_json(output / f'{label}.json', metrics)
            log({'phase': 'evaluation', 'vocabulary': size, 'split': label, 'metrics': metrics})
        score = candidate['calibration']['complete_sparse']
        candidate['passes_calibration'] = score['lpips_upper_2se'] <= .01 and score['relative_latent_mse'] <= .005
        manifest['candidates'][str(size)] = candidate
        write_json(args.output / 'results.json', manifest)
        del book, saved
        torch.cuda.empty_cache()
    passed = [int(k) for k, v in manifest['candidates'].items() if v['passes_calibration']]
    manifest['selected_vocabulary'] = min(passed) if passed else None
    manifest['phase'] = 'complete'
    manifest['unconditional_quality_tested'] = False
    write_json(args.output / 'results.json', manifest)
    log({'phase': 'complete', 'selected_vocabulary': manifest['selected_vocabulary']})


if __name__ == '__main__':
    main()
