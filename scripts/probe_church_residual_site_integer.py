#!/usr/bin/env python3
"""Measure complete sparse-site distortion at 27, 45, and 63 packed bits.

Fits bounded train-only Lloyd codebooks; no neural or stage-1 training. Also
tests exact support plus one 128-entry signed coefficient pattern (63 bits).
Every sparse reconstruction is recovered from the packed integer alone.
"""
import argparse
import codecs
import json
import os
from pathlib import Path
import shutil
import sys
import time

import torch
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_official_rqtransformer_laser_stage2 import LaserAux, atomic_torch_save
from scripts.tools.build_sign_probe_cache import sha256_file
from src.coefficient_pattern_codec import assign_coefficient_patterns, fit_coefficient_patterns, selected_support_grams
from src.complete_sparse_codec import sparse_latents
from src.learned_sparse_site_codec import SparseSiteProjector
from src.models.lpips import LPIPS
from src.residual_site_integer_codec import (
    pack_fields, unpack_fields, fit_residual_codebooks, encode_residual_fields, decode_residual_ids,
)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache', type=Path, default=ROOT / 'outputs/church-ffhq-recipe-20260911/continuous-cache.pt')
    p.add_argument('--stage1', type=Path, default=ROOT / 'outputs/lsun-church-sign-pattern-20260910/assets/best_rfid_slot1_model.pt')
    p.add_argument('--split-indices', type=Path, default=ROOT / 'outputs/church-complete-site-token-20260911/fit-indices.pt')
    p.add_argument('--fit-sites', type=int, default=262144)
    p.add_argument('--iterations', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=8)
    args = p.parse_args()
    if min(args.fit_sites, args.iterations, args.batch_size) < 1:
        raise ValueError('Counts must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.cuda.set_per_process_memory_fraction(.3)
    torch.serialization.add_safe_globals([codecs.encode])
    started = time.monotonic()

    def log(row):
        row = {'elapsed_seconds': time.monotonic() - started, **row}
        print(json.dumps(row, allow_nan=False), flush=True)
        with (args.output / 'history.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')

    def save_result():
        (args.output / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')

    raw = torch.load(args.cache, weights_only=False, map_location='cpu')
    split = torch.load(args.split_indices, weights_only=True, map_location='cpu')
    calibration = split['calibration_image_indices']
    fit_indices = split['fit_site_indices'][:args.fit_sites]
    assert not torch.isin(fit_indices // 64, calibration).any()
    keys = {s: set(raw[s]['keys']) for s in ('train', 'holdout', 'validation')}
    assert not keys['train'] & keys['holdout'] and not keys['train'] & keys['validation']
    assert not keys['holdout'] & keys['validation']
    checkpoint_hash = sha256_file(args.stage1)
    assert checkpoint_hash == raw['meta']['checkpoint_sha256']
    aux = LaserAux(args.stage1, 16384, 2048, 3., coeff_scales=raw['meta']['coeff_scales'],
                   sparsity_level=4, soft_target_physical=True, clamp_coeffs=False).cuda().eval().requires_grad_(False)
    perceptual = LPIPS().cuda().eval().requires_grad_(False)
    frozen = [(value, value._version) for module in (aux, perceptual) for value in (*module.parameters(), *module.buffers())]
    dictionary, bins, scales = aux.dictionary.t(), aux.coeff_bins, aux.coeff_scales
    projector = SparseSiteProjector(dictionary, bins, scales).cuda().eval()
    boundaries = (bins[1:] + bins[:-1]) / 2
    atoms = raw['train']['atoms'].reshape(-1, 4)[fit_indices].cuda().long()
    coefficients = raw['train']['coefficients'].reshape(-1, 4)[fit_indices].cuda()
    fit_z = (dictionary[atoms] * coefficients[..., None]).sum(-2)
    result = {'format': 'church_factorized_complete_site_integer_v1', 'phase': 'fitting',
              'checkpoint_sha256': checkpoint_hash, 'cache_sha256': sha256_file(args.cache),
              'split_indices_sha256': sha256_file(args.split_indices), 'fit_sites': len(fit_indices),
              'seed': 9711, 'iterations_per_stage': args.iterations, 'residual_vocabulary_per_stage': 512,
              'reserved_zero_residual_entry': True, 'calibration_excluded_from_fit': True,
              'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'reference': 'continuous frozen LASER reconstruction; compression, not unconditional generation',
              'scope': 'Each integer determines all four atoms AND signed coefficient bins. Internal fields remain factorized.',
              'acceptance': 'calibration LPIPS mean + 2 SE <= .01 AND relative latent MSE <= .005',
              'nominal_bits_exclude_shared_codebook_storage': True, 'neural_weights_updated': False,
              'splits': {}, 'source_sha256': {}}
    source_dir = args.output / 'source'
    source_dir.mkdir()
    for relative in ('scripts/probe_church_residual_site_integer.py', 'src/residual_site_integer_codec.py',
                     'src/coefficient_pattern_codec.py', 'src/learned_sparse_site_codec.py', 'src/complete_sparse_codec.py',
                     'scripts/train_official_rqtransformer_laser_stage2.py'):
        source = ROOT / relative
        result['source_sha256'][relative] = sha256_file(source)
        shutil.copy2(source, source_dir / source.name)
    save_result()
    log({'phase': 'fit_start', 'fit_sites': len(fit_indices)})
    books = fit_residual_codebooks(fit_z, stages=7, vocabulary=512, iterations=args.iterations, seed=9711,
        progress=lambda stage, iteration, error, empty: log({'phase': 'residual_lloyd', 'stage': stage,
            'iteration': iteration, 'mean_squared_latent_error': error, 'empty_clusters': empty}))
    grams = selected_support_grams(atoms, dictionary)
    patterns = fit_coefficient_patterns(coefficients, num_patterns=128, grams=grams, iterations=args.iterations,
        seed=9721, chunk_size=4096, progress=lambda iteration, error, empty: log({'phase': 'support_pattern_lloyd',
            'iteration': iteration, 'mean_squared_latent_error': error, 'empty_clusters': empty}))
    pattern_bins = torch.bucketize((patterns / scales).contiguous(), boundaries)
    patterns = bins[pattern_bins] * scales
    atomic_torch_save({'format': result['format'], 'checkpoint_sha256': checkpoint_hash,
        'residual_codebooks': books.cpu(), 'pattern_coefficient_ids': pattern_bins.cpu(),
        'coefficient_bins': bins.cpu(), 'coefficient_scales': scales.cpu(), 'fit_site_indices': fit_indices,
        'calibration_image_indices': calibration}, args.output / 'codebooks.pt')
    del fit_z, grams, atoms, coefficients

    def decode(z):
        return aux.decoder(aux.post_quant_conv(z.permute(0, 3, 1, 2).contiguous())).clamp(-1, 1)

    evaluations = [('calibration', 'train', calibration),
        ('holdout', 'holdout', torch.randperm(len(raw['holdout']['atoms']), generator=torch.Generator().manual_seed(9703))[:256]),
        ('validation', 'validation', torch.arange(len(raw['validation']['atoms'])))]
    log({'phase': 'evaluation_start'})
    for label, split_name, indices in evaluations:
        records, token_rows = {}, []
        for first in range(0, len(indices), args.batch_size):
            selected = indices[first:first + args.batch_size]
            atoms = raw[split_name]['atoms'][selected].cuda().long()
            coefficients = raw[split_name]['coefficients'][selected].cuda()
            z = (dictionary[atoms] * coefficients[..., None]).sum(-2)
            target = decode(z)
            fields = encode_residual_fields(z, books)
            scalar_bins = torch.bucketize((coefficients / scales).contiguous(), boundaries)
            conditions = {'nearest_scalar_100bit': sparse_latents(atoms, scalar_bins, dictionary, bins, scales)}
            tokens = {}
            for stages in (3, 5, 7):
                name = f'residual_{stages * 9}bit'
                ids = pack_fields(fields[..., :stages], [9] * stages)
                assert torch.equal(unpack_fields(ids, [9] * stages), fields[..., :stages])
                dense = decode_residual_ids(ids, books[:stages])
                recovered = projector(dense)
                torch.testing.assert_close(recovered['latents'], sparse_latents(recovered['atoms'],
                    recovered['coefficient_ids'], dictionary, bins, scales), atol=0, rtol=0)
                conditions[name] = recovered['latents']
                conditions[name + '_dense_diagnostic'] = dense
                tokens[name] = {'site_ids': ids.cpu(), 'atoms': recovered['atoms'].short().cpu(),
                                'coefficient_ids': recovered['coefficient_ids'].short().cpu()}
            gram = selected_support_grams(atoms.reshape(-1, 4), dictionary)
            pattern_ids, _ = assign_coefficient_patterns(coefficients.reshape(-1, 4), patterns, grams=gram)
            complete_fields = torch.cat((atoms, pattern_ids.reshape(*atoms.shape[:-1], 1)), -1)
            ids = pack_fields(complete_fields, [14] * 4 + [7])
            recovered = unpack_fields(ids, [14] * 4 + [7])
            assert torch.equal(recovered, complete_fields)
            recovered_bins = pattern_bins[recovered[..., 4]]
            conditions['support_pattern_63bit'] = sparse_latents(recovered[..., :4], recovered_bins, dictionary, bins, scales)
            tokens['support_pattern_63bit'] = {'site_ids': ids.cpu(), 'atoms': recovered[..., :4].short().cpu(),
                                              'coefficient_ids': recovered_bins.short().cpu()}
            token_rows.append(tokens)
            grid = [target]
            for name, reconstructed in conditions.items():
                image = decode(reconstructed)
                row = {'lpips': perceptual(image, target).flatten().cpu(),
                       'psnr': (-10 * ((image - target) / 2).square().mean((1, 2, 3)).clamp_min(1e-12).log10()).cpu(),
                       'relative_latent_mse': ((reconstructed - z).square().mean((1, 2, 3)) / z.square().mean((1, 2, 3))).cpu()}
                records.setdefault(name, []).append(row)
                if '_dense_' not in name and name != 'nearest_scalar_100bit':
                    grid.append(image)
            if first == 0:
                save_image((torch.stack(grid, 1).flatten(0, 1) + 1) / 2,
                           args.output / f'{label}-reconstructions.png', nrow=len(grid))
        per_image = {name: {k: torch.cat([row[k] for row in rows]) for k in rows[0]} for name, rows in records.items()}
        metrics = {name: {k: float(value.mean()) for k, value in rows.items()} for name, rows in per_image.items()}
        for name, row in per_image.items():
            metrics[name]['lpips_upper_2se'] = float(row['lpips'].mean() + 2 * row['lpips'].std() / len(indices) ** .5)
        result['splits'][label] = metrics
        token_data = {name: {k: torch.cat([row[name][k] for row in token_rows]) for k in token_rows[0][name]} for name in token_rows[0]}
        atomic_torch_save({'per_image': per_image, 'image_indices': indices, 'tokens': token_data}, args.output / f'{label}.pt')
        save_result()
        log({'phase': 'evaluation', 'split': label, 'images': len(indices), 'metrics': metrics})
    result['passes_calibration'] = {name: row['lpips_upper_2se'] <= .01 and row['relative_latent_mse'] <= .005
        for name, row in result['splits']['calibration'].items() if '_dense_' not in name and name != 'nearest_scalar_100bit'}
    assert all(value._version == version for value, version in frozen)
    assert all(value.grad is None for value, _ in frozen)
    result.update({'phase': 'complete', 'frozen_weight_versions_verified': True, 'unconditional_quality_tested': False,
                   'elapsed_seconds': time.monotonic() - started})
    save_result()
    log({'phase': 'complete', 'passes_calibration': result['passes_calibration']})


if __name__ == '__main__':
    main()
