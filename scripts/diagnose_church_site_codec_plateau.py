#!/usr/bin/env python3
"""Frozen ablations separating site-code selection, projection, and quantization."""
import argparse
import codecs
import json
import os
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
from src.learned_sparse_site_codec import LearnedSparseSiteCodec
from src.models.lpips import LPIPS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--run', type=Path, default=ROOT / 'outputs/church-learned-site-codec-20260911/train')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--inversion-images', type=int, default=32)
    p.add_argument('--inversion-steps', type=int, default=64)
    args = p.parse_args()
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
    saved = torch.load(args.run / 'best.pt', map_location='cpu', weights_only=False)
    config = saved['config']
    raw = torch.load(config['cache'], map_location='cpu', weights_only=False)
    table = torch.load(args.run / 'best-codebook.pt', map_location='cpu', weights_only=True)
    assert sha256_file(Path(config['stage1'])) == table['checkpoint_sha256'] == raw['meta']['checkpoint_sha256']
    model = LearnedSparseSiteCodec(saved['state_dict']['codewords'], width=config['architecture']['width'],
                                   layers=config['architecture']['layers']).cuda().eval().requires_grad_(False)
    model.load_state_dict(saved['state_dict'])
    hard_table = table['latents'].cuda()
    aux = LaserAux(Path(config['stage1']), 16384, 2048, 3., coeff_scales=raw['meta']['coeff_scales'],
                   sparsity_level=4, soft_target_physical=True).cuda().eval().requires_grad_(False)
    perceptual = LPIPS().cuda().eval().requires_grad_(False)
    initial = torch.load(config['initial_codebook'], map_location='cpu', weights_only=True)
    with torch.no_grad():
        reconstructed = aux.compound_embeddings(table['atoms'].cuda().long(), table['coefficient_ids'].cuda().long()).sum(-2)
        torch.testing.assert_close(reconstructed, hard_table, atol=1e-6, rtol=1e-6)
    started = time.monotonic()

    def log(row):
        row = {'elapsed_seconds': time.monotonic() - started, **row}
        print(json.dumps(row, allow_nan=False), flush=True)
        with (args.output / 'history.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')

    def decode(z):
        return aux.decoder(aux.post_quant_conv(z.float())).clamp(-1, 1)

    @torch.no_grad()
    def target_latent(split, indices):
        atoms = raw[split]['atoms'][indices].cuda().long()
        coefficients = raw[split]['coefficients'][indices].cuda()
        return (aux.dictionary.t()[atoms] * coefficients[..., None]).sum(-2).permute(0, 3, 1, 2).contiguous()

    @torch.no_grad()
    def nearest(z, words):
        flat = z.permute(0, 2, 3, 1).reshape(-1, 256)
        ids, _ = assign_coefficient_patterns(flat, words, chunk_size=1024)
        return ids.reshape(z.shape[0], z.shape[2], z.shape[3])

    def lookup(words, ids):
        return words[ids].permute(0, 3, 1, 2).contiguous()

    @torch.no_grad()
    def metrics(z, image, target_z, target):
        return {'lpips': perceptual(image, target).flatten(),
                'psnr': -10 * ((image - target) / 2).square().mean((1, 2, 3)).clamp_min(1e-12).log10(),
                'relative_latent_mse': (z - target_z).square().mean((1, 2, 3)) / target_z.square().mean((1, 2, 3))}

    dense = model.codewords.detach()
    old = initial['latents'].cuda()
    support_overlap = (table['atoms'][:, :, None] == initial['atoms'][:, None, :]).any(-1).float().mean()
    result = {'selected_step': saved['step'], 'weights_changed': False,
              'checkpoint_sha256': sha256_file(args.run / 'best.pt'),
              'codeword_relative_change': float(((dense - old).square().sum(-1) / old.square().sum(-1)).mean()),
              'dense_to_projected_relative_mse': float(((dense - hard_table).square().sum(-1) / dense.square().sum(-1)).mean()),
              'original_atom_support_retention': float(support_overlap),
              'scope': 'oracle reconstruction ablations; no unconditional generation', 'splits': {}}
    calibration = torch.load(config['split_indices'], weights_only=True)['calibration_image_indices']
    splits = [('calibration', 'train', calibration),
              ('holdout', 'holdout', torch.randperm(len(raw['holdout']['atoms']), generator=torch.Generator().manual_seed(9703))[:256]),
              ('validation', 'validation', torch.arange(len(raw['validation']['atoms'])))]
    for label, split, indices in splits:
        records = {}
        for first in range(0, len(indices), args.batch_size):
            with torch.no_grad():
                z = target_latent(split, indices[first:first + args.batch_size])
                target = decode(z)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    learned_ids, encoded = model.encode(z)
                hard_nearest = nearest(z, hard_table)
                dense_nearest = nearest(z, dense)
                conditions = [('learned_sparse', lookup(hard_table, learned_ids)),
                              ('nearest_sparse', lookup(hard_table, hard_nearest)),
                              ('learned_dense_diagnostic', lookup(dense, learned_ids)),
                              ('nearest_dense_diagnostic', lookup(dense, dense_nearest)),
                              ('continuous_encoder_diagnostic', encoded)]
                grid = [target]
                for name, value in conditions:
                    image = decode(value)
                    row = metrics(value, image, z, target)
                    records.setdefault(name, []).append({k: v.cpu() for k, v in row.items()})
                    grid.append(image)
                if first == 0:
                    save_image((torch.stack(grid, 1).flatten(0, 1) + 1) / 2,
                               args.output / f'{label}-ablations.png', nrow=len(grid))
        rows = {name: {k: torch.cat([r[k] for r in values]) for k in values[0]} for name, values in records.items()}
        values = {name: {k: float(v.mean()) for k, v in row.items()} for name, row in rows.items()}
        for metric in ('lpips', 'psnr', 'relative_latent_mse'):
            expected = saved['metrics'][label][metric]
            if abs(values['learned_sparse'][metric] - expected) > 1e-5:
                raise RuntimeError(f'Frozen baseline does not reproduce saved {label}/{metric}')
        for name in rows:
            delta = rows[name]['lpips'] - rows['learned_sparse']['lpips']
            values[name]['paired_lpips_delta'] = float(delta.mean())
            values[name]['paired_lpips_se'] = float(delta.std() / len(delta) ** .5)
        result['splits'][label] = values
        atomic_torch_save({'per_image': rows, 'image_indices': indices}, args.output / f'{label}.pt')
        log({'phase': 'ablation', 'split': label, 'metrics': values})
    (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')

    # Assignment-only inversion: optimize discrete IDs for known images. All
    # weights and codewords remain fixed. Every assessed image uses HARD tokens.
    # This is a restricted candidate search, not a global quantization bound.
    inversion_rows = []
    indices = calibration[:args.inversion_images]
    for first in range(0, len(indices), 4):
        with torch.no_grad():
            z = target_latent('train', indices[first:first + 4])
            target = decode(z)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                initial_ids, _ = model.encode(z)
            flat = z.permute(0, 2, 3, 1).reshape(-1, 256)
            distances = flat.square().sum(-1, keepdim=True) + hard_table.square().sum(-1)[None] - 2 * flat @ hard_table.t()
            candidates = distances.topk(32, largest=False).indices.reshape(*initial_ids.shape, 32)
            candidates = torch.cat([initial_ids[..., None], candidates], -1)
            candidate_words = hard_table[candidates]
            initial_z = lookup(hard_table, initial_ids)
            initial_image = decode(initial_z)
            initial_metrics = metrics(initial_z, initial_image, z, target)
            best = {k: v.clone() for k, v in initial_metrics.items()}
            best_ids = initial_ids.clone()
        logits = torch.zeros(candidates.shape, device='cuda')
        logits[..., 0] = 2.
        logits.requires_grad_(True)
        optimizer = torch.optim.Adam([logits], lr=.15)
        for iteration in range(args.inversion_steps):
            optimizer.zero_grad(set_to_none=True)
            probabilities = logits.softmax(-1)
            soft = (probabilities[..., None] * candidate_words).sum(-2).permute(0, 3, 1, 2)
            selected = candidates.gather(-1, logits.argmax(-1, keepdim=True)).squeeze(-1)
            hard = lookup(hard_table, selected)
            straight_through = hard + (soft - soft.detach())
            image = decode(straight_through)
            lpips = perceptual(image, target).flatten()
            relative = (straight_through - z).square().mean((1, 2, 3)) / z.square().mean((1, 2, 3))
            loss = lpips.mean() + .5 * (image - target).abs().mean() + .1 * relative.mean()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                # Select a discrete reconstruction per image using measured LPIPS.
                current = metrics(hard, image.detach(), z, target)
                improved = current['lpips'] < best['lpips']
                for key in best:
                    best[key] = torch.where(improved, current[key], best[key])
                best_ids[improved] = selected[improved]
        with torch.no_grad():
            final_z = lookup(hard_table, best_ids)
            final_image = decode(final_z)
            final_metrics = metrics(final_z, final_image, z, target)
            torch.testing.assert_close(final_metrics['lpips'], best['lpips'], atol=1e-6, rtol=1e-6)
            inversion_rows.append({'before': {k: v.cpu() for k, v in initial_metrics.items()},
                                   'after': {k: v.cpu() for k, v in final_metrics.items()}, 'site_ids': best_ids.cpu()})
            if first == 0:
                save_image((torch.stack([target, initial_image, final_image], 1).flatten(0, 1) + 1) / 2,
                           args.output / 'inversion.png', nrow=3)
        log({'phase': 'inversion', 'images_done': first + len(z), 'images': len(indices)})
    summary = {}
    for label in ('before', 'after'):
        summary[label] = {k: float(torch.cat([r[label][k] for r in inversion_rows]).mean()) for k in inversion_rows[0][label]}
    delta = torch.cat([r['after']['lpips'] - r['before']['lpips'] for r in inversion_rows])
    summary.update({'images': len(indices), 'steps': args.inversion_steps, 'candidates_per_site': 33,
                    'paired_lpips_delta': float(delta.mean()), 'paired_lpips_se': float(delta.std() / len(delta) ** .5),
                    'uses_known_target_images': True, 'all_outputs_decoded_from_hard_ids': True})
    result['assignment_inversion'] = summary
    atomic_torch_save({'rows': inversion_rows, 'image_indices': indices}, args.output / 'inversion.pt')
    (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    log({'phase': 'complete', 'inversion': summary})


if __name__ == '__main__':
    main()
