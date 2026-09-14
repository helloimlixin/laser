#!/usr/bin/env python3
"""Fit Church-style atom-specific levels on ImageNet and audit reconstruction."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / 'outputs/church-rq-baseline-scratch-20260912/upstream-source'
sys.path[:0] = [str(ROOT), str(UPSTREAM)]
os.environ.setdefault('TORCH_HOME', '/workspace/tmp/official-rqvae-eval-cache')

import numpy as np
import torch
import torch.distributed as dist
from src.adaptive_scaled_atom_rq import AdaptiveScaledAtomRQ, fit_atom_levels
from src.compact_rq_training import adaptive_quantizer
from src.scaled_atom_rq import FrozenSparseBackbone, ScaledAtomRQ, continuous_matching_pursuit, fit_signed_levels
from src.original_rq_training import atomic_json, file_sha256, state_sha256


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT / 'outputs/imagenet-rfid421-rq8-refit-20260913')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:3')
    parser.add_argument('--levels', type=int, nargs='+', default=[2, 4])
    parser.add_argument('--passes', type=int, default=8)
    parser.add_argument('--fit-images', type=int, default=4096)
    parser.add_argument('--images', type=int, choices=[4096, 50000], default=4096)
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--skip-control', action='store_true')
    parser.add_argument('--books', nargs='+', help='Named candidate files: name=/path/to/book.pt')
    args = parser.parse_args()
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    if world > 1:
        assert args.evaluate_only
        args.device = f'cuda:{os.environ["LOCAL_RANK"]}'
        torch.cuda.set_device(args.device)
        dist.init_process_group('nccl')
    torch.set_num_threads(8)
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    args.output.mkdir(parents=True, exist_ok=args.evaluate_only)
    started = time.time()

    def status(phase, **values):
        if rank != 0:
            return
        record = dict(phase=phase, elapsed_seconds=time.time()-started, updated_unix=time.time(), **values)
        atomic_json(args.output / 'status.json', record)
        print(json.dumps(record), flush=True)

    source = args.source.resolve()
    cache = json.loads((source / 'cache/complete.json').read_text())
    calibration = json.loads((source / 'temperature-calibration.json').read_text())
    old = torch.load(source / 'scaled-atom-codebook.pt', map_location='cpu', weights_only=True)
    assert file_sha256(cache['checkpoint']) == cache['checkpoint_sha256'] == old['checkpoint_sha256']
    assert file_sha256(source / 'scaled-atom-codebook.pt') == cache['codebook_sha256']
    dictionary = old['dictionary'].to(args.device)
    if not args.evaluate_only:
        values = np.load(source / 'cache/train-view0-latents.npy', mmap_mode='r')
        excluded = set(calibration['calibration_indices'])
        indices = [int(i) for i in np.random.default_rng(91332769).permutation(len(values))
                   if int(i) not in excluded][:args.fit_images]
        latents = np.array(values[indices], copy=True)
        assert len(indices) == args.fit_images and not excluded.intersection(indices)
        coefficients = []
        for start in range(0, len(latents), 16):
            z = torch.from_numpy(latents[start:start+16]).to(args.device)
            coefficients.append(continuous_matching_pursuit(z, dictionary)['coefficients'])
        coefficients = torch.cat(coefficients)
        for levels in args.levels:
            initial = fit_signed_levels(coefficients, levels)
            quantizer = AdaptiveScaledAtomRQ(dictionary, initial.expand(dictionary.shape[1], -1))
            trace = fit_atom_levels(quantizer, latents, passes=args.passes, batch_size=16, prior_weight=4.,
                callback=lambda row: status('fitting', levels=levels, **row))
            book = dict(format_version=1, kind='adaptive_scaled_atom_rq', dictionary=dictionary.cpu(),
                levels=quantizer.levels.cpu(), depth=4, zero_token=0, code_shape=[8,8,4],
                source_stage1_checkpoint=cache['checkpoint'],
                source_stage1_checkpoint_sha256=cache['checkpoint_sha256'])
            torch.save(book, args.output / f'compact{levels}-codebook.pt')
            atomic_json(args.output / f'fit{levels}.json', dict(fit_indices=indices, train_view=0,
                cache_manifest=str(source / 'cache/complete.json'),
                cache_manifest_sha256=file_sha256(source / 'cache/complete.json'),
                temperature_calibration_disjoint=True, fit_on_training_images_only=True,
                passes=args.passes, prior_weight=4., initial_shared_levels=initial.cpu().tolist(), trace=trace,
                source_hashes={str(p.relative_to(ROOT)):file_sha256(p) for p in
                    [Path(__file__), ROOT/'src/adaptive_scaled_atom_rq.py', ROOT/'src/scaled_atom_rq.py']}))
        del coefficients, latents, values
    backbone = FrozenSparseBackbone(Path(cache['checkpoint'])).to(args.device).eval()
    before = state_sha256(backbone)
    torch.testing.assert_close(backbone.dictionary, dictionary, rtol=0, atol=0)
    from rqvae.metrics.fid import get_inception_model, frechet_distance
    inception = get_inception_model().eval().requires_grad_(False).to(args.device)
    indices = sorted(np.random.default_rng(73421).choice(50000, args.images, replace=False).tolist())
    reference = ROOT / ('outputs/imagenet-tokenizer-fidelity-20260913/' +
        ('full-matched-first' if args.images == 50000 else 'rq8-refit-screen') + '/original-validation-statistics.npz')
    real = np.load(reference)
    assert int(real['images']) == args.images and np.array_equal(real['validation_indices'], indices)
    values = np.load(source / 'cache/val-view0-latents.npy', mmap_mode='r')
    quantizers = {}
    # Recompute the current control in the screen, using identical kernels and images.
    if args.images == 4096 and not args.skip_control:
        quantizers['shared8'] = ScaledAtomRQ(dictionary, old['levels']['8'].to(args.device))
    book_paths = ({name:Path(path) for name,path in (item.split('=',1) for item in args.books)} if args.books else
        {f'adaptive{levels}':args.output/f'compact{levels}-codebook.pt' for levels in args.levels})
    for name, book_path in book_paths.items():
        book = torch.load(book_path, map_location='cpu', weights_only=True)
        torch.testing.assert_close(book['dictionary'], dictionary.cpu(), rtol=0, atol=0)
        quantizers[name] = adaptive_quantizer(dictionary, book['levels'].to(args.device),depth=book['depth'])
    results = {}
    for name, quantizer in quantizers.items():
        if world > 1 and args.images == 50000:
            if rank == 0:
                a = np.lib.format.open_memmap(args.output/f'{name}-50000-codes.npy', mode='w+',
                    dtype=np.uint32, shape=(50000,8,8,4))
                del a
            dist.barrier()
        codes_cache = (np.lib.format.open_memmap(args.output/f'{name}-50000-codes.npy', mode='w+',
            dtype=np.uint32, shape=(50000,8,8,4)) if args.images==50000 and world==1 else
            np.load(args.output/f'{name}-50000-codes.npy', mmap_mode='r+') if args.images==50000 else None)
        total = torch.zeros(2048, device=args.device, dtype=torch.float64)
        cross = torch.zeros(2048, 2048, device=args.device, dtype=torch.float64)
        error = 0.
        local_indices = indices[rank::world]
        for start in range(0, len(local_indices), 16):
            selected = local_indices[start:start+16]
            z = torch.from_numpy(values[selected].copy()).to(args.device)
            result = quantizer.quantize(z)
            if codes_cache is not None:
                codes_cache[selected] = result['codes'].cpu().numpy().astype(np.uint32)
            torch.testing.assert_close(quantizer.embed(result['codes']).sum(-2), result['quantized'])
            error += (z-result['quantized']).square().sum().item()
            decoded = torch.cat([backbone.decode(chunk) for chunk in result['quantized'].split(8)])
            features = inception(decoded.mul(.5).add(.5).clamp(0,1)).double()
            total += features.sum(0)
            cross += features.T @ features
            if start % 256 == 0:
                status('reconstruction', variant=name, images=min(args.images,(start+len(z))*world), total=args.images)
        if codes_cache is not None:
            codes_cache.flush()
            del codes_cache
        if world > 1:
            dist.all_reduce(total)
            dist.all_reduce(cross)
            errors = torch.tensor(error, device=args.device, dtype=torch.float64)
            dist.all_reduce(errors)
            error = errors.item()
        mean = total / args.images
        covariance = (cross-args.images*torch.outer(mean,mean))/(args.images-1)
        if rank != 0:
            dist.barrier()
            continue
        mu, sigma = mean.cpu().numpy(), covariance.cpu().numpy()
        np.savez(args.output/f'{name}-{args.images}-statistics.npz', mu=mu, sigma=sigma, images=args.images)
        status('computing_fid', variant=name)
        score = float(frechet_distance(mu, sigma, real['mu'], real['sigma']))
        results[name] = dict(rfid=score, latent_mse=error/(args.images*64*256),
            vocabulary=quantizer.vocab_size)
        if name != 'shared8':
            results[name]['codebook_sha256'] = file_sha256(book_paths[name])
        atomic_json(args.output/f'results-{args.images}.json', dict(images=args.images, results=results,
            checkpoint_sha256=cache['checkpoint_sha256'], reference=str(reference),
            reference_sha256=file_sha256(reference), validation_indices=indices,
            precision='FP32; TF32 disabled', fit_on_training_images_only=True))
        status('result', variant=name, **results[name])
        if world > 1:
            dist.barrier()
    assert before == state_sha256(backbone)
    status('complete', frozen_backbone_unchanged=True, results=results)
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
