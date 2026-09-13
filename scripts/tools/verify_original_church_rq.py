#!/usr/bin/env python3
"""Exercise the released models, soft targets, causal cache and actual DDP update."""
import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.upstream.resolve()))
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from rqvae.img_datasets.lsun import LSUNClass
    from rqvae.img_datasets.transforms import create_transforms
    from rqvae.optimizer import create_scheduler
    from rqvae.metrics.fid import get_inception_model, frechet_distance
    from src.original_rq_training import (atomic_json, seed_all, state_sha256,
        load_stage2_config, fresh_transformer, load_tokenizer, accumulated_update,
        CachedLatents, validation_latents, evaluate_validation)

    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group('nccl', timeout=timedelta(minutes=15))
    started = time.time()
    output = args.output.resolve()
    if rank == 0:
        output.mkdir(exist_ok=False, parents=True)
    dist.barrier()
    seed_all(0)
    config = load_stage2_config(args.upstream)
    model, optimizer = fresh_transformer(config, device)
    initial = state_sha256(model)
    fingerprints = [None] * world
    dist.all_gather_object(fingerprints, initial)
    assert len(set(fingerprints)) == 1
    assert not optimizer.state
    assert sum(p.numel() for p in model.parameters()) > 350_000_000
    assert config.arch.body.n_layer == 24 and config.arch.head.n_layer == 4
    assert config.arch.body.block.n_head == 16 and config.arch.embed_dim == 1024
    assert config.loss.stochastic_codes and config.loss.temp == .5
    assert config.arch.input_emb_vqvae and config.arch.head_emb_vqvae and config.arch.cumsum_depth_ctx
    scheduler = create_scheduler(optimizer, config.optimizer.warmup, 494, 300)
    assert scheduler.warmup_scheduler is None
    assert scheduler.after_scheduler.T_max == 494 * 300
    assert scheduler.after_scheduler.eta_min == 0
    assert scheduler.get_last_lr() == [.0005]
    print(f'rank {rank}: fresh model and official config verified', flush=True)

    published = Path('/workspace/tmp/original-rqvae-473/published-stage1')
    tokenizer, _ = load_tokenizer(published / 'model.pt', published / 'config.yaml', device)
    tokenizer_hash = state_sha256(tokenizer.quantizer)
    ds = LSUNClass('/tmp/laser-sign-data', category_name='church',
                   transform=create_transforms(config.dataset, split='train'))
    xs = torch.stack([ds[rank * 2 + i][0] for i in range(2)]).to(device)
    ds.env.close()
    with torch.no_grad():
        latent = tokenizer.encode(xs)
        assert latent.dtype == torch.float32
        torch.manual_seed(17 + rank)
        direct_targets, direct_codes = tokenizer.get_soft_codes(xs, temp=.5, stochastic=True)
        path = output / f'latents-rank{rank}.npy'
        np.save(path, latent.cpu().numpy())
        cache = CachedLatents(path)
        reread = torch.stack([cache[0], cache[1]]).to(device)
        assert torch.equal(latent, reread)
        torch.manual_seed(17 + rank)
        targets, codes = tokenizer.quantizer.get_soft_codes(reread, temp=.5, stochastic=True)
        assert torch.equal(direct_codes, codes) and torch.equal(direct_targets, targets)
        assert targets.shape == (2, 8, 8, 4, 16384)
        assert torch.allclose(targets.sum(-1), torch.ones_like(targets.sum(-1)), atol=2e-6)
        # Independent squared-distance formula verifies conditioning on sampled depth-0 code.
        residual = latent[0, 0, 0] - tokenizer.quantizer.codebooks[0].weight[codes[0, 0, 0, 0]]
        distances = (tokenizer.quantizer.codebooks[0].weight[:-1] - residual).square().sum(-1)
        expected = torch.softmax(-distances / .5, -1)
        assert torch.allclose(expected, targets[0, 0, 0, 1], atol=2e-5, rtol=2e-3)

        model.eval()
        logits = model(codes[:1], model_aux=tokenizer, amp=False)
        altered = codes[:1].clone()
        cutoff = (2 * 8 + 3) * 4 + 1
        altered.view(-1)[cutoff:] = (altered.view(-1)[cutoff:] + 17) % 16384
        changed = model(altered, model_aux=tokenizer, amp=False)
        assert torch.equal(logits.reshape(-1, 16384)[:cutoff+1],
                           changed.reshape(-1, 16384)[:cutoff+1]), 'Future code leaked into causal logits'
        assert not torch.equal(logits, changed)
        # Every cached position must agree with a teacher-forced full pass.
        model.init_cache()
        max_cache_error = 0.
        for index in range(256):
            h, w, depth = index // 32, (index // 4) % 8, index % 4
            cached = model.cached_forward(codes[:1], model_aux=tokenizer,
                sample_loc=(h, w, depth), amp=False)
            error = (cached - logits[:, h, w, depth]).abs().max().item()
            max_cache_error = max(max_cache_error, error)
            assert torch.allclose(cached, logits[:, h, w, depth], atol=2e-4, rtol=2e-4), (index, error)
        model.init_cache()
        loss = model.compute_loss(logits.float(), targets[:1], use_soft_target=True)
        reference = torch.nn.functional.cross_entropy(logits.reshape(-1, 16384), targets[:1].reshape(-1, 16384))
        assert torch.allclose(loss, reference, atol=2e-6)
        recon = tokenizer(xs)[0]
        deterministic_codes = tokenizer.get_codes(xs)
        decoded = tokenizer.decode_code(deterministic_codes)
        print(json.dumps({'rank': rank, 'reconstruction_path_max_error': float((recon-decoded).abs().max()),
                          'reconstruction_path_mse': float((recon-decoded).square().mean())}), flush=True)
        z = tokenizer.encode(xs)
        quantized, _, reconstructed_codes = tokenizer.quantizer(z)
        embedded = tokenizer.quantizer.embed_code(deterministic_codes)
        assert torch.equal(deterministic_codes, reconstructed_codes)
        quantized_error = float((quantized - embedded).abs().max())
        assert quantized_error < 5e-7
        # The released straight-through expression x+(q-x) and embedding sum
        # differ at FP32 roundoff. TF32 convolutions can amplify boundary changes.
        # Check decoding equivalence with full-mantissa convolution arithmetic.
        original_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.allow_tf32 = False
        full_a = tokenizer.decode(quantized)
        full_b = tokenizer.decode(embedded)
        full_precision_error = float((full_a-full_b).abs().max())
        full_precision_mse = float((full_a-full_b).square().mean())
        print(json.dumps({'rank': rank, 'quantized_error': quantized_error,
            'full_precision_decoder_error': full_precision_error,
            'full_precision_decoder_mse': full_precision_mse}), flush=True)
        assert full_precision_error < 2e-4 and full_precision_mse < 1e-10
        torch.backends.cudnn.allow_tf32 = original_tf32
    print(f'rank {rank}: tokenizer, cache, causal masking and cached sampling verified', flush=True)

    ddp = DistributedDataParallel(model, device_ids=[device.index], broadcast_buffers=False)
    model.train()
    seed_all(rank)
    scaler = torch.amp.GradScaler('cuda')
    history = []
    for attempt in range(12):
        # Use the production microbatch and accumulation: 32 x 4 x 2 = 256.
        batches = [latent.detach().cpu().repeat(16, 1, 1, 1) for _ in range(4)]
        metrics = accumulated_update(ddp, tokenizer, optimizer, scaler, batches)
        history.append(metrics)
        print(json.dumps({'rank': rank, 'attempt': attempt, **metrics}), flush=True)
        if sum(row['optimizer_updated'] for row in history) >= 2:
            break
    assert sum(row['optimizer_updated'] for row in history) >= 2
    assert optimizer.state and initial != state_sha256(model)
    assert tokenizer_hash == state_sha256(tokenizer.quantizer)

    # Verify actual held-out population and evaluation loss path.
    heldout = validation_latents(tokenizer, config, device, rank, world)
    validation = evaluate_validation(model, tokenizer, heldout, device, rank)
    inception = get_inception_model().eval().to(device)
    with torch.no_grad():
        features = inception(decoded.mul(.5).add(.5).clamp(0, 1))
    assert features.shape == (2, 2048) and torch.isfinite(features).all()
    if rank == 0:
        real = np.load(ROOT / 'outputs/lsun-church-bar-20260910/assets/lsun_256_church.npz')
        identity_fid = float(frechet_distance(real['mu'], real['sigma'], real['mu'], real['sigma']))
        assert abs(identity_fid) < 1e-5
        report = {'passed': True, 'initial_optimizer_entries': 0,
            'random_initialization_verified': True, 'empty_optimizer_verified': True,
            'initial_weights_sha256': initial, 'two_rank_initialization_equal': True,
            'full_parameter_count': sum(p.numel() for p in model.parameters()),
            'stochastic_cached_and_uncached_targets_identical': True,
            'residual_prefix_conditioning_verified': True, 'causality_verified': True,
            'all_256_cached_positions_verified': True, 'max_cached_logit_error': max_cache_error,
            'quantized_decoder_input_error': quantized_error,
            'fp32_decoder_max_error': full_precision_error, 'fp32_decoder_mse': full_precision_mse,
            'global_batch': 256, 'microbatch': 32, 'accumulation': 4, 'world_size': world,
            'successful_full_batch_ddp_updates': sum(row['optimizer_updated'] for row in history),
            'updates': history, 'tokenizer_unchanged': True, 'heldout_validation': validation,
            'reference_self_fid': identity_fid, 'elapsed_seconds': time.time() - started}
        atomic_json(output / 'result.json', report)
        print(json.dumps(report), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
