#!/usr/bin/env python3
"""Exercise the production upstream-FID adapter with a known LASER checkpoint."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path

import train_church_laser_three_epoch_official_fid as recipe
import numpy as np
import torch
import torch.distributed as dist
from rqvae.models import create_model
from church_official_fid import evaluate_samples, score_saved_rgb


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    out = args.output.resolve()
    if rank == 0:
        out.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    previous = recipe.ROOT / 'outputs/church-laser-original-recipe-20260913/train'
    config = recipe.OmegaConf.load(previous / 'config.yaml')
    model, _ = create_model(config.arch, ema=False)
    state = torch.load(previous / 'epoch100_model.pt', map_location='cpu', weights_only=False, mmap=True)
    assert state['epoch'] == 100
    model.load_state_dict(state['state_dict'], strict=True)
    del state
    model.to(device).train().requires_grad_(False)
    tokenizer = recipe.FrozenCompactTokenizer(config.vqvae.ckpt, config.vqvae.codebook).to(device)
    before_hashes = [recipe.state_sha256(model), recipe.state_sha256(tokenizer)]
    torch.manual_seed(12000 + rank)
    cpu_rng = torch.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state(device).clone()
    reference = recipe.ROOT / 'outputs/lsun-church-bar-20260910/assets/lsun_256_church.npz'
    score = evaluate_samples(model, tokenizer, None, 4096, 100, out, reference, device, rank, world)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(gpu_rng, torch.cuda.get_rng_state(device))
    assert model.training and not tokenizer.training
    assert before_hashes == [recipe.state_sha256(model), recipe.state_sha256(tokenizer)]
    expected = json.loads((previous / 'fid-4096-epoch100.json').read_text())['fid']
    assert abs(score - expected) < 1e-4, (score, expected)
    receipt = dict(passed=True, samples=4096, epoch=100, fid=score, prior_streamed_fid=expected,
        difference=abs(score-expected), cpu_rng_restored=True, cuda_rng_restored=True,
        model_mode_restored=True, model_and_tokenizer_unchanged=True, rank=rank)
    recipe.atomic_json(out / f'verification-rank{rank}.json', receipt)
    if rank == 0:
        folder = out / 'official-fid/fid-4096-epoch100'
        saved = json.loads((folder / 'result.json').read_text())
        assert sum(row['images'] for row in saved['sample_files']) == 4096
        with np.load(folder / 'acts.npz') as values:
            assert values['acts'].shape == (4096, 2048)
        assert not list(folder.glob('samples*.pkl'))
        try:
            score_saved_rgb(folder, saved['sample_files'], 4096, reference, device)
        except AssertionError as error:
            assert 'pre-existing FID activation cache' in str(error)
        else:
            raise AssertionError('Stale activation cache was accepted')
        receipt.update(stale_cache_rejected=True, rgb_cleanup_verified=True,
                       full_inception_features_retained=True, upstream_function_used=True)
        recipe.atomic_json(out / 'verification.json', receipt)
        print(json.dumps(receipt), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
