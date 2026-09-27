#!/usr/bin/env python3
"""Rebuild FP32 encoder latents with the exact recovered three-epoch tokenizer."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from prepare_church_control_cache import ChurchImages


def verify_data(base):
    protocol = json.loads((base/'reference/data-protocol.json').read_text())
    records = {}
    for split, category in [('train', 'church'), ('val', 'church_val')]:
        dataset = ChurchImages(base/f'assets/data/church_outdoor_{split}_lmdb')
        expected = protocol['cache_stage2_fid']['datasets'][category]
        assert len(dataset) == expected['images']
        digest = hashlib.sha256()
        for key in dataset.keys:
            digest.update(len(key).to_bytes(8, 'little'))
            digest.update(key)
        assert digest.hexdigest() == expected['key_order_sha256']
        for index, sha in expected['pixel_probes'].items():
            assert hashlib.sha256(dataset[int(index)][0].numpy().tobytes()).hexdigest() == sha, (split, index)
        if dataset.env is not None:
            dataset.env.close()
        records[split] = dict(images=len(dataset), key_order_sha256=digest.hexdigest(),
                              verified_pixel_probes=len(expected['pixel_probes']))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    base = args.pipeline_dir.resolve()
    # This import selects the frozen source snapshot using --pipeline-dir.
    from resume_church_laser_ft3ep import load_frozen_tokenizer
    from src.original_rq_training import atomic_json, file_sha256, state_sha256
    rank, world, local_rank = (int(os.environ[x]) for x in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    gpu = torch.cuda.get_device_name(device)
    assert 'A100' in gpu or 'L40S' in gpu
    atomic_json(base/f'gpu-rank{rank}.json', dict(rank=rank, world_size=world, hostname=socket.gethostname(),
        local_rank=local_rank, gpu_name=gpu, memory_bytes=torch.cuda.get_device_properties(device).total_memory))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    dist.init_process_group('nccl', timeout=timedelta(hours=3))
    cache_dir = base/'cache'
    cache_dir.mkdir(exist_ok=True)
    if (cache_dir/'complete.json').is_file():
        cached = json.loads((cache_dir/'complete.json').read_text())
        if rank == 0:
            assert file_sha256(cached['latent_cache']) == cached['cache_sha256']
            assert file_sha256(cache_dir/'validation-all.npy') == cached['validation_all_sha256']
        dist.barrier()
        dist.destroy_process_group()
        return
    cache = json.loads((base/'recovery/cache/complete.json').read_text())
    cache.update(checkpoint=str(base/'recovery/tokenizer/epoch3-tokenizer.pt'),
                 codebook=str(base/'recovery/tokenizer/compact-codebook.pt'))
    assert file_sha256(cache['checkpoint']) == cache['checkpoint_sha256']
    assert file_sha256(cache['codebook']) == cache['codebook_sha256']
    if rank == 0:
        atomic_json(base/'data-preflight.json', verify_data(base))
    dist.barrier()
    tokenizer = load_frozen_tokenizer(cache).to(device).eval()
    train_path, val_path = cache_dir/'latents-fp32.npy', cache_dir/'validation-all.npy'
    if rank == 0:
        for path, count in ((train_path, 126227), (val_path, 300)):
            values = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(count,8,8,256))
            del values
    dist.barrier()
    started = time.time()
    with torch.inference_mode():
        for split, path in [('train', train_path), ('val', val_path)]:
            data_root = Path(os.environ['CHURCH_LOCAL_ROOT'])/'data'
            dataset = ChurchImages(data_root/f'church_outdoor_{split}_lmdb')
            loader = DataLoader(Subset(dataset, range(rank, len(dataset), world)), batch_size=args.batch_size,
                num_workers=2, pin_memory=True, shuffle=False)
            values = np.load(path, mmap_mode='r+')
            for index, (images, indices) in enumerate(loader):
                latents = tokenizer.encode(images.to(device, non_blocking=True))
                assert latents.dtype == torch.float32 and torch.isfinite(latents).all()
                values[indices.numpy()] = latents.cpu().numpy()
                if rank == 0 and (index % 50 == 0 or index+1 == len(loader)):
                    status = dict(phase='rebuilding_laser_cache',split=split,batch=index+1,batches=len(loader),
                                  elapsed_seconds=time.time()-started,updated_unix=time.time())
                    atomic_json(base/'cache-status.json',status)
                    print(json.dumps(status),flush=True)
            values.flush()
            del values, loader
    assert state_sha256(tokenizer) == cache['frozen_state_sha256']
    dist.barrier()
    if rank == 0:
        old_hash = cache['cache_sha256']
        cache.update(latent_cache=str(train_path),cache_sha256=file_sha256(train_path),world_size=world,
            validation_all_sha256=file_sha256(val_path),amarel_rebuild=dict(source_cache_sha256=old_hash,
                exact_tokenizer_verified=True,exact_real_fid_reference_recovered=True,
                exact_key_order_and_pixel_probes_verified=True,bitwise_original_latents=False))
        atomic_json(cache_dir/'complete.json',cache)
        print(json.dumps(dict(phase='cache_complete',**cache)),flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
