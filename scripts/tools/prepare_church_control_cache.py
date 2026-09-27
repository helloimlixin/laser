#!/usr/bin/env python3
"""Rebuild Church FP32 latents and a full-training-set FID reference on allocated GPUs."""
import argparse
from datetime import timedelta
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

import lmdb
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms


class ChurchImages(Dataset):
    def __init__(self, path):
        self.path = str(path)
        self.env = None
        with lmdb.open(self.path, readonly=True, lock=False, readahead=False) as env:
            with env.begin() as txn:
                self.keys = list(txn.cursor().iternext(keys=True, values=False))
        self.transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256),
            transforms.ToTensor(), transforms.Normalize([.5] * 3, [.5] * 3)])

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if self.env is None:
            self.env = lmdb.open(self.path, readonly=True, lock=False, readahead=False)
        with self.env.begin() as txn:
            data = txn.get(self.keys[index])
        return self.transform(Image.open(io.BytesIO(data)).convert('RGB')), index


def verify_keys(dataset, expected):
    assert len(dataset) == expected['images']
    # Reproduce the recorded key-order hash, allowing common delimiter formats.
    forms = [b'\n'.join(dataset.keys), b''.join(dataset.keys), b'\n'.join(dataset.keys) + b'\n']
    hashes = [hashlib.sha256(value).hexdigest() for value in forms]
    assert expected['key_order_sha256'] in hashes, (expected['key_order_sha256'], hashes)
    return expected['key_order_sha256']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=16)
    args = p.parse_args()
    base = args.base.resolve()
    sys.path[:0] = [str(base / 'source/upstream'), str(base / 'source')]
    from src.original_rq_training import atomic_json, file_sha256, state_sha256, load_tokenizer, FeatureMoments
    from rqvae.metrics.fid import get_inception_model
    rank, world, local_rank = (int(os.environ[x]) for x in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    gpu_name = torch.cuda.get_device_name(device)
    assert 'A100' in gpu_name or 'L40S' in gpu_name, gpu_name
    assert torch.cuda.device_count() == int(os.environ['LOCAL_WORLD_SIZE'])
    atomic_json(base / f'gpu-rank{rank}.json', dict(rank=rank, world_size=world,
        hostname=__import__('socket').gethostname(), local_rank=local_rank, gpu_name=gpu_name,
        memory_bytes=torch.cuda.get_device_properties(device).total_memory,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    dist.init_process_group('nccl', timeout=timedelta(hours=3))
    source = json.loads((base / 'checkpoint-metadata.json').read_text())
    assets = base / 'assets'
    cache_dir, reference_dir = base / 'cache', base / 'reference'
    for directory in (cache_dir, reference_dir):
        directory.mkdir(exist_ok=True)
    if (cache_dir / 'complete.json').exists() and (reference_dir / 'complete.json').exists():
        cached = json.loads((cache_dir / 'complete.json').read_text())
        if rank == 0:
            assert file_sha256(cached['latent_cache']) == cached['cache_sha256']
            ref = json.loads((reference_dir / 'complete.json').read_text())
            assert file_sha256(reference_dir / 'real-statistics.npz') == ref['sha256']
        dist.barrier()
        dist.destroy_process_group()
        return
    ckpt, config = assets / 'tokenizer/model.pt', assets / 'tokenizer/config.yaml'
    assert file_sha256(ckpt) == source['tokenizer']['checkpoint_sha256']
    assert file_sha256(config) == source['tokenizer']['config_sha256']
    tokenizer, _ = load_tokenizer(ckpt, config, device)
    frozen_hash = state_sha256(tokenizer)
    assert frozen_hash == source['tokenizer']['frozen_state_sha256']
    inception = get_inception_model().eval().requires_grad_(False).to(device)
    train = ChurchImages(assets / 'data/church_outdoor_train_lmdb')
    val = ChurchImages(assets / 'data/church_outdoor_val_lmdb')
    for dataset, name in ((train, 'church'), (val, 'church_val')):
        verify_keys(dataset, source['tokenizer']['data_verification'][name])
    latent_path = cache_dir / 'latents-fp32.npy'
    val_path = cache_dir / 'validation-all.npy'
    if rank == 0:
        for path, n in ((latent_path, len(train)), (val_path, len(val))):
            values = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(n, 8, 8, 256))
            del values
    dist.barrier()
    moments = FeatureMoments(device)
    started = time.time()
    with torch.inference_mode():
        for dataset, path, split in ((train, latent_path, 'train'), (val, val_path, 'val')):
            values = np.load(path, mmap_mode='r+')
            loader = DataLoader(Subset(dataset, range(rank, len(dataset), world)), batch_size=args.batch_size,
                                num_workers=4, pin_memory=True, shuffle=False)
            for index, (images, indices) in enumerate(loader):
                images = images.to(device)
                latents = tokenizer.encode(images)
                assert latents.dtype == torch.float32 and torch.isfinite(latents).all()
                values[indices.numpy()] = latents.cpu().numpy()
                if split == 'train':
                    moments.update(inception(images.mul(.5).add(.5).clamp(0, 1)))
                if rank == 0 and (index % 50 == 0 or index + 1 == len(loader)):
                    status = dict(phase='rebuilding_cache', split=split, batch=index + 1,
                                  batches=len(loader), elapsed_seconds=time.time() - started,
                                  updated_unix=time.time())
                    atomic_json(base / 'cache-status.json', status)
                    print(json.dumps(status), flush=True)
            values.flush()
            del values, loader
    assert state_sha256(tokenizer) == frozen_hash
    count, mu, sigma = moments.finish()
    assert count == 126227
    dist.barrier()
    if rank == 0:
        data_protocol = dict(source='LSUN original LMDB mirror RichardErkhov/LSUN',
            source_revision='1ee52c98c617f4222220fb124d761bef00ae5fbc',
            data_verification=source['tokenizer']['data_verification'],
            transform='PIL RGB; Resize(256) bilinear; CenterCrop(256); ToTensor; Normalize(.5,.5)',
            original_data_protocol_sha256=source['tokenizer']['data_protocol_sha256'],
            reference='All 126227 training images; FP32 Inception; float pixels; TF32 disabled')
        atomic_json(reference_dir / 'data-protocol.json', data_protocol)
        np.savez(reference_dir / 'real-statistics.npz', mu=mu, sigma=sigma, samples=count)
        reference_hash = file_sha256(reference_dir / 'real-statistics.npz')
        protocol_hash = file_sha256(reference_dir / 'data-protocol.json')
        validation = np.load(val_path, mmap_mode='r')
        val_hashes = []
        for stream in range(2):
            path = cache_dir / f'validation-rank{stream}.pt'
            torch.save(torch.from_numpy(validation[stream::2].copy()), path)
            val_hashes.append(file_sha256(path))
        report = dict(source['tokenizer'])
        report.update(checkpoint=str(ckpt), config=str(config), latent_cache=str(latent_path),
            cache_sha256=file_sha256(latent_path), validation_sha256=val_hashes,
            data_protocol_sha256=protocol_hash, world_size=world,
            amarel_rebuild=dict(source_cache_sha256=source['tokenizer']['cache_sha256'],
                source_reference_sha256=source['fid_protocol']['reference_sha256'],
                reference_sha256=reference_hash, original_tokenizer_verified=True,
                original_lmdb_key_order_verified=True, bitwise_original_cache=False))
        atomic_json(reference_dir / 'complete.json', dict(sha256=reference_hash,
                    data_protocol_sha256=protocol_hash, images=count))
        atomic_json(cache_dir / 'complete.json', report)
        print(json.dumps(dict(phase='cache_complete', **report)), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
