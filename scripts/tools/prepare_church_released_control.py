#!/usr/bin/env python3
"""Cache original Church images with the verified, frozen released RQ-VAE."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    sys.path[:0] = [str(root/'runtime/upstream'), str(root/'runtime')]
    os.environ.setdefault('TORCH_HOME', '/workspace/tmp/official-rqvae-eval-cache')
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader, Subset
    from rqvae.img_datasets.lsun import LSUNClass
    from rqvae.img_datasets.transforms import create_transforms
    from src.original_rq_training import (atomic_json, file_sha256, state_sha256,
        load_tokenizer, load_stage2_config, IndexedImages)

    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    assert world == 2
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.25, device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    dist.init_process_group('nccl', timeout=timedelta(hours=2))
    out = root/'cache'
    if rank == 0:
        out.mkdir(exist_ok=False)
    dist.barrier()
    started = time.time()
    def status(phase, **kwargs):
        if rank == 0:
            record = dict(phase=phase, updated_unix=time.time(), elapsed_seconds=time.time()-started, **kwargs)
            atomic_json(out/'status.json', record)
            print(json.dumps(record), flush=True)
    for name, digest in json.loads((root/'core-manifest.json').read_text()).items():
        assert file_sha256(root/'runtime'/name) == digest, name
    checkpoint = root/'tokenizer/model.pt'
    digest = file_sha256(checkpoint)
    assert digest == 'ba008ec2e192a6d4084a8fd511927a789c68a8459d6e8bfc22122ae02887b800'
    tokenizer, _ = load_tokenizer(checkpoint, root/'tokenizer/config.yaml', device)
    before = state_sha256(tokenizer)
    config = load_stage2_config(root/'runtime/upstream')
    protocol = json.loads((root/'reference/data-protocol.json').read_text())
    assert protocol['verified']
    expected = protocol['cache_stage2_fid']['datasets']
    reference = json.loads((root/'reference/complete.json').read_text())
    assert file_sha256(root/'reference/real-statistics.npz') == reference['sha256']
    assert file_sha256(root/'reference/data-protocol.json') == reference['data_protocol_sha256']
    LSUNClass.valid_categories = [*LSUNClass.valid_categories, 'church_val']
    receipts = {}
    with torch.inference_mode():
        for category, split in [('church', 'train'), ('church_val', 'val')]:
            dataset = LSUNClass('/tmp/laser-sign-data', category,
                create_transforms(config.dataset, split=split))
            assert len(dataset) == expected[category]['images']
            for index, wanted in expected[category]['pixel_probes'].items():
                pixels = dataset[int(index)][0]
                assert hashlib.sha256(pixels.numpy().tobytes()).hexdigest() == wanted, (category,index)
            receipts[category] = dict(images=len(dataset), pixel_probes_verified=len(expected[category]['pixel_probes']),
                key_order_sha256=hashlib.sha256(b'\n'.join(dataset.keys)).hexdigest())
            indices = list(range(rank, len(dataset), world))
            loader = DataLoader(Subset(IndexedImages(dataset), indices), batch_size=32,
                num_workers=4, pin_memory=True, shuffle=False)
            if category == 'church':
                path = out/'latents-fp32.npy'
                if rank == 0:
                    values = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(len(dataset),8,8,256))
                    del values
                dist.barrier()
                values = np.load(path, mmap_mode='r+')
            else:
                heldout = []
            completed = 0
            for xs, positions in loader:
                z = tokenizer.encode(xs.to(device))
                assert z.shape == (len(xs),8,8,256) and z.dtype == torch.float32 and torch.isfinite(z).all()
                if category == 'church':
                    values[positions.numpy()] = z.cpu().numpy()
                else:
                    heldout.append(z.cpu())
                completed += len(xs)
                if completed % 1024 == 0 or completed == len(indices):
                    status('caching', split=category, images_per_rank_done=completed, images_per_rank_total=len(indices))
            if category == 'church':
                values.flush()
                del values
            else:
                torch.save(torch.cat(heldout), out/f'validation-rank{rank}.pt')
            del loader
            dataset.env.close()
            dist.barrier()
        assert state_sha256(tokenizer) == before
    if rank == 0:
        path = out/'latents-fp32.npy'
        values = np.load(path, mmap_mode='r')
        assert all(np.isfinite(part).all() for part in np.array_split(values,128))
        atomic_json(out/'complete.json', dict(images=126227, shape=[126227,8,8,256], dtype='float32', world_size=2,
            checkpoint=str(checkpoint), checkpoint_sha256=digest, config=str(root/'tokenizer/config.yaml'),
            config_sha256=file_sha256(root/'tokenizer/config.yaml'), latent_cache=str(path), cache_sha256=file_sha256(path),
            frozen_state_sha256=before, frozen_state_unchanged=True, stage1_source='released Church RQVAE; no fine-tuning',
            validation_sha256=[file_sha256(out/f'validation-rank{r}.pt') for r in range(world)],
            data_protocol_sha256=file_sha256(root/'reference/data-protocol.json'), data_verification=receipts,
            hard_codes_cached=False, stochastic_codes_recomputed_each_visit=True,
            elapsed_seconds=time.time()-started))
        status('complete', images=126227, validation_images=300, frozen_state_unchanged=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
