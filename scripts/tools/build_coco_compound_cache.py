"""Encode each COCO image once, then retain every aligned training caption."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from omegaconf import OmegaConf
from src.data.coco2014 import COCO2014Dataset
from src.training.cc3m_compound import make_aux
from scripts.tools.build_cc3m_compound_cache import text_tokenizer, write_json


def transform():
    from torchvision import transforms as T
    return T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(256), T.ToTensor(), T.Normalize([.5]*3, [.5]*3)])


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_save(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.partial')
    torch.save(value, temp); temp.replace(path)


def caption_cache(dataset, atoms, coeffs, scales, sha):
    """Input tokens are in sorted unique image order; expand captions afterward."""
    rows = torch.tensor([index for index, _ in dataset.samples])
    captions = [caption for _, caption in dataset.samples]
    tokenizer = text_tokenizer()
    text_ids = torch.tensor([r.ids for r in tokenizer.encode_batch(captions)], dtype=torch.int16)
    coefficients = coeffs[rows] / scales.reshape(1,1,1,4)
    if not torch.isfinite(coefficients).all():
        raise ValueError('Nonfinite coefficients')
    return dict(atoms=atoms[rows], coeffs=coefficients, captions=captions, text_ids=text_ids,
        image_ids=torch.tensor([dataset.images[i]['id'] for i in rows.tolist()]),
        meta=dict(format='laser_coco2014_compound_v1', shape=[8,8,4], stage1_sha256=sha,
            split=dataset.split, caption_mode=dataset.caption_mode, items=len(rows),
            unique_images=len(dataset.images), coeff_scales=scales.tolist(),
            clip_coefficients=False, coefficient_storage='fp32', encoder_precision='fp32',
            transform='PIL_bicubic_resize_short256_center_crop256',
            text_tokenizer='bpe16k_huggingface', text_length=32))


@torch.inference_mode()
def prepare_coco_reference(options, device):
    from src.rqvae_metrics import DistributedOriginalRQVAEMetrics, _mean_covariance
    rank, world = dist.get_rank(), dist.get_world_size()
    dataset = COCO2014Dataset(options['data'], 'val', transform=transform())
    loader = DataLoader(Subset(dataset, range(rank, len(dataset), world)),
        batch_size=32, num_workers=4, pin_memory=True)
    metric = DistributedOriginalRQVAEMetrics(device)
    for images, _ in loader:
        metric.update((images.to(device)+1)*.5, real=True)
    for value in (metric.real_sum, metric.real_cross, metric.real_count):
        dist.all_reduce(value)
    if int(metric.real_count) != options['validation_items']:
        raise ValueError('COCO validation reference count mismatch')
    if rank == 0:
        mu, sigma = _mean_covariance(metric.real_sum, metric.real_cross, int(metric.real_count))
        path = Path(options['fid_reference_stats']); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mu=mu, sigma=sigma)
        write_json(path.with_suffix('.json'), dict(images=int(metric.real_count),
            protocol='official COCO val2014; one real image per ID',
            captions='lowest annotation ID per image for generation',
            annotation_sha256=digest(Path(options['data'])/'annotations/captions_val2014.json'),
            transform='PIL_bicubic_resize_short256_center_crop256', metric='original-rqvae-inception'))
    dist.barrier()
    del metric
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    options = OmegaConf.to_container(OmegaConf.load(args.config).options, resolve=True)
    rank, world, local = [int(os.environ[k]) for k in ('RANK','WORLD_SIZE','LOCAL_RANK')]
    torch.cuda.set_device(local); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl')
    sha = digest(options['checkpoint'])
    if sha != options['stage1_sha256']:
        raise ValueError('Unexpected COCO tokenizer checkpoint')
    aux = make_aux(options, [1.]*4, torch.device('cuda', local))
    for split in ('train','val'):
        dataset = COCO2014Dataset(options['data'], split, transform=transform())
        rows = list(range(rank, len(dataset), world))
        shard = args.output/f'{split}-{rank}.pt'
        receipt = shard.with_suffix('.json')
        if shard.is_file() and receipt.is_file():
            info = json.loads(receipt.read_text())
            if info['stage1_sha256'] != sha or info['world_size'] != world:
                raise ValueError('Cache shard provenance mismatch')
        else:
            loader = DataLoader(Subset(dataset, rows), batch_size=16, num_workers=4, pin_memory=True)
            atoms_parts, coeff_parts = [], []
            with torch.inference_mode():
                for number, (images, _) in enumerate(loader):
                    atoms, coeffs = aux.encode_sparse_components(images.cuda())
                    if atoms.shape[1:] != (8,8,4) or not torch.isfinite(coeffs).all():
                        raise ValueError('Invalid LASER encoding')
                    atoms_parts.append(atoms.cpu().short()); coeff_parts.append(coeffs.cpu().float())
                    if number % 100 == 0:
                        print(json.dumps(dict(split=split, rank=rank, batches=number)), flush=True)
            atomic_save(dict(rows=torch.tensor(rows), atoms=torch.cat(atoms_parts),
                            coeffs=torch.cat(coeff_parts)), shard)
            write_json(receipt, dict(stage1_sha256=sha, world_size=world, items=len(rows)))
        dist.barrier()
        if rank == 0:
            atoms = torch.empty((len(dataset),8,8,4), dtype=torch.int16)
            coeffs = torch.empty((len(dataset),8,8,4), dtype=torch.float32)
            seen = torch.zeros(len(dataset), dtype=torch.bool)
            for worker in range(world):
                data = torch.load(args.output/f'{split}-{worker}.pt', weights_only=True, mmap=True)
                if seen[data['rows']].any():
                    raise ValueError('Duplicate cache rows')
                seen[data['rows']] = True
                atoms[data['rows']], coeffs[data['rows']] = data['atoms'], data['coeffs']
            assert seen.all()
            if split == 'train':
                scales = coeffs.abs().reshape(-1,4).amax(0).clamp_min(1e-8)/3.
                write_json(args.output/'scales.json', scales.tolist())
            else:
                scales = torch.tensor(json.loads((args.output/'scales.json').read_text()))
            expanded = COCO2014Dataset(options['data'], split, caption_mode='all' if split == 'train' else 'first')
            cache = caption_cache(expanded, atoms, coeffs, scales, sha)
            expected = options['train_items' if split == 'train' else 'validation_items']
            if len(cache['atoms']) != expected:
                raise ValueError('Caption cache count mismatch')
            target = Path(options['token_cache' if split == 'train' else 'validation_cache'])
            atomic_save(cache, target)
            persistent = args.output.parent/'assets'/target.name
            persistent.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(target, persistent)
            write_json(persistent.with_suffix('.json'), cache['meta'])
            del cache, atoms, coeffs
        dist.barrier()
    del aux
    torch.cuda.empty_cache()
    prepare_coco_reference(options, torch.device('cuda',local))
    if rank == 0:
        write_json(args.output/'complete.json', dict(passed=True, stage1_sha256=sha,
            train_items=options['train_items'], validation_items=options['validation_items']))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
