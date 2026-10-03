"""Encode an entire ImageNet split with the active TorchMetrics FID network."""
import argparse
from collections import Counter
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import datasets, transforms
import torchmetrics
from torchmetrics.image.fid import FrechetInceptionDistance

ROOT = Path(os.environ.get('LASER_RUNTIME_ROOT', Path(__file__).resolve().parents[2]))
sys.path[:0] = [str(ROOT), str(ROOT / 'runtime')]
from src.training.fid_reference import SCHEMA, load_torchmetrics_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--split', choices=('train', 'val'), default='train')
    args = parser.parse_args()
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', timeout=timedelta(minutes=40))
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256),
        transforms.ToTensor(), transforms.Normalize([.5]*3, [.5]*3)])
    expected_samples = 1281167 if args.split == 'train' else 50000
    dataset = datasets.ImageFolder(args.data / args.split, transform=transform)
    assert len(dataset) == expected_samples and len(dataset.classes) == 1000
    # Strided shards cover every index exactly once, with no padded duplicates.
    indices = range(rank, len(dataset), world)
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, persistent_workers=True,
        prefetch_factor=3)
    metric = FrechetInceptionDistance(feature=2048, normalize=True,
        sync_on_compute=False).to(device).eval()
    started = time.monotonic()
    with torch.inference_mode():
        for batch, (images, _) in enumerate(loader):
            images = ((images.to(device, non_blocking=True).float()+1)*.5).clamp(0, 1)
            metric.update(images, real=True)
            if rank == 0 and (batch+1) % 25 == 0:
                count = int(metric.real_features_num_samples)
                elapsed = time.monotonic()-started
                progress = dict(phase='training_fid_reference', rank=rank,
                    images_on_rank=count, target_on_rank=len(indices),
                    approximate_global_images=min(len(dataset),count*world),
                    approximate_global_images_per_second=count*world/elapsed,
                    elapsed_seconds=elapsed)
                print(json.dumps(progress), flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.with_suffix('.progress.json').write_text(json.dumps(progress,indent=2)+'\n')
    assert int(metric.real_features_num_samples) == len(indices)
    for name in ('real_features_sum', 'real_features_cov_sum', 'real_features_num_samples'):
        dist.all_reduce(getattr(metric, name), op=dist.ReduceOp.SUM)
    assert int(metric.real_features_num_samples) == len(dataset)
    if rank == 0:
        index_digest = hashlib.sha256()
        for filename, target in dataset.samples:
            index_digest.update((str(Path(filename).relative_to(args.data))+'\t'+str(target)+'\n').encode())
        model_digest = hashlib.sha256()
        for name, tensor in metric.inception.state_dict().items():
            model_digest.update(name.encode())
            model_digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        metadata = dict(samples=len(dataset), real_split=args.split, classes=1000,
            images_per_class=dict(Counter(dataset.targets)),
            image_index_sha256=index_digest.hexdigest(),
            source_archive_md5=('1d675b47d978889d74fa0da5fadfb00e' if args.split == 'train'
                                else '29b22e2961454d5413ddabcf34fc5622'),
            inception_state_sha256=model_digest.hexdigest(), feature_dim=2048,
            torchmetrics_version=torchmetrics.__version__, torchvision_version=torchvision.__version__,
            transform='Resize256 CenterCrop256; training FID float normalization then uint8 conversion',
            network_precision='float32', accumulation_precision='float64',
            tf32_enabled=True,
            no_augmented_training_views=True, no_duplicate_indices=True,
            world_size=world, created_unix=time.time(), elapsed_seconds=time.monotonic()-started)
        payload = dict(schema=SCHEMA, feature_dim=2048, metadata=metadata,
            **{name:getattr(metric,name).cpu() for name in ('real_features_sum',
                'real_features_cov_sum','real_features_num_samples')})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix('.tmp')
        torch.save(payload, temporary)
        temporary.replace(args.output)
        load_torchmetrics_reference(args.output, expected_samples=expected_samples)
        with args.output.open('rb') as stream:
            metadata['statistics_sha256'] = hashlib.file_digest(stream,'sha256').hexdigest()
        args.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print(json.dumps(dict(phase='training_fid_reference_complete',
            images=len(dataset), output=str(args.output), **metadata)),flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
