#!/usr/bin/env python3
"""Matched full-validation rFID for the shared pair book and its source."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import rq_upstream
from src.models.laser_joint_tokenizer import FrozenJointLaserTokenizer, SharedAtomCoefficientRQ
from src.rqvae_metrics import DistributedOriginalRQVAEMetrics, OriginalRQVAEInception
from src.training.rq_imagenet_runtime import TarImageNet, sha256, write_json
from rqvae.img_datasets.transforms import create_transforms


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--variants', type=Path,
                        help='JSON list of named shared pair books to compare on identical images')
    args = parser.parse_args()
    config = json.loads((args.output / 'recipe.json').read_text())
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(minutes=30))
    started = time.monotonic()
    candidates = json.loads(args.variants.read_text()) if args.variants else None
    book_path = Path(candidates[0]['path']) if candidates else args.output / 'joint-codebook.pt'
    aux = FrozenJointLaserTokenizer(args.checkpoint, book_path).to(device)
    if candidates:
        quantizers = {}
        for candidate in candidates:
            spec = torch.load(candidate['path'], map_location='cpu', weights_only=True)
            quantizers[candidate['name']] = SharedAtomCoefficientRQ(spec['dictionary'], spec['atom_ids'],
                spec['coefficients'], spec['depth']).to(device)
    else:
        spec = torch.load(config['source_codebook'], map_location='cpu', weights_only=True)
        baseline = SharedAtomCoefficientRQ(spec['dictionary'], spec['atom_ids'],
                                          spec['coefficients'], config['depth']).to(device)
        quantizers = {'shared20481': baseline, 'shared24577': aux.quantizer}
    network = OriginalRQVAEInception().to(device).eval().requires_grad_(False)
    metrics = {name: DistributedOriginalRQVAEMetrics(device, inception=network) for name in quantizers}
    versions = [v._version for v in aux.state_dict().values()]
    transforms = create_transforms(OmegaConf.create(dict(transforms=dict(type='imagenet256x256'))), split='val')
    data = TarImageNet(args.data, 'val', transforms)
    assert len(data) == 50000 and np.array_equal(np.bincount(data.labels), np.full(1000, 50))
    indices = range(rank, len(data), world)
    loader = DataLoader(Subset(data, indices), batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, multiprocessing_context='spawn', persistent_workers=True)
    first_metric = next(iter(metrics.values()))
    seen = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        latent = aux.encode(images)
        first_metric.update(images.mul(.5).add(.5).clamp(0, 1), real=True)
        for name, q in quantizers.items():
            result = q.quantize(latent, chunk_size=512)
            if not seen:
                assert torch.equal(result['quantized'], q.embed(result['codes']).sum(-2))
            reconstructed = aux.decode(result['quantized']).mul(.5).add(.5).clamp(0, 1)
            metrics[name].update(reconstructed, real=False)
            if not seen and rank == 0:
                from torchvision.utils import save_image
                save_image(torch.cat((images[:8].mul(.5).add(.5).clamp(0, 1), reconstructed[:8])),
                           args.output / f'{name}-rfid-preview.png', nrow=8)
        seen += len(images)
        if rank == 0 and (seen % 128 == 0 or seen == len(indices)):
            progress = dict(phase='measuring_validation_rfid', images=seen * world,
                target=50000, elapsed_seconds=time.monotonic() - started, world_size=world)
            write_json(args.output / 'rfid-progress.json', progress)
            print(json.dumps(progress), flush=True)
    assert seen == len(indices)
    assert versions == [v._version for v in aux.state_dict().values()]
    # Copy local real statistics before either metric performs its all-reduce.
    for metric in list(metrics.values())[1:]:
        metric.real_sum.copy_(first_metric.real_sum)
        metric.real_cross.copy_(first_metric.real_cross)
        metric.real_count.copy_(first_metric.real_count)
    results = {}
    for name, metric in metrics.items():
        fid, _, _ = metric.compute()
        assert int(metric.real_count) == int(metric.fake_count) == 50000
        results[name] = dict(rfid=fid, vocabulary=quantizers[name].vocab_size,
                             depth=quantizers[name].depth, tokens_per_image=64 * quantizers[name].depth)
    if rank == 0:
        report = dict(completed=True, real_images=50000, reconstructed_images_per_candidate=50000,
            results=results, metric='original RQ-VAE TensorFlow-FID-compatible Inception',
            split='official ImageNet2012 validation', preprocessing=str(transforms),
            validation_used_for_fitting=False, shared_physical_pairs_at_every_depth=True,
            source_stage1_checkpoint_sha256=config['stage1_checkpoint_sha256'],
            codebook_sha256=sha256(book_path), encoder_decoder_frozen=True,
            seconds=time.monotonic() - started, stage2_training_started=False)
        # Reproduce the previously measured shared baseline as an independent check.
        control_name = 'baseline24577_d4' if candidates else 'shared20481'
        control_rfid = 4.871456618067668 if candidates else 4.96756247210908
        report['baseline_reproduction_error'] = abs(results[control_name]['rfid'] - control_rfid)
        report['baseline_reproduced'] = report['baseline_reproduction_error'] < .02
        if candidates:
            report['codebook_hashes'] = {row['name']: sha256(row['path']) for row in candidates}
            report['candidate_books'] = candidates
        write_json(args.output / 'full-rfid.json', report)
        print(json.dumps(report), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
