"""Matched native stochastic-teacher reconstruction evaluation; no training writes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/workspace/Projects/laser')
BASE = Path('/tmp/laser-rq24576-native-8h100-20261108')
WORK = Path('/tmp/laser-tokenizer-teacher-fidelity-20261110')
OUT = ROOT / 'outputs/tokenizer-teacher-fidelity-20261110'
sys.path[:0] = [str(ROOT), str(BASE)]
os.environ['RQ_UPSTREAM_ROOT'] = str(BASE / 'upstream')
os.environ['TORCH_HOME'] = str(BASE / 'torch-cache')

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from src import rq_upstream, rqvae_metrics as metrics
from src.training.rq_imagenet_runtime import TarImageNet
from runtime.tokenizer import NativeVectorTokenizer
from rqvae.img_datasets.transforms import create_transforms
from rqvae.models import create_model
from rqvae.utils.config import augment_arch_defaults, load_config


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(value, indent=2) + '\n')
    pending.replace(path)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world', type=int, default=8)
    parser.add_argument('--limit', type=int, default=50000)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--temperatures', type=float, nargs='+', default=[0., .25, .1, .0625])
    parser.add_argument('--without-original', action='store_true')
    parser.add_argument('--memory-gib', type=float, default=3.5)
    parser.add_argument('--tag', default='full50k')
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(args.memory_gib * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('highest')
    device = torch.device('cuda:0')
    aux = NativeVectorTokenizer(BASE / 'inputs/stage1-model.pt', BASE / 'inputs/joint-codebook.pt').to(device)
    original = None
    if not args.without_original:
        source = WORK / 'original-rq/stage1'
        original_config = load_config(source / 'config.yaml')
        original, _ = create_model(augment_arch_defaults(original_config.arch), ema=False)
        original.load_state_dict(torch.load(source / 'model.pt', map_location='cpu', weights_only=False)['state_dict'], strict=True)
        original.requires_grad_(False).eval().to(device)
        assert original.quantizer.shared_codebook
        assert tuple(original.code_shape) == (8, 8, 4)
    metrics.INCEPTION_PATH = BASE / 'upstream/rqvae/metrics/inception.py'
    inception = metrics.OriginalRQVAEInception().to(device).eval().requires_grad_(False)
    config = load_config(BASE / 'inputs/stage2-training-config.yaml')
    data = TarImageNet(BASE / 'data', 'val', create_transforms(config.dataset, split='val'), seed=261110)
    assert len(data) == 50000
    assert np.array_equal(np.bincount(data.labels), np.full(1000, 50))
    indices = list(range(args.rank, args.limit, args.world))
    loader = DataLoader(Subset(data, indices), batch_size=args.batch_size, num_workers=2,
                        pin_memory=True, persistent_workers=True, multiprocessing_context='spawn')
    variants = {f'laser_tau{t:g}': ('laser', t) for t in args.temperatures}
    if original is not None:
        variants.update(rq_original_greedy=('original', 0.), rq_original_tau05=('original', .5))
    directory = WORK / args.tag
    directory.mkdir(parents=True, exist_ok=True)
    features = {name: np.lib.format.open_memmap(directory / f'{name}-rank{args.rank}.npy', mode='w+', dtype=np.float32, shape=(len(indices), 2048)) for name in ['real', *variants]}
    # Independent streams keep each temperature reproducible when candidates change.
    seeds = {name: 261110000 + args.rank * 100 + index for index, name in enumerate(variants)}
    rng = {}
    for name, seed in seeds.items():
        torch.cuda.manual_seed(seed)
        rng[name] = torch.cuda.get_rng_state()
    statistics = {name: torch.zeros(10, dtype=torch.float64, device=device) for name in variants}
    seen = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True).float()
        latent = aux.encode(images)
        hard = aux.quantize_reconstruction(latent)['codes']
        old_latent = original.encode(images) if original is not None else None
        old_hard = original.quantizer.quantize(old_latent)[1] if original is not None else None
        real, _ = inception(images.mul(.5).add(.5).clamp(0, 1))
        features['real'][seen:seen + len(images)] = real.cpu().numpy()
        for name, (kind, temperature) in variants.items():
            model = aux if kind == 'laser' else original
            x, greedy = (latent, hard) if kind == 'laser' else (old_latent, old_hard)
            if temperature:
                torch.cuda.set_rng_state(rng[name])
                targets, codes = model.quantizer.get_soft_codes(x, temp=temperature, stochastic=True)
                rng[name] = torch.cuda.get_rng_state()
                entropy = -(targets * targets.clamp_min(1e-30).log()).sum(-1)
                statistics[name][2:6] += entropy.sum((0, 1, 2)).double()
                del targets, entropy
            else:
                codes = greedy
            quantized = model.quantizer.embed_code(codes)
            statistics[name][0] += (x - quantized).square().sum().double()
            statistics[name][1] += x.numel()
            statistics[name][6:10] += (codes != greedy).sum((0, 1, 2)).double()
            fake = model.decode(quantized).float().mul(.5).add(.5).clamp(0, 1)
            value, _ = inception(fake)
            features[name][seen:seen + len(images)] = value.cpu().numpy()
        seen += len(images)
        if seen % 64 == 0 or seen == len(indices):
            progress = dict(rank=args.rank, images=seen, target=len(indices), tag=args.tag,
                            seconds=time.monotonic() - started,
                            peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3)
            write(OUT / args.tag / f'progress-rank{args.rank}.json', progress)
            print(json.dumps(progress), flush=True)
    assert seen == len(indices)
    for value in features.values():
        value.flush()
    report = dict(rank=args.rank, images=seen, image_indices=indices, tag=args.tag,
                  native_codebook_sha256=hashlib.file_digest((BASE / 'inputs/joint-codebook.pt').open('rb'), 'sha256').hexdigest(),
                  original_checkpoint_sha256=(hashlib.file_digest((WORK / 'original-rq/stage1/model.pt').open('rb'), 'sha256').hexdigest() if original is not None else None),
                  statistics={name: value.cpu().tolist() for name, value in statistics.items()},
                  seeds=seeds, variants={name: [kind, temperature] for name, (kind, temperature) in variants.items()},
                  feature_directory=str(directory), seconds=time.monotonic() - started,
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                  training_state_modified=False)
    write(OUT / args.tag / f'result-rank{args.rank}.json', report)
    print(json.dumps(dict(completed=True, rank=args.rank, images=seen, seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    main()
