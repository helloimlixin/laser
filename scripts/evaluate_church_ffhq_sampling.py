#!/usr/bin/env python3
"""Bounded sampling-only comparison for a frozen Church FFHQ-recipe checkpoint."""
import argparse
import codecs
import json
from pathlib import Path
import sys
import time

import torch
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_official_rqtransformer_laser_stage2 import LaserAux, atomic_torch_save
from scripts.tools.build_sign_probe_cache import sha256_file
from src.church_ffhq_recipe import make_prior
from src.rqvae_metrics import DistributedOriginalRQVAEMetrics


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--atom-top-k', type=int, default=250)
    p.add_argument('--coeff-top-p', type=float, default=.85)
    p.add_argument('--coeff-temperature', type=float, default=1.)
    p.add_argument('--samples', type=int, default=4096)
    p.add_argument('--seed', type=int, default=15701)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.serialization.add_safe_globals([codecs.encode])
    torch.backends.cuda.matmul.allow_tf32 = True
    saved = torch.load(args.checkpoint, weights_only=False, map_location='cpu')
    cfg = saved['config']
    model = make_prior(cfg['architecture']).cuda().eval().requires_grad_(False)
    model.load_state_dict(saved['state_dict'], strict=True)
    aux = LaserAux(Path(cfg['stage1']), 16384, 2048, 3., coeff_scales=cfg['tokenizer']['coeff_scales'],
                   sparsity_level=4, soft_target_physical=False).cuda().eval().requires_grad_(False)
    metadata = {'checkpoint': str(args.checkpoint), 'checkpoint_sha256': sha256_file(args.checkpoint),
                'checkpoint_epoch': saved['epoch'], 'architecture': cfg['architecture'],
                'atom_top_k': args.atom_top_k, 'coeff_top_p': args.coeff_top_p,
                'coeff_temperature': args.coeff_temperature, 'atom_temperature': 1.,
                'samples': args.samples, 'seed': args.seed,
                'weights_changed': False, 'precision': 'BF16 AR; FP32 decoder; original RQ-VAE Inception'}
    del saved
    metric = DistributedOriginalRQVAEMetrics('cuda', reference_stats_path=Path(cfg['fid_stats']))
    atoms_saved, ids_saved = [], []
    torch.manual_seed(args.seed)
    start = time.monotonic()
    for first in range(0, args.samples, 128):
        n = min(128, args.samples - first)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            atoms, ids = model.sample_compound(n, aux, atom_top_k=args.atom_top_k, atom_top_p=1.,
                coeff_top_k=0, coeff_top_p=args.coeff_top_p, atom_temperature=1.,
                coeff_temperature=args.coeff_temperature, amp=True)
        atoms_saved.append(atoms.cpu().short())
        ids_saved.append(ids.cpu().short())
        for offset in range(0, n, 32):
            images = ((aux.decode_compound(atoms[offset:offset + 32], ids[offset:offset + 32]) + 1) / 2).clamp(0, 1)
            metric.update(images, real=False)
            if first == 0 and offset == 0:
                save_image(images, args.output / 'samples.png', nrow=8)
        if first == 0 or (first // 128) % 8 == 0 or first + n == args.samples:
            print(json.dumps({'generated': first + n, 'seconds': time.monotonic() - start}), flush=True)
    fid, _, _ = metric.compute()
    metadata.update(fid=float(fid), seconds=time.monotonic() - start)
    atomic_torch_save({'atoms': torch.cat(atoms_saved), 'coefficient_ids': torch.cat(ids_saved)}, args.output / 'generated-codes.pt')
    (args.output / 'metrics.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
