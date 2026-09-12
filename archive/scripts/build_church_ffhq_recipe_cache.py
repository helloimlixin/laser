#!/usr/bin/env python3
"""Re-encode frozen Church latents, preserving continuous OMP coefficients."""
import argparse
import codecs
import json
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.tools.build_sign_probe_cache import ChurchImages, sha256_file
from src.training.rqtransformer import LaserAux, atomic_torch_save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split-cache', type=Path, default=ROOT / 'outputs/lsun-church-bar-20260910/church-cache.pt')
    p.add_argument('--stage1', type=Path, default=ROOT / 'outputs/lsun-church-sign-pattern-20260910/assets/best_rfid_slot1_model.pt')
    p.add_argument('--data', type=Path, default=Path('/tmp/laser-sign-data/church'))
    p.add_argument('--batch-size', type=int, default=64)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(8)
    torch.serialization.add_safe_globals([codecs.encode])
    torch.backends.cuda.matmul.allow_tf32 = True
    old = torch.load(args.split_cache, map_location='cpu', weights_only=False)
    checkpoint_hash = sha256_file(args.stage1)
    assert old['meta']['checkpoint_sha256'] == checkpoint_hash
    aux = LaserAux(args.stage1, 16384, 2048, 3, coeff_scales=[1.] * 4,
                   sparsity_level=4, clamp_coeffs=False).cuda().eval()
    result = {'dictionary': aux.dictionary.cpu(), 'meta': {
        'format': 'church_ffhq_continuous_v1', 'checkpoint_sha256': checkpoint_hash,
        'split_cache_sha256': sha256_file(args.split_cache),
        'precision': 'BF16 encoder; FP32 OMP; FP32 physical coefficients',
        'transform': 'Resize(256), CenterCrop(256), normalize [-1,1]',
        'coeff_max': 3., 'coeff_vocab_size': 2048,
        'scale_fit': 'per-depth train absolute maximum / 3; no holdout or validation fitting',
        'soft_targets': 'FFHQ v4: exp(-(normalized coefficient - bin)^2 / 0.5)',
        'split_key_overlap': 0,
    }}
    keys = [set(old[s]['keys']) for s in ('train', 'holdout', 'validation')]
    assert not any(keys[i] & keys[j] for i in range(3) for j in range(i))
    start = time.monotonic()
    with torch.inference_mode():
        for split in ('train', 'holdout', 'validation'):
            path = args.data / ('church_outdoor_val_lmdb' if split == 'validation' else 'church_outdoor_train_lmdb')
            dataset = ChurchImages(path)
            dataset.keys = [key.encode('ascii') for key in old[split]['keys']]
            loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8,
                                pin_memory=True, shuffle=False)
            atoms, coeffs = [], []
            for step, images in enumerate(loader):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    a, c = aux.encode_sparse_components(images.cuda(non_blocking=True))
                assert torch.isfinite(c).all()
                atoms.append(a.cpu().short())
                coeffs.append(c.cpu().float())
                if step % 20 == 0:
                    print(json.dumps({'split': split, 'encoded': sum(map(len, atoms)),
                                      'total': len(dataset), 'seconds': time.monotonic() - start}), flush=True)
            result[split] = {'atoms': torch.cat(atoms), 'coefficients': torch.cat(coeffs),
                             'keys': old[split]['keys']}
    maximum = result['train']['coefficients'].abs().amax((0, 1, 2))
    assert (maximum > 0).all()
    result['meta']['coeff_scales'] = (maximum / 3).tolist()
    result['meta']['train_absolute_max'] = maximum.tolist()
    result['meta']['out_of_train_range_fraction'] = {
        s: (result[s]['coefficients'].abs() > maximum).float().mean((0, 1, 2)).tolist()
        for s in ('train', 'holdout', 'validation')}
    atomic_torch_save(result, args.output)
    args.output.with_suffix('.json').write_text(json.dumps(result['meta'], indent=2))
    print(json.dumps({'phase': 'complete', 'path': str(args.output), **result['meta']}), flush=True)


if __name__ == '__main__':
    main()
