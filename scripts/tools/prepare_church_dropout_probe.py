#!/usr/bin/env python3
"""Build fixed fresh-OMP train/validation diagnostics for the restored Church recipe."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    sys.path.insert(0, str(args.root / 'earlier-source/runtime'))
    import lmdb
    import numpy as np
    from PIL import Image
    import torch
    from src.training.rqtransformer import LaserAux, val_image_transform
    from src.stochastic_compound import stochastic_omp
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(2026092425)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    cache = torch.load(args.root / 'assets/compound-cache.pt', weights_only=True, mmap=True)
    meta = cache['meta']
    aux = LaserAux(Path('/mnt/laser-church/assets/tokenizer.pt'), 16384, 2048,
                   meta['coeff_max'], 1., coeff_scales=[1.]*4,
                   coeff_bin_centers=meta['coeff_bin_centers'], soft_target_physical=True,
                   clamp_coeffs=False, sparsity_level=4, attn_resolutions=(8,)).to(device).eval()
    transform = val_image_transform()
    gram = aux.dictionary.T @ aux.dictionary
    datasets, alignment = {}, []
    started = time.monotonic()
    with torch.inference_mode():
        for split in ['train', 'val']:
            path = Path('/workspace/Projects/data/lsun/lmdb') / f'church_outdoor_{split}_lmdb'
            env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False)
            if (path / 'keys.npy').exists():
                keys = np.load(path / 'keys.npy', allow_pickle=False)
            else:
                with env.begin() as txn:
                    keys = sorted(key for key, _ in txn.cursor())
            assert len(keys) == (126227 if split == 'train' else 300)
            indices = np.random.default_rng(2026092425).choice(len(keys), 300, replace=False).tolist()
            parts = {'atoms': [], 'coefficients': [], 'input_coefficient_ids': []}
            for start in range(0, len(indices), 8):
                rows = indices[start:start+8]
                with env.begin() as txn:
                    images = torch.stack([transform(Image.open(io.BytesIO(txn.get(bytes(keys[i])))).convert('RGB'))
                                          for i in rows]).to(device)
                latents = aux.quant_conv(aux.encoder(images)).permute(0, 2, 3, 1).contiguous().float()
                result = stochastic_omp(latents, aux.dictionary, depth=4, temperature=.0625, gram=gram)
                atoms, coefficients = result['atoms'], result['coefficients']
                ids, _ = aux.compound_coeff_ids(coefficients, temp=.125, stochastic=True)
                assert torch.isfinite(coefficients).all()
                if split == 'train' and start < 16:
                    cached_a = cache['atoms'][rows, :, :, 0].to(device).long()
                    cached_c = cache['coeffs'][rows, :, :, 0].to(device)
                    vectors = aux.dictionary.T[cached_a]
                    residual = latents - (vectors * cached_c[..., None]).sum(-2)
                    normal_error = (vectors * residual[..., None, :]).sum(-1).abs().max().item()
                    assert normal_error < 1e-4, normal_error
                    alignment.append({'indices': rows, 'normal_equation_max_abs_error': normal_error})
                parts['atoms'].append(atoms.cpu().to(torch.int16))
                parts['coefficients'].append(coefficients.cpu())
                parts['input_coefficient_ids'].append(ids.cpu().to(torch.int16))
            env.close()
            name = 'train_fresh' if split == 'train' else 'validation_fresh'
            datasets[name] = {k: torch.cat(v) for k, v in parts.items()}
            datasets[name]['indices'] = torch.tensor(indices)
            print(json.dumps({'split': split, 'images': len(indices), 'elapsed_seconds': time.monotonic()-started}), flush=True)
    output = args.root / 'assets/heldout-probe.pt'
    torch.save({'datasets': datasets, 'seed': 2026092425, 'atom_temperature': .0625,
                'coefficient_temperature': .125, 'stage1_sha256': meta['stage1_checkpoint_sha256']}, output)
    with output.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = {'passed': True, 'path': str(output), 'sha256': digest,
              'train_images': 300, 'validation_images': 300, 'training_images_removed': 0,
              'fixed_fresh_supports_and_coefficient_histories': True,
              'precision': 'FP32 encoder/OMP; TF32 disabled', 'cache_alignment': alignment,
              'elapsed_seconds': time.monotonic()-started}
    (args.root / 'assets/heldout-probe.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
