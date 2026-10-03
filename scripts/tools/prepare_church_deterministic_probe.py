#!/usr/bin/env python3
"""Build a common deterministic OMP probe and validate cached train examples."""
import hashlib
import io
import json
from pathlib import Path
import sys
import time

ROOT = Path('/mnt/laser-church/normalized-noise-comparison')
sys.path.insert(0, str(ROOT/'runtime'))
import lmdb
import numpy as np
from PIL import Image
import torch
from src.training.rqtransformer import LaserAux, val_image_transform


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(2026092425)
    cache = torch.load('/mnt/laser-church/assets/compound-cache.pt', weights_only=True, mmap=True)
    meta = cache['meta']
    aux = LaserAux(Path('/mnt/laser-church/assets/tokenizer.pt'), 16384, 2048, 3., 1.,
        coeff_scales=meta['coeff_scales'], soft_target_physical=False, clamp_coeffs=False,
        sparsity_level=4, attn_resolutions=(8,)).cuda().eval()
    datasets, alignment = {}, []
    transform = val_image_transform()
    started = time.monotonic()
    for split in ['train', 'val']:
        path = Path('/workspace/Projects/data/lsun/lmdb')/f'church_outdoor_{split}_lmdb'
        env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False)
        if (path/'keys.npy').exists():
            keys = np.load(path/'keys.npy', allow_pickle=False)
        else:
            with env.begin() as txn:
                keys = sorted(key for key, _ in txn.cursor())
        assert len(keys) == (126227 if split == 'train' else 300)
        indices = np.random.default_rng(2026092425).choice(len(keys), 300, replace=False).tolist()
        parts = {'atoms': [], 'coefficients': [], 'input_coefficient_ids': []}
        for start in range(0, 300, 8):
            rows = indices[start:start+8]
            if split == 'train' and start >= 32:
                atoms, coefficients = cache['atoms'][rows].cuda().long(), cache['coeffs'][rows].cuda()
            else:
                with env.begin() as txn:
                    images = torch.stack([transform(Image.open(io.BytesIO(txn.get(bytes(keys[i])))).convert('RGB')) for i in rows]).cuda()
                atoms, coefficients = aux.encode_sparse_components(images)
                if split == 'train':
                    expected_a, expected_c = cache['atoms'][rows].cuda().long(), cache['coeffs'][rows].cuda()
                    assert torch.equal(atoms, expected_a), 'Fresh OMP support disagrees with cache'
                    error = (coefficients-expected_c).abs().max().item()
                    assert error < 1e-5, error
                    alignment.append(dict(indices=rows, atoms_exact=True, coefficient_max_abs_error=error))
            assert torch.isfinite(coefficients).all()
            assert not (atoms.sort(-1).values[..., 1:] == atoms.sort(-1).values[..., :-1]).any()
            ids, _ = aux.compound_coeff_ids(coefficients, stochastic=False, temp=.25)
            for key, value in [('atoms', atoms.to(torch.int16)), ('coefficients', coefficients), ('input_coefficient_ids', ids.to(torch.int16))]:
                parts[key].append(value.cpu())
        env.close()
        name = 'train_fresh' if split == 'train' else 'validation_fresh'
        datasets[name] = {key: torch.cat(values) for key, values in parts.items()}
        datasets[name]['indices'] = torch.tensor(indices)
        print(json.dumps(dict(split=split, images=300, elapsed_seconds=time.monotonic()-started)), flush=True)
    output = ROOT/'heldout-probe.pt'
    torch.save(dict(datasets=datasets, seed=2026092425, atom_temperature=0., coefficient_history='nearest normalized bin; identical across variants',
                    stage1_sha256=meta['stage1_checkpoint_sha256']), output)
    with output.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = dict(passed=True, path=str(output), sha256=digest, train_images=300, validation_images=300,
                  training_images_removed=0, train_source='Cached deterministic OMP with 32 fresh image checks',
                  validation_source='Fresh deterministic OMP for all 300 held-out images',
                  fixed_coefficient_history='nearest normalized bin', precision='FP32, TF32 disabled',
                  cache_alignment=alignment, elapsed_seconds=time.monotonic()-started)
    (ROOT/'heldout-probe.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
