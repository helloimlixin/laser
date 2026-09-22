"""Build a sharded-on-write, complete stochastic VAR trajectory cache on all GPUs."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Subset

from src.training.cli import load_config
from src.training.var_laser import HFSquareImages
from src.models.scratch_var import build_scratch_tokenizer
from src.models.compound_var import compound_decompose
from src.models.sparse_token_codec import token_temperatures, validate_tokenized_checkpoint
from src.data.var_token_cache import FORMAT, restore_cached_codes
from src.original_rq_training import atomic_json, file_sha256


class IndexedImages(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, label = self.images[index]
        return index, image, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--variants', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    if args.variants < 2:
        raise ValueError('A stochastic cache needs at least two complete trajectories')
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise FileExistsError('Completed token cache already exists; refusing to overwrite')
    cfg = load_config(args.config)
    calibration = json.loads((Path(cfg.output_dir) / 'audit/calibration.json').read_text())
    vae = build_scratch_tokenizer(cfg.model, cfg.seed)
    state = torch.load(cfg.compound.tokenizer_checkpoint, weights_only=False, map_location='cpu', mmap=True)
    validate_tokenized_checkpoint(vae.quantize, state)
    vae.load_state_dict(state['model'], strict=True)
    del state
    vae.to(device).eval().requires_grad_(False)
    q = vae.quantize
    if q.tokenized_sparse_policy is not None:
        if (calibration['atom_temperatures'], calibration['coefficient_temperatures']) != token_temperatures(q):
            raise ValueError('Cache temperatures differ from the trained sparse-token policy')
    datasets = {split: HFSquareImages(cfg.data.root, split, False, cfg.seed, 256, resize_crop=False)
                for split in ('train', 'validation')}
    shape_by_split = {split: (len(data), 2 if split == 'train' else 1,
                             args.variants if split == 'train' else 1,
                             sum(p*p for p in q.v_patch_nums), q.sparsity)
                      for split, data in datasets.items()}
    if rank == 0:
        if file_sha256(cfg.compound.tokenizer_checkpoint) != calibration['tokenizer_sha256']:
            raise ValueError('Tokenizer does not match calibrated temperatures')
        for split, shape in shape_by_split.items():
            for name, dtype in [('atoms', 'uint16'), ('coefficients', 'uint16'), ('physical_coefficients', 'float32')]:
                np.lib.format.open_memmap(out / f'{split}-{name}.npy', mode='w+', dtype=dtype, shape=shape).flush()
            np.lib.format.open_memmap(out / f'{split}-labels.npy', mode='w+', dtype='int64', shape=shape[:1]).flush()
    dist.barrier()
    started, visits = time.monotonic(), 0
    checks = []
    generator = torch.Generator(device=device).manual_seed(20260921 + rank)
    with torch.no_grad():
        for split, dataset in datasets.items():
            arrays = {name: np.load(out / f'{split}-{name}.npy', mmap_mode='r+')
                      for name in ('atoms', 'coefficients', 'physical_coefficients', 'labels')}
            loader = DataLoader(Subset(IndexedImages(dataset), range(rank, len(dataset), world)),
                                batch_size=args.batch_size, num_workers=4, pin_memory=True)
            for batch_index, (indices, images, labels) in enumerate(loader):
                indices_np = indices.numpy()
                images = images.to(device, non_blocking=True)
                arrays['labels'][indices_np] = labels.numpy()
                for view in range(shape_by_split[split][1]):
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        latent = vae.quant_conv(vae.encoder(images.flip(-1) if view else images)).float()
                    previous = None
                    for variant in range(shape_by_split[split][2]):
                        codes = compound_decompose(q, latent, stochastic=split == 'train', generator=generator,
                            atom_temperatures=calibration['atom_temperatures'],
                            coefficient_temperatures=calibration['coefficient_temperatures'])
                        for name in ('atoms', 'coefficients', 'physical_coefficients'):
                            arrays[name][indices_np, view, variant] = codes[name].cpu().numpy()
                        if batch_index == 0 and view == 0:
                            restored = restore_cached_codes(q, {name: torch.from_numpy(
                                np.array(arrays[name][indices_np, view, variant])).long() if name != 'physical_coefficients'
                                else torch.from_numpy(np.array(arrays[name][indices_np, view, variant]))
                                for name in ('atoms', 'coefficients', 'physical_coefficients')},
                                calibration['coefficient_temperatures'] if split == 'train' else None)
                            for name in ('atoms', 'coefficients', 'inputs', 'latent', 'coefficient_probabilities'):
                                torch.testing.assert_close(restored[name], codes[name], rtol=0, atol=0)
                            if previous is not None:
                                changes = float((previous != codes['atoms']).any(-1).float().mean())
                                if split == 'train' and changes == 0:
                                    raise RuntimeError('Cached stochastic variants did not change')
                                checks.append(dict(split=split, variant=variant, changed_support_fraction=changes))
                            previous = codes['atoms'].clone()
                            del restored
                        visits += len(images)
                        del codes
                    del latent
                if batch_index % 5 == 0:
                    atomic_json(out / f'progress-rank{rank}.json', dict(split=split, images_done=min((batch_index+1)*args.batch_size, len(loader.dataset)),
                        images_total=len(loader.dataset), tokenized_visits=visits, elapsed_seconds=time.monotonic()-started))
            for array in arrays.values():
                array.flush()
    atomic_json(out / f'verification-rank{rank}.json', dict(exact_cached_context_and_soft_target_roundtrips=True,
                                                         variant_checks=checks, tokenized_visits=visits))
    dist.barrier()
    if rank == 0:
        files = {p.name: dict(bytes=p.stat().st_size, sha256=file_sha256(p)) for p in sorted(out.glob('*.npy'))}
        atomic_json(out / 'manifest.json', dict(format=FORMAT, tokenizer_sha256=calibration['tokenizer_sha256'],
                    calibration_sha256=file_sha256(Path(cfg.output_dir) / 'audit/calibration.json'),
                    patch_nums=list(q.v_patch_nums), sparsity=q.sparsity, shapes=shape_by_split,
                    dataset_fingerprints={split: data.dataset._fingerprint for split, data in datasets.items()},
                    views='original and horizontal flip; validation original only', variants=args.variants,
                    selection='one complete multiscale trajectory per image visit', physical_coefficients_dtype='float32',
                    coefficient_temperatures=calibration['coefficient_temperatures'], files=files,
                    builder_world_size=world, builder_batch_size=args.batch_size,
                    builder_seed=20260921, elapsed_seconds=time.monotonic()-started))
        print(json.dumps(dict(complete=True, directory=str(out), seconds=time.monotonic()-started)), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
