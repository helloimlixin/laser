#!/usr/bin/env python3
"""Extract a deterministic Church subset for the oracle joint-sign experiment."""

import argparse
import codecs
import hashlib
import io
import json
from pathlib import Path
import sys

import lmdb
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.training.rqtransformer import LaserAux, val_image_transform


class ChurchImages(Dataset):
    def __init__(self, path, indices=None):
        self.path = str(path)
        with lmdb.open(self.path, readonly=True, lock=False) as env:
            with env.begin() as transaction:
                self.keys = list(transaction.cursor().iternext(keys=True, values=False))
        if indices is not None:
            self.keys = [self.keys[i] for i in indices]
        self.environment = None
        self.transform = val_image_transform()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if self.environment is None:
            self.environment = lmdb.open(self.path, readonly=True, lock=False, readahead=False)
        with self.environment.begin() as transaction:
            raw = transaction.get(self.keys[index])
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        return self.transform(image)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-images", type=int, default=16384)
    parser.add_argument("--holdout-images", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2701)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace {args.output}")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    train_path = args.data / "church_outdoor_train_lmdb"
    validation_path = args.data / "church_outdoor_val_lmdb"
    population = ChurchImages(train_path)
    total = args.train_images + args.holdout_images
    if total > len(population) or min(args.train_images, args.holdout_images) <= 0:
        raise ValueError("invalid training/holdout counts")
    permutation = torch.randperm(len(population), generator=torch.Generator().manual_seed(args.seed))
    splits = {
        "train": ChurchImages(train_path, permutation[:args.train_images].tolist()),
        "holdout": ChurchImages(train_path, permutation[args.train_images:total].tolist()),
        "validation": ChurchImages(validation_path),
    }
    key_sets = [set(data.keys) for data in splits.values()]
    if any(key_sets[i] & key_sets[j] for i in range(3) for j in range(i)):
        raise ValueError("overlapping LMDB keys between splits")
    scales = [1.109375, 0.6234375238418579, 0.3685546815395355, 0.24746093153953552]
    # Older torch pickles encode byte metadata through this standard-library helper.
    torch.serialization.add_safe_globals([codecs.encode])
    aux = LaserAux(args.checkpoint, 16384, 2048, 20, coeff_scales=scales,
                   sparsity_level=4, clamp_coeffs=True).cuda().eval()
    payload = {"dictionary": aux.dictionary.cpu().clone(), "meta": {
        "format": "laser_sign_probe_v1", "seed": args.seed,
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256_file(args.checkpoint),
        "coefficient_units": "physical", "coefficient_target": "nearest of 2048 normalized bins; same depth scales as September 9 pair run",
        "coeff_scales": scales, "coeff_max": 20, "coeff_vocab_size": 2048,
        "transform": "Resize(256), CenterCrop(256), normalize [-1,1]",
        "precision": "BF16 encoder, FP32 OMP, float16 normalized cache round-trip before quantization",
        "training_population": len(population), "split_key_overlap": 0,
        "split_note": "Image-disjoint for these new sign priors; tokenizer trained on the original training population. Official validation was previously used in other experiments.",
    }}
    with torch.inference_mode():
        for name, dataset in splits.items():
            atoms, coefficients = [], []
            loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                                pin_memory=True, shuffle=False)
            for step, images in enumerate(loader):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    a, c = aux.encode_sparse_components(images.cuda(non_blocking=True))
                c = c.half().float()
                ids = (c[..., None] - aux.coeff_bins).abs().argmin(-1)
                physical = aux.coeff_bins[ids] * aux.coeff_scales
                atoms.append(a.cpu().short())
                coefficients.append(physical.cpu())
                if step % 20 == 0:
                    print(json.dumps({"split": name, "encoded": sum(len(v) for v in atoms), "total": len(dataset)}), flush=True)
            payload[name] = {"atoms": torch.cat(atoms), "coefficients": torch.cat(coefficients),
                             "keys": [key.decode("ascii") for key in dataset.keys]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(payload["meta"], indent=2))
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
