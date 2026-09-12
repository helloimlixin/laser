#!/usr/bin/env python3
"""Generate paired corrupted codes using a frozen AR prior and clean anchors."""
import argparse
import codecs
import json
from pathlib import Path
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_church_bar_coefficients import make_model, quantized_splits, LaserAux, atomic_torch_save
from scripts.tools.build_sign_probe_cache import sha256_file
from src.sparse_latent_repair import sample_anchored_span


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache", type=Path, default=ROOT / "outputs/lsun-church-bar-20260910/church-cache.pt")
    p.add_argument("--stage1", type=Path, default=ROOT / "outputs/lsun-church-sign-pattern-20260910/assets/best_rfid_slot1_model.pt")
    p.add_argument("--prior", type=Path, default=ROOT / "outputs/lsun-church-bar-20260910/assets/epoch50-source.pt")
    p.add_argument("--train-images", type=int, default=8192)
    p.add_argument("--holdout-images", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=6701)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.serialization.add_safe_globals([codecs.encode])
    cache = torch.load(args.cache, weights_only=False, map_location="cpu")
    splits = quantized_splits(cache)
    tokenizer_hash = sha256_file(args.stage1)
    assert tokenizer_hash == cache["meta"]["checkpoint_sha256"]
    model, prior_meta = make_model(args.prior, "categorical")
    model.cuda().eval().requires_grad_(False)
    aux = LaserAux(args.stage1, 16384, 2048, 20, coeff_scales=cache["meta"]["coeff_scales"], sparsity_level=4).cuda().eval()
    config = {**vars(args), "tokenizer_sha256": tokenizer_hash, "prior_sha256": sha256_file(args.prior),
              "prior_metadata": prior_meta, "script_sha256": sha256_file(Path(__file__)),
              "helper_sha256": sha256_file(ROOT / "src/sparse_latent_repair.py"),
              "corruption": "Alternating full-pair and coefficient-only AR spans of 4/8/16 sites; clean exterior",
              "trainable_parameters": 0}
    (args.output / "config.json").write_text(json.dumps(config, indent=2, default=str))
    started = time.monotonic()
    rng = torch.Generator().manual_seed(args.seed)
    for split, count in [("train", args.train_images), ("holdout", args.holdout_images)]:
        if not 0 < count <= len(splits[split]):
            raise ValueError("Invalid pair count")
        selected = torch.randperm(len(splits[split]), generator=rng)[:count]
        all_codes, all_starts, all_lengths, all_modes = [], [], [], []
        for batch_id, start in enumerate(range(0, count, args.batch_size)):
            truth = splits[split][selected[start:start + args.batch_size]].cuda()
            span = [4, 8, 16][batch_id % 3]
            first = int(torch.randint(0, 65 - span, (), generator=rng))
            full_pair = batch_id % 2 == 0
            uniforms = torch.rand((len(truth), span * 4, 2), generator=rng).cuda()
            corrupted = sample_anchored_span(model, aux, truth, first, span, uniforms, sample_atoms=full_pair)
            mask = torch.ones(64 * 4, dtype=torch.bool, device="cuda")
            mask[first * 4:(first + span) * 4] = False
            assert torch.equal(corrupted.flatten(1)[:, mask], truth.flatten(1)[:, mask])
            all_codes.append(corrupted.cpu().int())
            all_starts.append(torch.full((len(truth),), first))
            all_lengths.append(torch.full((len(truth),), span))
            all_modes.append(torch.full((len(truth),), full_pair))
            row = {"split": split, "images": start + len(truth), "total": count,
                   "seconds": time.monotonic() - started, "span_sites": span, "full_pair": full_pair,
                   "changed_fraction": float((corrupted != truth).float().mean())}
            print(json.dumps(row), flush=True)
            (args.output / "status.json").write_text(json.dumps(row, indent=2))
        payload = {"indices": selected, "corrupted": torch.cat(all_codes), "start_site": torch.cat(all_starts),
                   "span_sites": torch.cat(all_lengths), "full_pair": torch.cat(all_modes), "config": config,
                   "keys": [cache[split]["keys"][i] for i in selected.tolist()]}
        atomic_torch_save(payload, args.output / f"{split}.pt")
    (args.output / "complete.json").write_text(json.dumps({"seconds": time.monotonic() - started, "trainable_parameters": 0}))


if __name__ == "__main__":
    main()
