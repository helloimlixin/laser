#!/usr/bin/env python3
"""Cache an independent, unchanged-prior sample set for paired repair testing."""
import argparse
import codecs
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_church_bar_coefficients import make_model, LaserAux, generation_screen
from scripts.tools.build_sign_probe_cache import sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prior", type=Path, default=ROOT / "outputs/lsun-church-bar-20260910/assets/epoch50-source.pt")
    p.add_argument("--stage1", type=Path, default=ROOT / "outputs/lsun-church-sign-pattern-20260910/assets/best_rfid_slot1_model.pt")
    p.add_argument("--cache", type=Path, default=ROOT / "outputs/lsun-church-bar-20260910/church-cache.pt")
    p.add_argument("--fid-stats", type=Path, default=ROOT / "outputs/lsun-church-bar-20260910/assets/lsun_256_church.npz")
    p.add_argument("--fid-samples", type=int, default=50000)
    p.add_argument("--generation-batch", type=int, default=128)
    p.add_argument("--seed", type=int, default=7701, help="generation_screen adds 10000, giving sample seed 17701")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.serialization.add_safe_globals([codecs.encode])
    cache = torch.load(args.cache, weights_only=False, map_location="cpu")
    aux = LaserAux(args.stage1, 16384, 2048, 20, coeff_scales=cache["meta"]["coeff_scales"], sparsity_level=4).cuda().eval()
    model, metadata = make_model(args.prior, "categorical")
    model.cuda().eval().requires_grad_(False)
    config = {**vars(args), "prior_metadata": metadata, "prior_sha256": sha256_file(args.prior),
              "stage1_sha256": sha256_file(args.stage1), "sample_seed": args.seed + 10000,
              "purpose": "Independent samples, all base weights unchanged; reuse identical codes for each repair checkpoint"}
    (args.output / "config.json").write_text(json.dumps(config, indent=2, default=str))
    def log(row):
        print(json.dumps(row), flush=True)
        (args.output / "status.json").write_text(json.dumps(row, indent=2))
    result = generation_screen(model, aux, args, log)
    (args.output / "results.json").write_text(json.dumps(result, indent=2))
    log({"phase": "complete", **result})


if __name__ == "__main__":
    main()
