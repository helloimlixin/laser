#!/usr/bin/env python3
"""Apply a saved repair layer to saved AR tokens with paired image/FID checks."""
import argparse
import codecs
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from archive.scripts.train_church_latent_repair import generation_evaluation, write_json
from archive.scripts.train_church_bar_coefficients import LaserAux
from scripts.tools.build_sign_probe_cache import sha256_file
from src.models.lpips import LPIPS
from src.sparse_latent_repair import SparseLatentRepair


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--codes", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    if args.batch_size < 2 or args.batch_size % 2:
        p.error("Batch size must be even and at least two")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.manual_seed(6801)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    torch.serialization.add_safe_globals([codecs.encode])
    payload = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    config = payload["config"]
    assert sha256_file(config["stage1"]) == config["stage1_sha256"]
    cache = torch.load(config["cache"], weights_only=False, map_location="cpu")
    aux = LaserAux(config["stage1"], 16384, 2048, 20, coeff_scales=cache["meta"]["coeff_scales"], sparsity_level=4).cuda().eval()
    del cache
    repair = SparseLatentRepair(**config["architecture"]).cuda().eval().requires_grad_(False)
    repair.load_state_dict(payload["state_dict"], strict=True)
    perceptual = LPIPS().cuda().eval()
    codes = torch.load(args.codes, weights_only=True, map_location="cpu")
    if args.limit:
        codes = {key: value[:args.limit] for key, value in codes.items()}
    if len(codes["atoms"]) < 2 or len(codes["atoms"]) % 2:
        p.error("Use an even number of samples, at least two")
    def log(row):
        print(json.dumps(row), flush=True)
    result = generation_evaluation(repair, aux, perceptual, codes, config["fid_stats"], args.output, args.batch_size, log)
    write_json(args.output / "provenance.json", {"checkpoint": args.checkpoint, "checkpoint_sha256": sha256_file(args.checkpoint),
               "step": payload["step"], "source_codes": args.codes, "source_codes_sha256": sha256_file(args.codes),
               "samples": len(codes["atoms"]), "metrics": result})


if __name__ == "__main__":
    main()
