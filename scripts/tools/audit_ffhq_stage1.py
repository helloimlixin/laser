#!/usr/bin/env python3
"""Hard-gate an FFHQ LASER a2048/k2 stage-1 checkpoint before stage 2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.train_official_rqtransformer_laser_stage2 import load_stage1_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rfid", type=float, default=6.5)
    args = parser.parse_args()

    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    policy = json.loads(args.policy.read_text())
    selected = next(
        (item for item in policy.get("best", []) if item.get("path") == args.checkpoint.name),
        None,
    )
    payload = load_stage1_checkpoint(args.checkpoint)
    state = payload.get("state_dict", {})
    dictionary = state.get("quantizer.dictionary")
    hparams = config["arch"]["hparams"]
    ddconfig = config["arch"]["ddconfig"]
    gan_loss = config["gan"]["loss"]
    experiment = config["experiment"]
    checks = {
        "checkpoint_exists": args.checkpoint.is_file(),
        "checkpoint_listed_in_policy": selected is not None,
        "rfid_at_most_threshold": selected is not None and float(selected["rfid"]) <= args.max_rfid,
        "late_training_checkpoint": selected is not None and int(selected["epoch"]) >= 140,
        "dictionary_shape_256x2048": dictionary is not None and list(dictionary.shape) == [256, 2048],
        "laser_a2048_k2": (
            hparams.get("bottleneck_type") == "laser"
            and int(hparams.get("n_embed", -1)) == 2048
            and int(hparams.get("sparsity_level", -1)) == 2
            and list(hparams.get("code_shape", [])) == [8, 8, 2]
        ),
        "official_ffhq_backbone": (
            list(ddconfig.get("ch_mult", [])) == [1, 1, 2, 2, 4, 4]
            and int(ddconfig.get("num_res_blocks", -1)) == 2
            and list(ddconfig.get("attn_resolutions", [])) == [16]
        ),
        "full_adversarial_recipe": (
            int(experiment.get("epochs", -1)) == 150
            and float(gan_loss.get("disc_weight", -1)) == 0.75
            and float(gan_loss.get("perceptual_weight", -1)) == 1.0
            and int(gan_loss.get("disc_start", -1)) == 0
        ),
    }
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "checkpoint": str(args.checkpoint.resolve()),
        "epoch": None if selected is None else int(selected["epoch"]),
        "rfid": None if selected is None else float(selected["rfid"]),
        "wandb_run": "https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhq-a2048-k2-rqvae-strict-20260720-145706",
        "fallback_snapshot": "/scratch/xl598/submission_snapshots/laser_ffhq_celebahq_rqvae_strict_dict_sweep_20260720_145706",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise RuntimeError("stage-1 audit failed; stage 2 must not start")


if __name__ == "__main__":
    main()
