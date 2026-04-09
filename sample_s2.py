#!/usr/bin/env python3
"""Sample images from maintained stage-2 checkpoints."""

import argparse
from pathlib import Path

import torch

from src.s2 import load_run, pick_nrow, sample, sample_dir, sample_slide, save_dump, save_grid


def main():
    p = argparse.ArgumentParser(description="Sample from maintained stage-2 checkpoints.")
    p.add_argument("--ckpt", "--checkpoint", "--stage2-checkpoint", dest="checkpoint", type=Path, default=None, help="Stage-2 Lightning checkpoint (.ckpt).")
    p.add_argument("--cache", "--token-cache", "--token_cache", dest="token_cache", type=Path, default=None, help="Token cache used for training.")
    p.add_argument("--out", "--output-dir", "--output_dir", dest="output_dir", type=Path, default=None, help="Where to save sample grids.")
    p.add_argument("--root", "--output-root", "--output_root", dest="output_root", type=Path, default=Path("outputs"), help="Stage-1 output root used to infer checkpoints from token caches.")
    p.add_argument("--ar-dir", "--ar-output-dir", "--ar_output_dir", dest="ar_output_dir", type=Path, default=Path("outputs/ar"), help="Stage-2 output root used when inferring checkpoints and token caches.")
    p.add_argument("-n", "--num-samples", "--num_samples", dest="num_samples", type=int, default=16, help="Number of images to generate.")
    p.add_argument("-b", "--batch-size", "--batch_size", dest="batch_size", type=int, default=0, help="Optional generation batch size. Defaults to num_samples.")
    p.add_argument("--temp", "--temperature", dest="temperature", type=float, default=1.0, help="Sampling temperature.")
    p.add_argument("-k", "--top-k", "--top_k", dest="top_k", type=int, default=0, help="Top-k sampling cutoff; <=0 disables.")
    p.add_argument("--ctemp", "--coeff-temperature", "--coeff_temperature", dest="coeff_temperature", type=float, default=None, help="Optional coefficient sampling temperature for real-valued priors.")
    p.add_argument("--cmode", "--coeff-sample-mode", "--coeff_sample_mode", dest="coeff_sample_mode", type=str, default=None, choices=["gaussian", "mean"], help="Coefficient sampling rule for real-valued priors.")
    p.add_argument("--dev", "--device", dest="device", type=str, default="auto", help="Device string or 'auto'.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--nrow", type=int, default=None, help="Images per row in the output grid.")
    p.add_argument("--dump", "--save-payload", dest="save_payload", action="store_true", help="Also save the generated token payload as samples.pt.")
    p.add_argument("--latent-h", dest="latent_h", type=int, default=0, help="Optional target latent height for sliding-window high-resolution sampling.")
    p.add_argument("--latent-w", dest="latent_w", type=int, default=0, help="Optional target latent width for sliding-window high-resolution sampling.")
    args = p.parse_args()

    run = load_run(
        ckpt=args.checkpoint,
        cache_pt=args.token_cache,
        dev=args.device,
        out_root=args.output_root,
        ar_dir=args.ar_output_dir,
    )
    if run.dev.type == "cuda":
        torch.set_float32_matmul_precision("medium")
        torch.cuda.manual_seed_all(int(args.seed))
    torch.manual_seed(int(args.seed))

    target_hw = None
    if int(args.latent_h) > 0 or int(args.latent_w) > 0:
        if int(args.latent_h) <= 0 or int(args.latent_w) <= 0:
            raise ValueError("--latent-h and --latent-w must both be > 0 when either is set.")
        target_hw = (int(args.latent_h), int(args.latent_w))

    if target_hw is not None and target_hw != run.shape[:2]:
        out = sample_slide(
            run.net,
            run.s1,
            run.shape,
            out_h=target_hw[0],
            out_w=target_hw[1],
            n=int(args.num_samples),
            bs=(None if int(args.batch_size) <= 0 else int(args.batch_size)),
            temp=float(args.temperature),
            top_k=int(args.top_k),
            dev=run.dev,
        )
    else:
        out = sample(
            run.net,
            run.s1,
            run.shape,
            n=int(args.num_samples),
            bs=(None if int(args.batch_size) <= 0 else int(args.batch_size)),
            temp=float(args.temperature),
            top_k=int(args.top_k),
            ctemp=args.coeff_temperature,
            cmode=args.coeff_sample_mode,
            dev=run.dev,
        )

    out_dir = Path(args.output_dir or sample_dir(run.ckpt)).expanduser().resolve()
    raw, auto = save_grid(out.imgs, out_dir, nrow=pick_nrow(int(args.num_samples), args.nrow))
    dump = None
    if args.save_payload:
        dump = save_dump(out, run, out_dir / "samples.pt")

    print(f"Checkpoint: {run.ckpt}")
    print(f"Token cache: {run.cache_pt}")
    print(f"Stage-1 checkpoint: {run.s1.checkpoint_path}")
    if target_hw is not None:
        print(f"Latent shape: {target_hw}")
    print(f"Saved raw grid: {raw}")
    print(f"Saved auto grid: {auto}")
    if dump is not None:
        print(f"Saved sample payload: {dump}")


if __name__ == "__main__":
    main()
