"""Compatibility sampling wrapper with a shorter filename."""

import argparse
from pathlib import Path

import torch

from src.s2 import gen_dir, load_run, pick_nrow, sample, sample_slide, save_dump, save_grid


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample images from a trained sparse-token prior.")
    p.add_argument("--s1", "--stage1-checkpoint", dest="stage1_checkpoint", type=Path, default=None, help="Path to the LASER Lightning checkpoint.")
    p.add_argument("--s2", "--stage2-checkpoint", dest="stage2_checkpoint", type=Path, default=None, help="Path to the sparse prior Lightning checkpoint.")
    p.add_argument("--cache", "--token-cache", dest="token_cache", type=Path, default=None, help="Path to the stage-2 token cache used for training.")
    p.add_argument("--out", "--output-dir", dest="output_dir", type=Path, default=None, help="Directory to write samples into.")
    p.add_argument("-n", "--num-samples", dest="num_samples", type=int, default=64)
    p.add_argument("-b", "--batch-size", dest="batch_size", type=int, default=16)
    p.add_argument("--temp", "--temperature", dest="temperature", type=float, default=1.0)
    p.add_argument("-k", "--top-k", dest="top_k", type=int, default=0, help="0 disables top-k truncation.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dev", "--device", dest="device", type=str, default="auto", help="'auto', 'cpu', 'gpu', or a CUDA device like 'cuda:0'.")
    p.add_argument("--root", "--output-root", dest="output_root", type=Path, default=Path("outputs"))
    p.add_argument("--ar-dir", "--ar-output-dir", dest="ar_output_dir", type=Path, default=Path("outputs/ar"))
    p.add_argument("--arch", "--architecture", dest="architecture", type=str, default="auto", choices=["auto", "spatial_depth", "gpt", "mingpt"], help="Prior architecture override for older checkpoints.")
    p.add_argument("--heads", "--n-heads", dest="n_heads", type=int, default=None, help="Attention head override for older checkpoints.")
    p.add_argument("--drop", "--dropout", dest="dropout", type=float, default=None, help="Dropout override for older checkpoints.")
    p.add_argument("--cmax", "--coeff-max", dest="coeff_max", type=float, default=None, help="Coefficient range override when cache metadata is incomplete.")
    p.add_argument("--cquant", "--coeff-quantization", dest="coeff_quantization", type=str, default=None, help="Coefficient quantization override: 'uniform' or 'mu_law'.")
    p.add_argument("--cmu", "--coeff-mu", dest="coeff_mu", type=float, default=None, help="Mu parameter override for mu-law coefficient bins.")
    p.add_argument("--nrow", type=int, default=0, help="Grid columns for the saved sample sheet. 0 uses sqrt(num_samples).")
    p.add_argument("--ctemp", "--coeff-temperature", dest="coeff_temperature", type=float, default=None, help="Optional override for real-valued coefficient sampling temperature.")
    p.add_argument("--cmode", "--coeff-sample-mode", dest="coeff_sample_mode", type=str, default=None, choices=["gaussian", "mean"], help="Optional override for real-valued coefficient sampling mode.")
    p.add_argument("--latent-h", dest="latent_h", type=int, default=0, help="Optional target latent height for sliding-window high-resolution sampling.")
    p.add_argument("--latent-w", dest="latent_w", type=int, default=0, help="Optional target latent width for sliding-window high-resolution sampling.")
    return p.parse_args()


@torch.no_grad()
def main():
    args = _args()
    torch.manual_seed(int(args.seed))

    run = load_run(
        ckpt=args.stage2_checkpoint,
        cache_pt=args.token_cache,
        s1_ckpt=args.stage1_checkpoint,
        dev=args.device,
        out_root=args.output_root,
        ar_dir=args.ar_output_dir,
        arch=args.architecture,
        heads=args.n_heads,
        drop=args.dropout,
        cmax=args.coeff_max,
        cquant=args.coeff_quantization,
        cmu=args.coeff_mu,
    )
    if run.dev.type == "cuda":
        torch.set_float32_matmul_precision("medium")
        torch.cuda.manual_seed_all(int(args.seed))

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = gen_dir(run.ckpt, ar_dir=args.ar_output_dir)
        print(f"Defaulting output_dir to: {out_dir}")
    out_dir = Path(out_dir).expanduser().resolve()

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
    raw, auto = save_grid(
        out.imgs,
        out_dir,
        stem="samples",
        nrow=pick_nrow(int(args.num_samples), args.nrow),
    )
    dump = save_dump(out, run, out_dir / "samples.pt")

    print(f"Stage-2 checkpoint: {run.ckpt}")
    print(f"Token cache: {run.cache_pt}")
    print(f"Stage-1 checkpoint: {run.s1.checkpoint_path}")
    if target_hw is not None:
        print(f"Latent shape: {target_hw}")
    print(f"Saved raw grid: {raw}")
    print(f"Saved auto grid: {auto}")
    print(f"Saved sample payload: {dump}")


if __name__ == "__main__":
    main()
