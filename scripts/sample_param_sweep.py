#!/usr/bin/env python3
"""Sampling-parameter sweep for a stage-2 transformer prior.

Loads a checkpoint ONCE and renders a 4x4 grid of 16 samples for every
(temperature, top_k) config, then assembles a labeled contact sheet so configs
can be compared at a glance. For quantized priors (atom + coeff categorical
stream) temperature and top_k are the meaningful knobs.

Grids are always written to --out. Pass --wandb to ALSO log them to a Weights &
Biases run (a sortable table keyed by temp/top_k + the contact sheet image), so
the results are browsable in the W&B UI instead of disk-only.
"""
import argparse
import glob
import os
import re
from pathlib import Path

import torch

from src.s2 import load_run, sample, save_grid


def _floats(s):
    return [float(x) for x in str(s).split(",") if x.strip()]


def _ints(s):
    return [int(x) for x in str(s).split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--s1-ckpt", dest="s1_ckpt", default=None)
    ap.add_argument("--out-root", dest="out_root", required=True,
                    help="Stage-1 output root (e.g. the celebahq dir) used to locate the decoder.")
    ap.add_argument("--out", required=True, help="Directory for the grids + contact sheet.")
    ap.add_argument("-n", type=int, default=16)
    ap.add_argument("--nrow", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temps", default="0.7,0.8,0.9,1.0,1.1")
    ap.add_argument("--topks", default="0,32,64,128,256")
    # W&B logging (optional).
    ap.add_argument("--wandb", action="store_true", help="Also log grids + contact sheet to W&B.")
    ap.add_argument("--wandb-entity", dest="wandb_entity", default=None)
    ap.add_argument("--wandb-project", dest="wandb_project", default="laser")
    ap.add_argument("--wandb-group", dest="wandb_group", default=None)
    ap.add_argument("--wandb-name", dest="wandb_name", default="sampling-sweep")
    args = ap.parse_args()

    temps = _floats(args.temps)
    topks = _ints(args.topks)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    run = load_run(ckpt=args.ckpt, cache_pt=args.cache, s1_ckpt=args.s1_ckpt,
                   out_root=args.out_root, dev="auto")
    print(f"[sweep] loaded ckpt; latent shape={run.shape} dev={run.dev}", flush=True)
    print(f"[sweep] {len(temps)}x{len(topks)} = {len(temps) * len(topks)} configs, n={args.n} per grid", flush=True)

    grids = {}
    for t in temps:
        for k in topks:
            torch.manual_seed(args.seed)  # same draws across configs -> fair comparison
            batch = sample(run.net, run.s1, run.shape, n=args.n, temp=t, top_k=k, dev=run.dev)
            stem = f"grid_temp{t:.2f}_topk{k}"
            grids[(t, k)] = Path(save_grid(batch.imgs, out, stem=stem, nrow=args.nrow))
            print(f"[sweep] saved {stem} -> {grids[(t, k)]}", flush=True)

    sheet = _contact_sheet(out, temps, topks, grids)

    if args.wandb:
        _log_wandb(args, out, temps, topks, grids, sheet)

    print("[sweep] SWEEP DONE", flush=True)


def _contact_sheet(out, temps, topks, grids):
    """Labeled montage (rows=temp, cols=top_k). Returns the path, or None on failure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt

        nr, nc = len(temps), len(topks)
        fig, axes = plt.subplots(nr, nc, figsize=(3.0 * nc, 3.0 * nr), squeeze=False)
        for i, t in enumerate(temps):
            for j, k in enumerate(topks):
                ax = axes[i][j]
                ax.imshow(mpimg.imread(str(grids[(t, k)])))
                ax.set_xticks([]); ax.set_yticks([])
                if i == 0:
                    ax.set_title(f"top_k={k if k > 0 else 'off'}", fontsize=11)
                if j == 0:
                    ax.set_ylabel(f"temp={t:.2f}", fontsize=11)
        fig.suptitle("sampling sweep (each cell = 4x4 grid of 16 samples)", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        sheet = out / "contact_sheet.png"
        fig.savefig(str(sheet), dpi=110)
        print(f"[sweep] contact sheet -> {sheet}", flush=True)
        return sheet
    except Exception as err:
        print(f"[sweep] WARNING: contact sheet failed ({err})", flush=True)
        return None


def _log_wandb(args, out, temps, topks, grids, sheet):
    try:
        import wandb
        run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                         group=args.wandb_group, name=args.wandb_name, job_type="sampling_sweep",
                         config={"ckpt": str(args.ckpt), "n_per_grid": args.n,
                                 "temps": temps, "topks": topks, "seed": args.seed})
        table = wandb.Table(columns=["temperature", "top_k", "grid"])
        for t in temps:
            for k in topks:
                table.add_data(t, k, wandb.Image(str(grids[(t, k)]),
                                                 caption=f"temp={t:.2f} top_k={k if k > 0 else 'off'}"))
        run.log({"grid_table": table})
        if sheet is not None:
            run.log({"contact_sheet": wandb.Image(str(sheet), caption="contact sheet")})
        print(f"[sweep] W&B run: {run.url}", flush=True)
        run.finish()
    except Exception as err:
        print(f"[sweep] WARNING: W&B logging failed ({err}); grids are on disk in {out}", flush=True)


if __name__ == "__main__":
    main()
