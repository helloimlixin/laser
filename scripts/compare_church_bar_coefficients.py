#!/usr/bin/env python3
"""Summarize completed BAR/categorical runs without comparing unlike losses."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    modes = ["source-baseline", "categorical", "bar"]
    results = {m: json.loads((root / m / "results.json").read_text()) for m in modes}
    configs = {m: json.loads((root / m / "config.json").read_text()) for m in modes}
    for key in ["steps", "head_only_steps", "batch_size", "head_lr", "backbone_lr", "seed", "train_images", "source_checkpoint_sha256", "tokenizer_sha256"]:
        assert configs["categorical"][key] == configs["bar"][key], key
    for key in ["samples", "seed", "atom_top_k", "atom_temperature", "coefficient_temperature", "ar_precision", "decoder_precision"]:
        assert len({results[m]["generation"][key] for m in modes}) == 1, key
    paired = {m: torch.load(root / m / "validation-per-image.pt", weights_only=True) for m in modes[1:]}
    n = len(paired["bar"]["sign_accuracy"])
    draws = np.random.default_rng(2701).integers(0, n, (5000, n))
    intervals = {}
    for metric in ["sign_accuracy", "coefficient_accuracy", "physical_coefficient_mae", "latent_mse", "atom_nll"]:
        difference = (paired["bar"][metric] - paired["categorical"][metric]).numpy()
        intervals[metric] = {"bar_minus_categorical": float(difference.mean()),
                             "paired_image_bootstrap95": np.quantile(difference[draws].mean(1), [.025, .975]).tolist()}
    summary = {"results": results, "validation_differences": intervals,
               "note": "One training seed. Image bootstrap does not measure retraining or FID uncertainty. BAR BCE and categorical NLL are different objectives."}
    (root / "comparison.json").write_text(json.dumps(summary, indent=2))
    colors = {"source-baseline": "#7c8796", "categorical": "#4269c7", "bar": "#d56c29"}
    labels = {"source-baseline": "Original AR", "categorical": "Categorical continuation", "bar": "BAR coefficient head"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for mode in modes[1:]:
        rows = [json.loads(line) for line in (root / mode / "history.jsonl").read_text().splitlines()]
        rows = [row for row in rows if row["phase"] == "development256"]
        axes[0].plot([r["optimizer_step"] for r in rows], [r["physical_coefficient_mae"] for r in rows], marker="o", label=labels[mode], color=colors[mode])
    axes[0].set(xlabel="Continuation step", ylabel="Physical coefficient MAE", title="Monitoring split, true history")
    axes[0].legend(fontsize=8)
    for axis, metric, scale, title, ylabel in [
        (axes[1], "sign_accuracy", 100, "Official validation, true history", "Sign accuracy (%)"),
        (axes[2], "fid", 1, "Unconditional generation", f"FID ({results['bar']['generation']['samples']:,} samples)")]:
        values = [results[m]["generation" if metric == "fid" else "validation"][metric] * scale for m in modes]
        axis.bar(range(3), values, color=[colors[m] for m in modes])
        axis.set_xticks(range(3), ["Original", "Categorical", "BAR"])
        axis.set(title=title, ylabel=ylabel)
        for i, value in enumerate(values):
            axis.text(i, value, f"{value:.2f}", ha="center", va="bottom")
        axis.set_ylim(0, max(values) * 1.13)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.15)
        axis.set_axisbelow(True)
    fig.suptitle("LSUN Church: fixed tokenizer and pretrained AR initialization")
    fig.tight_layout()
    fig.savefig(root / "comparison.png", dpi=170)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
