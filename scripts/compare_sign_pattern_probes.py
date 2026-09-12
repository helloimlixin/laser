#!/usr/bin/env python3
"""Summarize a matched pair of oracle sign probes with image-level uncertainty."""

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
    parser.add_argument("root", type=Path, help="Directory containing joint/ and independent/")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2701)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("bootstrap count must be positive")
    modes = ["independent", "joint"]
    results = {m: json.loads((args.root / m / "results.json").read_text()) for m in modes}
    configs = {m: json.loads((args.root / m / "config.json").read_text()) for m in modes}
    for key in ["cache", "cache_meta", "seed", "steps", "batch_size", "width", "layers", "heads", "lr", "dropout"]:
        if configs["joint"][key] != configs["independent"][key]:
            raise ValueError(f"unmatched experimental setting: {key}")
    comparison = {
        "scope": "Paired bootstrap over official-validation images; intervals exclude retraining variability.",
        "bootstrap_resamples": args.bootstrap, "bootstrap_seed": args.seed,
        "joint_minus_independent": {}, "results": results,
    }
    rng = np.random.default_rng(args.seed)
    for setting in ["validation_teacher_forced", "validation_sign_rollout"]:
        arrays = {m: torch.load(args.root / m / (setting + "_per_image.pt"), weights_only=True) for m in modes}
        count = len(arrays["joint"]["sign_accuracy"])
        if len(arrays["independent"]["sign_accuracy"]) != count:
            raise ValueError("mismatched evaluation populations")
        indices = rng.integers(0, count, size=(args.bootstrap, count))
        comparison["joint_minus_independent"][setting] = {}
        for key in ["pattern_nll", "pattern_accuracy", "sign_accuracy", "latent_mse"]:
            if key not in arrays["joint"]:
                continue
            delta = (arrays["joint"][key] - arrays["independent"][key]).numpy()
            bounds = np.quantile(delta[indices].mean(1), [0.025, 0.975])
            comparison["joint_minus_independent"][setting][key] = {
                "mean": float(delta.mean()), "ci95": bounds.tolist(),
            }
    (args.root / "comparison.json").write_text(json.dumps(comparison, indent=2))
    print(json.dumps(comparison["joint_minus_independent"], indent=2))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    colors = {"independent": "#777777", "joint": "#2869b2"}
    labels = {"independent": "Independent signs", "joint": "Joint 16-pattern head"}
    for mode in modes:
        history = [json.loads(line) for line in (args.root / mode / "history.jsonl").read_text().splitlines()]
        history = [r for r in history if "holdout/pattern_nll" in r]
        axes[0].plot([r["step"] for r in history], [r["holdout/pattern_nll"] for r in history],
                     label=labels[mode], color=colors[mode])
    axes[0].set(title="Development sign-pattern NLL", xlabel="Optimizer step", ylabel="Nats per 4-sign pattern")
    axes[0].legend(frameon=False, fontsize=8)
    for axis, key, title, ylabel in [
        (axes[1], "pattern_accuracy", "Correct complete patterns", "Percent of sites"),
        (axes[2], "sign_accuracy", "Correct individual signs", "Percent of coefficients"),
    ]:
        for index, mode in enumerate(modes):
            values = [100 * results[mode][s][key] for s in ["validation_teacher_forced", "validation_sign_rollout"]]
            bars = axis.bar(np.arange(2) + (index - 0.5) * 0.35, values, width=0.35, color=colors[mode])
            axis.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
        axis.set(xticks=[0, 1], xticklabels=["True sign history", "Predicted sign history"], title=title, ylabel=ylabel)
        axis.set_ylim(0, max(100 * results[m][s][key] for m in modes for s in ["validation_teacher_forced", "validation_sign_rollout"]) * 1.25)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.15)
        axis.set_axisbelow(True)
    fig.suptitle("LSUN Church sign diagnostic — real supports and magnitudes throughout", fontsize=12)
    fig.savefig(args.root / "comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
