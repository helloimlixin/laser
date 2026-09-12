#!/usr/bin/env python3
"""Summarize atom interventions with paired bootstrap over validation images."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def interval(values, draws):
    values = np.asarray(values)
    return {"mean": float(values.mean()),
            "paired_image_bootstrap95": np.quantile(values[draws].mean(1), [.025, .975]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    data = torch.load(args.root / "local.pt", weights_only=True)
    count = len(data["packed"])
    draws = np.random.default_rng(3701).integers(0, count, (5000, count))
    base = data["baseline"]
    truth = data["physical_targets"]
    scales = data["scales"].repeat(64)
    bins = data["bins"]
    base_mae = (bins[base["prediction"]] * scales - truth).abs()
    base_sign = ((base["prediction"] >= 1024) == (truth > 0)).float()
    ranges = {"next_coefficient": (1, 2), "offsets_2_3": (2, 4),
              "offsets_4_15": (4, 16), "offsets_16_63": (16, 64),
              "same_depth_next_spatial_row": (32, 33), "offsets_64_plus": (64, 256)}
    local, local_curves = {}, {}
    for kind in ("near", "model"):
        cases = [c for c in data["cases"] if c["kind"] == kind]
        immediate = {
            "replacement_cosine": interval(torch.stack([c["cosine"] for c in cases]).mean(0), draws),
            "original_prediction_mae_at_intervention": interval(torch.stack([base_mae[:, c["position"]] for c in cases]).mean(0), draws),
            "replacement_prediction_mae_to_projection": interval(torch.stack([c["replacement_prediction_mae_to_projection"] for c in cases]).mean(0), draws),
            "adjusted_contribution_mse": interval(torch.stack([c["adjusted_contribution_mse"] for c in cases]).mean(0), draws),
            "predicted_contribution_mse": interval(torch.stack([c["predicted_contribution_mse"] for c in cases]).mean(0), draws),
        }
        local[kind] = {"immediate": immediate}
        for mode in ("adjusted", "predicted"):
            values = {}
            curves = []
            for case in cases:
                p = case["position"]
                stats = case[mode]
                pred_c = bins[stats["prediction"]] * scales
                mae_delta = (pred_c - truth).abs() - base_mae
                sign_delta = (((stats["prediction"] >= 1024) == (truth > 0)).float() - base_sign) * 100
                nll_delta = stats["nll"] - base["nll"]
                probability_shift = (stats["positive_probability"] - base["positive_probability"]).abs()
                for label, (lower, upper) in ranges.items():
                    first, last = p + lower, min(p + upper, 256)
                    if first >= last:
                        continue
                    destination = values.setdefault(label, {})
                    for name, array in [("mae_delta", mae_delta), ("sign_accuracy_delta_pp", sign_delta),
                                        ("nll_delta", nll_delta), ("positive_probability_absolute_shift", probability_shift)]:
                        destination.setdefault(name, []).append(array[:, first:last].mean(1))
                curve = torch.full((count, 64), torch.nan)
                length = min(64, 255 - p)
                curve[:, :length] = mae_delta[:, p + 1:p + 1 + length]
                curves.append(curve)
            local[kind][mode] = {label: {metric: interval(torch.stack(rows).mean(0), draws)
                                                    for metric, rows in metrics.items()}
                                       for label, metrics in values.items()}
            local_curves[f"{kind}-{mode}"] = torch.stack(curves).nanmean((0, 1)).numpy()

    rollouts, rollout_curves = {}, {}
    for sampling in ("greedy", "sample"):
        path = args.root / f"rollout-{sampling}.pt"
        if not path.exists():
            continue
        result = torch.load(path, weights_only=True)
        assert result["provenance"] == data["provenance"]
        oracle_path = args.root / f"oracle-{sampling}.pt"
        if oracle_path.exists():
            oracle = torch.load(oracle_path, weights_only=True)
            assert oracle["provenance"] == result["provenance"]
            assert oracle["seed"] == result["seed"]
            for original_site, oracle_site in zip(result["sites"], oracle["sites"], strict=True):
                assert original_site["start"] == oracle_site["start"]
                original_site["conditions"].update(oracle_site["conditions"])
        names = result["sites"][0]["conditions"]
        raw, per_site = {}, {}
        for name in names:
            accumulated = {}
            curves = []
            for site in result["sites"]:
                p = site["start"]
                condition = site["conditions"][name]
                m = condition["metrics"]
                values = {
                    "downstream_mae": m["mae"][:, p + 1:].mean(1),
                    "downstream_sign_accuracy_percent": m["correct_sign"][:, p + 1:].mean(1) * 100,
                    "latent_mse": m["latent_mse_by_site"].mean(1),
                    "psnr_to_true_code_reconstruction": m["psnr_to_true_code_reconstruction"],
                }
                for k, v in values.items():
                    accumulated.setdefault(k, []).append(v)
                per_site.setdefault(str(site["site"]), {})[name] = {k: float(v.mean()) for k, v in values.items()}
                curves.append(m["correct_sign"][:, p + 1:p + 96])
            raw[name] = {key: torch.stack(rows).mean(0) for key, rows in accumulated.items()}
            rollout_curves[(sampling, name)] = torch.stack(curves).mean((0, 1)).numpy() * 100
        differences = {}
        pairs = [("control", "teacher-forced"), ("control-clamped", "control"),
                 ("near-adjusted", "control-clamped"), ("model-adjusted", "control-clamped"),
                 ("near-predicted", "control"), ("model-predicted", "control")]
        pairs.extend((name, "control") for name in ("sign-oracle", "magnitude-oracle") if name in raw)
        for changed, reference in pairs:
            differences[f"{changed}_minus_{reference}"] = {metric: interval(raw[changed][metric] - raw[reference][metric], draws)
                                                           for metric in raw[reference]}
        rollouts[sampling] = {"means": {name: {k: float(v.mean()) for k, v in metrics.items()} for name, metrics in raw.items()},
                              "per_start_site": per_site, "paired_differences": differences}

    summary = {
        "images": count, "provenance": data["provenance"], "local": local, "rollouts": rollouts,
        "scope": "Fixed original model; oracle later atoms; equal average over intervention sites within each image. Bootstrap over images, not tokens. Downstream scores exclude the changed atom's coefficient. Reconstruction target is the true-code reconstruction. These are conditional diagnostics, not unconditional FID.",
    }
    (args.root / "summary.json").write_text(json.dumps(summary, indent=2))
    fig, axes_grid = plt.subplots(2, 2, figsize=(12, 8.3), layout="constrained")
    axes = axes_grid.flatten()
    for name, color in [("near-adjusted", "#4e79a7"), ("near-predicted", "#f28e2b"), ("model-predicted", "#e15759")]:
        axes[0].plot(np.arange(1, 65), local_curves[name], color=color, label=name)
    axes[0].axhline(0, color="#777777", lw=.7)
    axes[0].set(title="One changed pair; real later history", xlabel="Coefficient steps after intervention", ylabel="Change in coefficient MAE")
    axes[0].legend(fontsize=8, frameon=False)
    for ax, sampling in zip(axes[1:3], ("greedy", "sample")):
        if sampling not in rollouts:
            continue
        for name, label, color in [("teacher-forced", "Real coefficient history", "#59a14f"),
                                   ("control", "Generated coefficients", "#4e79a7"),
                                   ("near-predicted", "+ nearest wrong atom", "#f28e2b"),
                                   ("model-predicted", "+ model alternative atom", "#e15759")]:
            y = rollout_curves[(sampling, name)]
            ax.plot(np.arange(1, len(y) + 1), y, alpha=.45, color=color, lw=.7)
            # Centered 5-token smoothing, keeping only complete windows.
            if len(y) >= 5:
                ax.plot(np.arange(3, len(y) - 1), np.convolve(y, np.ones(5) / 5, mode="valid"), color=color, label=label)
        ax.set(title=f"Fixed-support rollout: {sampling}", xlabel="Coefficient steps after intervention", ylabel="Correct coefficient signs (%)")
        ax.legend(fontsize=8, frameon=False)
    for index, (name, label, color) in enumerate([
        ("control", "Generated coefficients", "#4e79a7"),
        ("sign-oracle", "Correct every sign", "#59a14f"),
        ("magnitude-oracle", "Correct every magnitude", "#f28e2b"),
    ]):
        if all(name in rollouts.get(mode, {}).get("means", {}) for mode in ("greedy", "sample")):
            values = [rollouts[mode]["means"][name]["psnr_to_true_code_reconstruction"] for mode in ("greedy", "sample")]
            bars = axes[3].bar(np.arange(2) + (index - 1) * .25, values, width=.25, color=color, label=label)
            axes[3].bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    axes[3].set(title="Oracle corrections during coefficient rollout", xticks=[0, 1], xticklabels=["Greedy", "Sampled"], ylabel="PSNR to true-code reconstruction (dB)")
    if axes[3].patches:
        axes[3].legend(fontsize=8, frameon=False, loc="lower left")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
    fig.suptitle("LSUN Church: separating atom mistakes from coefficient-history effects")
    fig.savefig(args.root / "diagnostic.png", dpi=170)
    plt.close(fig)
    compact = {"local_immediate": {kind: local[kind]["immediate"] for kind in local},
               "rollout_means": {mode: rollouts[mode]["means"] for mode in rollouts}}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
