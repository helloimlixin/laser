#!/usr/bin/env python3
"""Compare joint versus independent signs given real supports and magnitudes."""

import argparse
import codecs
from contextlib import nullcontext
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.sign_pattern_prior import (
    SignPatternPrior, pack_signs, pattern_log_probabilities, sign_metrics, sign_patterns,
)


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


@torch.no_grad()
def evaluate(model, data, batch_size, *, rollout=False):
    model.eval()
    arrays = {}
    for start in range(0, len(data["atoms"]), batch_size):
        atoms = data["atoms"][start:start + batch_size]
        coefficients = data["coefficients"][start:start + batch_size]
        with autocast(atoms.device):
            if rollout:
                prediction = model.rollout(atoms, coefficients.abs())
                directions = model.dictionary.t()[atoms]
                delta = ((prediction - coefficients).unsqueeze(-1) * directions).sum(-2)
                target = (coefficients.unsqueeze(-1) * directions).sum(-2)
                correct = (prediction >= 0) == (coefficients >= 0)
                values = {
                    "pattern_accuracy": correct.all(-1).float().mean(1),
                    "sign_accuracy": correct.float().mean((1, 2)),
                    "physical_coefficient_mse": (prediction - coefficients).square().mean((1, 2)),
                    "latent_mse": delta.square().mean((1, 2)),
                    "latent_energy": target.square().mean((1, 2)),
                    **{f"sign_accuracy_depth{d}": correct[..., d].float().mean(1)
                       for d in range(model.depth)},
                }
            else:
                logits = model(atoms, coefficients.abs(), coefficients)
                values = sign_metrics(logits, coefficients, atoms, model.dictionary, model.mode)
        for key, value in values.items():
            arrays.setdefault(key, []).append(value.float().cpu())
    arrays = {key: torch.cat(value) for key, value in arrays.items()}
    metrics = {key: float(value.mean()) for key, value in arrays.items()}
    metrics["latent_nmse"] = metrics["latent_mse"] / max(metrics["latent_energy"], 1e-12)
    return metrics, arrays


@torch.no_grad()
def decode_probe(model, data, checkpoint, output, batch_size=8):
    """Decode teacher-forced and sign-rollout outputs, keeping oracle magnitudes."""
    from scripts.train_official_rqtransformer_laser_stage2 import LaserAux
    from torchvision.utils import save_image
    device = next(model.parameters()).device
    torch.serialization.add_safe_globals([codecs.encode])
    aux = LaserAux(checkpoint, model.dictionary.shape[1], 2048, 20,
                   sparsity_level=model.depth).to(device).eval()
    if not torch.allclose(aux.dictionary, model.dictionary, atol=1e-6):
        raise ValueError("decoder checkpoint dictionary differs from cached dictionary")
    collected = {"teacher_forced": [], "sign_rollout": []}
    previews = {"oracle": [], "teacher_forced": [], "sign_rollout": []}
    for start in range(0, len(data["atoms"]), batch_size):
        atoms = data["atoms"][start:start + batch_size]
        coefficients = data["coefficients"][start:start + batch_size]
        with autocast(device):
            logits = model(atoms, coefficients.abs(), coefficients)
            probs = pattern_log_probabilities(logits, model.mode, model.depth)
            signs = sign_patterns(model.depth, device=device)[probs.argmax(-1)].float() * 2 - 1
            predicted = coefficients.abs() * signs
            rolled = model.rollout(atoms, coefficients.abs())
            decoded = {}
            for name, values in [("oracle", coefficients), ("teacher_forced", predicted), ("sign_rollout", rolled)]:
                z = (aux.dictionary.t()[atoms] * values.unsqueeze(-1)).sum(-2)
                side = math.isqrt(model.sites)
                z = z.reshape(-1, side, side, z.shape[-1]).permute(0, 3, 1, 2).contiguous()
                decoded[name] = ((aux.decoder(aux.post_quant_conv(z)).float().clamp(-1, 1) + 1) / 2)
                if start < 16:
                    previews[name].append(decoded[name][:min(len(atoms), 16 - start)].cpu())
            for name in collected:
                mse = (decoded[name] - decoded["oracle"]).square().mean((1, 2, 3))
                collected[name].append((-10 * mse.clamp_min(1e-12).log10()).cpu())
    # Rows preserve correspondence; these images are oracle reconstructions.
    grid = torch.cat([torch.cat(previews[name]) for name in previews])
    save_image(grid, output / "oracle-sign-reconstructions.png", nrow=16)
    return {name + "_psnr_vs_true_codes": float(torch.cat(values).mean())
            for name, values in collected.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["joint", "independent"], required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--seed", type=int, default=2701)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decode-checkpoint", type=Path)
    parser.add_argument("--wandb-id")
    parser.add_argument("--wandb-group", default="church-sign-pattern-20260910")
    parser.add_argument("--wandb-key-stdin", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.eval_every < 1 or args.batch_size < 1:
        parser.error("steps, eval-every and batch-size must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to replace a previous experiment: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.wandb_key_stdin:
        os.environ["WANDB_API_KEY"] = getpass.getpass("W&B API key: ")
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device)
    cache = torch.load(args.cache, weights_only=True, map_location="cpu")
    if cache["meta"]["format"] != "laser_sign_probe_v1":
        raise ValueError("unsupported cache format or coefficient units")
    dictionary = cache["dictionary"]
    data = {}
    for name in ["train", "holdout", "validation"]:
        part = cache[name]
        if not torch.isfinite(part["coefficients"]).all():
            raise ValueError("non-finite coefficients")
        data[name] = {
            "atoms": part["atoms"].flatten(1, 2).long().to(device),
            "coefficients": part["coefficients"].flatten(1, 2).float().to(device),
        }
    keys = [set(cache[name]["keys"]) for name in data]
    if any(keys[i] & keys[j] for i in range(3) for j in range(i)):
        raise ValueError("training, holdout and validation must be image-disjoint")
    rms = data["train"]["coefficients"].square().mean((0, 1)).sqrt().cpu().clamp_min(1e-6)
    depth = data["train"]["atoms"].shape[-1]
    sites = data["train"]["atoms"].shape[1]
    model_args = dict(depth=depth, sites=sites, width=args.width, layers=args.layers,
                      heads=args.heads, dropout=args.dropout, mode=args.mode)
    model = SignPatternPrior(dictionary, rms, **model_args).to(device)
    configuration = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                     if k != "wandb_key_stdin"}
    configuration.update({
        "cache_meta": cache["meta"], "model": model_args,
        "parameters": sum(p.numel() for p in model.parameters()),
        "split_counts": {k: len(v["atoms"]) for k, v in data.items()},
        "coefficient_rms": rms.tolist(),
        "conditioning": "current oracle support/magnitudes; only earlier signed raster sites",
        "selection": "minimum holdout pattern NLL; official validation evaluated after selection",
        "evaluation": "teacher forcing and deterministic MAP sign rollout; not unconditional generation",
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in [Path(__file__), ROOT / "src/sign_pattern_prior.py"]},
    })
    (args.output / "config.json").write_text(json.dumps(configuration, indent=2))
    run = None
    if args.wandb_id:
        import wandb
        run = wandb.init(entity="helloimlixin-rutgers", project="laser", id=args.wandb_id,
                         name=args.wandb_id, group=args.wandb_group, config=configuration,
                         dir=str(args.output), job_type="oracle-sign-diagnostic")
    history = open(args.output / "history.jsonl", "w", buffering=1)

    def log(record):
        history.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if run:
            run.log(record)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.03)
    generator = torch.Generator().manual_seed(args.seed + 1)
    best, best_step = float("inf"), 0
    started = time.monotonic()
    for step in range(args.steps + 1):
        if step:
            model.train()
            indices = torch.randint(len(data["train"]["atoms"]), (args.batch_size,), generator=generator).to(device)
            atoms = data["train"]["atoms"][indices]
            coefficients = data["train"]["coefficients"][indices]
            warmup = min(100, max(args.steps // 10, 1))
            lr = args.lr * min(step / warmup, 1) * (0.1 + 0.9 * (1 + math.cos(math.pi * max(step - warmup, 0) / max(args.steps - warmup, 1))) / 2)
            optimizer.param_groups[0]["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with autocast(device):
                logits = model(atoms, coefficients.abs(), coefficients)
                log_probs = pattern_log_probabilities(logits, args.mode, depth)
                loss = F.nll_loss(log_probs.flatten(0, 1), pack_signs(coefficients >= 0).flatten()) / depth
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            if not torch.isfinite(loss) or not torch.isfinite(grad):
                raise FloatingPointError("non-finite loss or gradient")
            optimizer.step()
            if step % 50 == 0:
                log({"step": step, "train/nll_per_sign": float(loss), "train/lr": lr,
                     "train/gradient_norm": float(grad), "elapsed_seconds": time.monotonic() - started})
        if step % args.eval_every == 0 or step == args.steps:
            metrics, _ = evaluate(model, data["holdout"], args.batch_size)
            log({"step": step, **{"holdout/" + k: v for k, v in metrics.items()}})
            if metrics["pattern_nll"] < best:
                best, best_step = metrics["pattern_nll"], step
                torch.save({"state_dict": model.state_dict(), "model_args": model_args,
                            "step": step, "holdout": metrics}, args.output / "best.pt")
    model.load_state_dict(torch.load(args.output / "best.pt", weights_only=True, map_location=device)["state_dict"])
    results = {"best_step": best_step, "holdout_pattern_nll": best, "mode": args.mode,
               "conditioning": configuration["conditioning"], "scope": configuration["evaluation"]}
    for name in ["holdout", "validation"]:
        for rollout in [False, True]:
            label = name + ("_sign_rollout" if rollout else "_teacher_forced")
            metrics, arrays = evaluate(model, data[name], args.batch_size, rollout=rollout)
            results[label] = metrics
            torch.save(arrays, args.output / (label + "_per_image.pt"))
            log({"step": args.steps, **{label + "/" + k: v for k, v in metrics.items()}})
    if args.decode_checkpoint:
        # Decoder evaluation uses the official validation images, once per arm.
        results["validation_reconstruction"] = decode_probe(model, data["validation"], args.decode_checkpoint, args.output)
        log({"step": args.steps, **{"validation_reconstruction/" + k: v
                                   for k, v in results["validation_reconstruction"].items()}})
        if run:
            run.log({"oracle/reconstructions": wandb.Image(str(args.output / "oracle-sign-reconstructions.png"),
                     caption="Rows: true codes; teacher-forced predicted signs; rolled-out predicted signs. Real supports and magnitudes throughout; not unconditional generation.")})
    (args.output / "results.json").write_text(json.dumps(results, indent=2))
    history.close()
    if run:
        run.summary["best_step"] = best_step
        run.save(str(args.output / "results.json"), base_path=str(args.output), policy="now")
        run.finish()


if __name__ == "__main__":
    main()
