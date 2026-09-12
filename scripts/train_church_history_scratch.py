#!/usr/bin/env python3
"""Long matched Church scratch training with causal coefficient-history recovery."""

import argparse
import codecs
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_church_bar_coefficients import (
    CoefficientExperimentPrior, make_model, quantized_splits, evaluate,
    generation_screen, LaserAux, build_model, atomic_torch_save,
)
from scripts.tools.build_sign_probe_cache import sha256_file
from src.coefficient_history_training import (
    EpochStream, recovery_mask, recovery_strength, sample_coefficient_span,
    scheduled_lr, sign_nll,
)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str))
    temporary.replace(path)


def initialize(path, seed):
    """Create one shared random prior; no pretrained stage-2 weights are read."""
    if path.exists():
        raise FileExistsError(path)
    torch.manual_seed(seed)
    model = build_model(18432, 16384, compound=True, coeff_vocab_size=2048,
                        compound_micro_transformer_layers=2,
                        compound_depth_specific_coeff_heads=True,
                        compound_pair_autoregressive=True,
                        sparsity_level=4, model_preset="lsun-church-350m")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save({"state_dict": model.state_dict(), "epoch": 0, "global_step": 0,
                       "initialization": "random; no stage-2 checkpoint loaded", "seed": seed}, path)
    write_json(path.with_suffix(".json"), {"initialization": "random", "seed": seed,
               "parameters": sum(p.numel() for p in model.parameters()), "sha256": sha256_file(path)})


def logits_and_losses(model, aux, packed, targets=None):
    if targets is None:
        targets = packed
    atoms, ids = model.unpack(targets)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(packed, model_aux=aux)
        logits = model.classify_coefficients(outputs["coefficient_context"]).float()
    coeff = F.cross_entropy(logits.flatten(0, -2), ids.flatten(), reduction="none").reshape_as(ids)
    atom = F.cross_entropy(outputs["atom_logits"].float().flatten(0, -2), atoms.flatten())
    return atom, coeff, logits


@torch.no_grad()
def rollout_evaluation(model, aux, packed, seed):
    model.eval()
    totals = []
    for start in (8, 40):
        rows = []
        uniform = torch.rand((len(packed), (64 - start) * 4), generator=torch.Generator().manual_seed(seed + start))
        for begin in range(0, len(packed), 128):
            truth = packed[begin:begin + 128].cuda()
            predicted = sample_coefficient_span(model, aux, truth, start, 64 - start,
                                                uniform[begin:begin + len(truth)].cuda())
            ids, target = (predicted % 2048).flatten(1), (truth % 2048).flatten(1)
            coeff = aux.coeff_bins[ids] * aux.coeff_scales.repeat(64)
            target_coeff = aux.coeff_bins[target] * aux.coeff_scales.repeat(64)
            rows.append({"sign_accuracy": ((ids[:, start * 4 + 1:] >= 1024) == (target[:, start * 4 + 1:] >= 1024)).float().mean(1).cpu(),
                         "physical_coefficient_mae": (coeff[:, start * 4 + 1:] - target_coeff[:, start * 4 + 1:]).abs().mean(1).cpu()})
        totals.append({key: torch.cat([row[key] for row in rows]) for key in rows[0]})
    per_image = {key: torch.stack([row[key] for row in totals]).mean(0) for key in totals[0]}
    return {key: float(value.mean()) for key, value in per_image.items()}, per_image


def evaluate_all(model, aux, splits, output, log, images=128, seed=4701):
    output.mkdir(parents=True, exist_ok=True)
    result = {}
    for split, packed in [("train_probe", splits["train"][:256]),
                          ("holdout", splits["holdout"]), ("validation", splits["validation"])]:
        values, per_image = evaluate(model, aux, packed)
        result[split] = values
        torch.save(per_image, output / f"{split}-per-image.pt")
        log({"phase": "evaluation", "split": split, **{f"{split}/{k}": v for k, v in values.items()}})
    for split in ("holdout", "validation"):
        values, per_image = rollout_evaluation(model, aux, splits[split][:images], seed)
        result[f"{split}_rollout"] = values
        torch.save(per_image, output / f"{split}-rollout-per-image.pt")
        log({"phase": "evaluation", "split": split + "_rollout", **{f"{split}_rollout/{k}": v for k, v in values.items()}})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=["control", "recovery"], default="recovery")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--initial-prior", type=Path, required=True)
    p.add_argument("--initialize-only", action="store_true")
    p.add_argument("--cache", type=Path, default=ROOT / "outputs/lsun-church-bar-20260910/church-cache.pt")
    p.add_argument("--stage1", type=Path, default=ROOT / "outputs/lsun-church-sign-pattern-20260910/assets/best_rfid_slot1_model.pt")
    p.add_argument("--fid-stats", type=Path, default=ROOT / "outputs/lsun-church-bar-20260910/assets/lsun_256_church.npz")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--microbatch", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup-epochs", type=float, default=5.)
    p.add_argument("--recovery-start-epoch", type=float, default=5.)
    p.add_argument("--recovery-ramp-epochs", type=float, default=10.)
    p.add_argument("--recovery-batch", type=int, default=32)
    p.add_argument("--recovery-max-sites", type=int, default=4)
    p.add_argument("--recovery-coeff-weight", type=float, default=.2)
    p.add_argument("--recovery-sign-weight", type=float, default=.5)
    p.add_argument("--eval-every-epochs", type=int, default=5)
    p.add_argument("--checkpoint-every-epochs", type=int, default=2)
    p.add_argument("--fid-samples", type=int, default=2048)
    p.add_argument("--final-fid-samples", type=int, default=50000)
    p.add_argument("--generation-batch", type=int, default=128)
    p.add_argument("--rollout-images", type=int, default=128)
    p.add_argument("--seed", type=int, default=4701)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after-step", type=int, default=0, help="Save and pause for a smoke or recovery check")
    p.add_argument("--wandb-id")
    p.add_argument("--upload-final-checkpoint", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(8)
    torch.serialization.add_safe_globals([codecs.encode])
    if args.initialize_only:
        initialize(args.initial_prior, args.seed)
        return
    if min(args.epochs, args.batch_size, args.microbatch, args.recovery_batch) < 1:
        p.error("Epochs and batch sizes must be positive")
    if not 1 <= args.recovery_max_sites <= 16:
        p.error("Recovery spans must contain 1 to 16 sites")
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    raw = torch.load(args.cache, weights_only=False, map_location="cpu")
    splits = quantized_splits(raw)
    cache_meta = raw["meta"]
    del raw
    initial_meta = torch.load(args.initial_prior, weights_only=False, map_location="cpu")
    assert initial_meta["initialization"] == "random; no stage-2 checkpoint loaded"
    assert initial_meta["seed"] == args.seed
    del initial_meta
    model, _ = make_model(args.initial_prior, "categorical")
    model.cuda().train()
    aux = LaserAux(args.stage1, 16384, 2048, 20, coeff_scales=cache_meta["coeff_scales"], sparsity_level=4).cuda().eval()
    aux.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(.9, .95), weight_decay=.03, fused=True)
    stream = EpochStream(len(splits["train"]), args.seed + 1)
    replay_rng = torch.Generator().manual_seed(args.seed + 2)
    config = {**vars(args), "initialization": "random; no pretrained stage-2 weights", "initial_prior_sha256": sha256_file(args.initial_prior),
              "tokenizer_sha256": sha256_file(args.stage1), "cache_metadata": cache_meta,
              "train_images": len(splits["train"]), "holdout_images": len(splits["holdout"]),
              "planned_image_presentations": args.epochs * len(splits["train"]),
              "steps_per_epoch": math.ceil(len(splits["train"]) / args.batch_size),
              "objective": "clean joint NLL plus matched coefficient/sign recovery loss; only auxiliary history differs between arms",
              "holdout_scope": "Excluded from stage-2 scratch training; frozen stage-1 tokenizer saw the training population",
              "checkpoint_selection": "minimum fixed-seed FID-2048 among periodic checkpoints; confirm with independent-seed FID-50000",
              "script_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ["scripts/train_church_history_scratch.py", "src/coefficient_history_training.py"]}}
    assert config["tokenizer_sha256"] == cache_meta["checkpoint_sha256"]
    step, best_fid, best_step, best_epoch = 0, float("inf"), 0, 0.
    total_elapsed = 0.
    if args.resume:
        payload = torch.load(args.output / "last.pt", weights_only=False, map_location="cpu")
        invariant = ["arm", "epochs", "batch_size", "microbatch", "lr", "min_lr", "warmup_epochs", "recovery_start_epoch", "recovery_ramp_epochs",
                     "recovery_batch", "recovery_max_sites", "recovery_coeff_weight", "recovery_sign_weight", "seed", "initial_prior_sha256", "tokenizer_sha256", "script_sha256"]
        for key in invariant:
            if str(payload["config"][key]) != str(config[key]):
                raise ValueError(f"Resume setting changed: {key}")
        model.load_state_dict(payload["state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        stream.load_state_dict(payload["stream"])
        replay_rng.set_state(payload["replay_rng"].cpu())
        step, best_fid, best_step, best_epoch = [payload[k] for k in ("step", "best_fid", "best_step", "best_epoch")]
        total_elapsed = payload["elapsed_seconds"]
        del payload
    write_json(args.output / "config.json", config)
    wb = None
    if args.wandb_id:
        import wandb
        wb = wandb.init(entity="helloimlixin-rutgers", project="laser", id=args.wandb_id,
                        name=args.wandb_id, group="church-history-scratch-20260910",
                        job_type="scratch-coefficient-history-recovery", dir=str(args.output),
                        config=json.loads(json.dumps(config, default=str)), resume="must" if args.resume else "never")
    start = time.monotonic()
    stopping = {"signal": None}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda number, frame: stopping.update(signal=number))

    def log(row):
        full = {"optimizer_step": step, "epoch_progress": stream.epoch + stream.position / stream.size,
                "elapsed_seconds": total_elapsed + time.monotonic() - start, **row}
        print(json.dumps(full), flush=True)
        with (args.output / "history.jsonl").open("a") as f:
            f.write(json.dumps(full) + "\n")
        write_json(args.output / "status.json", {"pid": os.getpid(), "arm": args.arm,
                    "planned_epochs": args.epochs, "planned_steps": args.epochs * config["steps_per_epoch"],
                    "best_screen_fid": best_fid if math.isfinite(best_fid) else None, "best_step": best_step, **full})
        if wb:
            wb.log(full)

    def save_last():
        model.init_cache()
        atomic_torch_save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "stream": stream.state_dict(),
                           "replay_rng": replay_rng.get_state(), "step": step, "best_fid": best_fid, "best_step": best_step,
                           "best_epoch": best_epoch, "config": config, "elapsed_seconds": total_elapsed + time.monotonic() - start},
                          args.output / "last.pt")
        log({"phase": "checkpoint_saved", "path": str(args.output / "last.pt")})

    log({"phase": "ready", "parameters": sum(p.numel() for p in model.parameters()), "resumed": args.resume})
    try:
        while stream.epoch + stream.position / stream.size < args.epochs:
            selected, progress, epoch_end = stream.next(args.batch_size)
            step += 1
            batch = splits["train"][selected]
            lr = scheduled_lr(progress, args.epochs, args.lr, args.min_lr, args.warmup_epochs)
            optimizer.param_groups[0]["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            model.train()
            clean_atom, clean_coeff = 0., 0.
            iteration_start = time.monotonic()
            for micro_index, chunk in enumerate(batch.split(args.microbatch)):
                torch.manual_seed(args.seed + step * 100 + micro_index)
                atom_loss, coefficient_loss, logits = logits_and_losses(model, aux, chunk.cuda())
                coeff_loss = coefficient_loss.mean()
                weight = len(chunk) / len(batch)
                ((atom_loss + coeff_loss) * .5 * weight).backward()
                clean_atom += float(atom_loss.detach()) * weight
                clean_coeff += float(coeff_loss.detach()) * weight
                del atom_loss, coefficient_loss, coeff_loss, logits
            strength = recovery_strength(progress, args.recovery_start_epoch, args.recovery_ramp_epochs)
            replay_metrics = {"recovery/strength": strength, "recovery/loss": 0.}
            if strength > 0:
                span = max(1, math.ceil(args.recovery_max_sites * strength))
                first = int(torch.randint(1, 64 - span, (), generator=replay_rng))
                original = batch[:min(args.recovery_batch, len(batch))].cuda()
                uniform = torch.rand((len(original), span * 4), generator=replay_rng).cuda()
                history = (sample_coefficient_span(model, aux, original, first, span, uniform)
                           if args.arm == "recovery" else original)
                mask = recovery_mask(model.block_size, first, span, device="cuda").expand_as(original)
                torch.manual_seed(args.seed + step * 100 + 99)
                _, coefficient_loss, logits = logits_and_losses(model, aux, history, original)
                coeff_recovery = coefficient_loss[mask].mean()
                sign_recovery = sign_nll(logits, original % 2048)[mask].mean()
                recovery = strength * (args.recovery_coeff_weight * coeff_recovery + args.recovery_sign_weight * sign_recovery)
                recovery.backward()
                injected = slice(first * 4, (first + span) * 4)
                true_ids, predicted_ids = (original % 2048).flatten(1)[:, injected], (history % 2048).flatten(1)[:, injected]
                replay_metrics.update({"recovery/loss": float(recovery.detach()), "recovery/coefficient_nll": float(coeff_recovery.detach()),
                                       "recovery/sign_nll": float(sign_recovery.detach()), "recovery/span_sites": span,
                                       "recovery/changed_fraction": float((true_ids != predicted_ids).float().mean()),
                                       "recovery/sign_changed_fraction": float(((true_ids >= 1024) != (predicted_ids >= 1024)).float().mean())})
                del original, history, coefficient_loss, logits, coeff_recovery, sign_recovery, recovery
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError(f"Nonfinite gradient at step {step}; prior durable checkpoint retained")
            optimizer.step()
            if step == 1 or step % 10 == 0 or epoch_end:
                seconds = time.monotonic() - iteration_start
                log({"phase": "train", "train/atom_nll": clean_atom, "train/coefficient_nll": clean_coeff,
                     "train/clean_joint_nll": (clean_atom + clean_coeff) / 2,
                     "train/lr": lr, "train/gradient_norm": float(norm), "train/step_seconds": seconds,
                     "train/images_per_second": len(batch) / seconds, "train/images_seen": stream.epoch * stream.size + stream.position,
                     "gpu/peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30, **replay_metrics})
            if stopping["signal"] or (args.stop_after_step and step >= args.stop_after_step):
                save_last()
                log({"phase": "paused", "signal": stopping["signal"]})
                if wb:
                    wb.summary["training_status"] = "paused"
                    wb.finish()
                return
            if epoch_end:
                completed_epoch = stream.epoch + 1
                evaluate_now = completed_epoch == 1 or completed_epoch % args.eval_every_epochs == 0 or completed_epoch == args.epochs
                if evaluate_now:
                    model.eval()
                    optimizer.zero_grad(set_to_none=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                    directory = args.output / f"evaluations/epoch-{completed_epoch:03d}"
                    values = evaluate_all(model, aux, splits, directory, log, args.rollout_images, args.seed)
                    if args.fid_samples:
                        setting = SimpleNamespace(seed=args.seed, fid_stats=args.fid_stats, fid_samples=args.fid_samples,
                                                  generation_batch=args.generation_batch, output=directory)
                        values["generation"] = generation_screen(model, aux, setting, log)
                        fid = values["generation"]["fid"]
                        if fid < best_fid:
                            best_fid, best_step, best_epoch = fid, step, progress
                            atomic_torch_save({"state_dict": model.state_dict(), "config": config,
                                               "step": step, "epoch": progress, "fid": fid}, args.output / "best-screen.pt")
                        log({"phase": "screen_complete", "screen/fid": fid, "screen/epoch": completed_epoch})
                        if wb:
                            import wandb
                            wb.log({"samples": wandb.Image(str(directory / "samples.png")), "optimizer_step": step})
                    write_json(directory / "results.json", values)
                    save_last()
                elif completed_epoch % args.checkpoint_every_epochs == 0:
                    save_last()
        save_last()
        model.eval()
        if (args.output / "best-screen.pt").exists():
            selected = torch.load(args.output / "best-screen.pt", weights_only=False, map_location="cpu")
            model.load_state_dict(selected["state_dict"], strict=True)
            del selected
        else:
            best_step, best_epoch = step, float(args.epochs)
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()
        final_dir = args.output / "final-evaluation"
        results = evaluate_all(model, aux, splits, final_dir, log, 300, args.seed + 1)
        if args.final_fid_samples:
            setting = SimpleNamespace(seed=args.seed + 1, fid_stats=args.fid_stats, fid_samples=args.final_fid_samples,
                                      generation_batch=args.generation_batch, output=final_dir)
            results["generation"] = generation_screen(model, aux, setting, log)
        results["selected_checkpoint"] = {"step": best_step, "epoch": best_epoch, "screen_fid": best_fid}
        write_json(args.output / "results.json", results)
        log({"phase": "complete", "selected_step": best_step, **{f"final/{k}": v for k, v in results.get("generation", {}).items()}})
        if wb:
            for split, values in results.items():
                for key, value in values.items():
                    wb.summary[f"final/{split}/{key}"] = value
            wb.summary["training_status"] = "complete"
            wb.save(str(args.output / "results.json"), base_path=str(args.output))
            if args.upload_final_checkpoint and (args.output / "best-screen.pt").exists():
                wb.save(str(args.output / "best-screen.pt"), base_path=str(args.output))
            wb.finish()
    except Exception as error:
        log({"phase": "failed", "error": repr(error), "last_checkpoint": str(args.output / "last.pt")})
        if wb:
            wb.summary["training_status"] = "failed"
            wb.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
