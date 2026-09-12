#!/usr/bin/env python3
"""Train and evaluate only a new Church latent repair network."""
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

import torch
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_church_bar_coefficients import LaserAux, quantized_splits, atomic_torch_save
from scripts.tools.build_sign_probe_cache import sha256_file
from src.sparse_latent_repair import SparseLatentRepair, synthetic_corruption
from src.models.lpips import LPIPS
from src.rqvae_metrics import DistributedOriginalRQVAEMetrics


def write_json(path, payload):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, default=str))
    temp.replace(path)


@torch.no_grad()
def latent(aux, packed):
    return aux.compound_embeddings(packed // 2048, packed % 2048).sum(-2).permute(0, 3, 1, 2).contiguous()


def decode(aux, z):
    # The weights are frozen, but autograd must retain the input gradient.
    return aux.decoder(aux.post_quant_conv(z)).clamp(-1, 1)


@torch.no_grad()
def nearest_atoms(dictionary, k=8):
    result = []
    for begin in range(0, dictionary.shape[1], 512):
        scores = dictionary[:, begin:begin + 512].t() @ dictionary
        scores[torch.arange(len(scores), device=scores.device), torch.arange(begin, begin + len(scores), device=scores.device)] = -torch.inf
        result.append(scores.topk(k, -1).indices)
    return torch.cat(result)


@torch.no_grad()
def repair_forward(repair, z):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return repair(z).float()


@torch.no_grad()
def reconstruction_evaluation(repair, aux, perceptual, conditions, output, batch_size, log):
    repair.eval()
    output.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, (noisy, truth) in conditions.items():
        rows = []
        for begin in range(0, len(truth), batch_size):
            clean_z = latent(aux, truth[begin:begin + batch_size].cuda().long())
            noisy_z = latent(aux, noisy[begin:begin + batch_size].cuda().long())
            reference = decode(aux, clean_z)
            original = decode(aux, noisy_z)
            repaired_z = repair_forward(repair, noisy_z)
            repaired = decode(aux, repaired_z)
            row = {}
            for label, z, image in [("baseline", noisy_z, original), ("repair", repaired_z, repaired)]:
                row[label + "/latent_mse"] = (z - clean_z).square().mean((1, 2, 3)).cpu()
                row[label + "/lpips"] = perceptual(image, reference).flatten().cpu()
                row[label + "/psnr"] = (-10 * ((image - reference) / 2).square().mean((1, 2, 3)).clamp_min(1e-12).log10()).cpu()
            rows.append(row)
            if begin == 0:
                grid = torch.stack([reference[:8], original[:8], repaired[:8]], 1).flatten(0, 1)
                save_image((grid + 1) / 2, output / (name + ".png"), nrow=3)
        per_image = {key: torch.cat([row[key] for row in rows]) for key in rows[0]}
        torch.save(per_image, output / (name + ".pt"))
        values = {key: float(value.mean()) for key, value in per_image.items()}
        for metric in ["lpips", "psnr", "latent_mse"]:
            change = per_image["repair/" + metric] - per_image["baseline/" + metric]
            values["paired_delta/" + metric] = float(change.mean())
            values["paired_standard_error/" + metric] = float(change.std() / len(change) ** .5)
        values["images"] = len(truth)
        result[name] = values
        log({"phase": "reconstruction_evaluation", "condition": name, **values})
    write_json(output / "metrics.json", result)
    return result


@torch.no_grad()
def generation_evaluation(repair, aux, perceptual, codes, stats, output, batch_size, log):
    repair.eval()
    output.mkdir(parents=True, exist_ok=True)
    original_metric = DistributedOriginalRQVAEMetrics("cuda", reference_stats_path=stats)
    repaired_metric = DistributedOriginalRQVAEMetrics("cuda", reference_stats_path=stats, inception=original_metric.inception)
    packed = codes["atoms"].long() * 2048 + codes["coefficient_ids"].long()
    latents, differences = [], []
    diversity = {"baseline": [], "repair": []}
    for begin in range(0, len(packed), batch_size):
        z = latent(aux, packed[begin:begin + batch_size].cuda())
        repaired_z = repair_forward(repair, z)
        original, repaired = decode(aux, z), decode(aux, repaired_z)
        original_metric.update(((original + 1) / 2).clamp(0, 1), real=False)
        repaired_metric.update(((repaired + 1) / 2).clamp(0, 1), real=False)
        if begin < 512:
            pair_count = min(len(original), 512 - begin) // 2
            for name, images in [("baseline", original), ("repair", repaired)]:
                diversity[name].append(perceptual(images[:pair_count * 2:2], images[1:pair_count * 2:2]).flatten().cpu())
        latents.append(repaired_z.cpu().half())
        differences.append((repaired_z - z).square().mean((1, 2, 3)).cpu())
        if begin == 0:
            grid = torch.stack([original[:16], repaired[:16]], 1).flatten(0, 1)
            save_image((grid + 1) / 2, output / "paired-images.png", nrow=8)
    result = {"samples": len(packed), "paired_source_tokens": True, "latent_mse_change": float(torch.cat(differences).mean())}
    for name, metric in [("baseline", original_metric), ("repair", repaired_metric)]:
        fid, _, _ = metric.compute()
        n = len(packed)
        trace = (metric.fake_cross.diag().sum() - metric.fake_sum.square().sum() / n) / (n - 1)
        pair_distances = torch.cat(diversity[name])
        result[name] = {"fid": fid, "inception_covariance_trace": float(trace),
                        "image_pair_lpips": float(pair_distances.mean()), "diversity_pairs": len(pair_distances)}
        log({"phase": "generation_evaluation", "method": name, **result[name]})
    result["fid_delta"] = result["repair"]["fid"] - result["baseline"]["fid"]
    result["diversity_note"] = "Covariance trace is only a dispersion diagnostic, not a test of mode coverage."
    torch.save({"repaired_latents": torch.cat(latents), "note": "Continuous latents, not re-quantized sparse tokens"}, output / "repaired-latents.pt")
    write_json(output / "metrics.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--cache", type=Path, default=ROOT / "outputs/lsun-church-bar-20260910/church-cache.pt")
    p.add_argument("--stage1", type=Path, default=ROOT / "outputs/lsun-church-sign-pattern-20260910/assets/best_rfid_slot1_model.pt")
    p.add_argument("--generated-codes", type=Path, default=ROOT / "outputs/lsun-church-bar-20260910/source-baseline/generated-codes.pt")
    p.add_argument("--fid-stats", type=Path, default=ROOT / "outputs/lsun-church-bar-20260910/assets/lsun_256_church.npz")
    p.add_argument("--rollout", type=Path, default=ROOT / "outputs/lsun-church-atom-errors-20260910/main/rollout-sample.pt")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--microbatch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-images", type=int, default=256)
    p.add_argument("--fid-samples", type=int, default=2048)
    p.add_argument("--seed", type=int, default=6801)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after-step", type=int, default=0)
    p.add_argument("--wandb-id")
    args = p.parse_args()
    if args.batch_size % 4 or min(args.steps, args.microbatch, args.eval_every) < 1:
        p.error("Batch must be divisible by four; step counts must be positive")
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    torch.serialization.add_safe_globals([codecs.encode])
    cache = torch.load(args.cache, weights_only=False, map_location="cpu")
    splits = quantized_splits(cache)
    pairs = {name: torch.load(args.pairs / (name + ".pt"), weights_only=False, map_location="cpu") for name in ["train", "holdout"]}
    stage1_hash = sha256_file(args.stage1)
    assert stage1_hash == cache["meta"]["checkpoint_sha256"]
    for name, data in pairs.items():
        assert data["config"]["tokenizer_sha256"] == stage1_hash
        assert data["keys"] == [cache[name]["keys"][i] for i in data["indices"].tolist()]
    assert not set(pairs["train"]["keys"]) & set(pairs["holdout"]["keys"])
    aux = LaserAux(args.stage1, 16384, 2048, 20, coeff_scales=cache["meta"]["coeff_scales"], sparsity_level=4).cuda().eval()
    perceptual = LPIPS().cuda().eval()
    versions = {name: parameter._version for name, parameter in aux.named_parameters()}
    buffer_versions = {name: value._version for name, value in aux.named_buffers()}
    repair = SparseLatentRepair().cuda()
    repair.set_normalization(latent(aux, splits["train"][:2048].cuda()))
    neighbors = nearest_atoms(aux.dictionary)
    optimizer = torch.optim.AdamW(repair.parameters(), lr=args.lr, weight_decay=.01, betas=(.9, .95), fused=True)
    assert not {id(p) for p in repair.parameters()} & {id(p) for p in aux.parameters()}
    cpu_rng = torch.Generator().manual_seed(args.seed + 1)
    gpu_rng = torch.Generator(device="cuda").manual_seed(args.seed + 2)
    config = {**vars(args), "trainable_parameters": sum(p.numel() for p in repair.parameters()),
              "architecture": repair.config, "stage1_sha256": stage1_hash,
              "frozen_components": ["encoder", "dictionary", "coefficient_bins", "decoder", "AR_prior"],
              "target": "Clean quantized latent and its frozen-decoder reconstruction; no original-image enhancement objective",
              "batch_mix": "25% clean, 25% synthetic errors, 50% frozen-AR span errors",
              "loss": "Normalized latent MSE + 0.5 LPIPS + 0.1 pixel L1; clean examples weighted 4",
              "selection": "Minimum holdout noisy LPIPS + 2 * clean LPIPS, subject to clean LPIPS <= 0.01; identity eligible",
              "holdout_scope": "Repair training excludes holdout and validation; base tokenizer and AR previously saw the training population",
              "pairs_sha256": {name: sha256_file(args.pairs / (name + ".pt")) for name in pairs},
              "script_sha256": {name: sha256_file(ROOT / name) for name in ["scripts/train_church_latent_repair.py", "src/sparse_latent_repair.py"]}}
    step, best_score, best_step = 0, float("inf"), 0
    if args.resume:
        saved = torch.load(args.output / "last.pt", weights_only=False, map_location="cpu")
        for key in ["steps", "batch_size", "microbatch", "lr", "seed", "stage1_sha256", "pairs_sha256", "script_sha256"]:
            if saved["config"][key] != config[key]:
                raise ValueError("Resume invariant changed: " + key)
        repair.load_state_dict(saved["state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        cpu_rng.set_state(saved["cpu_rng"])
        gpu_rng.set_state(saved["gpu_rng"])
        step, best_score, best_step = saved["step"], saved["best_score"], saved["best_step"]
        del saved
    write_json(args.output / "config.json", config)
    wb = None
    if args.wandb_id:
        import wandb
        wb = wandb.init(entity="helloimlixin-rutgers", project="laser", id=args.wandb_id,
                        name=args.wandb_id, group="church-neural-repair-20260910", job_type="frozen-base-latent-denoiser",
                        dir=str(args.output), config=json.loads(json.dumps(config, default=str)), resume="must" if args.resume else "never")
    started = time.monotonic()
    stop = {"signal": None}
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, lambda number, frame: stop.update(signal=number))

    def log(row):
        row = {"optimizer_step": step, "seconds": time.monotonic() - started, **row}
        print(json.dumps(row), flush=True)
        with (args.output / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        write_json(args.output / "status.json", {"pid": os.getpid(), "planned_steps": args.steps, "best_step": best_step, **row})
        if wb:
            wb.log(row)

    def save_last():
        atomic_torch_save({"state_dict": repair.state_dict(), "optimizer": optimizer.state_dict(),
                           "cpu_rng": cpu_rng.get_state(), "gpu_rng": gpu_rng.get_state(),
                           "step": step, "best_score": best_score, "best_step": best_step, "config": config}, args.output / "last.pt")

    hold = pairs["holdout"]
    hn = min(args.eval_images, len(hold["indices"]))
    # Uniformly span the saved batches, covering all span sizes and both modes.
    hi = torch.linspace(0, len(hold["indices"]) - 1, hn).round().long()
    clean_holdout = splits["holdout"][hold["indices"][hi]]
    conditions = {"holdout-clean": (clean_holdout, clean_holdout),
                  "holdout-ar-errors": (hold["corrupted"][hi], clean_holdout)}

    def assess():
        nonlocal best_score, best_step
        metrics = reconstruction_evaluation(repair, aux, perceptual, conditions,
                                            args.output / f"evaluation-{step:06d}", args.microbatch, log)
        clean = metrics["holdout-clean"]["repair/lpips"]
        score = metrics["holdout-ar-errors"]["repair/lpips"] + 2 * clean
        if clean <= .01 and score < best_score:
            best_score, best_step = score, step
            atomic_torch_save({"state_dict": repair.state_dict(), "config": config, "step": step, "score": score}, args.output / "best.pt")
        log({"phase": "checkpoint_selection", "holdout_score": score, "clean_lpips": clean,
             "best_step": best_step, "best_score": best_score})
        save_last()

    log({"phase": "ready", "trainable_parameters": config["trainable_parameters"], "resumed": args.resume})
    if not args.resume:
        assess()
    try:
        while step < args.steps:
            step += 1
            repair.train()
            quarter = args.batch_size // 4
            selected = torch.randint(len(splits["train"]), (quarter * 2,), generator=cpu_rng)
            clean = splits["train"][selected[:quarter]].cuda()
            synthetic_target = splits["train"][selected[quarter:]].cuda()
            synthetic = synthetic_corruption(synthetic_target, neighbors, gpu_rng)
            chosen = torch.randint(len(pairs["train"]["indices"]), (quarter * 2,), generator=cpu_rng)
            replay_input = pairs["train"]["corrupted"][chosen].cuda().long()
            replay_target = splits["train"][pairs["train"]["indices"][chosen]].cuda()
            inputs = torch.cat([clean, synthetic, replay_input])
            targets = torch.cat([clean, synthetic_target, replay_target])
            weights = torch.ones(args.batch_size, device="cuda")
            weights[:quarter] = 4
            warmup = min(100, max(1, args.steps // 10))
            if step <= warmup:
                lr = args.lr * step / warmup
            else:
                progress = (step - warmup) / max(args.steps - warmup, 1)
                lr = args.lr * (.1 + .9 * (1 + math.cos(math.pi * progress)) / 2)
            optimizer.param_groups[0]["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            totals = {"loss": 0., "latent": 0., "lpips": 0., "pixel": 0.}
            iteration_start = time.monotonic()
            for begin in range(0, args.batch_size, args.microbatch):
                noisy_z = latent(aux, inputs[begin:begin + args.microbatch])
                clean_z = latent(aux, targets[begin:begin + args.microbatch])
                with torch.no_grad():
                    target_image = decode(aux, clean_z)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    repaired_z = repair(noisy_z)
                repaired_image = decode(aux, repaired_z.float())
                latent_loss = ((repaired_z - clean_z) / repair.scale).square().mean((1, 2, 3))
                perceptual_loss = perceptual(repaired_image, target_image).flatten()
                pixel_loss = (repaired_image - target_image).abs().mean((1, 2, 3))
                loss = ((latent_loss + .5 * perceptual_loss + .1 * pixel_loss) * weights[begin:begin + len(clean_z)]).sum() / args.batch_size
                loss.backward()
                totals["loss"] += float(loss.detach())
                for name, value in [("latent", latent_loss), ("lpips", perceptual_loss), ("pixel", pixel_loss)]:
                    totals[name] += float(value.detach().sum()) / args.batch_size
            gradient_norm = torch.nn.utils.clip_grad_norm_(repair.parameters(), 1.)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Repair gradient became nonfinite")
            optimizer.step()
            if step == 1 or step % 25 == 0:
                log({"phase": "train", "lr": lr, "gradient_norm": float(gradient_norm),
                     "step_seconds": time.monotonic() - iteration_start, **{f"train/{k}": v for k, v in totals.items()}})
            if step % args.eval_every == 0 or step == args.steps:
                assess()
            if stop["signal"] is not None or args.stop_after_step and step >= args.stop_after_step:
                save_last()
                log({"phase": "paused", "signal": stop["signal"]})
                if wb:
                    wb.finish()
                return
        save_last()
        last_weights = {k: v.detach().cpu() for k, v in repair.state_dict().items()}
        atomic_torch_save({"state_dict": last_weights, "config": config, "step": step}, args.output / "final.pt")
        best = torch.load(args.output / "best.pt", weights_only=False, map_location="cpu")
        repair.load_state_dict(best["state_dict"], strict=True)
        repair.eval()
        final_conditions = {"validation-clean": (splits["validation"], splits["validation"])}
        rollout = torch.load(args.rollout, weights_only=True, map_location="cpu")
        local = torch.load(args.rollout.parent / "local.pt", weights_only=True, map_location="cpu")
        assert torch.equal(local["packed"], splits["validation"])
        for site in rollout["sites"]:
            case = site["conditions"]["control"]
            final_conditions[f"validation-rollout-site{site['site']}"] = (case["atoms"].long() * 2048 + case["coefficient_ids"].long(), splits["validation"])
        final_recon = reconstruction_evaluation(repair, aux, perceptual, final_conditions,
                                                args.output / "selected-reconstruction", args.microbatch, log)
        codes = torch.load(args.generated_codes, weights_only=True, map_location="cpu")
        codes = {key: value[:args.fid_samples] for key, value in codes.items()}
        generation = generation_evaluation(repair, aux, perceptual, codes, args.fid_stats, args.output / "selected-generation", 16, log)
        final_checkpoint = None
        if best_step != step:
            repair.load_state_dict(last_weights, strict=True)
            final_checkpoint = {"step": step,
                "reconstruction": reconstruction_evaluation(repair, aux, perceptual, final_conditions,
                    args.output / "final-reconstruction", args.microbatch, log),
                "generation": generation_evaluation(repair, aux, perceptual, codes, args.fid_stats,
                    args.output / "final-generation", 16, log)}
        frozen_check = all(p._version == versions[name] and p.grad is None and not p.requires_grad for name, p in aux.named_parameters())
        frozen_check &= all(value._version == buffer_versions[name] for name, value in aux.named_buffers())
        assert frozen_check
        result = {"selected_step": best_step, "trained_steps": step, "selected_holdout_score": best_score,
                  "frozen_base_weights_verified": frozen_check, "reconstruction": final_recon, "generation": generation,
                  "final_checkpoint": final_checkpoint,
                  "note": "FID-2048 is a screen, not a confirmed FID-50000 improvement; selected checkpoint uses holdout reconstruction only"}
        write_json(args.output / "results.json", result)
        log({"phase": "complete", "selected_step": best_step, "fid_delta": generation["fid_delta"], "frozen_base_weights_verified": frozen_check})
        if wb:
            wb.summary.update({"selected_step": best_step, "baseline_fid": generation["baseline"]["fid"],
                               "repair_fid": generation["repair"]["fid"], "frozen_base_weights_verified": frozen_check})
            artifact = wandb.Artifact(args.wandb_id + "-repair", type="model")
            artifact.add_file(str(args.output / "best.pt"))
            artifact.add_file(str(args.output / "results.json"))
            wb.log_artifact(artifact)
            wb.log({"paired_samples": wandb.Image(str(args.output / "selected-generation/paired-images.png"))})
            wb.finish()
    except BaseException:
        # Keep the last successfully completed training checkpoint intact.
        if wb:
            wb.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
