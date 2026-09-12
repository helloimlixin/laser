#!/usr/bin/env python3
"""Matched Church continuation with categorical or BAR-inspired coefficients."""

import argparse
import codecs
import gc
import getpass
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_official_rqtransformer_laser_stage2 import (
    CompoundLaserRQTransformer, LaserAux, build_model, atomic_torch_save,
)
from src.masked_coefficient_head import MaskedCoefficientHead
from src.models.rqtransformer.transformers import sample_from_logits
from src.rqvae_metrics import DistributedOriginalRQVAEMetrics
from scripts.tools.build_sign_probe_cache import sha256_file


class CoefficientExperimentPrior(CompoundLaserRQTransformer):
    def __init__(self, config, num_atoms, coeff_vocab_size, mode="bar", width=512, **kwargs):
        super().__init__(config, num_atoms, coeff_vocab_size, **kwargs)
        self.coefficient_mode = mode
        if mode == "bar":
            self.bar_head = MaskedCoefficientHead(config.embed_dim, width=width,
                                                  bits=int(math.log2(coeff_vocab_size)))
            if 2 ** self.bar_head.bits != coeff_vocab_size:
                raise ValueError("BAR experiment requires power-of-two bins")

    def classify_head_outputs(self, head_outputs):
        atoms = self._teacher_atoms
        refined = self.refine_coefficient_hidden(
            head_outputs, self._model_aux.dictionary.t()[atoms.long()])
        return {
            "atom_logits": self.mask_seen_atoms(self.classifier(head_outputs), atoms),
            "coefficient_context": refined,
        }

    def coefficient_loss(self, refined, targets, generator=None):
        if self.coefficient_mode == "categorical":
            logits = self.classify_coefficients(refined).float()
            return F.cross_entropy(logits.flatten(0, -2), targets.flatten()), {}
        depth = torch.arange(refined.shape[-2], device=refined.device).expand(refined.shape[:-1])
        loss, metrics = self.bar_head.loss(refined, targets, depth, generator)
        # Vector-scale surrogate; this is not exact categorical token NLL.
        return loss * self.bar_head.bits, metrics

    def choose_coefficient(self, refined, depth_index=None, greedy=False, generator=None):
        if self.coefficient_mode == "categorical":
            logits = self.classify_coefficients(refined, depth_index).float()
            if greedy:
                return logits.argmax(-1)
            return torch.multinomial(logits.softmax(-1).reshape(-1, logits.shape[-1]),
                                     1, generator=generator).reshape(logits.shape[:-1])
        if depth_index is None:
            depth = torch.arange(refined.shape[-2], device=refined.device).expand(refined.shape[:-1])
        else:
            depth = torch.full(refined.shape[:-1], depth_index, device=refined.device, dtype=torch.long)
        return self.bar_head.sample(refined, depth, greedy=greedy, generator=generator)

    @torch.no_grad()
    def generate(self, aux, batch, forced_atoms=None, greedy_coefficients=False):
        h, w, depth = self.block_size
        device = next(self.parameters()).device
        atoms = torch.zeros((batch, h, w, depth), device=device, dtype=torch.long)
        coefficients = torch.full_like(atoms, self.coeff_vocab_size // 2)
        packed = atoms * self.coeff_vocab_size + coefficients
        self.init_cache()
        try:
            for row in range(h):
                for col in range(w):
                    for d in range(depth):
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                            hidden = self.cached_head_output(packed, aux, None, (row, col, d), amp=False)
                            if forced_atoms is None:
                                logits = self.classifier(hidden).float()
                                if d:
                                    logits.scatter_(1, atoms[:, row, col, :d], -torch.inf)
                                atom = sample_from_logits(logits, temperature=1, top_k=min(250, self.num_atoms), top_p=1)
                            else:
                                atom = forced_atoms[:, row, col, d]
                            refined = self.refine_coefficient_hidden(hidden, aux.dictionary.t()[atom])
                            coefficient = self.choose_coefficient(refined, d, greedy_coefficients)
                        atoms[:, row, col, d] = atom
                        coefficients[:, row, col, d] = coefficient
                        packed[:, row, col, d] = atom * self.coeff_vocab_size + coefficient
        finally:
            self.init_cache()
        return atoms, coefficients


def make_model(checkpoint, mode, width=512):
    # Source checkpoint is the authenticated, user-owned W&B model snapshot.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    base = build_model(18432, 16384, compound=True, coeff_vocab_size=2048,
                       compound_micro_transformer_layers=2,
                       compound_depth_specific_coeff_heads=True,
                       compound_pair_autoregressive=True,
                       sparsity_level=4, model_preset="lsun-church-350m")
    config = base.config
    del base
    model = CoefficientExperimentPrior(config, 16384, 2048, mode=mode, width=width,
                                      micro_transformer_layers=2,
                                      depth_specific_coeff_heads=True,
                                      pair_autoregressive=True)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if unexpected or any(not name.startswith("bar_head.") for name in missing):
        raise RuntimeError(f"checkpoint mismatch: {missing}, {unexpected}")
    return model, {k: payload[k] for k in ["epoch", "global_step", "fid"] if k in payload}


def quantized_splits(cache):
    bins = torch.linspace(-cache["meta"]["coeff_max"], cache["meta"]["coeff_max"], 2048)
    scales = torch.tensor(cache["meta"]["coeff_scales"])
    splits = {}
    for name in ["train", "holdout", "validation"]:
        data = cache[name]
        c = data["coefficients"] / scales
        ids = ((c + 20) * (2047 / 40)).round().long().clamp(0, 2047)
        assert torch.allclose(bins[ids] * scales, data["coefficients"], atol=2e-6)
        splits[name] = data["atoms"].long() * 2048 + ids
    return splits


@torch.no_grad()
def evaluate(model, aux, packed, batch_size=32):
    model.eval()
    records = []
    for chunk in packed.split(batch_size):
        chunk = chunk.cuda()
        atoms, truth = model.unpack(chunk)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(chunk, model_aux=aux)
            prediction = model.choose_coefficient(outputs["coefficient_context"], greedy=True)
        true_c = aux.coeff_bins[truth] * aux.coeff_scales
        pred_c = aux.coeff_bins[prediction] * aux.coeff_scales
        atom_nll = F.cross_entropy(outputs["atom_logits"].float().flatten(0, -2), atoms.flatten(), reduction="none").reshape_as(atoms)
        error = pred_c - true_c
        latent_error = (aux.dictionary.t()[atoms] * error.unsqueeze(-1)).sum(-2)
        row = {
            "atom_nll": atom_nll.mean((1, 2, 3)),
            "coefficient_accuracy": (prediction == truth).float().mean((1, 2, 3)),
            "sign_accuracy": ((prediction >= 1024) == (truth >= 1024)).float().mean((1, 2, 3)),
            "physical_coefficient_mae": error.abs().mean((1, 2, 3)),
            "latent_mse": latent_error.square().mean((1, 2, 3)),
        }
        for d in range(4):
            row[f"sign_accuracy_d{d}"] = ((prediction[..., d] >= 1024) == (truth[..., d] >= 1024)).float().mean((1, 2))
        if model.coefficient_mode == "categorical":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model.classify_coefficients(outputs["coefficient_context"])
            row["coefficient_nll"] = F.cross_entropy(logits.float().flatten(0, -2), truth.flatten(), reduction="none").reshape_as(truth).mean((1, 2, 3))
        records.append({k: v.cpu() for k, v in row.items()})
    per_image = {k: torch.cat([r[k] for r in records]) for k in records[0]}
    return {k: float(v.mean()) for k, v in per_image.items()}, per_image


@torch.no_grad()
def generation_screen(model, aux, args, log):
    model.eval()
    torch.manual_seed(args.seed + 10000)
    metric = DistributedOriginalRQVAEMetrics("cuda", reference_stats_path=args.fid_stats)
    saved_atoms, saved_coefficients = [], []
    start = time.monotonic()
    for offset in range(0, args.fid_samples, args.generation_batch):
        n = min(args.generation_batch, args.fid_samples - offset)
        atoms, coeff = model.generate(aux, n)
        saved_atoms.append(atoms.cpu().short())
        saved_coefficients.append(coeff.cpu().short())
        # Decode in FP32, matching the established September FID evaluator.
        for start_index in range(0, n, 32):
            images = (aux.decode_compound(atoms[start_index:start_index+32], coeff[start_index:start_index+32]).float() + 1) / 2
            images = images.clamp(0, 1)
            metric.update(images, real=False)
            if offset == 0 and start_index == 0:
                save_image(images, args.output / "samples.png", nrow=8)
        log({"phase": "generation", "generated": offset + n, "seconds": time.monotonic() - start})
    fid, _, _ = metric.compute()
    torch.save({"atoms": torch.cat(saved_atoms), "coefficient_ids": torch.cat(saved_coefficients)}, args.output / "generated-codes.pt")
    return {"fid": fid, "samples": args.fid_samples, "seed": args.seed + 10000,
            "seconds": time.monotonic() - start, "atom_top_k": 250,
            "atom_temperature": 1.0, "coefficient_temperature": 1.0,
            "ar_precision": "bfloat16", "decoder_precision": "float32"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["bar", "categorical"], required=True)
    p.add_argument("--source-checkpoint", type=Path, required=True)
    p.add_argument("--stage1", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fid-stats", type=Path)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--head-only-steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--head-lr", type=float, default=3e-4)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=2701)
    p.add_argument("--fid-samples", type=int, default=2048)
    p.add_argument("--generation-batch", type=int, default=128)
    p.add_argument("--wandb-id")
    p.add_argument("--wandb-key-stdin", action="store_true")
    p.add_argument("--evaluate-checkpoint", type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    if args.wandb_key_stdin:
        os.environ["WANDB_API_KEY"] = getpass.getpass("W&B API key: ")
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.serialization.add_safe_globals([codecs.encode])
    cache = torch.load(args.cache, weights_only=False, map_location="cpu")
    splits = quantized_splits(cache)
    metadata = cache["meta"]
    del cache
    model, source = make_model(args.source_checkpoint, args.mode, args.width)
    if args.evaluate_checkpoint:
        model.load_state_dict(torch.load(args.evaluate_checkpoint, map_location="cpu", weights_only=False)["state_dict"], strict=True)
    aux = LaserAux(args.stage1, 16384, 2048, 20, coeff_scales=metadata["coeff_scales"], sparsity_level=4).cuda().eval()
    aux.requires_grad_(False)
    model.cuda()
    config = {**vars(args), "source": source, "tokenizer_sha256": sha256_file(args.stage1),
              "source_checkpoint_sha256": sha256_file(args.source_checkpoint), "cache_metadata": metadata,
              "train_images": len(splits["train"]), "holdout_note": "Excluded from continuation; original pretrained AR saw the whole training population.",
              "bar_schedule": [2, 3, 3, 3], "bar_loss": "masked bit BCE times 11; not exact token NLL",
              "checkpoint_selection": "fixed final step; no validation-based selection"}
    assert config["tokenizer_sha256"] == metadata["checkpoint_sha256"]
    (args.output / "config.json").write_text(json.dumps(config, default=str, indent=2))
    wb = None
    if args.wandb_id:
        import wandb
        wb = wandb.init(entity="helloimlixin-rutgers", project="laser", id=args.wandb_id,
                        name=args.wandb_id, group="church-bar-coefficients-20260910",
                        job_type="matched-coefficient-head-continuation", dir=str(args.output),
                        config=json.loads(json.dumps(config, default=str)), resume="never")

    def log(row):
        print(json.dumps(row), flush=True)
        with (args.output / "history.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        if wb:
            wb.log(row)

    head_names = ("bar_head.",) if args.mode == "bar" else ("coeff_classifier.",)
    output_parameters, backbone_parameters = [], []
    for name, param in model.named_parameters():
        if args.mode == "bar" and name.startswith("coeff_classifier."):
            param.requires_grad_(False)
        elif name.startswith(head_names):
            output_parameters.append(param)
        else:
            backbone_parameters.append(param)
    optimizer = torch.optim.AdamW([
        {"params": output_parameters, "lr": args.head_lr},
        {"params": backbone_parameters, "lr": args.backbone_lr},
    ], betas=(.9, .95), weight_decay=.03)
    for param in backbone_parameters:
        param.requires_grad_(False)
    sampler = torch.Generator().manual_seed(args.seed + 1)
    masks = torch.Generator(device="cuda").manual_seed(args.seed + 2)
    permutation = torch.randperm(len(splits["train"]), generator=sampler)
    position = 0
    start = time.monotonic()
    log({"phase": "ready", "parameters": sum(p.numel() for p in model.parameters()), "source_epoch": source["epoch"]})
    initial, _ = evaluate(model, aux, splits["validation"][:64])
    log({"phase": "initial_validation64", **initial})
    for step in range(1, args.steps + 1):
        if step == args.head_only_steps + 1:
            for param in backbone_parameters:
                param.requires_grad_(True)
        model.train()
        # Fixed pretrained context during head warmup, without dropout drift.
        if step <= args.head_only_steps:
            model.eval()
            if args.mode == "bar":
                model.bar_head.train()
        if position + args.batch_size > len(permutation):
            permutation = torch.randperm(len(splits["train"]), generator=sampler)
            position = 0
        selected = permutation[position:position + args.batch_size]
        position += args.batch_size
        packed = splits["train"][selected].cuda()
        atoms, ids = model.unpack(packed)
        # Reset the main RNG each step so matched arms use identical dropout.
        torch.manual_seed(args.seed + step)
        multiplier = .2 + .8 * .5 * (1 + math.cos(math.pi * (step - 1) / max(args.steps, 1)))
        optimizer.param_groups[0]["lr"] = args.head_lr * multiplier
        optimizer.param_groups[1]["lr"] = args.backbone_lr * multiplier
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(packed, model_aux=aux)
            coeff_loss, metrics = model.coefficient_loss(result["coefficient_context"], ids, masks)
            atom_loss = F.cross_entropy(result["atom_logits"].float().flatten(0, -2), atoms.flatten())
            loss = (coeff_loss + atom_loss) / 2
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError(f"nonfinite gradient at step {step}")
        optimizer.step()
        if step == 1 or step % 25 == 0:
            log({"phase": "train", "optimizer_step": step, "loss": float(loss),
                 "atom_nll": float(atom_loss), "coefficient_objective": float(coeff_loss),
                 "gradient_norm": float(norm), "seconds": time.monotonic() - start,
                 **{k: float(v) for k, v in metrics.items()}})
        if step % args.eval_every == 0 or step == args.steps:
            validation, _ = evaluate(model, aux, splits["holdout"][:256])
            log({"phase": "development256", "optimizer_step": step, **validation})
    model.eval()
    atomic_torch_save({"state_dict": model.state_dict(), "config": config,
                       "optimizer_step": args.steps}, args.output / "final.pt")
    results = {}
    for split in ["holdout", "validation"]:
        values, per_image = evaluate(model, aux, splits[split])
        results[split] = values
        torch.save(per_image, args.output / f"{split}-per-image.pt")
        log({"phase": split, **values})
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    if args.fid_samples:
        results["generation"] = generation_screen(model, aux, args, log)
        log({"phase": "generation_complete", **results["generation"]})
    (args.output / "results.json").write_text(json.dumps(results, indent=2))
    if wb:
        import wandb
        for split, values in results.items():
            for key, value in values.items():
                wb.summary[f"{split}/{key}"] = value
        if (args.output / "samples.png").exists():
            wb.log({"samples": wandb.Image(str(args.output / "samples.png"))})
        wb.save(str(args.output / "results.json"), base_path=str(args.output))
        wb.finish()


if __name__ == "__main__":
    main()
