"""Compare scalar K4 sampling policies on an immutable stage-2 checkpoint.

This produces matched previews and code diagnostics, not a FID estimate or a
sampler selection. It never changes the active trainer or its configuration.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=261002)
    parser.add_argument("--samples-per-class", type=int, default=4)
    args = parser.parse_args()
    sys.path.insert(0, str(args.runtime.resolve()))
    import torch
    from omegaconf import OmegaConf
    from src.data.imagenet_labels import class_names_for_dataset
    from src.training import rqtransformer as training
    from src.training.k4_checkpoint_io import _checkpoint_upload_source

    torch.set_num_threads(8)
    torch.cuda.set_device(0)
    options = OmegaConf.load(args.recipe).options
    if options.compound_tokens or options.sparsity_level != 4:
        raise ValueError("This audit requires the noncompound K4 recipe")
    args.output.mkdir(parents=True, exist_ok=True)
    # Pin a local immutable inode so the trainer can rotate its last checkpoint.
    source = _checkpoint_upload_source(args.checkpoint)
    if source == args.checkpoint:
        raise RuntimeError("A verified local checkpoint copy is required")
    pinned = args.output / "checkpoint-input.pt"
    os.link(source, pinned)
    try:
        payload = torch.load(pinned, map_location="cpu", mmap=True, weights_only=False)
        step = int(payload["global_step"])
        with torch.device("meta"):
            model = training.build_model(
                int(options.num_atoms + options.coeff_vocab_size),
                int(options.num_atoms), compound=False,
                coeff_vocab_size=int(options.coeff_vocab_size),
                sparsity_level=4, model_preset=str(options.model_preset),
            )
        model.load_state_dict(payload["state_dict"], strict=True, assign=True)
        model = model.cuda().eval().requires_grad_(False)
        del payload
        gc.collect()
        aux = training.LaserAux(
            Path(options.checkpoint), int(options.num_atoms),
            int(options.coeff_vocab_size), float(options.coeff_max),
            coeff_scale=float(options.coeff_scale),
            coeff_scales=list(options.coeff_scales), sparsity_level=4,
            soft_target_physical=False, clamp_coeffs=False,
        ).cuda().eval().requires_grad_(False)
        cache = torch.load(options.token_cache, map_location="cpu", mmap=True, weights_only=True)
        real_coeffs = cache["coeffs"][:512].float()
        physical_real = real_coeffs * aux.coeff_scales.cpu()
        chosen = torch.tensor([883, 434, 151, 19, 645, 854, 882, 736], device="cuda")
        labels = chosen.repeat_interleave(args.samples_per_class)
        names = class_names_for_dataset("imagenet")
        policies = [
            dict(name="baseline", atom_temperature=1., coeff_temperature=1., atom_top_p=.92, coeff_top_p=.92),
            dict(name="milder_atoms", atom_temperature=.9, coeff_temperature=1., atom_top_p=.92, coeff_top_p=.92),
            dict(name="milder_coefficients", atom_temperature=1., coeff_temperature=.9, atom_top_p=.92, coeff_top_p=.92),
            dict(name="milder_both", atom_temperature=.9, coeff_temperature=.9, atom_top_p=.92, coeff_top_p=.92),
            dict(name="broader_nucleus", atom_temperature=1., coeff_temperature=1., atom_top_p=.98, coeff_top_p=.98),
        ]
        report = dict(
            checkpoint_step=step, source_checkpoint=str(source),
            seed=args.seed, labels=labels.cpu().tolist(),
            recipe_sha256=hashlib.sha256(args.recipe.read_bytes()).hexdigest(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            coefficient_target_temperature=float(options.coeff_target_temperature),
            coefficient_scales=list(options.coeff_scales),
            reference_physical_coefficient_rms=physical_real.square().mean((0, 1, 2)).sqrt().tolist(),
            interpretation="Matched previews and scalar-code diagnostics only; no FID ranking or production policy change.",
            cached_forward_parity={}, results=[],
        )
        # Compare real cached predictions with a full teacher-forced forward at
        # two spatial positions. All eight depth slots must agree causally.
        atoms = cache["atoms"][:1].long().cuda()
        coeffs = cache["coeffs"][:1].float().cuda()
        cond = cache["labels"][:1].long().cuda()
        tokens, _ = aux.sparse_targets(atoms, coeffs, stochastic=False, compact=True)
        with torch.no_grad():
            full = model(tokens, aux, cond=cond, amp=True)
            model.init_cache()
            errors = []
            for w in range(2):
                for d in range(8):
                    cached = model.cached_forward(tokens[:, :1], aux, cond=cond, amp=True, sample_loc=(0, w, d))
                    if d % 2 == 0:
                        cached = cached[:, :aux.num_atoms].float()
                        expected = full["atom_logits"][:, 0, w, d // 2].float()
                    else:
                        cached = cached[:, aux.num_atoms:].float()
                        expected = full["coeff_logits"][:, 0, w, d // 2].float()
                    assert torch.isfinite(cached).all()
                    errors.append(float((cached - expected).norm() / expected.norm().clamp_min(1e-12)))
            model.init_cache()
            report["cached_forward_parity"] = dict(relative_l2_by_position_depth=errors, max_relative_l2=max(errors))
            if max(errors) > .015:
                raise RuntimeError(f"Cached/full logits disagree: {max(errors):.6f}")
            del full, cached, expected, tokens
        del cache
        for policy in policies:
            settings = {key: value for key, value in policy.items() if key != "name"}
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            start = time.monotonic()
            with torch.no_grad():
                sampled = model.sample_sparse(len(labels), aux, cond=labels, **settings, amp=True)
                sampled_atoms = sampled[..., 0::2]
                coeff_ids = sampled[..., 1::2] - aux.num_atoms
                assert sampled_atoms.min() >= 0 and sampled_atoms.max() < aux.num_atoms
                assert coeff_ids.min() >= 0 and coeff_ids.max() < aux.coeff_vocab_size
                physical = aux.coeff_bins[coeff_ids] * aux.coeff_scales
                assert torch.isfinite(physical).all()
                duplicates = sampled_atoms.sort(-1).values.diff(dim=-1).eq(0).any(-1).float().mean()
                images = aux.decode_tokens(sampled)
                assert torch.isfinite(images).all()
                images = ((images.float().cpu() + 1) * .5).clamp(0, 1)
                filename = args.output / (policy["name"] + ".png")
                training.save_class_labeled_grid(images, chosen, names, filename, samples_per_class=args.samples_per_class)
                torch.save(dict(tokens=sampled.cpu(), labels=labels.cpu(), settings=settings, checkpoint_step=step, seed=args.seed), args.output / (policy["name"] + "-tokens.pt"))
                result = dict(
                    **policy, images=len(labels), elapsed_seconds=time.monotonic() - start,
                    grid=str(filename),
                    physical_coefficient_rms=physical.square().mean((0, 1, 2)).sqrt().cpu().tolist(),
                    physical_coefficient_mean=physical.mean((0, 1, 2)).cpu().tolist(),
                    coefficient_boundary_fraction=((coeff_ids == 0) | (coeff_ids == aux.coeff_vocab_size - 1)).float().mean().item(),
                    duplicate_support_fraction=float(duplicates),
                    peak_cuda_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
                )
            report["results"].append(result)
            (args.output / "sampling-audit.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(result), flush=True)
            del sampled, images, sampled_atoms, coeff_ids, physical
        print(json.dumps(dict(status="complete", checkpoint_step=step, policies=len(policies))), flush=True)
    finally:
        pinned.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
