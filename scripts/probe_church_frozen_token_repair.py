#!/usr/bin/env python3
"""Test one frozen decode/encode/quantize cycle; no optimizer or model updates.

This is a consistency baseline, not a trained denoiser or an ECC guarantee.
Generated samples are paired by using exactly the same saved AR tokens.
"""

import argparse
import codecs
import json
from pathlib import Path
import sys
import time

import torch
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_church_bar_coefficients import LaserAux, quantized_splits
from scripts.tools.build_sign_probe_cache import sha256_file
from src.models.lpips import LPIPS
from src.rqvae_metrics import DistributedOriginalRQVAEMetrics


def write_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str))
    temporary.replace(path)


@torch.no_grad()
def cycle(aux, image):
    # Match the original cache encoder/OMP/binning precision exactly.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        atoms, coefficients = aux.encode_sparse_components(image)
    coefficients = coefficients.half().float()
    ids = (coefficients[..., None] - aux.coeff_bins).abs().argmin(-1)
    return atoms, ids


@torch.no_grad()
def reconstruction_probe(aux, perceptual, atoms, ids, truth, batch_size, output, log):
    rows = []
    for start in range(0, len(atoms), batch_size):
        a, c = atoms[start:start + batch_size].cuda().long(), ids[start:start + batch_size].cuda().long()
        reference = truth[start:start + batch_size].cuda().long()
        ta, tc = reference // 2048, reference % 2048
        target_z = aux.compound_embeddings(ta, tc).sum(-2)
        target = aux.decode_compound(ta, tc)
        original = aux.decode_compound(a, c)
        ra, rc = cycle(aux, original)
        repaired = aux.decode_compound(ra, rc)
        row = {}
        for name, image, current_a, current_c in [("baseline", original, a, c), ("cycle", repaired, ra, rc)]:
            z = aux.compound_embeddings(current_a, current_c).sum(-2)
            row[name + "/latent_mse"] = (z - target_z).square().mean((1, 2, 3)).cpu()
            mse = ((image - target) / 2).square().mean((1, 2, 3))
            row[name + "/psnr_to_clean_reconstruction"] = -10 * mse.clamp_min(1e-12).log10().cpu()
            row[name + "/lpips_to_clean_reconstruction"] = perceptual(image, target).flatten().cpu()
        row["cycle/ordered_atom_change_fraction"] = (ra != a).float().mean((1, 2, 3)).cpu()
        row["cycle/ordered_coefficient_change_fraction"] = (rc != c).float().mean((1, 2, 3)).cpu()
        rows.append(row)
        if start == 0:
            grid = torch.stack([target[:8], original[:8], repaired[:8]], 1).flatten(0, 1)
            save_image((grid + 1) / 2, output.with_suffix(".png"), nrow=3)
        if start == 0 or start + batch_size >= len(atoms):
            log({"phase": "reconstruction", "condition": output.name, "images": min(start + batch_size, len(atoms))})
    per_image = {key: torch.cat([row[key] for row in rows]) for key in rows[0]}
    torch.save(per_image, output.with_suffix(".pt"))
    result = {key: float(value.mean()) for key, value in per_image.items()}
    result["images"] = len(atoms)
    for metric in ["latent_mse", "psnr_to_clean_reconstruction", "lpips_to_clean_reconstruction"]:
        difference = per_image["cycle/" + metric] - per_image["baseline/" + metric]
        result["paired_delta/" + metric] = float(difference.mean())
        result["paired_standard_error/" + metric] = float(difference.std() / len(difference) ** .5)
    return result


@torch.no_grad()
def generation_probe(aux, codes, args, log):
    baseline = DistributedOriginalRQVAEMetrics("cuda", reference_stats_path=args.fid_stats)
    corrected = DistributedOriginalRQVAEMetrics("cuda", reference_stats_path=args.fid_stats, inception=baseline.inception)
    saved_atoms, saved_ids = [], []
    n = min(args.fid_samples, len(codes["atoms"]))
    for start in range(0, n, args.batch_size):
        atoms = codes["atoms"][start:start + min(args.batch_size, n-start)].long().cuda()
        ids = codes["coefficient_ids"][start:start + min(args.batch_size, n-start)].long().cuda()
        image = aux.decode_compound(atoms, ids)
        new_atoms, new_ids = cycle(aux, image)
        new_image = aux.decode_compound(new_atoms, new_ids)
        baseline.update(((image + 1) / 2).clamp(0, 1), real=False)
        corrected.update(((new_image + 1) / 2).clamp(0, 1), real=False)
        saved_atoms.append(new_atoms.cpu().short())
        saved_ids.append(new_ids.cpu().short())
        if start == 0:
            grid = torch.stack([image[:16], new_image[:16]], 1).flatten(0, 1)
            save_image((grid + 1) / 2, args.output / "paired-unconditional.png", nrow=8)
        if start == 0 or (start + len(atoms)) % 256 == 0 or start + len(atoms) == n:
            log({"phase": "generation_decode", "images": start + len(atoms), "total": n})
    torch.save({"atoms": torch.cat(saved_atoms), "coefficient_ids": torch.cat(saved_ids)}, args.output / "cycle-generated-codes.pt")
    result = {"samples": n, "same_source_tokens": True, "source": str(args.generated_codes)}
    for name, metric in [("baseline", baseline), ("cycle", corrected)]:
        fid, _, _ = metric.compute()
        # A dispersion diagnostic only; this does not establish mode coverage.
        covariance_trace = (metric.fake_cross.diag().sum() - metric.fake_sum.square().sum() / n) / (n - 1)
        result[name] = {"fid": fid, "inception_covariance_trace": float(covariance_trace)}
        log({"phase": "generation_metric", "method": name, **result[name]})
    result["fid_delta_cycle_minus_baseline"] = result["cycle"]["fid"] - result["baseline"]["fid"]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stage1", type=Path, default=ROOT / "outputs/lsun-church-sign-pattern-20260910/assets/best_rfid_slot1_model.pt")
    p.add_argument("--cache", type=Path, default=ROOT / "outputs/lsun-church-bar-20260910/church-cache.pt")
    p.add_argument("--rollout", type=Path, default=ROOT / "outputs/lsun-church-atom-errors-20260910/main/rollout-sample.pt")
    p.add_argument("--generated-codes", type=Path, default=ROOT / "outputs/lsun-church-bar-20260910/source-baseline/generated-codes.pt")
    p.add_argument("--fid-stats", type=Path, default=ROOT / "outputs/lsun-church-bar-20260910/assets/lsun_256_church.npz")
    p.add_argument("--images", type=int, default=300)
    p.add_argument("--fid-samples", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--phase", choices=["reconstruction", "generation", "all"], default="all")
    args = p.parse_args()
    if min(args.images, args.batch_size, args.fid_samples) < 2:
        p.error("Counts must be at least two")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.manual_seed(5701)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.serialization.add_safe_globals([codecs.encode])
    cache = torch.load(args.cache, weights_only=False, map_location="cpu")
    checkpoint_hash = sha256_file(args.stage1)
    assert checkpoint_hash == cache["meta"]["checkpoint_sha256"]
    aux = LaserAux(args.stage1, 16384, 2048, 20, coeff_scales=cache["meta"]["coeff_scales"], sparsity_level=4).cuda().eval()
    assert all(not p.requires_grad for p in aux.parameters())
    config = {**vars(args), "stage1_sha256": checkpoint_hash, "updated_model_parameters": 0,
              "method": "One frozen D -> E -> OMP4 -> 2048-bin quantization -> D pass",
              "source_codes_sha256": sha256_file(args.generated_codes),
              "script_sha256": sha256_file(Path(__file__)),
              "note": "Conditional rollouts keep true atom supports; FID uses unconditional samples. No training or hyperparameter search."}
    write_json(args.output / "config.json", config)
    started = time.monotonic()

    def log(row):
        row = {"seconds": time.monotonic() - started, **row}
        print(json.dumps(row), flush=True)
        with (args.output / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        write_json(args.output / "status.json", row)

    result = {"config": config}
    if args.phase in ["reconstruction", "all"]:
        packed = quantized_splits(cache)["validation"][:args.images]
        perceptual = LPIPS().cuda().eval()
        rollout = torch.load(args.rollout, weights_only=True, map_location="cpu")
        local = torch.load(args.rollout.parent / "local.pt", weights_only=True, map_location="cpu")
        assert rollout["provenance"]["tokenizer_sha256"] == checkpoint_hash
        assert torch.equal(local["packed"][:len(packed)], packed), "Rollout/reference image order differs"
        result["clean"] = reconstruction_probe(aux, perceptual, packed // 2048, packed % 2048, packed,
                                                args.batch_size, args.output / "clean", log)
        for site in rollout["sites"]:
            condition = site["conditions"]["control"]
            name = f"rollout-site{site['site']}"
            result[name] = reconstruction_probe(aux, perceptual, condition["atoms"][:len(packed)],
                          condition["coefficient_ids"][:len(packed)], packed, args.batch_size, args.output / name, log)
        del perceptual
    if args.phase in ["generation", "all"]:
        codes = torch.load(args.generated_codes, weights_only=True, map_location="cpu")
        result["generation"] = generation_probe(aux, codes, args, log)
    write_json(args.output / "summary.json", result)
    log({"phase": "complete", "updated_model_parameters": 0})


if __name__ == "__main__":
    main()
