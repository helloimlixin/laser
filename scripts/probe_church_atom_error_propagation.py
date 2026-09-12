#!/usr/bin/env python3
"""Controlled single-atom interventions and fixed-support coefficient rollouts.

Uses the original Church AR checkpoint without training. A changed atom is never
scored against the old atom's coefficient target. Later atom identities stay fixed.
"""

import argparse
import codecs
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_church_bar_coefficients import make_model, quantized_splits, LaserAux
from scripts.tools.build_sign_probe_cache import sha256_file


def scalar_projection_coefficient(old_vector, new_vector, old_coefficient):
    """Least-squares replacement coefficient, holding the other atoms fixed."""
    return old_coefficient * (old_vector * new_vector).sum(-1) / new_vector.square().sum(-1).clamp_min(1e-12)


def exclude_site_atoms(scores, site_atoms):
    return scores.clone().scatter_(-1, site_atoms.long(), -torch.inf)


def select_coefficients(logits, uniforms=None):
    if uniforms is None:
        return logits.argmax(-1)
    cdf = logits.float().softmax(-1).cumsum(-1).contiguous()
    return torch.searchsorted(cdf, uniforms.unsqueeze(-1).contiguous()).squeeze(-1).clamp_max(logits.shape[-1] - 1)


def restore_component(predicted, truth, component):
    """Oracle diagnostic on the symmetric 2048-bin grid, not an inference fix."""
    same_sign = (predicted >= 1024) == (truth >= 1024)
    if component == "sign":
        return torch.where(same_sign, predicted, 2047 - predicted)
    if component == "magnitude":
        return torch.where(same_sign, truth, 2047 - truth)
    if component == "full":
        return truth
    raise ValueError(component)


def physical(aux, ids):
    scales = aux.coeff_scales.repeat(ids.shape[-1] // len(aux.coeff_scales))
    return aux.coeff_bins[ids] * scales


def nearest_id(aux, value, depth):
    bins = aux.coeff_bins
    unit = value / aux.coeff_scales[depth]
    return ((unit - bins[0]) * (len(bins) - 1) / (bins[-1] - bins[0])).round().long().clamp(0, len(bins) - 1)


@torch.no_grad()
def predictions(model, aux, packed, uniforms, positions, batch_size):
    records, candidates = [], []
    normalized_dictionary = F.normalize(aux.dictionary.t().float(), dim=-1)
    for begin in range(0, len(packed), batch_size):
        chunk = packed[begin:begin + batch_size].cuda()
        atoms, targets = model.unpack(chunk)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(chunk, model_aux=aux)
            logits = model.classify_coefficients(outputs["coefficient_context"]).float().flatten(1, 3)
        probability = logits.softmax(-1)
        targets = targets.flatten(1)
        row = {
            "prediction": logits.argmax(-1),
            "sample": select_coefficients(logits, uniforms[begin:begin + len(chunk)].cuda()),
            "nll": F.cross_entropy(logits.transpose(1, 2), targets, reduction="none"),
            "positive_probability": probability[..., 1024:].sum(-1),
            "physical_mean": (probability * aux.coeff_bins).sum(-1) * aux.coeff_scales.repeat(64),
        }
        records.append({k: v.cpu() for k, v in row.items()})
        if positions:
            atom_logits = outputs["atom_logits"].float().flatten(1, 3)[:, positions]
            selected_atoms = atoms.flatten(1)[:, positions]
            site_atoms = atoms.flatten(1, 2)[:, [p // 4 for p in positions]]
            alternative = exclude_site_atoms(atom_logits, site_atoms).argmax(-1)
            similarity = normalized_dictionary[selected_atoms] @ normalized_dictionary.t()
            nearest = exclude_site_atoms(similarity, site_atoms).argmax(-1)
            candidates.append({
                "near": nearest.cpu(), "model": alternative.cpu(),
                "near_cosine": similarity.gather(-1, nearest.unsqueeze(-1)).squeeze(-1).cpu(),
                "model_cosine": similarity.gather(-1, alternative.unsqueeze(-1)).squeeze(-1).cpu(),
            })
    combined = {k: torch.cat([row[k] for row in records]) for k in records[0]}
    choices = {k: torch.cat([row[k] for row in candidates]) for k in candidates[0]} if candidates else None
    return combined, choices


def load(args):
    torch.set_num_threads(8)
    torch.serialization.add_safe_globals([codecs.encode])
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    source = args.assets / "assets/epoch50-source.pt"
    tokenizer = ROOT / "outputs/lsun-church-sign-pattern-20260910/assets/best_rfid_slot1_model.pt"
    cache = torch.load(args.assets / "church-cache.pt", map_location="cpu", weights_only=False)
    meta = cache["meta"]
    packed = quantized_splits(cache)["validation"][:args.images]
    del cache
    model, source_meta = make_model(source, "categorical")
    model.cuda().eval().requires_grad_(False)
    aux = LaserAux(tokenizer, 16384, 2048, 20, coeff_scales=meta["coeff_scales"], sparsity_level=4).cuda().eval()
    aux.requires_grad_(False)
    hashes = {"tokenizer_sha256": sha256_file(tokenizer), "source_sha256": sha256_file(source)}
    assert hashes["tokenizer_sha256"] == meta["checkpoint_sha256"]
    return model, aux, packed, {**hashes, "source": source_meta}


def log(args, row):
    print(json.dumps(row), flush=True)
    with (args.output / f"{args.phase}-{args.sampling}.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")


@torch.no_grad()
def local(args, model, aux, packed, provenance):
    positions = [4 * site + depth for site in args.sites for depth in args.depths]
    uniforms = torch.rand((len(packed), 256), generator=torch.Generator().manual_seed(args.seed))
    base, candidates = predictions(model, aux, packed, uniforms, positions, args.batch_size)
    truth_atoms, truth_ids = model.unpack(packed)
    truth_atoms, truth_ids = truth_atoms.flatten(1), truth_ids.flatten(1)
    true_c = physical(aux, truth_ids.cuda()).cpu()
    cases = []
    for j, p in enumerate(positions):
        d = p % 4
        old_vectors = aux.dictionary.t()[truth_atoms[:, p].cuda()].float()
        for kind in ("near", "model"):
            replacement = candidates[kind][:, j]
            new_vectors = aux.dictionary.t()[replacement.cuda()].float()
            optimum = scalar_projection_coefficient(old_vectors, new_vectors, true_c[:, p].cuda())
            adjusted_id = nearest_id(aux, optimum, d).cpu()
            changed = packed.clone().flatten(1)
            changed[:, p] = replacement * 2048 + adjusted_id
            adjusted, _ = predictions(model, aux, changed.reshape_as(packed), uniforms, [], args.batch_size)
            predicted_id = adjusted["prediction"][:, p]
            changed[:, p] = replacement * 2048 + predicted_id
            predicted, _ = predictions(model, aux, changed.reshape_as(packed), uniforms, [], args.batch_size)
            # The intervention cannot change earlier predictions; current context
            # must not depend on the injected current coefficient either.
            assert torch.equal(base["prediction"][:, :p], adjusted["prediction"][:, :p])
            assert torch.equal(base["prediction"][:, :p], predicted["prediction"][:, :p])
            assert torch.equal(adjusted["prediction"][:, p], predicted["prediction"][:, p])
            adjusted_c = aux.coeff_bins[adjusted_id.cuda()] * aux.coeff_scales[d]
            predicted_c = aux.coeff_bins[predicted_id.cuda()] * aux.coeff_scales[d]
            old_contribution = old_vectors * true_c[:, p].cuda().unsqueeze(-1)
            case = {
                "position": p, "site": p // 4, "depth": d, "kind": kind,
                "replacement_atom": replacement, "adjusted_id": adjusted_id,
                "predicted_id": predicted_id, "cosine": candidates[f"{kind}_cosine"][:, j],
                "adjusted": adjusted, "predicted": predicted,
                "replacement_prediction_mae_to_projection": (predicted_c - adjusted_c).abs().cpu(),
                "adjusted_contribution_mse": (new_vectors * adjusted_c.unsqueeze(-1) - old_contribution).square().mean(-1).cpu(),
                "predicted_contribution_mse": (new_vectors * predicted_c.unsqueeze(-1) - old_contribution).square().mean(-1).cpu(),
            }
            cases.append(case)
            log(args, {"phase": "single_intervention", "site": p // 4, "depth": d, "kind": kind,
                       "mean_cosine": float(case["cosine"].mean()),
                       "coefficient_mae_to_adjusted_target": float(case["replacement_prediction_mae_to_projection"].mean())})
    result = {"provenance": provenance, "seed": args.seed, "packed": packed, "uniforms": uniforms,
              "baseline": base, "cases": cases, "physical_targets": true_c,
              "scales": aux.coeff_scales.cpu(), "bins": aux.coeff_bins.cpu()}
    torch.save(result, args.output / "local.pt")
    log(args, {"phase": "local_complete", "images": len(packed), "cases": len(cases),
               "baseline_sign_accuracy": float(((base["prediction"] >= 1024) == (truth_ids >= 1024)).float().mean())})


@torch.no_grad()
def coefficient_rollout(model, aux, truth, start, replacement=None, adjusted=None,
                        clamp_initial=False, uniforms=None, oracle_component=None):
    """True prefix, then generated coefficients; only one atom may be changed."""
    packed = truth.clone()
    atoms, coefficient_ids = model.unpack(packed)
    atoms, coefficient_ids = atoms.clone(), coefficient_ids.clone()
    if replacement is not None:
        atoms.flatten(1)[:, start] = replacement
    model.init_cache()
    try:
        for position in range(256):
            site, depth = divmod(position, 4)
            row, col = divmod(site, 8)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = model.cached_head_output(packed, aux, None, (row, col, depth), amp=False)
                if position < start:
                    continue
                if position == start and adjusted is not None:
                    coefficient = adjusted
                elif position == start and clamp_initial:
                    coefficient = coefficient_ids[:, row, col, depth]
                else:
                    atom = atoms[:, row, col, depth]
                    refined = model.refine_coefficient_hidden(hidden, aux.dictionary.t()[atom])
                    logits = model.classify_coefficients(refined, depth).float()
                    coefficient = select_coefficients(logits, None if uniforms is None else uniforms[:, position])
            if oracle_component:
                coefficient = restore_component(coefficient, truth[:, row, col, depth] % 2048, oracle_component)
            coefficient_ids[:, row, col, depth] = coefficient
            packed[:, row, col, depth] = atoms[:, row, col, depth] * 2048 + coefficient
    finally:
        model.init_cache()
    return atoms, coefficient_ids


@torch.no_grad()
def reconstruction_metrics(aux, atoms, ids, true_atoms, true_ids):
    records, previews = [], []
    for begin in range(0, len(atoms), 32):
        a, c, ta, tc = [x[begin:begin + 32].cuda() for x in (atoms, ids, true_atoms, true_ids)]
        image = ((aux.decode_compound(a, c).float() + 1) / 2).clamp(0, 1)
        target = ((aux.decode_compound(ta, tc).float() + 1) / 2).clamp(0, 1)
        pc, truth_c = physical(aux, c.flatten(1)), physical(aux, tc.flatten(1))
        latent = (aux.dictionary.t()[a] * pc.reshape_as(c).unsqueeze(-1)).sum(-2)
        target_latent = (aux.dictionary.t()[ta] * truth_c.reshape_as(tc).unsqueeze(-1)).sum(-2)
        record = {
            "mae": (pc - truth_c).abs(),
            "correct_sign": ((c.flatten(1) >= 1024) == (tc.flatten(1) >= 1024)).float(),
            "latent_mse_by_site": (latent - target_latent).square().mean(-1).flatten(1),
            "psnr_to_true_code_reconstruction": -10 * (image - target).square().mean((1, 2, 3)).clamp_min(1e-12).log10(),
        }
        records.append({k: v.cpu() for k, v in record.items()})
        if begin == 0:
            previews = image[:4].cpu()
    return {k: torch.cat([r[k] for r in records]) for k in records[0]}, previews


@torch.no_grad()
def rollout(args, model, aux, packed, provenance):
    data = torch.load(args.output / "local.pt", weights_only=True)
    assert data["provenance"] == provenance
    assert torch.equal(data["packed"], packed)
    assert data["seed"] == args.seed
    all_results = []
    true_atoms, true_ids = model.unpack(packed)
    for site in args.rollout_sites:
        start = 4 * site
        selected = {case["kind"]: case for case in data["cases"] if case["position"] == start}
        assert set(selected) == {"near", "model"}
        names = (["sign-oracle", "magnitude-oracle", "full-oracle"] if args.phase == "oracle" else
                 ["teacher-forced", "control", "control-clamped", "near-adjusted", "near-predicted", "model-adjusted", "model-predicted"])
        site_results, previews = {}, {}
        for name in names:
            if name == "teacher-forced":
                atoms = true_atoms.clone()
                ids = true_ids.clone().flatten(1)
                choice = "prediction" if args.sampling == "greedy" else "sample"
                ids[:, start:] = data["baseline"][choice][:, start:]
                ids = ids.reshape_as(true_ids)
            else:
                collected = []
                for begin in range(0, len(packed), args.batch_size):
                    end = min(begin + args.batch_size, len(packed))
                    uniforms = data["uniforms"][begin:end].cuda() if args.sampling == "sample" else None
                    case = selected[name.split("-")[0]] if name.split("-")[0] in selected else None
                    replacement = case["replacement_atom"][begin:end].cuda() if case else None
                    adjusted = case["adjusted_id"][begin:end].cuda() if case and name.endswith("adjusted") else None
                    oracle = name.split("-")[0] if name.endswith("oracle") else None
                    a, c = coefficient_rollout(model, aux, packed[begin:end].cuda(), start,
                                               replacement, adjusted, name == "control-clamped", uniforms, oracle)
                    collected.append((a.cpu(), c.cpu()))
                atoms = torch.cat([pair[0] for pair in collected])
                ids = torch.cat([pair[1] for pair in collected])
            assert torch.equal(ids.flatten(1)[:, :start], true_ids.flatten(1)[:, :start])
            assert torch.equal(atoms.flatten(1)[:, start + 1:], true_atoms.flatten(1)[:, start + 1:])
            if name == "full-oracle":
                assert torch.equal(atoms, true_atoms)
                assert torch.equal(ids, true_ids)
            metrics, preview = reconstruction_metrics(aux, atoms, ids, true_atoms, true_ids)
            site_results[name] = {"atoms": atoms, "coefficient_ids": ids, "metrics": metrics}
            previews[name] = preview
            log(args, {"phase": "rollout_condition", "sampling": args.sampling, "site": site, "condition": name,
                       "downstream_mae": float(metrics["mae"][:, start + 1:].mean()),
                       "downstream_sign_accuracy": float(metrics["correct_sign"][:, start + 1:].mean()),
                       "psnr": float(metrics["psnr_to_true_code_reconstruction"].mean())})
        reference = ((aux.decode_compound(true_atoms[:4].cuda(), true_ids[:4].cuda()).float() + 1) / 2).clamp(0, 1).cpu()
        columns = [reference] + [previews[name] for name in names]
        save_image(torch.stack(columns, dim=1).flatten(0, 1), args.output / f"{args.phase}-{args.sampling}-site{site}.png", nrow=len(columns))
        all_results.append({"site": site, "start": start, "conditions": site_results})
    torch.save({"provenance": provenance, "seed": args.seed, "sampling": args.sampling,
                "sites": all_results}, args.output / f"{args.phase}-{args.sampling}.pt")
    log(args, {"phase": f"{args.phase}_complete", "sampling": args.sampling})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["local", "rollout", "oracle"], required=True)
    parser.add_argument("--sampling", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--assets", type=Path, default=ROOT / "outputs/lsun-church-bar-20260910")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sites", type=int, nargs="+", default=[8, 24, 40, 56])
    parser.add_argument("--depths", type=int, nargs="+", default=[0, 2])
    parser.add_argument("--rollout-sites", type=int, nargs="+", default=[8, 40])
    parser.add_argument("--seed", type=int, default=3701)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / ("local.pt" if args.phase == "local" else f"{args.phase}-{args.sampling}.pt")
    if target.exists():
        raise FileExistsError(target)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.output / f"config-{args.phase}-{args.sampling}.json").write_text(json.dumps(config, indent=2))
    start = time.monotonic()
    model, aux, packed, provenance = load(args)
    (local if args.phase == "local" else rollout)(args, model, aux, packed, provenance)
    log(args, {"phase": "complete", "seconds": time.monotonic() - start})


if __name__ == "__main__":
    main()
