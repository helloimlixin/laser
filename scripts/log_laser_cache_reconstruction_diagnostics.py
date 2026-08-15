#!/usr/bin/env python3
"""Log aligned LASER token-cache reconstruction diagnostics to one W&B run."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_upstream_laser_rfid import FlatImages
from scripts.train_official_rqtransformer_laser_stage2 import LaserAux, val_image_transform


def decode_cache(
    aux: LaserAux,
    cache: dict,
    count: int,
    mode: str,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Decode the first ``count`` row-aligned cache entries on one GPU."""
    reconstructions = []
    sparsity_level = int(cache["meta"]["shape"][-1])
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            atoms = cache["atoms"][start:stop].to(device, dtype=torch.long)
            coeffs = cache["coeffs"][start:stop].to(device, dtype=torch.float32)
            if mode == "quantized":
                coeff_ids, _ = aux.compound_coeff_ids(
                    coeffs, stochastic=False, hard=True
                )
                decoded = aux.decode_compound(atoms, coeff_ids)
            else:
                vectors = aux.dictionary.t()[atoms]
                physical_coeffs = coeffs * aux.coeff_scales.view(
                    1, 1, 1, sparsity_level
                )
                z_q = (vectors * physical_coeffs[..., None]).sum(dim=-2)
                z_q = aux.post_quant_conv(
                    z_q.permute(0, 3, 1, 2).contiguous()
                )
                decoded = aux.decoder(z_q).clamp(-1.0, 1.0)
            reconstructions.append(
                ((decoded.float() + 1.0) * 0.5).clamp(0, 1).cpu()
            )
    return torch.cat(reconstructions, dim=0)


def load_rfid(path: Path) -> tuple[float, dict]:
    payload = json.loads(path.read_text())
    return float(payload["rfid"]), payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset", choices=("ffhq", "celebahq"), default="ffhq")
    parser.add_argument(
        "--cache-spec",
        action="append",
        nargs=4,
        metavar=("LABEL", "CACHE", "CONTINUOUS_RFID_JSON", "QUANTIZED_RFID_JSON"),
        required=True,
        help="May be repeated to compare multiple coefficient tokenizers.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--decode-batch-size", type=int, default=8)
    parser.add_argument("--wandb-entity", default="helloimlixin-rutgers")
    parser.add_argument("--wandb-project", default="laser")
    parser.add_argument("--wandb-id", required=True)
    parser.add_argument("--wandb-name", required=True)
    args = parser.parse_args()

    side = int(args.grid_size ** 0.5)
    if side * side != args.grid_size:
        parser.error("--grid-size must be a perfect square")
    if args.grid_size != 64:
        parser.error("this diagnostic is intentionally fixed to an aligned 8x8 grid (64 images)")
    if args.decode_batch_size <= 0:
        parser.error("--decode-batch-size must be positive")

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    dataset = FlatImages(args.data, transform=val_image_transform())
    if len(dataset) < args.grid_size:
        raise ValueError(f"dataset contains only {len(dataset)} images")
    # FlatImages sorts paths, so every cache comparison uses exactly the same
    # source rows in a deterministic order.
    source = torch.stack(
        [((dataset[index][0].float() + 1.0) * 0.5).clamp(0, 1)
         for index in range(args.grid_size)]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_path = args.output_dir / "source_grid_8x8.png"
    save_image(source, source_path, nrow=8, padding=2)

    import wandb

    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        id=args.wandb_id,
        name=args.wandb_name,
        resume="allow",
        mode="online",
        config={
            "diagnostic_checkpoint": str(args.checkpoint.resolve()),
            "diagnostic_dataset": args.dataset,
            "diagnostic_grid_indices": list(range(args.grid_size)),
            "diagnostic_grid_layout": "8x8",
        },
    )
    log_payload = {
        "diagnostics/source_grid_8x8": wandb.Image(
            str(source_path), caption="FFHQ source rows 0-63 (sorted paths)"
        ),
        "diagnostics/reconstruction_rfid_num_images": 70_000,
    }

    for raw_label, cache_arg, continuous_json_arg, quantized_json_arg in args.cache_spec:
        label = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_label).strip("_")
        if not label:
            raise ValueError(f"invalid empty cache label: {raw_label!r}")
        cache_path = Path(cache_arg)
        continuous_json = Path(continuous_json_arg)
        quantized_json = Path(quantized_json_arg)
        cache = torch.load(cache_path, map_location="cpu", weights_only=True, mmap=True)
        meta = dict(cache["meta"])
        if meta.get("dataset") != args.dataset:
            raise ValueError(
                f"{label}: cache dataset {meta.get('dataset')!r} != {args.dataset!r}"
            )
        if len(cache["atoms"]) < args.grid_size:
            raise ValueError(f"{label}: cache has fewer than {args.grid_size} rows")

        aux = LaserAux(
            args.checkpoint,
            int(meta["num_atoms"]),
            int(meta["coeff_vocab_size"]),
            float(meta["coeff_max"]),
            float(meta.get("coeff_scale", 1.0)),
            attn_resolutions=(16,),
            coeff_scales=meta.get("coeff_scales"),
            clamp_coeffs=True,
            coeff_bin_centers=meta.get("coeff_bin_centers"),
            sparsity_level=int(meta["shape"][-1]),
        ).to(device).eval()

        continuous = decode_cache(
            aux, cache, args.grid_size, "continuous", args.decode_batch_size, device
        )
        quantized = decode_cache(
            aux, cache, args.grid_size, "quantized", args.decode_batch_size, device
        )
        continuous_path = args.output_dir / f"{label}_continuous_grid_8x8.png"
        quantized_path = args.output_dir / f"{label}_quantized_grid_8x8.png"
        save_image(continuous, continuous_path, nrow=8, padding=2)
        save_image(quantized, quantized_path, nrow=8, padding=2)

        continuous_rfid, continuous_payload = load_rfid(continuous_json)
        quantized_rfid, quantized_payload = load_rfid(quantized_json)
        for mode, metric_payload in (
            ("continuous", continuous_payload),
            ("quantized", quantized_payload),
        ):
            if int(metric_payload["num_images"]) != 70_000:
                raise ValueError(
                    f"{label}/{mode}: expected a 70,000-image rFID result"
                )
            if Path(metric_payload["token_cache"]).resolve() != cache_path.resolve():
                raise ValueError(
                    f"{label}/{mode}: rFID JSON was computed from a different cache"
                )

        prefix = f"diagnostics/{label}"
        log_payload.update({
            f"{prefix}/continuous_cache_reconstruction_grid_8x8": wandb.Image(
                str(continuous_path),
                caption=f"{label}: cached continuous coefficients",
            ),
            f"{prefix}/quantized_cache_reconstruction_grid_8x8": wandb.Image(
                str(quantized_path),
                caption=f"{label}: nearest-bin coefficient tokens",
            ),
            f"{prefix}/continuous_cache_reconstruction_rfid": continuous_rfid,
            f"{prefix}/quantized_cache_reconstruction_rfid": quantized_rfid,
            f"{prefix}/coefficient_vocab_size": int(meta["coeff_vocab_size"]),
            f"{prefix}/coefficient_scales_depth_0": float(meta["coeff_scales"][0]),
            f"{prefix}/coefficient_scales_depth_1": float(meta["coeff_scales"][1]),
        })
        del aux, continuous, quantized, cache
        torch.cuda.empty_cache()

    run.log(log_payload)
    run.finish()
    print(f"Logged aligned 8x8 cache diagnostics to {run.url}", flush=True)


if __name__ == "__main__":
    main()
