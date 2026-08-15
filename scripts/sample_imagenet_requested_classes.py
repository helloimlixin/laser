#!/usr/bin/env python3
"""Sample fixed ImageNet classes from an official LASER RQ-Transformer run."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_official_rqtransformer_laser_stage2 import (  # noqa: E402
    LaserAux,
    build_model,
    save_class_labeled_grid,
)
from src.data.imagenet_labels import class_names_for_dataset  # noqa: E402


DEFAULT_CLASSES = (
    "ostrich",
    "bald eagle",
    "lorikeet",
    "tibetan terrier",
    "snow leopard",
    "teapot",
    "wombat",
    "red fox",
    "samoyed",
    "hotpot",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES))
    parser.add_argument("--samples-per-class", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb-run", default=None)
    parser.add_argument("--wandb-key", default="samples/requested_classes_10x8")
    return parser.parse_args()


def resolve_classes(requested: list[str], class_names: list[str]) -> tuple[list[int], list[str]]:
    normalized = {name.casefold(): index for index, name in enumerate(class_names)}
    ids = []
    display_names = []
    for requested_name in requested:
        key = str(requested_name).strip().replace("_", " ").casefold()
        if key == "hotpot":
            key = "hot pot"
        if key not in normalized:
            raise ValueError(f"unknown canonical ImageNet class: {requested_name!r}")
        class_id = normalized[key]
        ids.append(class_id)
        display_names.append(class_names[class_id])
    if len(set(ids)) != len(ids):
        raise ValueError("requested classes must be unique")
    return ids, display_names


@torch.inference_mode()
def sample_images(
    model,
    aux,
    class_ids: list[int],
    *,
    samples_per_class: int,
    batch_size: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> torch.Tensor:
    labels = torch.tensor(class_ids, device=device, dtype=torch.long).repeat_interleave(
        samples_per_class
    )
    batches = []
    for start in range(0, int(labels.numel()), batch_size):
        batch_labels = labels[start : start + batch_size]
        partial = torch.zeros(
            int(batch_labels.numel()), 8, 8, 4, device=device, dtype=torch.long
        )
        tokens = model.sample(
            partial,
            model_aux=aux,
            cond=batch_labels,
            temperature=temperature,
            top_k=aux.num_atoms,
            top_p=top_p,
            amp=True,
            cached=True,
            is_tqdm=False,
        )
        decoded = aux.decode_tokens(tokens)
        batches.append(((decoded.float().cpu() + 1.0) * 0.5).clamp(0, 1))
        print(
            f"sampled {min(start + batch_size, int(labels.numel()))}/{int(labels.numel())}",
            flush=True,
        )
    return torch.cat(batches)


def upload_to_wandb(
    run_path: str,
    image_path: Path,
    manifest_path: Path,
    *,
    key: str,
    source_step: int,
) -> str:
    import wandb

    parts = run_path.strip("/").split("/")
    if len(parts) != 3:
        raise ValueError("--wandb-run must be entity/project/run_id")
    entity, project, run_id = parts
    run = wandb.init(entity=entity, project=project, id=run_id, resume="must")
    run.log(
        {
            key: wandb.Image(str(image_path), caption="10 requested ImageNet classes, 8 samples each"),
            "sampling/requested_classes/source_global_step": source_step,
        }
    )
    run.save(str(manifest_path), base_path=str(manifest_path.parent), policy="now")
    run.finish()
    return f"https://wandb.ai/{entity}/{project}/runs/{run_id}"


def main() -> int:
    args = parse_args()
    if args.samples_per_class <= 0 or args.batch_size <= 0:
        raise ValueError("samples per class and batch size must be positive")
    if not args.stage1.is_file():
        raise FileNotFoundError(args.stage1)
    if not args.stage2.is_file():
        raise FileNotFoundError(args.stage2)

    class_names = class_names_for_dataset("imagenet")
    if len(class_names) != 1000:
        raise RuntimeError("canonical ImageNet-1k class names are unavailable")
    class_ids, display_names = resolve_classes(args.classes, class_names)
    print("classes: " + ", ".join(f"{idx}:{name}" for idx, name in zip(class_ids, display_names)))

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    payload = torch.load(args.stage2, map_location="cpu", weights_only=False, mmap=True)
    config = dict(payload.get("config", {}))
    if bool(config.get("compound_tokens", False)):
        raise ValueError("this sampler expects the run's non-compound checkpoint")
    num_atoms = int(config.get("num_atoms", 16_384))
    coeff_vocab_size = int(config.get("coeff_vocab_size", 2_048))
    coeff_max = float(config.get("coeff_max", 20.0))
    coeff_scale = float(config.get("coeff_scale", 6.4))
    sparsity_level = int(config.get("sparsity_level", 2))
    source_step = int(payload.get("global_step", -1))
    source_epoch = int(payload.get("epoch", -1))

    aux = LaserAux(
        args.stage1,
        num_atoms,
        coeff_vocab_size,
        coeff_max,
        coeff_scale,
        sparsity_level=sparsity_level,
    ).to(device).eval()
    model = build_model(
        num_atoms + coeff_vocab_size,
        num_atoms,
        coeff_vocab_size=coeff_vocab_size,
        sparsity_level=sparsity_level,
        model_preset=str(config.get("model_preset", "imagenet-1400m")),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    del payload
    gc.collect()
    model = model.to(device).eval()

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    images = sample_images(
        model,
        aux,
        class_ids,
        samples_per_class=args.samples_per_class,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    chosen = torch.tensor(class_ids, dtype=torch.long)
    grid_class_names = list(class_names)
    for class_id, requested_name in zip(class_ids, args.classes):
        grid_class_names[class_id] = str(requested_name).strip().replace("_", " ")
    save_class_labeled_grid(
        images,
        chosen,
        grid_class_names,
        args.output,
        samples_per_class=args.samples_per_class,
    )
    manifest_path = args.output.with_suffix(".json")
    manifest_path.write_text(
        json.dumps(
            {
                "stage1": str(args.stage1.resolve()),
                "stage2": str(args.stage2.resolve()),
                "source_epoch": source_epoch,
                "source_global_step": source_step,
                "seed": args.seed,
                "temperature": args.temperature,
                "top_k": num_atoms,
                "top_p": args.top_p,
                "samples_per_class": args.samples_per_class,
                "classes": [
                    {"id": idx, "name": requested, "canonical_name": canonical}
                    for idx, requested, canonical in zip(
                        class_ids, args.classes, display_names
                    )
                ],
                "image": str(args.output.resolve()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved {args.output}", flush=True)
    print(f"saved {manifest_path}", flush=True)

    if args.wandb_run:
        url = upload_to_wandb(
            args.wandb_run,
            args.output,
            manifest_path,
            key=args.wandb_key,
            source_step=source_step,
        )
        print(f"uploaded {url}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
