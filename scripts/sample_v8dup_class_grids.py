#!/usr/bin/env python3
"""Generate subtitle-free 8x8 ImageNet class grids from a LASER RQTransformer."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import timedelta
import gc
import json
import os
from pathlib import Path
import random
import sys

import torch
import torch.distributed as dist
from PIL import Image
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "rq-vae-transformer"))

from scripts.evaluate_laser_full_imagenet import SamplingConfig, sample_images  # noqa: E402
from scripts.train_official_rqtransformer_laser_stage2 import (  # noqa: E402
    LaserAux,
    build_model,
    release_cuda_memory,
)


@dataclass(frozen=True)
class ClassSpec:
    class_id: int
    requested_name: str
    canonical_name: str

    @property
    def slug(self) -> str:
        return self.requested_name.lower().replace(" ", "-")


CLASSES = (
    ClassSpec(22, "bald eagle", "bald eagle"),
    ClassSpec(106, "wombat", "wombat"),
    ClassSpec(277, "red fox", "red fox"),
    ClassSpec(200, "tibetan terrier", "Tibetan terrier"),
    ClassSpec(926, "hotpot", "hot pot"),
    ClassSpec(289, "snow leopard", "snow leopard"),
    ClassSpec(258, "samoyed", "Samoyed"),
    ClassSpec(375, "colobus", "colobus"),
    ClassSpec(7, "cock", "cock"),
    ClassSpec(747, "punching bag", "punching bag"),
    ClassSpec(338, "guinea pig", "guinea pig"),
    ClassSpec(0, "tench", "tench"),
    ClassSpec(128, "black stork", "black stork"),
    ClassSpec(462, "broom", "broom"),
    ClassSpec(849, "teapot", "teapot"),
    ClassSpec(933, "cheeseburger", "cheeseburger"),
    ClassSpec(994, "stinkhorn", "stinkhorn"),
)


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def world() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def is_rank0() -> bool:
    return rank() == 0


def atomic_json(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def class_grid_path(output: Path, spec: ClassSpec) -> Path:
    return output / f"class-{spec.class_id:04d}-{spec.slug}.png"


def configure_seed(seed: int, class_id: int) -> None:
    value = int(seed) + 1009 * int(class_id)
    random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


@torch.no_grad()
def generate_class_grid(
    model,
    aux: LaserAux,
    spec: ClassSpec,
    output: Path,
    *,
    samples_per_class: int,
    batch_size: int,
    grid_columns: int,
    sampling: SamplingConfig,
    seed: int,
) -> Path:
    configure_seed(seed, spec.class_id)
    device = next(model.parameters()).device
    batches: list[torch.Tensor] = []
    generated = 0
    while generated < samples_per_class:
        current = min(batch_size, samples_per_class - generated)
        labels = torch.full((current,), spec.class_id, dtype=torch.long, device=device)
        images = sample_images(model, aux, labels, sampling)
        batches.append(images.detach().float().cpu())
        generated += current
        del images, labels
        release_cuda_memory(model)

    grid_images = torch.cat(batches, dim=0)
    if grid_images.shape[0] != samples_per_class:
        raise RuntimeError(
            f"{spec.requested_name}: generated {grid_images.shape[0]}, expected {samples_per_class}"
        )
    path = class_grid_path(output, spec)
    temporary = path.with_name(path.stem + ".tmp.png")
    save_image(grid_images, temporary, nrow=grid_columns, padding=0)
    os.replace(temporary, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--source-artifact-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-class", type=int, default=64)
    parser.add_argument("--grid-columns", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--top-k", type=int, default=0, help="<=0 means the full vocabulary")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-atoms", type=int, default=16384)
    parser.add_argument("--coeff-vocab-size", type=int, default=2048)
    parser.add_argument("--coeff-max", type=float, default=20.0)
    parser.add_argument("--coeff-scale", type=float, default=6.4)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expected-epoch", type=int, default=100)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-entity", default="helloimlixin-rutgers")
    parser.add_argument("--wandb-project", default="laser")
    parser.add_argument("--wandb-name", default="v8dup-epoch100-selected-class-grids")
    parser.add_argument("--wandb-id", default=None)
    parser.add_argument("--wandb-group", default="v8dup0731113220-class-grids")
    args = parser.parse_args()

    if args.samples_per_class <= 0:
        raise ValueError("--samples-per-class must be positive")
    if args.grid_columns <= 0 or args.samples_per_class % args.grid_columns:
        raise ValueError("samples per class must divide evenly into grid columns")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("--top-p must be in (0, 1]")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    args.output.mkdir(parents=True, exist_ok=True)
    aux = LaserAux(
        args.stage1,
        args.num_atoms,
        args.coeff_vocab_size,
        args.coeff_max,
        args.coeff_scale,
    ).to(device).eval()
    model = build_model(args.num_atoms + args.coeff_vocab_size, args.num_atoms).to(device).eval()
    payload = torch.load(args.stage2, map_location="cpu", weights_only=False, mmap=True)
    checkpoint_epoch = int(payload.get("epoch", -1))
    checkpoint_step = int(payload.get("global_step", -1))
    if args.expected_epoch >= 0 and checkpoint_epoch != args.expected_epoch:
        raise RuntimeError(
            f"Expected epoch {args.expected_epoch} checkpoint, found epoch {checkpoint_epoch}"
        )
    model.load_state_dict(payload["state_dict"], strict=True)
    del payload
    gc.collect()
    for module in (aux, model):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    sampling = SamplingConfig(
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=None if args.top_k <= 0 else int(args.top_k),
    )
    assigned = CLASSES[rank() :: world()]
    for spec in assigned:
        path = class_grid_path(args.output, spec)
        if args.resume and path.is_file():
            print(f"rank={rank()} reuse class={spec.requested_name} path={path}", flush=True)
            continue
        print(
            f"rank={rank()} sample class_id={spec.class_id} class={spec.requested_name}",
            flush=True,
        )
        generate_class_grid(
            model,
            aux,
            spec,
            args.output,
            samples_per_class=args.samples_per_class,
            batch_size=args.batch_size,
            grid_columns=args.grid_columns,
            sampling=sampling,
            seed=args.seed,
        )

    if dist.is_initialized():
        dist.barrier()

    if is_rank0():
        rows = args.samples_per_class // args.grid_columns
        expected_size = (args.grid_columns * 256, rows * 256)
        grids = []
        for spec in CLASSES:
            path = class_grid_path(args.output, spec)
            if not path.is_file():
                raise FileNotFoundError(f"Missing completed grid: {path}")
            with Image.open(path) as image:
                if image.size != expected_size:
                    raise RuntimeError(
                        f"{path} has size {image.size}, expected {expected_size} with no padding"
                    )
            grids.append({**asdict(spec), "path": str(path), "size": list(expected_size)})

        source_artifact = None
        if args.source_artifact_manifest is not None and args.source_artifact_manifest.is_file():
            source_artifact = json.loads(args.source_artifact_manifest.read_text())
        manifest = {
            "checkpoint": str(args.stage2),
            "checkpoint_epoch": checkpoint_epoch,
            "checkpoint_step": checkpoint_step,
            "source_artifact": source_artifact,
            "samples_per_class": args.samples_per_class,
            "grid": {"columns": args.grid_columns, "rows": rows, "padding": 0},
            "sampling": {
                "temperature": sampling.temperature,
                "top_p": sampling.top_p,
                "top_k": "full" if sampling.top_k is None else sampling.top_k,
                "seed": args.seed,
            },
            "classes": grids,
            "total_images": len(CLASSES) * args.samples_per_class,
        }
        manifest_path = args.output / "class_grid_manifest.json"
        atomic_json(manifest, manifest_path)

        if args.wandb:
            import wandb

            wb = wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=args.wandb_name,
                id=args.wandb_id or None,
                group=args.wandb_group or None,
                job_type="class_grid_sampling",
                config={
                    **{key: str(value) for key, value in vars(args).items()},
                    "checkpoint_epoch": checkpoint_epoch,
                    "checkpoint_step": checkpoint_step,
                    "source_artifact": source_artifact,
                    "classes": [asdict(spec) for spec in CLASSES],
                },
            )
            media = {
                f"class_grids/{spec.slug}": wandb.Image(str(class_grid_path(args.output, spec)))
                for spec in CLASSES
            }
            wb.log({
                **media,
                "sampling/grids": len(CLASSES),
                "sampling/images": len(CLASSES) * args.samples_per_class,
                "sampling/checkpoint_epoch": checkpoint_epoch,
                "sampling/checkpoint_step": checkpoint_step,
            })
            wb.save(str(manifest_path), policy="now")
            wb.finish()
        print(f"completed {len(CLASSES)} grids; manifest={manifest_path}", flush=True)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
