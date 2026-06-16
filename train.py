#!/usr/bin/env python3
"""Unified training launcher for LASER experiments.

This is a small argparse facade over the maintained Hydra entrypoints.  It keeps
paper-facing commands short while preserving the existing training code paths.
Example:

    python train.py --stage 1 --adversarial true --num_gpus 8 \
      --dataset imagenet --modality image --conditioning class
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_DATASETS = {
    "celeba",
    "celebahq",
    "cifar10",
    "coco",
    "ffhq",
    "imagenet",
    "imagenette2",
    "stl10",
}
AUDIO_DATASETS = {"vctk", "maestro"}


@dataclass(frozen=True)
class ResourcePlan:
    total_gpus: int
    num_nodes: int
    devices_per_node: int
    strategy: str


def _str_bool(raw) -> bool:
    if isinstance(raw, bool):
        return raw
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {raw!r}")


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _stage(raw: str) -> str:
    value = str(raw).strip().lower()
    if value in {"1", "stage1", "stage-1"}:
        return "1"
    if value in {"2", "stage2", "stage-2"}:
        return "2"
    raise argparse.ArgumentTypeError("stage must be 1 or 2")


def _default_output_root() -> str:
    user = os.environ.get("USER", "unknown")
    scratch = Path("/scratch") / user
    if scratch.exists():
        return str(scratch / "runs" / "laser_cli")
    return "runs/laser_cli"


def _slurm_num_nodes() -> int | None:
    for key in ("SLURM_JOB_NUM_NODES", "SLURM_NNODES"):
        raw = os.environ.get(key)
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                pass
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified LASER training launcher. Unknown trailing args are passed through as Hydra overrides.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stage", required=True, type=_stage, help="Training stage: 1 for autoencoder, 2 for prior.")
    parser.add_argument("--dataset", required=True, help="Dataset config name, e.g. imagenet, celebahq, ffhq, vctk.")
    parser.add_argument("--modality", required=True, choices=("image", "audio"))
    parser.add_argument("--conditioning", default="none", choices=("none", "class", "text"))
    parser.add_argument("--adversarial", type=_str_bool, default=False, help="Enable stage-1 adversarial loss.")
    parser.add_argument("--num_gpus", type=_positive_int, default=1, help="Total GPUs requested for the run.")
    parser.add_argument(
        "--num_nodes",
        default="auto",
        help="Number of nodes. Use auto to infer from SLURM_JOB_NUM_NODES/SLURM_NNODES when present.",
    )
    parser.add_argument(
        "--devices_per_node",
        type=_positive_int,
        default=None,
        help="Override Lightning train.devices/train_ar.devices directly.",
    )
    parser.add_argument("--downsample_layers", type=int, default=5, choices=(5, 6))
    parser.add_argument("--sparsity_level", type=_positive_int, default=3)
    parser.add_argument("--num_embeddings", type=_positive_int, default=None)
    parser.add_argument("--embedding_dim", type=_positive_int, default=None)
    parser.add_argument("--image_size", type=_positive_int, default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--batch_size", type=_positive_int, default=None, help="Per-process batch size.")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--epochs", type=_positive_int, default=None)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--learning_rate", default=None)
    parser.add_argument("--dict_learning_rate", default=None)
    parser.add_argument("--output_root", default=_default_output_root())
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--project", default="laser")
    parser.add_argument("--token_cache_path", default=None, help="Required for --stage 2.")
    parser.add_argument("--num_classes", type=_positive_int, default=None)
    parser.add_argument("--dry_run", action="store_true", help="Print the translated command without executing it.")
    return parser


def _normalize_unknown(unknown: Iterable[str]) -> list[str]:
    values = list(unknown)
    if values and values[0] == "--":
        values = values[1:]
    return values


def _resolve_resources(total_gpus: int, num_nodes_arg: str, devices_per_node: int | None) -> ResourcePlan:
    if str(num_nodes_arg).strip().lower() == "auto":
        num_nodes = _slurm_num_nodes() or 1
    else:
        num_nodes = _positive_int(str(num_nodes_arg))
    if devices_per_node is not None:
        devices = devices_per_node
    elif num_nodes > 1:
        if total_gpus % num_nodes != 0:
            raise SystemExit(
                f"--num_gpus ({total_gpus}) must be divisible by inferred --num_nodes ({num_nodes}); "
                "or pass --devices_per_node explicitly."
            )
        devices = max(1, total_gpus // num_nodes)
    else:
        devices = total_gpus
    strategy = "ddp" if int(num_nodes) * int(devices) > 1 else "auto"
    return ResourcePlan(total_gpus=total_gpus, num_nodes=num_nodes, devices_per_node=devices, strategy=strategy)


def _dataset_config(dataset: str, modality: str) -> str:
    name = dataset.strip().lower()
    if modality == "image":
        if name not in IMAGE_DATASETS:
            raise SystemExit(f"Unknown or non-image dataset for --modality image: {dataset!r}")
        return name
    if name not in AUDIO_DATASETS:
        raise SystemExit(f"Unknown or non-audio dataset for --modality audio: {dataset!r}")
    if name == "vctk":
        return "vctk_waveform"
    if name == "maestro":
        return "maestro_waveform"
    return name


def _model_config(modality: str, downsample_layers: int) -> str:
    if modality == "image":
        return f"laser_image_nonpatch_d{downsample_layers}"
    return f"laser_audio_waveform_nonpatch_d{downsample_layers}"


def _default_batch_size(modality: str, dataset: str) -> int:
    if modality == "audio":
        return 2
    if dataset.strip().lower() == "imagenet":
        return 2
    return 2


def _default_workers(modality: str, dataset: str) -> int:
    if modality == "audio":
        return 4
    if dataset.strip().lower() == "imagenet":
        return 8
    return 4


def _default_lr(modality: str) -> str:
    return "1.5e-4" if modality == "audio" else "1.0e-4"


def _default_dict_lr(modality: str) -> str:
    return "1.5e-4" if modality == "audio" else "2.5e-4"


def _default_epochs(stage: str) -> int:
    return 75 if stage == "1" else 300


def _run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    adv = "adv" if args.adversarial and args.stage == "1" else "recon"
    if args.stage == "2":
        adv = "prior"
    return (
        f"stage{args.stage}-{args.dataset}-{args.modality}-{args.conditioning}-"
        f"d{args.downsample_layers}-k{args.sparsity_level}-{adv}"
    )


def _output_dir(args: argparse.Namespace, run_name: str) -> str:
    if args.output_dir:
        return args.output_dir
    return str(Path(args.output_root).expanduser() / run_name)


def _tags(args: argparse.Namespace) -> str:
    tags = [
        f"stage{args.stage}",
        args.modality,
        args.dataset.strip().lower(),
        args.conditioning,
        "nonpatch",
        f"d{args.downsample_layers}",
        f"k{args.sparsity_level}",
    ]
    if args.stage == "1":
        tags.append("adversarial" if args.adversarial else "reconstruction")
    return "[" + ",".join(tags) + "]"


def _validate(args: argparse.Namespace) -> None:
    dataset = args.dataset.strip().lower()
    if args.conditioning == "class" and args.modality != "image":
        raise SystemExit("--conditioning class is only valid with --modality image.")
    if args.conditioning == "text" and args.modality != "audio":
        raise SystemExit("--conditioning text is only valid with --modality audio.")
    if args.conditioning == "class" and dataset not in {"imagenet", "imagenette2", "stl10", "cifar10"}:
        raise SystemExit(f"--conditioning class is not configured for dataset {dataset!r}.")
    if args.stage == "2" and not args.token_cache_path:
        raise SystemExit("--stage 2 requires --token_cache_path.")


def _common_overrides(args: argparse.Namespace, resources: ResourcePlan, run_name: str, output_dir: str) -> list[str]:
    data_cfg = _dataset_config(args.dataset, args.modality)
    overrides = [
        f"data={data_cfg}",
        f"output_dir={output_dir}",
        f"seed=42",
        f"wandb.project={args.project}",
        f"wandb.name={run_name}",
        f"wandb.group={run_name}",
        f"wandb.tags={_tags(args)}",
        "wandb.append_timestamp=false",
    ]
    if args.data_dir:
        overrides.append(f"data.data_dir={Path(args.data_dir).expanduser()}")
    if args.image_size:
        overrides.append(f"data.image_size={args.image_size}")
    overrides.append(f"data.batch_size={args.batch_size or _default_batch_size(args.modality, args.dataset)}")
    overrides.append(f"data.num_workers={_default_workers(args.modality, args.dataset) if args.num_workers is None else args.num_workers}")
    return overrides


def _stage1_adversarial_overrides(args: argparse.Namespace) -> list[str]:
    if not args.adversarial:
        return [
            "model.adversarial_weight=0.0",
            "model.adversarial_start_step=1000000000",
            "model.adversarial_warmup_steps=0",
            "model.disc_start_step=1000000000",
        ]
    if args.modality == "audio":
        return [
            "model.adversarial_weight=0.03",
            "model.adversarial_start_step=0",
            "model.adversarial_warmup_steps=0",
            "model.disc_start_step=0",
            "model.audio_adversarial_type=hifigan",
            "model.audio_disc_periods=[2,3,5,7,11]",
            "model.audio_disc_num_scales=3",
            "model.audio_disc_max_channels=512",
            "model.disc_channels=32",
            "model.disc_num_layers=3",
            "model.disc_learning_rate=5.0e-5",
            "model.disc_loss=hinge",
            "model.use_adaptive_disc_weight=true",
        ]
    return [
        "model.adversarial_weight=0.05",
        "model.adversarial_start_step=0",
        "model.adversarial_warmup_steps=0",
        "model.disc_start_step=0",
        "model.disc_channels=64",
        "model.disc_num_layers=3",
        "model.disc_norm=group",
        "model.disc_learning_rate=5.0e-5",
        "model.disc_loss=hinge",
        "model.use_adaptive_disc_weight=true",
    ]


def _stage1_command(args: argparse.Namespace, resources: ResourcePlan, extra: list[str]) -> list[str]:
    run_name = _run_name(args)
    output_dir = _output_dir(args, run_name)
    script = Path(__file__).resolve().with_name("train_stage1_autoencoder.py")
    overrides = [
        f"model={_model_config(args.modality, args.downsample_layers)}",
        *_common_overrides(args, resources, run_name, output_dir),
        f"model.sparsity_level={args.sparsity_level}",
        "model.patch_based=false",
        "model.patch_size=1",
        "model.patch_stride=1",
        f"train.max_epochs={args.epochs or _default_epochs('1')}",
        f"train.max_steps={args.max_steps}",
        f"train.accelerator=gpu",
        f"train.num_nodes={resources.num_nodes}",
        f"train.devices={resources.devices_per_node}",
        f"train.strategy={resources.strategy}",
        f"train.precision={args.precision}",
        f"train.learning_rate={args.learning_rate or _default_lr(args.modality)}",
        f"model.dict_learning_rate={args.dict_learning_rate or _default_dict_lr(args.modality)}",
        "train.gradient_clip_val=1.0",
        "train.deterministic=false",
        "train.run_test_after_fit=false",
        "train.min_lr_ratio=0.03",
        "checkpoint.save_top_k=1",
    ]
    if args.modality == "image":
        overrides.extend(["train.warmup_steps=500", "train.val_check_interval=0.25"])
        if args.dataset.strip().lower() == "imagenet":
            overrides.extend(["train.limit_val_batches=512", "train.limit_test_batches=512"])
        else:
            overrides.extend(["train.limit_val_batches=256", "train.limit_test_batches=256"])
    else:
        overrides.extend(["train.warmup_steps=750", "data.audio_representation=waveform"])
    if args.num_embeddings is not None:
        overrides.append(f"model.num_embeddings={args.num_embeddings}")
    if args.embedding_dim is not None:
        overrides.append(f"model.embedding_dim={args.embedding_dim}")
    overrides.extend(_stage1_adversarial_overrides(args))
    overrides.extend(extra)
    return [sys.executable, str(script), *overrides]


def _stage2_command(args: argparse.Namespace, resources: ResourcePlan, extra: list[str]) -> list[str]:
    run_name = _run_name(args)
    output_dir = _output_dir(args, run_name)
    script = Path(__file__).resolve().with_name("train_stage2_prior.py")
    overrides = [
        f"token_cache_path={Path(args.token_cache_path).expanduser()}",
        f"output_dir={output_dir}",
        f"data.dataset={args.dataset.strip().lower()}",
        f"wandb.project={args.project}",
        f"wandb.name={run_name}",
        f"wandb.group={run_name}",
        f"wandb.tags={_tags(args)}",
        "wandb.append_timestamp=false",
        "ar.type=sparse_spatial_depth",
        f"ar.max_steps={args.max_steps}",
        "ar.d_model=768",
        "ar.n_heads=12",
        "ar.n_layers=18",
        "ar.d_ff=3072",
        "ar.n_global_spatial_tokens=16",
        f"train_ar.max_epochs={args.epochs or _default_epochs('2')}",
        f"train_ar.batch_size={args.batch_size or _default_batch_size(args.modality, args.dataset)}",
        f"train_ar.accelerator=gpu",
        f"train_ar.num_nodes={resources.num_nodes}",
        f"train_ar.devices={resources.devices_per_node}",
        f"train_ar.strategy={resources.strategy}",
        f"train_ar.precision={args.precision}",
        "train_ar.sample_log_to_wandb=false",
        "train_ar.run_test_after_fit=false",
        "train_ar.save_final_samples_after_fit=false",
    ]
    if args.conditioning == "class":
        num_classes = args.num_classes or (1000 if args.dataset.strip().lower() == "imagenet" else 0)
        overrides.extend(
            [
                "ar.class_conditional=true",
                f"ar.num_classes={num_classes}",
                "train_ar.sample_class_labels=[0,1,2,3,4,5,6,7]",
            ]
        )
    elif args.conditioning == "text":
        overrides.extend(
            [
                "ar.text_conditional=true",
                "ar.text_prefix_length=16",
                "train_ar.sample_text_prompts=[\"The quick brown fox jumps over the lazy dog.\",\"A calm voice reads this sentence clearly.\"]",
            ]
        )
    overrides.extend(extra)
    return [sys.executable, str(script), *overrides]


def build_command(argv: list[str] | None = None) -> list[str]:
    parser = _build_parser()
    args, unknown = parser.parse_known_args(argv)
    extra = _normalize_unknown(unknown)
    _validate(args)
    resources = _resolve_resources(args.num_gpus, args.num_nodes, args.devices_per_node)
    if args.stage == "1":
        return _stage1_command(args, resources, extra)
    return _stage2_command(args, resources, extra)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args, unknown = parser.parse_known_args(argv)
    extra = _normalize_unknown(unknown)
    _validate(args)
    resources = _resolve_resources(args.num_gpus, args.num_nodes, args.devices_per_node)
    cmd = _stage1_command(args, resources, extra) if args.stage == "1" else _stage2_command(args, resources, extra)
    print("Launching:", shlex.join(cmd), flush=True)
    if args.dry_run:
        return 0
    os.environ.setdefault("LASER_DISABLE_WANDB_MEDIA", "1")
    os.execv(cmd[0], cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
