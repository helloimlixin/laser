#!/usr/bin/env python3
"""Unified training launcher for LASER experiments.

This is a small argparse facade over the maintained Hydra stage implementations.
It keeps paper-facing commands short while preserving raw Hydra override access.
Example:

    python train.py stage1 --adversarial true --num_gpus 8 \
      --dataset imagenet --modality image --conditioning class

    python train.py stage1 model=laser data=cifar10
    python train.py stage2 token_cache_path=/path/to/token_cache.pt
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


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
STAGE_TOKEN_ALIASES = {
    "1": "1",
    "stage1": "1",
    "stage-1": "1",
    "2": "2",
    "stage2": "2",
    "stage-2": "2",
    "pipeline": "pipeline",
    "full": "pipeline",
    "full-pipeline": "pipeline",
}
FACADE_FLAGS = {
    "--config",
    "--stage",
    "--dataset",
    "--modality",
    "--conditioning",
    "--adversarial",
    "--num_gpus",
    "--num-gpus",
    "--num_nodes",
    "--num-nodes",
    "--devices_per_node",
    "--devices-per-node",
    "--downsample_layers",
    "--downsample-layers",
    "--sparsity_level",
    "--sparsity-level",
    "--num_embeddings",
    "--num-embeddings",
    "--embedding_dim",
    "--embedding-dim",
    "--image_size",
    "--image-size",
    "--data_dir",
    "--data-dir",
    "--batch_size",
    "--batch-size",
    "--num_workers",
    "--num-workers",
    "--epochs",
    "--max_steps",
    "--max-steps",
    "--precision",
    "--learning_rate",
    "--learning-rate",
    "--dict_learning_rate",
    "--dict-learning-rate",
    "--output_root",
    "--output-root",
    "--output_dir",
    "--output-dir",
    "--run_name",
    "--run-name",
    "--project",
    "--token_cache_path",
    "--token-cache-path",
    "--num_classes",
    "--num-classes",
    "--dry_run",
    "--dry-run",
}
FACADE_CONFIG_KEYS = {
    "stage": "--stage",
    "dataset": "--dataset",
    "modality": "--modality",
    "conditioning": "--conditioning",
    "adversarial": "--adversarial",
    "num_gpus": "--num-gpus",
    "num_nodes": "--num-nodes",
    "devices_per_node": "--devices-per-node",
    "downsample_layers": "--downsample-layers",
    "sparsity_level": "--sparsity-level",
    "num_embeddings": "--num-embeddings",
    "embedding_dim": "--embedding-dim",
    "image_size": "--image-size",
    "data_dir": "--data-dir",
    "batch_size": "--batch-size",
    "num_workers": "--num-workers",
    "epochs": "--epochs",
    "max_steps": "--max-steps",
    "precision": "--precision",
    "learning_rate": "--learning-rate",
    "dict_learning_rate": "--dict-learning-rate",
    "output_root": "--output-root",
    "output_dir": "--output-dir",
    "run_name": "--run-name",
    "project": "--project",
    "token_cache_path": "--token-cache-path",
    "num_classes": "--num-classes",
    "dry_run": "--dry-run",
}
CONFIG_META_KEYS = {
    "config",
    "description",
    "direct",
    "hydra_overrides",
    "launcher",
    "mode",
    "name",
    "overrides",
}


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
    if value in STAGE_TOKEN_ALIASES:
        return STAGE_TOKEN_ALIASES[value]
    raise argparse.ArgumentTypeError("stage must be 1, 2, or pipeline")


def _stage_token(raw: str) -> str | None:
    return STAGE_TOKEN_ALIASES.get(str(raw).strip().lower())


def _split_flag_name(arg: str) -> str:
    return arg.split("=", 1)[0]


def _inject_stage_option(argv: list[str]) -> list[str]:
    stage = _stage_token(argv[0]) if argv else None
    if stage:
        return ["--stage", stage, *argv[1:]]
    return argv


def _looks_like_facade_args(argv: Iterable[str]) -> bool:
    return any(_split_flag_name(arg) in FACADE_FLAGS for arg in argv)


def _normalize_config_key(key: str) -> str:
    return str(key).strip().replace("-", "_")


def _hydra_literal(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_hydra_literal(item) for item in value) + "]"
    text = str(value)
    if text == "" or any(ch in text for ch in " \t\n,[]{}:=#"):
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return text


def _flatten_hydra_overrides(prefix: str, value: Any) -> list[str]:
    if isinstance(value, dict):
        flattened: list[str] = []
        for key, child in value.items():
            child_key = str(key).strip()
            child_prefix = f"{prefix}.{child_key}" if prefix else child_key
            flattened.extend(_flatten_hydra_overrides(child_prefix, child))
        return flattened
    return [f"{prefix}={_hydra_literal(value)}"]


def _coerce_hydra_overrides(value: Any, *, source: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        flattened: list[str] = []
        for key, child in value.items():
            flattened.extend(_flatten_hydra_overrides(str(key).strip(), child))
        return flattened
    if isinstance(value, (list, tuple)):
        flattened = []
        for item in value:
            if isinstance(item, str):
                flattened.append(item)
            elif isinstance(item, dict):
                flattened.extend(_coerce_hydra_overrides(item, source=source))
            else:
                raise SystemExit(f"{source} entries must be strings or mappings, got {type(item).__name__}.")
        return flattened
    if isinstance(value, str):
        return [value]
    raise SystemExit(f"{source} must be a string, list, or mapping, got {type(value).__name__}.")


def _load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise SystemExit(f"Config file not found: {config_path}")
    try:
        from omegaconf import OmegaConf

        loaded = OmegaConf.load(config_path)
        data = OmegaConf.to_container(loaded, resolve=True)
    except Exception as exc:
        raise SystemExit(f"Could not read config file {config_path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SystemExit(f"Config file {config_path} must contain a YAML mapping at the top level.")
    return dict(data)


def _extract_config_arg(argv: list[str]) -> tuple[str | None, list[str]]:
    config_path = None
    rest: list[str] = []
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--config":
            if idx + 1 >= len(argv):
                raise SystemExit("--config requires a YAML file path.")
            config_path = argv[idx + 1]
            idx += 2
            continue
        if arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]
            idx += 1
            continue
        rest.append(arg)
        idx += 1
    return config_path, rest


def _merge_launcher_config(config: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    launcher = config.get("launcher")
    if launcher is not None:
        if not isinstance(launcher, dict):
            raise SystemExit("launcher must be a YAML mapping when present.")
        merged.update(launcher)
    for key, value in config.items():
        if key != "launcher":
            merged[key] = value
    return merged


def _config_lookup(config: dict[str, Any], normalized_key: str):
    for key, value in config.items():
        if _normalize_config_key(key) == normalized_key:
            return key, value
    return None, None


def _config_mode(config: dict[str, Any]) -> str:
    _, explicit = _config_lookup(config, "mode")
    if explicit is not None:
        mode = str(explicit).strip().lower()
        if mode in {"direct", "hydra", "raw"}:
            return "direct"
        if mode in {"facade", "launcher", "shortcut"}:
            return "facade"
        raise SystemExit(f"Unsupported config mode {explicit!r}; use direct or facade.")
    _, direct = _config_lookup(config, "direct")
    if bool(direct):
        return "direct"
    dataset_key, _ = _config_lookup(config, "dataset")
    modality_key, _ = _config_lookup(config, "modality")
    return "facade" if dataset_key and modality_key else "direct"


def _config_stage(config: dict[str, Any]) -> str:
    _, raw_stage = _config_lookup(config, "stage")
    if raw_stage is None:
        raise SystemExit("YAML config requires a stage: stage1, stage2, or pipeline.")
    try:
        stage = _stage(str(raw_stage))
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(str(exc)) from exc
    if stage == "pipeline":
        return "pipeline"
    return "stage1" if stage == "1" else "stage2"


def _config_hydra_overrides(config: dict[str, Any], *, consumed: set[str]) -> list[str]:
    overrides: list[str] = []
    for special in ("overrides", "hydra_overrides"):
        _, value = _config_lookup(config, special)
        overrides.extend(_coerce_hydra_overrides(value, source=special))
    for key, value in config.items():
        normalized = _normalize_config_key(key)
        if normalized in consumed or normalized in CONFIG_META_KEYS:
            continue
        overrides.extend(_flatten_hydra_overrides(str(key).strip(), value))
    return overrides


def _config_facade_argv(config: dict[str, Any]) -> list[str]:
    consumed = set(CONFIG_META_KEYS)
    argv: list[str] = []
    for normalized_key, flag in FACADE_CONFIG_KEYS.items():
        actual_key, value = _config_lookup(config, normalized_key)
        if actual_key is None or value is None:
            continue
        consumed.add(_normalize_config_key(actual_key))
        if normalized_key == "dry_run":
            if bool(value):
                argv.append(flag)
            continue
        argv.extend([flag, _hydra_literal(value) if isinstance(value, (list, tuple)) else str(value)])
    argv.extend(_config_hydra_overrides(config, consumed=consumed))
    return argv


def _config_direct_argv(config: dict[str, Any]) -> list[str]:
    consumed = set(CONFIG_META_KEYS)
    consumed.add("stage")
    stage = _config_stage(config)
    argv = [stage]
    _, dry_run = _config_lookup(config, "dry_run")
    if dry_run is not None:
        consumed.add("dry_run")
    argv.extend(_config_hydra_overrides(config, consumed=consumed))
    if bool(dry_run):
        argv.append("--dry-run")
    return argv


def _config_pipeline_argv(path: str | Path, config: dict[str, Any]) -> list[str]:
    _, dry_run = _config_lookup(config, "dry_run")
    argv = ["pipeline", "--pipeline-config", str(path)]
    if bool(dry_run):
        argv.append("--dry-run")
    return argv


def _nested_stage_argv(config: dict[str, Any], key: str, stage: str) -> list[str]:
    section = config.get(key)
    if not isinstance(section, dict):
        raise SystemExit(f"pipeline config requires a '{key}:' YAML mapping.")
    nested = dict(section)
    nested.setdefault("mode", "direct")
    nested["stage"] = stage
    return _config_direct_argv(nested)


def _pipeline_commands(config_path: str | Path) -> tuple[list[tuple[str, list[str]]], dict[str, Any], bool]:
    path = Path(config_path)
    config = _merge_launcher_config(_load_yaml_config(path))
    if _config_stage(config) != "pipeline":
        raise SystemExit(f"{config_path} is not a pipeline config.")
    _, dry_run = _config_lookup(config, "dry_run")
    stage1_argv = _nested_stage_argv(config, "stage1", "stage1")
    stage2_argv = _nested_stage_argv(config, "stage2", "stage2")
    commands: list[tuple[str, list[str]]] = [
        ("stage 1", [sys.executable, str(Path(__file__).resolve()), *stage1_argv]),
    ]
    if isinstance(config.get("stage1_adv"), dict):
        stage1_adv_argv = _nested_stage_argv(config, "stage1_adv", "stage1")
        commands.append(
            (
                "stage 1 adversarial",
                [sys.executable, str(Path(__file__).resolve()), *stage1_adv_argv],
            )
        )
    commands.append(("stage 2", [sys.executable, str(Path(__file__).resolve()), *stage2_argv]))
    return commands, config, bool(dry_run)


def _pipeline_has_stage1_init(cmd: list[str]) -> bool:
    return any(arg.startswith("init_ckpt_path=") or arg.startswith("ckpt_path=") for arg in cmd)


def _pipeline_section_model_type(section: dict[str, Any]) -> str:
    model = section.get("model")
    if isinstance(model, dict):
        model_type = model.get("type")
        if model_type:
            return str(model_type).strip().lower()
    for override in _coerce_hydra_overrides(section.get("overrides"), source="overrides"):
        if override.startswith("model.type="):
            return override.split("=", 1)[1].strip().lower()
    return "laser"


def _pipeline_latest_stage1_checkpoint(config: dict[str, Any]) -> Path:
    section = config.get("stage1")
    if not isinstance(section, dict):
        raise SystemExit("pipeline config requires a 'stage1:' YAML mapping.")
    output_root = section.get("output_dir", "outputs")
    model_type = _pipeline_section_model_type(section)
    from src.stage2_paths import infer_latest_stage1_checkpoint

    checkpoint = infer_latest_stage1_checkpoint(output_root=output_root, model_type=model_type)
    if checkpoint is None:
        raise SystemExit(f"No stage-1 checkpoint found under {output_root!r} for model type {model_type!r}.")
    return checkpoint


def _argv_from_config(path: str | Path) -> tuple[list[str], bool]:
    config = _merge_launcher_config(_load_yaml_config(path))
    mode = _config_mode(config)
    if _config_stage(config) == "pipeline":
        return _config_pipeline_argv(path, config), True
    if mode == "direct":
        return _config_direct_argv(config), True
    return _config_facade_argv(config), False


def _expand_config_argv(argv: list[str]) -> tuple[list[str], bool]:
    config_path, rest = _extract_config_arg(argv)
    if config_path is None:
        return argv, False
    config_argv, direct = _argv_from_config(config_path)
    if rest and (stage := _stage_token(rest[0])):
        rest = ["--stage", stage, *rest[1:]]
    return [*config_argv, *rest], direct


def _strip_direct_dry_run(argv: list[str]) -> tuple[list[str], bool]:
    cleaned = []
    dry_run = False
    for arg in argv:
        if arg in {"--dry-run", "--dry_run"}:
            dry_run = True
        else:
            cleaned.append(arg)
    return cleaned, dry_run


def _optional_container_for_cli_tests(value):
    if value is None:
        return None
    try:
        from omegaconf import OmegaConf
    except Exception:
        return value
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _cfg_attr(obj, name: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _sample_text_prompts(cfg) -> list[str]:
    train_ar = _cfg_attr(cfg, "train_ar")
    prompts = _optional_container_for_cli_tests(_cfg_attr(train_ar, "sample_text_prompts"))
    if prompts:
        return list(prompts)
    ar = _cfg_attr(cfg, "ar")
    prompts = _optional_container_for_cli_tests(_cfg_attr(ar, "sample_text_prompts"))
    return list(prompts or [])


_STAGE_ENTRYPOINTS = None


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
        description=(
            "Unified LASER training launcher. Pass stage1/stage2 as the first argument "
            "or use --stage. Use --config configs/exp.yaml to load run settings from YAML. "
            "Unknown trailing args are passed through as Hydra overrides."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="YAML experiment config for launcher settings and Hydra overrides.")
    parser.add_argument("--stage", default=None, type=_stage, help="Training stage: stage1 for autoencoder, stage2 for prior.")
    parser.add_argument("--dataset", default=None, help="Dataset config name, e.g. imagenet, celebahq, ffhq, vctk.")
    parser.add_argument("--modality", default=None, choices=("image", "audio"))
    parser.add_argument("--conditioning", default="none", choices=("none", "class", "text"))
    parser.add_argument("--adversarial", type=_str_bool, default=False, help="Enable stage-1 adversarial loss.")
    parser.add_argument("--num_gpus", "--num-gpus", type=_positive_int, default=1, help="Total GPUs requested for the run.")
    parser.add_argument(
        "--num_nodes",
        "--num-nodes",
        default="auto",
        help="Number of nodes. Use auto to infer from SLURM_JOB_NUM_NODES/SLURM_NNODES when present.",
    )
    parser.add_argument(
        "--devices_per_node",
        "--devices-per-node",
        type=_positive_int,
        default=None,
        help="Override Lightning train.devices/train_ar.devices directly.",
    )
    parser.add_argument("--downsample_layers", "--downsample-layers", type=int, default=5, choices=(5, 6))
    parser.add_argument("--sparsity_level", "--sparsity-level", type=_positive_int, default=3)
    parser.add_argument("--num_embeddings", "--num-embeddings", type=_positive_int, default=None)
    parser.add_argument("--embedding_dim", "--embedding-dim", type=_positive_int, default=None)
    parser.add_argument("--image_size", "--image-size", type=_positive_int, default=None)
    parser.add_argument("--data_dir", "--data-dir", default=None)
    parser.add_argument("--batch_size", "--batch-size", type=_positive_int, default=None, help="Per-process batch size.")
    parser.add_argument("--num_workers", "--num-workers", type=int, default=None)
    parser.add_argument("--epochs", type=_positive_int, default=None)
    parser.add_argument("--max_steps", "--max-steps", type=int, default=-1)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--learning_rate", "--learning-rate", default=None)
    parser.add_argument("--dict_learning_rate", "--dict-learning-rate", default=None)
    parser.add_argument("--output_root", "--output-root", default=_default_output_root())
    parser.add_argument("--output_dir", "--output-dir", default=None)
    parser.add_argument("--run_name", "--run-name", default=None)
    parser.add_argument("--project", default="laser")
    parser.add_argument("--token_cache_path", "--token-cache-path", default=None, help="Required for --stage 2.")
    parser.add_argument("--num_classes", "--num-classes", type=_positive_int, default=None)
    parser.add_argument("--dry_run", "--dry-run", action="store_true", help="Print the translated command without executing it.")
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


def _dedupe_tags(tags: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for tag in tags:
        text = str(tag).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _selected_checkpoint_paths(checkpoint_callback) -> list[Path]:
    """Return monitored top-k checkpoints plus last.ckpt, preserving order."""
    paths: list[Path] = []
    best_k = getattr(checkpoint_callback, "best_k_models", None) or {}
    if best_k:
        mode = str(getattr(checkpoint_callback, "mode", "min") or "min").lower()

        def _score(item):
            score = item[1]
            detach = getattr(score, "detach", None)
            if callable(detach):
                score = detach()
            cpu = getattr(score, "cpu", None)
            if callable(cpu):
                score = cpu()
            item_method = getattr(score, "item", None)
            if callable(item_method):
                return float(item_method())
            return float(score)

        paths.extend(
            Path(path)
            for path, _ in sorted(
                best_k.items(),
                key=_score,
                reverse=(mode == "max"),
            )
        )
    best_model_path = str(getattr(checkpoint_callback, "best_model_path", "") or "").strip()
    if best_model_path:
        paths.append(Path(best_model_path))
    last_model_path = str(getattr(checkpoint_callback, "last_model_path", "") or "").strip()
    if last_model_path:
        paths.append(Path(last_model_path))

    selected: list[Path] = []
    seen = set()
    for path in paths:
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        selected.append(resolved)
    return selected


def _make_selected_checkpoint_artifact_callback(callback_base):
    class SelectedCheckpointArtifactCallback(callback_base):
        """Upload top-k model checkpoints plus last.ckpt as one W&B artifact."""

        def __init__(
            self,
            checkpoint_callback,
            *,
            artifact_prefix: str = "model",
            every_n_epochs: int = 1,
        ):
            super().__init__()
            self.checkpoint_callback = checkpoint_callback
            self.artifact_prefix = str(artifact_prefix or "model")
            self.every_n_epochs = max(1, int(every_n_epochs or 1))
            self._last_signature = None

        def _file_digest(self, path: Path) -> str:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        def _signature(self, paths: list[Path]) -> tuple:
            return tuple((path.name, path.stat().st_size, self._file_digest(path)) for path in paths)

        def _upload(self, trainer, *, reason: str) -> None:
            if not bool(getattr(trainer, "is_global_zero", True)):
                return
            logger = getattr(trainer, "logger", None)
            experiment = getattr(logger, "experiment", None)
            if experiment is None or not hasattr(experiment, "log_artifact"):
                return
            paths = _selected_checkpoint_paths(self.checkpoint_callback)
            if not paths:
                return
            signature = self._signature(paths)
            if signature == self._last_signature:
                return

            import wandb

            run_id = str(getattr(experiment, "id", "") or getattr(logger, "version", "") or "run")
            artifact = wandb.Artifact(
                name=f"{self.artifact_prefix}-{run_id}-selected-checkpoints",
                type="model",
                description="Automatically uploaded selected checkpoints: monitored top-k plus last.ckpt.",
                metadata={
                    "reason": reason,
                    "epoch": int(getattr(trainer, "current_epoch", -1)),
                    "global_step": int(getattr(trainer, "global_step", -1)),
                    "monitor": str(getattr(self.checkpoint_callback, "monitor", "") or ""),
                    "mode": str(getattr(self.checkpoint_callback, "mode", "") or ""),
                    "save_top_k": int(getattr(self.checkpoint_callback, "save_top_k", -1)),
                    "save_last": bool(getattr(self.checkpoint_callback, "save_last", False)),
                    "checkpoint_paths": [str(path) for path in paths],
                },
            )
            for path in paths:
                artifact.add_file(str(path), name=path.name)
            epoch = int(getattr(trainer, "current_epoch", 0))
            aliases = ["latest", "best-plus-last", f"epoch-{epoch:03d}"]
            experiment.log_artifact(artifact, aliases=aliases)
            self._last_signature = signature

        def on_validation_end(self, trainer, pl_module) -> None:
            if bool(getattr(trainer, "sanity_checking", False)):
                return
            epoch = int(getattr(trainer, "current_epoch", 0))
            if (epoch + 1) % self.every_n_epochs != 0:
                return
            self._upload(trainer, reason="validation_end")

        def on_train_end(self, trainer, pl_module) -> None:
            self._upload(trainer, reason="train_end")

    return SelectedCheckpointArtifactCallback


def _stage1_wandb_tags(cfg) -> list[str]:
    model_cfg = cfg.model
    backbone = str(getattr(model_cfg, "backbone", "") or "unknown").strip().lower()
    channel_multipliers = getattr(model_cfg, "channel_multipliers", None)
    if backbone == "ddpm" and channel_multipliers:
        num_downsamples = max(0, len(channel_multipliers) - 1)
    else:
        num_downsamples = int(getattr(model_cfg, "num_downsamples", 0) or 0)

    patch_based = bool(getattr(model_cfg, "patch_based", False))
    tags = [
        f"backbone={backbone}",
        f"downsamples={num_downsamples}",
        f"sparsity={int(getattr(model_cfg, 'sparsity_level', 0) or 0)}",
        f"patch_based={str(patch_based).lower()}",
    ]
    if patch_based:
        tags.append(f"patch_size={int(getattr(model_cfg, 'patch_size', 0) or 0)}")
    return tags


def _validate(args: argparse.Namespace) -> None:
    if not args.stage:
        raise SystemExit("Specify a stage with stage1/stage2, --stage, or a YAML config containing stage.")
    if not args.dataset:
        raise SystemExit("--dataset is required for launcher-style runs. Direct YAML configs can use data=... instead.")
    if not args.modality:
        raise SystemExit("--modality is required for launcher-style runs. Direct YAML configs can use data=... instead.")
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
    script = Path(__file__).resolve()
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
    return [sys.executable, str(script), "stage1", *overrides]


def _stage2_command(args: argparse.Namespace, resources: ResourcePlan, extra: list[str]) -> list[str]:
    run_name = _run_name(args)
    output_dir = _output_dir(args, run_name)
    script = Path(__file__).resolve()
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
    return [sys.executable, str(script), "stage2", *overrides]


def build_command(argv: list[str] | None = None) -> list[str]:
    parser = _build_parser()
    expanded, direct_config = _expand_config_argv(list(argv) if argv is not None else sys.argv[1:])
    if direct_config:
        return [sys.executable, str(Path(__file__).resolve()), *_strip_direct_dry_run(expanded)[0]]
    normalized = _inject_stage_option(expanded)
    args, unknown = parser.parse_known_args(normalized)
    extra = _normalize_unknown(unknown)
    _validate(args)
    resources = _resolve_resources(args.num_gpus, args.num_nodes, args.devices_per_node)
    if args.stage == "1":
        return _stage1_command(args, resources, extra)
    return _stage2_command(args, resources, extra)

def _load_stage_entrypoints():
    global _STAGE_ENTRYPOINTS
    if _STAGE_ENTRYPOINTS is not None:
        return _STAGE_ENTRYPOINTS

    # -----------------------------------------------------------------------------
    # Stage 1 Hydra implementation
    # -----------------------------------------------------------------------------
    """Train the maintained stage-1 autoencoder."""

    import os
    import subprocess
    import sys
    import warnings

    if sys.version_info < (3, 10):
        raise SystemExit(
            "ERROR: train.py stage1 requires Python >= 3.10. "
            "Set PYTHON_BIN to a supported environment or run through scripts/run.sh."
        )

    # Windows: PyTorch (LLVM OpenMP) and MKL/NumPy (Intel OpenMP) can both load and trigger OMP #15.
    if os.name == "nt":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    # Suppress TF32 deprecation warnings (PyTorch 2.9 with Lightning compatibility)
    warnings.filterwarnings('ignore', message='.*TF32.*')
    os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'

    import torch

    from src.hydra_argparse_compat import patch_argparse_for_hydra_on_py314

    patch_argparse_for_hydra_on_py314()
    import hydra
    from omegaconf import DictConfig, open_dict

    from src.lightning_warning_filters import register as register_lightning_warning_filters

    register_lightning_warning_filters()
    import lightning as pl
    from lightning.pytorch.callbacks import (
        Callback,
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
        RichProgressBar,
    )
    from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch.plugins.environments import LightningEnvironment
    import wandb
    from datetime import datetime

    # Reduce DeepSpeed info logs
    os.environ.setdefault("DEEPSPEED_LOG_LEVEL", "warning")
    # Required by cuBLAS for deterministic kernels on supported CUDA paths.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    torch.set_float32_matmul_precision('medium')

    from src.pl_trainer_util import resolve_val_check_interval
    from src.stage1_setup import (
        build_stage1_datamodule,
        build_stage1_model,
        data_config_from_section,
        infer_data_channels,
    )

    # Configure progress bar theme
    progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="green1",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82"
        ),
        leave=True
    )

    STAGE1_TITLE = "STAGE 1: AUTOENCODER TRAINING"
    STAGE1_MODE = "stage1_autoencoder"
    CHECKPOINT_SAVE_TOP_K = 3
    CHECKPOINT_SAVE_LAST = True
    CHECKPOINT_EVERY_N_EPOCHS = 1
    CHECKPOINT_UPLOAD_TO_WANDB = True
    CHECKPOINT_UPLOAD_EVERY_N_EPOCHS = 1


    def _default_stage1_run_name(model_type: str) -> str:
        return f"{str(model_type).strip().lower()}-autoencoder"


    def _resolve_ckpt_file(path: str) -> str:
        path = os.path.expanduser(str(path))
        if os.path.isdir(path):
            preferred = [
                "final.ckpt",
                "last.ckpt",
                "mp_rank_00_model_states.pt",
                "model.pth",
                "model.pt",
                "state_dict.pth",
                "state_dict.pt",
                "weights.pt",
                "weights.pth",
            ]
            for name in preferred:
                cand = os.path.join(path, name)
                if os.path.isfile(cand):
                    return cand
            for root, _, files in os.walk(path):
                for filename in sorted(files):
                    if filename.endswith((".pt", ".pth", ".ckpt", ".bin")):
                        return os.path.join(root, filename)
            raise FileNotFoundError(f"No checkpoint file found under directory: {path}")
        return path


    @hydra.main(config_path="configs", config_name="config", version_base="1.2")
    def train_stage1(cfg: DictConfig):
        """
        Main training function using Hydra for configuration.
    
        Args:
            cfg: Hydra configuration object containing model and training parameters
        """
        ckpt_path = getattr(cfg, "ckpt_path", None)
        if ckpt_path:
            ckpt_path = _resolve_ckpt_file(ckpt_path)
            print(f"\nResume checkpoint: {ckpt_path}")
        init_ckpt_path = getattr(cfg, "init_ckpt_path", None)
        if init_ckpt_path:
            init_ckpt_path = _resolve_ckpt_file(init_ckpt_path)
            print(f"\nInitialize weights from checkpoint: {init_ckpt_path}")
        if ckpt_path and init_ckpt_path:
            raise ValueError("Use ckpt_path to resume training or init_ckpt_path to initialize weights, not both.")
        deterministic = bool(getattr(cfg.train, "deterministic", False))
        resolved_in_channels = infer_data_channels(cfg.data)
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = deterministic
            torch.backends.cudnn.benchmark = not deterministic

        # Print detailed experiment configuration
        print("\n" + "=" * 60)
        print(STAGE1_TITLE)
        print("=" * 60)
        print("\nExperiment Configuration:")
    
        print("\nGeneral Settings:")
        print("Stage Role: autoencoder training")
        print(f"Random Seed: {cfg.seed}")
        print(f"Output Directory: {cfg.output_dir}")
    
        print("\nDataset Configuration:")
        print(f"Dataset: {cfg.data.dataset}")
        print(f"Data Directory: {cfg.data.data_dir}")
        print(f"Batch Size: {cfg.data.batch_size}")
        print(f"Number of Workers: {cfg.data.num_workers}")
        print(f"Image Size: {cfg.data.image_size}")
        print(f"Mean: {cfg.data.mean}")
        print(f"Std: {cfg.data.std}")
        if str(cfg.data.dataset).strip().lower() in {"vctk", "maestro"}:
            print(f"Audio Representation: {getattr(cfg.data, 'audio_representation', 'spectrogram')}")
            print(f"Sample Rate: {cfg.data.sample_rate}")
            print(f"Audio Samples Per Clip: {cfg.data.audio_num_samples}")
            print(f"STFT FFT Size: {cfg.data.stft_n_fft}")
            print(f"STFT Hop Length: {cfg.data.stft_hop_length}")

        print("\nModel Configuration:")
        print(f"Model Type: {cfg.model.type}")
        print(f"Input Channels: {resolved_in_channels}")
        print(f"Hidden Dimensions: {cfg.model.num_hiddens}")
        print(f"Embedding Dimensions: {cfg.model.embedding_dim}")
        print(f"Number of Residual Blocks: {cfg.model.num_residual_blocks}")
        print(f"Residual Hidden Dimensions: {cfg.model.num_residual_hiddens}")
        if cfg.model.type == "laser":
            print(f"Backbone: {getattr(cfg.model, 'backbone', 'simple')}")
            print(f"Dictionary Size: {cfg.model.num_embeddings}")
            print(f"Sparsity: {cfg.model.sparsity_level}")
            print(f"Bypass Bottleneck: {bool(getattr(cfg.model, 'bypass_bottleneck', False))}")
            print(f"Coefficient Quantization Bound: {getattr(cfg.model, 'coef_max', None)}")
            if str(getattr(cfg.model, 'backbone', 'simple')).strip().lower() != "simple":
                print(f"Downsamples: {getattr(cfg.model, 'num_downsamples', 2)}")
                print(f"Attention Resolutions: {tuple(getattr(cfg.model, 'attn_resolutions', ())) or ()}")
                print(f"Use Mid Attention: {bool(getattr(cfg.model, 'use_mid_attention', True))}")
                channel_multipliers = getattr(cfg.model, 'channel_multipliers', None)
                if channel_multipliers not in (None, "", ()):
                    print(f"Channel Multipliers: {tuple(channel_multipliers)}")
                print(
                    "Backbone Latent Channels: "
                    f"{getattr(cfg.model, 'backbone_latent_channels', cfg.model.embedding_dim)}"
                )
        elif cfg.model.type == "vqvae":
            print(f"Number of Embeddings: {cfg.model.num_embeddings}")
        else:
            raise ValueError(f"Unsupported model type: {cfg.model.type}")
    
        print("\nTraining Configuration:")
        print(f"Learning Rate: {cfg.train.learning_rate}")
        print(f"Reconstruction MSE Weight: {float(getattr(cfg.model, 'recon_mse_weight', 1.0))}")
        print(f"Reconstruction L1 Weight: {float(getattr(cfg.model, 'recon_l1_weight', 0.0))}")
        print(f"Reconstruction Edge Weight: {float(getattr(cfg.model, 'recon_edge_weight', 0.0))}")
        print(f"Audio Multi-Resolution Loss Weight: {float(getattr(cfg.model, 'audio_multires_loss_weight', 0.0))}")
        print(f"Audio Multi-Resolution Scales: {tuple(getattr(cfg.model, 'audio_multires_scales', (1, 2, 4, 8)))}")
        print(f"Perceptual Weight: {float(getattr(cfg.model, 'perceptual_weight', 0.0))}")
        print(f"Perceptual Start Step: {int(getattr(cfg.model, 'perceptual_start_step', 0))}")
        print(f"Perceptual Warmup Steps: {int(getattr(cfg.model, 'perceptual_warmup_steps', 0))}")
        print(f"Adversarial Weight: {float(getattr(cfg.model, 'adversarial_weight', 0.0))}")
        print(f"Adversarial Start Step: {int(getattr(cfg.model, 'adversarial_start_step', 0))}")
        print(f"Adversarial Warmup Steps: {int(getattr(cfg.model, 'adversarial_warmup_steps', 0))}")
        print(f"Adversarial Start Recon MSE: {getattr(cfg.model, 'adversarial_start_recon_mse', None)}")
        print(f"Adversarial Quality EMA Decay: {float(getattr(cfg.model, 'adversarial_quality_ema_decay', 0.99))}")
        print(f"Beta: {cfg.train.beta}")
        print(f"Max Epochs: {cfg.train.max_epochs}")
        print(f"Max Steps: {getattr(cfg.train, 'max_steps', -1)}")
        print(f"Accelerator: {cfg.train.accelerator}")
        print(f"Num Nodes: {getattr(cfg.train, 'num_nodes', 1)}")
        print(f"Devices: {cfg.train.devices}")
        print(f"Precision: {cfg.train.precision}")
        print(f"Gradient Clip Value: {cfg.train.gradient_clip_val}")
        print(f"Deterministic: {deterministic}")
        print(f"Limit Train Batches: {getattr(cfg.train, 'limit_train_batches', 1.0)}")
        print(f"Limit Val Batches: {getattr(cfg.train, 'limit_val_batches', 1.0)}")
        print(f"Limit Test Batches: {getattr(cfg.train, 'limit_test_batches', 1.0)}")
        print(f"Run Test After Fit: {bool(getattr(cfg.train, 'run_test_after_fit', False))}")
    
        print("\nWandB Configuration:")
        print(f"Project: {cfg.wandb.project}")
        print(f"Run Name: {cfg.wandb.name}")
        print(f"Save Directory: {cfg.wandb.save_dir}")
    
        # Resolve checkpoint directory (base from config + run timestamp + model type)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_ckpt_dir = getattr(cfg.checkpoint, "dirpath", os.path.join(cfg.output_dir, "checkpoints"))
        run_ckpt_dir = os.path.join(base_ckpt_dir, f'run_{timestamp}', cfg.model.type)
        os.makedirs(run_ckpt_dir, exist_ok=True)

        # Determine monitor key and mode (configurable, with safe defaults per model)
        configured_monitor = getattr(cfg.checkpoint, "monitor", None)
        if configured_monitor:
            monitor_key = configured_monitor
        else:
            monitor_key = "val/loss"
        monitor_mode = getattr(cfg.checkpoint, "mode", "min")
        filename_template = getattr(cfg.checkpoint, "filename", f"{cfg.model.type}-{{epoch:03d}}")
        checkpoint_upload_to_wandb = bool(
            getattr(cfg.checkpoint, "upload_to_wandb", CHECKPOINT_UPLOAD_TO_WANDB)
        )
        checkpoint_upload_every_n_epochs = max(
            1,
            int(getattr(cfg.checkpoint, "upload_every_n_epochs", CHECKPOINT_UPLOAD_EVERY_N_EPOCHS) or 1),
        )
        with open_dict(cfg):
            cfg.checkpoint.save_top_k = CHECKPOINT_SAVE_TOP_K
            cfg.checkpoint.save_last = CHECKPOINT_SAVE_LAST
            cfg.checkpoint.upload_to_wandb = checkpoint_upload_to_wandb
            cfg.checkpoint.upload_every_n_epochs = checkpoint_upload_every_n_epochs

        print("\nCheckpoint Configuration:")
        print(f"Base Save Directory: {base_ckpt_dir}")
        print(f"Run Save Directory:  {run_ckpt_dir}")
        print(f"Filename Template:   {filename_template}")
        print(f"Monitor:             {monitor_key} (mode={monitor_mode})")
        print(f"Save Top K:          {cfg.checkpoint.save_top_k}")
        print(f"Save Last:           {cfg.checkpoint.save_last}")
        print(f"Every N Epochs:      {CHECKPOINT_EVERY_N_EPOCHS}")
        print(f"W&B Selected Checkpoint Upload: {checkpoint_upload_to_wandb}")
        print(f"W&B Upload Every N Epochs:      {checkpoint_upload_every_n_epochs}")
        print("=" * 60 + "\n")

        # Set random seed for reproducibility
        pl.seed_everything(cfg.seed, workers=True)
    
        # Print GPU information
        print(f"GPU available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
    
        # Initialize data module
        print(f"Initializing data module for dataset: {cfg.data.dataset}")
        data_config = data_config_from_section(cfg.data)
        datamodule = build_stage1_datamodule(data_config)

        # Print dataset info for debugging
        print(f"Using dataset: {cfg.data.dataset}")
        print(f"Data module type: {type(datamodule).__name__}")

        if (
            str(getattr(cfg.model, "type", "")).strip().lower() == "vqvae"
            and bool(getattr(cfg.model, "speaker_conditioning", False))
            and int(getattr(cfg.model, "speaker_conditioning_num_speakers", 0) or 0) <= 0
            and str(getattr(cfg.data, "dataset", "")).strip().lower() in {"vctk", "maestro"}
        ):
            datamodule.prepare_data()
            datamodule.setup("fit")
            num_speakers = int(getattr(datamodule, "num_speakers", 0) or 0)
            if num_speakers <= 0:
                raise RuntimeError("Speaker conditioning was requested, but no speakers were found in the data module.")
            with open_dict(cfg):
                cfg.model.speaker_conditioning_num_speakers = num_speakers
            print(f"Inferred VQ-VAE speaker conditioning classes: {num_speakers}")

        if int(cfg.model.in_channels) != resolved_in_channels:
            print(
                f"Adjusting model input channels from {int(cfg.model.in_channels)} "
                f"to {resolved_in_channels} to match dataset {cfg.data.dataset}."
            )
    
        model = build_stage1_model(cfg.model, cfg.train, cfg.data)
        if init_ckpt_path:
            from src.checkpoint_io import extract_state_dict, load_torch_payload

            payload = load_torch_payload(init_ckpt_path, map_location="cpu")
            state_dict = extract_state_dict(payload)
            if not isinstance(state_dict, dict):
                raise RuntimeError(f"Checkpoint payload does not contain a state_dict: {init_ckpt_path}")
            incompatible = model.load_state_dict(state_dict, strict=False)
            bottleneck = getattr(model, "bottleneck", None)
            data_initialized = getattr(bottleneck, "_data_initialized", None)
            if data_initialized is not None and hasattr(data_initialized, "fill_"):
                data_initialized.fill_(True)
            missing = len(getattr(incompatible, "missing_keys", ()) or ())
            unexpected = len(getattr(incompatible, "unexpected_keys", ()) or ())
            print(
                "Initialized stage-1 model weights "
                f"from {init_ckpt_path} (missing={missing}, unexpected={unexpected})."
            )
        trainer_gradient_clip_val = float(getattr(cfg.train, "gradient_clip_val", 0.0) or 0.0)
        uses_manual_opt = not bool(getattr(model, "automatic_optimization", True))
        if uses_manual_opt:
            model.manual_gradient_clip_val = trainer_gradient_clip_val
            trainer_gradient_clip_val = 0.0
            print(
                "Manual optimization active (adversarial); applying gradient clipping "
                f"inside the model with value {model.manual_gradient_clip_val}."
            )

        if ckpt_path:
            # Older checkpoints may contain metric module state such as
            # val_rfid/test_fid. Current models instantiate those lazily, so strict
            # resume would reject otherwise valid training checkpoints.
            model.strict_loading = False

        # Initialize wandb logger
        base_run_name = str(getattr(cfg.wandb, "name", "") or "").strip() or _default_stage1_run_name(cfg.model.type)
        if bool(getattr(cfg.wandb, "append_timestamp", False)):
            run_name = f"{base_run_name}_{timestamp}"
        else:
            run_name = base_run_name
        run_group = str(getattr(cfg.wandb, "group", "") or "").strip() or None
        run_tags = _dedupe_tags([*(getattr(cfg.wandb, "tags", []) or []), *_stage1_wandb_tags(cfg)])
        devices_cfg = cfg.train.devices
        try:
            num_devices = int(devices_cfg) if isinstance(devices_cfg, (int, str)) else len(devices_cfg)
        except Exception:
            num_devices = 1
        if num_devices > 1:
            wandb.setup()
        wandb_logger = WandbLogger(
            project=cfg.wandb.project,
            name=run_name,
            save_dir=cfg.wandb.save_dir,
            group=run_group,
            tags=run_tags if run_tags else None,
            log_model=False,
        )
        wandb_logger.log_hyperparams(
            {
                "training_stage": "stage1",
                "stage_role": "autoencoder_training",
                "training_mode": STAGE1_MODE,
                "model_type": cfg.model.type,
                "dataset": cfg.data.dataset,
                "input_channels": resolved_in_channels,
            }
        )

        # Initialize callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=run_ckpt_dir,
            filename=filename_template,
            save_top_k=CHECKPOINT_SAVE_TOP_K,
            monitor=monitor_key,
            mode=monitor_mode,
            save_last=CHECKPOINT_SAVE_LAST,
            every_n_epochs=CHECKPOINT_EVERY_N_EPOCHS,
        )
        callbacks = [checkpoint_callback]
        if checkpoint_upload_to_wandb:
            SelectedCheckpointArtifactCallback = _make_selected_checkpoint_artifact_callback(Callback)
            callbacks.append(
                SelectedCheckpointArtifactCallback(
                    checkpoint_callback,
                    artifact_prefix="model",
                    every_n_epochs=checkpoint_upload_every_n_epochs,
                )
            )
        callbacks.extend([
            LearningRateMonitor(logging_interval='step'),
            progress_bar,
        ])
        # Add EarlyStopping only if configured
        if getattr(cfg.train, "early_stopping_patience", None):
            callbacks.insert(1, EarlyStopping(
                monitor=monitor_key,
                patience=cfg.train.early_stopping_patience,
                mode=monitor_mode
            ))

        # Initialize trainer
        # Choose DDP only when using >1 device. Single-GPU DDP still inits torch.distributed (NCCL on CUDA),
        # which is unavailable on many Windows PyTorch builds — use auto instead.
        strategy_cfg = getattr(cfg.train, "strategy", None)
        if strategy_cfg is None:
            if cfg.model.type == "vqvae" and num_devices and num_devices > 1:
                strategy_cfg = "ddp"
        # Lightning rejects strategy=None; null / unset in config means default (auto).
        if strategy_cfg is None:
            strategy_cfg = "auto"
        strat_lower = str(strategy_cfg).lower()
        if num_devices <= 1 and strat_lower in ("ddp", "ddp_spawn", "ddp_notebook"):
            strategy_cfg = "auto"
            strat_lower = "auto"
        if "find_unused" in strat_lower:
            strategy_cfg = "ddp"
            strat_lower = "ddp"
            print("Using standard DDP; unused-parameter detection is disabled.")
        val_check_interval = resolve_val_check_interval(
            datamodule, getattr(cfg.train, "val_check_interval", 1.0)
        )
        max_steps = int(getattr(cfg.train, "max_steps", -1) or -1)
        trainer_plugins = [LightningEnvironment()] if num_devices > 1 and strat_lower.startswith("ddp") else None
        trainer = pl.Trainer(
            max_epochs=cfg.train.max_epochs,
            max_steps=max_steps,
            accelerator=cfg.train.accelerator,
            num_nodes=int(getattr(cfg.train, "num_nodes", 1) or 1),
            devices=cfg.train.devices,
            strategy=strategy_cfg,
            plugins=trainer_plugins,
            logger=wandb_logger,
            callbacks=callbacks,
            precision=cfg.train.precision,
            gradient_clip_val=trainer_gradient_clip_val,
            log_every_n_steps=cfg.train.log_every_n_steps,
            val_check_interval=val_check_interval,
            limit_train_batches=getattr(cfg.train, "limit_train_batches", 1.0),
            limit_val_batches=getattr(cfg.train, "limit_val_batches", 1.0),
            limit_test_batches=getattr(cfg.train, "limit_test_batches", 1.0),
            deterministic=deterministic,
            enable_progress_bar=True,
            enable_model_summary=(str(cfg.train.precision) == "32"),
            reload_dataloaders_every_n_epochs=0,
            num_sanity_val_steps=0,
        )

        # Train and test model (use PyTorch defaults for matmul precision to avoid API mixing)
        print("\nStarting autoencoder training...")
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
        print("\nAutoencoder training complete.")

        final_ckpt_path = os.path.join(run_ckpt_dir, "final.ckpt")
        # In DDP, Lightning's checkpoint path may involve strategy collectives. All
        # ranks need to enter the call; Lightning handles rank-zero-only file writes.
        trainer.save_checkpoint(final_ckpt_path)
        if trainer.is_global_zero:
            print(f"Saved final stage-1 checkpoint: {final_ckpt_path}")

        if not bool(getattr(cfg.train, "run_test_after_fit", False)):
            return

        # Keep DDP test evaluation distributed. Launching a new single-rank trainer
        # while the original process group/environment is still alive can hang when
        # model logs use sync_dist=True.
        if num_devices > 1 and str(strategy_cfg).lower().startswith("ddp"):
            if trainer.is_global_zero:
                print("\nRunning autoencoder test evaluation with the DDP trainer...")
            trainer.test(model, datamodule=datamodule)
            if trainer.is_global_zero:
                print("\nAutoencoder evaluation complete.")
            return

        if not trainer.is_global_zero:
            return

        print("\nRunning autoencoder test evaluation...")
        test_trainer = pl.Trainer(
            accelerator=('gpu' if (cfg.train.accelerator == 'gpu' and torch.cuda.is_available()) else 'cpu'),
            devices=1,
            logger=wandb_logger,
            precision=cfg.train.precision,
            deterministic=deterministic,
            limit_test_batches=getattr(cfg.train, "limit_test_batches", 1.0),
            enable_progress_bar=True,
            enable_model_summary=(str(cfg.train.precision) == "32")
        )
        test_trainer.test(model, datamodule=datamodule)
        print("\nAutoencoder evaluation complete.")

    # -----------------------------------------------------------------------------
    # Stage 2 Hydra implementation
    # -----------------------------------------------------------------------------
    """Train the maintained stage-2 transformer prior and save generation previews."""

    import os
    import sys
    import warnings
    from datetime import datetime
    from pathlib import Path

    if sys.version_info < (3, 10):
        raise SystemExit(
            "ERROR: train.py stage2 requires Python >= 3.10. "
            "Set PYTHON_BIN to a supported environment or run through scripts/run.sh."
        )

    if os.name == "nt":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    warnings.filterwarnings("ignore", message=".*TF32.*")
    os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

    from src.hydra_argparse_compat import patch_argparse_for_hydra_on_py314

    patch_argparse_for_hydra_on_py314()
    import hydra

    from src.lightning_warning_filters import register as reg_lit_warn

    reg_lit_warn()
    import lightning as pl
    import torch
    import wandb
    from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint, RichProgressBar
    from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch.plugins.environments import LightningEnvironment
    from omegaconf import DictConfig, OmegaConf

    from src.data.token_cache import TokenCacheDataModule
    from src.models.sparse_token_prior import (
        SparseTokenPriorModule,
        build_sparse_prior_from_cache,
        infer_sparse_vocab_sizes,
    )
    from src.pl_trainer_util import resolve_val_check_interval
    from src.stage2_compat import ensure_stage2_cache_metadata as add_cache_meta
    from src.stage2_metrics import build_stage2_metrics_payload
    from src.stage2_paths import (
        default_token_cache_path,
        infer_latest_stage1_checkpoint,
        infer_latest_token_cache as pick_cache,
    )
    from src.stage2_preview import (
        Stage2SamplePreviewCallback,
        save_final_generation_preview,
    )
    from src.wandb_media import log_wandb_payload

    torch.set_float32_matmul_precision("medium")

    CHECKPOINT_SAVE_TOP_K = 3
    CHECKPOINT_SAVE_LAST = True
    CHECKPOINT_EVERY_N_EPOCHS = 1
    CHECKPOINT_UPLOAD_TO_WANDB = True
    CHECKPOINT_UPLOAD_EVERY_N_EPOCHS = 1

    bar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="green1",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
        ),
        leave=True,
    )

    STAGE2_TITLE = "STAGE 2: TRANSFORMER PRIOR TRAINING + GENERATION"
    STAGE2_MODE = "stage2_transformer_generation"


    def _default_stage2_run_name() -> str:
        return "stage2-transformer"


    def _optional_container(value):
        if value is None:
            return None
        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
        return value

    def _sample_text_prompts(cfg) -> list[str]:
        train_ar = getattr(cfg, "train_ar", None)
        prompts = _optional_container(getattr(train_ar, "sample_text_prompts", None)) if train_ar is not None else None
        if prompts:
            return list(prompts)
        ar = getattr(cfg, "ar", None)
        prompts = _optional_container(getattr(ar, "sample_text_prompts", None)) if ar is not None else None
        return list(prompts or [])


    def arch_name(raw) -> str:
        text = str(raw or "sparse_spatial_depth").strip().lower()
        if text in {"sparse_spatial_depth", "spatial_depth"}:
            return "spatial_depth"
        if text in {"mingpt", "gpt"}:
            return "gpt"
        raise ValueError(f"Unsupported stage-2 architecture: {raw!r}")


    def _cfg_value(section, key: str, default=None):
        if section is None:
            return default
        value = getattr(section, key, default)
        if value is None:
            return default
        return value


    def _float_sequence(value, *, fallback):
        if value is None:
            value = fallback
        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
        return [str(float(v)) for v in value]


    def _resolve_token_cache_output(cfg, cache_cfg) -> Path:
        explicit = str(_cfg_value(cache_cfg, "output", "") or "").strip()
        if explicit:
            return Path(explicit).expanduser().resolve()
        current = str(getattr(cfg, "token_cache_path", "") or "").strip()
        if current:
            return Path(current).expanduser().resolve()
        return default_token_cache_path(
            ar_output_dir=cfg.output_dir,
            dataset=str(_cfg_value(cfg.data, "dataset", "celeba")),
            split=str(_cfg_value(cache_cfg, "split", "train")),
            image_size=int(_cfg_value(cfg.data, "image_size", 128)),
            coeff_bins=int(_cfg_value(cache_cfg, "coeff_vocab_size", 16)),
            coeff_quantization=str(_cfg_value(cache_cfg, "coeff_quantization", "uniform")),
            coeff_mu=float(_cfg_value(cache_cfg, "coeff_mu", 0.0)),
        ).resolve()


    def _maybe_build_token_cache(cfg) -> None:
        cache_cfg = getattr(cfg, "token_cache", None)
        if cache_cfg is None or not bool(_cfg_value(cache_cfg, "build", False)):
            return

        output_path = _resolve_token_cache_output(cfg, cache_cfg)
        force = bool(_cfg_value(cache_cfg, "force", False))
        if output_path.is_file() and not force:
            cfg.token_cache_path = str(output_path)
            print(f"Using existing token cache: {output_path}")
            return

        stage1_checkpoint = str(_cfg_value(cache_cfg, "stage1_checkpoint", "") or "").strip()
        if stage1_checkpoint:
            stage1_checkpoint_path = Path(stage1_checkpoint).expanduser().resolve()
        else:
            stage1_root = _cfg_value(cache_cfg, "stage1_output_root", None)
            if stage1_root is None:
                stage1_root = Path(str(cfg.output_dir)).expanduser().resolve().parent / "stage1"
            stage1_checkpoint_path = infer_latest_stage1_checkpoint(
                output_root=stage1_root,
                model_type="laser",
            )
            if stage1_checkpoint_path is None:
                raise FileNotFoundError(
                    "Could not infer a stage-1 checkpoint for token cache extraction. "
                    "Set token_cache.stage1_checkpoint or token_cache.stage1_output_root."
                )
        if not stage1_checkpoint_path.is_file():
            raise FileNotFoundError(f"Stage-1 checkpoint not found: {stage1_checkpoint_path}")

        dataset = str(_cfg_value(cfg.data, "dataset", "celeba")).strip().lower()
        data_dir = str(_cfg_value(cfg.data, "data_dir", "") or "").strip()
        if not data_dir:
            raise ValueError("token_cache.build=true requires data.data_dir")
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "scripts" / "tools" / "build_token_cache.py"),
            "--stage1_checkpoint",
            str(stage1_checkpoint_path),
            "--dataset",
            dataset,
            "--data_dir",
            data_dir,
            "--split",
            str(_cfg_value(cache_cfg, "split", "train")),
            "--cache_mode",
            str(_cfg_value(cache_cfg, "cache_mode", "quantized")),
            "--image_size",
            str(int(_cfg_value(cfg.data, "image_size", 128))),
            "--batch_size",
            str(int(_cfg_value(cache_cfg, "batch_size", _cfg_value(cfg.data, "batch_size", 64)))),
            "--num_workers",
            str(int(_cfg_value(cache_cfg, "num_workers", _cfg_value(cfg.data, "num_workers", 4)))),
            "--mean",
            *_float_sequence(_cfg_value(cfg.data, "mean", None), fallback=(0.5, 0.5, 0.5)),
            "--std",
            *_float_sequence(_cfg_value(cfg.data, "std", None), fallback=(0.5, 0.5, 0.5)),
            "--coeff_vocab_size",
            str(int(_cfg_value(cache_cfg, "coeff_vocab_size", 16))),
            "--coeff_quantization",
            str(_cfg_value(cache_cfg, "coeff_quantization", "uniform")),
            "--coeff_mu",
            str(float(_cfg_value(cache_cfg, "coeff_mu", 0.0))),
            "--coeff_calibration_percentile",
            str(float(_cfg_value(cache_cfg, "coeff_calibration_percentile", 99.5))),
            "--output",
            str(output_path),
            "--max_items",
            str(int(_cfg_value(cache_cfg, "max_items", 0))),
            "--device",
            str(_cfg_value(cache_cfg, "device", "auto")),
        ]
        coeff_max = _cfg_value(cache_cfg, "coeff_max", None)
        if coeff_max is not None:
            cmd.extend(["--coeff_max", str(coeff_max)])

        print("Building token cache before stage 2:")
        print("  Stage-1 checkpoint:", stage1_checkpoint_path)
        print("  Output cache:", output_path)
        subprocess.run(cmd, check=True)
        cfg.token_cache_path = str(output_path)


    def _preferred_module_device(module: torch.nn.Module) -> torch.device:
        """Return the module device, falling back to the current CUDA device."""
        try:
            param = next(module.parameters())
            if param.device.type != "cpu":
                return param.device
        except StopIteration:
            pass
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")


    def build(cfg: DictConfig):
        if not cfg.token_cache_path:
            cache_pt = pick_cache(ar_output_dir=cfg.output_dir)
            if cache_pt is None:
                need = Path(str(cfg.output_dir)).expanduser().resolve() / "token_cache"
                raise ValueError(
                    f"Sparse prior runs require token_cache_path, and no cache could be inferred under {need}"
                )
            cfg.token_cache_path = str(cache_pt)
            print(f"Inferred token_cache_path: {cfg.token_cache_path}")

        dm = TokenCacheDataModule(
            cache_path=cfg.token_cache_path,
            batch_size=cfg.train_ar.batch_size,
            num_workers=cfg.data.num_workers,
            seed=cfg.seed,
            validation_fraction=getattr(cfg.train_ar, "validation_split", 0.05),
            test_fraction=getattr(cfg.train_ar, "test_split", 0.05),
            max_items=getattr(cfg.train_ar, "max_items", 0),
            crop_h_sites=getattr(cfg.train_ar, "crop_h_sites", 0),
            crop_w_sites=getattr(cfg.train_ar, "crop_w_sites", 0),
        )
        dm.setup("fit")
        dm.cache = add_cache_meta(
            dm.cache,
            token_cache_path=cfg.token_cache_path,
            output_root=Path(str(cfg.output_dir)).expanduser().resolve().parent,
        )

        arch = arch_name(getattr(cfg.ar, "type", "sparse_spatial_depth"))
        real = dm.cache.get("coeffs_flat") is not None
        if real and arch != "spatial_depth":
            raise ValueError("Real-valued sparse-token caches are only supported with ar.type=sparse_spatial_depth.")
        if arch == "gpt":
            H, W, D = dm.token_shape
            if int(D) % 2 != 0:
                raise ValueError(
                    "ar.type=gpt requires an even token depth because it models an interleaved atom/coeff stream. "
                    f"Got token shape {(H, W, D)}. VQ-VAE caches use depth 1, so keep ar.type=sparse_spatial_depth."
                )
            seq_len = int(H * W * D)
            window_sites = int(getattr(cfg.ar, "window_sites", 0) or 0)
            if seq_len > 8192 and window_sites <= 0:
                raise ValueError(
                    "ar.type=gpt without ar.window_sites is a full-sequence GPT over the flattened sparse token stream and "
                    f"is not practical for sequence length {seq_len} from token shape {(H, W, D)}. "
                    "Use a much shorter token grid, set ar.window_sites for local sliding-window attention, "
                    "or keep ar.type=sparse_spatial_depth for this cache."
                )

        cache_meta = dm.cache.get("meta", {}) if isinstance(dm.cache, dict) else {}
        class_conditional = bool(getattr(cfg.ar, "class_conditional", False))
        text_conditional = bool(getattr(cfg.ar, "text_conditional", False))
        if class_conditional and dm.cache.get("class_labels") is None:
            raise ValueError("ar.class_conditional=true requires class_labels in the token cache.")
        if text_conditional and dm.cache.get("text_tokens") is None:
            raise ValueError("ar.text_conditional=true requires text_tokens in the token cache.")

        resolved_num_classes = int(getattr(cfg.ar, "num_classes", 0) or cache_meta.get("num_classes", 0) or 0)
        if class_conditional and resolved_num_classes <= 0:
            raise ValueError("ar.class_conditional=true requires ar.num_classes or cache meta num_classes > 0.")
        cfg.ar.num_classes = resolved_num_classes

        resolved_text_vocab_size = int(getattr(cfg.ar, "text_vocab_size", 0) or cache_meta.get("text_vocab_size", 0) or 0)
        resolved_text_max_length = int(getattr(cfg.ar, "text_max_length", 0) or cache_meta.get("text_max_length", 0) or 0)
        resolved_text_pad_id = int(getattr(cfg.ar, "text_pad_id", 0) or cache_meta.get("text_pad_id", 0) or 0)
        if text_conditional:
            if resolved_text_vocab_size <= 2 or resolved_text_max_length <= 0:
                raise ValueError(
                    "ar.text_conditional=true requires text_vocab_size > 2 and text_max_length > 0 "
                    "from config or token cache metadata."
                )
            cfg.ar.text_vocab_size = resolved_text_vocab_size
            cfg.ar.text_max_length = resolved_text_max_length
            cfg.ar.text_pad_id = resolved_text_pad_id

        prior = build_sparse_prior_from_cache(
            dm.cache,
            architecture=arch,
            total_vocab_size=cfg.ar.vocab_size,
            atom_vocab_size=cfg.ar.atom_vocab_size,
            coeff_vocab_size=cfg.ar.coeff_vocab_size,
            grid_shape=dm.token_shape,
            window_sites=getattr(cfg.ar, "window_sites", 0),
            d_model=cfg.ar.d_model,
            n_heads=cfg.ar.n_heads,
            n_layers=cfg.ar.n_layers,
            d_ff=cfg.ar.d_ff,
            dropout=cfg.ar.dropout,
            n_global_spatial_tokens=cfg.ar.n_global_spatial_tokens,
            autoregressive_coeffs=cfg.ar.autoregressive_coeffs,
            class_conditional=class_conditional,
            num_classes=resolved_num_classes,
            text_conditional=text_conditional,
            text_vocab_size=resolved_text_vocab_size,
            text_max_length=resolved_text_max_length,
            text_pad_id=resolved_text_pad_id,
            text_prefix_length=cfg.ar.text_prefix_length,
        )
        if real:
            atom_vocab = int(prior.atom_vocab_size)
            vocab = int(prior.cfg.vocab_size)
            coeff_vocab = 0
        else:
            vocab, atom_vocab, coeff_vocab = infer_sparse_vocab_sizes(
                dm.cache,
                total_vocab_size=cfg.ar.vocab_size,
                atom_vocab_size=cfg.ar.atom_vocab_size,
                coeff_vocab_size=cfg.ar.coeff_vocab_size,
            )
        if cfg.ar.vocab_size != vocab:
            print(f"Adjusting sparse vocab_size from {cfg.ar.vocab_size} to {vocab} from cache metadata")
            cfg.ar.vocab_size = vocab
        if cfg.ar.atom_vocab_size != atom_vocab:
            print(f"Resolved atom_vocab_size = {atom_vocab}")
            cfg.ar.atom_vocab_size = atom_vocab
        coeff_cfg = None if real else int(coeff_vocab)
        if cfg.ar.coeff_vocab_size != coeff_cfg:
            print(f"Resolved coeff_vocab_size = {coeff_cfg}")
            cfg.ar.coeff_vocab_size = coeff_cfg

        model = SparseTokenPriorModule(
            prior=prior,
            learning_rate=cfg.ar.learning_rate,
            weight_decay=cfg.ar.weight_decay,
            warmup_steps=cfg.ar.warmup_steps,
            min_lr_ratio=cfg.ar.min_lr_ratio,
            atom_loss_weight=cfg.ar.atom_loss_weight,
            coeff_loss_weight=cfg.ar.coeff_loss_weight,
            coeff_depth_weighting=cfg.ar.coeff_depth_weighting,
            coeff_focal_gamma=cfg.ar.coeff_focal_gamma,
            coeff_loss_type=cfg.ar.coeff_loss_type,
            coeff_huber_delta=cfg.ar.coeff_huber_delta,
            sample_coeff_temperature=cfg.ar.sample_coeff_temperature,
            sample_coeff_mode=cfg.ar.sample_coeff_mode,
        )
        model.save_hyperparameters(
            {
                "token_cache_path": str(Path(str(cfg.token_cache_path)).expanduser().resolve()),
                "resolved_atom_vocab_size": int(atom_vocab),
                "resolved_coeff_vocab_size": int(coeff_vocab),
                "resolved_total_vocab_size": int(vocab),
                "token_cache_real_valued": bool(real),
                "class_conditional": class_conditional,
                "num_classes": int(resolved_num_classes),
                "text_conditional": text_conditional,
                "text_vocab_size": int(resolved_text_vocab_size),
                "text_max_length": int(resolved_text_max_length),
            }
        )
        return model, dm


    @hydra.main(config_path="configs", config_name="config_ar", version_base="1.2")
    def train_stage2(cfg: DictConfig):
        mode = arch_name(getattr(cfg.ar, "type", "sparse_spatial_depth"))
        cfg.ar.type = mode
        _maybe_build_token_cache(cfg)

        print("\n" + "=" * 60)
        print(STAGE2_TITLE)
        print("=" * 60)
        print(f"\nTransformer Mode: {mode}")
        print(f"Token Cache: {cfg.token_cache_path}")

        print("\nModel Config:")
        print(f"  Vocab Size: {cfg.ar.vocab_size}")
        print(f"  Model Dim: {cfg.ar.d_model}")
        print(f"  Heads: {cfg.ar.n_heads}")
        print(f"  Layers: {cfg.ar.n_layers}")
        print(f"  FF Dim: {cfg.ar.d_ff}")
        print(f"  Dropout: {cfg.ar.dropout}")
        print(f"  Atom Vocab Size: {cfg.ar.atom_vocab_size}")
        print(f"  Coeff Vocab Size: {cfg.ar.coeff_vocab_size}")
        print(f"  Window Sites: {getattr(cfg.ar, 'window_sites', 0)}")
        print(f"  Global Spatial Tokens: {cfg.ar.n_global_spatial_tokens}")
        print(f"  Autoregressive Coeffs: {cfg.ar.autoregressive_coeffs}")
        print(f"  Coeff Loss Type: {cfg.ar.coeff_loss_type}")

        print("\nTrain Config:")
        print(f"  Learning Rate: {cfg.ar.learning_rate}")
        print(f"  Warmup Steps: {cfg.ar.warmup_steps}")
        print(f"  Max Steps: {cfg.ar.max_steps}")
        print(f"  Num Nodes: {getattr(cfg.train_ar, 'num_nodes', 1)}")
        print(f"  Batch Size: {cfg.train_ar.batch_size}")
        print(f"  Max Epochs: {cfg.train_ar.max_epochs}")
        print(f"  Limit Train Batches: {getattr(cfg.train_ar, 'limit_train_batches', 1.0)}")
        print(f"  Limit Val Batches: {getattr(cfg.train_ar, 'limit_val_batches', 1.0)}")
        print(f"  Limit Test Batches: {getattr(cfg.train_ar, 'limit_test_batches', 1.0)}")
        print(f"  Crop H Sites: {getattr(cfg.train_ar, 'crop_h_sites', 0)}")
        print(f"  Crop W Sites: {getattr(cfg.train_ar, 'crop_w_sites', 0)}")
        print("=" * 60 + "\n")

        pl.seed_everything(cfg.seed, workers=True)

        print(f"GPU available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU device: {torch.cuda.get_device_name(0)}")

        model, dm = build(cfg)
        print(f"Transformer Token Grid Shape: {dm.token_shape}")
        cache_meta = dm.metadata
        stage1_checkpoint = str(cache_meta.get("stage1_checkpoint") or "").strip()
        if stage1_checkpoint:
            print(f"Stage-1 decoder checkpoint: {stage1_checkpoint}")
        else:
            print("Stage-1 decoder checkpoint: unresolved from token cache metadata")
        cache_dataset = str(cache_meta.get("dataset") or "").strip()
        cfg_dataset = str(getattr(cfg.data, "dataset", "") or "").strip()
        if cache_dataset:
            if cfg_dataset and cfg_dataset.lower() != cache_dataset.lower():
                print(f"Stage-2 transformer cache dataset: {cache_dataset} (ignoring data.dataset={cfg_dataset})")
            else:
                print(f"Stage-2 transformer cache dataset: {cache_dataset}")

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total:,} total, {trainable:,} trainable")

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_dir = os.path.join(cfg.output_dir, "checkpoints", f"s2_{stamp}")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"\nCheckpoint dir: {ckpt_dir}")
        print(
            "Checkpoint policy: "
            f"save_top_k={CHECKPOINT_SAVE_TOP_K}, "
            f"save_last={CHECKPOINT_SAVE_LAST}, "
            f"every_n_epochs={CHECKPOINT_EVERY_N_EPOCHS}, "
            f"wandb_selected_checkpoint_upload={CHECKPOINT_UPLOAD_TO_WANDB}, "
            f"wandb_upload_every_n_epochs={CHECKPOINT_UPLOAD_EVERY_N_EPOCHS}"
        )

        base_run = str(getattr(cfg.wandb, "name", "") or "").strip() or _default_stage2_run_name()
        if bool(getattr(cfg.wandb, "append_timestamp", False)):
            run = f"{base_run}_{stamp}"
        else:
            run = base_run
        run_group = str(getattr(cfg.wandb, "group", "") or "").strip() or None
        run_tags = list(getattr(cfg.wandb, "tags", []) or [])
        devices_cfg = cfg.train_ar.devices
        try:
            num_devices = int(devices_cfg) if isinstance(devices_cfg, (int, str)) else len(devices_cfg)
        except Exception:
            num_devices = 1
        if num_devices > 1:
            wandb.setup()
        wandb_id = str(getattr(cfg.wandb, "id", "") or "").strip() or None
        wandb_resume = str(getattr(cfg.wandb, "resume", "") or "").strip() or None
        wandb_kwargs = {}
        if wandb_resume:
            wandb_kwargs["resume"] = wandb_resume
        wb = WandbLogger(
            project=cfg.wandb.project,
            name=run,
            save_dir=cfg.wandb.save_dir,
            group=run_group,
            tags=run_tags if run_tags else None,
            id=wandb_id,
            log_model=False,
            **wandb_kwargs,
        )
        step_every = int(getattr(cfg.train_ar, "sample_every_n_steps", 0) or 0)
        epoch_every = int(getattr(cfg.train_ar, "sample_every_n_epochs", 0) or 0)
        sample_out_dir = os.path.join(cfg.output_dir, "samples", f"s2_{stamp}")
        sample_variants = _optional_container(getattr(cfg.train_ar, "sample_variants", None))
        sample_text_prompts = _sample_text_prompts(cfg)
        sample_class_labels = _optional_container(getattr(cfg.train_ar, "sample_class_labels", None)) or []
        wb.log_hyperparams(
            {
                "training_stage": "stage2",
                "stage_role": "transformer_generation",
                "training_mode": STAGE2_MODE,
                "legacy_training_mode": "s2",
                "token_cache_path": cfg.token_cache_path,
                "ar_vocab_size": cfg.ar.vocab_size,
                "ar_d_model": cfg.ar.d_model,
                "ar_n_heads": cfg.ar.n_heads,
                "ar_n_layers": cfg.ar.n_layers,
                "ar_d_ff": cfg.ar.d_ff,
                "ar_dropout": cfg.ar.dropout,
                "dataset": cache_dataset or cfg_dataset or None,
                "config_dataset": cfg_dataset or None,
                "stage1_checkpoint": stage1_checkpoint or None,
                "sample_every_n_steps": step_every,
                "sample_every_n_epochs": epoch_every,
                "sample_num_images": int(getattr(cfg.train_ar, "sample_num_images", 4) or 4),
                "sample_temperature": float(getattr(cfg.train_ar, "sample_temperature", 1.0) or 1.0),
                "sample_top_k": int(getattr(cfg.train_ar, "sample_top_k", 0) or 0),
                "sample_class_labels": sample_class_labels,
                "sample_text_prompts": sample_text_prompts,
            }
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="s2-{epoch:03d}-{val/loss:.4f}",
            save_top_k=CHECKPOINT_SAVE_TOP_K,
            monitor="val/loss",
            mode="min",
            save_last=CHECKPOINT_SAVE_LAST,
            every_n_epochs=CHECKPOINT_EVERY_N_EPOCHS,
        )
        cbs = [checkpoint_callback]
        if CHECKPOINT_UPLOAD_TO_WANDB:
            SelectedCheckpointArtifactCallback = _make_selected_checkpoint_artifact_callback(Callback)
            cbs.append(
                SelectedCheckpointArtifactCallback(
                    checkpoint_callback,
                    artifact_prefix="model-stage2",
                    every_n_epochs=CHECKPOINT_UPLOAD_EVERY_N_EPOCHS,
                )
            )
        cbs.extend([
            LearningRateMonitor(logging_interval="step"),
            bar,
        ])
        if step_every > 0 or epoch_every > 0:
            cbs.append(
                Stage2SamplePreviewCallback(
                    cache_pt=str(cfg.token_cache_path),
                    out_dir=sample_out_dir,
                    step_every=step_every,
                    epoch_every=epoch_every,
                    n=int(getattr(cfg.train_ar, "sample_num_images", 4) or 4),
                    temp=float(getattr(cfg.train_ar, "sample_temperature", 1.0) or 1.0),
                    top_k=int(getattr(cfg.train_ar, "sample_top_k", 0) or 0),
                    ctemp=getattr(cfg.train_ar, "sample_coeff_temperature", cfg.ar.sample_coeff_temperature),
                    cmode=getattr(cfg.train_ar, "sample_coeff_mode", cfg.ar.sample_coeff_mode),
                    sample_variants=sample_variants,
                    s1_root=str(Path(str(cfg.output_dir)).expanduser().resolve().parent),
                    use_wandb=bool(getattr(cfg.train_ar, "sample_log_to_wandb", False)),
                    text_prompts=sample_text_prompts,
                    class_labels=sample_class_labels,
                )
            )

        val_every = resolve_val_check_interval(dm, getattr(cfg.train_ar, "val_check_interval", 1.0))
        strategy_cfg = getattr(cfg.train_ar, "strategy", "auto")
        strategy_lower = str(strategy_cfg).lower()
        if num_devices <= 1 and strategy_lower in ("ddp", "ddp_spawn", "ddp_notebook"):
            strategy_cfg = "auto"
            strategy_lower = "auto"
        trainer_plugins = [LightningEnvironment()] if num_devices > 1 and strategy_lower.startswith("ddp") else None
        trainer = pl.Trainer(
            max_epochs=cfg.train_ar.max_epochs,
            max_steps=int(getattr(cfg.ar, "max_steps", -1) or -1),
            accelerator=cfg.train_ar.accelerator,
            num_nodes=int(getattr(cfg.train_ar, "num_nodes", 1) or 1),
            devices=cfg.train_ar.devices,
            strategy=strategy_cfg,
            plugins=trainer_plugins,
            logger=wb,
            callbacks=cbs,
            precision=cfg.train_ar.precision,
            gradient_clip_val=cfg.train_ar.gradient_clip_val,
            log_every_n_steps=cfg.train_ar.log_every_n_steps,
            val_check_interval=val_every,
            limit_train_batches=getattr(cfg.train_ar, "limit_train_batches", 1.0),
            limit_val_batches=getattr(cfg.train_ar, "limit_val_batches", 1.0),
            limit_test_batches=getattr(cfg.train_ar, "limit_test_batches", 1.0),
            deterministic=True,
            enable_progress_bar=True,
            num_sanity_val_steps=2,
        )

        ckpt_path = str(getattr(cfg, "ckpt_path", "") or "").strip() or None
        if ckpt_path:
            print(f"\nResuming transformer prior training from: {ckpt_path}")
        else:
            print("\nStarting transformer prior training...")
        trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)

        run_test_cfg = getattr(cfg.train_ar, "run_test_after_fit", None)
        run_test_after_fit = (num_devices <= 1) if run_test_cfg is None else bool(run_test_cfg)
        if run_test_after_fit:
            print("\nRunning transformer prior test evaluation...")
            trainer.test(model, datamodule=dm)
        elif trainer.is_global_zero:
            print("\nSkipping post-fit transformer test evaluation for multi-device training.")

        compute_generation_fid = bool(getattr(cfg.train_ar, "compute_generation_fid", False))
        compute_audio_generation = bool(getattr(cfg.train_ar, "compute_audio_generation_metrics", False))
        generation_metric_samples = int(getattr(cfg.train_ar, "generation_metric_num_samples", 0) or 0)
        sample_count = int(getattr(cfg.train_ar, "sample_num_images", 4) or 0)
        if compute_generation_fid or compute_audio_generation:
            sample_count = max(sample_count, generation_metric_samples)
        final_samples_cfg = getattr(cfg.train_ar, "save_final_samples_after_fit", None)
        save_final_samples = (num_devices <= 1) if final_samples_cfg is None else bool(final_samples_cfg)
        if not save_final_samples and trainer.is_global_zero and sample_count > 0:
            print("Skipping final post-fit generation preview for multi-device training.")
        if save_final_samples and trainer.is_global_zero and sample_count > 0:
            try:
                sample_device = _preferred_module_device(model)
                if sample_device.type != "cpu":
                    model.to(sample_device)
                final_result = save_final_generation_preview(
                    trainer=trainer,
                    mod=model,
                    cache_pt=str(cfg.token_cache_path),
                    out_dir=sample_out_dir,
                    n=sample_count,
                    temp=float(getattr(cfg.train_ar, "sample_temperature", 1.0) or 1.0),
                    top_k=int(getattr(cfg.train_ar, "sample_top_k", 0) or 0),
                    ctemp=getattr(cfg.train_ar, "sample_coeff_temperature", cfg.ar.sample_coeff_temperature),
                    cmode=getattr(cfg.train_ar, "sample_coeff_mode", cfg.ar.sample_coeff_mode),
                    s1_root=str(Path(str(cfg.output_dir)).expanduser().resolve().parent),
                    use_wandb=bool(getattr(cfg.train_ar, "sample_log_to_wandb", False)),
                    text_prompts=sample_text_prompts,
                    class_labels=sample_class_labels,
                    return_batch=(compute_generation_fid or compute_audio_generation),
                )
                if isinstance(final_result, tuple):
                    final_raw, final_batch, final_cache = final_result
                else:
                    final_raw, final_batch, final_cache = final_result, None, None
                print(f"\nSaved final transformer generation preview: {final_raw}")
                if final_batch is not None and final_cache is not None and generation_metric_samples > 0:
                    metric_payload = build_stage2_metrics_payload(
                        final_batch.imgs.to(sample_device),
                        cfg=cfg,
                        cache=final_cache,
                        max_items=generation_metric_samples,
                        compute_fid=compute_generation_fid,
                        compute_audio=compute_audio_generation,
                    )
                    if metric_payload:
                        step = max(1, int(getattr(trainer, "global_step", 0) or 0))
                        log_wandb_payload(wb, metric_payload, step=step)
                        print(f"Logged final generation metrics: {sorted(metric_payload)}")
            except Exception as err:
                print(f"\nWarning: could not save final transformer generation preview ({err})")
        if save_final_samples:
            Stage2SamplePreviewCallback._barrier_if_needed(trainer)

        print("\nTransformer training complete.")
        print(f"Best checkpoint: {cbs[0].best_model_path}")
        return cbs[0].best_model_path


    train_ar = train_stage2

    _STAGE_ENTRYPOINTS = (train_stage1, train_stage2)
    return _STAGE_ENTRYPOINTS


def _dispatch_stage(stage: str, stage_argv: list[str]) -> int:
    sys.argv = [sys.argv[0], *stage_argv]
    train_stage1_entry, train_stage2_entry = _load_stage_entrypoints()
    if stage == "1":
        train_stage1_entry()
    else:
        train_stage2_entry()
    return 0


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    expanded, direct_config = _expand_config_argv(raw_argv)
    if direct_config:
        direct_argv, dry_run = _strip_direct_dry_run(expanded)
        stage = _stage_token(direct_argv[0]) if direct_argv else None
        if not stage:
            raise SystemExit("Direct YAML config requires stage: stage1, stage2, or pipeline.")
        if stage == "pipeline":
            try:
                config_idx = direct_argv.index("--pipeline-config")
                config_path = direct_argv[config_idx + 1]
            except (ValueError, IndexError) as exc:
                raise SystemExit("Pipeline config dispatch requires --pipeline-config PATH.") from exc
            commands, pipeline_config, config_dry_run = _pipeline_commands(config_path)
            dry_run = dry_run or config_dry_run
            for label, cmd in commands:
                printable = cmd
                if label == "stage 1 adversarial" and not _pipeline_has_stage1_init(cmd):
                    printable = [*cmd, "init_ckpt_path=<latest stage-1 checkpoint>"]
                print(f"Launching {label}:", shlex.join(printable), flush=True)
            if dry_run:
                return 0
            for label, cmd in commands:
                if label == "stage 1 adversarial" and not _pipeline_has_stage1_init(cmd):
                    checkpoint = _pipeline_latest_stage1_checkpoint(pipeline_config)
                    cmd = [*cmd, f"init_ckpt_path={checkpoint}"]
                    print(f"Initializing stage 1 adversarial from: {checkpoint}", flush=True)
                subprocess.run(cmd, check=True)
            return 0
        cmd = [sys.executable, str(Path(__file__).resolve()), *direct_argv]
        if dry_run:
            print("Launching:", shlex.join(cmd), flush=True)
            return 0
        return _dispatch_stage(stage, direct_argv[1:])

    stage = _stage_token(expanded[0]) if expanded else None
    if stage == "pipeline":
        raise SystemExit("Pipeline runs are supported through YAML configs: python train.py --config configs/exp1.yaml")
    if stage and not _looks_like_facade_args(raw_argv[1:]):
        return _dispatch_stage(stage, expanded[1:])

    parser = _build_parser()
    normalized = _inject_stage_option(expanded)
    args, unknown = parser.parse_known_args(normalized)
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
