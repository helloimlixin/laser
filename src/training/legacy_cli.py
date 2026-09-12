"""Compatibility for old training flags, raw Hydra commands, and archived pipelines."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.training.paths import TRAIN_SCRIPT, CONFIG_DIR
from src.training.common import _dataset_key


IMAGE_DATASETS = {
    "cc3m",
    "celeba",
    "celebahq",
    "cifar10",
    "coco",
    "ffhq",
    "imagenet",
    "imagenette2",
    "lsun_bedroom",
    "lsun_church",
    "lsun_cat",
    "stl10",
}
POST_FIT_RFID_DATASETS = {
    "celeba",
    "celebahq",
    "cifar10",
    "ffhq",
    "imagenet",
    "imagenette2",
    "lsun_bedroom",
    "lsun_church",
    "lsun_cat",
}
AUDIO_DATASETS = {"vctk", "maestro"}
_DISPATCH_STAGE_ENV = "LASER_TRAIN_DISPATCH_STAGE"
_DISPATCH_ARGV_ENV = "LASER_TRAIN_DISPATCH_ARGV"


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
        ("stage 1", [sys.executable, str(TRAIN_SCRIPT), *stage1_argv]),
    ]
    if isinstance(config.get("stage1_adv"), dict):
        stage1_adv_argv = _nested_stage_argv(config, "stage1_adv", "stage1")
        commands.append(
            (
                "stage 1 adversarial",
                [sys.executable, str(TRAIN_SCRIPT), *stage1_adv_argv],
            )
        )
    commands.append(("stage 2", [sys.executable, str(TRAIN_SCRIPT), *stage2_argv]))
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
    name = _dataset_key(dataset)
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
    if not args.stage:
        raise SystemExit("Specify a stage with stage1/stage2, --stage, or a YAML config containing stage.")
    if not args.dataset:
        raise SystemExit("--dataset is required for launcher-style runs. Direct YAML configs can use data=... instead.")
    if not args.modality:
        raise SystemExit("--modality is required for launcher-style runs. Direct YAML configs can use data=... instead.")
    dataset = args.dataset.strip().lower()
    if args.conditioning == "class" and args.modality != "image":
        raise SystemExit("--conditioning class is only valid with --modality image.")
    if args.conditioning == "text" and args.modality != "audio" and dataset not in {"cc3m"}:
        raise SystemExit("--conditioning text is only valid with audio datasets or CC3M image runs.")
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
    script = TRAIN_SCRIPT
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
        if _dataset_key(args.dataset) in POST_FIT_RFID_DATASETS:
            overrides.extend([
                "train.compute_rfid_after_fit=true",
                "train.rfid_split=val",
                "train.rfid_batch_size=100",
                "train.rfid_num_workers=8",
                "train.rfid_max_samples=0",
                "train.rfid_device=auto",
                "train.rfid_feature=2048",
            ])
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
    script = TRAIN_SCRIPT
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
        dataset_name = args.dataset.strip().lower()
        overrides.extend(
            [
                "ar.text_conditional=true",
                "ar.text_prefix_length=16",
                "train_ar.sample_text_prompts=[\"The quick brown fox jumps over the lazy dog.\",\"A calm voice reads this sentence clearly.\"]",
            ]
        )
        if dataset_name == "cc3m":
            overrides.extend(
                [
                    "ar.text_conditioning_mode=rq_prefix",
                    "ar.text_prefix_length=32",
                    "ar.text_loss_weight=0.1",
                    "ar.image_loss_weight=0.9",
                    "ar.n_global_spatial_tokens=0",
                ]
            )
    overrides.extend(extra)
    return [sys.executable, str(script), "stage2", *overrides]


def build_command(argv: list[str] | None = None) -> list[str]:
    parser = _build_parser()
    expanded, direct_config = _expand_config_argv(list(argv) if argv is not None else sys.argv[1:])
    if direct_config:
        return [sys.executable, str(TRAIN_SCRIPT), *_strip_direct_dry_run(expanded)[0]]
    normalized = _inject_stage_option(expanded)
    args, unknown = parser.parse_known_args(normalized)
    extra = _normalize_unknown(unknown)
    _validate(args)
    resources = _resolve_resources(args.num_gpus, args.num_nodes, args.devices_per_node)
    if args.stage == "1":
        return _stage1_command(args, resources, extra)
    return _stage2_command(args, resources, extra)

def _dispatch_stage(stage: str, stage_argv: list[str]) -> int:
    os.environ[_DISPATCH_STAGE_ENV] = stage
    os.environ[_DISPATCH_ARGV_ENV] = json.dumps(stage_argv)
    sys.argv = [sys.argv[0], *stage_argv]
    import hydra
    from importlib import import_module
    module = import_module(f"src.training.stage{stage}")
    config_name = "config" if stage == "1" else "config_ar"
    hydra.main(config_path=str(CONFIG_DIR), config_name=config_name, version_base="1.2")(module.run)()
    return 0


def _dispatch_from_env(expanded: list[str]) -> tuple[str, list[str]] | None:
    stage = _stage_token(os.environ.get(_DISPATCH_STAGE_ENV, ""))
    if not stage:
        return None
    if expanded:
        return stage, expanded
    raw_argv = os.environ.get(_DISPATCH_ARGV_ENV, "")
    if not raw_argv:
        return None
    try:
        stage_argv = json.loads(raw_argv)
    except json.JSONDecodeError:
        return None
    if not isinstance(stage_argv, list) or not all(isinstance(arg, str) for arg in stage_argv):
        return None
    return stage, stage_argv


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
        cmd = [sys.executable, str(TRAIN_SCRIPT), *direct_argv]
        if dry_run:
            print("Launching:", shlex.join(cmd), flush=True)
            return 0
        return _dispatch_stage(stage, direct_argv[1:])

    stage = _stage_token(expanded[0]) if expanded else None
    if not stage and (env_dispatch := _dispatch_from_env(expanded)):
        env_stage, env_argv = env_dispatch
        return _dispatch_stage(env_stage, env_argv)
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
    os.environ.setdefault("LASER_DISABLE_WANDB_MEDIA", "0")
    os.execv(cmd[0], cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
