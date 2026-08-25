#!/usr/bin/env python3
"""Create an audited Stage-1 resume checkpoint with a short cosine LR tail.

By default the model, discriminator, Adam moments, RNG state, data-loader
cursor, and partial-epoch accumulator are preserved.  The optional
``--reset-main-moments`` trust-region mode clears only the generator Adam
first/second moments while preserving every optimizer step counter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from rqvae.utils.checkpoint import (
    build_resume_signature,
    validate_resume_checkpoint,
)
from rqvae.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def rebase_optimizer(state: dict[str, Any], learning_rate: float) -> list[float]:
    groups = state.get("param_groups")
    if not isinstance(groups, list) or not groups:
        raise RuntimeError("optimizer has no parameter groups")
    old_rates = []
    for group in groups:
        if not isinstance(group, dict) or "lr" not in group:
            raise RuntimeError("optimizer parameter group has no learning rate")
        old_rates.append(float(group["lr"]))
        group["lr"] = learning_rate
        group["initial_lr"] = learning_rate
    return old_rates


def reset_adam_moments(state: dict[str, Any]) -> dict[str, int]:
    """Clear Adam momentum tensors without changing parameter step counters."""
    optimizer_state = state.get("state")
    if not isinstance(optimizer_state, dict) or not optimizer_state:
        raise RuntimeError("optimizer has no parameter state to reset")

    tensor_counts = {
        "parameter_states": 0,
        "exp_avg": 0,
        "exp_avg_sq": 0,
        "max_exp_avg_sq": 0,
    }
    for parameter_state in optimizer_state.values():
        if not isinstance(parameter_state, dict):
            raise RuntimeError("optimizer parameter state must be a mapping")
        tensor_counts["parameter_states"] += 1
        reset_any = False
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = parameter_state.get(key)
            if value is None:
                continue
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(f"optimizer {key} must be a tensor")
            value.zero_()
            tensor_counts[key] += 1
            reset_any = True
        if not reset_any:
            raise RuntimeError("optimizer parameter state has no Adam moments")
        if "step" not in parameter_state:
            raise RuntimeError("optimizer parameter state has no step counter")
    return tensor_counts


def rebase_scheduler(
    state: dict[str, Any],
    *,
    learning_rate: float,
    minimum_learning_rate: float,
    decay_steps: int,
    global_step: int,
) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise RuntimeError("scheduler state must be a mapping")
    warmup = state.get("warmup")
    after = state.get("after")
    if not isinstance(warmup, dict) or not isinstance(after, dict):
        raise RuntimeError("expected the RQ-VAE warmup-plus-cosine scheduler")
    if int(warmup.get("last_epoch", -1)) != global_step:
        raise RuntimeError("warmup scheduler cursor does not match global step")
    base_lrs = after.get("base_lrs")
    if not isinstance(base_lrs, list) or not base_lrs:
        raise RuntimeError("cosine scheduler has no base learning rates")
    group_count = len(base_lrs)
    old = {
        "base_lrs": [float(value) for value in base_lrs],
        "eta_min": float(after.get("eta_min", 0.0)),
        "T_max": float(after.get("T_max", -1.0)),
        "last_epoch": int(after.get("last_epoch", -1)),
    }

    warmup["base_lrs"] = [learning_rate] * group_count
    warmup["_last_lr"] = [learning_rate] * group_count
    after["base_lrs"] = [learning_rate] * group_count
    after["eta_min"] = minimum_learning_rate
    after["T_max"] = int(decay_steps)
    # The outer warmup cursor remains at global_step for checkpoint validation.
    # Reset only the nested cosine phase so the next update is cosine step one.
    after["last_epoch"] = 0
    after["_step_count"] = 1
    after["_last_lr"] = [learning_rate] * group_count
    return old


def semantic_config(config: Any) -> dict[str, Any]:
    return {
        name: OmegaConf.to_container(config[name], resolve=True)
        for name in ("arch", "dataset", "optimizer", "experiment", "gan")
    }


def validate_bypass_only_rebase(
    *,
    baseline_config: Any,
    target_config: Any,
    saved_signature: dict[str, Any],
    expected_signature: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed unless the only semantic change is the audited block mode."""
    baseline_signature = build_resume_signature(
        baseline_config,
        project_root=ROOT,
    )
    if baseline_signature.get("config_sha256") != saved_signature.get(
        "config_sha256"
    ):
        raise RuntimeError(
            "baseline config does not match the checkpoint config signature"
        )

    saved_sources = saved_signature.get("source_sha256")
    expected_sources = expected_signature.get("source_sha256")
    if not isinstance(saved_sources, dict) or not isinstance(expected_sources, dict):
        raise RuntimeError("resume signatures need complete source manifests")
    if set(saved_sources) != set(expected_sources):
        raise RuntimeError("resume source manifest keys changed")
    changed_sources = sorted(
        name
        for name in saved_sources
        if saved_sources[name] != expected_sources[name]
    )
    allowed_sources = ["rqvae/trainers/trainer_rqvae.py"]
    if changed_sources != allowed_sources:
        raise RuntimeError(
            "bypass-only rebase has unexpected source changes: "
            + ", ".join(changed_sources)
        )

    baseline_semantic = semantic_config(baseline_config)
    target_semantic = semantic_config(target_config)
    baseline_experiment = baseline_semantic["experiment"]
    target_experiment = target_semantic["experiment"]
    if baseline_experiment.get("train_bypass_alternating") is not True:
        raise RuntimeError("baseline is not the paired alternating objective")
    required_target = {
        "train_bypass_alternating": False,
        "train_bypass_only": True,
        "freeze_discriminator_stats": True,
    }
    for key, expected in required_target.items():
        if target_experiment.get(key) is not expected:
            raise RuntimeError(f"target experiment must set {key}={expected}")
    allowed_config_changes = sorted(required_target)
    for key in allowed_config_changes:
        baseline_experiment.pop(key, None)
        target_experiment.pop(key, None)
    if baseline_semantic != target_semantic:
        raise RuntimeError(
            "bypass-only rebase changes config fields outside the audited mode flags"
        )
    return {
        "baseline_config_sha256": saved_signature["config_sha256"],
        "target_config_sha256": expected_signature["config_sha256"],
        "allowed_config_changes": allowed_config_changes,
        "changed_sources": changed_sources,
        "updated_state_scope": "resume compatibility signature and training mode",
    }


def validate_compatible_trainer_source_rebase(
    *,
    config: Any,
    saved_signature: dict[str, Any],
    expected_signature: dict[str, Any],
    baseline_trainer_source: Path,
    expected_current_trainer_sha256: str,
) -> dict[str, Any]:
    """Adopt a trainer revision whose new modes are dormant for this config."""
    if expected_signature.get("config_sha256") != saved_signature.get(
        "config_sha256"
    ):
        raise RuntimeError(
            "compatible trainer-source rebase cannot change the semantic config"
        )

    saved_sources = saved_signature.get("source_sha256")
    expected_sources = expected_signature.get("source_sha256")
    if not isinstance(saved_sources, dict) or not isinstance(expected_sources, dict):
        raise RuntimeError("resume signatures need complete source manifests")
    if set(saved_sources) != set(expected_sources):
        raise RuntimeError("resume source manifest keys changed")
    changed_sources = sorted(
        name
        for name in saved_sources
        if saved_sources[name] != expected_sources[name]
    )
    trainer_key = "rqvae/trainers/trainer_rqvae.py"
    if changed_sources != [trainer_key]:
        raise RuntimeError(
            "compatible trainer-source rebase has unexpected source changes: "
            + ", ".join(changed_sources)
        )

    baseline_sha256 = sha256_file(baseline_trainer_source)
    if baseline_sha256 != saved_sources[trainer_key]:
        raise RuntimeError(
            "baseline trainer snapshot does not match the checkpoint signature"
        )
    expected_current_sha256 = str(expected_current_trainer_sha256).lower()
    if expected_sources[trainer_key] != expected_current_sha256:
        raise RuntimeError(
            "current trainer source does not match the explicitly audited SHA-256"
        )

    experiment = semantic_config(config)["experiment"]
    if experiment.get("train_bypass_alternating") is not True:
        raise RuntimeError("compatible trainer-source rebase requires alternating mode")
    if bool(experiment.get("train_bypass_only", False)):
        raise RuntimeError("bypass-only mode must be dormant for this source rebase")
    if bool(experiment.get("freeze_discriminator_stats", False)):
        raise RuntimeError("fixed-critic mode must be dormant for this source rebase")

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_trainer_source": baseline_trainer_source.as_posix(),
        "baseline_trainer_sha256": baseline_sha256,
        "current_trainer_sha256": expected_current_sha256,
        "changed_sources": changed_sources,
        "dormant_mode_flags": {
            "train_bypass_only": False,
            "freeze_discriminator_stats": False,
        },
        "updated_state_scope": "resume compatibility signature only",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-checkpoint", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path)
    parser.add_argument("--baseline-trainer-source", type=Path)
    parser.add_argument("--expected-current-trainer-sha256")
    parser.add_argument("--main-lr", type=float, required=True)
    parser.add_argument("--disc-lr", type=float, required=True)
    parser.add_argument("--main-min-lr", type=float, default=0.0)
    parser.add_argument("--disc-min-lr", type=float, default=0.0)
    parser.add_argument("--decay-steps", type=int, required=True)
    parser.add_argument("--expected-input-sha256")
    parser.add_argument("--source-rfid", type=float, required=True)
    parser.add_argument("--source-bypass-rfid", type=float, required=True)
    parser.add_argument("--cycle", type=int, default=3)
    parser.add_argument(
        "--reset-main-moments",
        action="store_true",
        help="zero generator Adam moments while preserving optimizer steps",
    )
    parser.add_argument(
        "--allow-bypass-only-rebase",
        action="store_true",
        help=(
            "audit and adopt the bypass-only/fixed-critic config plus the "
            "corresponding trainer source change"
        ),
    )
    parser.add_argument(
        "--allow-compatible-trainer-source-rebase",
        action="store_true",
        help=(
            "adopt an explicitly audited trainer-only source revision while "
            "its newly added bypass-only and fixed-critic modes remain dormant"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input_checkpoint.expanduser().resolve()
    output_path = args.output_checkpoint.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    baseline_config_path = (
        args.baseline_config.expanduser().resolve()
        if args.baseline_config is not None
        else None
    )
    baseline_trainer_source = (
        args.baseline_trainer_source.expanduser().resolve()
        if args.baseline_trainer_source is not None
        else None
    )
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if baseline_config_path is not None and not baseline_config_path.is_file():
        raise FileNotFoundError(baseline_config_path)
    if baseline_trainer_source is not None and not baseline_trainer_source.is_file():
        raise FileNotFoundError(baseline_trainer_source)
    if args.allow_bypass_only_rebase != (baseline_config_path is not None):
        raise ValueError(
            "--allow-bypass-only-rebase and --baseline-config must be used together"
        )
    compatible_source_inputs = (
        baseline_trainer_source is not None
        and args.expected_current_trainer_sha256 is not None
    )
    if args.allow_compatible_trainer_source_rebase != compatible_source_inputs:
        raise ValueError(
            "--allow-compatible-trainer-source-rebase requires both "
            "--baseline-trainer-source and --expected-current-trainer-sha256"
        )
    if args.allow_bypass_only_rebase and args.allow_compatible_trainer_source_rebase:
        raise ValueError("resume-signature rebase modes are mutually exclusive")
    if output_path == input_path:
        raise ValueError("output checkpoint must differ from input checkpoint")
    if output_path.exists():
        raise FileExistsError(output_path)
    if int(args.decay_steps) <= 0:
        raise ValueError("--decay-steps must be positive")
    if int(args.cycle) <= 0:
        raise ValueError("--cycle must be positive")

    main_lr = finite_nonnegative(args.main_lr, "--main-lr")
    disc_lr = finite_nonnegative(args.disc_lr, "--disc-lr")
    main_min_lr = finite_nonnegative(args.main_min_lr, "--main-min-lr")
    disc_min_lr = finite_nonnegative(args.disc_min_lr, "--disc-min-lr")
    if main_min_lr > main_lr or disc_min_lr > disc_lr:
        raise ValueError("minimum learning rates cannot exceed starting rates")

    input_sha256 = sha256_file(input_path)
    if (
        args.expected_input_sha256 is not None
        and input_sha256 != str(args.expected_input_sha256).lower()
    ):
        raise RuntimeError(
            "input checkpoint SHA-256 mismatch: "
            f"expected={args.expected_input_sha256}, actual={input_sha256}"
        )

    checkpoint = torch.load(
        input_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    steps_per_epoch = int(checkpoint.get("steps_per_epoch", -1))
    world_size = int(checkpoint.get("checkpoint_world_size", -1))
    if steps_per_epoch <= 0 or world_size <= 0:
        raise RuntimeError("checkpoint has invalid data-loader metadata")
    validate_resume_checkpoint(
        checkpoint,
        steps_per_epoch=steps_per_epoch,
        world_size=world_size,
        expected_resume_signature=checkpoint.get("resume_signature"),
    )

    # A resume uses the canonical, already-augmented config saved beside the
    # checkpoint.  Re-applying defaults here would change its semantic hash.
    config = load_config(config_path)
    expected_signature = build_resume_signature(config, project_root=ROOT)
    saved_signature = checkpoint.get("resume_signature", {})
    objective_rebase = None
    compatible_trainer_source_rebase = None
    if expected_signature.get("sha256") != saved_signature.get("sha256"):
        if args.allow_bypass_only_rebase:
            baseline_config = load_config(baseline_config_path)
            objective_rebase = validate_bypass_only_rebase(
                baseline_config=baseline_config,
                target_config=config,
                saved_signature=saved_signature,
                expected_signature=expected_signature,
            )
            objective_rebase["created_utc"] = datetime.now(timezone.utc).isoformat()
            objective_rebase["baseline_config"] = baseline_config_path.as_posix()
            objective_rebase["target_config"] = config_path.as_posix()
        elif args.allow_compatible_trainer_source_rebase:
            compatible_trainer_source_rebase = (
                validate_compatible_trainer_source_rebase(
                    config=config,
                    saved_signature=saved_signature,
                    expected_signature=expected_signature,
                    baseline_trainer_source=baseline_trainer_source,
                    expected_current_trainer_sha256=(
                        args.expected_current_trainer_sha256
                    ),
                )
            )
        else:
            raise RuntimeError(
                "the supplied canonical config/current source does not match the "
                "checkpoint resume signature"
            )
        checkpoint["resume_signature"] = expected_signature
    elif (
        args.allow_bypass_only_rebase
        or args.allow_compatible_trainer_source_rebase
    ):
        raise RuntimeError("requested resume-signature rebase made no change")

    global_step = int(checkpoint["global_step"])
    source_checkpoint_id = str(checkpoint["checkpoint_id"])
    old_main_rates = rebase_optimizer(checkpoint["optimizer"], main_lr)
    reset_main_moments = None
    if args.reset_main_moments:
        reset_main_moments = reset_adam_moments(checkpoint["optimizer"])
    old_main_schedule = rebase_scheduler(
        checkpoint["scheduler"],
        learning_rate=main_lr,
        minimum_learning_rate=main_min_lr,
        decay_steps=int(args.decay_steps),
        global_step=global_step,
    )
    old_disc_rates = rebase_optimizer(
        checkpoint["discriminator_optimizer"], disc_lr
    )
    old_disc_schedule = rebase_scheduler(
        checkpoint["discriminator_scheduler"],
        learning_rate=disc_lr,
        minimum_learning_rate=disc_min_lr,
        decay_steps=int(args.decay_steps),
        global_step=global_step,
    )

    checkpoint["parent_checkpoint_id"] = source_checkpoint_id
    checkpoint["checkpoint_id"] = uuid.uuid4().hex
    checkpoint["lineage_exact"] = False
    checkpoint["lineage_origin"] = (
        str(checkpoint.get("lineage_origin", "unknown"))
        + f"|cosine_lr_trust_region_rebase_step_{global_step}_"
        + f"main_{main_lr:.12g}_disc_{disc_lr:.12g}_"
        + f"decay_{int(args.decay_steps)}"
        + ("_reset_main_adam_moments" if args.reset_main_moments else "")
        + ("_bypass_only_fixed_critic" if objective_rebase else "")
        + (
            "_compatible_trainer_source"
            if compatible_trainer_source_rebase
            else ""
        )
    )
    checkpoint["learning_rate_rebase"] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": input_path.as_posix(),
        "source_sha256": input_sha256,
        "source_checkpoint_id": source_checkpoint_id,
        "global_step_preserved": global_step,
        "main_lr_before": old_main_rates,
        "main_lr_after": main_lr,
        "main_min_lr_after": main_min_lr,
        "main_scheduler_before": old_main_schedule,
        "discriminator_lr_before": old_disc_rates,
        "discriminator_lr_after": disc_lr,
        "discriminator_min_lr_after": disc_min_lr,
        "discriminator_scheduler_before": old_disc_schedule,
        "cosine_decay_steps": int(args.decay_steps),
        "main_adam_moments_reset": bool(args.reset_main_moments),
        "main_adam_moment_reset_counts": reset_main_moments,
        "preserved_state_scope": (
            "model, discriminator, optimizer steps, "
            + (
                "discriminator Adam moments, "
                if args.reset_main_moments
                else "all Adam moments, "
            )
            + "RNG, loader cursor, partial-epoch accumulator, "
            "quantizer calibration, and global step"
        ),
        "changed_state_scope": (
            "optimizer param-group LR fields, nested cosine scheduler tail, "
            + ("generator Adam first/second moments, " if args.reset_main_moments else "")
            + "and checkpoint lineage/provenance"
        ),
    }
    if objective_rebase is not None:
        checkpoint["training_objective_rebase"] = objective_rebase
    if compatible_trainer_source_rebase is not None:
        checkpoint["compatible_trainer_source_rebase"] = (
            compatible_trainer_source_rebase
        )
    checkpoint["continuation_cycle"] = {
        "cycle": int(args.cycle),
        "source_checkpoint": input_path.as_posix(),
        "source_checkpoint_id": source_checkpoint_id,
        "source_rfid": float(args.source_rfid),
        "source_bypass_rfid": float(args.source_bypass_rfid),
        "planned_backbone_updates": int(args.decay_steps),
        "training_pattern": (
            (
                "bypass reconstruction on every global step; fixed critic; "
                "dictionary update on every batch"
            )
            if objective_rebase
            else (
                "LASER on odd global steps; bypass on even global steps; "
                "dictionary update on every batch"
            )
        ),
        "gate": (
            "reject immediately unless bypass rFID improves; run a frozen-"
            "backbone dictionary catch-up only when bypass improves but full "
            "rFID does not"
        ),
    }

    # The validator must still accept the exact data/optimizer cursor after the
    # deliberate schedule rebase.
    validate_resume_checkpoint(
        checkpoint,
        steps_per_epoch=steps_per_epoch,
        world_size=world_size,
        expected_resume_signature=checkpoint["resume_signature"],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_config = output_path.parent / "config.yaml"
    if canonical_config.exists():
        if canonical_config.read_bytes() != config_path.read_bytes():
            raise RuntimeError(f"existing canonical config differs: {canonical_config}")
    else:
        shutil.copy2(config_path, canonical_config)

    temporary_path = output_path.with_name(
        f".{output_path.name}.{uuid.uuid4().hex}.tmp"
    )
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, output_path)
    output_sha256 = sha256_file(output_path)
    result = {
        "output_checkpoint": output_path.as_posix(),
        "output_sha256": output_sha256,
        "checkpoint_id": checkpoint["checkpoint_id"],
        "parent_checkpoint_id": source_checkpoint_id,
        "global_step": global_step,
        "main_lr": main_lr,
        "main_min_lr": main_min_lr,
        "discriminator_lr": disc_lr,
        "discriminator_min_lr": disc_min_lr,
        "cosine_decay_steps": int(args.decay_steps),
        "main_adam_moments_reset": bool(args.reset_main_moments),
        "main_adam_moment_reset_counts": reset_main_moments,
        "training_objective_rebase": objective_rebase,
        "compatible_trainer_source_rebase": compatible_trainer_source_rebase,
        "source_rfid": float(args.source_rfid),
        "source_bypass_rfid": float(args.source_bypass_rfid),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
