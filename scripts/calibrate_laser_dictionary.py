#!/usr/bin/env python3
"""Catch a LASER dictionary up to a frozen Stage-1 encoder.

This deliberately is not a training mode.  It reads ImageNet training batches,
encodes them without gradients, applies only the fixed-code residual dictionary
update, and writes a child checkpoint in which only ``quantizer.*`` state is
replaced.  Encoder, decoder, GAN, optimizer, scheduler, RNG, and loader state
remain byte-for-byte sourced from the input checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from rqvae.img_datasets import create_dataset
from rqvae.models import create_model
from rqvae.utils.config import augment_defaults, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-checkpoint", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=125)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=80375)
    parser.add_argument("--dataset-root", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initialize_distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("dictionary calibration requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    return rank, world_size, local_rank, device


def seed_rank(seed: int, rank: int) -> torch.Generator:
    rank_seed = int(seed) + int(rank)
    random.seed(rank_seed)
    np.random.seed(rank_seed % (2**32))
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed(rank_seed)
    generator = torch.Generator()
    generator.manual_seed(rank_seed)
    return generator


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu().item())


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")

    config_path = args.config.resolve()
    input_path = args.input_checkpoint.resolve()
    output_path = args.output_checkpoint.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path == input_path:
        raise ValueError("output checkpoint must differ from input checkpoint")
    if output_path.exists():
        raise FileExistsError(output_path)

    rank, world_size, local_rank, device = initialize_distributed()
    loader_generator = seed_rank(args.seed, rank)
    torch.backends.cudnn.benchmark = True

    config = augment_defaults(load_config(config_path))
    if args.dataset_root is not None:
        config.dataset.root = args.dataset_root.resolve().as_posix()
    bottleneck_type = str(config.arch.hparams.get("bottleneck_type", "rq")).lower()
    update_mode = str(
        config.arch.hparams.get("dictionary_update_mode", "gradient")
    ).lower()
    if bottleneck_type != "laser" or update_mode != "alternating_residual":
        raise ValueError(
            "calibration requires bottleneck_type=laser and "
            "dictionary_update_mode=alternating_residual"
        )

    dataset_train, _ = create_dataset(config, is_eval=False)
    sampler = DistributedSampler(
        dataset_train,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=int(args.seed),
        drop_last=True,
    )
    sampler.set_epoch(0)
    loader = DataLoader(
        dataset_train,
        batch_size=int(args.batch_size),
        sampler=sampler,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
        persistent_workers=bool(args.num_workers),
        generator=loader_generator,
    )
    if args.steps > len(loader):
        raise ValueError(f"requested {args.steps} steps, but loader has {len(loader)}")

    checkpoint = torch.load(
        input_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("state_dict"), dict
    ):
        raise RuntimeError("input is not a Stage-1 training checkpoint")

    model, model_ema = create_model(config.arch, ema=False)
    if model_ema is not None:
        raise RuntimeError("dictionary calibration does not support EMA models")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.requires_grad_(False)
    model.to(device)
    model.eval()
    quantizer = model.quantizer
    quantizer.train()
    if str(quantizer.dictionary_update_mode).lower() != "alternating_residual":
        raise RuntimeError("loaded model does not expose the residual dictionary update")

    original_dictionary = checkpoint["state_dict"]["quantizer.dictionary"].float()
    initial_dictionary_step = int(
        checkpoint["state_dict"]["quantizer._dictionary_update_step"].item()
    )
    global_step = int(checkpoint.get("global_step", -1))
    if global_step < 0:
        raise RuntimeError("input checkpoint has no valid global step")
    raw_prior_offset = checkpoint.get("quantizer_step_offset")
    if raw_prior_offset is None:
        # Backward compatibility for the first calibrated checkpoints, which
        # recorded the dictionary counters but predated the explicit offset.
        prior_quantizer_step_offset = initial_dictionary_step - global_step
        if prior_quantizer_step_offset:
            prior_calibration = checkpoint.get("dictionary_calibration")
            if not isinstance(prior_calibration, dict) or (
                prior_calibration.get("updated_state_scope") != "quantizer.* only"
            ):
                raise RuntimeError(
                    "cannot infer a quantizer step offset without calibration provenance"
                )
    else:
        prior_quantizer_step_offset = int(raw_prior_offset)
    if prior_quantizer_step_offset < 0:
        raise RuntimeError("input checkpoint has a negative quantizer step offset")
    expected_initial_quantizer_step = global_step + prior_quantizer_step_offset
    initial_revival_step = int(
        checkpoint["state_dict"]["quantizer._revival_step"].item()
    )
    if (
        initial_dictionary_step != expected_initial_quantizer_step
        or initial_revival_step != expected_initial_quantizer_step
    ):
        raise RuntimeError(
            "input quantizer counters do not match global step plus calibration offset"
        )
    source_sha256 = sha256_file(input_path) if rank == 0 else None
    if world_size > 1:
        payload = [source_sha256]
        dist.broadcast_object_list(payload, src=0)
        source_sha256 = payload[0]

    if rank == 0:
        print(
            json.dumps(
                {
                    "event": "dictionary_calibration_start",
                    "input": input_path.as_posix(),
                    "source_sha256": source_sha256,
                    "global_step_preserved": int(checkpoint.get("global_step", -1)),
                    "quantizer_step_offset_before": prior_quantizer_step_offset,
                    "calibration_steps": int(args.steps),
                    "global_batch_size": int(args.batch_size) * world_size,
                    "world_size": world_size,
                    "device": torch.cuda.get_device_name(local_rank),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    objective_sum = 0.0
    relative_improvement_sum = 0.0
    updated_atom_sum = 0
    revived_atom_sum = 0
    start_time = time.monotonic()
    iterator = iter(loader)
    with torch.inference_mode():
        for step in range(1, int(args.steps) + 1):
            batch = next(iterator)
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = images.to(device, non_blocking=True)
            encoded = model.encode(images)
            with torch.autocast(device_type="cuda", enabled=False):
                quantizer(encoded.float())
            objective_sum += scalar(quantizer._last_final_dictionary_loss)
            updated_atoms = int(quantizer.alternating_dictionary_update_after_step_())
            quantizer.normalize_dictionary_()
            revived_atoms = int(quantizer.revive_dead_atoms_after_step_(None))
            updated_atom_sum += updated_atoms
            revived_atom_sum += revived_atoms
            relative_improvement_sum += scalar(
                quantizer._last_dictionary_update_relative_improvement
            )
            if rank == 0 and (step == 1 or step % 25 == 0 or step == args.steps):
                print(
                    json.dumps(
                        {
                            "event": "dictionary_calibration_progress",
                            "step": step,
                            "steps": int(args.steps),
                            "fixed_code_mse": scalar(
                                quantizer._last_final_dictionary_loss
                            ),
                            "updated_atoms": updated_atoms,
                            "accepted_relaxation": scalar(
                                quantizer._last_dictionary_update_relaxation
                            ),
                            "fixed_code_relative_improvement": scalar(
                                quantizer._last_dictionary_update_relative_improvement
                            ),
                            "revived_atoms": revived_atoms,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    torch.cuda.synchronize(device)
    elapsed_seconds = time.monotonic() - start_time
    if world_size > 1:
        dist.barrier()

    calibrated_state = model.state_dict()
    quantizer_keys = {
        key for key in checkpoint["state_dict"] if key.startswith("quantizer.")
    }
    if not quantizer_keys or quantizer_keys.difference(calibrated_state):
        raise RuntimeError("checkpoint and calibrated quantizer state do not match")

    if rank == 0:
        calibrated_dictionary = calibrated_state["quantizer.dictionary"].detach().cpu().float()
        dictionary_delta = calibrated_dictionary - original_dictionary
        original_checkpoint_id = checkpoint.get("checkpoint_id")
        checkpoint["parent_checkpoint_id"] = original_checkpoint_id
        checkpoint["checkpoint_id"] = uuid.uuid4().hex
        checkpoint["lineage_exact"] = False
        checkpoint["lineage_origin"] = (
            str(checkpoint.get("lineage_origin", "unknown"))
            + f"|dictionary_only_calibration_{int(args.steps)}_steps_seed_{int(args.seed)}"
        )
        for key in sorted(quantizer_keys):
            checkpoint["state_dict"][key] = calibrated_state[key].detach().cpu()
        checkpoint["quantizer_step_offset"] = (
            prior_quantizer_step_offset + int(args.steps)
        )
        checkpoint["dictionary_calibration"] = {
            "source_checkpoint": input_path.as_posix(),
            "source_sha256": source_sha256,
            "source_checkpoint_id": original_checkpoint_id,
            "steps": int(args.steps),
            "batch_size_per_rank": int(args.batch_size),
            "global_batch_size": int(args.batch_size) * world_size,
            "world_size": world_size,
            "seed": int(args.seed),
            "sampler_epoch": 0,
            "dataset": "ImageNet training split",
            "updated_state_scope": "quantizer.* only",
            "preserved_state_scope": (
                "encoder, decoder, discriminator, optimizers, schedulers, "
                "global step, RNG, and loader cursor"
            ),
            "dictionary_update_step_before": initial_dictionary_step,
            "dictionary_update_step_after": int(
                calibrated_state["quantizer._dictionary_update_step"].item()
            ),
            "quantizer_step_offset_before": prior_quantizer_step_offset,
            "quantizer_step_offset_after": checkpoint["quantizer_step_offset"],
            "mean_fixed_code_mse_before_update": objective_sum / int(args.steps),
            "mean_fixed_code_relative_improvement": (
                relative_improvement_sum / int(args.steps)
            ),
            "mean_updated_atoms": updated_atom_sum / int(args.steps),
            "total_revived_atoms": revived_atom_sum,
            "dictionary_delta_l2": float(dictionary_delta.norm().item()),
            "dictionary_delta_rms": float(dictionary_delta.square().mean().sqrt().item()),
            "elapsed_seconds": elapsed_seconds,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(
            f".{output_path.name}.{uuid.uuid4().hex}.tmp"
        )
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, output_path)
        output_sha256 = sha256_file(output_path)
        print(
            json.dumps(
                {
                    "event": "dictionary_calibration_complete",
                    "output": output_path.as_posix(),
                    "output_sha256": output_sha256,
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "global_step_preserved": int(checkpoint.get("global_step", -1)),
                    **checkpoint["dictionary_calibration"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
