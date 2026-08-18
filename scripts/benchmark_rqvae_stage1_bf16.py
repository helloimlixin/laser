#!/usr/bin/env python3
"""Two-rank FP32/BF16 regression benchmark for the ImageNet Stage-1 trainer.

This loads a real checkpoint but never writes training state or contacts W&B.
It is intended to run through ``torchrun`` on otherwise idle GPUs.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder


ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT / "third_party" / "rq-vae-transformer"
if str(THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rqvae.models import create_model
from rqvae.img_datasets.transforms import create_transforms
from rqvae.optimizer import create_optimizer, create_scheduler
from rqvae.trainers.trainer_rqvae import Trainer
from rqvae.utils import dist as dist_utils
from rqvae.utils.dist import DistEnv


class RepeatedImageDataset(Dataset):
    def __init__(self, length):
        self.length = int(length)
        generator = torch.Generator().manual_seed(20260817)
        self.image = torch.rand(3, 256, 256, generator=generator).mul_(2).sub_(1)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.image, 0


class NullWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def add_image(self, *args, **kwargs):
        return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--precision", choices=("float32", "bfloat16"), required=True)
    parser.add_argument(
        "--omp-precision",
        choices=("float32", "bfloat16"),
        default="float32",
        help=(
            "OMP matrix-product precision. BF16 still uses FP32 for the small "
            "Cholesky solves because CUDA has no BF16 solve kernels."
        ),
    )
    parser.add_argument("--local-batch-size", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--imagenet-root", type=Path, default=None)
    parser.add_argument("--snapshot-only", action="store_true")
    parser.add_argument("--disable-activation-checkpointing", action="store_true")
    return parser.parse_args()


def make_loader(*, steps, local_batch_size, world_size, rank):
    dataset = RepeatedImageDataset(steps * local_batch_size * world_size)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )
    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=local_batch_size,
        num_workers=0,
        pin_memory=True,
    )


def scalar_dict(values):
    return {
        key: float(value.detach().float().cpu()) if torch.is_tensor(value) else float(value)
        for key, value in values.items()
    }


def numerical_snapshot(trainer, local_batch_size, dataset=None):
    model = trainer.model
    model.eval()
    trainer.discriminator.eval()
    if dataset is None:
        loader = make_loader(
            steps=1,
            local_batch_size=local_batch_size,
            world_size=trainer.distenv.world_size,
            rank=trainer.distenv.world_rank,
        )
    else:
        sampler = DistributedSampler(
            dataset,
            num_replicas=trainer.distenv.world_size,
            rank=trainer.distenv.world_rank,
            shuffle=False,
        )
        loader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=local_batch_size,
            num_workers=4,
            pin_memory=True,
        )
    xs = next(iter(loader))[0].to(trainer.device, non_blocking=True)
    with torch.no_grad(), trainer.autocast_context():
        raw_outputs = model(xs)
        reconstructed = raw_outputs[0]
        losses = model.module.compute_loss(*raw_outputs, xs=xs)
        perceptual = trainer.perceptual_loss(xs, reconstructed)
        gen_loss, disc_loss, _ = trainer.gan_loss(xs, reconstructed, mode="eval")

    snapshot = {
        "reconstruction": reconstructed[:2].float().cpu(),
        "support": losses["codes"][0].cpu(),
        "losses": {
            "loss_total": float(losses["loss_total"].detach().float().cpu()),
            "loss_recon": float(losses["loss_recon"].detach().float().cpu()),
            "loss_latent": float(losses["loss_latent"].detach().float().cpu()),
            "loss_perceptual": float(perceptual.detach().float().cpu()),
            "loss_gen": float(gen_loss.detach().float().cpu()),
            "loss_disc": float(disc_loss.detach().float().cpu()),
        },
    }
    model.train()
    trainer.discriminator.train()
    return snapshot


def compare_snapshot(reference, candidate):
    ref_recon = reference["reconstruction"].float()
    new_recon = candidate["reconstruction"].float()
    delta = new_recon - ref_recon
    reference_scale = ref_recon.square().mean().sqrt().clamp_min(1.0e-12)
    loss_relative = {}
    for name, ref_value in reference["losses"].items():
        new_value = candidate["losses"][name]
        loss_relative[name] = abs(new_value - ref_value) / max(abs(ref_value), 1.0e-12)
    support_equal = candidate["support"] == reference["support"]
    return {
        "support_agreement": float(support_equal.float().mean()),
        "ordered_code_exact_agreement": float(
            support_equal.all(dim=-1).float().mean()
        ),
        "unordered_support_exact_agreement": float(
            candidate["support"].sort(dim=-1).values.eq(
                reference["support"].sort(dim=-1).values
            ).all(dim=-1).float().mean()
        ),
        "support_agreement_by_depth": [
            float(support_equal[..., depth].float().mean())
            for depth in range(int(support_equal.size(-1)))
        ],
        "reconstruction_mae": float(delta.abs().mean()),
        "reconstruction_rmse": float(delta.square().mean().sqrt()),
        "reconstruction_relative_rmse": float(delta.square().mean().sqrt() / reference_scale),
        "reconstruction_max_abs": float(delta.abs().max()),
        "loss_relative_error": loss_relative,
    }


def main():
    args = parse_args()
    if args.local_batch_size <= 0 or args.steps <= 0 or args.warmup_steps <= 0:
        raise ValueError("batch size, warmup steps, and measured steps must be positive")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    distenv = DistEnv(
        world_size=world_size,
        world_rank=rank,
        local_rank=local_rank,
        num_gpus=1,
        master=rank == 0,
        device_name=torch.cuda.get_device_name(device),
    )

    config = OmegaConf.load(args.config)
    config.experiment.batch_size = args.local_batch_size
    config.experiment.total_batch_size = args.local_batch_size * world_size
    config.experiment.precision = args.precision
    config.arch.hparams.omp_compute_precision = args.omp_precision
    config.experiment.amp = False
    config.experiment.recovery_ckpt_freq_steps = 0
    if args.disable_activation_checkpointing:
        config.arch.checkpointing = False
    config.seed = int(config.get("seed", 0))
    config.result_path = str(args.output.resolve().parent)
    config.runtime = {"distenv": distenv}

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    original_steps_per_epoch = int(checkpoint["steps_per_epoch"])

    model, model_ema = create_model(config.arch, ema=False)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(
        optimizer,
        config.optimizer.warmup,
        original_steps_per_epoch,
        config.experiment.epochs,
        distenv,
    )
    # Match main_stage1.py ordering: scheduler construction mutates optimizer
    # LR, so checkpoint optimizer state must be restored after construction.
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    model = dist_utils.dataparallel_and_sync(distenv, model)

    original_length = original_steps_per_epoch * args.local_batch_size * world_size
    original_dataset = RepeatedImageDataset(original_length)
    trainer = Trainer(
        model,
        model_ema,
        original_dataset,
        original_dataset,
        config,
        NullWriter(),
        device,
        distenv,
        disc_state_dict=checkpoint["discriminator"],
        disc_optimizer_state_dict=checkpoint["discriminator_optimizer"],
        disc_scheduler_state_dict=checkpoint["discriminator_scheduler"],
        lineage_exact=False,
        lineage_origin="benchmark",
    )

    torch.manual_seed(918273 + rank)
    torch.cuda.manual_seed(918273 + rank)
    snapshot_dataset = None
    if args.imagenet_root is not None:
        snapshot_dataset = ImageFolder(
            args.imagenet_root / "val",
            transform=create_transforms(config.dataset, split="val", is_eval=True),
        )
    snapshot = numerical_snapshot(
        trainer,
        args.local_batch_size,
        dataset=snapshot_dataset,
    )

    if args.snapshot_only:
        if rank == 0:
            result = {
                "precision": args.precision,
                "omp_precision": args.omp_precision,
                "world_size": world_size,
                "local_batch_size": args.local_batch_size,
                "snapshot_losses": snapshot["losses"],
            }
            if args.reference is not None:
                reference = torch.load(
                    args.reference, map_location="cpu", weights_only=False
                )
                result["reference_path"] = str(args.reference.resolve())
                result["comparison_to_reference"] = compare_snapshot(reference, snapshot)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(snapshot, args.output)
            args.output.with_suffix(".json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n"
            )
            print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        dist.barrier()
        dist.destroy_process_group()
        return

    trainer.loader_trn = make_loader(
        steps=args.warmup_steps,
        local_batch_size=args.local_batch_size,
        world_size=world_size,
        rank=rank,
    )
    trainer.train(optimizer, scheduler, epoch=0)

    trainer.loader_trn = make_loader(
        steps=args.steps,
        local_batch_size=args.local_batch_size,
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.reset_peak_memory_stats(device)
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    summary = trainer.train(optimizer, scheduler, epoch=0)
    end.record()
    end.synchronize()
    elapsed_seconds = start.elapsed_time(end) / 1000.0

    elapsed = torch.tensor(elapsed_seconds, device=device)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    peak_memory = torch.tensor(
        torch.cuda.max_memory_allocated(device),
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)

    if rank == 0:
        result = {
            "precision": args.precision,
            "omp_precision": args.omp_precision,
            "world_size": world_size,
            "local_batch_size": args.local_batch_size,
            "steps": args.steps,
            "elapsed_seconds": float(elapsed.cpu()),
            "seconds_per_step": float(elapsed.cpu()) / args.steps,
            "images_per_second": (
                args.local_batch_size * world_size * args.steps / float(elapsed.cpu())
            ),
            "peak_memory_gib": float(peak_memory.cpu()) / (1024 ** 3),
            "activation_checkpointing": not args.disable_activation_checkpointing,
            "training_metrics": scalar_dict(summary.metrics),
            "snapshot_losses": snapshot["losses"],
        }
        if args.reference is not None:
            reference = torch.load(args.reference, map_location="cpu", weights_only=False)
            result["reference_path"] = str(args.reference.resolve())
            result["comparison_to_reference"] = compare_snapshot(reference, snapshot)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(snapshot, args.output)
        json_path = args.output.with_suffix(".json")
        json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
