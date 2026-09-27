#!/usr/bin/env python3
"""Full ImageNet generation FID/IS sweep for LASER RQTransformer checkpoints."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import sys
import time
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "rq-vae-transformer"))
from rqvae.metrics.fid import frechet_distance, get_inception_model  # noqa: E402
from scripts.train_official_rqtransformer_laser_stage2 import (  # noqa: E402
    CompoundLaserRQTransformer,
    LaserAux,
    build_model,
    release_cuda_memory,
    val_image_transform,
)


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float
    top_p: float
    top_k: int | None
    rejection_factor: int = 1

    @property
    def label(self) -> str:
        top_k_label = "full" if self.top_k is None else str(self.top_k)
        base = f"temp{self.temperature:g}_topP{self.top_p:g}_topK{top_k_label}"
        if self.rejection_factor > 1:
            return f"{base}_rej{self.rejection_factor}"
        return base


@dataclass
class OfficialFIDReference:
    """Official RQ-Transformer FID stats and matching feature extractor."""

    path: Path
    mu: np.ndarray
    sigma: np.ndarray
    model: torch.nn.Module


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def world() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def is_rank0() -> bool:
    return rank() == 0


def atomic_json(payload, path: Path) -> None:
    if not is_rank0():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_floats(raw: str) -> list[float]:
    return [float(item) for item in str(raw).replace(";", ",").split(",") if item.strip()]


def parse_top_ks(raw: str) -> list[int | None]:
    values: list[int | None] = []
    for item in str(raw).replace(";", ",").split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token in {"0", "none", "full", "all"}:
            values.append(None)
        else:
            values.append(int(token))
    return values


def parse_ints(raw: str) -> list[int]:
    return [int(item) for item in str(raw).replace(";", ",").split(",") if item.strip()]


def build_sampling_configs(args: argparse.Namespace) -> list[SamplingConfig]:
    temperatures = parse_floats(args.temperatures)
    top_ps = parse_floats(args.top_ps)
    top_ks = parse_top_ks(args.top_ks)
    rejection_factors = [factor for factor in parse_ints(args.rejection_factors) if factor > 1]

    configs: list[SamplingConfig] = []
    for temperature in temperatures:
        for top_p in top_ps:
            for top_k in top_ks:
                configs.append(SamplingConfig(temperature, top_p, top_k, 1))

    rejection_base = args.rejection_base.strip()
    if rejection_factors and rejection_base:
        base_parts = rejection_base.split(",")
        if len(base_parts) != 3:
            raise ValueError("--rejection-base must be temperature,top_p,top_k")
        base_temperature = float(base_parts[0])
        base_top_p = float(base_parts[1])
        base_top_k = parse_top_ks(base_parts[2])[0]
        for factor in rejection_factors:
            configs.append(SamplingConfig(base_temperature, base_top_p, base_top_k, factor))
    return configs


def configure_seed(seed: int, config_index: int) -> None:
    value = int(seed) + 1009 * int(config_index) + rank()
    random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


@torch.no_grad()
def reconstruction_fid(aux: LaserAux, loader: DataLoader, device: torch.device) -> tuple[float, int]:
    metric = FrechetInceptionDistance(
        feature=2048, normalize=True, sync_on_compute=dist.is_initialized()
    ).to(device)
    seen = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        tokens, _ = aux.encode_sparse(images, temp=0.5, stochastic=False)
        recon = ((aux.decode_tokens(tokens).float() + 1.0) * 0.5).clamp(0, 1)
        real = ((images.float() + 1.0) * 0.5).clamp(0, 1)
        metric.update(real, real=True)
        metric.update(recon, real=False)
        seen += images.size(0)
    return float(metric.compute().item()), seen


@torch.no_grad()
def build_real_fid(loader: DataLoader, device: torch.device, num_samples: int):
    metric = FrechetInceptionDistance(
        feature=2048, normalize=True, sync_on_compute=dist.is_initialized()
    ).to(device)
    local_samples = num_samples // world() + (rank() < num_samples % world())
    seen = 0
    for images, _ in loader:
        images = ((images.to(device, non_blocking=True).float() + 1.0) * 0.5).clamp(0, 1)
        keep = min(images.size(0), local_samples - seen)
        if keep > 0:
            metric.update(images[:keep], real=True)
            seen += keep
        if seen >= local_samples:
            break
    if seen != local_samples:
        raise RuntimeError(f"rank {rank()} saw {seen} real samples, expected {local_samples}")
    return metric


def build_official_fid_reference(path: Path, device: torch.device) -> OfficialFIDReference:
    if not path.is_file():
        raise FileNotFoundError(f"official FID statistics not found: {path}")
    with np.load(path) as payload:
        if "mu" not in payload or "sigma" not in payload:
            raise ValueError(f"{path} must contain mu and sigma arrays")
        mu = np.asarray(payload["mu"], dtype=np.float64)
        sigma = np.asarray(payload["sigma"], dtype=np.float64)
    if mu.shape != (2048,) or sigma.shape != (2048, 2048):
        raise ValueError(
            f"unexpected official FID stat shapes in {path}: mu={mu.shape}, sigma={sigma.shape}"
        )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if dist.is_initialized() and local_rank != 0:
        dist.barrier()
    model = get_inception_model(dims=2048).to(device).eval()
    if dist.is_initialized() and local_rank == 0:
        dist.barrier()
    return OfficialFIDReference(path=path, mu=mu, sigma=sigma, model=model)


def official_fake_fid(
    reference: OfficialFIDReference,
    feature_sum: torch.Tensor,
    feature_outer_sum: torch.Tensor,
    feature_count: torch.Tensor,
) -> float:
    if dist.is_initialized():
        dist.all_reduce(feature_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(feature_outer_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(feature_count, op=dist.ReduceOp.SUM)
    count = int(feature_count.item())
    if count < 2:
        raise RuntimeError(f"need at least two generated features for FID; got {count}")

    mean = feature_sum / count
    covariance = (feature_outer_sum - count * torch.outer(mean, mean)) / (count - 1)
    fid_tensor = torch.zeros((), dtype=torch.float64, device=feature_sum.device)
    if is_rank0():
        fid_tensor.fill_(float(frechet_distance(
            reference.mu,
            reference.sigma,
            mean.cpu().numpy(),
            covariance.cpu().numpy(),
        )))
    if dist.is_initialized():
        dist.broadcast(fid_tensor, src=0)
    return float(fid_tensor.item())


def build_rejection_classifier(device: torch.device, arch: str):
    arch = str(arch).strip().lower()
    if arch == "none":
        return None, None, None, None, arch
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if dist.is_initialized() and local_rank != 0:
        dist.barrier()
    if arch == "resnet50":
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights).eval().to(device)
        image_size = 224
    elif arch == "inception_v3":
        from torchvision.models import Inception_V3_Weights, inception_v3

        weights = Inception_V3_Weights.IMAGENET1K_V1
        model = inception_v3(weights=weights).eval().to(device)
        image_size = 299
    else:
        raise ValueError(f"Unsupported rejection classifier: {arch}")
    if dist.is_initialized() and local_rank == 0:
        dist.barrier()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    mean = torch.tensor(weights.transforms().mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(weights.transforms().std, device=device).view(1, 3, 1, 1)
    return model, mean, std, image_size, arch


@torch.no_grad()
def classifier_scores(classifier, mean, std, image_size: int, images: torch.Tensor, labels: torch.Tensor):
    resized = F.interpolate(images.float(), size=(image_size, image_size), mode="bilinear", align_corners=False)
    normalized = (resized - mean) / std
    logits = classifier(normalized)
    if isinstance(logits, tuple):
        logits = logits[0]
    probs = logits.float().softmax(dim=1)
    scores = probs.gather(1, labels.long().view(-1, 1)).squeeze(1)
    predicted = probs.argmax(dim=1)
    return scores, predicted


@torch.no_grad()
def sample_images(model, aux: LaserAux, labels: torch.Tensor, config: SamplingConfig):
    device = labels.device
    if isinstance(model, CompoundLaserRQTransformer):
        top_k = aux.num_atoms if config.top_k is None else min(int(config.top_k), aux.num_atoms)
        atoms, coeff_ids = model.sample_compound(
            labels.numel(),
            aux,
            cond=labels,
            temperature=float(config.temperature),
            atom_top_k=top_k,
            atom_top_p=float(config.top_p),
            coeff_top_p=float(config.top_p),
            amp=True,
        )
        return ((aux.decode_compound(atoms, coeff_ids).float() + 1.0) * 0.5).clamp(0, 1)

    partial = torch.zeros(labels.numel(), 8, 8, 4, device=device, dtype=torch.long)
    tokens = model.sample(
        partial,
        model_aux=aux,
        cond=labels,
        temperature=float(config.temperature),
        top_k=config.top_k,
        top_p=float(config.top_p),
        amp=True,
        cached=True,
        is_tqdm=False,
    )
    return ((aux.decode_tokens(tokens).float() + 1.0) * 0.5).clamp(0, 1)


def _uniform_imagenet_labels(start: int, count: int, device: torch.device) -> torch.Tensor:
    local_indices = torch.arange(start, start + count, device=device, dtype=torch.long)
    return (local_indices * world() + rank()).remainder(1000)


@torch.no_grad()
def generation_metrics(
    model,
    aux: LaserAux,
    real_metric,
    device: torch.device,
    num_samples: int,
    batch_size: int,
    config: SamplingConfig,
    classifier_bundle,
) -> dict[str, object]:
    official_reference = real_metric if isinstance(real_metric, OfficialFIDReference) else None
    metric = None if official_reference is not None else copy.deepcopy(real_metric)
    feature_sum = torch.zeros(2048, dtype=torch.float64, device=device)
    feature_outer_sum = torch.zeros(2048, 2048, dtype=torch.float64, device=device)
    feature_count = torch.zeros((), dtype=torch.long, device=device)
    inception_score = InceptionScore(
        feature="logits_unbiased", normalize=True, splits=10,
        sync_on_compute=dist.is_initialized(),
    ).to(device)
    local_samples = num_samples // world() + (rank() < num_samples % world())
    generated = 0
    selected_score_sum = torch.zeros((), device=device)
    selected_match_count = torch.zeros((), device=device)
    selected_count = torch.zeros((), device=device)
    start_time = time.monotonic()

    classifier, classifier_mean, classifier_std, classifier_image_size, classifier_name = classifier_bundle
    use_rejection = config.rejection_factor > 1
    if use_rejection and classifier is None:
        raise ValueError("rejection sampling requires a rejection classifier")

    while generated < local_samples:
        current = min(int(batch_size), local_samples - generated)
        labels = _uniform_imagenet_labels(generated, current, device)
        if not use_rejection:
            images = sample_images(model, aux, labels, config)
        else:
            best_images = None
            best_scores = None
            best_pred = None
            for _ in range(config.rejection_factor):
                candidate = sample_images(model, aux, labels, SamplingConfig(
                    config.temperature, config.top_p, config.top_k, 1
                ))
                scores, predicted = classifier_scores(
                    classifier, classifier_mean, classifier_std,
                    classifier_image_size, candidate, labels
                )
                if best_scores is None:
                    best_images = candidate.detach().clone()
                    best_scores = scores
                    best_pred = predicted
                else:
                    better = scores > best_scores
                    best_images[better] = candidate[better]
                    best_scores = torch.where(better, scores, best_scores)
                    best_pred = torch.where(better, predicted, best_pred)
                del candidate, scores, predicted
                release_cuda_memory(model)
            images = best_images
            selected_score_sum += best_scores.sum()
            selected_match_count += (best_pred == labels).float().sum()
            selected_count += labels.numel()

        if official_reference is not None:
            features = official_reference.model(images).double()
            feature_sum += features.sum(dim=0)
            feature_outer_sum += features.T @ features
            feature_count += features.shape[0]
            del features
        else:
            metric.update(images, real=False)
        inception_score.update(images)
        generated += current
        del images, labels
        release_cuda_memory(model)

    if official_reference is not None:
        fid = official_fake_fid(
            official_reference, feature_sum, feature_outer_sum, feature_count
        )
    else:
        fid = float(metric.compute().item())
    is_mean, is_std = inception_score.compute()

    if dist.is_initialized() and use_rejection:
        dist.all_reduce(selected_score_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(selected_match_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(selected_count, op=dist.ReduceOp.SUM)

    result: dict[str, object] = {
        "label": config.label,
        "temperature": float(config.temperature),
        "top_p": float(config.top_p),
        "top_k": "full" if config.top_k is None else int(config.top_k),
        "rejection_factor": int(config.rejection_factor),
        "fid": fid,
        "inception_score_mean": float(is_mean.item()),
        "inception_score_std": float(is_std.item()),
        "inception_score_splits": 10,
        "num_samples": int(num_samples),
        "world_size": int(world()),
        "elapsed_seconds": float(time.monotonic() - start_time),
    }
    if use_rejection:
        denom = float(selected_count.item()) if selected_count.item() > 0 else 1.0
        result.update({
            "rejection_classifier": str(classifier_name),
            "rejection_avg_target_probability": float(selected_score_sum.item() / denom),
            "rejection_selected_top1_match_fraction": float(selected_match_count.item() / denom),
        })
    return result


def log_wandb(wb, payload: dict[str, object]) -> None:
    if wb is None or not is_rank0():
        return
    wb.log(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--real-split", choices=["train", "val"], default="train")
    parser.add_argument(
        "--real-stats",
        type=Path,
        default=None,
        help=(
            "Precomputed RQ-Transformer FID .npz (mu/sigma). When set, use its "
            "matching Inception implementation instead of scanning real images."
        ),
    )
    parser.add_argument(
        "--real-num-samples",
        type=int,
        default=0,
        help="Number of real reference images; <=0 uses the entire selected split.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--temperatures", default="0.9,1.0,1.05")
    parser.add_argument("--top-ps", default="0.90,0.92,0.95")
    parser.add_argument("--top-ks", default="full")
    parser.add_argument(
        "--rejection-factors",
        default="",
        help="Comma-separated best-of-N rejection factors. Values <=1 are ignored.",
    )
    parser.add_argument(
        "--rejection-base",
        default="1.0,0.92,full",
        help="Single sampler point for rejection as temperature,top_p,top_k.",
    )
    parser.add_argument("--rejection-classifier", default="resnet50", choices=["none", "resnet50", "inception_v3"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-atoms", type=int, default=16384)
    parser.add_argument("--coeff-vocab-size", type=int, default=2048)
    parser.add_argument("--coeff-max", type=float, default=20.0)
    parser.add_argument("--coeff-scale", type=float, default=6.4)
    parser.add_argument("--compound-tokens", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-reconstruction-fid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-entity", default="helloimlixin-rutgers")
    parser.add_argument("--wandb-project", default="laser")
    parser.add_argument("--wandb-name", default="v8dup-full-imagenet-sampling-sweep")
    parser.add_argument("--wandb-id", default=None)
    parser.add_argument("--wandb-group", default="v8dup-sampling-eval")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output / "full_imagenet_sampling_sweep.json"
    configs = build_sampling_configs(args)
    if not configs:
        raise ValueError("sampling sweep has no configs")

    real_data_path = args.data / args.real_split
    dataset = None
    loader = None
    real_num_samples = int(args.real_num_samples)
    if args.real_stats is None or not args.skip_reconstruction_fid:
        dataset = datasets.ImageFolder(real_data_path, transform=val_image_transform())
        if real_num_samples <= 0:
            real_num_samples = len(dataset)
        if real_num_samples > len(dataset):
            raise ValueError(
                f"requested {real_num_samples} real images from {real_data_path}, "
                f"but the split contains only {len(dataset)}"
            )
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=False) if world() > 1 else None
        loader = DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            sampler=sampler,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )
    elif real_num_samples <= 0:
        raise ValueError("--real-num-samples must identify the reference population with --real-stats")

    aux = LaserAux(
        args.stage1, args.num_atoms, args.coeff_vocab_size, args.coeff_max, args.coeff_scale
    ).to(device).eval()
    model = build_model(
        args.num_atoms + args.coeff_vocab_size,
        args.num_atoms,
        compound=args.compound_tokens,
        coeff_vocab_size=args.coeff_vocab_size,
    ).to(device).eval()
    payload = torch.load(args.stage2, map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(payload["state_dict"], strict=True)

    checkpoint_epoch = int(payload.get("epoch", -1))
    checkpoint_step = int(payload.get("global_step", -1))
    results: dict[str, object] = {
        "checkpoint": str(args.stage2),
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_step": checkpoint_step,
        "data": str(args.data),
        "real_data_path": str(real_data_path),
        "real_split": str(args.real_split),
        "real_stats": str(args.real_stats) if args.real_stats is not None else None,
        "real_num_samples": int(real_num_samples),
        "num_samples": int(args.num_samples),
        "batch_size": int(args.batch_size),
        "world_size": int(world()),
        "reconstruction_fid": None,
        "reconstruction_images": None,
        "sampling_sweep": [],
        "completed": False,
    }

    wb = None
    if args.wandb and is_rank0():
        import wandb

        wb = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_name,
            id=args.wandb_id or None,
            group=args.wandb_group or None,
            job_type="full_imagenet_sampling_sweep",
            config={
                **{key: str(value) for key, value in vars(args).items()},
                "checkpoint_epoch": checkpoint_epoch,
                "checkpoint_step": checkpoint_step,
                "real_data_path_resolved": str(real_data_path),
                "real_stats_resolved": str(args.real_stats) if args.real_stats is not None else None,
                "real_num_samples_resolved": int(real_num_samples),
                "configs": [config.__dict__ for config in configs],
            },
        )

    atomic_json(results, results_path)
    if is_rank0():
        print(
            f"Loaded checkpoint epoch={checkpoint_epoch} step={checkpoint_step}; "
            f"{len(configs)} sweep configs; generated_per_config={args.num_samples}; "
            f"real_reference={args.real_stats or real_data_path} real_images={real_num_samples}; "
            f"results={results_path}",
            flush=True,
        )

    if not args.skip_reconstruction_fid:
        assert loader is not None
        rfid, local_seen = reconstruction_fid(aux, loader, device)
        results["reconstruction_fid"] = rfid
        results["reconstruction_images"] = int(local_seen * world())
        atomic_json(results, results_path)
        log_wandb(wb, {
            "reconstruction/fid": rfid,
            "reconstruction/images": int(local_seen * world()),
        })
        if is_rank0():
            print(f"Full ImageNet tokenizer reconstruction FID: {rfid:.6f}", flush=True)
        if dist.is_initialized():
            dist.barrier()

    if args.real_stats is not None:
        real_metric = build_official_fid_reference(args.real_stats, device)
    else:
        assert loader is not None
        real_metric = build_real_fid(loader, device, real_num_samples)
    classifier_bundle = (None, None, None, None, "none")
    if any(config.rejection_factor > 1 for config in configs):
        classifier_bundle = build_rejection_classifier(device, args.rejection_classifier)

    for index, config in enumerate(configs):
        configure_seed(args.seed, index)
        if dist.is_initialized():
            dist.barrier()
        if is_rank0():
            print(f"Starting {config.label}", flush=True)
        result = generation_metrics(
            model,
            aux,
            real_metric,
            device,
            args.num_samples,
            args.batch_size,
            config,
            classifier_bundle,
        )
        cast_results = results["sampling_sweep"]
        assert isinstance(cast_results, list)
        cast_results.append(result)
        atomic_json(results, results_path)
        log_wandb(wb, {
            "eval/fid": result["fid"],
            "eval/inception_score": result["inception_score_mean"],
            "eval/inception_score_std": result["inception_score_std"],
            "eval/temperature": result["temperature"],
            "eval/top_p": result["top_p"],
            "eval/rejection_factor": result["rejection_factor"],
            "eval/elapsed_seconds": result["elapsed_seconds"],
            "eval/config_index": index,
        })
        if is_rank0():
            print(
                f"{config.label}: FID={result['fid']:.6f} "
                f"IS={result['inception_score_mean']:.6f}+/-{result['inception_score_std']:.6f} "
                f"elapsed={result['elapsed_seconds']:.1f}s",
                flush=True,
            )

    results["completed"] = True
    atomic_json(results, results_path)
    if wb is not None:
        wb.save(str(results_path))
        wb.finish()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
