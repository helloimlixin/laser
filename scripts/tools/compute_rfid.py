#!/usr/bin/env python3
"""Compute stage-1 reconstruction FID (rFID) for a LASER or VQVAE checkpoint."""

import argparse
import logging
import os
import re
import sys

if sys.version_info < (3, 10):
    raise SystemExit("ERROR: scripts/tools/compute_rfid.py requires Python >= 3.10.")

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


_RUN_STAMP_RE = re.compile(r"run_(\d{8}_\d{6})")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_timestamp(text: str) -> Optional[datetime]:
    text = str(text).strip()
    if not text:
        return None
    for fmt in ("%Y%m%d_%H%M%S", "%Y-%m-%d_%H-%M-%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    match = _RUN_STAMP_RE.search(text)
    if match is not None:
        try:
            return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
        except ValueError:
            return None
    return None


def infer_config_path(ckpt_path: str | Path) -> Optional[Path]:
    ckpt = Path(ckpt_path).expanduser().resolve()
    roots = [ckpt.parent, *ckpt.parents[:6]]
    target_ts = None
    for part in ckpt.parts[::-1]:
        target_ts = _parse_timestamp(part)
        if target_ts is not None:
            break

    candidates: list[tuple[tuple[float, int, str], Path]] = []
    seen: set[Path] = set()
    rels = (Path(".hydra/config.yaml"), Path("config.yaml"))

    def _consider(path: Path, base_cost: int) -> None:
        resolved = path.resolve()
        if resolved in seen or not resolved.is_file():
            return
        seen.add(resolved)
        owner_ts = _parse_timestamp(resolved.parent.parent.name if resolved.parent.name == ".hydra" else resolved.parent.name)
        ts_cost = 1e18
        if target_ts is not None and owner_ts is not None:
            ts_cost = abs((owner_ts - target_ts).total_seconds())
        elif target_ts is None:
            ts_cost = 0.0
        candidates.append(((ts_cost, int(base_cost), str(resolved)), resolved))

    for root_idx, root in enumerate(roots):
        for rel in rels:
            _consider(root / rel, base_cost=root_idx)
        try:
            children = [child for child in root.iterdir() if child.is_dir()]
        except OSError:
            children = []
        for child in children:
            for rel in rels:
                _consider(child / rel, base_cost=100 + root_idx)

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _extract_hparams(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("hyper_parameters"), dict):
        return dict(payload["hyper_parameters"])
    return {}


def infer_model_type(hparams: dict[str, Any], config_type: Optional[str], explicit: str) -> str:
    if explicit != "auto":
        return explicit
    if config_type in {"laser", "vqvae"}:
        return str(config_type)
    if "sparsity_level" in hparams or "patch_based" in hparams:
        return "laser"
    if "decay" in hparams:
        return "vqvae"
    raise ValueError("Could not infer model type from checkpoint. Pass --model laser or --model vqvae.")


def normalize_dataset_name(dataset: str) -> str:
    return str(dataset).strip().lower().replace("-", "_")


def _default_stats(dataset: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    name = normalize_dataset_name(dataset)
    if name == "cifar10":
        return (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    return (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


def _first_int(*values: Any) -> int:
    for value in values:
        if value in (None, ""):
            continue
        if isinstance(value, str) and "${" in value:
            continue
        return int(value)
    raise ValueError("Expected at least one integer-compatible value.")


_CELEBA_DATASETS = {"celeba", "celebahq"}
_IMAGE_FOLDER_DATASETS = {"ffhq", "imagenet", "lsun_bedroom", "lsun_church", "lsun_cat"}


def datamodule_kind_for_dataset(dataset: str) -> str:
    name = normalize_dataset_name(dataset)
    if name == "cifar10":
        return "cifar10"
    if name in _CELEBA_DATASETS:
        return "celeba"
    if name == "imagenette2":
        return "imagenette2"
    if name in _IMAGE_FOLDER_DATASETS:
        return "image_folder"
    raise ValueError(f"Unsupported dataset for rFID: {dataset!r}")


@dataclass
class DataArgs:
    dataset: str
    data_dir: str
    image_size: int
    batch_size: int
    num_workers: int
    seed: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


def build_data_args(args: argparse.Namespace, cfg_dict: Optional[dict[str, Any]]) -> DataArgs:
    data_cfg = dict((cfg_dict or {}).get("data") or {})
    dataset = normalize_dataset_name(args.dataset or data_cfg.get("dataset") or "")
    if not dataset:
        raise ValueError("Need a dataset name. Pass --dataset or provide a config with data.dataset.")

    data_dir = str(args.data_dir or data_cfg.get("data_dir") or "").strip()
    if "${" in data_dir:
        data_dir = ""

    image_size_raw = args.image_size if args.image_size > 0 else data_cfg.get("image_size")
    if image_size_raw in (None, ""):
        raise ValueError("Need an image size. Pass --image-size or provide a config with data.image_size.")
    if isinstance(image_size_raw, (list, tuple)):
        image_size = int(image_size_raw[0])
    else:
        image_size = int(image_size_raw)

    batch_size = int(args.batch_size if args.batch_size > 0 else (data_cfg.get("batch_size") or 100))
    num_workers = int(args.num_workers if args.num_workers >= 0 else (data_cfg.get("num_workers") or 4))
    seed = _first_int(data_cfg.get("seed"), (cfg_dict or {}).get("seed"), 42)

    default_mean, default_std = _default_stats(dataset)
    mean_raw = tuple(args.mean) if args.mean is not None else tuple(data_cfg.get("mean") or default_mean)
    std_raw = tuple(args.std) if args.std is not None else tuple(data_cfg.get("std") or default_std)
    mean = tuple(float(v) for v in mean_raw)
    std = tuple(float(v) for v in std_raw)

    return DataArgs(
        dataset=dataset,
        data_dir=data_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        mean=mean,
        std=std,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute rFID for a stage-1 LASER or VQVAE checkpoint.")
    p.add_argument("--ckpt", required=True, type=Path, help="Stage-1 Lightning checkpoint (.ckpt).")
    p.add_argument("--config", type=Path, default=None, help="Optional Hydra config.yaml for dataset settings.")
    p.add_argument("--model", choices=["auto", "laser", "vqvae"], default="auto", help="Checkpoint model type.")
    p.add_argument("--split", choices=["train", "val", "valid", "test"], default="val", help="Dataset split.")
    p.add_argument("--batch-size", type=int, default=100, help="Eval batch size.")
    p.add_argument("--num-workers", type=int, default=-1, help="Dataloader workers override.")
    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional debug cap on evaluated images; 0 uses the full split for paper-comparable rFID.",
    )
    p.add_argument("--device", type=str, default="auto", help="'auto', 'cpu', or CUDA device like 'cuda:0'.")
    p.add_argument("--feature", type=int, default=2048, help="Inception feature size for FID.")
    p.add_argument("--dataset", type=str, default=None, help="Dataset override when config inference is unavailable.")
    p.add_argument("--data-dir", type=str, default=None, help="Dataset root override.")
    p.add_argument("--image-size", type=int, default=0, help="Image size override.")
    p.add_argument("--mean", type=float, nargs=3, default=None, help="Normalization mean override.")
    p.add_argument("--std", type=float, nargs=3, default=None, help="Normalization std override.")
    return p.parse_args()


def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("compute_rfid")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def main() -> None:
    args = _parse_args()

    ckpt = args.ckpt.expanduser().resolve()
    cfg_path = args.config.expanduser().resolve() if args.config is not None else infer_config_path(ckpt)
    log_path = ckpt.parent / "rfid.log"
    logger = _setup_logger(log_path)

    import torch
    from omegaconf import OmegaConf
    from torchmetrics.image.fid import FrechetInceptionDistance

    from src.checkpoint_io import load_lightning_module, load_torch_payload
    from src.data.celeba import CelebADataModule
    from src.data.cifar10 import CIFAR10DataModule
    from src.data.config import DataConfig
    from src.data.image_folder import ImageFolderDataModule
    from src.data.imagenette2 import Imagenette2DataModule
    from src.models.laser import LASER
    from src.models.vqvae import VQVAE

    def load_payload(path: Path):
        return load_torch_payload(path, map_location="cpu")

    cfg_dict: Optional[dict[str, Any]] = None
    if cfg_path is not None and cfg_path.is_file():
        try:
            cfg_obj = OmegaConf.load(cfg_path)
            try:
                cfg_dict = OmegaConf.to_container(cfg_obj, resolve=True)  # type: ignore[assignment]
            except Exception:
                cfg_dict = OmegaConf.to_container(cfg_obj, resolve=False)  # type: ignore[assignment]
            logger.info("Loaded config: %s", cfg_path)
        except Exception as exc:
            logger.warning("Could not load config from %s: %s", cfg_path, exc)
            cfg_dict = None
    else:
        logger.info("No config inferred near %s; using CLI dataset arguments only.", ckpt)

    payload = load_payload(ckpt)
    hparams = _extract_hparams(payload)
    cfg_model_type = str(((cfg_dict or {}).get("model") or {}).get("type") or "").strip().lower() or None
    model_type = infer_model_type(hparams, cfg_model_type, args.model)
    data_args = build_data_args(args, cfg_dict)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info("Checkpoint: %s", ckpt)
    logger.info("Model: %s", model_type)
    logger.info("Split: %s", args.split)
    logger.info("Feature dim: %d", int(args.feature))
    logger.info("Log file: %s", log_path)
    logger.info(
        "Data: dataset=%s data_dir=%s image_size=%s batch_size=%s workers=%s",
        data_args.dataset,
        (data_args.data_dir or "<auto>"),
        data_args.image_size,
        data_args.batch_size,
        data_args.num_workers,
    )

    if model_type == "laser":
        model = load_lightning_module(
            LASER,
            ckpt,
            map_location="cpu",
            strict=False,
            compute_fid=False,
        )
    else:
        model = load_lightning_module(
            VQVAE,
            ckpt,
            map_location="cpu",
            strict=False,
            compute_fid=False,
        )
    model = model.eval().to(device)
    model.requires_grad_(False)

    dm_cfg = DataConfig(
        dataset=data_args.dataset,
        data_dir=data_args.data_dir,
        batch_size=data_args.batch_size,
        num_workers=data_args.num_workers,
        image_size=data_args.image_size,
        seed=data_args.seed,
        mean=data_args.mean,
        std=data_args.std,
        augment=False,
    )
    datamodule_kind = datamodule_kind_for_dataset(data_args.dataset)
    if datamodule_kind == "cifar10":
        dm = CIFAR10DataModule(dm_cfg)
    elif datamodule_kind == "celeba":
        dm = CelebADataModule(dm_cfg)
    elif datamodule_kind == "imagenette2":
        dm = Imagenette2DataModule(OmegaConf.create(dm_cfg.__dict__))
    elif datamodule_kind == "image_folder":
        dm = ImageFolderDataModule(dm_cfg)
    else:
        raise AssertionError(f"Unhandled rFID datamodule kind: {datamodule_kind!r}")
    dm.setup(None)

    split = str(args.split).strip().lower()
    if split == "train":
        loader = dm.train_dataloader()
    elif split in {"val", "valid"}:
        loader = dm.val_dataloader()
    else:
        loader = dm.test_dataloader()

    metric = FrechetInceptionDistance(feature=int(args.feature), normalize=True).to(device)
    metric.eval()

    mean = torch.tensor(data_args.mean, device=device, dtype=torch.float32).view(1, -1, 1, 1)
    std = torch.tensor(data_args.std, device=device, dtype=torch.float32).view(1, -1, 1, 1)

    def _images_only(batch):
        return batch[0] if isinstance(batch, (tuple, list)) else batch

    def _to_unit_range(x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=device, dtype=torch.float32)
        x = x * std + mean
        return torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)

    max_samples = int(args.max_samples)
    seen = 0
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            imgs = _images_only(batch)
            if max_samples > 0 and seen >= max_samples:
                break
            if max_samples > 0:
                keep = min(int(imgs.size(0)), max_samples - seen)
                imgs = imgs[:keep]
            imgs = imgs.to(device, non_blocking=True)
            recon = model(imgs)[0]

            real = _to_unit_range(imgs)
            fake = _to_unit_range(recon)
            metric.update(real, real=True)
            metric.update(fake, real=False)

            seen += int(real.size(0))
            if batch_idx == 0:
                logger.info(
                    "Batch0 stats: real[min=%.4f max=%.4f mean=%.4f] recon[min=%.4f max=%.4f mean=%.4f]",
                    float(real.min().item()),
                    float(real.max().item()),
                    float(real.mean().item()),
                    float(fake.min().item()),
                    float(fake.max().item()),
                    float(fake.mean().item()),
                )

    if seen <= 0:
        raise RuntimeError("No samples were processed for rFID.")

    score = float(metric.compute().item())
    logger.info("Processed samples: %d", seen)
    logger.info("rFID: %.4f", score)


if __name__ == "__main__":
    main()
