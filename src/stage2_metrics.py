"""Optional final metrics for decoded stage-2 generation samples."""

from __future__ import annotations

import torch
from omegaconf import DictConfig

from src.audio_logging import compute_audio_generation_metrics


def _is_null_config_value(value) -> bool:
    if value is None:
        return True
    text = str(value).strip().lower()
    return text in {"", "none", "null"}


def _cfg_or_meta(cfg_section, cache_meta: dict, key: str, default=None):
    """Prefer explicit Hydra config, then fall back to token-cache metadata."""
    value = getattr(cfg_section, key, None)
    if not _is_null_config_value(value):
        return value
    value = cache_meta.get(key) if isinstance(cache_meta, dict) else None
    if not _is_null_config_value(value):
        return value
    return default


def _float_tuple(value, *, fallback, channels: int) -> tuple:
    if _is_null_config_value(value):
        value = fallback
    if value is None:
        value = (0.5,) * int(channels)
    if not isinstance(value, (list, tuple)):
        try:
            value = tuple(value)
        except TypeError:
            value = (float(value),) * int(channels)
    out = tuple(float(v) for v in value)
    if len(out) == 1 and channels > 1:
        out = out * int(channels)
    return out[: int(channels)]


def _to_unit_range(images: torch.Tensor, *, mean: tuple, std: tuple) -> torch.Tensor:
    channels = int(images.size(1))
    mean_t = torch.tensor(mean, dtype=torch.float32, device=images.device).view(1, channels, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32, device=images.device).view(1, channels, 1, 1)
    unit = images.to(dtype=torch.float32) * std_t + mean_t
    return torch.nan_to_num(unit, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)


def _compute_generation_fid(
    generated: torch.Tensor,
    *,
    cfg: DictConfig,
    cache_meta: dict,
    max_items: int,
) -> dict:
    # FID is image-only and optional, so keep torchmetrics/data imports inside
    # this path instead of making every stage-2 run pay that dependency cost.
    if max_items <= 0 or not torch.is_tensor(generated) or generated.ndim != 4 or int(generated.size(1)) != 3:
        return {}

    dataset = str(_cfg_or_meta(cfg.data, cache_meta, "dataset", "") or "").strip().lower()
    data_dir = _cfg_or_meta(cfg.data, cache_meta, "data_dir", None)
    if _is_null_config_value(data_dir):
        return {}

    from torchmetrics.image.fid import FrechetInceptionDistance

    from src.data.celeba import CelebADataModule
    from src.data.config import DataConfig
    from src.data.image_folder import ImageFolderDataModule, PAPER_IMAGE_FOLDER_DATASETS

    if dataset in {"celeba", "celebahq"}:
        dm_cls = CelebADataModule
    elif dataset in PAPER_IMAGE_FOLDER_DATASETS:
        dm_cls = ImageFolderDataModule
    else:
        return {}

    count = min(int(max_items), int(generated.size(0)))
    if count <= 0:
        return {}

    image_size = int(_cfg_or_meta(cfg.data, cache_meta, "image_size", 256))
    num_workers = int(_cfg_or_meta(cfg.data, cache_meta, "num_workers", 4) or 0)
    mean = _float_tuple(getattr(cfg.data, "mean", None), fallback=cache_meta.get("mean"), channels=3)
    std = _float_tuple(getattr(cfg.data, "std", None), fallback=cache_meta.get("std"), channels=3)
    batch_size = min(max(1, int(count)), max(1, int(getattr(cfg.train_ar, "batch_size", count) or count)))
    dm = dm_cls(
        DataConfig(
            dataset=dataset,
            data_dir=str(data_dir),
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            seed=int(getattr(cfg, "seed", 42)),
            mean=mean,
            std=std,
            augment=False,
        )
    )
    dm.prepare_data()
    dm.setup("fit")
    loader = dm.val_dataloader()
    if loader is None:
        return {}

    device = generated.device
    if device.type == "cpu" and torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type == "cpu":
        print("Warning: skipping generation FID because torch-fidelity has no CPU backend here")
        return {}

    metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    metric.eval()
    fake = _to_unit_range(generated[:count].to(device=device), mean=mean, std=std)
    metric.update(fake, real=False)

    seen = 0
    with torch.inference_mode():
        for batch in loader:
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            keep = min(int(images.size(0)), count - seen)
            if keep <= 0:
                break
            real = _to_unit_range(
                images[:keep].to(device=device, non_blocking=(device.type == "cuda")),
                mean=mean,
                std=std,
            )
            metric.update(real, real=True)
            seen += int(keep)
            if seen >= count:
                break

    if seen <= 0:
        return {}
    score = float(metric.compute().item())
    return {
        "generation/fid": score,
        "s2/generation_fid": score,
    }


def build_stage2_metrics_payload(
    generated: torch.Tensor,
    *,
    cfg: DictConfig,
    cache: dict,
    max_items: int,
    compute_fid: bool,
    compute_audio: bool,
) -> dict:
    """Build W&B scalar payloads for the optional post-training generation pass."""
    cache_meta = dict(cache.get("meta", {}) or {}) if isinstance(cache, dict) else {}
    payload = {}
    if compute_fid:
        try:
            payload.update(
                _compute_generation_fid(
                    generated,
                    cfg=cfg,
                    cache_meta=cache_meta,
                    max_items=max_items,
                )
            )
        except Exception as err:
            print(f"Warning: could not compute generation FID ({err})")
    if compute_audio:
        try:
            audio_metrics = compute_audio_generation_metrics(
                generated,
                audio_source=cache_meta,
                audio_meta=cache.get("audio_meta") if isinstance(cache, dict) else None,
                max_items=max_items,
            )
            for name, value in audio_metrics.items():
                scalar = float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value)
                payload[f"generation/{name}"] = scalar
                payload[f"s2/{name}"] = scalar
        except Exception as err:
            print(f"Warning: could not compute audio generation metrics ({err})")
    return payload
