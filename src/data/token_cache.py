"""Cached sparse-token datasets for maintained stage-2 Lightning training."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from src.cache_sort import canonicalize_cache


def load_token_cache(path: str | Path, *, canonicalize: bool = False):
    resolved = Path(path).expanduser().resolve()
    try:
        cache = torch.load(resolved, map_location="cpu", weights_only=True)
    except TypeError:
        cache = torch.load(resolved, map_location="cpu")
    if canonicalize:
        return canonicalize_cache(cache)
    return cache


class CachedTokenDataset(Dataset):
    """Dataset backed by a precomputed token-cache payload."""

    def __init__(
        self,
        tokens_flat: torch.Tensor,
        coeffs_flat: Optional[torch.Tensor] = None,
        *,
        shape: Optional[tuple[int, int, int]] = None,
        crop_shape: Optional[tuple[int, int, int]] = None,
        crop_mode: str = "full",
        indices: Optional[list[int]] = None,
    ):
        if not torch.is_tensor(tokens_flat) or tokens_flat.ndim != 2:
            raise ValueError("tokens_flat must be a rank-2 tensor")
        if coeffs_flat is not None:
            if not torch.is_tensor(coeffs_flat) or coeffs_flat.ndim != 2:
                raise ValueError("coeffs_flat must be a rank-2 tensor when provided")
            if coeffs_flat.shape != tokens_flat.shape:
                raise ValueError(
                    f"coeffs_flat shape {tuple(coeffs_flat.shape)} must match tokens_flat shape {tuple(tokens_flat.shape)}"
                )
            self.coeffs_flat = coeffs_flat.to(torch.float32).contiguous()
        else:
            self.coeffs_flat = None
        self.tokens_flat = tokens_flat.to(torch.long).contiguous()
        self.shape = None if shape is None else tuple(int(v) for v in shape)
        self.crop_shape = None if crop_shape is None else tuple(int(v) for v in crop_shape)
        self.crop_mode = str(crop_mode).strip().lower()
        self.indices = None if indices is None else [int(v) for v in indices]

        if self.crop_shape is not None:
            if self.shape is None:
                raise ValueError("shape is required when crop_shape is provided")
            full_h, full_w, full_d = self.shape
            crop_h, crop_w, crop_d = self.crop_shape
            if crop_d != full_d:
                raise ValueError(
                    f"crop depth must match cache depth, got crop D={crop_d}, full D={full_d}"
                )
            if crop_h <= 0 or crop_w <= 0:
                raise ValueError(f"crop_shape must be positive, got {self.crop_shape}")
            if crop_h > full_h or crop_w > full_w:
                raise ValueError(
                    f"crop_shape {self.crop_shape} exceeds full token grid {self.shape}"
                )
            if self.crop_mode not in {"random", "center"}:
                raise ValueError(f"crop_mode must be 'random' or 'center', got {crop_mode!r}")
        elif self.crop_mode not in {"", "full"}:
            raise ValueError(f"crop_mode must be 'full' when crop_shape is unset, got {crop_mode!r}")

    def __len__(self) -> int:
        if self.indices is not None:
            return len(self.indices)
        return int(self.tokens_flat.size(0))

    def _crop_start(self, size: int, crop: int) -> int:
        if crop >= size:
            return 0
        if self.crop_mode == "center":
            return int((size - crop) // 2)
        return int(torch.randint(0, size - crop + 1, size=()).item())

    def __getitem__(self, idx: int):
        real_idx = int(self.indices[idx]) if self.indices is not None else int(idx)
        tokens = self.tokens_flat[real_idx]
        coeffs = None if self.coeffs_flat is None else self.coeffs_flat[real_idx]
        if self.crop_shape is not None:
            full_h, full_w, full_d = self.shape
            crop_h, crop_w, _ = self.crop_shape
            top = self._crop_start(full_h, crop_h)
            left = self._crop_start(full_w, crop_w)
            tokens = tokens.view(full_h, full_w, full_d)[top : top + crop_h, left : left + crop_w, :]
            tokens = tokens.reshape(-1)
            if coeffs is not None:
                coeffs = coeffs.view(full_h, full_w, full_d)[top : top + crop_h, left : left + crop_w, :]
                coeffs = coeffs.reshape(-1)
        if coeffs is None:
            return tokens
        return tokens, coeffs


class TokenCacheDataModule(pl.LightningDataModule):
    """Lightning datamodule for train/val/test splits from a cached token grid."""

    def __init__(
        self,
        cache_path: str,
        batch_size: int = 256,
        num_workers: int = 0,
        seed: int = 42,
        validation_fraction: float = 0.05,
        test_fraction: float = 0.05,
        max_items: int = 0,
        crop_h_sites: int = 0,
        crop_w_sites: int = 0,
    ):
        super().__init__()
        self.cache_path = str(cache_path)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.seed = int(seed)
        self.validation_fraction = float(max(0.0, validation_fraction))
        self.test_fraction = float(max(0.0, test_fraction))
        self.max_items = int(max_items)
        self.crop_h_sites = int(crop_h_sites)
        self.crop_w_sites = int(crop_w_sites)
        self.cache = None
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._token_shape = None
        self._full_token_shape = None

    def _split_lengths(self, total_items: int) -> tuple[int, int, int]:
        if total_items <= 1:
            return total_items, 0, 0

        test_items = 0
        if self.test_fraction > 0.0 and total_items >= 3:
            test_items = max(1, int(round(total_items * self.test_fraction)))
            test_items = min(test_items, total_items - 2)

        remaining = total_items - test_items
        val_items = 0
        if self.validation_fraction > 0.0 and remaining >= 2:
            val_items = max(1, int(round(remaining * self.validation_fraction)))
            val_items = min(val_items, remaining - 1)

        train_items = total_items - val_items - test_items
        return train_items, val_items, test_items

    def setup(self, stage: Optional[str] = None):
        if self.dataset is not None:
            return

        cache = load_token_cache(self.cache_path)
        tokens_flat = cache.get("tokens_flat")
        coeffs_flat = cache.get("coeffs_flat")
        if self.max_items > 0:
            tokens_flat = tokens_flat[: self.max_items]
            if coeffs_flat is not None:
                coeffs_flat = coeffs_flat[: self.max_items]

        self.cache = dict(cache)
        self.cache["tokens_flat"] = tokens_flat
        if coeffs_flat is not None:
            self.cache["coeffs_flat"] = coeffs_flat

        full_shape = tuple(int(v) for v in cache.get("shape"))
        self._full_token_shape = full_shape
        crop_h = int(self.crop_h_sites or 0)
        crop_w = int(self.crop_w_sites or 0)
        if crop_h > 0 or crop_w > 0:
            if crop_h <= 0:
                crop_h = full_shape[0]
            if crop_w <= 0:
                crop_w = full_shape[1]
            self._token_shape = (crop_h, crop_w, full_shape[2])
            meta = dict(self.cache.get("meta", {}) or {})
            meta["full_token_shape"] = full_shape
            meta["train_crop_shape"] = self._token_shape
            self.cache["meta"] = meta
        else:
            self._token_shape = full_shape

        self.dataset = CachedTokenDataset(
            tokens_flat=tokens_flat,
            coeffs_flat=coeffs_flat,
            shape=full_shape,
        )
        total_items = len(self.dataset)
        train_items, val_items, test_items = self._split_lengths(total_items)

        permutation = torch.randperm(total_items, generator=torch.Generator().manual_seed(self.seed)).tolist()
        train_idx = permutation[:train_items]
        val_idx = permutation[train_items : train_items + val_items]
        test_idx = permutation[train_items + val_items : train_items + val_items + test_items]

        crop_shape = None if self._token_shape == full_shape else self._token_shape
        self.train_dataset = CachedTokenDataset(
            tokens_flat=tokens_flat,
            coeffs_flat=coeffs_flat,
            shape=full_shape,
            crop_shape=crop_shape,
            crop_mode=("random" if crop_shape is not None else "full"),
            indices=train_idx,
        )
        self.val_dataset = (
            CachedTokenDataset(
                tokens_flat=tokens_flat,
                coeffs_flat=coeffs_flat,
                shape=full_shape,
                crop_shape=crop_shape,
                crop_mode=("center" if crop_shape is not None else "full"),
                indices=val_idx,
            )
            if val_idx
            else None
        )
        self.test_dataset = (
            CachedTokenDataset(
                tokens_flat=tokens_flat,
                coeffs_flat=coeffs_flat,
                shape=full_shape,
                crop_shape=crop_shape,
                crop_mode=("center" if crop_shape is not None else "full"),
                indices=test_idx,
            )
            if test_idx
            else self.val_dataset
        )

    @property
    def token_shape(self) -> tuple[int, int, int]:
        if self.cache is None or self._token_shape is None:
            raise RuntimeError("Token cache is not loaded yet. Call setup() first.")
        return tuple(int(v) for v in self._token_shape)

    @property
    def metadata(self) -> dict:
        if self.cache is None:
            raise RuntimeError("Token cache is not loaded yet. Call setup() first.")
        return dict(self.cache.get("meta", {}))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=(len(self.train_dataset) >= self.batch_size),
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
