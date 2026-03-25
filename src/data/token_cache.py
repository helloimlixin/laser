"""Cached sparse-token datasets for maintained stage-2 Lightning training."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Subset


def load_token_cache(path: str | Path):
    resolved = Path(path).expanduser().resolve()
    try:
        return torch.load(resolved, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(resolved, map_location="cpu")


class CachedTokenDataset(Dataset):
    """Dataset backed by a precomputed token-cache payload."""

    def __init__(self, tokens_flat: torch.Tensor, coeffs_flat: Optional[torch.Tensor] = None):
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

    def __len__(self) -> int:
        return int(self.tokens_flat.size(0))

    def __getitem__(self, idx: int):
        if self.coeffs_flat is None:
            return self.tokens_flat[idx]
        return self.tokens_flat[idx], self.coeffs_flat[idx]


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
    ):
        super().__init__()
        self.cache_path = str(cache_path)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.seed = int(seed)
        self.validation_fraction = float(max(0.0, validation_fraction))
        self.test_fraction = float(max(0.0, test_fraction))
        self.max_items = int(max_items)
        self.cache = None
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

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

        self.dataset = CachedTokenDataset(tokens_flat=tokens_flat, coeffs_flat=coeffs_flat)
        total_items = len(self.dataset)
        train_items, val_items, test_items = self._split_lengths(total_items)

        permutation = torch.randperm(total_items, generator=torch.Generator().manual_seed(self.seed)).tolist()
        train_idx = permutation[:train_items]
        val_idx = permutation[train_items : train_items + val_items]
        test_idx = permutation[train_items + val_items : train_items + val_items + test_items]

        self.train_dataset = Subset(self.dataset, train_idx)
        self.val_dataset = Subset(self.dataset, val_idx) if val_idx else None
        self.test_dataset = Subset(self.dataset, test_idx) if test_idx else self.val_dataset

    @property
    def token_shape(self) -> tuple[int, int, int]:
        if self.cache is None:
            raise RuntimeError("Token cache is not loaded yet. Call setup() first.")
        shape = self.cache.get("shape")
        if not isinstance(shape, (tuple, list)) or len(shape) != 3:
            raise ValueError("cache['shape'] must be a length-3 tuple/list")
        return int(shape[0]), int(shape[1]), int(shape[2])

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
