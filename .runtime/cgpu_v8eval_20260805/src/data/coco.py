import os
from pathlib import Path
from typing import List, Optional, Tuple

import lightning as pl
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from src.data.celeba import FlatImageDataset, IMG_EXTENSIONS
from src.data.config import DataConfig


def _list_flat_image_paths(split_dir: Path) -> List[Path]:
    """List COCO split images without recursive pathlib stat calls."""
    paths: List[Path] = []
    with os.scandir(split_dir) as entries:
        for entry in entries:
            suffix = Path(entry.name).suffix.lower()
            if suffix not in IMG_EXTENSIONS:
                continue
            try:
                if not entry.is_file(follow_symlinks=False):
                    continue
            except OSError:
                continue
            paths.append(Path(entry.path))
    paths.sort()
    if not paths:
        raise RuntimeError(f"No COCO images found under {split_dir}")
    return paths


class COCODataModule(pl.LightningDataModule):
    """COCO image datamodule for train2017/val2017 style directory layouts."""

    def __init__(self, config: DataConfig):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _loader_generator(self, offset: int = 0) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(int(self.config.seed) + int(offset))
        return generator

    def _resolve_data_dir(self) -> Path:
        candidates = [
            Path(str(self.config.data_dir)).expanduser(),
            Path("/scratch/xl598/data/coco"),
            Path("/scratch/xl598/datasets/coco"),
            Path("/scratch/xl598/datasets/COCO"),
        ]
        for candidate in candidates:
            train_dir = candidate / "train2017"
            val_dir = candidate / "val2017"
            if train_dir.is_dir() and val_dir.is_dir():
                return candidate
        tried = ", ".join(str(path) for path in candidates)
        raise RuntimeError(
            "COCO data not found. Expected train2017/ and val2017/ under one of: "
            f"{tried}"
        )

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if self.train_dataset is not None:
            return

        root = self._resolve_data_dir()
        image_size = self.config.image_size
        if isinstance(image_size, int):
            resize_to: Tuple[int, int] = (int(image_size), int(image_size))
        else:
            resize_to = (int(image_size[0]), int(image_size[1]))

        train_ops = [transforms.Resize(resize_to)]
        if bool(self.config.augment):
            train_ops.append(transforms.RandomHorizontalFlip())
        train_ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.config.mean, self.config.std),
            ]
        )
        eval_ops = [
            transforms.Resize(resize_to),
            transforms.ToTensor(),
            transforms.Normalize(self.config.mean, self.config.std),
        ]

        train_dir = root / "train2017"
        val_dir = root / "val2017"
        train_paths = _list_flat_image_paths(train_dir)
        val_paths = _list_flat_image_paths(val_dir)
        self.train_dataset = FlatImageDataset(
            train_dir,
            transform=transforms.Compose(train_ops),
            paths=train_paths,
        )
        self.val_dataset = FlatImageDataset(
            val_dir,
            transform=transforms.Compose(eval_ops),
            paths=val_paths,
        )
        self.test_dataset = FlatImageDataset(
            val_dir,
            transform=transforms.Compose(eval_ops),
            paths=val_paths,
        )

    def train_dataloader(self):
        return self._build_loader(
            dataset=self.train_dataset,
            batch_size=int(self.config.batch_size),
            shuffle=True,
            num_workers=int(self.config.num_workers),
            seed_offset=0,
        )

    def val_dataloader(self):
        val_workers = min(2, int(self.config.num_workers)) if int(self.config.num_workers) > 0 else 0
        return self._build_loader(
            dataset=self.val_dataset,
            batch_size=int(self.config.batch_size),
            shuffle=False,
            num_workers=val_workers,
            seed_offset=1,
        )

    def test_dataloader(self):
        test_workers = min(2, int(self.config.num_workers)) if int(self.config.num_workers) > 0 else 0
        return self._build_loader(
            dataset=self.test_dataset,
            batch_size=int(self.config.batch_size),
            shuffle=False,
            num_workers=test_workers,
            seed_offset=2,
        )

    def _build_loader(self, dataset, batch_size: int, shuffle: bool, num_workers: int, seed_offset: int):
        if dataset is None:
            return None
        kwargs = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=(num_workers > 0),
            generator=self._loader_generator(seed_offset),
            timeout=0,
        )
        if num_workers > 0:
            kwargs["prefetch_factor"] = 2
        else:
            kwargs["timeout"] = 0
        return DataLoader(**kwargs)
