from pathlib import Path
from typing import Optional, Tuple

import lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

from src.data.celeba import FlatImageDataset
from src.data.config import DataConfig


PAPER_IMAGE_FOLDER_DATASETS = {"ffhq", "lsun_bedroom", "lsun_church", "lsun_cat"}


def normalize_image_folder_dataset_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


class ImageFolderDataModule(pl.LightningDataModule):
    """Generic RGB image-folder datamodule for paper-aligned image datasets.

    Supported layouts:
      root/train, root/val, optional root/test
      root/train/<class-or-category>, root/val/<class-or-category>
      a flat or recursive image root, split deterministically into 90/5/5
    """

    def __init__(self, config: DataConfig):
        super().__init__()
        self.config = config
        self.dataset_name = normalize_image_folder_dataset_name(config.dataset)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _loader_generator(self, offset: int = 0) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(int(self.config.seed) + int(offset))
        return generator

    def _resolve_data_dir(self) -> Path:
        raw = Path(str(self.config.data_dir)).expanduser()
        candidates = [raw]
        stem = self.dataset_name
        if stem.startswith("lsun_"):
            category = stem.removeprefix("lsun_")
            candidates.extend(
                [
                    Path(f"/scratch/{Path.home().name}/datasets/lsun") / category,
                    Path(f"/scratch/{Path.home().name}/data/lsun") / category,
                    Path(f"/scratch/{Path.home().name}/datasets") / stem,
                ]
            )
        else:
            candidates.extend(
                [
                    Path(f"/scratch/{Path.home().name}/datasets") / stem,
                    Path(f"/scratch/{Path.home().name}/data") / stem,
                ]
            )
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        tried = ", ".join(str(path) for path in candidates)
        raise RuntimeError(f"{self.dataset_name} data not found. Tried: {tried}")

    def _resize_to(self) -> Tuple[int, int]:
        return self._as_hw(self.config.image_size)

    @staticmethod
    def _as_hw(value) -> Tuple[int, int]:
        if isinstance(value, int):
            return int(value), int(value)
        return int(value[0]), int(value[1])

    def _train_crop_to(self) -> Optional[Tuple[int, int]]:
        crop_size = getattr(self.config, "train_crop_size", None)
        if crop_size is None:
            return None
        if isinstance(crop_size, int) and int(crop_size) <= 0:
            return None
        crop_to = self._as_hw(crop_size)
        if crop_to[0] <= 0 or crop_to[1] <= 0:
            return None
        resize_to = self._resize_to()
        if crop_to[0] > resize_to[0] or crop_to[1] > resize_to[1]:
            raise ValueError(f"train_crop_size={crop_to} cannot exceed image_size={resize_to}")
        return crop_to

    def _eval_batch_size(self) -> int:
        raw = getattr(self.config, "eval_batch_size", None)
        if raw is None or int(raw) <= 0:
            return int(self.config.batch_size)
        return int(raw)

    def _train_transform(self):
        ops = [transforms.Resize(self._resize_to())]
        train_crop = self._train_crop_to()
        if train_crop is not None and train_crop != self._resize_to():
            ops.append(transforms.RandomCrop(train_crop))
        if bool(self.config.augment):
            ops.append(transforms.RandomHorizontalFlip())
        ops.extend([transforms.ToTensor(), transforms.Normalize(self.config.mean, self.config.std)])
        return transforms.Compose(ops)

    def _eval_transform(self):
        return transforms.Compose(
            [
                transforms.Resize(self._resize_to()),
                transforms.ToTensor(),
                transforms.Normalize(self.config.mean, self.config.std),
            ]
        )

    @staticmethod
    def _first_existing_dir(root: Path, names: tuple[str, ...]) -> Optional[Path]:
        for name in names:
            candidate = root / name
            if candidate.is_dir():
                return candidate
        return None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if self.train_dataset is not None:
            return

        root = self._resolve_data_dir()
        train_dir = self._first_existing_dir(root, ("train", "training", "train256", "images_train"))
        val_dir = self._first_existing_dir(root, ("val", "valid", "validation", "test", "images_val"))
        test_dir = self._first_existing_dir(root, ("test", "testing"))

        if train_dir is not None and val_dir is not None:
            self.train_dataset = FlatImageDataset(train_dir, transform=self._train_transform())
            self.val_dataset = FlatImageDataset(val_dir, transform=self._eval_transform())
            self.test_dataset = FlatImageDataset(
                test_dir if test_dir is not None else val_dir,
                transform=self._eval_transform(),
            )
            return

        base_dataset = FlatImageDataset(root, transform=None)
        num_items = len(base_dataset)
        if num_items < 3:
            raise RuntimeError(f"{self.dataset_name} needs at least 3 images, found {num_items} under {root}.")

        num_train = int(0.90 * num_items)
        num_val = max(1, int(0.05 * num_items))
        num_test = num_items - num_train - num_val
        if num_test <= 0:
            num_train = max(1, num_train - 1)
            num_test = num_items - num_train - num_val

        generator = self._loader_generator(0)
        train_subset, val_subset, test_subset = random_split(
            base_dataset,
            [num_train, num_val, num_test],
            generator=generator,
        )
        paths = base_dataset.image_paths

        def _subset_dataset(subset, transform):
            subset_paths = [paths[int(i)] for i in subset.indices]
            return FlatImageDataset(root, transform=transform, paths=subset_paths)

        self.train_dataset = _subset_dataset(train_subset, self._train_transform())
        self.val_dataset = _subset_dataset(val_subset, self._eval_transform())
        self.test_dataset = _subset_dataset(test_subset, self._eval_transform())

    def train_dataloader(self):
        return self._build_loader(
            self.train_dataset,
            batch_size=int(self.config.batch_size),
            shuffle=True,
            seed_offset=0,
        )

    def val_dataloader(self):
        return self._build_loader(
            self.val_dataset,
            batch_size=self._eval_batch_size(),
            shuffle=False,
            seed_offset=1,
        )

    def test_dataloader(self):
        return self._build_loader(
            self.test_dataset,
            batch_size=self._eval_batch_size(),
            shuffle=False,
            seed_offset=2,
        )

    def _build_loader(self, dataset, *, batch_size: int, shuffle: bool, seed_offset: int):
        if dataset is None:
            return None
        num_workers = int(self.config.num_workers)
        kwargs = dict(
            dataset=dataset,
            batch_size=int(batch_size),
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=(num_workers > 0),
            generator=self._loader_generator(seed_offset),
            timeout=0,
        )
        if num_workers > 0:
            kwargs["prefetch_factor"] = 2
        return DataLoader(**kwargs)
