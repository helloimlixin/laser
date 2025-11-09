import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union, List

import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms as transforms

from src.data.config import DataConfig


IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


class FlatImageDataset(Dataset):
    def __init__(self, root: Union[str, Path], transform=None):
        self.root = Path(root)
        self.transform = transform
        image_paths: List[Path] = []
        for path in self.root.rglob('*'):
            if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS:
                image_paths.append(path)
        image_paths.sort()
        if not image_paths:
            raise RuntimeError(f'No images found under {self.root}')
        self.image_paths = image_paths
        self._placeholder_size = (64, 64)  # used only as a last-resort fallback

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        # Robust image loading: retry a few nearby indices if an image fails
        num_items = len(self.image_paths)
        attempts = 0
        last_exc = None
        while attempts < 5:
            path = self.image_paths[index % num_items]
            try:
                with Image.open(path) as img:
                    img = img.convert('RGB')
                if self.transform is not None:
                    img = self.transform(img)
                return img, 0
            except Exception as exc:
                last_exc = exc
                index += 1
                attempts += 1
                continue
        # Last-resort fallback: return a black image of nominal size (transforms will resize/normalize)
        try:
            img = Image.new('RGB', self._placeholder_size, (0, 0, 0))
            if self.transform is not None:
                img = self.transform(img)
            return img, 0
        except Exception:
            # Re-raise the original exception if even placeholder fails (should be extremely rare)
            raise last_exc


class CelebADataModule(pl.LightningDataModule):
    def __init__(self, config: DataConfig):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _resolve_data_dir(self) -> str:
        """Resolve a directory that actually contains CelebA images."""
        def has_images(path_str: str) -> bool:
            if not path_str:
                return False
            p = Path(path_str)
            if not p.exists() or not p.is_dir():
                return False
            # quickly check if any image exists
            for ext in IMG_EXTENSIONS:
                if any(p.rglob(f'*{ext}')):
                    return True
            return False

        # Candidate directories in order of preference
        candidates = [
            str(self.config.data_dir) if getattr(self.config, "data_dir", "") else "",
            os.environ.get('CELEBA_DIR', ''),
            '/home/xl598/Data/celeba/img_align_celeba',
            '/home/xl598/Data/celeba',
            str(Path.cwd() / 'data' / 'celeba'),
            str(Path(__file__).resolve().parents[2] / 'data' / 'celeba'),
        ]

        for c in candidates:
            if has_images(c):
                return c

        # If none matched, raise with guidance
        raise RuntimeError(
            "CelebA images not found. Set CELEBA_DIR to your images folder "
            "(e.g., /home/xl598/Data/celeba/img_align_celeba) or update data.data_dir."
        )

    def prepare_data(self):
        # No automatic download for CelebA here; rely on local images folder
        # Users should set CELEBA_DIR or data.data_dir to the images directory.
        pass

    def setup(self, stage=None):
        data_dir = self._resolve_data_dir()
        augment = self.config.augment

        # Transforms: center-crop if the images are 178x218, then resize to configured size.
        # Use Normalize with config mean/std.
        image_size = self.config.image_size
        if isinstance(image_size, int):
            resize_to: Tuple[int, int] = (image_size, image_size)
        else:
            resize_to = tuple(image_size)  # type: ignore[arg-type]

        train_transforms = transforms.Compose([
            transforms.Resize(resize_to),
            transforms.RandomHorizontalFlip() if augment else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(self.config.mean, self.config.std),
        ])

        eval_transforms = transforms.Compose([
            transforms.Resize(resize_to),
            transforms.ToTensor(),
            transforms.Normalize(self.config.mean, self.config.std),
        ])

        full_dataset = FlatImageDataset(data_dir, transform=train_transforms)

        # Simple split: 90% train, 5% val, 5% test
        num_items = len(full_dataset)
        num_train = int(0.90 * num_items)
        num_val = int(0.05 * num_items)
        num_test = num_items - num_train - num_val

        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            lengths=[num_train, num_val, num_test],
            generator=torch.Generator().manual_seed(42),
        )

        # Override transforms for val/test
        val_dataset.dataset.transform = eval_transforms  # type: ignore[attr-defined]
        test_dataset.dataset.transform = eval_transforms  # type: ignore[attr-defined]

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=True,
            timeout=120,
            multiprocessing_context='spawn',
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,  # single-process to avoid worker aborts during validation
            pin_memory=False,
            persistent_workers=True,
            timeout=120,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,  # single-process to avoid worker aborts during test
            pin_memory=False,
            persistent_workers=True,
            timeout=120,
        )


