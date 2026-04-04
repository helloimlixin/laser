import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple, Union, List

import lightning as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms as transforms

from src.data.config import DataConfig


IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def _packed_celeba_file(root: Union[str, Path], image_size: int) -> Path:
    return Path(root) / f"celeba_{int(image_size)}x{int(image_size)}_rgb_uint8.npy"


def _is_readable_rgb_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.convert("RGB").load()
        return True
    except Exception:
        return False


def _filter_readable_paths(paths: List[Path]) -> List[Path]:
    if not paths:
        return []
    workers = min(32, max(4, (os.cpu_count() or 4) * 2))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        ok = list(pool.map(_is_readable_rgb_image, paths))
    return [p for p, keep in zip(paths, ok) if keep]


class FlatImageDataset(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        transform=None,
        paths: Optional[List[Path]] = None,
    ):
        self.root = Path(root)
        self.transform = transform
        if paths is not None:
            if not paths:
                raise RuntimeError("No image paths provided for dataset subset.")
            self.image_paths = [Path(p) for p in paths]
        else:
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
        # Robust image loading: skip over unreadable images instead of injecting black placeholders
        num_items = len(self.image_paths)
        attempts = 0
        failures = []
        last_exc = None
        while attempts < min(256, num_items):
            path = self.image_paths[index % num_items]
            try:
                with Image.open(path) as img:
                    img = img.convert('RGB')
                if self.transform is not None:
                    img = self.transform(img)
                return img, 0
            except Exception as exc:
                failures.append(str(path))
                last_exc = exc
                index += 1
                attempts += 1
                continue
        warnings.warn(
            f"Failed to load images after {attempts} attempts. "
            f"Last error on {failures[-1] if failures else 'unknown'}: {last_exc}"
        )
        raise RuntimeError(f"Unrecoverable image loading failure for: {failures}") from last_exc


class PackedRGBImageDataset(Dataset):
    """Read resized RGB uint8 images from a single NumPy memmap file."""

    def __init__(
        self,
        root: Union[str, Path],
        image_size: int,
        *,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        random_horizontal_flip: bool = False,
        indices: Optional[List[int]] = None,
    ):
        self.root = Path(root)
        self.image_size = int(image_size)
        self.random_horizontal_flip = bool(random_horizontal_flip)
        self.packed_path = _packed_celeba_file(self.root, self.image_size)
        if not self.packed_path.exists():
            raise FileNotFoundError(f"Packed dataset not found: {self.packed_path}")

        self.images = np.load(self.packed_path, mmap_mode="r")
        expected_shape = (self.image_size, self.image_size, 3)
        if self.images.ndim != 4 or tuple(self.images.shape[1:]) != expected_shape:
            raise ValueError(
                f"Packed dataset at {self.packed_path} has shape {tuple(self.images.shape)}; "
                f"expected [N, {self.image_size}, {self.image_size}, 3]"
            )
        if self.images.dtype != np.uint8:
            raise ValueError(
                f"Packed dataset at {self.packed_path} has dtype {self.images.dtype}; expected uint8"
            )

        self.indices = list(indices) if indices is not None else None
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        if self.indices is not None:
            return len(self.indices)
        return int(self.images.shape[0])

    def __getitem__(self, idx: int):
        if self.indices is not None:
            idx = int(self.indices[idx])
        image = torch.from_numpy(np.array(self.images[idx], copy=True)).permute(2, 0, 1).float().div_(255.0)
        if self.random_horizontal_flip and torch.rand((), dtype=torch.float32).item() < 0.5:
            image = torch.flip(image, dims=[2])
        image = (image - self.mean) / self.std
        return image, 0


class CelebADataModule(pl.LightningDataModule):
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

    def _packed_image_size(self) -> Optional[int]:
        image_size = self.config.image_size
        if isinstance(image_size, int):
            return int(image_size)
        if len(image_size) == 2 and int(image_size[0]) == int(image_size[1]):
            return int(image_size[0])
        return None

    def _resolve_data_dir(self) -> str:
        """Resolve a directory that contains raw CelebA images or a packed npy file."""

        packed_image_size = self._packed_image_size()

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

        def has_packed(path_str: str) -> bool:
            if not path_str or packed_image_size is None:
                return False
            p = Path(path_str)
            if not p.exists() or not p.is_dir():
                return False
            return _packed_celeba_file(p, packed_image_size).exists()

        # Candidate directories in order of preference
        candidates = [
            str(self.config.data_dir) if getattr(self.config, "data_dir", "") else "",
            os.environ.get('CELEBA_DIR', ''),
            '/home/xl598/Data/celeba/img_align_celeba',
            '/home/xl598/Data/celeba',
            str((Path.cwd() / '..' / 'data' / 'celeba').resolve()),
            str(Path(__file__).resolve().parents[3] / 'data' / 'celeba'),
        ]

        for c in candidates:
            if has_packed(c) or has_images(c):
                return c

        # If none matched, raise with guidance
        raise RuntimeError(
            "CelebA data not found. Set CELEBA_DIR/data.data_dir to either a raw images folder "
            "(e.g., /home/xl598/Data/celeba/img_align_celeba) or a directory containing "
            f"celeba_{packed_image_size}x{packed_image_size}_rgb_uint8.npy."
        )

    def prepare_data(self):
        # No automatic download for CelebA here; rely on local images folder
        # Users should set CELEBA_DIR or data.data_dir to the images directory.
        pass

    def setup(self, stage=None):
        if self.train_dataset is not None:
            return
        data_dir = self._resolve_data_dir()
        augment = self.config.augment

        # Transforms: center-crop if the images are 178x218, then resize to configured size.
        # Use Normalize with config mean/std.
        image_size = self.config.image_size
        if isinstance(image_size, int):
            resize_to: Tuple[int, int] = (image_size, image_size)
        else:
            resize_to = tuple(image_size)  # type: ignore[arg-type]

        packed_image_size = self._packed_image_size()
        packed_path = (
            _packed_celeba_file(data_dir, packed_image_size)
            if packed_image_size is not None
            else None
        )

        if packed_path is not None and packed_path.exists():
            print(f"[Data] using packed CelebA dataset at {packed_path}")
            num_items = int(np.load(packed_path, mmap_mode="r").shape[0])
            if num_items < 3:
                raise RuntimeError("CelebA packed dataset must contain at least three images for train/val/test splits.")

            generator = torch.Generator().manual_seed(int(self.config.seed))
            indices = torch.randperm(num_items, generator=generator)
            num_train = int(0.90 * num_items)
            num_val = int(0.05 * num_items)
            train_idx = indices[:num_train].tolist()
            val_idx = indices[num_train:num_train + num_val].tolist()
            test_idx = indices[num_train + num_val:].tolist()

            mean = tuple(float(x) for x in self.config.mean)
            std = tuple(float(x) for x in self.config.std)

            def _make_packed(indices, flip):
                return PackedRGBImageDataset(
                    data_dir, packed_image_size,
                    mean=mean, std=std,
                    random_horizontal_flip=flip, indices=indices,
                )

            self.train_dataset = _make_packed(train_idx, flip=augment)
            self.val_dataset = _make_packed(val_idx, flip=False)
            self.test_dataset = _make_packed(test_idx, flip=False)
            return

        train_ops = [transforms.Resize(resize_to)]
        if augment:
            train_ops.append(transforms.RandomHorizontalFlip())
        train_ops.extend([
            transforms.ToTensor(),
            transforms.Normalize(self.config.mean, self.config.std),
        ])
        train_transforms = transforms.Compose(train_ops)

        eval_transforms = transforms.Compose([
            transforms.Resize(resize_to),
            transforms.ToTensor(),
            transforms.Normalize(self.config.mean, self.config.std),
        ])

        base_dataset = FlatImageDataset(data_dir, transform=None)
        raw_paths = base_dataset.image_paths
        valid_paths = _filter_readable_paths(raw_paths)
        dropped = len(raw_paths) - len(valid_paths)
        if dropped:
            warnings.warn(
                f"CelebA: skipped {dropped} unreadable or corrupt file(s) under {data_dir}. "
                f"Re-download or repair those images if you need the full dataset."
            )
        if len(valid_paths) < 3:
            raise RuntimeError(
                f"Fewer than 3 readable images in {data_dir} after validation "
                f"({len(valid_paths)} left, {dropped} dropped)."
            )
        base_dataset = FlatImageDataset(data_dir, transform=None, paths=valid_paths)
        num_items = len(base_dataset)
        num_train = int(0.90 * num_items)
        num_val = int(0.05 * num_items)
        num_test = num_items - num_train - num_val

        if num_items < 3:
            raise RuntimeError("CelebA dataset must contain at least three images for train/val/test splits.")

        generator = torch.Generator().manual_seed(int(self.config.seed))
        indices = torch.randperm(num_items, generator=generator)
        train_idx = indices[:num_train]
        val_idx = indices[num_train:num_train + num_val]
        test_idx = indices[num_train + num_val:]

        def _gather_paths(idxs: torch.Tensor) -> List[Path]:
            return [base_dataset.image_paths[i] for i in idxs.tolist()]

        train_paths = _gather_paths(train_idx)
        val_paths = _gather_paths(val_idx)
        test_paths = _gather_paths(test_idx)

        self.train_dataset = FlatImageDataset(
            data_dir,
            transform=train_transforms,
            paths=train_paths,
        )
        self.val_dataset = FlatImageDataset(
            data_dir,
            transform=eval_transforms,
            paths=val_paths,
        )
        self.test_dataset = FlatImageDataset(
            data_dir,
            transform=eval_transforms,
            paths=test_paths,
        )
    def train_dataloader(self):
        return self._build_loader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            seed_offset=0,
        )

    def val_dataloader(self):
        val_workers = min(2, self.config.num_workers) if self.config.num_workers > 0 else 0
        return self._build_loader(
            dataset=self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=val_workers,
            seed_offset=1,
        )

    def test_dataloader(self):
        test_workers = min(2, self.config.num_workers) if self.config.num_workers > 0 else 0
        return self._build_loader(
            dataset=self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=test_workers,
            seed_offset=2,
        )

    def _build_loader(self, dataset, batch_size, shuffle, num_workers, seed_offset):
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
        )
        if num_workers > 0:
            kwargs['timeout'] = 120
            kwargs['multiprocessing_context'] = 'spawn'
        else:
            kwargs['timeout'] = 0
        return DataLoader(**kwargs)
