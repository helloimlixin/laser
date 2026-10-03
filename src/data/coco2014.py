"""Official COCO2014 image/caption splits, from ZIPs or extracted images."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import zipfile

import lightning as pl
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data.config import DataConfig


class COCO2014Dataset(Dataset):
    """Return (RGB image, caption); never substitute images or mix splits.

    ``first`` visits each image once using its lowest-ID caption. ``all`` visits
    every annotation, so an image with five captions appears five times. Caption
    text and original image bytes are preserved. ZIP handles are worker-local.
    """

    def __init__(self, root, split="train", *, transform=None, caption_mode="first", max_items=0):
        self.root = Path(root).expanduser().resolve()
        if split not in {"train", "val"}:
            raise ValueError("COCO2014 caption splits are 'train' and 'val', not the hidden test set.")
        if caption_mode not in {"first", "all"}:
            raise ValueError("caption_mode must be 'first' or 'all'")
        self.split = split
        self.transform = transform
        self.caption_mode = caption_mode
        self.split_name = f"{split}2014"
        annotation_path = self.root / "annotations" / f"captions_{self.split_name}.json"
        with annotation_path.open() as f:
            payload = json.load(f)
        self.images = sorted(payload["images"], key=lambda im: int(im["id"]))
        by_id = {im["id"]: idx for idx, im in enumerate(self.images)}
        if not self.images or len(by_id) != len(self.images):
            raise ValueError(f"Empty or duplicate image IDs in {annotation_path}")
        self.captions = {image_id: [] for image_id in by_id}
        annotation_ids = set()
        for ann in sorted(payload["annotations"], key=lambda ann: int(ann["id"])):
            if ann["image_id"] not in by_id or not isinstance(ann["caption"], str) or not ann["caption"].strip():
                raise ValueError(f"Invalid caption annotation {ann['id']} in {annotation_path}")
            if ann["id"] in annotation_ids:
                raise ValueError(f"Duplicate caption ID {ann['id']} in {annotation_path}")
            annotation_ids.add(ann["id"])
            self.captions[ann["image_id"]].append(ann["caption"])
        self.samples = []
        for idx, im in enumerate(self.images):
            if Path(im["file_name"]).name != im["file_name"]:
                raise ValueError(f"Expected a plain COCO image filename: {im['file_name']}")
            captions = self.captions[im["id"]]
            if not captions:
                raise ValueError(f"Image {im['id']} has no caption in {annotation_path}")
            self.samples.extend((idx, caption) for caption in (captions if caption_mode == "all" else captions[:1]))
        if max_items > 0:
            self.samples = self.samples[:int(max_items)]
        self.archive_path = self.root / f"{self.split_name}.zip"
        self.image_dir = self.root / self.split_name
        self.use_zip = not self.image_dir.is_dir()
        if self.use_zip:
            with zipfile.ZipFile(self.archive_path) as z:
                names = set(z.namelist())
            missing = [self._member(idx) for idx, _ in self.samples if self._member(idx) not in names]
        else:
            names = set(os.listdir(self.image_dir))
            missing = [self.images[idx]["file_name"] for idx, _ in self.samples
                       if self.images[idx]["file_name"] not in names]
        if missing:
            raise FileNotFoundError(f"Missing COCO image: {missing[0]}")
        self.image_paths = [
            f"{self.archive_path}::{self._member(idx)}" if self.use_zip
            else str(self.image_dir / self.images[idx]["file_name"])
            for idx, _ in self.samples
        ]
        self._zip = None
        self._pid = None

    def _member(self, idx):
        return f"{self.split_name}/{self.images[idx]['file_name']}"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        idx, caption = self.samples[index]
        if self.use_zip:
            if self._zip is None or self._pid != os.getpid():
                self.close()
                self._zip = zipfile.ZipFile(self.archive_path)
                self._pid = os.getpid()
            source = io.BytesIO(self._zip.read(self._member(idx)))
        else:
            source = self.image_dir / self.images[idx]["file_name"]
        with Image.open(source) as image:
            rgb = image.convert("RGB")
        return (self.transform(rgb) if self.transform is not None else rgb), caption

    def close(self):
        if getattr(self, "_zip", None) is not None:
            self._zip.close()
        self._zip = None
        self._pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_zip=None, _pid=None)
        return state

    def __del__(self):
        self.close()


class COCO2014DataModule(pl.LightningDataModule):
    def __init__(self, config: DataConfig):
        super().__init__()
        self.config = config
        self.train_dataset = self.val_dataset = self.test_dataset = None

    def _transform(self, training):
        size = self.config.image_size
        size = size if isinstance(size, int) else tuple(size)
        ops = [transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
               transforms.CenterCrop(size)]
        if training and self.config.augment:
            ops.append(transforms.RandomHorizontalFlip())
        return transforms.Compose(ops + [transforms.ToTensor(),
                                  transforms.Normalize(self.config.mean, self.config.std)])

    def setup(self, stage=None):
        if self.train_dataset is not None:
            return
        self.train_dataset = COCO2014Dataset(
            self.config.data_dir, "train", transform=self._transform(True),
            caption_mode=self.config.coco_caption_mode, max_items=self.config.max_items,
        )
        self.val_dataset = COCO2014Dataset(
            self.config.data_dir, "val", transform=self._transform(False),
            caption_mode="first", max_items=self.config.max_items,
        )
        # There are no public test captions: test_dataloader evaluates val2014.
        self.test_dataset = self.val_dataset

    def _loader(self, dataset, training):
        workers = int(self.config.num_workers)
        kwargs = dict(
            batch_size=int(self.config.batch_size if training else self.config.eval_batch_size or self.config.batch_size),
            shuffle=training, num_workers=workers, pin_memory=self.config.pin_memory,
            persistent_workers=workers > 0,
            generator=torch.Generator().manual_seed(int(self.config.seed) + (0 if training else 1)),
        )
        if workers > 0 and self.config.prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(self.config.prefetch_factor)
        return DataLoader(dataset, **kwargs)

    def train_dataloader(self):
        return self._loader(self.train_dataset, True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, False)
