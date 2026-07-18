"""CC3M WebDataset-style tar datamodule with caption strings."""

from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile
from typing import Optional, Sequence

import lightning as pl
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from src.data.config import DataConfig


CC3M_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _as_hw(value) -> tuple[int, int]:
    if isinstance(value, int):
        return int(value), int(value)
    return int(value[0]), int(value[1])


def _candidate_roots(data_dir: str | Path) -> list[Path]:
    raw = Path(str(data_dir)).expanduser()
    return [
        raw,
        raw / "wds",
        raw / "webdataset",
        Path("/workspace/Projects/data/cc3m"),
        Path("/workspace/Projects/data/cc3m/wds"),
        Path(f"/scratch/{Path.home().name}/data/cc3m"),
        Path(f"/scratch/{Path.home().name}/data/cc3m/wds"),
        Path(f"/scratch/{Path.home().name}/datasets/cc3m"),
        Path(f"/scratch/{Path.home().name}/datasets/cc3m/wds"),
    ]


def _find_shards(data_dir: str | Path, split: str, *, allow_fallback: bool = True) -> list[Path]:
    split = str(split or "train").strip().lower()
    roots = _candidate_roots(data_dir)
    split_patterns = {
        "train": ("*train*.tar", "*.tar"),
        "val": ("*val*.tar", "*valid*.tar", "*validation*.tar"),
        "test": ("*test*.tar",),
    }
    patterns = split_patterns.get(split, (f"*{split}*.tar",))

    for root in roots:
        if root.is_file() and root.suffix == ".tar":
            return [root]
        if not root.is_dir():
            continue
        for pattern in patterns:
            shards = sorted(path for path in root.glob(pattern) if path.is_file())
            if shards:
                return shards

    if allow_fallback and split != "train":
        return _find_shards(data_dir, "train", allow_fallback=False)

    tried = ", ".join(str(path) for path in roots)
    raise RuntimeError(f"CC3M {split} shards not found. Tried: {tried}")


def _record_key(member_name: str) -> str:
    path = Path(member_name)
    return str(path.with_suffix(""))


def _scan_shards(shards: Sequence[Path], *, max_items: int = 0) -> list[tuple[str, str, str, str, str]]:
    records: list[tuple[str, str, str, str, str]] = []
    cap = max(0, int(max_items or 0))
    for shard in shards:
        grouped: dict[str, dict[str, str]] = {}
        with tarfile.open(shard, "r:*") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                suffix = Path(member.name).suffix.lower()
                key = _record_key(member.name)
                item = grouped.setdefault(key, {})
                if suffix in CC3M_IMAGE_EXTENSIONS:
                    item.setdefault("image", member.name)
                elif suffix == ".txt":
                    item.setdefault("text", member.name)
                elif suffix == ".json":
                    item.setdefault("json", member.name)

        shard_text = str(Path(shard).expanduser().resolve())
        for key in sorted(grouped):
            item = grouped[key]
            image = item.get("image", "")
            if not image:
                continue
            text = item.get("text", "")
            meta = item.get("json", "")
            if not text and not meta:
                continue
            records.append((shard_text, image, text, meta, key))
            if cap > 0 and len(records) >= cap:
                return records
    return records


class CC3MDataset(Dataset):
    """Map-style reader for CC3M shards containing image/caption pairs."""

    def __init__(
        self,
        records: Sequence[tuple[str, str, str, str, str]],
        *,
        transform=None,
        fallback_size: tuple[int, int] = (256, 256),
    ):
        self.records = list(records)
        if not self.records:
            raise RuntimeError("CC3M dataset received no image/caption records.")
        self.transform = transform
        self.fallback_size = (int(fallback_size[0]), int(fallback_size[1]))
        self.image_paths = [f"{shard}::{image}" for shard, image, _, _, _ in self.records]
        self._tar_path: str | None = None
        self._tar: tarfile.TarFile | None = None

    def __len__(self) -> int:
        return len(self.records)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_tar_path"] = None
        state["_tar"] = None
        return state

    def _tarfile(self, shard: str) -> tarfile.TarFile:
        if self._tar is None or self._tar_path != shard:
            if self._tar is not None:
                self._tar.close()
            self._tar_path = shard
            self._tar = tarfile.open(shard, "r:*")
        return self._tar

    def _read_member(self, shard: str, member: str) -> bytes:
        tf = self._tarfile(shard)
        extracted = tf.extractfile(member)
        if extracted is None:
            raise RuntimeError(f"Could not read {member!r} from {shard}")
        return extracted.read()

    def _read_caption(self, shard: str, text_member: str, json_member: str) -> str:
        if text_member:
            return self._read_member(shard, text_member).decode("utf-8", errors="replace").strip()
        if json_member:
            raw = self._read_member(shard, json_member).decode("utf-8", errors="replace")
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                return ""
            return str(payload.get("caption", "") or "").strip()
        return ""

    def _read_image(self, shard: str, image_member: str) -> Image.Image:
        try:
            raw = self._read_member(shard, image_member)
            with Image.open(io.BytesIO(raw)) as img:
                return img.convert("RGB")
        except (OSError, UnidentifiedImageError, RuntimeError):
            return Image.new("RGB", self.fallback_size, color=(0, 0, 0))

    def __getitem__(self, idx: int):
        shard, image_member, text_member, json_member, _ = self.records[int(idx)]
        image = self._read_image(shard, image_member)
        caption = self._read_caption(shard, text_member, json_member)
        if self.transform is not None:
            image = self.transform(image)
        return image, caption

    def __del__(self):
        try:
            if self._tar is not None:
                self._tar.close()
        except Exception:
            pass


class CC3MDataModule(pl.LightningDataModule):
    """Lightning datamodule for CC3M shards laid out as ``*.jpg``/``*.txt`` tars."""

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

    def _resize_to(self) -> tuple[int, int]:
        return _as_hw(self.config.image_size)

    def _train_transform(self):
        ops = [transforms.Resize(self._resize_to())]
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

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if self.train_dataset is not None:
            return

        max_items = max(0, int(getattr(self.config, "max_items", 0) or 0))
        train_records = _scan_shards(
            _find_shards(self.config.data_dir, "train"),
            max_items=max_items,
        )
        if not train_records:
            raise RuntimeError(f"No CC3M training records found under {self.config.data_dir}")

        eval_cap = max(1, min(len(train_records), 1000))
        try:
            val_records = _scan_shards(_find_shards(self.config.data_dir, "val", allow_fallback=False), max_items=eval_cap)
        except RuntimeError:
            val_records = train_records[:eval_cap]
        try:
            test_records = _scan_shards(_find_shards(self.config.data_dir, "test", allow_fallback=False), max_items=eval_cap)
        except RuntimeError:
            test_records = val_records

        self.train_dataset = CC3MDataset(
            train_records,
            transform=self._train_transform(),
            fallback_size=self._resize_to(),
        )
        self.val_dataset = CC3MDataset(
            val_records,
            transform=self._eval_transform(),
            fallback_size=self._resize_to(),
        )
        self.test_dataset = CC3MDataset(
            test_records,
            transform=self._eval_transform(),
            fallback_size=self._resize_to(),
        )

    def train_dataloader(self):
        return self._build_loader(
            self.train_dataset,
            batch_size=int(self.config.batch_size),
            shuffle=False,
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

    def _eval_batch_size(self) -> int:
        raw = getattr(self.config, "eval_batch_size", None)
        if raw is None or int(raw) <= 0:
            return int(self.config.batch_size)
        return int(raw)

    def _build_loader(self, dataset, *, batch_size: int, shuffle: bool, seed_offset: int):
        if dataset is None:
            return None
        num_workers = int(self.config.num_workers)
        pin_memory = bool(getattr(self.config, "pin_memory", False))
        prefetch_factor = getattr(self.config, "prefetch_factor", 2)
        kwargs = dict(
            dataset=dataset,
            batch_size=int(batch_size),
            shuffle=bool(shuffle),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            generator=self._loader_generator(seed_offset),
            timeout=0,
        )
        if num_workers > 0:
            try:
                prefetch_factor = int(prefetch_factor)
            except (TypeError, ValueError):
                prefetch_factor = 2
            if prefetch_factor > 0:
                kwargs["prefetch_factor"] = prefetch_factor
        return DataLoader(**kwargs)
