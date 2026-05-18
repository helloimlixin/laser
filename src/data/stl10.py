from torch.utils.data import ConcatDataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import STL10
import lightning as pl
import torch

from src.data.config import DataConfig


class STL10DataModule(pl.LightningDataModule):
    def __init__(self, config: DataConfig = None):
        super().__init__()
        if config is None:
            config = DataConfig(
                dataset="stl10",
                data_dir="../data/stl10",
                batch_size=64,
                num_workers=4,
                image_size=96,
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
                augment=True,
            )
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _loader_generator(self, offset: int = 0) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(int(self.config.seed) + int(offset))
        return generator

    def prepare_data(self):
        STL10(self.config.data_dir, split="train", download=True)
        STL10(self.config.data_dir, split="test", download=True)
        STL10(self.config.data_dir, split="unlabeled", download=True)

    def setup(self, stage=None):
        image_size = (
            int(self.config.image_size)
            if isinstance(self.config.image_size, int)
            else int(self.config.image_size[0])
        )
        resize_ops = []
        if image_size != 96:
            resize_ops.append(transforms.Resize((image_size, image_size), antialias=True))

        train_ops = list(resize_ops)
        if bool(self.config.augment):
            train_ops.append(transforms.RandomHorizontalFlip())
        train_ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.config.mean, self.config.std),
            ]
        )
        eval_ops = list(resize_ops) + [
            transforms.ToTensor(),
            transforms.Normalize(self.config.mean, self.config.std),
        ]

        train_transform = transforms.Compose(train_ops)
        eval_transform = transforms.Compose(eval_ops)

        train_split = STL10(
            self.config.data_dir,
            split="train",
            transform=train_transform,
            download=False,
        )
        unlabeled_split = STL10(
            self.config.data_dir,
            split="unlabeled",
            transform=train_transform,
            download=False,
        )
        test_split = STL10(
            self.config.data_dir,
            split="test",
            transform=eval_transform,
            download=False,
        )

        self.train_dataset = ConcatDataset([train_split, unlabeled_split])
        self.val_dataset = test_split
        self.test_dataset = test_split

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            generator=self._loader_generator(0),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            generator=self._loader_generator(1),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            generator=self._loader_generator(2),
        )
