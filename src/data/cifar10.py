import os
import lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataclasses import dataclass
from typing import List
from omegaconf import DictConfig

@dataclass
class CIFAR10Config:
    data_dir: str
    batch_size: int
    num_workers: int
    image_size: int
    mean: List[float]
    std: List[float]

    @staticmethod
    def from_dict(config: DictConfig) -> 'CIFAR10Config':
        return CIFAR10Config(
            data_dir=config.data_dir,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            image_size=config.image_size,
            mean=config.mean,
            std=config.std
        )

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, config: CIFAR10Config):
        super().__init__()
        self.config = config
        self.transform = None
        self.train_dataset = None
        self.val_dataset = None
        
        # Create data directory if it doesn't exist
        os.makedirs(self.config.data_dir, exist_ok=True)

    def setup_transforms(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.config.mean, self.config.std)
        ])

    def setup(self, stage=None):
        if self.transform is None:
            self.setup_transforms()

        if stage == 'fit' or stage is None:
            try:
                self.train_dataset = datasets.CIFAR10(
                    root=self.config.data_dir,
                    train=True,
                    download=True,
                    transform=self.transform
                )
                
                self.val_dataset = datasets.CIFAR10(
                    root=self.config.data_dir,
                    train=False,
                    download=True,
                    transform=self.transform
                )
                print(f"CIFAR10 dataset loaded successfully: {len(self.train_dataset)} training samples, "
                      f"{len(self.val_dataset)} validation samples")
            except Exception as e:
                print(f"Error setting up CIFAR10 dataset: {str(e)}")
                raise

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        ) 