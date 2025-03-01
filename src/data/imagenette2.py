import os
from typing import Optional, Tuple, List
import lightning as pl
from torch.utils.data import DataLoader
from dataclasses import dataclass
import logging
from omegaconf import DictConfig

from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    dataset: str
    data_dir: str
    image_size: int
    batch_size: int
    num_workers: int
    mean: List[float]
    std: List[float]

    @staticmethod
    def from_dict(config: DictConfig) -> 'DataConfig':
        return DataConfig(
            dataset=config.dataset,
            data_dir=config.data_dir,
            image_size=config.image_size,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            mean=config.mean,
            std=config.std
        )

    def validate(self):
        """Validate config parameters"""
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {self.num_workers}")


class Imagenette2DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for Imagenette2 dataset.
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = DataConfig.from_dict(config)
        self.config.validate()
        self.train_dataset = None
        self.val_dataset = None
        self.transform = None

    def setup_transforms(self) -> None:
        """Initialize data transforms."""
        try:
            self.transform = Compose([
                Resize((self.config.image_size, self.config.image_size)),
                ToTensor(),
                Normalize(self.config.mean, self.config.std)
            ])
        except Exception as e:
            logger.error(f"Failed to setup transforms: {str(e)}")
            raise

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for training and validation."""
        if self.transform is None:
            self.setup_transforms()

        if stage == 'fit' or stage is None:
            train_dir = os.path.join(self.config.data_dir, 'train')
            val_dir = os.path.join(self.config.data_dir, 'val')

            if not os.path.exists(train_dir):
                raise FileNotFoundError(f"Training directory not found: {train_dir}")
            if not os.path.exists(val_dir):
                raise FileNotFoundError(f"Validation directory not found: {val_dir}")

            try:
                self.train_dataset = ImageFolder(
                    root=train_dir,
                    transform=self.transform
                )

                self.val_dataset = ImageFolder(
                    root=val_dir,
                    transform=self.transform
                )

                if len(self.train_dataset) == 0:
                    raise RuntimeError(f"No valid images found in training directory {train_dir}")
                if len(self.val_dataset) == 0:
                    raise RuntimeError(f"No valid images found in validation directory {val_dir}")

                logger.info(f"Dataset loaded: {len(self.train_dataset)} training samples, "
                            f"{len(self.val_dataset)} validation samples")

            except Exception as e:
                logger.error(f"Failed to create dataset: {str(e)}")
                raise

    def train_dataloader(self) -> DataLoader:
        """Create the training data loader."""
        if self.train_dataset is None:
            raise RuntimeError("Training dataset not initialized. Call setup() first.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=False
        )

    def val_dataloader(self) -> DataLoader:
        """Create the validation data loader."""
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset not initialized. Call setup() first.")

        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=False
        )
