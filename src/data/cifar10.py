import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import lightning as pl
from src.data.config import DataConfig

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, config: DataConfig = None):
        super().__init__()
        if config is None:
            config = DataConfig(
                dataset='cifar10',
                data_dir='./data',
                batch_size=128,
                num_workers=4,
                image_size=32,
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
                augment=True
            )
        self.config = config
        self.cifar_train = None
        self.cifar_val = None  # Rename this variable

    def prepare_data(self):
        """Download data if needed"""
        CIFAR10(self.config.data_dir, train=True, download=True)
        CIFAR10(self.config.data_dir, train=False, download=True)

    def setup(self, stage=None):
        """Setup datasets"""
        # Define transforms
        if self.config.augment:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.config.mean, self.config.std)
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.config.mean, self.config.std)
            ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.config.mean, self.config.std)
        ])
        
        # Setup training data
        self.cifar_train = CIFAR10(
            self.config.data_dir,
            train=True,
            transform=train_transform
        )
        
        # Setup validation data - using test set for validation
        self.cifar_val = CIFAR10(  # Changed from cifar_test to cifar_val
            self.config.data_dir,
            train=False,
            transform=test_transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.cifar_train,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.cifar_val,  # Changed from cifar_test to cifar_val
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return self.val_dataloader()
