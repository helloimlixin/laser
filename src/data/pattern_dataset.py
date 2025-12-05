"""
Dataset for extracting pattern indices from a pretrained LASER model.

This module provides a dataset that:
1. Loads images from an existing image dataset
2. Passes them through a pretrained LASER model with pattern quantization
3. Returns the pattern indices for training an autoregressive model

Usage:
    laser_model = LASER.load_from_checkpoint('path/to/checkpoint.ckpt')
    base_dataset = CIFAR10(...)
    pattern_dataset = PatternDataset(base_dataset, laser_model)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Any
import lightning as pl
from tqdm import tqdm


class PatternDataset(Dataset):
    """
    Dataset that extracts pattern indices from images using a pretrained LASER model.

    Can operate in two modes:
    1. Online mode (cache=False): Extracts patterns on-the-fly during training
    2. Cached mode (cache=True): Pre-extracts all patterns and stores them in memory

    For AR training, cached mode is recommended as it's much faster.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        laser_model: Optional[torch.nn.Module] = None,
        cache: bool = True,
        device: str = 'cuda',
        batch_size: int = 64,
        show_progress: bool = True,
    ):
        """
        Args:
            base_dataset: Image dataset (e.g., CIFAR10, CelebA)
            laser_model: Pretrained LASER model with pattern quantization enabled
            cache: If True, pre-extract all patterns; if False, extract on-the-fly
            device: Device for pattern extraction
            batch_size: Batch size for pattern extraction (only used if cache=True)
            show_progress: Show progress bar during caching
        """
        self.base_dataset = base_dataset
        self.laser_model = laser_model
        self.cache = cache
        self.device = device
        self.cached_patterns = None

        if cache and laser_model is not None:
            self._cache_patterns(batch_size, show_progress)

    def _cache_patterns(self, batch_size: int, show_progress: bool):
        """Pre-extract all pattern indices and cache them."""
        self.laser_model.eval()
        self.laser_model.to(self.device)

        # Create a simple dataloader for batch processing
        loader = DataLoader(
            self.base_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        all_patterns = []

        with torch.no_grad():
            iterator = tqdm(loader, desc="Caching pattern indices") if show_progress else loader
            for batch in iterator:
                # Handle different batch formats
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch

                images = images.to(self.device)

                # Get pattern indices from LASER
                pattern_indices = self._extract_patterns(images)
                all_patterns.append(pattern_indices.cpu())

        self.cached_patterns = torch.cat(all_patterns, dim=0)
        print(f"Cached {len(self.cached_patterns)} pattern sequences")

        # Free GPU memory
        self.laser_model.cpu()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _extract_patterns(self, images: torch.Tensor) -> torch.Tensor:
        """Extract pattern indices from images using LASER model."""
        # Forward through LASER
        outputs = self.laser_model(images)

        # outputs format with pattern quantization:
        # (recon, bottleneck_loss, coefficients, pattern_indices, pattern_info)
        if isinstance(outputs, tuple) and len(outputs) >= 4:
            pattern_indices = outputs[3]  # [B, num_patches]
        else:
            raise ValueError(
                "LASER model output doesn't contain pattern indices. "
                "Make sure use_pattern_quantizer=True in the model config."
            )

        return pattern_indices

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns:
            pattern_indices: [seq_len] tensor of pattern indices
        """
        if self.cached_patterns is not None:
            return self.cached_patterns[idx]

        # Online mode: extract on the fly
        if self.laser_model is None:
            raise ValueError("LASER model required for online pattern extraction")

        # Get image from base dataset
        item = self.base_dataset[idx]
        if isinstance(item, (list, tuple)):
            image = item[0]
        else:
            image = item

        # Add batch dimension
        image = image.unsqueeze(0).to(self.device)

        # Extract pattern
        self.laser_model.eval()
        pattern_indices = self._extract_patterns(image)

        return pattern_indices.squeeze(0).cpu()


class PatternDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for pattern index datasets.

    Handles loading LASER model and creating pattern datasets for train/val/test.
    """

    def __init__(
        self,
        laser_checkpoint: str,
        base_datamodule: pl.LightningDataModule,
        batch_size: int = 256,
        num_workers: int = 4,
        cache: bool = True,
        device: str = 'cuda',
    ):
        """
        Args:
            laser_checkpoint: Path to pretrained LASER checkpoint
            base_datamodule: DataModule for image data (e.g., CIFAR10DataModule)
            batch_size: Batch size for AR training
            num_workers: Number of data loading workers
            cache: Whether to cache pattern indices
            device: Device for pattern extraction
        """
        super().__init__()
        self.laser_checkpoint = laser_checkpoint
        self.base_datamodule = base_datamodule
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache = cache
        self.device = device

        self.laser_model = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """Download data if needed."""
        self.base_datamodule.prepare_data()

    def setup(self, stage: Optional[str] = None):
        """Set up datasets."""
        # Load LASER model
        if self.laser_model is None:
            from src.models.laser import LASER
            self.laser_model = LASER.load_from_checkpoint(
                self.laser_checkpoint,
                map_location='cpu'
            )
            self.laser_model.eval()

            # Verify pattern quantization is enabled
            if not self.laser_model.use_pattern_quantizer:
                raise ValueError(
                    "LASER model doesn't have pattern quantization enabled. "
                    "Train LASER with use_pattern_quantizer=True first."
                )

        # Setup base datamodule
        self.base_datamodule.setup(stage)

        if stage == 'fit' or stage is None:
            # Create train dataset
            self.train_dataset = PatternDataset(
                self.base_datamodule.train_dataset,
                self.laser_model,
                cache=self.cache,
                device=self.device,
            )

            # Create val dataset
            self.val_dataset = PatternDataset(
                self.base_datamodule.val_dataset,
                self.laser_model,
                cache=self.cache,
                device=self.device,
            )

        if stage == 'test' or stage is None:
            # Create test dataset
            self.test_dataset = PatternDataset(
                self.base_datamodule.test_dataset,
                self.laser_model,
                cache=self.cache,
                device=self.device,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class CachedPatternDataset(Dataset):
    """
    Dataset that loads pre-extracted pattern indices from a file.

    Use this for faster training when patterns have already been extracted.
    """

    def __init__(self, pattern_file: str):
        """
        Args:
            pattern_file: Path to .pt file containing pattern indices
        """
        self.patterns = torch.load(pattern_file)
        print(f"Loaded {len(self.patterns)} pattern sequences from {pattern_file}")

    def __len__(self) -> int:
        return len(self.patterns)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.patterns[idx]


def extract_and_save_patterns(
    laser_checkpoint: str,
    base_datamodule: pl.LightningDataModule,
    output_dir: str,
    device: str = 'cuda',
    batch_size: int = 64,
):
    """
    Extract pattern indices from all images and save to files.

    This is useful for creating cached datasets that can be loaded quickly
    without needing the LASER model during AR training.

    Args:
        laser_checkpoint: Path to pretrained LASER checkpoint
        base_datamodule: DataModule for image data
        output_dir: Directory to save pattern files
        device: Device for extraction
        batch_size: Batch size for extraction
    """
    import os
    from src.models.laser import LASER

    os.makedirs(output_dir, exist_ok=True)

    # Load LASER model
    print(f"Loading LASER model from {laser_checkpoint}")
    laser_model = LASER.load_from_checkpoint(laser_checkpoint, map_location='cpu')
    laser_model.eval()
    laser_model.to(device)

    if not laser_model.use_pattern_quantizer:
        raise ValueError("LASER model doesn't have pattern quantization enabled")

    # Setup datamodule
    base_datamodule.setup()

    # Extract patterns for each split
    for split_name, dataset in [
        ('train', base_datamodule.train_dataset),
        ('val', base_datamodule.val_dataset),
        ('test', base_datamodule.test_dataset),
    ]:
        if dataset is None:
            continue

        print(f"\nExtracting patterns for {split_name} split...")
        pattern_dataset = PatternDataset(
            dataset,
            laser_model,
            cache=True,
            device=device,
            batch_size=batch_size,
        )

        # Save patterns
        output_path = os.path.join(output_dir, f'{split_name}_patterns.pt')
        torch.save(pattern_dataset.cached_patterns, output_path)
        print(f"Saved {len(pattern_dataset)} patterns to {output_path}")

    print("\nPattern extraction complete!")
