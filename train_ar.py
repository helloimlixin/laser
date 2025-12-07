"""
Training script for Autoregressive Transformer on LASER pattern indices.

This script:
1. Loads a pretrained LASER model with pattern quantization
2. Extracts pattern indices from the image dataset
3. Trains an autoregressive transformer to predict pattern sequences

Usage:
    python train_ar.py laser_ckpt=path/to/laser.ckpt
    python train_ar.py laser_ckpt=path/to/laser.ckpt data.dataset=cifar10
"""

import os
import warnings

warnings.filterwarnings('ignore', message='.*TF32.*')
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import WandbLogger
from datetime import datetime
import math

torch.set_float32_matmul_precision('medium')

from src.models.ar_transformer import ARTransformer
from src.models.laser import LASER
from src.data.pattern_dataset import PatternDataset, PatternDataModule
from src.data.cifar10 import CIFAR10DataModule
from src.data.imagenette2 import Imagenette2DataModule
from src.data.celeba import CelebADataModule
from src.data.config import DataConfig

# Configure progress bar theme
progress_bar = RichProgressBar(
    theme=RichProgressBarTheme(
        description="green_yellow",
        progress_bar="green1",
        progress_bar_finished="green1",
        progress_bar_pulse="green1",
        batch_progress="green_yellow",
        time="grey82",
        processing_speed="grey82",
        metrics="grey82"
    ),
    leave=True
)


@hydra.main(config_path="configs", config_name="config_ar", version_base="1.2")
def train_ar(cfg: DictConfig):
    """
    Main training function for AR transformer.

    Args:
        cfg: Hydra configuration object
    """
    # Print configuration
    print("\n" + "="*60)
    print("AR TRANSFORMER TRAINING")
    print("="*60)

    print("\nLASER Model:")
    print(f"  Checkpoint: {cfg.laser_ckpt}")

    print("\nAR Transformer Config:")
    print(f"  Vocab Size: {cfg.ar.vocab_size}")
    print(f"  Sequence Length: {cfg.ar.seq_len}")
    print(f"  Model Dim: {cfg.ar.d_model}")
    print(f"  Heads: {cfg.ar.n_heads}")
    print(f"  Layers: {cfg.ar.n_layers}")
    print(f"  FF Dim: {cfg.ar.d_ff}")
    print(f"  Dropout: {cfg.ar.dropout}")

    print("\nTraining Config:")
    print(f"  Learning Rate: {cfg.ar.learning_rate}")
    print(f"  Warmup Steps: {cfg.ar.warmup_steps}")
    print(f"  Max Steps: {cfg.ar.max_steps}")
    print(f"  Batch Size: {cfg.train_ar.batch_size}")
    print(f"  Max Epochs: {cfg.train_ar.max_epochs}")

    print("\nDataset:")
    print(f"  Dataset: {cfg.data.dataset}")
    print("="*60 + "\n")

    # Set seed
    pl.seed_everything(cfg.seed)

    # Check GPU
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")

    # Load pretrained LASER model
    print(f"\nLoading LASER model from: {cfg.laser_ckpt}")
    laser_model = LASER.load_from_checkpoint(cfg.laser_ckpt, map_location='cpu')
    laser_model.eval()

    # Verify pattern quantization is enabled
    if not laser_model.use_pattern_quantizer:
        raise ValueError(
            "LASER bottleneck quantization is disabled. Use the external sparse-code "
            "+ k-means quantization pipeline to create patterns before AR training."
        )

    print(f"LASER model loaded. Pattern vocab size: {laser_model.bottleneck.num_patterns}")

    # Get sequence length from LASER model
    # seq_len = num_patches = (latent_h / patch_h) * (latent_w / patch_w)
    # For simplicity, we'll get it from config or infer from a sample

    # Initialize base data module
    print(f"\nInitializing data module for: {cfg.data.dataset}")
    if cfg.data.dataset == 'cifar10':
        base_datamodule = CIFAR10DataModule(DataConfig.from_dict(cfg.data))
    elif cfg.data.dataset == 'imagenette2':
        base_datamodule = Imagenette2DataModule(DataConfig.from_dict(cfg.data))
    elif cfg.data.dataset == 'celeba':
        base_datamodule = CelebADataModule(DataConfig.from_dict(cfg.data))
    else:
        raise ValueError(f"Unsupported dataset: {cfg.data.dataset}")

    # Create pattern data module
    pattern_datamodule = PatternDataModule(
        laser_checkpoint=cfg.laser_ckpt,
        base_datamodule=base_datamodule,
        batch_size=cfg.train_ar.batch_size,
        num_workers=cfg.data.num_workers,
        cache=True,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    # Infer AR sequence length from LASER bottleneck + image size to avoid mismatches
    def _to_tuple(val):
        if isinstance(val, (list, tuple)):
            return int(val[0]), int(val[1])
        return int(val), int(val)

    patch_h, patch_w = _to_tuple(laser_model.hparams.patch_size)
    patch_stride_cfg = getattr(laser_model.hparams, "patch_stride", None)
    stride_h, stride_w = _to_tuple(patch_stride_cfg) if patch_stride_cfg is not None else (patch_h, patch_w)
    latent_h = math.ceil(cfg.data.image_size / 4)
    latent_w = math.ceil(cfg.data.image_size / 4)
    pad_h = (patch_h - (latent_h % patch_h)) % patch_h
    pad_w = (patch_w - (latent_w % patch_w)) % patch_w
    padded_h = latent_h + pad_h
    padded_w = latent_w + pad_w
    n_h = (padded_h - patch_h) // stride_h + 1
    n_w = (padded_w - patch_w) // stride_w + 1
    expected_seq_len = n_h * n_w

    if cfg.ar.seq_len != expected_seq_len:
        print(f"Adjusting AR seq_len from {cfg.ar.seq_len} to {expected_seq_len} to match patch grid ({n_h}x{n_w})")
        cfg.ar.seq_len = expected_seq_len

    # Initialize AR Transformer
    print("\nInitializing AR Transformer...")
    ar_model = ARTransformer(
        vocab_size=cfg.ar.vocab_size,
        seq_len=cfg.ar.seq_len,
        d_model=cfg.ar.d_model,
        n_heads=cfg.ar.n_heads,
        n_layers=cfg.ar.n_layers,
        d_ff=cfg.ar.d_ff,
        dropout=cfg.ar.dropout,
        learning_rate=cfg.ar.learning_rate,
        weight_decay=cfg.ar.weight_decay,
        warmup_steps=cfg.ar.warmup_steps,
        max_steps=cfg.ar.max_steps,
        use_bos=cfg.ar.use_bos,
        use_eos=cfg.ar.use_eos,
    )

    # Count parameters
    total_params = sum(p.numel() for p in ar_model.parameters())
    trainable_params = sum(p.numel() for p in ar_model.parameters() if p.requires_grad)
    print(f"AR Transformer parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Set LASER model for generation visualization
    ar_model.set_laser_model(laser_model, log_images_every_n_epochs=1, num_samples=8)
    print("LASER model set for generation visualization")

    # Setup checkpoint directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    ckpt_dir = os.path.join(cfg.output_dir, "checkpoints", f"ar_{timestamp}")
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\nCheckpoint directory: {ckpt_dir}")

    # Initialize wandb logger
    run_name = f"ar_transformer_{cfg.data.dataset}_{timestamp}"
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=run_name,
        save_dir=cfg.wandb.save_dir,
    )

    # Log hyperparameters
    wandb_logger.log_hyperparams({
        'laser_ckpt': cfg.laser_ckpt,
        'ar_vocab_size': cfg.ar.vocab_size,
        'ar_seq_len': cfg.ar.seq_len,
        'ar_d_model': cfg.ar.d_model,
        'ar_n_heads': cfg.ar.n_heads,
        'ar_n_layers': cfg.ar.n_layers,
        'ar_d_ff': cfg.ar.d_ff,
        'ar_dropout': cfg.ar.dropout,
        'ar_learning_rate': cfg.ar.learning_rate,
        'dataset': cfg.data.dataset,
    })

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='ar-{epoch:03d}-{val/loss:.4f}',
            save_top_k=3,
            monitor='val/loss',
            mode='min',
            save_last=True,
        ),
        LearningRateMonitor(logging_interval='step'),
        progress_bar,
    ]

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.train_ar.max_epochs,
        accelerator=cfg.train_ar.accelerator,
        devices=cfg.train_ar.devices,
        logger=wandb_logger,
        callbacks=callbacks,
        precision=cfg.train_ar.precision,
        gradient_clip_val=cfg.train_ar.gradient_clip_val,
        log_every_n_steps=cfg.train_ar.log_every_n_steps,
        val_check_interval=getattr(cfg.train_ar, 'val_check_interval', 1.0),
        deterministic=True,
        enable_progress_bar=True,
        num_sanity_val_steps=2,
    )

    # Train
    print("\nStarting AR Transformer training...")
    trainer.fit(ar_model, datamodule=pattern_datamodule)

    # Test
    print("\nRunning test evaluation...")
    trainer.test(ar_model, datamodule=pattern_datamodule)

    print("\nTraining complete!")
    print(f"Best checkpoint: {callbacks[0].best_model_path}")

    return callbacks[0].best_model_path


if __name__ == "__main__":
    train_ar()
