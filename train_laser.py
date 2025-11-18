#!/usr/bin/env python
"""
Train K-SVD VAE model on CIFAR-10 or CelebA dataset.

Usage:
    python train_ksvd.py --dataset cifar10 --epochs 100
    python train_ksvd.py --dataset celeba --epochs 50 --batch_size 32
"""

import os
import sys
from pathlib import Path
import argparse

# Ensure project sources are importable
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from src.models.laser import LASER
from src.data.cifar10 import CIFAR10DataModule
from src.data.celeba import CelebADataModule
from src.data.config import DataConfig


def main():
    parser = argparse.ArgumentParser(description='Train K-SVD VAE')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'celeba'],
                        help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory (default: ./data for CIFAR-10, ~/data/celeba for CelebA)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--image_size', type=int, default=64,
                        help='Image size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num_embeddings', type=int, default=128,
                        help='Number of dictionary atoms')
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--sparsity_level', type=int, default=8,
                        help='Sparsity level (number of non-zero coefficients)')
    parser.add_argument('--ksvd_iterations', type=int, default=2,
                        help='Number of K-SVD update iterations per forward pass')
    parser.add_argument('--patch_size', type=int, default=1,
                        help='Patch size for dictionary learning (1 = pixel-level)')
    parser.add_argument('--perceptual_weight', type=float, default=1.0,
                        help='Weight for perceptual loss')
    parser.add_argument('--commitment_cost', type=float, default=0.25,
                        help='Commitment cost')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases logging')
    parser.add_argument('--project_name', type=str, default='ksvd-vae',
                        help='W&B project name')
    parser.add_argument('--run_name', type=str, default=None,
                        help='W&B run name')
    parser.add_argument('--output_dir', type=str, default='./outputs/ksvd',
                        help='Output directory for checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    pl.seed_everything(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup data
    if args.dataset == 'cifar10':
        data_dir = args.data_dir if args.data_dir else './data'
        data_config = DataConfig(
            dataset='cifar10',
            data_dir=data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
        )
        dm = CIFAR10DataModule(data_config)
    elif args.dataset == 'celeba':
        data_dir = args.data_dir if args.data_dir else str(Path.home() / 'data' / 'celeba')
        data_config = DataConfig(
            dataset='celeba',
            data_dir=data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
        )
        dm = CelebADataModule(data_config)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Setup model
    model = LASER(
        in_channels=3,
        num_hiddens=128,
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
        sparsity_level=args.sparsity_level,
        num_residual_blocks=1,
        num_residual_hiddens=32,
        commitment_cost=args.commitment_cost,
        learning_rate=args.learning_rate,
        beta=0.9,
        ksvd_iterations=args.ksvd_iterations,
        perceptual_weight=args.perceptual_weight,
        compute_fid=False,
        patch_size=args.patch_size,
        use_backprop_only=True,  # Use only backprop for dictionary learning
    )
    
    # Setup logger
    if args.use_wandb:
        run_name = args.run_name if args.run_name else f"ksvd_{args.dataset}_k{args.num_embeddings}_s{args.sparsity_level}"
        logger = WandbLogger(
            project=args.project_name,
            name=run_name,
            save_dir=str(output_dir),
        )
    else:
        logger = None
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / 'checkpoints',
        filename='ksvd-vae-{epoch:02d}-{val/loss:.4f}',
        save_top_k=3,
        monitor='val/loss',
        mode='min',
        save_last=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='auto',
        devices=1,
        precision="16-mixed",  # Enable automatic mixed precision for faster training
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
    )
    
    # Print configuration
    print("\n" + "="*60)
    print("K-SVD VAE TRAINING CONFIGURATION")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Data directory: {data_dir}")
    print(f"Image size: {args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"\nModel Configuration:")
    print(f"  Dictionary atoms: {args.num_embeddings}")
    print(f"  Embedding dim: {args.embedding_dim}")
    print(f"  Sparsity level: {args.sparsity_level}")
    print(f"  K-SVD iterations: {args.ksvd_iterations}")
    print(f"  Patch size: {args.patch_size}")
    print(f"  Perceptual weight: {args.perceptual_weight}")
    print(f"  Commitment cost: {args.commitment_cost}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Using W&B: {args.use_wandb}")
    print("="*60 + "\n")
    
    # Train
    trainer.fit(model, dm)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Last checkpoint: {checkpoint_callback.last_model_path}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
