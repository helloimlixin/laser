import os
import torch
import hydra
from omegaconf import DictConfig
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import WandbLogger
from datetime import datetime

from src.models.vqvae import VQVAE
from src.models.dlvae import DLVAE
from src.data.cifar10 import CIFAR10DataModule
from src.data.config import DataConfig
from src.data.imagenette2 import Imagenette2DataModule
from src.data.celeba import CelebADataModule

# Create a unique directory for each run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
checkpoint_dir = os.path.join('outputs', 'checkpoints', f'run_{timestamp}')
os.makedirs(checkpoint_dir, exist_ok=True)

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

@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def train(cfg: DictConfig):
    """
    Main training function using Hydra for configuration.
    
    Args:
        cfg: Hydra configuration object containing model and training parameters
    """
    # Print detailed experiment configuration
    print("\n" + "="*50)
    print("EXPERIMENT CONFIGURATION")
    print("="*50)
    
    print("\nGeneral Settings:")
    print(f"Random Seed: {cfg.seed}")
    print(f"Output Directory: {cfg.output_dir}")
    
    print("\nDataset Configuration:")
    print(f"Dataset: {cfg.data.dataset}")
    print(f"Data Directory: {cfg.data.data_dir}")
    print(f"Batch Size: {cfg.data.batch_size}")
    print(f"Number of Workers: {cfg.data.num_workers}")
    print(f"Image Size: {cfg.data.image_size}")
    print(f"Mean: {cfg.data.mean}")
    print(f"Std: {cfg.data.std}")
    
    print("\nModel Configuration:")
    print(f"Model Type: {cfg.model.type}")
    print(f"Input Channels: {cfg.model.in_channels}")
    print(f"Hidden Dimensions: {cfg.model.num_hiddens}")
    print(f"Embedding Dimensions: {cfg.model.embedding_dim}")
    print(f"Number of Residual Blocks: {cfg.model.num_residual_blocks}")
    print(f"Residual Hidden Dimensions: {cfg.model.num_residual_hiddens}")
    if cfg.model.type == "vqvae":
        print(f"Number of Embeddings: {cfg.model.num_embeddings}")
    elif cfg.model.type == "dlvae":
        print(f"Dictionary Size: {cfg.model.num_embeddings}")
        print(f"Sparsity: {cfg.model.sparsity_level}")
        if hasattr(cfg.model, "omp_tolerance"):
            print(f"OMP Tolerance: {cfg.model.omp_tolerance}")
        if hasattr(cfg.model, "omp_debug"):
            print(f"OMP Debug: {cfg.model.omp_debug}")
    
    print("\nTraining Configuration:")
    print(f"Learning Rate: {cfg.train.learning_rate}")
    print(f"Beta: {cfg.train.beta}")
    print(f"Max Epochs: {cfg.train.max_epochs}")
    print(f"Accelerator: {cfg.train.accelerator}")
    print(f"Devices: {cfg.train.devices}")
    print(f"Precision: {cfg.train.precision}")
    print(f"Gradient Clip Value: {cfg.train.gradient_clip_val}")
    
    print("\nWandB Configuration:")
    print(f"Project: {cfg.wandb.project}")
    print(f"Run Name: {cfg.wandb.name}")
    print(f"Save Directory: {cfg.wandb.save_dir}")
    
    print("\nCheckpoint Configuration:")
    print(f"Save Directory: {checkpoint_dir}")
    print(f"Filename Template: {cfg.model.type}-{{epoch}}-{{val_loss:.2f}}")
    print(f"Save Top K: 3")
    print("="*50 + "\n")

    # Set random seed for reproducibility
    pl.seed_everything(cfg.seed)
    
    # Print GPU information
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    
    # Initialize data module
    print(f"Initializing data module for dataset: {cfg.data.dataset}")
    if cfg.data.dataset == 'cifar10':
        datamodule = CIFAR10DataModule(DataConfig.from_dict(cfg.data))
    elif cfg.data.dataset == 'imagenette2':
        datamodule = Imagenette2DataModule(DataConfig.from_dict(cfg.data))
    elif cfg.data.dataset == 'celeba':
        datamodule = CelebADataModule(DataConfig.from_dict(cfg.data))
    else:
        raise ValueError(f"Unsupported dataset: {cfg.data.dataset}")

    # Print dataset info for debugging
    print(f"Using dataset: {cfg.data.dataset}")
    print(f"Data module type: {type(datamodule).__name__}")
    
    # Initialize model based on type
    model_params = {
        'in_channels': cfg.model.in_channels,
        'num_hiddens': cfg.model.num_hiddens,
        'num_embeddings': cfg.model.num_embeddings,
        'embedding_dim': cfg.model.embedding_dim,
        'num_residual_blocks': cfg.model.num_residual_blocks,
        'num_residual_hiddens': cfg.model.num_residual_hiddens,
        'commitment_cost': cfg.model.commitment_cost,
        'decay': cfg.model.decay,
        'perceptual_weight': cfg.model.perceptual_weight,
        'learning_rate': cfg.train.learning_rate,
        'beta': cfg.train.beta,
        'compute_fid': cfg.model.compute_fid
    }

    if cfg.model.type == "vqvae":
        model = VQVAE(**model_params)
    elif cfg.model.type == "dlvae":
        model_params['sparsity_level'] = cfg.model.sparsity_level
        # Optional OMP params with sensible defaults
        model_params['omp_tolerance'] = getattr(cfg.model, 'omp_tolerance', 1e-7)
        model_params['omp_debug'] = getattr(cfg.model, 'omp_debug', False)
        model = DLVAE(**model_params)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")

    # Initialize wandb logger
    run_name = f"{cfg.wandb.name}_{cfg.model.type}_{timestamp}"
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=run_name,
        save_dir=cfg.wandb.save_dir
    )

    # Initialize callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(checkpoint_dir, cfg.model.type),
            filename=f"{cfg.model.type}-{{epoch}}-{{val_loss_epoch:.2f}}",
            save_top_k=cfg.checkpoint.save_top_k,
            monitor='val/loss_epoch',
            mode='min',
            save_last=True
        ),
        LearningRateMonitor(logging_interval='step'),
        progress_bar
    ]
    # Add EarlyStopping only if configured
    if getattr(cfg.train, "early_stopping_patience", None):
        callbacks.insert(1, EarlyStopping(
            monitor="val/loss_epoch",
            patience=cfg.train.early_stopping_patience,
            mode="min"
        ))

    # Initialize trainer
    # Prefer plain DDP for DLVAE to avoid unnecessary find_unused traversal
    strategy_cfg = getattr(cfg.train, "strategy", None)
    if cfg.model.type == "dlvae" and strategy_cfg == "ddp_find_unused_parameters_true":
        strategy_cfg = "ddp"
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        strategy=strategy_cfg,
        logger=wandb_logger,
        callbacks=callbacks,
        precision=cfg.train.precision,
        gradient_clip_val=cfg.train.gradient_clip_val,
        log_every_n_steps=cfg.train.log_every_n_steps,
        deterministic=True,
        enable_progress_bar=True,
        enable_model_summary=(str(cfg.train.precision) == "32"),
        reload_dataloaders_every_n_epochs=0,
        num_sanity_val_steps=0,
    )

    # Train and test model
    torch.set_float32_matmul_precision('medium')
    trainer.fit(model, datamodule=datamodule)
    # Run test on a single device to avoid DistributedSampler duplications
    test_trainer = pl.Trainer(
        accelerator=('gpu' if (cfg.train.accelerator == 'gpu' and torch.cuda.is_available()) else 'cpu'),
        devices=1,
        logger=wandb_logger,
        precision=cfg.train.precision,
        deterministic=True,
        enable_progress_bar=True,
        enable_model_summary=(str(cfg.train.precision) == "32")
    )
    test_trainer.test(model, datamodule=datamodule)

if __name__ == "__main__":
    train()