import os
import torch
import hydra
from omegaconf import DictConfig
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import WandbLogger
from datetime import datetime

from models.vqvae import VQVAE
from models.dlvae import DLVAE
from data.cifar10 import CIFAR10DataModule, CIFAR10Config
from data.imagenette2 import Imagenette2DataModule, DataConfig

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

@hydra.main(config_path="../configs", config_name="train")
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
        print(f"Dictionary Size: {cfg.model.dict_size}")
        print(f"Sparsity: {cfg.model.sparsity}")
    
    print("\nTraining Configuration:")
    print(f"Learning Rate: {cfg.training.learning_rate}")
    print(f"Beta: {cfg.training.beta}")
    print(f"Max Epochs: {cfg.training.max_epochs}")
    print(f"Accelerator: {cfg.training.accelerator}")
    print(f"Devices: {cfg.training.devices}")
    print(f"Precision: {cfg.training.precision}")
    print(f"Gradient Clip Value: {cfg.training.gradient_clip_val}")
    
    print("\nWandB Configuration:")
    print(f"Project: {cfg.wandb.project}")
    print(f"Run Name: {cfg.wandb.name}")
    print(f"Save Directory: {cfg.wandb.save_dir}")
    
    print("\nCheckpoint Configuration:")
    print(f"Save Directory: {cfg.checkpoint.dirpath}")
    print(f"Filename Template: {cfg.checkpoint.filename}")
    print(f"Save Top K: {cfg.checkpoint.save_top_k}")
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
        datamodule = CIFAR10DataModule(CIFAR10Config.from_dict(cfg.data))
    elif cfg.data.dataset == 'imagenette2':
        datamodule = Imagenette2DataModule(DataConfig.from_dict(cfg.data))
    else:
        raise ValueError(f"Unsupported dataset: {cfg.data.dataset}")

    # Print dataset info for debugging
    print(f"Using dataset: {cfg.data.dataset}")
    print(f"Data module type: {type(datamodule).__name__}")
    
    # Initialize model based on type
    model_params = {
        'in_channels': cfg.model.in_channels,
        'num_hiddens': cfg.model.num_hiddens,
        'embedding_dim': cfg.model.embedding_dim,
        'num_residual_blocks': cfg.model.num_residual_blocks,
        'num_residual_hiddens': cfg.model.num_residual_hiddens,
        'commitment_cost': cfg.model.commitment_cost,
        'decay': cfg.model.decay,
        'perceptual_weight': cfg.model.perceptual_weight,
        'learning_rate': cfg.training.learning_rate,
        'beta': cfg.training.beta,
        'compute_fid': cfg.model.compute_fid
    }

    if cfg.model.type == "vqvae":
        model_params['num_embeddings'] = cfg.model.num_embeddings
        model = VQVAE(**model_params)
    elif cfg.model.type == "dlvae":
        model_params['dict_size'] = cfg.model.dict_size
        model_params['sparsity'] = cfg.model.sparsity
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
            filename=f"{cfg.model.type}-{{epoch}}-{{val_loss:.2f}}",
            save_top_k=cfg.checkpoint.save_top_k,
            monitor='val/loss',
            mode='min',
            save_last=True
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=cfg.training.early_stopping_patience,
            mode="min"
        ),
        LearningRateMonitor(logging_interval='step'),
        progress_bar
    ]

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        logger=wandb_logger,
        callbacks=callbacks,
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        log_every_n_steps=cfg.training.log_every_n_steps,
        deterministic=True,
        enable_progress_bar=True,
        enable_model_summary=True
    )

    # Train and test model
    torch.set_float32_matmul_precision('medium')
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

if __name__ == "__main__":
    train()