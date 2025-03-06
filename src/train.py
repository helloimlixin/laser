import os
import torch
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from models.vqvae import VQVAE
from models.dlvae import DLVAE
from data import ImageDataModule

@hydra.main(config_path="../configs", config_name="train")
def train(cfg: DictConfig):
    """
    Main training function using Hydra for configuration.
    
    Args:
        cfg: Hydra configuration object containing model and training parameters
    """
    # Set random seed for reproducibility
    pl.seed_everything(cfg.seed)
    
    # Initialize data module
    datamodule = ImageDataModule(
        data_dir=cfg.data.path,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        image_size=cfg.data.image_size
    )
    
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

    # Initialize wandb logger with model type in name
    run_name = f"{cfg.wandb.name}_{cfg.model.type}"
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=run_name,
        save_dir=cfg.wandb.save_dir
    )

    # Initialize callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(cfg.checkpoint.dirpath, cfg.model.type),
            filename=f"{cfg.model.type}-" + cfg.checkpoint.filename,
            save_top_k=cfg.checkpoint.save_top_k,
            monitor='val/loss',
            mode='min',
            save_last=True
        ),
        LearningRateMonitor(logging_interval='step')
    ]

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        logger=wandb_logger,
        callbacks=callbacks,
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.gradient_clip_val
    )

    # Train and test model
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

if __name__ == "__main__":
    train()