import os
import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import WandbLogger
import torch

from src.data.imagenette2 import Imagenette2DataModule
from src.models.vqvae import VQVAE

from datetime import datetime
import os

# Create a unique directory for each run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
checkpoint_dir = os.path.join('outputs', 'checkpoints', f'run_{timestamp}')
os.makedirs(checkpoint_dir, exist_ok=True)

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
def main(config: DictConfig):
    # Set random seed
    pl.seed_everything(config.seed)

    # Print GPU information
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")

    # Initialize data module
    data_module = Imagenette2DataModule(config.data)

    # Initialize model
    model = VQVAE(
        in_channels=config.model.in_channels,
        hidden_dims=config.model.hidden_dims,
        num_embeddings=config.model.num_embeddings,
        embedding_dim=config.model.embedding_dim,
        n_residual_blocks=config.model.n_residual_blocks,
        commitment_cost=config.model.commitment_cost,
        decay=config.model.decay,
        perceptual_weight=config.model.perceptual_weight,
        learning_rate=config.model.learning_rate,
        beta=config.model.beta
    )

    # Convert config to plain dictionary for wandb
    wandb_config = OmegaConf.to_container(config, resolve=True)

    # Set up logging
    wandb_logger = WandbLogger(
        project=config.wandb.project,
        name=config.wandb.name,
        config=wandb_config,  # Use the converted config
        save_dir=config.output_dir,
        version=''
    )

    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='{epoch}-{val_loss:.2f}',
            monitor="val/loss",
            mode="min",
            save_last=True,
            save_top_k=3,
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=config.trainer.early_stopping_patience,
            mode="min"
        ),
        LearningRateMonitor(logging_interval='epoch'),
        progress_bar
    ]

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config.trainer.max_epochs,
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=config.trainer.log_every_n_steps,
        deterministic=True,
        enable_progress_bar=True,
        enable_model_summary=True
    )

    # Train the model
    torch.set_float32_matmul_precision("medium")
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
