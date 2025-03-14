import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import torchvision

from .encoder import Encoder
from .decoder import Decoder
from .bottleneck import DictionaryLearningBottleneck
from src.lpips import LPIPS

import torchmetrics
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
import wandb

class DLVAE(pl.LightningModule):
    def __init__(
            self,
            in_channels,
            num_hiddens,
            num_embeddings,
            embedding_dim,
            sparsity,
            num_residual_blocks,
            num_residual_hiddens,
            commitment_cost,
            decay,
            perceptual_weight,
            learning_rate,
            beta,
            compute_fid=False
    ):
        """Initialize DLVAE model.

        Args:
            in_channels: Number of input channels (3 for RGB)
            num_hiddens: Number of hidden units
            num_embeddings: Number of embeddings
            embedding_dim: Dimension of latent space
            sparsity: Sparsity parameter for DictionaryLearningBottleneck
            num_residual_blocks: Number of residual blocks
            num_residual_hiddens: Number of hidden units in residual blocks
            commitment_cost: Commitment cost for DictionaryLearningBottleneck
            decay: Decay parameter for DictionaryLearningBottleneck
            perceptual_weight: Weight for perceptual loss
            learning_rate: Learning rate
            beta: Beta parameter for Adam optimizer
            compute_fid: Whether to compute FID
        """
        super(DLVAE, self).__init__()

        # Store parameters
        self.learning_rate = learning_rate
        self.beta = beta
        self.perceptual_weight = perceptual_weight
        self.log_images_every_n_steps = 100
        self.compute_fid = compute_fid

        # Initialize encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            num_hiddens=num_hiddens,
            num_residual_blocks=num_residual_blocks,
            num_residual_hiddens=num_residual_hiddens
        )

        self.pre_bottleneck = nn.Conv2d(in_channels=num_hiddens,
                                        out_channels=embedding_dim,
                                        kernel_size=1,
                                        stride=1)

        # Initialize bottleneck
        self.bottleneck = DictionaryLearningBottleneck(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            sparsity=sparsity,
            commitment_cost=commitment_cost,
            decay=decay
        )

        self.post_bottleneck = nn.Conv2d(in_channels=embedding_dim,
                                         out_channels=num_hiddens,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1)

        # Initialize decoder
        self.decoder = Decoder(
            in_channels=embedding_dim,
            num_hiddens=num_hiddens,
            num_residual_blocks=num_residual_blocks,
            num_residual_hiddens=num_residual_hiddens
        )

        # Initialize LPIPS for perceptual loss
        self.lpips = LPIPS()

        # Initialize metrics
        self._init_metrics()

        # Save hyperparameters
        self.save_hyperparameters()

    def _init_metrics(self):
        # Training metrics
        self.train_recon_loss = torchmetrics.MeanMetric()
        self.train_commit_loss = torchmetrics.MeanMetric()
        self.train_perceptual_loss = torchmetrics.MeanMetric()
        self.train_psnr = PeakSignalNoiseRatio()

        # Validation metrics
        self.val_recon_loss = torchmetrics.MeanMetric()
        self.val_commit_loss = torchmetrics.MeanMetric()
        self.val_perceptual_loss = torchmetrics.MeanMetric()
        self.val_psnr = PeakSignalNoiseRatio()

        # Test metrics
        self.test_metrics = torchmetrics.MeanMetric()
        self.test_psnr = PeakSignalNoiseRatio()

        if self.compute_fid:
            self.test_fid = FrechetInceptionDistance(feature=2048, normalize=True)
        else:
            self.test_fid = None

    def forward(self, x):
        # Encode
        h = self.encoder(x)
        h = self.pre_bottleneck(h)
        
        # Apply bottleneck
        z_q, loss, coefficients = self.bottleneck(h)
        
        # Decode
        recon = self.decoder(z_q)
        
        return recon, loss, coefficients

    def compute_metrics(self, batch, prefix='train'):
        """Compute metrics for a batch."""
        # Get input
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        
        # Forward pass
        recon, commit_loss, coefficients = self(x)
        
        # Compute losses
        recon_loss = F.mse_loss(recon, x).mean()
        
        # Perceptual loss
        x_norm = x * 2.0 - 1.0
        x_recon_norm = recon * 2.0 - 1.0
        perceptual_loss = self.lpips(x_recon_norm, x_norm).mean()
        
        # Total loss
        total_loss = recon_loss + commit_loss + self.perceptual_weight * perceptual_loss
        
        # Update metrics
        if hasattr(self, f'{prefix}_recon_loss'):
            getattr(self, f'{prefix}_recon_loss')(recon_loss)
            getattr(self, f'{prefix}_commit_loss')(commit_loss)
            getattr(self, f'{prefix}_perceptual_loss')(perceptual_loss)

        # Handle FID for test
        if prefix == 'test' and self.test_fid is not None:
            x_recon_fid = torch.clamp(recon, 0, 1)
            x_fid = torch.clamp(x, 0, 1)
            self.test_fid.update(x_recon_fid, real=False)
            self.test_fid.update(x_fid, real=True)

        # Log metrics
        self.log(f'{prefix}/loss', total_loss, on_step=True, on_epoch=True)
        self.log(f'{prefix}/recon_loss', recon_loss, on_step=True, on_epoch=True)
        self.log(f'{prefix}/commit_loss', commit_loss, on_step=True, on_epoch=True)
        self.log(f'{prefix}/perceptual_loss', perceptual_loss, on_step=True, on_epoch=True)

        # Add PSNR calculation
        if hasattr(self, f'{prefix}_psnr'):
            x_psnr = torch.clamp(x, 0, 1)
            recon_psnr = torch.clamp(recon, 0, 1)
            psnr_value = getattr(self, f'{prefix}_psnr')(recon_psnr, x_psnr)
            self.log(f'{prefix}/psnr', psnr_value, on_step=True, on_epoch=True)

        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'commit_loss': commit_loss,
            'perceptual_loss': perceptual_loss,
            'x': x,
            'x_recon': recon,
            'coefficients': coefficients
        }

    def training_step(self, batch, batch_idx):
        metrics = self.compute_metrics(batch, prefix='train')
        
        if batch_idx % self.log_images_every_n_steps == 0:
            self._log_images(metrics['x'], metrics['x_recon'], split='train')
            
        return metrics

    def validation_step(self, batch, batch_idx):
        metrics = self.compute_metrics(batch, prefix='val')
        
        if batch_idx % self.log_images_every_n_steps == 0:
            self._log_images(metrics['x'], metrics['x_recon'], split='val')
            
        return metrics

    def test_step(self, batch, batch_idx):
        metrics = self.compute_metrics(batch, prefix='test')
        
        if batch_idx % self.log_images_every_n_steps == 0:
            self._log_images(metrics['x'], metrics['x_recon'], split='test')
            
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(self.beta, 0.999))
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1
            }
        }

    def _log_images(self, x, x_recon, split='train'):
        """Log images to wandb."""
        # Take first 32 images
        x = x[:32]
        x_recon = x_recon[:32]

        # Create image grids
        x_grid = torchvision.utils.make_grid(x, nrow=8, normalize=True, value_range=(-1, 1))
        x_recon_grid = torchvision.utils.make_grid(x_recon, nrow=8, normalize=True, value_range=(-1, 1))

        # Convert to numpy
        x_grid = x_grid.cpu().numpy().transpose(1, 2, 0)
        x_recon_grid = x_recon_grid.cpu().numpy().transpose(1, 2, 0)

        # Log to wandb
        self.logger.experiment.log({
            f"{split}/images": [
                wandb.Image(x_grid, caption="Original"),
                wandb.Image(x_recon_grid, caption="Reconstructed")
            ],
            "global_step": self.global_step
        })
        
        