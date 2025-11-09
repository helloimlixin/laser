import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import torchvision
from torchmetrics.functional.text import perplexity

from .encoder import Encoder
from .decoder import Decoder
from .bottleneck import VectorQuantizer
from src.lpips import LPIPS

import torchmetrics
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
import wandb

class VQVAE(pl.LightningModule):
    def __init__(
            self,
            in_channels,
            num_hiddens,
            num_embeddings,
            embedding_dim,
            num_residual_blocks,
            num_residual_hiddens,
            commitment_cost,
            decay,
            perceptual_weight,
            learning_rate,
            beta,
            compute_fid=False
    ):
        """Initialize VQVAE model.

        Args:
            in_channels: Number of input channels, 3 for RGB images
            num_hiddens: number of hidden units (hidden dimensions)
            num_embeddings: Number of embeddings in codebook
            embedding_dim: Dimension of each embedding
            num_residual_blocks: Number of residual blocks in encoder and decoder
            commitment_cost: Commitment cost for VQ
            decay: Decay factor for EMA
            perceptual_weight: Weight for perceptual loss
            learning_rate: Learning rate for optimization
            beta: Beta parameter for optimizer
            compute_fid: Whether to compute FID metric
        """
        super().__init__()

        # Store model parameters
        self.learning_rate = learning_rate
        self.beta = beta
        self.perceptual_weight = perceptual_weight
        self.log_images_every_n_steps = 100
        self.compute_fid = compute_fid

        # Initialize model components
        self.encoder = Encoder(in_channels=in_channels,
                               num_hiddens=num_hiddens,
                               num_residual_blocks=num_residual_blocks,
                               num_residual_hiddens=num_residual_hiddens)
        
        self.pre_bottleneck = nn.Conv2d(in_channels=num_hiddens,
                                        out_channels=embedding_dim,
                                        kernel_size=1,
                                        stride=1)
        
        self.vector_quantizer = VectorQuantizer(
            num_embeddings, embedding_dim, commitment_cost, decay
        )

        self.post_bottleneck = nn.Conv2d(in_channels=embedding_dim,
                                         out_channels=num_hiddens,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1)
        
        self.decoder = Decoder(
            in_channels=num_hiddens,  # Input channels for decoder matches hidden dims
            num_hiddens=num_hiddens,
            num_residual_blocks=num_residual_blocks,
            num_residual_hiddens=num_residual_hiddens
        )

        # Initialize LPIPS only if used
        self.lpips = LPIPS() if self.perceptual_weight > 0 else None

        if self.compute_fid:
            self.test_fid = torchmetrics.image.FrechetInceptionDistance(
                feature=64,
                normalize=True
            )
        else:
            self.test_fid = None

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)

        # Save hyperparameters for logging
        self.save_hyperparameters()

    def encode(self, x):
        """
        Encode input to latent representation
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            z_q: Quantized latent representation
            indices: Indices of the codebook entries
        """
        z = self.encoder(x)
        z = self.pre_bottleneck(z)
        z_q, quantization_loss, _, _ = self.vector_quantizer(z)
        return z_q, quantization_loss
    
    def decode(self, z_q):
        """
        Decode latent representation to reconstruction
        
        Args:
            z_q: Quantized latent representation
        
        Returns:
            x_recon: Reconstructed input
        """
        z_q = self.post_bottleneck(z_q)
        x_recon = self.decoder(z_q)
        return x_recon

    def forward(self, x):
        z = self.encoder(x)
        z = self.pre_bottleneck(z)
        z_q, vq_loss, perplexity, encodings = self.vector_quantizer(z)
        z_q = self.post_bottleneck(z_q)
        recon = self.decoder(z_q)

        # Return as tuple instead of dict
        return recon, vq_loss, perplexity

    def compute_metrics(self, batch, prefix='train'):
        """Compute all metrics for training, validation, and test steps.

        Args:
            batch: Input batch of data
            prefix: Metric prefix ('train', 'val', or 'test')

        Returns:
            dict: Dictionary containing all computed metrics, losses, and reconstructed images
        """
        # Unpack the batch - ensure we get a single tensor
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        
        # Forward pass
        recon, vq_loss, perplexity = self(x)

        # Compute reconstruction loss (ensure it's a scalar)
        recon_loss = F.mse_loss(recon, x).mean()

        # Compute perceptual loss only if enabled
        if self.perceptual_weight > 0 and self.lpips is not None:
            x_norm = x * 2.0 - 1.0
            x_recon_norm = recon * 2.0 - 1.0
            perceptual_loss = self.lpips(x_recon_norm, x_norm).mean()
        else:
            perceptual_loss = torch.zeros((), device=x.device, dtype=recon.dtype)

        # Compute total loss
        total_loss = (1 - self.perceptual_weight) * recon_loss + vq_loss + self.perceptual_weight * perceptual_loss

        # Special handling for test metrics
        if prefix == 'test':
            if self.test_fid is not None:
                x_recon_fid = torch.clamp(recon, 0, 1)
                x_fid = torch.clamp(x, 0, 1)
                self.test_fid.update(x_recon_fid, real=False)
                self.test_fid.update(x_fid, real=True)

        # Log all metrics (now they're all scalars)
        self.log(f'{prefix}/loss', total_loss, on_step=True, on_epoch=True)
        self.log(f'{prefix}/recon_loss', recon_loss, on_step=True, on_epoch=True)
        self.log(f'{prefix}/vq_loss', vq_loss, on_step=True, on_epoch=True)
        self.log(f'{prefix}/perceptual_loss', perceptual_loss, on_step=True, on_epoch=True)
        self.log(f'{prefix}/perplexity', perplexity, on_step=True, on_epoch=True)

        # Add PSNR calculation
        psnr = self.psnr(x, recon)
        self.log(f'{prefix}/psnr', psnr, on_step=True, on_epoch=True)

        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'vq_loss': vq_loss,
            'perceptual_loss': perceptual_loss,
            'perplexity': perplexity,
            'psnr': psnr,
            'x': x,
            'x_recon': recon
        }

    def training_step(self, batch, batch_idx):
        """Perform the training step."""
        metrics = self.compute_metrics(batch, prefix='train')

        if batch_idx % self.log_images_every_n_steps == 0:
            self._log_images(metrics['x'], metrics['x_recon'], split='train')

        return metrics

    def validation_step(self, batch, batch_idx):
        """Perform the validation step."""
        metrics = self.compute_metrics(batch, prefix='val')

        if batch_idx % self.log_images_every_n_steps == 0:
            self._log_images(metrics['x'], metrics['x_recon'], split='val')

        return metrics

    def test_step(self, batch, batch_idx):
        """Perform the test step."""
        # Compute all metrics
        metrics = self.compute_metrics(batch, prefix='test')

        # Log images periodically
        if batch_idx % self.log_images_every_n_steps == 0:
            self._log_images(metrics['x'], metrics['x_recon'], split='test')

        # Existing FID handling
        if self.test_fid is not None:
            fid_score = self.test_fid.compute()
            self.log('test/fid', fid_score)
            self.test_fid.reset()

        return metrics

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Store the scheduler as an attribute
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss_epoch",  # Metric to monitor
            }
        }

    def _log_images(self, x, x_recon, split='train'):
        """
        Log images to Weights & Biases.

        Args:
            x (torch.Tensor): Original images
            x_recon (torch.Tensor): Reconstructed images
            split (str): Data split (train/val/test)
        """
        # Take first 16 images
        x = x[:32]
        x_recon = x_recon[:32]

        # Create grids with smaller size
        x_grid = torchvision.utils.make_grid(
            x,
            nrow=8,
            normalize=True,
            value_range=(-1, 1)
        )

        x_recon_grid = torchvision.utils.make_grid(
            x_recon,
            nrow=8,
            normalize=True,
            value_range=(-1, 1)
        )

        # Convert to numpy and transpose to correct format (H,W,C)
        x_grid = x_grid.cpu().numpy().transpose(1, 2, 0)
        x_recon_grid = x_recon_grid.cpu().numpy().transpose(1, 2, 0)

        # Log to wandb using the experiment attribute
        self.logger.experiment.log({
            f"{split}/images": [
                wandb.Image(x_grid, caption="Original"),
                wandb.Image(x_recon_grid, caption="Reconstructed")
            ],
            f"{split}/reconstruction_error": F.mse_loss(x_recon, x).item(),
            "global_step": self.global_step
        })


# test the VQVAE model
if __name__ == "__main__":
    vqvae = VQVAE(in_channels=3, num_hiddens=128, num_residual_blocks=2, num_residual_hiddens=32, num_embeddings=1024, embedding_dim=32, commitment_cost=0.25, decay=0.99, perceptual_weight=0.1, learning_rate=1e-4, beta=1.0, compute_fid=True)
    x = torch.randn(4, 3, 256, 256)  # batch_size x 3 x 256 x 256
    print(vqvae(x).shape)  # batch_size x 3 x 256 x 256





