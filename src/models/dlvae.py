import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import torchvision

from .encoder import Encoder
from .decoder import Decoder
from .bottleneck import DictionaryLearning
from .lpips import LPIPS

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
            sparsity_level,
            num_residual_blocks,
            num_residual_hiddens,
            commitment_cost,
            decay,
            perceptual_weight,
            learning_rate,
            beta,
            compute_fid=False,
            omp_tolerance=1e-7,
            omp_debug=False
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
            omp_tolerance: Early stopping tolerance for BatchOMP residual
            omp_debug: Enable BatchOMP debug logging
        """
        super(DLVAE, self).__init__()

        # Store parameters
        self.learning_rate = learning_rate
        self.beta = beta
        self.perceptual_weight = perceptual_weight
        self.log_images_every_n_steps = 100
        self.compute_fid = compute_fid
        self.omp_tolerance = omp_tolerance
        self.omp_debug = omp_debug

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
        self.bottleneck = DictionaryLearning(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            sparsity_level=sparsity_level,
            commitment_cost=commitment_cost,
            decay=decay,
            tolerance=self.omp_tolerance,
            omp_debug=self.omp_debug
        )

        self.post_bottleneck = nn.Conv2d(in_channels=embedding_dim,
                                         out_channels=num_hiddens,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1)

        # Initialize decoder
        self.decoder = Decoder(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_blocks=num_residual_blocks,
            num_residual_hiddens=num_residual_hiddens
        )

        # Initialize LPIPS for perceptual loss only if used
        self.lpips = LPIPS() if self.perceptual_weight > 0 else None

        # Initialize the PSNR metric (evaluate on de-normalized [0,1] images)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)

        # Initialize the SSIM metric
        self.ssim = StructuralSimilarityIndexMeasure()

        # Initialize metrics for Frechet Inception Distance (FID Score)
        if self.compute_fid:
            self.test_fid = FrechetInceptionDistance(feature=64, normalize=True)
        else:
            self.test_fid = None

        # Save hyperparameters
        self.save_hyperparameters()

    def encode(self, x):
        """
        Encode input to latent representation.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            z_dl: latent representation reconstructed from the dictionary learning bottleneck
            bottleneck_loss: loss from the dictionary learning bottleneck
        """
        z_e = self.encoder(x)
        z_e = self.pre_bottleneck(z_e)
        z_dl, bottleneck_loss, coefficients = self.bottleneck(z_e)
        return z_dl, bottleneck_loss, coefficients

    def decode(self, z_dl):
        """
        Decode latent representation to reconstruction.

        Args:
            z_dl: latent representation reconstructed from the dictionary learning bottleneck

        Returns:
            x_recon: reconstruction of the input
        """
        z_dl = self.post_bottleneck(z_dl)
        x_recon = self.decoder(z_dl)
        return x_recon

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            tuple: (recon, bottleneck_loss, coefficients)
        """
        z = self.encoder(x)
        z = self.pre_bottleneck(z)
        z_dl, bottleneck_loss, coefficients = self.bottleneck(z)
        z_dl = self.post_bottleneck(z_dl)
        recon = self.decoder(z_dl)

        return recon, bottleneck_loss, coefficients

    def compute_metrics(self, batch, prefix='train'):
        """Compute metrics for a batch."""
        # Get input
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        
        # Forward pass
        recon_raw, dl_loss, coefficients = self(x)
        # Keep raw tensors for loss; create sanitized copies for metrics/visualization only
        recon_vis = torch.nan_to_num(recon_raw.detach(), nan=0.0, posinf=1.0, neginf=-1.0).clamp_(-1.0, 1.0)
        x_vis = torch.nan_to_num(x.detach(), nan=0.0, posinf=1.0, neginf=-1.0).clamp_(-1.0, 1.0)
        
        # Compute losses
        recon_loss = F.mse_loss(recon_raw, x).mean()
        
        # Perceptual loss (optional)
        if self.perceptual_weight > 0 and self.lpips is not None:
            x_norm = x * 2.0 - 1.0
            x_recon_norm = recon_raw * 2.0 - 1.0
            perceptual_loss = self.lpips(x_recon_norm, x_norm).mean()
        else:
            perceptual_loss = torch.tensor(0.0, device=self.device, dtype=recon_raw.dtype)
        
        # Total loss
        total_loss = (1 - self.perceptual_weight) * recon_loss + dl_loss + self.perceptual_weight * perceptual_loss

        # Handle FID for test
        if prefix == 'test' and self.test_fid is not None:
            x_recon_fid = torch.clamp(recon_raw, 0, 1)
            x_fid = torch.clamp(x, 0, 1)
            self.test_fid.update(x_recon_fid, real=False)
            self.test_fid.update(x_fid, real=True)

        # Log metrics (sync across devices for epoch-level aggregation)
        self.log(f'{prefix}/loss', total_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f'{prefix}/recon_loss', recon_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f'{prefix}/dl_loss', dl_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f'{prefix}/perceptual_loss', perceptual_loss, on_step=True, on_epoch=True, sync_dist=True)

        # Add PSNR calculation on de-normalized [0,1] images to avoid scale drift
        dm = getattr(getattr(self, "trainer", None), "datamodule", None)
        if dm is not None and hasattr(dm, "config") and hasattr(dm.config, "mean") and hasattr(dm.config, "std"):
            mean = torch.tensor(dm.config.mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
            std = torch.tensor(dm.config.std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
            x_dn = (x * std + mean).clamp(0.0, 1.0)
            recon_dn = (recon_raw * std + mean).clamp(0.0, 1.0)
        else:
            x_dn = ((x + 1.0) / 2.0).clamp(0.0, 1.0)
            recon_dn = ((recon_raw + 1.0) / 2.0).clamp(0.0, 1.0)
        psnr = self.psnr(x_dn, recon_dn)
        self.log(f'{prefix}/psnr', psnr, on_step=True, on_epoch=True, sync_dist=True)

        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'dl_loss': dl_loss,
            'perceptual_loss': perceptual_loss,
            'psnr': psnr,
            'x': x_vis,
            'x_recon': recon_vis
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

        if self.test_fid is not None:
            fid_score = self.test_fid.compute()
            self.log("test/fid", fid_score, on_epoch=True)
            self.test_fid.reset()
            
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss_epoch"
            }
        }

    def _log_images(self, x, x_recon, split='train'):
        """Log images to wandb."""
        # Only log from rank zero in DDP to avoid multi-process logger contention
        if getattr(getattr(self, "trainer", None), "is_global_zero", False) is False:
            return
        # If logger is disabled or doesn't support experiment logging, skip
        if not getattr(self, "logger", None) or not hasattr(self.logger, "experiment"):
            return
        
        # Take first 32 images
        x = x[:32]
        x_recon = x_recon[:32]

        # Create image grids
        # De-normalize using datamodule config if available; otherwise assume [-1,1] → [0,1]
        dm = getattr(getattr(self, "trainer", None), "datamodule", None)
        if dm is not None and hasattr(dm, "config") and hasattr(dm.config, "mean") and hasattr(dm.config, "std"):
            mean = torch.tensor(dm.config.mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
            std = torch.tensor(dm.config.std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
            x_disp = x * std + mean
            x_recon_disp = x_recon * std + mean
        else:
            x_disp = (x + 1.0) / 2.0
            x_recon_disp = (x_recon + 1.0) / 2.0
        x_disp = x_disp.clamp(0.0, 1.0)
        x_recon_disp = x_recon_disp.clamp(0.0, 1.0)
        x_grid = torchvision.utils.make_grid(x_disp, nrow=8, normalize=False)
        x_recon_grid = torchvision.utils.make_grid(x_recon_disp, nrow=8, normalize=False)

        # Sanitize NaN/Inf and clamp to [0,1] before converting to numpy
        x_grid = torch.nan_to_num(x_grid, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        x_recon_grid = torch.nan_to_num(x_recon_grid, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)

        # Convert to numpy
        x_grid = x_grid.cpu().numpy().transpose(1, 2, 0)
        x_recon_grid = x_recon_grid.cpu().numpy().transpose(1, 2, 0)

        # Log to wandb
        self.logger.experiment.log({
            f"{split}/images": [
                wandb.Image(x_grid, caption="Original"),
                wandb.Image(x_recon_grid, caption="Reconstructed")
            ],
            f"{split}/reconstruction_error": F.mse_loss(x_recon, x).item(),
            "global_step": self.global_step
        })
