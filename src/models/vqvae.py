import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import torchvision

from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.models.bottleneck import VectorQuantizer

import torchmetrics
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
import wandb

class VQVAE(pl.LightningModule):
    def __init__(self, in_channels=3, hidden_dims=None,
                 num_embeddings=512, embedding_dim=64,
                 n_residual_blocks=2, commitment_cost=0.25, decay=0.99,
                 perceptual_weight=1.0, learning_rate=1e-3, beta=1.0, log_images_every_n_steps=100):
        super().__init__()

        # Save hyperparameters for easy checkpointing and reproducibility
        if hidden_dims is None:
            hidden_dims = [128, 256, 512]
        self.save_hyperparameters()
        
        # Model components
        self.encoder = Encoder(in_channels, hidden_dims, n_residual_blocks)
        self.pre_quantization = nn.Conv2d(hidden_dims[-1], embedding_dim, kernel_size=1)
        self.vector_quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost, decay=decay)
        self.post_quantization = nn.Conv2d(embedding_dim, hidden_dims[-1], kernel_size=1)
        self.decoder = Decoder(in_channels, hidden_dims, n_residual_blocks)
        
        # Training parameters
        self.learning_rate = learning_rate
        self.beta = beta
        self.log_images_every_n_steps = log_images_every_n_steps
        self.perceptual_weight = perceptual_weight

        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')

    
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
        z = self.pre_quantization(z)
        z_q, _, indices = self.vector_quantizer(z)
        return z_q, indices
    
    def decode(self, z_q):
        """
        Decode latent representation to reconstruction
        
        Args:
            z_q: Quantized latent representation
        
        Returns:
            x_recon: Reconstructed input
        """
        z_q = self.post_quantization(z_q)
        x_recon = self.decoder(z_q)
        return x_recon
    
    def decode_indices(self, indices, latent_shape):
        """
        Decode indices to reconstruction
        
        Args:
            indices: Indices of the codebook entries [B, H, W]
            latent_shape: Shape of the latent representation [H, W]
        
        Returns:
            x_recon: Reconstructed input
        """
        # Reshape indices to match latent shape
        indices = indices.view(-1, *latent_shape)
        
        # Get codebook entries
        z_q = self.vector_quantizer.get_codebook_entry(indices)
        z_q = z_q.view(-1, self.vector_quantizer.embedding_dim, *latent_shape)
        
        # Decode
        return self.decode(z_q)

    def forward(self, x):
        z = self.encoder(x)
        z = self.pre_quantization(z)
        z_q, vq_loss, perplexity = self.vector_quantizer(z)
        z_q = self.post_quantization(z_q)
        recon = self.decoder(z_q)

        # Return as tuple instead of dict
        return recon, vq_loss, perplexity

    def compute_metrics(self, batch, output, stage='train'):
        metrics_dict = {}
        x = batch
        recon = output['recon']

        # Get the appropriate metrics dictionary
        metrics = self.train_metrics if stage == 'train' else self.val_metrics
        perplexity_tracker = self.train_perplexity if stage == 'train' else self.val_perplexity
        vq_loss_tracker = self.train_vq_loss if stage == 'train' else self.val_vq_loss
        fid_metric = self.train_fid if stage == 'train' else self.val_fid

        # Update per-batch metrics
        metrics[f'{stage}_mse'](recon, x)
        metrics[f'{stage}_psnr'](recon, x)
        metrics[f'{stage}_ssim'](recon, x)
        metrics[f'{stage}_lpips'](recon, x)

        # Update FID metric
        # Ensure images are in [0, 1] range
        if x.min() < 0 or x.max() > 1:
            x = (x + 1) / 2  # Assuming [-1, 1] range
            recon = (recon + 1) / 2

        # FID expects images in [0, 1] and of shape [B, C, H, W]
        fid_metric.update(x, real=True)
        fid_metric.update(recon, real=False)

        # Track perplexity and vq_loss
        perplexity_tracker(output['perplexity'])
        vq_loss_tracker(output['vq_loss'])

        return metrics_dict

    def training_step(self, batch, batch_idx):
        images, labels = batch

        # Forward pass
        recon, vq_loss, perplexity = self.forward(images)

        # Compute reconstruction loss
        recon_loss = F.mse_loss(recon, images)

        # Normalize images to [-1, 1] for LPIPS if they're not already normalized
        recon_normalized = recon * 2 - 1
        recon_normalized = torch.clamp(recon_normalized, -1, 1)
        images_normalized = images * 2 - 1
        images_normalized = torch.clamp(images_normalized, -1, 1)

        # Compute perceptual loss
        perceptual_loss = self.lpips(recon_normalized, images_normalized)

        # Combine losses
        total_loss = (1 - self.perceptual_weight) * recon_loss + vq_loss + self.perceptual_weight * perceptual_loss

        # Update metrics
        self.train_perplexity.update(perplexity.float().mean())
        self.train_vq_loss.update(vq_loss)
        self.train_recon_loss.update(recon_loss)
        self.train_perceptual_loss.update(perceptual_loss)

        # Log metrics
        self.log('train/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/recon_loss', recon_loss, on_step=True, on_epoch=True)
        self.log('train/perceptual_loss', perceptual_loss, on_step=True, on_epoch=True)
        self.log('train/vq_loss', vq_loss, on_step=True, on_epoch=True)
        self.log('train/perplexity', perplexity.float().mean(), on_step=True, on_epoch=True)

        # Log sample images periodically
        if batch_idx == 0:
            self._log_images(images, recon, split='train')

        return total_loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        # Forward pass
        recon, vq_loss, perplexity = self.forward(images)

        # Compute reconstruction loss
        recon_loss = F.mse_loss(recon, images)

        # Normalize images to [-1, 1] for LPIPS
        recon_normalized = recon * 2 - 1
        recon_normalized = torch.clamp(recon_normalized, -1, 1)
        images_normalized = images * 2 - 1
        images_normalized = torch.clamp(images_normalized, -1, 1)

        # Compute perceptual loss with normalized images
        perceptual_loss = self.lpips(recon_normalized, images_normalized)

        total_loss = (1 - self.perceptual_weight) * recon_loss + vq_loss + self.perceptual_weight * perceptual_loss

        # Calculate PSNR
        psnr = PeakSignalNoiseRatio(data_range=1.0).to(images.device)(recon, images)
        self.log('val/psnr', psnr, on_step=True, on_epoch=True, prog_bar=True)

        # Update metrics
        self.val_perplexity.update(perplexity.float().mean())
        self.val_vq_loss.update(vq_loss)
        self.val_recon_loss.update(recon_loss)
        self.val_perceptual_loss.update(perceptual_loss)

        # Log metrics
        self.log('val/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val/recon_loss', recon_loss, on_step=True, on_epoch=True)
        self.log('val/perceptual_loss', perceptual_loss, on_step=True, on_epoch=True)
        self.log('val/vq_loss', vq_loss, on_step=True, on_epoch=True)
        self.log('val/perplexity', perplexity.float().mean(), on_step=True, on_epoch=True)

        # Log sample images
        if batch_idx == 0:
            self._log_images(images, recon, split='val')

        return total_loss

    def test_step(self, batch, batch_idx):
        """Perform the test step.

        Args:
            batch: Input batch of data
            batch_idx: Index of the current batch

        Returns:
            dict: Dictionary containing test metrics
        """
        # Unpack batch and perform forward pass
        images, labels = batch
        # Forward pass
        recon, vq_loss, perplexity = self.forward(images)

        # Compute reconstruction loss
        recon_loss = F.mse_loss(recon, images)

        # Normalize images to [-1, 1] for LPIPS
        recon_normalized = recon * 2 - 1
        recon_normalized = torch.clamp(recon_normalized, -1, 1)
        images_normalized = images * 2 - 1
        images_normalized = torch.clamp(images_normalized, -1, 1)

        perceptual_loss = self.lpips(recon_normalized, images_normalized)

        # Combine losses
        total_loss = recon_loss + vq_loss + self.perceptual_weight * perceptual_loss

        # Calculate PSNR
        psnr = PeakSignalNoiseRatio(data_range=1.0).to(images.device)(recon, images)
        self.log('test/psnr', psnr, on_step=True, on_epoch=True, prog_bar=True)

        # Update test metrics
        self.test_metrics.update(total_loss)

        # Update FID if enabled
        if self.test_fid is not None:
            # Ensure images are in [0, 1] range
            x_recon_fid = torch.clamp(recon, 0, 1)
            x_fid = torch.clamp(images, 0, 1)
            self.test_fid.update(x_recon_fid, real=False)
            self.test_fid.update(x_fid, real=True)

        # Log step metrics
        self.log('test/loss', total_loss, on_step=True, on_epoch=True)
        self.log('test/recon_loss', recon_loss, on_step=True, on_epoch=True)
        self.log('test/vq_loss', vq_loss, on_step=True, on_epoch=True)
        self.log('test/perceptual_loss', perceptual_loss, on_step=True, on_epoch=True)
        self.log('test/perplexity', perplexity, on_step=True, on_epoch=True)

        # Log images periodically
        if batch_idx % self.log_images_every_n_steps == 0:
            self._log_images(images, recon, split='test')

        return {
            'test_loss': total_loss,
            'recon_loss': recon_loss,
            'vq_loss': vq_loss,
            'perceptual_loss': perceptual_loss,
            'perplexity': perplexity
        }

    def setup(self, stage=None):
        """Setup metrics for each stage of training."""
        if stage == 'fit' or stage is None:
            # Training metrics
            self.train_perplexity = torchmetrics.MeanMetric()
            self.train_vq_loss = torchmetrics.MeanMetric()
            self.train_recon_loss = torchmetrics.MeanMetric()  # Add reconstruction loss metric
            self.train_perceptual_loss = torchmetrics.MeanMetric()  # Add perceptual loss metric

            # Validation metrics
            self.val_perplexity = torchmetrics.MeanMetric()
            self.val_vq_loss = torchmetrics.MeanMetric()
            self.val_recon_loss = torchmetrics.MeanMetric()  # Add reconstruction loss metric
            self.val_perceptual_loss = torchmetrics.MeanMetric()  # Add perceptual loss metric

        if stage == 'test' or stage is None:
            # Test metrics
            self.test_metrics = torchmetrics.MeanMetric()

            # Initialize FID for test set if needed
            if self.compute_fid:
                self.test_fid = torchmetrics.image.FrechetInceptionDistance(
                    feature=2048,
                    normalize=True
                )
            else:
                self.test_fid = None

    def on_train_epoch_end(self):
        """Log epoch-level metrics for training."""
        # Compute and log the epoch-averaged metrics
        perplexity = self.train_perplexity.compute()
        vq_loss = self.train_vq_loss.compute()

        # Log metrics
        self.log('train/epoch_perplexity', perplexity)
        self.log('train/epoch_vq_loss', vq_loss)

        # Reset metrics
        self.train_perplexity.reset()
        self.train_vq_loss.reset()

    def on_validation_epoch_end(self):
        """Log epoch-level metrics for validation."""
        # Log the epoch-averaged metrics
        perplexity = self.val_perplexity.compute()
        vq_loss = self.val_vq_loss.compute()

        # Log metrics
        self.log('val/epoch_perplexity', perplexity)
        self.log('val/epoch_vq_loss', vq_loss)

        # Reset metrics
        self.val_perplexity.reset()
        self.val_vq_loss.reset()
        self.lpips.reset()  # Reset LPIPS metric

    def on_test_epoch_end(self):
        """Log epoch-level metrics for testing."""
        # Log final test metrics
        test_loss = self.test_metrics.compute()
        self.log('test/epoch_loss', test_loss)

        # Log FID if available
        if self.test_fid is not None:
            fid_score = self.test_fid.compute()
            self.log('test/fid', fid_score)

        # Reset metrics
        self.test_metrics.reset()
        if self.test_fid is not None:
            self.test_fid.reset()
        self.lpips.reset()  # Reset LPIPS metric

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Store the scheduler as an attribute
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": self.lr_scheduler,
                "monitor": "val/loss",  # Metric to monitor
                "interval": "epoch",
                "frequency": 100
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
            value_range=(-1, 1),
            pad_value=1
        )

        x_recon_grid = torchvision.utils.make_grid(
            x_recon,
            nrow=8,
            normalize=True,
            value_range=(-1, 1),
            pad_value=1
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






