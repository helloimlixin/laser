import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import torchvision

from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.models.bottleneck import VectorQuantizer
import wandb

class VQVAE(pl.LightningModule):
    def __init__(self, in_channels=3, hidden_dims=None,
                 num_embeddings=512, embedding_dim=64,
                 n_residual_blocks=2, commitment_cost=0.25, decay=0.99,
                 learning_rate=1e-3, beta=1.0, log_images_every_n_steps=100):
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
        """
        Forward pass
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            x_recon: Reconstructed input
            vq_loss: Vector quantization loss
            indices: Indices of the codebook entries
        """
        # Encode
        z = self.encoder(x)
        z = self.pre_quantization(z)
        
        # Vector quantization
        z_q, vq_loss, indices = self.vector_quantizer(z)
        
        # Decode
        z_q = self.post_quantization(z_q)
        x_recon = self.decoder(z_q)
        
        return x_recon, vq_loss, indices
    
    def calculate_loss(self, x, beta=None):
        """
        Calculate loss
        
        Args:
            x: Input tensor [B, C, H, W]
            beta: Weight for the commitment loss (overrides self.beta if provided)
        
        Returns:
            loss: Total loss
            recon_loss: Reconstruction loss
            vq_loss: Vector quantization loss
        """
        # Use provided beta or default to self.beta
        beta = beta if beta is not None else self.beta
        
        # Forward pass
        x_recon, vq_loss, _ = self(x)
        
        # Reconstruction loss (mean squared error)
        recon_loss = F.mse_loss(x_recon, x)
        
        # Total loss
        loss = recon_loss + beta * vq_loss
        
        return loss, x_recon, recon_loss, vq_loss
    
    # PyTorch Lightning methods
    def training_step(self, batch, batch_idx):
        x, _ = batch
        loss, x_recon, recon_loss, vq_loss = self.calculate_loss(x)

        # Enhanced logging
        self.log_dict({
            'train/total_loss': loss,
            # 'train/recon_loss': recon_loss,
            # 'train/vq_loss': vq_loss,
        }, prog_bar=True, on_step=True, on_epoch=True)

        # Log images periodically
        if batch_idx == 0:  # Log first batch of each epoch
            self._log_images(x, x_recon, split='train')

        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch  # Ignore labels
        
        # Calculate loss
        loss, x_recon, recon_loss, vq_loss = self.calculate_loss(x)

        # Calculate PSNR
        mse = F.mse_loss(x_recon, x)
        psnr = -10 * torch.log10(mse)
        
        # log validation metrics
        metrics = {
            'val_loss': loss,
            'val_psnr': psnr.item(),
            # 'val_recon_loss': recon_loss,
            # 'val_vq_loss': vq_loss
        }
        
        # Log images from the first batch
        if batch_idx == 0:
            self._log_images(x, x_recon, split='val')

        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, _ = batch  # Ignore labels
        
        # Calculate loss
        loss, recon_loss, vq_loss = self.calculate_loss(x)
        
        # Log metrics
        self.log('test_loss', loss)
        self.log('test_recon_loss', recon_loss)
        self.log('test_vq_loss', vq_loss)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def _log_images(self, x, x_recon, split='train'):
        """
        Log images to Weights & Biases.

        Args:
            x (torch.Tensor): Original images
            x_recon (torch.Tensor): Reconstructed images
            split (str): Data split (train/val/test)
        """
        # Take first 8 images
        x = x[:8]
        x_recon = x_recon[:8]

        # Create grids with smaller size
        x_grid = torchvision.utils.make_grid(
            x,
            nrow=4,
            normalize=True,
            value_range=(-1, 1),
            pad_value=1
        )

        x_recon_grid = torchvision.utils.make_grid(
            x_recon,
            nrow=4,
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






