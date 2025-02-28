import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.models.bottleneck import VectorQuantizer
import wandb

# Replace torchvision.utils.make_grid with a custom implementation
def make_grid(tensor, nrow=8, padding=2, normalize=False, value_range=None, scale_each=False, pad_value=0):
    """
    Simple implementation of torchvision's make_grid function to avoid dependency issues
    """
    if value_range is not None:
        if len(value_range) != 2:
            raise ValueError("value_range has to be a tuple (min, max)")
        vmin, vmax = value_range
    elif normalize:
        vmin, vmax = tensor.min(), tensor.max()
    else:
        vmin, vmax = 0, 1
    
    if normalize:
        tensor = (tensor - vmin) / (vmax - vmin)
    
    # Make a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(nmaps / xmaps + 0.5)
    height, width = tensor.size(2) + padding, tensor.size(3) + padding
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[:, y * height + padding:(y + 1) * height, x * width + padding:(x + 1) * width] = tensor[k]
            k += 1
    
    return grid

class VQVAE(pl.LightningModule):
    def __init__(self, in_channels=3, hidden_dims=None,
                 num_embeddings=512, embedding_dim=64,
                 n_residual_blocks=2, commitment_cost=0.25,
                 learning_rate=1e-3, beta=1.0, log_images_every_n_steps=100):
        super().__init__()
        
        # Save hyperparameters for easy checkpointing and reproducibility
        if hidden_dims is None:
            hidden_dims = [128, 256, 512]
        self.save_hyperparameters()
        
        # Model components
        self.encoder = Encoder(in_channels, hidden_dims, n_residual_blocks)
        self.pre_quantization = nn.Conv2d(hidden_dims[-1], embedding_dim, kernel_size=1)
        self.vector_quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
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
        
        return loss, recon_loss, vq_loss
    
    # PyTorch Lightning methods
    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_recon, vq_loss, _ = self(x)  # Unpack the tuple correctly
        loss, recon_loss, _ = self.calculate_loss(x)

        # Enhanced logging
        self.log_dict({
            'train/total_loss': loss,
            # 'train/recon_loss': recon_loss,
            # 'train/vq_loss': vq_loss,
        }, prog_bar=True, on_step=True, on_epoch=True)

        # Log images periodically
        if batch_idx == 0:  # Log first batch of each epoch
            self.logger.experiment.log({
                "train/reconstructions": [
                    wandb.Image(
                        x_i,
                        caption=f"Epoch {self.current_epoch}"
                    ) for x_i in x_recon[:4]  # Use x_recon instead of x_hat
                ],
                "train/originals": [
                    wandb.Image(
                        x_i,
                        caption=f"Epoch {self.current_epoch}"
                    ) for x_i in x[:4]
                ]
            })

        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch  # Ignore labels
        
        # Calculate loss
        loss, recon_loss, vq_loss = self.calculate_loss(x)
        
        # log validation metrics
        metrics = {
            'val_loss': loss,
            # 'val_recon_loss': recon_loss,
            # 'val_vq_loss': vq_loss
        }
        
        # Log images from the first batch
        if batch_idx % self.log_images_every_n_steps == 0:
            x_recon = self._log_images(x, step_name='val')

            # Calculate PSNR
            mse = F.mse_loss(x_recon, x)
            psnr = -10 * torch.log10(mse)
            metrics['val_psnr'] = psnr.item()

        self.log("val_loss", metrics['val_loss'], prog_bar=True, sync_dist=True)
        if 'val_psnr' in metrics:
            self.log("val_psnr", metrics['val_psnr'], prog_bar=True, sync_dist=True)
        
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

    def _log_images(self, x, step_name='train'):
        """Log original and reconstructed images"""
        # Get reconstructions
        x_recon, _, _ = self(x)

        # Select up to 8 images to avoid cluttering the logs
        n_images = min(8, x.size(0))

        # Create pairs of original and reconstructed images
        images = []
        for idx in range(n_images):
            # Convert tensors to images that wandb can handle
            original = x[idx].detach().cpu()
            recon = x_recon[idx].detach().cpu()

            # Normalize to [0, 1] range if needed
            if original.min() < 0:
                original = (original + 1) / 2
                recon = (recon + 1) / 2

            # Create wandb image with both original and reconstruction
            images.append(wandb.Image(
                torch.cat([original, recon], dim=2),  # Concatenate horizontally
                caption=f"{step_name} - Original vs Reconstruction"
            ))

        # Log to wandb
        self.logger.experiment.log({
            f"{step_name}_reconstructions": images,
            "global_step": self.global_step
        })

        return x_recon

