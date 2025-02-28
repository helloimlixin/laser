import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import Encoder
from models.decoder import Decoder
from models.bottleneck import VectorQuantizer

class VQVAE(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=[128, 256, 512], 
                 num_embeddings=512, embedding_dim=64, 
                 n_residual_blocks=2, commitment_cost=0.25):
        super().__init__()
        
        self.encoder = Encoder(in_channels, hidden_dims, n_residual_blocks)
        self.pre_quantization = nn.Conv2d(hidden_dims[-1], embedding_dim, kernel_size=1)
        self.vector_quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.post_quantization = nn.Conv2d(embedding_dim, hidden_dims[-1], kernel_size=1)
        self.decoder = Decoder(in_channels, hidden_dims, n_residual_blocks)
    
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
    
    def calculate_loss(self, x, beta=1.0):
        """
        Calculate loss
        
        Args:
            x: Input tensor [B, C, H, W]
            beta: Weight for the commitment loss
        
        Returns:
            loss: Total loss
            recon_loss: Reconstruction loss
            vq_loss: Vector quantization loss
        """
        # Forward pass
        x_recon, vq_loss, _ = self(x)
        
        # Reconstruction loss (mean squared error)
        recon_loss = F.mse_loss(x_recon, x)
        
        # Total loss
        loss = recon_loss + beta * vq_loss
        
        return loss, recon_loss, vq_loss
