import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    Vector Quantizer implementation for VQ-VAE.
    
    The Vector Quantizer maps continuous encodings to discrete codes from a learned codebook.
    This is the key component that enables VQ-VAE to learn discrete representations.
    
    The quantization process involves:
    1. Finding the nearest embedding vector in the codebook for each spatial position in the input
    2. Replacing the input vectors with their corresponding codebook vectors
    3. Computing loss terms to train both the encoder and the codebook
    4. Using a straight-through estimator to allow gradient flow through the discrete bottleneck
    
    The codebook can be updated using either:
    - EMA updates (when decay > 0): More stable, not directly influenced by optimizer
    - Gradient descent (when decay = 0): Standard backpropagation approach
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25,
                 decay=0.99, epsilon=1e-5):
        """
        Initialize the Vector Quantizer.
        
        Args:
            num_embeddings (int): Size of the embedding dictionary (codebook size, typically 512 or 1024)
            embedding_dim (int): Dimension of each embedding vector in the codebook
            commitment_cost (float): Weight for the commitment loss, balancing encoder vs codebook training
            decay (float): Decay factor for exponential moving average updates of embeddings
                           If 0, standard backpropagation is used to update embeddings
            epsilon (float): Small constant to prevent division by zero in EMA update normalization
        """
        super().__init__()

        self.embedding_dim = embedding_dim  # Dimension of each embedding vector
        self.num_embeddings = num_embeddings  # Number of embedding vectors in the codebook
        self.commitment_cost = commitment_cost  # Weighting for commitment loss
        self.use_ema = bool(decay > 0.0)  # Whether to use EMA updates for the codebook
        self.decay = decay  # EMA decay factor (higher = slower updates)
        self.epsilon = epsilon  # Small constant for numerical stability

        # Create embedding table (codebook)
        # This is the dictionary of codes that continuous vectors will be mapped to
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # Initialize embedding weights with small random values
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

        if self.use_ema:
            print('Using EMA updates for codebook...')
            # Register buffers for EMA updates (not model parameters - not directly optimized)
            # ema_cluster_size: Tracks how often each codebook entry is used
            self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
            # ema_w: EMA of the encoder outputs assigned to each codebook entry
            self.register_buffer('ema_w', self.embedding.weight.data.clone())
            # Flag to control when EMA updates are performed
            self.register_buffer('ema_updating', torch.ones(1))

    def forward(self, z):
        """
        Forward pass through the vector quantizer.
        
        Args:
            z (torch.Tensor): Output from encoder with shape [B, D, H, W]
                              B: batch size, D: embedding dimension, H, W: spatial dimensions
        
        Returns:
            z_q (torch.Tensor): Quantized tensor with same shape as input [B, D, H, W]
            loss (torch.Tensor): VQ loss (codebook loss + commitment loss)
            min_encoding_indices (torch.Tensor): Indices of the nearest embedding vectors [B*H*W]
        """
        # z shape: [B, D, H, W]

        # Reshape z to [B*H*W, D] for easier processing
        # The permute operation reorders dimensions to [B, H, W, D] before reshaping
        z_flattened = z.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)
        
        # Calculate squared distances between z_flattened and embedding vectors
        # This uses the || x - e ||^2 = ||x||^2 + ||e||^2 - 2*x^T*e formula for efficiency
        # Rather than computing distances directly, which would be more expensive
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())

        # Find nearest embedding for each z_flattened vector
        # min_encoding_indices contains indices of the closest codebook entry for each position
        min_encoding_indices = torch.argmin(d, dim=1)
        
        # Convert indices to one-hot encodings for gathering embeddings
        # min_encodings shape: [B*H*W, num_embeddings]
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.num_embeddings, device=z.device)
        min_encodings.scatter_(1, min_encoding_indices.unsqueeze(1), 1)

        # Update codebook using EMA if enabled and in training mode
        if self.use_ema and self.training:
            # EMA update for the codebook
            
            # 1. Update cluster size (how many vectors are assigned to each codebook entry)
            # Count occurrences of each embedding being used in this batch
            encodings_sum = min_encodings.sum(0)
            # Update the exponential moving average of cluster sizes
            self.ema_cluster_size.data.mul_(self.decay).add_(
                encodings_sum, alpha=(1 - self.decay)
            )

            # 2. Update embedding vectors based on assigned encoder outputs
            # Compute sum of all z vectors assigned to each embedding
            dw = torch.matmul(min_encodings.t(), z_flattened)
            # Update the exponential moving average of assigned vectors
            self.ema_w.data.mul_(self.decay).add_(dw, alpha=(1 - self.decay))

            # 3. Normalize the updated embeddings
            # Get total cluster size (with smoothing to prevent division by zero)
            n = self.ema_cluster_size.sum()
            # Normalize cluster sizes with Laplace smoothing
            cluster_size = ((self.ema_cluster_size + self.epsilon) /
                            (n + self.num_embeddings * self.epsilon) * n)
            # Normalize embeddings by their corresponding cluster sizes
            embed_normalized = self.ema_w / cluster_size.unsqueeze(1)
            # Update the actual embedding weights with the EMA-updated version
            self.embedding.weight.data.copy_(embed_normalized)

        # Get the quantized latent vectors using embedding lookup
        # Multiply one-hot encodings by embedding weights to get quantized vectors
        # Then reshape back to the original tensor shape [B, H, W, D]
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(
            z.shape[0], z.shape[2], z.shape[3], self.embedding_dim)
        # Permute back to match input tensor ordering [B, D, H, W]
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        # Compute VQ losses:
        
        # 1. Codebook loss (makes codebook vectors move towards encoder outputs)
        # The detach() on z prevents gradients from flowing back to the encoder
        vq_loss = F.mse_loss(z_q.detach(), z)
        
        # 2. Commitment loss (makes encoder outputs move towards codebook vectors)
        # The detach() on z_q prevents gradients from flowing back to the codebook
        commitment_loss = F.mse_loss(z_q, z.detach())
        
        # Combine losses with commitment_cost weighting
        loss = vq_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator
        # This allows gradients to flow back to encoder even though quantization is discrete
        # In the forward pass: z_q = selected embeddings
        # In the backward pass: gradients flow directly from z_q to z, bypassing quantization
        z_q = z + (z_q - z).detach()

        return z_q, loss, min_encoding_indices

    def get_codebook_entry(self, indices):
        """
        Retrieve specific vectors from the codebook using their indices.
        
        This is used during inference to map from discrete codes back to latent vectors.
        
        Args:
            indices (torch.Tensor): Indices of the codebook entries to retrieve
        
        Returns:
            z_q (torch.Tensor): The corresponding vectors from the codebook
        """
        # Flatten indices to 1D
        indices = indices.view(-1)
        
        # Convert indices to one-hot encodings
        min_encodings = torch.zeros(indices.shape[0], self.num_embeddings, device=indices.device)
        min_encodings.scatter_(1, indices.unsqueeze(1), 1)
        
        # Get corresponding embedding vectors using matmul with one-hot encodings
        z_q = torch.matmul(min_encodings, self.embedding.weight)
        
        # Reshape to match expected dimensions
        z_q = z_q.view(indices.shape[0], self.embedding_dim)
        
        return z_q

# Add this testing code at the bottom of the file
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Test parameters
    batch_size = 4
    embedding_dim = 64
    num_embeddings = 512
    height, width = 16, 16  # Spatial dimensions for test
    
    # Create a vector quantizer instance (with EMA updates)
    vq_with_ema = VectorQuantizer(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        commitment_cost=0.25,
        decay=0.99  # Using EMA updates
    )
    
    # Create another vector quantizer instance (without EMA updates)
    vq_without_ema = VectorQuantizer(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        commitment_cost=0.25,
        decay=0.0  # No EMA updates
    )
    
    # Generate random encoder output
    z = torch.randn(batch_size, embedding_dim, height, width)
    print(f"Input shape: {z.shape}")
    
    # Test forward pass with EMA updates
    print("\n--- Testing VectorQuantizer with EMA updates ---")
    vq_with_ema.train()  # Set to training mode to enable EMA updates
    z_q_ema, loss_ema, indices_ema = vq_with_ema(z)
    
    print(f"Quantized output shape: {z_q_ema.shape}")
    print(f"VQ Loss: {loss_ema.item():.6f}")
    print(f"Indices shape: {indices_ema.shape}")
    
    # Test codebook usage (perplexity)
    encodings = torch.zeros(indices_ema.shape[0], num_embeddings)
    encodings.scatter_(1, indices_ema.unsqueeze(1), 1)
    avg_probs = torch.mean(encodings, dim=0)
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
    codebook_usage = torch.sum(avg_probs > 0)
    
    print(f"Perplexity: {perplexity.item():.2f} (effective codebook usage)")
    print(f"Codebook entries used: {codebook_usage.item()} out of {num_embeddings}")
    
    # Test forward pass without EMA updates
    print("\n--- Testing VectorQuantizer without EMA updates ---")
    vq_without_ema.train()
    z_q_no_ema, loss_no_ema, indices_no_ema = vq_without_ema(z)
    
    print(f"Quantized output shape: {z_q_no_ema.shape}")
    print(f"VQ Loss: {loss_no_ema.item():.6f}")
    
    # Test get_codebook_entry method
    print("\n--- Testing get_codebook_entry ---")
    # Get the first 10 indices
    test_indices = indices_ema[:10]
    retrieved_embeddings = vq_with_ema.get_codebook_entry(test_indices)
    print(f"Retrieved embeddings shape: {retrieved_embeddings.shape}")
    
    # Verify reconstruction error is minimal (should be zero or very close)
    # Get the original quantized vectors for these indices
    original_flat = z_q_ema.permute(0, 2, 3, 1).contiguous().view(-1, embedding_dim)
    original_subset = original_flat[:10]
    reconstruction_error = F.mse_loss(retrieved_embeddings, original_subset)
    print(f"Reconstruction error: {reconstruction_error.item():.10f} (should be close to zero)")
    
    # Visualize codebook usage
    print("\n--- Visualizing Codebook Usage ---")
    used_indices = indices_ema.cpu().numpy()
    plt.figure(figsize=(10, 5))
    plt.hist(used_indices, bins=50, alpha=0.7)
    plt.title("Codebook Usage Distribution")
    plt.xlabel("Codebook Index")
    plt.ylabel("Frequency")
    plt.savefig("codebook_usage.png")
    print(f"Saved codebook usage visualization to 'codebook_usage.png'")
    
    # Test inference mode
    print("\n--- Testing inference mode ---")
    vq_with_ema.eval()  # Set to evaluation mode
    z_q_eval, loss_eval, indices_eval = vq_with_ema(z)
    
    print(f"Inference mode - Quantized output shape: {z_q_eval.shape}")
    print(f"Inference mode - VQ Loss: {loss_eval.item():.6f}")
    
    print("\nTest completed successfully!")
