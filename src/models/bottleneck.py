import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import math

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
        else:
            print('Using standard backpropagation for codebook... (no EMA)')

        # Enable TF32 math for faster operations on modern GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Pre-allocate buffers for OMP
        self.register_buffer('_corr_buffer', torch.Tensor())
        self.register_buffer('_residual_buffer', torch.Tensor())

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

        # Add explicit memory logging
        # torch.cuda.synchronize()
        # print(f"Allocated: {torch.cuda.memory_allocated()/1e6:.2f}MB")

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


class DictionaryLearningBottleneck(nn.Module):
    """
    Dictionary Learning Bottleneck with vectorized Batch Orthogonal Matching Pursuit (OMP) sparse coding.
    """
    def __init__(
        self,
        dict_size=512,
        embedding_dim=64,
        sparsity=5,
        commitment_cost=0.25,
        decay=0.99,
        epsilon=1e-10,
        use_ema=True,
    ):
        super().__init__()
        
        self.dict_size = dict_size
        self.embedding_dim = embedding_dim
        self.sparsity = sparsity
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.use_ema = use_ema
        
        # Initialize buffers for OMP
        self.register_buffer('_corr_buffer', torch.Tensor())
        self.register_buffer('_residual_buffer', torch.Tensor())
        self.register_buffer('_coefficients_buffer', torch.Tensor())
        self.register_buffer('_active_set_buffer', torch.Tensor())
        self.register_buffer('_L_buffer', torch.Tensor())  # For Cholesky
        
        # Initialize dictionary with random atoms
        self.dictionary = nn.Parameter(
            torch.randn(embedding_dim, dict_size), requires_grad=True
        )
        self._normalize_dictionary()

    def _normalize_dictionary(self):
        """Normalize all dictionary atoms to have unit L2 norm."""
        with torch.no_grad():
            norms = torch.norm(self.dictionary, p=2, dim=0, keepdim=True)
            self.dictionary.data = self.dictionary.data / norms

    def batch_omp(self, signals, dictionary, debug=False):
        device = signals.device
        batch_size = signals.size(1)
        
        # 2. Initialize buffers
        residual = signals.clone()
        coefficients = torch.zeros(batch_size, self.dict_size, device=device)
        active_set = torch.zeros(batch_size, self.dict_size, dtype=torch.bool, device=device)
        
        # 3. Precompute DtD and Dt once
        Dt = dictionary.t()
        DtD = torch.matmul(Dt, dictionary)
        DtD = DtD + torch.eye(self.dict_size, device=device) * self.epsilon
        
        # 4. Main OMP loop
        for k in range(self.sparsity):
            # Compute correlations
            correlations = torch.matmul(Dt, residual)
            correlations[active_set.t()] = -float('inf')
            
            # Select new atoms
            new_atoms = torch.argmax(correlations, dim=0)
            row_indices = torch.arange(batch_size, device=device)
            active_set[row_indices, new_atoms] = True
            
            # Get active indices
            active_indices = active_set.nonzero()[:,1].view(batch_size, k+1)
            
            try:
                # Solve least squares
                selected_DtD = DtD[active_indices.unsqueeze(-1), active_indices.unsqueeze(-2)]
                y = torch.bmm(Dt[active_indices], residual.t().unsqueeze(-1)).squeeze(-1)
                coeffs = torch.linalg.solve(selected_DtD, y.unsqueeze(-1)).squeeze(-1)
                
                # Update residual
                update = torch.bmm(
                    dictionary[:,active_indices].permute(1,0,2),
                    coeffs.unsqueeze(-1)
                ).squeeze(-1).transpose(0,1)
                residual = signals - update
                
                # Store coefficients
                coefficients.scatter_(1, active_indices, coeffs)
                
            except RuntimeError as e:
                if debug:
                    print(f"Error at iteration {k}: {e}")
                break
        
        return coefficients.t()
    
    def forward(self, z):
        """
        Forward pass through the dictionary learning bottleneck.
        
        Args:
            z: Input tensor of shape [batch_size, embedding_dim, height, width]
            
        Returns:
            z_q: Quantized tensor of shape [batch_size, embedding_dim, height, width]
            loss: Commitment loss
            coefficients: Sparse coefficients
        """
        # Save input shape
        input_shape = z.shape
        batch_size, embedding_dim, height, width = input_shape
        
        # Reshape for sparse coding with proper normalization
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, embedding_dim).t()
        
        # Dictionary updates during training
        if self.training:
            self._normalize_dictionary()
        
        # Sparse coding with OMP
        with torch.no_grad():
            coefficients = self.batch_omp(z_flat, self.dictionary, debug=False)

            # Clip extreme values
            coefficients = torch.clamp(coefficients, -1e7, 1e7)
        
        # Reconstruct with stable computation
        z_q_flat = torch.matmul(self.dictionary, coefficients)
        
        # Reshape back with proper normalization
        z_q_reshaped = z_q_flat.t().view(batch_size, height, width, embedding_dim).permute(0, 3, 1, 2)
        
        # Compute losses with gradient clipping
        z_flat_orig = z.permute(0, 2, 3, 1).reshape(-1, embedding_dim).t()
        rec_loss = F.mse_loss(z_q_flat.detach(), z_flat_orig)
        commit_loss = F.mse_loss(z_q_flat, z_flat_orig.detach())
        
        # Scale commitment cost with sparsity
        effective_cost = self.commitment_cost * (1 + self.sparsity / 10)
        
        # Add L1 regularization to coefficients
        coeff_loss = torch.mean(torch.abs(coefficients))
        
        # Scale losses to prevent explosion
        loss = rec_loss + effective_cost * commit_loss + 0.001 * coeff_loss
        
        # Straight-through estimator with gradient clipping
        z_q = z + torch.clamp(z_q_reshaped - z, -1, 1).detach()

        return z_q, loss, coefficients


def test_vector_quantizer():
    print("Testing VectorQuantizer...")
    
    # Parameters
    batch_size = 2
    embedding_dim = 64
    height, width = 8, 8
    num_embeddings = 512
    
    # Create random test input
    z = torch.randn(batch_size, embedding_dim, height, width)
    
    # Initialize VQ model
    vq = VectorQuantizer(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        commitment_cost=0.25,
        decay=0.99
    )
    
    # Test forward pass
    z_q, loss, indices = vq(z)
    
    # Print shapes and stats
    print(f"Input shape: {z.shape}")
    print(f"Quantized output shape: {z_q.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Indices shape: {indices.shape}")
    
    # Test codebook lookup
    random_indices = torch.randint(0, num_embeddings, (10,))
    codebook_vectors = vq.get_codebook_entry(random_indices)
    print(f"Retrieved codebook vectors shape: {codebook_vectors.shape}")
    
    # Visualize codebook usage after forward pass
    usage_count = torch.bincount(indices, minlength=num_embeddings)
    
    plt.figure(figsize=(10, 4))
    plt.bar(range(num_embeddings), usage_count.cpu().numpy())
    plt.title("VQ Codebook Usage")
    plt.xlabel("Codebook Index")
    plt.ylabel("Usage Count")
    plt.savefig("vq_codebook_usage.png")
    print("Saved codebook usage visualization to vq_codebook_usage.png")
    
    return vq, z, z_q, loss, indices

def test_dictionary_learning_bottleneck():
    print("\nTesting OnlineDictionaryLearningBottleneck...")
    
    # Parameters
    batch_size = 2
    embedding_dim = 64
    height, width = 8, 8
    dict_size = 512
    sparsity = 5
    
    # Create random test input
    z = torch.randn(batch_size, embedding_dim, height, width)
    
    # Initialize ODL model
    odl = DictionaryLearningBottleneck(
        dict_size=dict_size,
        embedding_dim=embedding_dim,
        sparsity=sparsity,
        commitment_cost=0.25,
        decay=0.99,
        use_ema=True
    )
    
    # Test forward pass
    z_q, loss, coefficients = odl(z)
    
    # Print shapes and stats
    print(f"Input shape: {z.shape}")
    print(f"Quantized output shape: {z_q.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Coefficients shape: {coefficients.shape}")
    
    # Check sparsity of coefficients
    nonzero_ratio = (coefficients.abs() > 1e-6).float().mean().item()
    print(f"Nonzero coefficient ratio: {nonzero_ratio:.4f}")
    print(f"Expected sparsity ratio: {sparsity/dict_size:.4f}")
    
    # Visualize dictionary atom usage
    atom_usage = (coefficients.abs() > 1e-6).float().sum(dim=1).cpu().numpy()
    
    plt.figure(figsize=(10, 4))
    plt.bar(range(dict_size), atom_usage)
    plt.title("Dictionary Atom Usage")
    plt.xlabel("Atom Index")
    plt.ylabel("Usage Count")
    plt.savefig("dictionary_atom_usage.png")
    print("Saved dictionary atom usage visualization to dictionary_atom_usage.png")
    
    # Visualize a few dictionary atoms
    num_atoms_to_show = 10
    atoms_to_show = np.random.choice(dict_size, num_atoms_to_show, replace=False)
    
    plt.figure(figsize=(12, 6))
    for i, atom_idx in enumerate(atoms_to_show):
        plt.subplot(2, 5, i+1)
        plt.plot(odl.dictionary[:, atom_idx].detach().cpu().numpy())
        plt.title(f"Atom {atom_idx}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("dictionary_atoms.png")
    print("Saved dictionary atoms visualization to dictionary_atoms.png")
    
    return odl, z, z_q, loss, coefficients

def compare_reconstructions(vq_data, odl_data):
    print("\nComparing reconstructions...")
    
    # Unpack data
    _, z_vq, z_q_vq, loss_vq, _ = vq_data
    _, z_odl, z_q_odl, loss_odl, _ = odl_data
    
    # Calculate reconstruction errors
    vq_mse = torch.mean((z_vq - z_q_vq) ** 2).item()
    odl_mse = torch.mean((z_odl - z_q_odl) ** 2).item()
    
    print(f"VQ-VAE MSE: {vq_mse:.6f}")
    print(f"ODL MSE: {odl_mse:.6f}")
    
    # Visualize sample reconstructions
    batch_idx = 0
    channel_idx = 0
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(z_vq[batch_idx, channel_idx].detach().cpu().numpy())
    plt.title("Original")
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.imshow(z_q_vq[batch_idx, channel_idx].detach().cpu().numpy())
    plt.title(f"VQ-VAE (MSE: {vq_mse:.6f})")
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.imshow(z_q_odl[batch_idx, channel_idx].detach().cpu().numpy())
    plt.title(f"ODL (MSE: {odl_mse:.6f})")
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig("reconstruction_comparison.png")
    print("Saved reconstruction comparison to reconstruction_comparison.png")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run tests
    vq_data = test_vector_quantizer()
    odl_data = test_dictionary_learning_bottleneck()
    
    # Compare reconstructions
    compare_reconstructions(vq_data, odl_data)
    
    print("\nAll tests completed!")