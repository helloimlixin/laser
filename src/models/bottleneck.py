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
    2. Replacing the input vectors with their corresponding codebook vectors from nearest neighbor search
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
        super(VectorQuantizer, self).__init__()

        self.embedding_dim = embedding_dim  # Dimension of each embedding vector
        self.num_embeddings = num_embeddings  # Number of embedding vectors in the codebook
        self.commitment_cost = commitment_cost  # Weighting for commitment loss
        self.use_ema = bool(decay > 0.0)  # Whether to use EMA updates for the codebook
        self.decay = decay  # EMA decay factor (higher = slower updates)
        self.epsilon = epsilon  # Small constant for numerical stability

        # Create embedding table (codebook)
        # This is the dictionary of codes that continuous vectors will be mapped to
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        if self.use_ema:
            print('Using EMA updates for codebook...')
            # Initialize the embedding weights using normal distribution
            self.embedding.weight.data.normal_()
        else:
            print('Using standard backpropagation for codebook... (no EMA)')
            # Initialize embedding weights with small random values
            self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

        # Register buffers for EMA updates (not model parameters - not directly optimized)
        # ema_cluster_size: Tracks how often each codebook entry is used
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        # ema_w: EMA of the encoder outputs assigned to each codebook entry
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        # Flag to control when EMA updates are performed
        self._ema_w.data.normal_()

    def forward(self, z_e):
        """
        Forward pass through the vector quantizer.
        
        Args:
            z_e (torch.Tensor): Output from encoder with shape [B, D, H, W]
                              B: batch size, D: embedding dimension, H, W: spatial dimensions
        
        Returns:
            z_q (torch.Tensor): Quantized tensor with same shape as input [B, D, H, W]
            loss (torch.Tensor): VQ loss (codebook loss + commitment loss)
            min_encoding_indices (torch.Tensor): Indices of the nearest embedding vectors [B*H*W]
        """
        # z shape: [B, D, H, W]

        # Reshape z to [B*H*W, D] for easier processing
        # The permute operation reorders dimensions to [B, H, W, D] before reshaping
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        input_shape = z_e.shape

        # Flatten the input
        ze_flattened = z_e.view(-1, self.embedding_dim)
        
        # Calculate squared distances between z_flattened and embedding vectors
        # This uses the || x - e ||^2 = ||x||^2 + ||e||^2 - 2*x^T*e formula for efficiency
        # Rather than computing distances directly, which would be more expensive
        distances = torch.sum(ze_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * torch.matmul(ze_flattened, self.embedding.weight.t())

        # Find nearest embedding for each z_flattened vector
        # encoding_indices contains indices of the closest codebook entry for each position
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        # Convert indices to one-hot encodings for gathering embeddings
        # encodings shape: [B*H*W, num_embeddings]
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=z_e.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Get the quantized latent vectors using embedding lookup
        # Multiply one-hot encodings by embedding weights to get quantized vectors
        # Then reshape back to the original tensor shape [B, H, W, D]
        z_q = torch.matmul(encodings, self.embedding.weight).view(input_shape)

        # Update codebook using EMA if enabled and in training mode
        if self.use_ema and self.training:
            # EMA update for the codebook
            
            # 1. Update cluster size (how many vectors are assigned to each codebook entry)
            # Count occurrences of each embedding being used in this batch
            # Update the exponential moving average of cluster sizes
            self._ema_cluster_size = self._ema_cluster_size * self.decay + \
                (1 - self.decay) * encodings.sum(0)

            # 2. Update embedding vectors based on assigned encoder outputs
            # Compute sum of all z vectors assigned to each embedding
            dw = torch.matmul(encodings.t(), ze_flattened)
            # Update the exponential moving average of assigned vectors
            self._ema_w = nn.Parameter(self._ema_w * self.decay + (1 - self.decay) * dw)

            # 3. Normalize the updated embeddings
            # Get total cluster size (with smoothing to prevent division by zero)
            n = torch.sum(self._ema_cluster_size.data)
            # Normalize cluster sizes with Laplace smoothing
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n)
            # Update the actual embedding weights with the EMA-updated version
            self.embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Compute VQ losses:

        # 1. Codebook loss (makes codebook vectors move towards encoder outputs)
        # The detach() on z prevents gradients from flowing back to the quantization
        e_latent_loss = F.mse_loss(z_q.detach(), z_e)
        
        # 2. Commitment loss (makes encoder outputs move towards codebook vectors)
        # The detach() on z_q prevents gradients from flowing back to the codebook
        commitment_loss = self.commitment_cost * e_latent_loss
        
        # Combine losses with commitment_cost weighting
        if self.use_ema:
            loss = commitment_loss
        else:
            q_latent_loss = F.mse_loss(z_q, z_e.detach())
            loss = q_latent_loss + commitment_loss

        # Straight-through estimator
        # This allows gradients to flow back to encoder even though quantization is discrete
        # In the forward pass: z_q = selected embeddings
        # In the backward pass: gradients flow directly from z_q to z, bypassing quantization
        z_q = z_e + (z_q - z_e).detach()

        # Compute perplexity evaluation
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q.permute(0, 3, 1, 2).contiguous(), loss, perplexity, encodings


class DictionaryLearning(nn.Module):
    """
    Dictionary Learning Bottleneck with vectorized Batch Orthogonal Matching Pursuit (OMP) sparse coding.
    """
    def __init__(
        self,
        num_embeddings=512,
        embedding_dim=64,
        sparsity_level=5,
        commitment_cost=0.25,
        decay=0.99,
        epsilon=1e-10,
        use_ema=True
    ):
        super(DictionaryLearning, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.sparsity_level = sparsity_level
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.use_ema = use_ema
        
        # Initialize dictionary with random atoms
        self.dictionary = nn.Parameter(
            torch.randn(embedding_dim, num_embeddings), requires_grad=True
        )
        self._normalize_dictionary()

    def _normalize_dictionary(self):
        """Normalize all dictionary atoms to have unit L2 norm."""
        with torch.no_grad():
            self.dictionary.data = self.dictionary.data / torch.linalg.norm(self.dictionary.data, dim=0)

    def batch_omp(self, signals, dictionary, debug=False):
        """
        Sparce coding using Batch Orthogonal Matching Pursuit (OMP).

        References:
            - Rubinstein, R., Zibulevsky, M. and Elad, M., "Efficient Implementation of the K-SVD Algorithm using
                Batch Orthogonal Matching Pursuit," CS Technion, 2008.

        Args:
            signals (torch.Tensor): Input tensor of shape [batch_size, embedding_dim, height, width]
            dictionary (torch.Tensor): Dictionary of shape [embedding_dim, num_embeddings]
            debug (bool): If True, print debug information during

        Returns:
            coefficients (torch.Tensor): Sparse coefficients of shape [batch_size, num_embeddings]
        """
        # 1. Initialize variables and Precompute Dt and DtD
        embedding_dim, num_signals = signals.size()
        device = signals.device
        dtype = signals.dtype
        dictionary_t = dictionary.t()  # save the transposed dictionary for faster computation
        gram_matrix = torch.mm(dictionary_t, dictionary)  # precompute the Gram matrix [num_embeddings, num_embeddings]
        init_correlations = torch.mm(dictionary_t, signals).t()  # initial correlations [num_embeddings, batch_size]

        # 2. Initialize buffers
        residual = torch.norm(signals, p=2, dim=0)  # initial residual [batch_size], which is the L2 norm of each signal
        delta = torch.zeros(num_signals, device=device)  # placeholder for the residual update
        coefficients = torch.zeros_like(init_correlations, dtype=dtype)  # placeholder for the sparse coefficients
        correlations = init_correlations  # correlations are initialized to the initial correlations
        L = torch.ones(num_signals, 1, 1, device=device)  # contains the progressive Cholesky of the Gram matrix in the selected indices
        I = torch.zeros(num_signals, 0, dtype=torch.long, device=device)  # placeholder for the selected indices
        omega = torch.ones_like(init_correlations, dtype=torch.bool)  # operator to mask out zero elements in the correlations
        signal_idx = torch.arange(num_signals, device=device)  # indices for the signals
        
        # 3. Main OMP loop
        k = 0

        while k < self.sparsity_level:
            k += 1
            k_hats = torch.argmax(torch.abs(correlations * omega), dim=1)  # find the index of the maximum correlation
            # update the mask to make sure we do not select the same atoms again
            omega.scatter_(1, k_hats.unsqueeze(1), 0)
            expanded_signal_idx = signal_idx.unsqueeze(0).expand(k, num_signals).t()  # expand the signal indices for batch processing, more efficient than repeating

            if  k > 1:
                G_ = gram_matrix[I[signal_idx, :], k_hats[expanded_signal_idx[...,: -1]]].view(num_signals, k - 1, 1)  # compute for all the signals in a vectorized manner
                w = torch.linalg.solve_triangular(L, G_, upper=False).view(-1, 1, k - 1)  # solve the triangular system for the current signal
                w_br = torch.sqrt(1 - (w ** 2).sum(dim=2, keepdim=True))  # L bottom-right corner element: sqrt(1 - w.t() @ w)

                # concatenate into the new Cholesky: L = [[L, 0], [w, w_br]]
                k_zeros = torch.zeros(num_signals, k - 1, 1, device=signals.device)
                L = torch.cat((
                    torch.cat((L, k_zeros), dim=2),
                    torch.cat((w, w_br), dim=2)
                ), dim=1)

            # update non-zero indices
            I = torch.cat((I, k_hats.unsqueeze(1)), dim=1)

            # solve for L
            correlations_ = init_correlations[expanded_signal_idx, I[signal_idx, :]].view(num_signals, k, 1)
            coefficients_ = torch.cholesky_solve(correlations_, L)

            # de-stack the coefficients into the non-zero elements
            coefficients[signal_idx.unsqueeze(1), I[signal_idx]] = coefficients_[signal_idx].squeeze(-1)

            # beta = G_I @ coefficients
            beta = torch.bmm(coefficients[signal_idx.unsqueeze(1), I[signal_idx]].unsqueeze(1),
                                gram_matrix[I[signal_idx], :]).squeeze(1)  # [batch_size, num_embeddings]

            # update the correlations
            correlations = init_correlations - beta

            # update the residual
            new_delta = (coefficients * beta).sum(dim=1)  # [batch_size]
            residual += delta - new_delta
            delta = new_delta

            if debug and k % 1 == 0:
                print('Step {}, residual: {:.4f}, below tolerance: {:.4f}'.format(k, residual.max(), (residual < 1e-7).float().mean().item()))

        return coefficients.t()
    
    def forward(self, z_e):
        """
        Forward pass through the dictionary learning bottleneck.
        
        Args:
            z_e: Input tensor of shape [batch_size, embedding_dim, height, width]
            
        Returns:
            z_dl: latent representation reconstructed from the dictionary learning bottleneck
            loss: loss from the dictionary learning bottleneck
            coefficients: sparse coefficients
        """
        # z shape: [batch_size, embedding_dim, height, width]

        z_e = z_e.permute(0, 2, 3, 1).contiguous()  # [batch_size, height, width, embedding_dim]
        input_shape = z_e.shape
        
        # Flatten the input
        ze_flattened = z_e.view(self.embedding_dim, -1)  # [batch_size * height * width, embedding_dim]
        
        '''
        Sparse coding stage
        '''
        self._normalize_dictionary()
        coefficients = self.batch_omp(ze_flattened, self.dictionary, debug=False)  # [num_embeddings, batch_size * height * width]

        z_dl = self.dictionary @ coefficients  # [embedding_dim, batch_size * height * width]
        z_dl = z_dl.view(input_shape) # [batch_size, embedding_dim, height, width]

        # Compute the commitment loss
        e_latent_loss = F.mse_loss(z_dl.detach(), z_e)  # [batch_size, height, width, embedding_dim]
        dl_latent_loss = F.mse_loss(z_dl, z_e.detach())  # [batch_size, height, width, embedding_dim]

        # Compute the total loss
        loss = self.commitment_cost * e_latent_loss + dl_latent_loss

        # Straight-through estimator
        z_dl = z_e + (z_dl - z_e).detach()  # Allow gradients to flow back to encoder

        return z_dl.permute(0, 3, 1, 2).contiguous(), loss, coefficients  # Return the reconstructed latent representation, loss, and sparse coefficients


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
    num_embeddings = 512
    sparsity = 5
    
    # Create random test input
    z = torch.randn(batch_size, embedding_dim, height, width)
    
    # Initialize ODL model
    odl = DictionaryLearning(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        sparsity_level=sparsity,
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
    
    # Check sparsity_level of coefficients
    nonzero_ratio = (coefficients.abs() > 1e-6).float().mean().item()
    print(f"Nonzero coefficient ratio: {nonzero_ratio:.4f}")
    print(f"Expected sparsity ratio: {sparsity/num_embeddings:.4f}")
    
    # Visualize dictionary atom usage
    atom_usage = (coefficients.abs() > 1e-6).float().sum(dim=1).cpu().numpy()
    
    plt.figure(figsize=(10, 4))
    plt.bar(range(num_embeddings), atom_usage)
    plt.title("Dictionary Atom Usage")
    plt.xlabel("Atom Index")
    plt.ylabel("Usage Count")
    plt.savefig("dictionary_atom_usage.png")
    print("Saved dictionary atom usage visualization to dictionary_atom_usage.png")
    
    # Visualize a few dictionary atoms
    num_atoms_to_show = 10
    atoms_to_show = np.random.choice(num_embeddings, num_atoms_to_show, replace=False)
    
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