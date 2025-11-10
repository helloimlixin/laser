import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

def _update_logical(logical, to_add):
    """Mark selected indices as used in the logical mask."""
    running_idx = torch.arange(to_add.shape[0], device=to_add.device)
    logical[running_idx, to_add] = 1


def BatchOMP(data, dictionary, max_nonzero, tolerance=1e-7, debug=False):
    """
    Vectorized Batch Orthogonal Matching Pursuit (OMP).
    Adapted from: https://github.com/amzn/sparse-vqvae/blob/main/utils/pyomp.py
    Reference links:
    - https://sparse-plex.readthedocs.io/en/latest/book/pursuit/omp/batch_omp.html
    - http://www.cs.technion.ac.il/~ronrubin/Publications/KSVD-OMP-v2.pdf

    Args:
        data (torch.Tensor): (vector_dim, batch_size)
        dictionary (torch.Tensor): (vector_dim, num_atoms), column-normalized recommended
        max_nonzero (int): Maximum number of non-zero coefficients per signal
        tolerance (float): Early stopping tolerance on residual norm
        debug (bool): If True, prints per-iteration diagnostics

    Returns:
        torch.Tensor: (num_atoms, batch_size) sparse coefficients
    """
    vector_dim, batch_size = data.size()
    dictionary_t = dictionary.t()  # (num_atoms, vector_dim)
    G = dictionary_t.mm(dictionary)  # (num_atoms, num_atoms) Gram matrix

    # residual norms initialized as ||x||_2
    eps = torch.norm(data, dim=0)  # (batch_size,)

    # initial correlation vector, transposed to (batch_size, num_atoms)
    h_bar = dictionary_t.mm(data).t()

    # working variables
    h = h_bar
    x = torch.zeros_like(h_bar)  # (batch_size, num_atoms) resulting sparse code
    L = torch.ones(batch_size, 1, 1, device=h.device)  # progressive Cholesky of G in selected indices
    I = torch.ones(batch_size, 0, device=h.device).long()
    I_logic = torch.zeros_like(h_bar).bool()  # mask to avoid reselecting same index
    delta = torch.zeros(batch_size, device=h.device)  # to track errors

    k = 0
    while k < max_nonzero:
        k += 1
        # select next index per sample, masking already selected positions
        index = (h * (~I_logic).float()).abs().argmax(dim=1)  # (batch_size,)
        _update_logical(I_logic, index)

        batch_idx = torch.arange(batch_size, device=G.device)
        expanded_batch_idx = batch_idx.unsqueeze(0).expand(k, batch_size).t()  # (batch_size, k)

        if k > 1:
            # Gather G[I, index] efficiently across batch
            G_stack = G[I[batch_idx, :], index[expanded_batch_idx[..., :-1]]].view(batch_size, k - 1, 1)
            # Solve L w = G_stack for w using triangular solve (lower triangular)
            w = torch.linalg.solve_triangular(L, G_stack, upper=False).view(-1, 1, k - 1)
            # Corner element: sqrt(max(1 - ||w||^2, 0)) for numerical stability
            w_corner = torch.sqrt(torch.clamp(1 - (w ** 2).sum(dim=2, keepdim=True), min=0.0))

            # Concatenate into new Cholesky factor
            k_zeros = torch.zeros(batch_size, k - 1, 1, device=h.device)
            L = torch.cat((
                torch.cat((L, k_zeros), dim=2),
                torch.cat((w, w_corner), dim=2),
            ), dim=1)

        # update non-zero indices
        I = torch.cat([I, index.unsqueeze(1)], dim=1)  # (batch_size, k)

        # Solve for x_I with the Cholesky factor: G_I x_I = h_bar_I
        h_stack = h_bar[expanded_batch_idx, I[batch_idx, :]].view(batch_size, k, 1)  # (B, k, 1)
        # Use two triangular solves instead of deprecated cholesky_solve:
        # 1) L y = h_stack
        y_stack = torch.linalg.solve_triangular(L, h_stack, upper=False)  # (B, k, 1)
        # 2) L^T x = y
        x_stack = torch.linalg.solve_triangular(L.transpose(1, 2), y_stack, upper=True)  # (B, k, 1)

        # Scatter x_stack into x at active indices
        x[batch_idx.unsqueeze(1), I[batch_idx]] = x_stack.squeeze(-1)

        # beta = G_I * x_I (projected correlations)
        beta = x[batch_idx.unsqueeze(1), I[batch_idx]].unsqueeze(1).bmm(G[I[batch_idx], :]).squeeze(1)

        # Update correlations for next selection step
        h = h_bar - beta

        # Update residual via energy tracking
        new_delta = (x * beta).sum(dim=1)
        eps += delta - new_delta
        delta = new_delta

        if debug:
            print('OMP step {}, residual: {:.4f}, below tolerance: {:.4f}'.format(
                k, eps.max().item(), (eps < tolerance).float().mean().item()))

    # Return transposed to (num_atoms, batch_size)
    return x.t()

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
                 decay=0.0, epsilon=1e-5):
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
        self.decay = 0.0  # Non-EMA for Zalando-style VQ
        self.epsilon = epsilon  # Small constant for numerical stability

        # Create embedding table (codebook)
        # This is the dictionary of codes that continuous vectors will be mapped to
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # Initialize embedding weights with small random values (Zalando)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

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
        ze_flattened = z_e.view(-1, self.embedding_dim)  # [N, D]
        N = ze_flattened.size(0)

        # Memory-efficient nearest neighbor search in chunks (Zalando)
        emb = self.embedding.weight  # [K, D]
        emb_norm2 = (emb ** 2).sum(dim=1)  # [K]
        chunk_size = 32768
        all_indices = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            x = ze_flattened[start:end]                      # [n, D]
            x_norm2 = (x ** 2).sum(dim=1, keepdim=True)      # [n,1]
            logits = x @ emb.t()                             # [n, K]
            dists = x_norm2 + emb_norm2.unsqueeze(0) - 2.0 * logits
            idx = torch.argmin(dists, dim=1)                 # [n]
            all_indices.append(idx)
        encoding_indices = torch.cat(all_indices, dim=0)     # [N]

        # Quantize via gather and reshape back
        z_q_flat = emb[encoding_indices]                     # [N, D]
        z_q = z_q_flat.view(input_shape)                     # [B, H, W, D]

        # Losses (Zalando): codebook + commitment
        e_latent_loss = F.mse_loss(z_q.detach(), z_e)
        q_latent_loss = F.mse_loss(z_q, z_e.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        # This allows gradients to flow back to encoder even though quantization is discrete
        # In the forward pass: z_q = selected embeddings
        # In the backward pass: gradients flow directly from z_q to z, bypassing quantization
        z_q = z_e + (z_q - z_e).detach()

        # Compute perplexity evaluation
        counts_for_pp = torch.bincount(encoding_indices, minlength=self.num_embeddings).to(z_e.dtype)
        avg_probs = counts_for_pp / float(N)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Return indices instead of full one-hot to avoid memory blow-up; caller ignores it
        return z_q.permute(0, 3, 1, 2).contiguous(), loss, perplexity, encoding_indices


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
        use_ema=True,
        tolerance=1e-7,
        omp_debug=False
    ):
        super(DictionaryLearning, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.sparsity_level = sparsity_level
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.use_ema = use_ema
        self.tolerance = tolerance
        self.omp_debug = omp_debug
        
        # Initialize dictionary with random atoms
        self.dictionary = nn.Parameter(
            torch.randn(embedding_dim, num_embeddings), requires_grad=True
        )
        self._normalize_dictionary()

    def _normalize_dictionary(self):
        """Normalize all dictionary atoms to have unit L2 norm."""
        with torch.no_grad():
            self.dictionary.data = self.dictionary.data / torch.linalg.norm(self.dictionary.data, dim=0)

    def batch_omp(self, X, D):
        """
        Batched Orthogonal Matching Pursuit.

        Args:
            X (torch.Tensor): Input signals of shape (B, M).
            D (torch.Tensor): Dictionary of shape (M, N), where each column is an atom of dimension M.

        Returns:
            coefficients: (N, B) Tensor with the corresponding coefficients.
        """
        M, B = X.shape
        _, N = D.shape
        device = X.device

        # Initialize the full coefficients matrix (N, B) with zeros.
        coefficients = torch.zeros(N, B, device=device, dtype=X.dtype)

        # Initialize residual as the input signals.
        residual = X.clone()  # shape (M, B)

        for k in range(self.sparsity_level):
            # Compute the correlations (projections): D^T (shape N x M) x residual (M x B) = (N x B)
            correlations = torch.mm(D.t(), residual)  # shape (N, B)

            # For each signal (each column), select the atom with the highest absolute correlation / projection
            idx = torch.argmax(torch.abs(correlations), dim=0)  # shape (B,)

            # Gather the selected atoms: for each signal i, d_selected[:, i] = D[:, idx[i]]
            # D is (M, N) and idx is (B,), so D[:, idx] is (M, B).
            d_selected = D[:, idx]  # shape (M, B)

            # Compute coefficients for each signal:
            # alpha[i] = (residual[:, i] @ d_selected[:, i]) / (|| d_selected[:, i] ||^2)
            numerator = (residual * d_selected).sum(dim=0)  # shape (B,)
            denominator = (d_selected ** 2).sum(dim=0)  # shape (B,)
            alpha = numerator / (denominator + self.epsilon)  # shape (B,)

            # Update the full coefficient matrix.
            sample_indices = torch.arange(B, device=device)  # shape (B,)
            coefficients.index_put_((idx, sample_indices), alpha, accumulate=True)

            # Update the residual: residual = X - D @ coefficients
            residual = residual - d_selected * alpha.unsqueeze(0)  # shape (M, B)

        return coefficients
    
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
        ze_flattened = z_e.view(self.embedding_dim, -1)  # [embedding_dim, batch_size * height * width]
        
        '''
        Sparse coding stage
        '''
        self._normalize_dictionary()
        # Run OMP in float32 to avoid AMP dtype mismatches; cast back after
        orig_dtype = ze_flattened.dtype
        with torch.amp.autocast('cuda', enabled=False):
            ze_flattened_f32 = ze_flattened.to(torch.float32)
            dictionary_f32 = self.dictionary.to(torch.float32)
            coefficients = BatchOMP(
                data=ze_flattened_f32,
                dictionary=dictionary_f32,
                max_nonzero=self.sparsity_level,
                tolerance=self.tolerance,
                debug=self.omp_debug
            )  # [num_embeddings, batch_size * height * width]
            z_dl_f32 = dictionary_f32 @ coefficients  # [embedding_dim, batch_size * height * width]

        # # validate the coefficients sparsity level (number of non-zero coefficients along the first dimension)
        # sparsity_level = (coefficients.abs() > 1e-6).float().sum(dim=0).mean().item()
        # print(f"DEBUG Sparsity level: {sparsity_level:.4f}")

        z_dl = z_dl_f32.to(orig_dtype)  # cast back to original dtype (could be fp16 under AMP)
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