import torch
import torch.nn as nn
import torch.nn.functional as F

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
    dictionary_t = dictionary.t()
    G = dictionary_t.mm(dictionary)  # Gram matrix
    # Add regularization to Gram matrix for numerical stability
    # Larger epsilon needed for low-dimensional problems like RGB (d=3)
    G = G + torch.eye(G.size(0), device=G.device) * 1e-4
    eps = torch.norm(data, dim=0)  # residual, initialized as L2 norm of signal
    h_bar = dictionary_t.mm(data).t()  # initial correlation vector, transposed

    h = h_bar
    x = torch.zeros_like(h_bar)  # resulting sparse code
    L = torch.ones(batch_size, 1, 1, device=h.device)  # progressive Cholesky
    I = torch.ones(batch_size, 0, device=h.device).long()
    I_logic = torch.zeros_like(h_bar).bool()  # mask to avoid reselecting same index
    delta = torch.zeros(batch_size, device=h.device)

    k = 0
    while k < max_nonzero and eps.max() > tolerance:
        k += 1
        # select next index per sample, masking already selected positions
        index = (h * (~I_logic).float()).abs().argmax(dim=1)
        _update_logical(I_logic, index)
        
        batch_idx = torch.arange(batch_size, device=G.device)
        expanded_batch_idx = batch_idx.unsqueeze(0).expand(k, batch_size).t()

        if k > 1:
            # Cholesky update
            G_stack = G[I[batch_idx, :], index[expanded_batch_idx[..., :-1]]].view(batch_size, k - 1, 1)
            # Use modern triangular_solve replacement
            w = torch.linalg.solve_triangular(L, G_stack, upper=False).view(-1, 1, k - 1)
            # Add small epsilon to prevent numerical issues
            w_corner = torch.sqrt(torch.clamp(1 - (w ** 2).sum(dim=2, keepdim=True), min=1e-10))

            # Concatenate into new Cholesky: L <- [[L, 0], [w, w_corner]]
            k_zeros = torch.zeros(batch_size, k - 1, 1, device=h.device)
            L = torch.cat((
                torch.cat((L, k_zeros), dim=2),
                torch.cat((w, w_corner), dim=2),
            ), dim=1)

        # update non-zero indices
        I = torch.cat([I, index.unsqueeze(1)], dim=1)

        # Solve for x using Cholesky factor
        h_stack = h_bar[expanded_batch_idx, I[batch_idx, :]].view(batch_size, k, 1)
        # Replace deprecated cholesky_solve with two triangular solves
        y_stack = torch.linalg.solve_triangular(L, h_stack, upper=False)
        x_stack = torch.linalg.solve_triangular(L.transpose(1, 2), y_stack, upper=True)

        # Scatter x_stack into x at active indices
        x[batch_idx.unsqueeze(1), I[batch_idx]] = x_stack.squeeze(-1)

        # beta = G_I * x_I
        beta = x[batch_idx.unsqueeze(1), I[batch_idx]].unsqueeze(1).bmm(G[I[batch_idx], :]).squeeze(1)

        h = h_bar - beta

        # update residual
        new_delta = (x * beta).sum(dim=1)
        eps += delta - new_delta
        delta = new_delta

        if debug:
            print('Step {}, residual: {:.4f}, below tolerance: {:.4f}'.format(
                k, eps.max().item(), (eps < tolerance).float().mean().item()))

    return x.t()  # transpose since sparse codes should be used as D * x

class VectorQuantizer(nn.Module):
    """
    Vector Quantizer implementation for VQ-VAE.

    The Vector Quantizer maps continuous encodings to discrete codes from a learned codebook.
    This is the key component that enables VQ-VAE to learn discrete representations.
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        commitment_cost=0.25,
        decay=0.0,
        epsilon=1e-5,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

        if self.decay > 0:
            self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
            self.register_buffer("_ema_w", torch.zeros(num_embeddings, embedding_dim))
            self._ema_w.data.copy_(self.embedding.weight.detach())

    def forward(self, z_e: torch.Tensor):
        """
        Args:
            z_e: Input tensor of shape [B, D, H, W]

        Returns:
            Tuple of (quantized tensor, loss, perplexity, one-hot encodings)
        """
        # Permute to [B, H, W, D] for convenience and flatten to [N, D]
        z = z_e.permute(0, 2, 3, 1).contiguous()
        flat_inputs = z.view(-1, self.embedding_dim)
        num_vectors = flat_inputs.size(0)

        embeddings = self.embedding.weight
        emb_norm_sq = (embeddings ** 2).sum(dim=1)

        # Chunked nearest neighbour search
        chunk_size = 32768
        indices_list = []
        for start in range(0, num_vectors, chunk_size):
            end = min(start + chunk_size, num_vectors)
            x = flat_inputs[start:end]
            x_norm_sq = (x ** 2).sum(dim=1, keepdim=True)
            distances = x_norm_sq + emb_norm_sq.unsqueeze(0) - 2.0 * x @ embeddings.t()
            indices_list.append(torch.argmin(distances, dim=1))
        encoding_indices = torch.cat(indices_list, dim=0)  # [N]

        # One-hot encodings and quantized latents
        encodings = F.one_hot(encoding_indices, self.num_embeddings).to(flat_inputs.dtype)
        z_q_flat = encodings @ embeddings  # [N, D]
        z_q = z_q_flat.view_as(z)

        # Loss terms
        e_latent_loss = F.mse_loss(z_q.detach(), z)
        q_latent_loss = F.mse_loss(z_q, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        z_q = z_q.to(z.dtype)
        z_st = z + (z_q - z).detach()

        # Perplexity (codebook usage)
        encoding_sum = encodings.float().sum(dim=0)
        avg_probs = encoding_sum / float(num_vectors)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # EMA updates when enabled and in training mode
        if self.training and self.decay > 0:
            self._ema_cluster_size = (
                self._ema_cluster_size * self.decay
                + (1 - self.decay) * encoding_sum
            )

            ema_w = encodings.float().t() @ flat_inputs.float()
            self._ema_w = self._ema_w * self.decay + (1 - self.decay) * ema_w

            n = self._ema_cluster_size.sum()
            cluster_size = (
                (self._ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon)
                * n
            )
            self.embedding.weight.data.copy_(
                (self._ema_w / cluster_size.unsqueeze(1)).to(self.embedding.weight.data.dtype)
            )

        encodings = encodings.view(num_vectors, self.num_embeddings)
        return z_st.permute(0, 3, 1, 2).contiguous(), loss, perplexity, encodings


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
        omp_debug=False,
        normalize_atoms=True
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
        self.normalize_atoms = normalize_atoms
        
        # Initialize dictionary with random atoms
        self.dictionary = nn.Parameter(
            torch.randn(embedding_dim, num_embeddings), requires_grad=True
        )
        if self.normalize_atoms:
            self._normalize_dictionary()

    def _normalize_dictionary(self):
        """Normalize all dictionary atoms to have unit L2 norm."""
        with torch.no_grad():
            self.dictionary.data = self.dictionary.data / torch.linalg.norm(self.dictionary.data, dim=0)

    def batch_omp(self, X, D):
        """
        Batched Orthogonal Matching Pursuit (greedy selection only).
        
        Fast implementation using greedy atom selection without LS refinement.

        Args:
            X (torch.Tensor): Input signals of shape (M, B).
            D (torch.Tensor): Dictionary of shape (M, N), where each column is an atom of dimension M.

        Returns:
            coefficients: (N, B) Tensor with the corresponding coefficients.
        """
        M, B = X.shape
        _, N = D.shape
        device = X.device

        # Vectorized greedy atom selection with diversity penalty
        coefficients = torch.zeros(N, B, device=device, dtype=X.dtype)
        residual = X.clone()
        
        # Mask to prevent reselecting the same atom per signal (N, B) format
        mask = torch.ones(N, B, device=device, dtype=X.dtype)  # Use float for efficient multiply
        batch_idx = torch.arange(B, device=device)
        
        # Track global atom usage for diversity penalty
        global_usage = torch.zeros(N, device=device, dtype=X.dtype)
        diversity_weight = 0.001  # Small bonus to encourage diversity
        
        for k in range(self.sparsity_level):
            # Compute correlations (keep in N, B format - avoid transpose)
            correlations = torch.mm(D.t(), residual)  # (N, B)
            abs_corr = torch.abs(correlations)
            
            # Apply diversity bonus (vectorized)
            if k > 0:
                avg_usage = global_usage.sum() / N
                diversity_bonus = diversity_weight * (avg_usage - global_usage).clamp(min=0).unsqueeze(1)
                abs_corr = abs_corr + diversity_bonus
            
            # Apply mask and find argmax (avoid transpose by using dim=0)
            abs_corr_masked = abs_corr * mask  # (N, B) - already in right shape!
            idx = torch.argmax(abs_corr_masked, dim=0)  # (B,)
            
            # Update mask (vectorized) - set selected atoms to 0
            mask[idx, batch_idx] = 0.0
            
            # Update global usage (vectorized with scatter_add)
            global_usage.scatter_add_(0, idx, torch.ones(B, device=device, dtype=X.dtype))
            
            # Gather selected atoms
            d_selected = D[:, idx]  # (M, B)
            
            # Compute coefficients
            numerator = (residual * d_selected).sum(dim=0)  # (B,)
            denominator = (d_selected ** 2).sum(dim=0)  # (B,)
            alpha = numerator / (denominator + self.epsilon)  # (B,)
            
            # Update coefficient matrix
            coefficients[idx, batch_idx] = alpha
            
            # Update residual (not in-place to preserve gradients)
            residual = residual - d_selected * alpha.unsqueeze(0)
        
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
        
        # Flatten the input: [B, H, W, C] -> [B*H*W, C] -> [C, B*H*W]
        ze_flattened = z_e.reshape(-1, self.embedding_dim).t().contiguous()  # [embedding_dim, batch_size * height * width]
        
        '''
        Sparse coding stage
        '''
        # Run OMP in float32 to avoid AMP dtype mismatches; cast back after
        orig_dtype = ze_flattened.dtype
        with torch.amp.autocast('cuda', enabled=False):
            ze_flattened_f32 = ze_flattened.to(torch.float32)
            dictionary_f32 = self.dictionary.to(torch.float32)
            
            # For pixel-level data without atom normalization, OMP still needs normalized dictionary
            # The dictionary atoms store the actual RGB color values
            if self.normalize_atoms:
                # Standard DL: normalize and keep normalized
                dict_norms = torch.linalg.norm(dictionary_f32, dim=0, keepdim=True)
                dict_for_omp = dictionary_f32 / (dict_norms + 1e-10)
                dict_for_recon = dict_for_omp
            else:
                # Pixel-level: use atoms as-is (they're already RGB colors from k-means)
                dict_for_omp = dictionary_f32
                dict_for_recon = dictionary_f32
            
            # Use simpler batch_omp method (more stable for pixel-level data)
            coefficients = self.batch_omp(ze_flattened_f32, dict_for_omp)
            z_dl_f32 = dict_for_recon @ coefficients

        # # validate the coefficients sparsity level (number of non-zero coefficients along the first dimension)
        # sparsity_level = (coefficients.abs() > 1e-6).float().sum(dim=0).mean().item()
        # print(f"DEBUG Sparsity level: {sparsity_level:.4f}")

        z_dl = z_dl_f32.to(orig_dtype)  # cast back to original dtype (could be fp16 under AMP)
        # Reshape back: [C, B*H*W] -> [B*H*W, C] -> [B, H, W, C]
        z_dl = z_dl.t().reshape(input_shape).contiguous()  # [batch_size, height, width, embedding_dim]

        # Compute the commitment loss
        e_latent_loss = F.mse_loss(z_dl.detach(), z_e)  # [batch_size, height, width, embedding_dim]
        dl_latent_loss = F.mse_loss(z_dl, z_e.detach())  # [batch_size, height, width, embedding_dim]

        # Compute the total loss
        loss = self.commitment_cost * e_latent_loss + dl_latent_loss

        # Straight-through estimator
        z_dl = z_e + (z_dl - z_e).detach()  # Allow gradients to flow back to encoder

        return z_dl.permute(0, 3, 1, 2).contiguous(), loss, coefficients  # Return the reconstructed latent representation, loss, and sparse coefficients



