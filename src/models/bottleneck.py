import warnings
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparsePatternQuantizer(nn.Module):
    """
    Maps sparse codes to discrete pattern tokens for autoregressive generation.

    Each pattern is a learned prototype sparse combination. During forward pass,
    we find the nearest pattern for each sparse code and return discrete indices.

    This enables efficient tokenization:
    - Input: sparse codes [num_atoms, num_patches] with ~32 non-zero per column
    - Output: discrete indices [num_patches] from vocabulary of num_patterns

    For a 128x128 image with patch_size=8:
    - Without quantizer: 16 patches × 32 atoms = 512 tokens
    - With quantizer: 16 patches × 1 pattern = 16 tokens

    Training uses straight-through estimator for discrete indices.
    Commitment loss encourages sparse codes to match their assigned patterns.
    """

    def __init__(
        self,
        num_patterns: int = 2048,
        num_atoms: int = 2048,
        sparsity: int = 32,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        use_ema: bool = True,
        temperature: float = 1.0,
    ):
        """
        Args:
            num_patterns: Size of pattern vocabulary (number of discrete tokens)
            num_atoms: Dictionary size (must match DictionaryLearning.num_embeddings)
            sparsity: Target sparsity level for patterns
            commitment_cost: Weight for commitment loss
            ema_decay: EMA decay for pattern updates (if use_ema=True)
            use_ema: Use EMA updates for patterns instead of gradients
            temperature: Temperature for soft pattern matching (lower = harder)
        """
        super().__init__()
        self.num_patterns = num_patterns
        self.num_atoms = num_atoms
        self.sparsity = sparsity
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.use_ema = use_ema
        self.temperature = temperature

        # Learnable pattern codebook: each pattern is a sparse coefficient vector
        # Shape: [num_atoms, num_patterns] - matches coefficient matrix layout
        # Initialize with small random values, will be updated during training
        self.patterns = nn.Parameter(torch.randn(num_atoms, num_patterns) * 0.02)

        if use_ema:
            # EMA cluster size and sum for stable updates
            self.register_buffer('ema_cluster_size', torch.zeros(num_patterns))
            self.register_buffer('ema_pattern_sum', torch.zeros(num_atoms, num_patterns))
            self.register_buffer('initialized', torch.tensor(False))

    def _initialize_patterns_from_data(self, sparse_codes: torch.Tensor):
        """Initialize patterns from first batch of sparse codes using k-means++."""
        # sparse_codes: [num_atoms, num_samples]
        num_samples = sparse_codes.shape[1]
        device = sparse_codes.device

        if num_samples < self.num_patterns:
            # Not enough samples, use random subset + noise
            indices = torch.randperm(num_samples, device=device)
            self.patterns.data[:, :num_samples] = sparse_codes[:, indices]
            # Fill rest with noisy copies
            for i in range(num_samples, self.num_patterns):
                src_idx = i % num_samples
                noise = torch.randn_like(sparse_codes[:, src_idx]) * 0.01
                self.patterns.data[:, i] = sparse_codes[:, src_idx] + noise
        else:
            # k-means++ initialization
            indices = []
            # First center: random
            idx = torch.randint(num_samples, (1,), device=device).item()
            indices.append(idx)

            for _ in range(1, min(self.num_patterns, num_samples)):
                # Compute distances to nearest center
                centers = sparse_codes[:, indices]  # [num_atoms, num_centers]
                # Squared L2 distance: ||x - c||^2
                dists = torch.cdist(sparse_codes.t(), centers.t(), p=2).pow(2)  # [num_samples, num_centers]
                min_dists = dists.min(dim=1).values  # [num_samples]
                # Sample proportional to distance squared
                probs = min_dists / (min_dists.sum() + 1e-10)
                idx = torch.multinomial(probs, 1).item()
                indices.append(idx)

            self.patterns.data = sparse_codes[:, indices[:self.num_patterns]].clone()

        # Apply sparsity constraint to initial patterns
        self._enforce_sparsity()

        if self.use_ema:
            self.ema_pattern_sum.data = self.patterns.data.clone()
            self.ema_cluster_size.fill_(1.0)
            self.initialized.fill_(True)

    def _enforce_sparsity(self):
        """Enforce sparsity constraint on patterns (keep top-k coefficients)."""
        with torch.no_grad():
            # Keep only top-k coefficients per pattern
            abs_patterns = self.patterns.abs()
            topk_vals, topk_idx = abs_patterns.topk(self.sparsity, dim=0)

            # Create sparse patterns
            sparse_patterns = torch.zeros_like(self.patterns)
            sparse_patterns.scatter_(0, topk_idx, self.patterns.gather(0, topk_idx))
            self.patterns.data = sparse_patterns

    def forward(self, sparse_codes: torch.Tensor):
        """
        Quantize sparse codes to discrete pattern indices.

        Args:
            sparse_codes: Sparse coefficient matrix [num_atoms, num_samples]
                         where num_samples = batch_size * num_patches

        Returns:
            pattern_indices: Discrete pattern indices [num_samples]
            quantized_codes: Quantized sparse codes [num_atoms, num_samples]
            pattern_loss: Commitment + codebook loss for training
            info: Dict with additional metrics
        """
        # sparse_codes: [num_atoms, num_samples]
        num_atoms, num_samples = sparse_codes.shape
        device = sparse_codes.device

        # Initialize patterns from first batch
        if self.use_ema and not self.initialized and self.training:
            self._initialize_patterns_from_data(sparse_codes.detach())

        # Normalize for cosine similarity (more robust than L2 for sparse vectors)
        codes_norm = F.normalize(sparse_codes.t(), dim=1, eps=1e-8)  # [num_samples, num_atoms]
        patterns_norm = F.normalize(self.patterns.t(), dim=1, eps=1e-8)  # [num_patterns, num_atoms]

        # Compute similarities: [num_samples, num_patterns]
        similarities = codes_norm @ patterns_norm.t() / self.temperature

        # Find nearest pattern (hard assignment)
        pattern_indices = similarities.argmax(dim=1)  # [num_samples]

        # Get quantized codes from patterns
        quantized_codes = self.patterns[:, pattern_indices]  # [num_atoms, num_samples]

        # Compute losses
        # Commitment loss: encourages sparse codes to stay close to assigned patterns
        commitment_loss = F.mse_loss(sparse_codes, quantized_codes.detach())

        # Codebook loss: encourages patterns to move toward assigned codes
        codebook_loss = F.mse_loss(quantized_codes, sparse_codes.detach())

        pattern_loss = self.commitment_cost * commitment_loss + codebook_loss

        # EMA updates for patterns (more stable than gradient updates)
        if self.training and self.use_ema:
            with torch.no_grad():
                # One-hot encoding of assignments
                encodings = F.one_hot(pattern_indices, self.num_patterns).float()  # [num_samples, num_patterns]

                # Update cluster sizes
                cluster_size = encodings.sum(dim=0)  # [num_patterns]
                self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)

                # Update pattern sums
                pattern_sum = sparse_codes @ encodings  # [num_atoms, num_patterns]
                self.ema_pattern_sum.mul_(self.ema_decay).add_(pattern_sum, alpha=1 - self.ema_decay)

                # Laplace smoothing to avoid empty clusters
                n = self.ema_cluster_size.sum()
                cluster_size_smoothed = (
                    (self.ema_cluster_size + 1e-5) / (n + self.num_patterns * 1e-5) * n
                )

                # Update patterns
                self.patterns.data = self.ema_pattern_sum / cluster_size_smoothed.unsqueeze(0)

                # Enforce sparsity periodically
                self._enforce_sparsity()

        # Straight-through estimator: forward uses quantized, backward uses original
        quantized_codes_st = sparse_codes + (quantized_codes - sparse_codes).detach()

        # Compute perplexity (effective codebook usage)
        avg_probs = F.one_hot(pattern_indices, self.num_patterns).float().mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        info = {
            'perplexity': perplexity,
            'commitment_loss': commitment_loss,
            'codebook_loss': codebook_loss,
            'pattern_usage': (avg_probs > 0).sum().float() / self.num_patterns,
        }

        return pattern_indices, quantized_codes_st, pattern_loss, info

    def decode_patterns(self, pattern_indices: torch.Tensor) -> torch.Tensor:
        """
        Convert pattern indices back to sparse codes.

        Args:
            pattern_indices: Discrete indices [num_samples] or [batch, num_patches]

        Returns:
            sparse_codes: Sparse coefficient matrix [num_atoms, num_samples]
        """
        # Flatten if needed
        original_shape = pattern_indices.shape
        indices_flat = pattern_indices.view(-1)  # [num_samples]

        # Look up patterns
        sparse_codes = self.patterns[:, indices_flat]  # [num_atoms, num_samples]

        return sparse_codes

    def get_pattern_atoms(self, pattern_idx: int) -> torch.Tensor:
        """Get the atom indices used by a specific pattern (for interpretability)."""
        pattern = self.patterns[:, pattern_idx]
        topk_vals, topk_idx = pattern.abs().topk(self.sparsity)
        return topk_idx, pattern[topk_idx]


class LISTALayer(nn.Module):
    """
    Single layer of Learned ISTA (LISTA).

    Implements one iteration of the LISTA update:
        z = soft_threshold(We @ x + S @ z_prev, threshold)

    Where We and S are learned matrices, and threshold is a learnable parameter.

    This layer can operate in two modes:
    1. Learned mode (default): We and S are independent learnable parameters
    2. Dictionary-coupled mode: We = step * D^T, S = I - step * D^T @ D
       where D is provided at forward time, ensuring gradient flow to dictionary

    Reference: "Learning Fast Approximations of Sparse Coding"
               by Gregor and LeCun (2010)
    """

    def __init__(self, input_dim, code_dim, initial_threshold=0.1, use_dictionary=False,
                 initial_step_size=None):
        super().__init__()
        self.use_dictionary = use_dictionary
        self.code_dim = code_dim

        if use_dictionary:
            # Dictionary-coupled mode: only learn step size and threshold
            # We = step * D^T and S = I - step * D^T @ D are computed from dictionary
            # For convergence: step < 2/L where L = max eigenvalue of D^T @ D
            # For overcomplete dictionaries (code_dim > input_dim), L can be >> 1
            if initial_step_size is None:
                # Heuristic: for overcomplete dict, L ≈ code_dim/input_dim (rough upper bound)
                # Use step = 0.4/overcompleteness for near-critical stability (learning is faster)
                overcompleteness = max(1.0, code_dim / input_dim)
                initial_step_size = 0.4 / overcompleteness
            self.step_size = nn.Parameter(torch.tensor(initial_step_size))
            # Scale threshold with step size: pre_activation ≈ step * correlations
            # With normalized inputs and dictionary, correlations ∈ [-1, 1]
            # So pre_activation ≈ step * [-1, 1], threshold should be < step
            scaled_threshold = initial_threshold * initial_step_size
        else:
            # Learned mode: independent We and S matrices
            self.We = nn.Linear(input_dim, code_dim, bias=False)
            self.S = nn.Linear(code_dim, code_dim, bias=False)
            # Initialize S close to identity for stable start
            nn.init.eye_(self.S.weight)
            # Scale down We initially
            nn.init.xavier_uniform_(self.We.weight, gain=0.1)
            scaled_threshold = initial_threshold

        # Learnable soft threshold (one per code dimension for flexibility)
        self.threshold = nn.Parameter(torch.full((code_dim,), scaled_threshold))

    def soft_threshold(self, x, thresh):
        """Soft thresholding: sign(x) * max(|x| - thresh, 0)"""
        thresh = thresh.abs()  # Ensure positive threshold
        return torch.sign(x) * F.relu(torch.abs(x) - thresh)

    def forward(self, x, z_prev, dictionary=None):
        """
        Args:
            x: Input signal (batch, input_dim)
            z_prev: Previous code estimate (batch, code_dim)
            dictionary: Optional dictionary (input_dim, code_dim) for coupled mode
        Returns:
            z: Updated code estimate (batch, code_dim)
        """
        if self.use_dictionary:
            if dictionary is None:
                raise ValueError("Dictionary required for dictionary-coupled LISTA layer")
            # Compute We and S from dictionary: We = step * D^T, S = I - step * D^T @ D
            # D is (input_dim, code_dim), so D^T is (code_dim, input_dim)
            step = self.step_size.abs()  # Ensure positive step size
            DtD = dictionary.t() @ dictionary  # (code_dim, code_dim)
            # We @ x = step * D^T @ x
            We_x = step * (x @ dictionary)  # (batch, code_dim)
            # S @ z_prev = (I - step * D^T @ D) @ z_prev
            S_z = z_prev - step * (z_prev @ DtD)  # (batch, code_dim)
            pre_activation = We_x + S_z
        else:
            # Standard LISTA update with learned matrices
            pre_activation = self.We(x) + self.S(z_prev)

        z = self.soft_threshold(pre_activation, self.threshold)
        return z


class LISTA(nn.Module):
    """
    Learned ISTA (LISTA) for fast approximate sparse coding.

    Unrolls ISTA into a fixed number of learnable layers. Each layer has:
    - Encoding matrix We (input -> code)
    - Recurrent matrix S (code -> code)
    - Learned soft threshold per dimension

    Two operating modes:
    1. Independent mode (couple_dictionary=False): We and S are learned independently.
       Fast but gradients don't flow to the dictionary.
    2. Coupled mode (couple_dictionary=True): We = step * D^T, S = I - step * D^T @ D
       Slower but ensures gradients flow through the dictionary, making the
       reconstruction D @ coefficients consistent with coefficient computation.

    Advantages over IHT:
    - Learns optimal step sizes and thresholds for the data distribution
    - Typically needs only 3-5 layers vs 15+ IHT iterations
    - End-to-end differentiable with the encoder

    Can optionally share weights across layers (tied LISTA) or use
    separate weights per layer (untied LISTA, more expressive).
    """

    def __init__(
        self,
        input_dim,
        code_dim,
        num_layers=5,
        initial_threshold=0.1,
        tied_weights=False,
        sparsity_level=None,
        couple_dictionary=True,
        initial_step_size=None,
        use_fista=True,
    ):
        """
        Args:
            input_dim: Dimension of input signal (atom_dim)
            code_dim: Dimension of sparse code (num_embeddings)
            num_layers: Number of LISTA iterations/layers
            initial_threshold: Initial soft threshold value
            tied_weights: If True, share We and S across all layers
            sparsity_level: If set, apply hard top-k after soft thresholding
            couple_dictionary: If True, derive We/S from dictionary for gradient flow
            initial_step_size: Initial step size for dictionary-coupled mode. If None, uses
                              a conservative default based on dictionary dimensions.
            use_fista: If True, use FISTA momentum acceleration (recommended)
        """
        super().__init__()
        self.num_layers = num_layers
        self.code_dim = code_dim
        self.tied_weights = tied_weights
        self.sparsity_level = sparsity_level
        self.couple_dictionary = couple_dictionary
        self.use_fista = use_fista

        if tied_weights:
            # Single shared layer
            self.layers = nn.ModuleList([
                LISTALayer(input_dim, code_dim, initial_threshold,
                          use_dictionary=couple_dictionary, initial_step_size=initial_step_size)
            ])
        else:
            # Separate layer for each iteration (more expressive)
            self.layers = nn.ModuleList([
                LISTALayer(input_dim, code_dim, initial_threshold,
                          use_dictionary=couple_dictionary, initial_step_size=initial_step_size)
                for _ in range(num_layers)
            ])

        # Precompute FISTA momentum coefficients for each layer
        # t_{k+1} = (1 + sqrt(1 + 4*t_k^2)) / 2
        # momentum_k = (t_k - 1) / t_{k+1}
        if use_fista:
            t = 1.0
            momentum_coeffs = []
            for _ in range(num_layers):
                t_next = (1.0 + (1.0 + 4.0 * t * t) ** 0.5) / 2.0
                momentum = (t - 1.0) / t_next
                momentum_coeffs.append(momentum)
                t = t_next
            self.register_buffer('momentum_coeffs', torch.tensor(momentum_coeffs))

    def forward(self, x, dictionary=None):
        """
        Compute sparse codes for input signals using FISTA acceleration.

        FISTA (Fast ISTA) adds Nesterov momentum to accelerate convergence:
        - y_k = z_k + momentum_k * (z_k - z_{k-1})  (extrapolation step)
        - z_{k+1} = prox(y_k - step * grad)          (proximal step)

        This achieves O(1/k²) convergence vs O(1/k) for standard ISTA.

        Args:
            x: Input signals (input_dim, batch) - matches OMP/IHT convention
            dictionary: Dictionary matrix (input_dim, code_dim). Required if
                       couple_dictionary=True, optional otherwise.

        Returns:
            z: Sparse codes (code_dim, batch) to match IHT/OMP output format
        """
        # Input is (input_dim, batch) from _sparse_encode - transpose for nn.Linear
        # nn.Linear expects (batch, input_dim)
        x = x.t()  # (batch, input_dim)
        batch_size = x.shape[0]

        # Initialize codes to zero
        z = torch.zeros(batch_size, self.code_dim, device=x.device, dtype=x.dtype)
        z_prev = z  # For FISTA momentum

        # Iterate through LISTA layers with optional FISTA momentum
        for i in range(self.num_layers):
            layer_idx = 0 if self.tied_weights else i

            if self.use_fista and i > 0:
                # FISTA extrapolation: y = z + momentum * (z - z_prev)
                # momentum_{k} = (t_{k-1} - 1) / t_k (stored at index k-1)
                momentum = self.momentum_coeffs[i - 1]
                y = z + momentum * (z - z_prev)
            else:
                y = z

            z_prev = z
            z = self.layers[layer_idx](x, y, dictionary=dictionary)

        # Optional: enforce hard sparsity constraint after soft thresholding
        if self.sparsity_level is not None and self.sparsity_level < self.code_dim:
            # Keep only top-k by magnitude
            abs_z = torch.abs(z)
            _, topk_idx = torch.topk(abs_z, self.sparsity_level, dim=1)
            mask = torch.zeros_like(z)
            mask.scatter_(1, topk_idx, 1.0)
            z = z * mask

        # Transpose to (code_dim, batch) format for compatibility with OMP/IHT
        z = z.t()

        return z


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
    Dictionary Learning Bottleneck with sparse coding and flexible dictionary updates.

    Supports multiple sparse coding algorithms and dictionary update methods:

    Sparse Solvers:
    - OMP: Orthogonal Matching Pursuit (slow, best quality)
    - IHT: Iterative Hard Thresholding (fast, good quality)
    - Top-K: Simple correlation-based selection (fastest, lowest quality)
    - LISTA: Learned ISTA (fast, learns optimal thresholds, recommended)

    Dictionary Update Methods:
    - Backprop-only: Dictionary learned via gradients (fast, integrates with encoder/decoder)
    - K-SVD: Classical SVD-based atom updates (high quality, interpretable)
    - Online Learning: Fast gradient-like updates (good balance)

    Key features:
    - Always normalizes dictionary atoms for numerical stability
    - LISTA learns optimal sparse coding for the data distribution
    - Patch-based processing for efficiency
    - Straight-through estimator for gradient flow

    References:
    - K-SVD: Aharon, Elad, Bruckstein (2006)
    - LISTA: Gregor and LeCun (2010)
    """
    
    def __init__(
        self,
        num_embeddings=512,
        embedding_dim=64,
        sparsity_level=5,
        commitment_cost=0.25,
        ksvd_iterations=1,
        decay=None,
        patch_size=1,
        patch_stride=None,
        epsilon=1e-10,
        sparse_solver="omp",
        iht_iterations=10,
        iht_step_size=None,
        lista_layers=5,
        lista_tied_weights=False,
        lista_initial_threshold=0.1,
        fista_alpha=0.1,
        fista_tolerance=1e-3,
        fista_max_steps=50,
        use_online_learning=True,
        dict_learning_rate=0.1,
        use_backprop_only=False,
        tolerance=None,
        omp_debug=False,
        per_pixel_sparse_coding=False,
        patch_flatten_order='channel_first',
        # Pattern quantization for autoregressive generation
        use_pattern_quantizer=False,
        num_patterns=2048,
        pattern_commitment_cost=0.25,
        pattern_ema_decay=0.99,
        pattern_temperature=1.0,
    ):
        """
        Args:
            num_embeddings: Number of dictionary atoms (dictionary size)
            embedding_dim: Channel dimension of encoder output (NOT full atom dimension!)
            sparsity_level: Max non-zero coefficients per patch
            commitment_cost: Weight for encoder commitment loss
            ksvd_iterations: Number of K-SVD iterations per forward pass
            decay: Deprecated EMA decay parameter (accepted for compatibility)
            patch_size: Spatial patch size (int or tuple). Determines actual atom dimension.
            patch_stride: Optional patch stride (int or tuple). Defaults to patch_size (no overlap).
            epsilon: Small constant for numerical stability
            sparse_solver: 'omp', 'iht', 'topk', or 'lista'
            iht_iterations: Number of IHT iterations (if sparse_solver='iht')
            iht_step_size: IHT step size (None=auto)
            lista_layers: Number of LISTA layers/iterations (if sparse_solver='lista')
            lista_tied_weights: Share weights across LISTA layers (smaller model)
            lista_initial_threshold: Initial soft threshold for LISTA
            fista_alpha: Shrinkage parameter for FISTA (if sparse_solver='fista')
            fista_tolerance: Convergence tolerance for FISTA
            fista_max_steps: Maximum iterations for FISTA
            use_online_learning: Use online learning instead of K-SVD
            dict_learning_rate: Learning rate for online updates
            use_backprop_only: Learn dictionary via gradients only (no K-SVD/online)
            tolerance: Optional OMP residual tolerance (legacy parameter)
            omp_debug: Enable verbose logging for OMP (legacy parameter)
            per_pixel_sparse_coding: If True, apply sparse coding per-pixel within patches
                instead of per-patch. This keeps atom_dim=embedding_dim regardless of
                patch_size, avoiding high-dimensional sparse coding issues.
            use_pattern_quantizer: Enable pattern quantization for autoregressive generation.
                Maps sparse codes to discrete tokens (16 tokens per 128x128 image).
            num_patterns: Size of pattern vocabulary (default 2048)
            pattern_commitment_cost: Weight for pattern commitment loss (default 0.25)
            pattern_ema_decay: EMA decay for pattern updates (default 0.99)
            pattern_temperature: Temperature for pattern matching (default 1.0)

        Note:
            When per_pixel_sparse_coding=False (default):
                atom_dim = embedding_dim × patch_size²
                E.g., embedding_dim=64, patch_size=4 → atom_dim=1024
            When per_pixel_sparse_coding=True:
                atom_dim = embedding_dim (constant regardless of patch_size)
                Patches only affect grouping/efficiency, not sparse coding dimension.
            patch_flatten_order: Order for flattening patches ('channel_first' or 'spatial_first')
                'channel_first': [C, H, W] → [C*H*W] (default, channel-major)
                'spatial_first': [H, W, C] → [H*W*C] (groups channel info together)
        """
        super(DictionaryLearning, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.sparsity_level = sparsity_level
        self.commitment_cost = commitment_cost
        self.ksvd_iterations = ksvd_iterations
        self.epsilon = epsilon
        self.sparse_solver = sparse_solver.lower()
        self.iht_iterations = iht_iterations
        self.iht_step_size = iht_step_size
        self.lista_layers = lista_layers
        self.lista_tied_weights = lista_tied_weights
        self.lista_initial_threshold = lista_initial_threshold
        self.fista_alpha = fista_alpha
        self.fista_tolerance = fista_tolerance
        self.fista_max_steps = fista_max_steps
        self.omp_tolerance = tolerance
        self.omp_debug = omp_debug
        self.decay = decay
        if decay not in (None, 0):
            warnings.warn(
                "DictionaryLearning no longer uses the `decay` parameter; it is ignored.",
                stacklevel=2,
            )
        
        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)
        else:
            if len(patch_size) != 2:
                raise ValueError("patch_size must be an int or a tuple of two ints")
            self.patch_size = tuple(int(p) for p in patch_size)
        self.patch_area = self.patch_size[0] * self.patch_size[1]

        if patch_stride is None:
            self.patch_stride = self.patch_size
        else:
            if isinstance(patch_stride, int):
                self.patch_stride = (patch_stride, patch_stride)
            else:
                if len(patch_stride) != 2:
                    raise ValueError("patch_stride must be an int or a tuple of two ints")
                self.patch_stride = tuple(int(s) for s in patch_stride)

        # Per-pixel sparse coding: each pixel within a patch is a separate signal
        self.per_pixel_sparse_coding = per_pixel_sparse_coding

        # Patch flattening order
        self.patch_flatten_order = patch_flatten_order.lower()

        # IMPORTANT: atom_dim is the ACTUAL dictionary atom dimension
        if per_pixel_sparse_coding:
            # Per-pixel mode: atom_dim = embedding_dim (constant regardless of patch_size)
            # Each pixel is sparse-coded independently, patches only affect grouping
            self.atom_dim = self.embedding_dim
        else:
            # Per-patch mode: atom_dim = embedding_dim × patch_area
            # Each patch is flattened and sparse-coded as a single high-dimensional signal
            self.atom_dim = self.embedding_dim * self.patch_area
        
        # Use backprop-only mode for faster training
        self.use_backprop_only = use_backprop_only
        # Scale dictionary LR down when using overlapping patches (more updates per pixel)
        stride_area = self.patch_stride[0] * self.patch_stride[1]
        self.overlap_factor = max(1.0, self.patch_area / max(stride_area, 1))
        lr_scale = 1.0 / self.overlap_factor
        
        # Initialize dictionary with random atoms (will be replaced by data-driven init)
        # Always make the dictionary trainable so optimizers can update it
        # Tests expect gradients to flow to the dictionary even when K-SVD/online
        # updates are enabled.
        self.dictionary = nn.Parameter(
            torch.randn(self.atom_dim, num_embeddings), requires_grad=True
        )
        # Keep K-SVD/online updates optional for users that want them in addition
        # to gradient-based learning.
        self.enable_ksvd_update = not self.use_backprop_only

        # Track whether dictionary has been initialized from data
        # For high-dimensional patches, random init doesn't work with OMP
        self.register_buffer("_dict_initialized", torch.tensor(False))

        # Always normalize dictionary atoms at initialization
        self._normalize_dictionary()

        # Online dictionary learning parameters (only used if not backprop_only)
        self.dict_learning_rate = dict_learning_rate * lr_scale
        self.use_online_learning = use_online_learning

        # Initialize LISTA if selected as sparse solver
        if self.sparse_solver == "lista":
            self.lista = LISTA(
                input_dim=self.atom_dim,
                code_dim=num_embeddings,
                num_layers=lista_layers,
                initial_threshold=lista_initial_threshold,
                tied_weights=lista_tied_weights,
                sparsity_level=sparsity_level,
                couple_dictionary=True,  # Ensure gradients flow to dictionary
            )
        else:
            self.lista = None

        # Initialize pattern quantizer for autoregressive generation
        self.use_pattern_quantizer = use_pattern_quantizer
        self.num_patterns = num_patterns
        if use_pattern_quantizer:
            self.pattern_quantizer = SparsePatternQuantizer(
                num_patterns=num_patterns,
                num_atoms=num_embeddings,
                sparsity=sparsity_level,
                commitment_cost=pattern_commitment_cost,
                ema_decay=pattern_ema_decay,
                use_ema=True,
                temperature=pattern_temperature,
            )
        else:
            self.pattern_quantizer = None

        # Cache for smooth blending window (created on first use)
        self._blend_window = None

    def _get_blend_window(self, device, dtype):
        """
        Create or retrieve cached smooth blending window for overlapping patches.

        Uses a 2D Hann window to smoothly blend overlapping patch reconstructions,
        reducing visible seam artifacts at patch boundaries.

        Returns:
            window: Tensor of shape (1, patch_h * patch_w, 1) for use with fold weights
        """
        patch_h, patch_w = self.patch_size
        if self._blend_window is not None:
            cached = self._blend_window
            if cached.device == device and cached.dtype == dtype:
                return cached

        # Create 2D Hann window: smooth falloff from center to edges
        # Hann window: 0.5 * (1 - cos(2*pi*n / (N-1)))
        if patch_h > 1:
            win_h = torch.hann_window(patch_h, periodic=False, device=device, dtype=dtype)
        else:
            win_h = torch.ones(1, device=device, dtype=dtype)

        if patch_w > 1:
            win_w = torch.hann_window(patch_w, periodic=False, device=device, dtype=dtype)
        else:
            win_w = torch.ones(1, device=device, dtype=dtype)

        # Outer product to get 2D window
        window_2d = win_h.unsqueeze(1) * win_w.unsqueeze(0)  # (patch_h, patch_w)

        # Flatten and add dimensions for broadcasting with unfold/fold
        # Shape: (1, patch_h * patch_w, 1) to broadcast across batch and num_patches
        window = window_2d.flatten().unsqueeze(0).unsqueeze(-1)

        self._blend_window = window
        return window

    def _normalize_dictionary(self):
        """Normalize all dictionary atoms to have unit L2 norm."""
        with torch.no_grad():
            data = torch.nan_to_num(
                self.dictionary.data,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            norms = torch.linalg.norm(data, dim=0, keepdim=True)
            bad_atoms = (~torch.isfinite(norms)) | (norms < self.epsilon)

            if bad_atoms.any():
                num_bad = int(bad_atoms.sum().item())
                replacement = torch.randn(
                    self.atom_dim, num_bad, device=data.device, dtype=data.dtype
                )
                replacement = replacement / (
                    torch.linalg.norm(replacement, dim=0, keepdim=True) + self.epsilon
                )
                data[:, bad_atoms.squeeze(0)] = replacement
                norms = torch.linalg.norm(data, dim=0, keepdim=True)

            self.dictionary.data.copy_(data / (norms + self.epsilon))

    @torch.no_grad()
    def _initialize_dictionary_from_data(self, patch_tokens: torch.Tensor):
        """
        Initialize dictionary atoms from data using PCA + k-means.

        For high-dimensional patches (large patch_size), random initialization
        leads to near-zero correlations with OMP. This method initializes atoms
        to lie in the data subspace, enabling effective sparse coding from the start.

        Args:
            patch_tokens: Patch data of shape (atom_dim, num_patches)
        """
        if self._dict_initialized:
            return

        M, B = patch_tokens.shape  # M = atom_dim, B = num_patches
        K = self.num_embeddings

        # Center the data
        mean = patch_tokens.mean(dim=1, keepdim=True)
        centered = patch_tokens - mean

        # Compute covariance and get principal components
        # For efficiency, use SVD on the data matrix directly
        # If B < M, it's more efficient to compute B×B covariance
        if B < M:
            # Compute V from X = U @ S @ V.T, then principal components are X @ V
            gram = centered.t() @ centered  # (B, B)
            # Add small regularization for numerical stability
            gram = gram + self.epsilon * torch.eye(B, device=gram.device, dtype=gram.dtype)
            eigenvalues, V = torch.linalg.eigh(gram)
            # Sort by descending eigenvalue
            idx = torch.argsort(eigenvalues, descending=True)
            V = V[:, idx]
            # Principal components in original space
            components = centered @ V  # (M, B)
            # Normalize to get principal directions
            comp_norms = torch.linalg.norm(components, dim=0, keepdim=True).clamp_min(self.epsilon)
            components = components / comp_norms
        else:
            # Standard covariance approach
            cov = centered @ centered.t() / (B - 1)  # (M, M)
            cov = cov + self.epsilon * torch.eye(M, device=cov.device, dtype=cov.dtype)
            eigenvalues, components = torch.linalg.eigh(cov)
            idx = torch.argsort(eigenvalues, descending=True)
            components = components[:, idx]

        # Use top principal components as initial atoms
        num_pca_atoms = min(K, components.shape[1], M)

        if num_pca_atoms >= K:
            # Enough PCA components to fill dictionary
            init_atoms = components[:, :K]
        else:
            # Supplement PCA atoms with k-means centroids from data
            init_atoms = torch.zeros(M, K, device=patch_tokens.device, dtype=patch_tokens.dtype)
            init_atoms[:, :num_pca_atoms] = components[:, :num_pca_atoms]

            # K-means on residuals after projecting out PCA components
            remaining = K - num_pca_atoms
            if remaining > 0 and B > remaining:
                # Project data onto orthogonal complement of PCA space
                pca_proj = components[:, :num_pca_atoms] @ (components[:, :num_pca_atoms].t() @ centered)
                residuals = centered - pca_proj

                # Simple k-means++ initialization on residuals
                # Select diverse samples as additional atoms
                norms = torch.linalg.norm(residuals, dim=0)
                probs = norms / norms.sum()

                selected = []
                for i in range(remaining):
                    if i == 0:
                        # First center: sample proportional to norm
                        idx = torch.multinomial(probs, 1).item()
                    else:
                        # Subsequent: sample proportional to distance from nearest center
                        centers = residuals[:, selected]
                        dists = torch.cdist(residuals.t(), centers.t()).min(dim=1).values
                        probs = dists / dists.sum().clamp_min(self.epsilon)
                        idx = torch.multinomial(probs, 1).item()
                    selected.append(idx)

                additional = residuals[:, selected]
                add_norms = torch.linalg.norm(additional, dim=0, keepdim=True).clamp_min(self.epsilon)
                init_atoms[:, num_pca_atoms:] = additional / add_norms

        # Copy to dictionary and normalize
        self.dictionary.data.copy_(init_atoms)
        self._normalize_dictionary()
        self._dict_initialized.fill_(True)

    def batch_omp(self, X, D):
        """
        Batched Orthogonal Matching Pursuit adapted from amzn/sparse-vqvae utils/pyomp.py.

        Args:
            X: Input signals of shape (M, B)
            D: Dictionary of shape (M, N) with normalized columns.

        Returns:
            coefficients: Sparse coefficient matrix of shape (N, B)
        """
        M, B = X.shape
        _, N = D.shape
        k_max = self.sparsity_level
        tol = self.omp_tolerance if self.omp_tolerance is not None else 1e-7

        dictionary_t = D.t()
        # Gram with small jitter to stabilize Cholesky updates
        diag_eps = 1e-5
        G = dictionary_t @ D
        G = G + diag_eps * torch.eye(N, device=X.device, dtype=X.dtype)
        eps = torch.norm(X, dim=0)  # residual norms
        h_bar = (dictionary_t @ X).t()  # (B, N)

        h = h_bar.clone()
        x = torch.zeros_like(h_bar)
        # Progressive Cholesky factors per batch element
        L = torch.ones(B, 1, 1, device=X.device, dtype=X.dtype)
        I = torch.ones(B, 0, device=X.device, dtype=torch.long)
        I_logic = torch.zeros_like(h_bar, dtype=torch.bool)
        delta = torch.zeros(B, device=X.device, dtype=X.dtype)

        def _update_logical(logical, to_add):
            running_idx = torch.arange(to_add.shape[0], device=to_add.device)
            logical[running_idx, to_add] = True

        k = 0
        batch_idx = torch.arange(B, device=X.device)
        while k < k_max and eps.max() > tol:
            k += 1

            # Select next atom per batch
            index = (h * (~I_logic).float()).abs().argmax(dim=1)
            _update_logical(I_logic, index)
            expanded_batch_idx = batch_idx.unsqueeze(0).expand(k, B).t()

            if k > 1:
                # Cholesky update for each batch element
                G_stack = G[I[batch_idx, :], index[expanded_batch_idx[..., :-1]]].view(B, k - 1, 1)
                # Solve L w = G_stack for w (lower triangular)
                try:
                    w = torch.linalg.solve_triangular(L, G_stack, upper=False)
                except AttributeError:
                    w = torch.triangular_solve(G_stack, L, upper=False).solution
                w = w.view(B, 1, k - 1)
                w_corner = torch.sqrt(torch.clamp(1 - (w ** 2).sum(dim=2, keepdim=True), min=diag_eps))

                # Build new L = [[L, 0], [w, w_corner]]
                k_zeros = torch.zeros(B, k - 1, 1, device=X.device, dtype=X.dtype)
                L = torch.cat(
                    (
                        torch.cat((L, k_zeros), dim=2),
                        torch.cat((w, w_corner), dim=2),
                    ),
                    dim=1,
                )

            I = torch.cat([I, index.unsqueeze(1)], dim=1)

            # Solve for coefficients on active set via Cholesky solve
            h_stack = h_bar[expanded_batch_idx, I[batch_idx, :]].view(B, k, 1)
            try:
                x_stack = torch.cholesky_solve(h_stack, L)
            except AttributeError:
                x_stack = torch.linalg.cholesky_solve(h_stack, L)
            x[batch_idx.unsqueeze(1), I[batch_idx]] = x_stack.squeeze(-1)

            beta = x[batch_idx.unsqueeze(1), I[batch_idx]].unsqueeze(1).bmm(G[I[batch_idx], :]).squeeze(1)
            h = h_bar - beta

            new_delta = (x * beta).sum(dim=1)
            eps = eps + delta - new_delta
            delta = new_delta

            # NaN/inf guard: break early if instability detected
            if not torch.isfinite(x).all() or not torch.isfinite(L).all() or not torch.isfinite(eps).all():
                break

            if self.omp_debug and k % 1 == 0:
                print(
                    f"OMP step {k}, residual max={eps.max().item():.4f}, below tol={(eps < tol).float().mean().item():.4f}"
                )

        return x.t()

    def _sparse_encode(self, tokens, dictionary):
        """
        Sparse coding dispatcher.

        Args:
            tokens: Input signals (M, B)
            dictionary: Normalized dictionary (M, N)
        """
        # Normalize input tokens to prevent coefficient explosion
        token_norms = torch.linalg.norm(tokens, dim=0, keepdim=True).clamp_min(self.epsilon)
        tokens_normalized = tokens / token_norms

        if self.sparse_solver == "lista":
            # LISTA is end-to-end learnable with dictionary coupling
            # We = step * D^T, S = I - step * D^T @ D ensures gradients flow to dictionary
            coefficients = self.lista(tokens_normalized, dictionary)
        elif self.sparse_solver == "fista":
            coefficients = self.fista_sparse_coding(tokens_normalized, dictionary)
        elif self.sparse_solver == "iht":
            coefficients = self.iterative_hard_thresholding(tokens_normalized, dictionary)
        elif self.sparse_solver == "topk":
            coefficients = self.topk_sparse_coding(tokens_normalized, dictionary)
        else:
            coefficients = self.batch_omp(tokens_normalized, dictionary)

        # Scale coefficients back by token norms to preserve reconstruction
        coefficients = coefficients * token_norms

        # Clamp coefficient magnitudes to prevent explosion during early training
        # Max magnitude of ~10 is reasonable for normalized reconstructions
        coeff_max = 10.0
        coefficients = coefficients.clamp(-coeff_max, coeff_max)

        # Guard against numerical issues: replace NaN/Inf with safe values
        coefficients = torch.nan_to_num(coefficients, nan=0.0, posinf=coeff_max, neginf=-coeff_max)

        return coefficients

    def topk_sparse_coding(self, X, D):
        """
        Fast approximate sparse coding using top-k selection with thresholding.

        Much faster than IHT/OMP: single matrix multiplication + top-k selection.
        Good approximation when dictionary atoms are well-learned.
        
        Now includes adaptive thresholding to prevent weak atoms from being selected,
        reducing overfitting.

        Args:
            X: Input signals of shape (M, B)
            D: Dictionary of shape (M, N) with normalized columns

        Returns:
            coefficients: (N, B) with top-k non-zero entries
        """
        M, B = X.shape
        _, N = D.shape
        device = X.device

        # Single matrix multiplication to get all correlations
        correlations = D.t() @ X  # (N, B)

        # Keep only top-k by absolute value
        abs_corr = torch.abs(correlations)
        topk_vals, topk_idx = torch.topk(abs_corr, self.sparsity_level, dim=0)

        # Adaptive thresholding: only keep coefficients above a threshold
        # relative to the maximum correlation per sample
        max_corr = abs_corr.max(dim=0, keepdim=True)[0]  # (1, B)
        threshold = 0.1 * max_corr  # Keep only atoms with >10% of max correlation
        
        # Create sparse coefficient matrix
        coefficients = torch.zeros(N, B, device=device, dtype=X.dtype)
        batch_idx = torch.arange(B, device=device).unsqueeze(0).expand(self.sparsity_level, -1)

        # Gather original correlation values (with sign)
        selected_corr = torch.gather(correlations, 0, topk_idx)
        
        # Apply threshold mask
        threshold_mask = topk_vals > threshold
        selected_corr = selected_corr * threshold_mask
        
        coefficients[topk_idx, batch_idx] = selected_corr

        return coefficients
    
    def _hard_threshold(self, coef):
        """Keep only the top-k coefficients (by magnitude) per sample."""
        k = self.sparsity_level
        if k >= coef.size(0):
            return coef
        abs_coef = torch.abs(coef)
        topk = torch.topk(abs_coef, k, dim=0)
        mask = torch.zeros_like(coef, dtype=torch.bool)
        mask.scatter_(0, topk.indices, True)
        return coef * mask

    def fista_sparse_coding(self, X, D):
        """
        FISTA sparse coding adapted from amzn/sparse-vqvae utils/pyfista.py.

        Uses soft-thresholding with Nesterov momentum; enforces k-sparsity each
        iteration to stay consistent with the rest of the bottleneck.
        """
        _, B = X.shape
        _, N = D.shape
        device = X.device
        dtype = X.dtype

        # Lipschitz constant of gradient: ||D||_2^2
        with torch.no_grad():
            L = torch.linalg.norm(D, 2).pow(2).clamp_min(self.epsilon)

        alpha = self.fista_alpha
        tol = self.fista_tolerance
        max_steps = max(1, int(self.fista_max_steps))

        Z = torch.zeros(N, B, device=device, dtype=dtype)
        Y = Z
        t = torch.tensor(1.0, device=device, dtype=dtype)

        one = torch.tensor(1.0, device=device, dtype=dtype)
        for _ in range(max_steps):
            residual = D @ Y - X
            grad = D.t() @ residual
            Z_next = Y - grad / L

            # Soft threshold
            Z_next = torch.sign(Z_next) * torch.clamp(torch.abs(Z_next) - alpha / L, min=0.0)

            # Enforce hard k-sparsity
            Z_next = self._hard_threshold(Z_next)

            # Nesterov update
            t_next = 0.5 * (one + torch.sqrt(one + 4.0 * t * t))
            momentum = (t - one) / t_next
            Y = Z_next + momentum * (Z_next - Z)

            diff = torch.linalg.norm(Z_next - Z) / (torch.linalg.norm(Z) + self.epsilon)
            Z = Z_next
            t = t_next
            if diff < tol:
                break

        return Z

    def iterative_hard_thresholding(self, X, D):
        """
        Iterative Hard Thresholding (IHT) sparse coding.
        
        IHT is a proper sparse coding algorithm that iteratively:
        1. Takes a gradient step to minimize reconstruction error
        2. Hard-thresholds to maintain sparsity constraint
        
        This is true sparse coding (unlike top-k which is just selection).

        Args:
            X: Input signals of shape (M, B)
            D: Dictionary of shape (M, N) with normalized columns

        Returns:
            coefficients: (N, B) sparse coefficient matrix
        """
        M, B = X.shape
        _, N = D.shape
        device = X.device

        # Initialize coefficients to zero
        coefficients = torch.zeros(N, B, device=device, dtype=X.dtype)
        Dt = D.t()

        # Compute step size from Lipschitz constant
        if self.iht_step_size is not None:
            step_size = self.iht_step_size
        else:
            # For normalized dictionary D, largest singular value is ~1
            # L = ||D||_2^2 ≈ 1, so step_size = 1/L ≈ 1
            # Use slightly smaller for stability
            with torch.no_grad():
                try:
                    # Fast approximation: power iteration for largest eigenvalue
                    v = torch.randn(N, 1, device=device, dtype=X.dtype)
                    for _ in range(3):  # 3 iterations usually enough
                        v = Dt @ (D @ v)
                        v = v / (torch.linalg.norm(v) + self.epsilon)
                    spectral_sq = (v.t() @ Dt @ D @ v).item()
                    step_size = 0.9 / (spectral_sq + self.epsilon)  # Conservative
                except:
                    # Fallback: assume normalized dict has spectral norm ≈ 1
                    step_size = 0.9

        # IHT iterations
        for _ in range(self.iht_iterations):
            # Compute residual: r = X - D*coefficients
            residual = X - D @ coefficients
            
            # Gradient step: coefficients += step_size * D^T * residual
            gradient = Dt @ residual
            coefficients = coefficients + step_size * gradient
            
            # Hard threshold: keep only top-k by magnitude
            coefficients = self._hard_threshold(coefficients)

        return coefficients

    def ksvd_update(self, X, coefficients):
        """
        K-SVD dictionary update step following dictlearn's efficient approach.

        Uses closed-form weighted sum update instead of SVD for efficiency.
        Reference: https://github.com/permfl/dictlearn/blob/master/dictlearn/optimize.py

        Dead atoms (those with no active coefficients) are replaced with the
        worst-reconstructed patches (highest residual norm) rather than random
        noise. This ensures new atoms are immediately useful for sparse coding.
        """
        M, B = X.shape
        N = self.num_embeddings

        with torch.no_grad():
            if not torch.isfinite(X).all() or not torch.isfinite(coefficients).all():
                self._normalize_dictionary()
                return

            # Pre-compute residual for dead atom replacement
            residual = X - self.dictionary @ coefficients
            if not torch.isfinite(residual).all():
                self._normalize_dictionary()
                return

            for k in range(N):
                # Find signals using this atom
                row_k = coefficients[k, :]
                omega_k = torch.abs(row_k) > self.epsilon

                if not omega_k.any():
                    # Replace dead atom with worst-reconstructed patch
                    residual_norms = torch.linalg.norm(residual, dim=0)
                    worst_idx = torch.argmax(residual_norms)
                    new_atom = residual[:, worst_idx].clone()
                    new_atom_norm = torch.linalg.norm(new_atom)
                    if new_atom_norm > self.epsilon:
                        self.dictionary.data[:, k] = new_atom / new_atom_norm
                    else:
                        # Fallback to random if residual is near-zero
                        self.dictionary.data[:, k] = torch.randn(
                            M, device=X.device, dtype=X.dtype
                        )
                        self.dictionary.data[:, k] /= (
                            torch.linalg.norm(self.dictionary.data[:, k]) + self.epsilon
                        )
                    continue

                # Get active signals and coefficients
                w = omega_k.nonzero(as_tuple=True)[0]
                g = coefficients[k, w]  # Coefficients for atom k at active signals
                signals_w = X[:, w]  # Active signals [M, |w|]
                decomp_w = coefficients[:, w]  # All coefficients for active signals [N, |w|]

                # Zero out current atom for reconstruction
                self.dictionary.data[:, k] = 0

                # Compute reconstruction without current atom
                dict_d_w = self.dictionary @ decomp_w  # [M, |w|]

                # dictlearn-style update:
                # d = signals_w @ g - dict_d_w @ g (weighted sum of residuals)
                d = signals_w @ g - dict_d_w @ g  # [M]
                d_norm = torch.linalg.norm(d)

                if d_norm < self.epsilon:
                    # Atom contributes nothing, replace with random
                    d = torch.randn(M, device=X.device, dtype=X.dtype)
                    d_norm = torch.linalg.norm(d)

                d = d / d_norm  # Normalize new atom

                # Update coefficients: g = signals_w.T @ d - dict_d_w.T @ d
                g_new = signals_w.t() @ d - dict_d_w.t() @ d  # [|w|]

                # Update dictionary and coefficients
                self.dictionary.data[:, k] = d
                coefficients[k, w] = g_new

            # Always normalize dictionary after K-SVD update
            self._normalize_dictionary()
    
    def online_dict_update(self, X, coefficients):
        """
        Online dictionary learning update using gradient-like updates.
        
        Fast vectorized implementation that updates all atoms in parallel.
        
        Args:
            X: Input signals of shape (M, B)
            coefficients: Current sparse coefficients of shape (N, B)
        """
        with torch.no_grad():
            if not torch.isfinite(X).all() or not torch.isfinite(coefficients).all():
                self._normalize_dictionary()
                return

            # Compute reconstruction and residual
            recon = self.dictionary @ coefficients
            residual = X - recon
            if not torch.isfinite(residual).all():
                self._normalize_dictionary()
                return
            
            # Vectorized update: D += lr * (residual @ coefficients.T)
            # This computes the gradient for all atoms at once
            # Shape: (M, B) @ (B, N) = (M, N)
            gradient = residual @ coefficients.t()
            if not torch.isfinite(gradient).all():
                self._normalize_dictionary()
                return
            
            # Normalize by usage count (number of non-zero coefficients per atom)
            usage_counts = (torch.abs(coefficients) > 1e-6).sum(dim=1, keepdim=True).t()  # (1, N)
            usage_counts = usage_counts.clamp_min(1.0)  # Avoid division by zero
            
            # Update all atoms at once
            self.dictionary.data += self.dict_learning_rate * gradient / usage_counts
            
            # Always normalize dictionary atoms after update
            self._normalize_dictionary()
    
    def forward(self, z_e):
        """
        Forward pass through K-SVD dictionary learning bottleneck.
        
        Args:
            z_e: Input tensor of shape [batch_size, embedding_dim, height, width]
        
        Returns:
            z_q: Quantized representation with straight-through gradients
            loss: Reconstruction + commitment loss
            coefficients: Sparse coefficients (num_embeddings, batch_size * num_patches)
        """
        z_e = z_e.contiguous()
        batch_size, channels, height, width = z_e.shape
        orig_height, orig_width = height, width
        original_z_e = z_e
        patch_h, patch_w = self.patch_size
        stride = self.patch_stride
        # Ensure dictionary stays finite/normalized before coding
        self._normalize_dictionary()

        # Mirror the reference implementation by padding to make patch tiling well-defined
        pad_h = (patch_h - (height % patch_h)) % patch_h
        pad_w = (patch_w - (width % patch_w)) % patch_w
        z_e_padded = z_e
        if pad_h or pad_w:
            z_e_padded = F.pad(z_e, (0, pad_w, 0, pad_h))
            height = orig_height + pad_h
            width = orig_width + pad_w

        # Extract patches using the same unfolding/reshaping strategy as the provided code
        patches = F.unfold(
            z_e_padded,
            kernel_size=self.patch_size,
            stride=stride,
        ).permute(2, 0, 1).contiguous()  # [num_patches, batch_size, C*P*P]

        patches_shape = patches.shape
        num_patches = patches_shape[0]

        if self.per_pixel_sparse_coding:
            # Per-pixel mode: each pixel within a patch is a separate signal
            # Reshape from (num_patches, batch_size, C*P*P) to (C, num_patches*batch_size*P*P)
            # This treats each spatial position as an independent C-dimensional signal
            patches_reshaped = patches.view(
                num_patches, batch_size, channels, self.patch_area
            )  # [num_patches, batch_size, C, P*P]
            patches_reshaped = patches_reshaped.permute(2, 0, 1, 3).contiguous()  # [C, num_patches, batch_size, P*P]
            patch_tokens = patches_reshaped.view(channels, -1).contiguous()  # [C, num_patches*batch_size*P*P]
        else:
            # Per-patch mode: each patch is flattened into a single high-dimensional signal
            patches_nchw = patches.view(
                patches_shape[0] * patches_shape[1],
                channels,
                patch_h,
                patch_w,
            ).contiguous()

            if self.patch_flatten_order == 'spatial_first':
                # Spatial-first: [N, C, H, W] → [N, H, W, C] → [H*W*C]
                # Groups channel info together (consecutive elements are from same pixel)
                patches_nhwc = patches_nchw.permute(0, 2, 3, 1).contiguous()
                patch_tokens = patches_nhwc.view(-1, self.atom_dim).t().contiguous()  # [H*W*C, num_patches*batch_size]
            else:
                # Channel-first (default): [N, C, H, W] → [C*H*W]
                patch_tokens = patches_nchw.view(-1, self.atom_dim).t().contiguous()  # [C*H*W, num_patches*batch_size]
        
        # Store original dtype for consistency
        orig_dtype = patch_tokens.dtype

        # Always use sparse coding (IHT/OMP/TopK) - operate in float32 for numerical stability
        with torch.amp.autocast('cuda', enabled=False):
            patch_tokens_f32 = patch_tokens.float()  # Preserves gradient connection
            patch_tokens_f32 = torch.nan_to_num(
                patch_tokens_f32, nan=0.0, posinf=0.0, neginf=0.0
            )

            # Initialize dictionary from data on first forward pass
            # Critical for high-dimensional patches where random init fails
            if self.training and not self._dict_initialized:
                self._initialize_dictionary_from_data(patch_tokens_f32)

            # Use .float() on the parameter to preserve gradient flow to self.dictionary
            dictionary_f32 = torch.nan_to_num(
                self.dictionary.float(), nan=0.0, posinf=0.0, neginf=0.0
            )

            atom_norms = torch.linalg.norm(dictionary_f32, dim=0).clamp_min(self.epsilon)
            dict_normalized = dictionary_f32 / atom_norms.unsqueeze(0)

            # Sparse coding to get coefficients (always enforces sparsity)
            coefficients = self._sparse_encode(patch_tokens_f32, dict_normalized)

            # Dictionary update (only if NOT using backprop-only mode)
            if self.training and self.enable_ksvd_update and not self.use_backprop_only:
                if self.use_online_learning:
                    with torch.no_grad():
                        self.online_dict_update(patch_tokens_f32, coefficients.detach())
                    dictionary_f32 = torch.nan_to_num(
                        self.dictionary.float(), nan=0.0, posinf=0.0, neginf=0.0
                    )
                    atom_norms = torch.linalg.norm(dictionary_f32, dim=0).clamp_min(self.epsilon)
                    dict_normalized = dictionary_f32 / atom_norms.unsqueeze(0)
                    coefficients = self._sparse_encode(patch_tokens_f32, dict_normalized)
                else:
                    for _ in range(self.ksvd_iterations):
                        with torch.no_grad():
                            self.ksvd_update(patch_tokens_f32, coefficients.detach())
                        dictionary_f32 = torch.nan_to_num(
                            self.dictionary.float(), nan=0.0, posinf=0.0, neginf=0.0
                        )
                        atom_norms = torch.linalg.norm(dictionary_f32, dim=0).clamp_min(self.epsilon)
                        dict_normalized = dictionary_f32 / atom_norms.unsqueeze(0)
                        coefficients = self._sparse_encode(patch_tokens_f32, dict_normalized)

            # Always use normalized dictionary for reconstruction (numerical stability)
            z_dl_f32 = dict_normalized @ coefficients

        # Reshape back to feature maps
        if self.per_pixel_sparse_coding:
            # Per-pixel mode: z_dl_f32 is (C, num_patches*batch_size*P*P)
            # Reshape back to patches_shape (num_patches, batch_size, C*P*P)
            z_dl_reshaped = z_dl_f32.view(
                channels, num_patches, batch_size, self.patch_area
            )  # [C, num_patches, batch_size, P*P]
            z_dl_reshaped = z_dl_reshaped.permute(1, 2, 0, 3).contiguous()  # [num_patches, batch_size, C, P*P]
            z_dl_patches_raw = z_dl_reshaped.view(*patches_shape)  # [num_patches, batch_size, C*P*P]
            z_dl_patches = z_dl_patches_raw.permute(1, 2, 0).contiguous()  # [batch_size, C*P*P, num_patches]
        else:
            # Per-patch mode: z_dl_f32 is (atom_dim, num_patches*batch_size)
            if self.patch_flatten_order == 'spatial_first':
                # Spatial-first: z_dl_f32 is (H*W*C, num_patches*batch)
                # Need to convert back to (C*H*W) order for fold
                z_dl_nhwc = z_dl_f32.t().view(-1, patch_h, patch_w, channels)  # [num_patches*batch, H, W, C]
                z_dl_nchw = z_dl_nhwc.permute(0, 3, 1, 2).contiguous()  # [num_patches*batch, C, H, W]
                z_dl_flat = z_dl_nchw.view(-1, self.atom_dim)  # [num_patches*batch, C*H*W]
                z_dl_patches = (
                    z_dl_flat
                    .view(*patches_shape)
                    .permute(1, 2, 0)
                    .contiguous()
                )  # [batch_size, C*P*P, num_patches]
            else:
                # Channel-first: z_dl_f32 is (C*H*W, num_patches*batch)
                z_dl_patches = (
                    z_dl_f32.t()
                    .view(*patches_shape)
                    .permute(1, 2, 0)
                    .contiguous()
                )  # [batch_size, C*P*P, num_patches]

        if stride != self.patch_size:
            # Use smooth Hann window blending for overlapping patches
            # This reduces visible seam artifacts at patch boundaries
            blend_window = self._get_blend_window(z_e.device, z_dl_patches.dtype)

            # z_dl_patches shape: (batch_size, channels * patch_h * patch_w, num_patches)
            # We need to apply window per-channel, so reshape appropriately
            patch_h, patch_w = self.patch_size
            num_patches = z_dl_patches.shape[2]

            # Reshape to (batch_size, channels, patch_h * patch_w, num_patches)
            z_dl_reshaped = z_dl_patches.view(batch_size, channels, patch_h * patch_w, num_patches)

            # Apply window weights: broadcast (1, 1, patch_h * patch_w, 1) over batch, channels, patches
            window_4d = blend_window.unsqueeze(0)  # (1, 1, patch_h * patch_w, 1)
            z_dl_weighted = z_dl_reshaped * window_4d

            # Reshape back to fold input format
            z_dl_weighted = z_dl_weighted.view(batch_size, channels * patch_h * patch_w, num_patches)

            # Fold with weighted patches
            z_dl_nchw = F.fold(
                z_dl_weighted,
                output_size=(height, width),
                kernel_size=self.patch_size,
                stride=stride,
            )

            # Compute normalization using same window weights
            # Create weight tensor matching unfold output shape
            ones = torch.ones(
                (batch_size, channels, height, width),
                device=z_e.device,
                dtype=z_e.dtype,
            )
            unfold_ones = F.unfold(ones, kernel_size=self.patch_size, stride=stride)
            # Reshape and apply window
            unfold_ones_reshaped = unfold_ones.view(batch_size, channels, patch_h * patch_w, num_patches)
            unfold_ones_weighted = unfold_ones_reshaped * window_4d
            unfold_ones_weighted = unfold_ones_weighted.view(batch_size, channels * patch_h * patch_w, num_patches)

            fold_weights = F.fold(
                unfold_ones_weighted,
                output_size=(height, width),
                kernel_size=self.patch_size,
                stride=stride,
            ).clamp_min(self.epsilon)

            z_dl_nchw = z_dl_nchw / fold_weights
        else:
            # Non-overlapping patches: simple fold without blending
            z_dl_nchw = F.fold(
                z_dl_patches,
                output_size=(height, width),
                kernel_size=self.patch_size,
                stride=stride,
            )

        if pad_h or pad_w:
            z_dl_nchw = z_dl_nchw[:, :, :orig_height, :orig_width]

        z_dl = z_dl_nchw.to(orig_dtype)

        # Compute losses for monitoring and encoder training
        z_e_nhwc = original_z_e.permute(0, 2, 3, 1).contiguous()
        z_dl_nhwc = z_dl.permute(0, 2, 3, 1).contiguous()

        # Encoder commitment loss: encourages encoder to match reconstruction
        e_latent_loss = F.mse_loss(z_dl_nhwc.detach(), z_e_nhwc)
        # Dictionary reconstruction loss: encourages good reconstruction
        dl_latent_loss = F.mse_loss(z_dl_nhwc, z_e_nhwc.detach())
        # Overlap-aware scaling keeps per-pixel gradient magnitude consistent when stride < patch_size
        overlap_scale = 1.0 / self.overlap_factor
        loss = overlap_scale * (self.commitment_cost * e_latent_loss + dl_latent_loss)

        # Apply pattern quantization for autoregressive generation
        pattern_indices = None
        pattern_info = None
        pattern_loss = torch.tensor(0.0, device=z_e.device, dtype=z_e.dtype)
        if self.use_pattern_quantizer and self.pattern_quantizer is not None:
            # coefficients shape: [num_atoms, num_patches * batch_size]
            pattern_indices, quantized_coeffs, pattern_loss, pattern_info = self.pattern_quantizer(coefficients)
            # Add pattern loss to total loss
            loss = loss + pattern_loss
            # Reshape pattern_indices to [batch_size, num_patches]
            # Note: pattern_indices is in PATCH-MAJOR order [p0b0, p0b1, ..., p1b0, ...]
            # so we reshape to [num_patches, batch_size] then transpose to get [batch_size, num_patches]
            pattern_indices = pattern_indices.view(num_patches, batch_size).t().contiguous()

        # Stash component losses for external logging/debugging
        self._last_bottleneck_losses = {
            'e_latent_loss': e_latent_loss.detach(),
            'dl_latent_loss': dl_latent_loss.detach(),
            'pattern_loss': pattern_loss.detach(),
        }

        # Straight-through estimator (same as VQ-VAE)
        # Forward: uses z_dl (dictionary reconstruction)
        # Backward: gradients copy through to encoder via z_e
        # Dictionary gradients flow through dl_latent_loss above
        z_dl_out = original_z_e + (z_dl - original_z_e).detach()

        # Convert coefficients to output dtype
        coefficients_out = coefficients.to(orig_dtype)

        # Return pattern indices if pattern quantization is enabled
        if self.use_pattern_quantizer:
            return z_dl_out, loss, coefficients_out, pattern_indices, pattern_info
        return z_dl_out, loss, coefficients_out

    def orthogonality_loss(self):
        """
        Penalize correlated atoms; zero when dictionary columns are orthonormal.
        """
        with torch.no_grad():
            self._normalize_dictionary()
        D = self.dictionary
        gram = D.t() @ D
        eye = torch.eye(self.num_embeddings, device=gram.device, dtype=gram.dtype)
        return ((gram - eye) ** 2).mean()

    @torch.no_grad()
    def decode_pattern_indices_to_latent(self, pattern_indices: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct a latent feature map from discrete pattern indices.

        Assumes non-overlapping patches (patch_stride == patch_size). For overlapping
        stride, extend this to use the Hann blending path from the forward pass.

        Args:
            pattern_indices: LongTensor [B, num_patches]

        Returns:
            z_dl_nchw: FloatTensor [B, embedding_dim, latent_h, latent_w]
        """
        if not self.use_pattern_quantizer or self.pattern_quantizer is None:
            raise ValueError("Pattern quantizer is not enabled on this bottleneck")

        B, num_patches = pattern_indices.shape

        # Infer grid (assume square). This matches training for stride=patch_size.
        grid = int(math.sqrt(num_patches))
        if grid * grid != num_patches:
            raise ValueError(f"num_patches={num_patches} is not a perfect square; generation reshape is ambiguous")

        patch_h, patch_w = self.patch_size
        latent_h = grid * patch_h
        latent_w = grid * patch_w

        # Decode patterns to sparse codes: [num_atoms, B*num_patches]
        # Switch back to patch-major flattening expected by the dictionary
        flat_indices = pattern_indices.transpose(0, 1).reshape(-1)
        sparse_codes = self.pattern_quantizer.decode_patterns(flat_indices)  # [num_atoms, B*num_patches]

        # Normalize dictionary before reconstruction (matches training path)
        dictionary = torch.nan_to_num(self.dictionary, nan=0.0, posinf=0.0, neginf=0.0)
        atom_norms = torch.linalg.norm(dictionary, dim=0, keepdim=True).clamp_min(self.epsilon)
        dict_normalized = dictionary / atom_norms

        # Reconstruct flattened patches: [atom_dim, B*num_patches]
        z_dl_flat = dict_normalized @ sparse_codes  # (atom_dim, B*num_patches)

        # Reshape to fold input
        # Note: sparse_codes are in PATCH-MAJOR order [num_patches * B], so we need to
        # reshape to [num_patches, B, ...] first, then transpose to [B, num_patches, ...]
        if self.patch_flatten_order == 'spatial_first':
            # (H*W*C, N) -> [num_patches*B, H, W, C] -> [num_patches, B, C, H, W] -> [B, C*H*W, num_patches]
            z_dl_nhwc = z_dl_flat.t().contiguous().view(-1, patch_h, patch_w, self.embedding_dim)
            z_dl_nchw = z_dl_nhwc.permute(0, 3, 1, 2).contiguous()  # [num_patches*B, C, H, W]
            # Reshape to patch-major then transpose to batch-major
            z_dl_patches = (
                z_dl_nchw.view(num_patches, B, -1)  # [num_patches, B, C*H*W]
                .permute(1, 2, 0)  # [B, C*H*W, num_patches]
                .contiguous()
            )
        else:
            # channel_first: (C*H*W, N) -> [num_patches, B, C*H*W] -> [B, C*H*W, num_patches]
            z_dl_patches = (
                z_dl_flat.t().contiguous()
                .view(num_patches, B, -1)  # [num_patches, B, C*H*W]
                .permute(1, 2, 0)  # [B, C*H*W, num_patches]
                .contiguous()
            )

        # Non-overlapping fold back to latent map
        z_dl_nchw = F.fold(
            z_dl_patches,
            output_size=(latent_h, latent_w),
            kernel_size=self.patch_size,
            stride=self.patch_stride,
        )

        return z_dl_nchw
