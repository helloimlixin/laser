import warnings
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """Vector Quantizer implementation for VQ-VAE without EMA updates.

    The Vector Quantizer maps continuous encodings to discrete codes from a learned
    codebook. This is the key component that enables VQ-VAE to learn discrete
    representations.
    
    Args:
        num_embeddings: number of discrete codebook embeddings, denoted as K
        embedding_dim: dimensionality of each embedding vector, denoted as D
        commitment_cost: weight for the commitment loss term, default is 0.25 as in the original VQ-VAE paper
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        commitment_cost=0.25
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim) # codebook embeddings, shape [K, D]
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z_e: torch.Tensor):
        """
        Args:
            z_e: input tensor of shape [B, C, H, W] from the encoder

        Returns:
            Tuple of (quantized tensor, loss, perplexity, one-hot encodings)
        """
        if z_e.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected channel dim {self.embedding_dim} but received {z_e.shape[1]}"
            )

        # permute to [B, H, W, C] for convenience and flatten to [N, C]
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        ze_shape = z_e.shape
        
        # flatten input z_e to [N, C]
        ze_flat = z_e.view(-1, self.embedding_dim)

        # compute distances between latent feature vectors and codebook embedding vectors
        # the distance matrix looks like
        #
        #   [[d(ze_1, e_1), d(ze_1, e_2), ..., d(ze_1, e_K)],
        #    [d(ze_2, e_1), d(ze_2, e_2), ..., d(ze_2, e_K)],
        #    ...
        #    [d(ze_N, e_1), d(ze_N, e_2), ..., d(ze_N, e_K)]]
        #
        # where d(ze, e) = ||ze - e||^2 is the squared Euclidean distance, N = B x H x W and K is the
        # number of codebook embedding vectors. The formula of
        # the squared Euclidean distance is thus expanded to:
        # d(ze, e) = ||ze||^2 + ||e||^2 - 2 * ze^T * e
        distances = (
            torch.sum(ze_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(ze_flat, self.embedding.weight.t())
        )
        
        # derive the encoding indices
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0],
            self.num_embeddings,
            device=z_e.device,
            dtype=z_e.dtype,
        )
        
        # create one-hot encodings with the scatter_ method, with dimension [N, K]
        encodings.scatter_(1, encoding_indices, 1)
        
        # quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(ze_shape) # [B, H, W, C]
        
        # compute loss
        e_latent_loss = F.mse_loss(quantized.detach(), z_e)
        q_latent_loss = F.mse_loss(quantized, z_e.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss 
        
        # use straight-through estimator
        quantized = z_e + (quantized - z_e).detach()
        
        # compute perplexity
        avg_probs = torch.mean(encodings, dim=0) # compute average probability of each codebook entry by simply averaging the one-hot encodings
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # reshape quantized back to [B, C, H, W]
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        return quantized, loss, perplexity, encodings

class DictionaryLearning(nn.Module):
    """
    Dictionary Learning Bottleneck with sparse coding and flexible dictionary updates.

    Supports multiple sparse coding algorithms and dictionary update methods:

    Sparse Solvers:
    - OMP: Orthogonal Matching Pursuit (slow, best quality)

    Dictionary Update Methods:
    - Backprop-only: Dictionary learned via gradients (fast, integrates with encoder/decoder)

    Key features:
    - Always normalizes dictionary atoms for numerical stability
    - Patch-based processing for efficiency
    - Straight-through estimator for gradient flow
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
        # Simplified: force OMP-only sparse coding
        self.lista_layers = 0
        self.lista_tied_weights = False
        self.lista_initial_threshold = 0.0
        self.fista_alpha = 0.0
        self.fista_tolerance = 0.0
        self.fista_max_steps = 0
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
        # Only backprop-based dictionary learning is supported in this simplified setup
        self.use_backprop_only = True
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

        # Always normalize dictionary atoms at initialization
        self._normalize_dictionary()

        # Online dictionary learning disabled
        self.dict_learning_rate = 0.0

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
