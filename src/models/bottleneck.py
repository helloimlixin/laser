import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class _VectorQuantizerBase(nn.Module):
    """Shared helpers for VectorQuantizer variants."""

    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self._last_diag = {}

    def _check_input(self, z_e):
        if z_e.dim() != 4:
            raise ValueError(f"Expected input [B, C, H, W], got {tuple(z_e.shape)}")
        if z_e.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected channel dim {self.embedding_dim} but received {z_e.shape[1]}"
            )

    def _flatten(self, z_e):
        # Move channels last to simplify flattening into N = B*H*W vectors.
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        return z_e, z_e.view(-1, self.embedding_dim)

    def _compute_distances(self, z_flat):
        # Squared L2 distance: ||z - e||^2 = ||z||^2 + ||e||^2 - 2 z^T e.
        codebook = self.embedding.weight
        z_sq = torch.sum(z_flat ** 2, dim=1, keepdim=True)
        e_sq = torch.sum(codebook ** 2, dim=1)
        return z_sq + e_sq - 2 * torch.matmul(z_flat, codebook.t())

    def _build_encodings(self, indices, dtype, device):
        # Dense one-hot encodings [N, K] used for diagnostics and EMA updates.
        encodings = torch.zeros(
            indices.shape[0],
            self.num_embeddings,
            device=device,
            dtype=dtype,
        )
        encodings.scatter_(1, indices.unsqueeze(1), 1)
        return encodings

    def _compute_perplexity(self, encodings):
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return perplexity, avg_probs

    def _update_diag(self, distances, avg_probs, perplexity):
        with torch.no_grad():
            self._last_diag = {
                "codes_used": (avg_probs > 0).sum(),
                "code_usage": (avg_probs > 0).float().mean(),
                "perplexity": perplexity.detach(),
                "dist_min": distances.min().detach(),
                "dist_max": distances.max().detach(),
                "dist_mean": distances.mean().detach(),
            }

    def _encode(self, z_e):
        z_e_nhwc, z_flat = self._flatten(z_e)
        distances = self._compute_distances(z_flat)
        indices = torch.argmin(distances, dim=1)
        encodings = self._build_encodings(indices, dtype=z_flat.dtype, device=z_flat.device)
        return z_e_nhwc, z_flat, distances, indices, encodings


class VectorQuantizer(_VectorQuantizerBase):
    """Vector Quantizer implementation for VQ-VAE without EMA updates."""

    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__(num_embeddings, embedding_dim, commitment_cost)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z_e: torch.Tensor):
        """
        Args:
            z_e: input tensor of shape [B, C, H, W] from the encoder

        Returns:
            Tuple of (quantized tensor, loss, perplexity, one-hot encodings)
        """
        self._check_input(z_e)

        z_e_nhwc, _, distances, indices, encodings = self._encode(z_e)
        quantized = self.embedding(indices).view_as(z_e_nhwc)

        # Codebook loss encourages embedding vectors to match encoder outputs.
        q_latent_loss = F.mse_loss(quantized, z_e_nhwc.detach())
        # Commitment loss prevents encoder outputs from fluctuating far from the codebook.
        e_latent_loss = F.mse_loss(quantized.detach(), z_e_nhwc)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator: forward uses quantized, backward sees identity.
        quantized = z_e_nhwc + (quantized - z_e_nhwc).detach()

        perplexity, avg_probs = self._compute_perplexity(encodings)
        self._update_diag(distances, avg_probs, perplexity)

        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return quantized, loss, perplexity, encodings


class VectorQuantizerEMA(_VectorQuantizerBase):
    """Vector Quantizer implementation for VQ-VAE with EMA updates."""

    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, ema_decay=0.99, epsilon=1e-10):
        super().__init__(num_embeddings, embedding_dim, commitment_cost)

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.normal_()
        # EMA codebooks are updated manually; disable gradient tracking.
        self.embedding.weight.requires_grad = False

        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.randn(num_embeddings, embedding_dim), requires_grad=False)

        self.ema_decay = ema_decay
        self.epsilon = epsilon

    def forward(self, z_e: torch.Tensor):
        """
        Args:
            z_e: input tensor of shape [B, C, H, W] from the encoder

        Returns:
            Tuple of (quantized tensor, loss, perplexity, one-hot encodings)
        """
        self._check_input(z_e)

        z_e_nhwc, z_flat, distances, indices, encodings = self._encode(z_e)
        quantized = self.embedding(indices).view_as(z_e_nhwc)

        if self.training:
            # EMA update uses assignment counts and accumulated latent sums.
            with torch.no_grad():
                counts = encodings.sum(dim=0)
                self.ema_cluster_size.mul_(self.ema_decay).add_(counts, alpha=1 - self.ema_decay)

                dw = encodings.t() @ z_flat
                self._ema_w.mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)

                n = self.ema_cluster_size.sum()
                # Laplace smoothing avoids empty clusters causing NaNs.
                cluster_size = (
                    (self.ema_cluster_size + self.epsilon)
                    / (n + self.num_embeddings * self.epsilon) * n
                )
                self.embedding.weight.copy_(self._ema_w / cluster_size.unsqueeze(1))

        # Encoder commitment only; codebook is updated via EMA.
        e_latent_loss = F.mse_loss(quantized.detach(), z_e_nhwc)
        loss = self.commitment_cost * e_latent_loss

        # Straight-through estimator keeps encoder gradients alive.
        quantized = z_e_nhwc + (quantized - z_e_nhwc).detach()

        perplexity, avg_probs = self._compute_perplexity(encodings)
        self._update_diag(distances, avg_probs, perplexity)

        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return quantized, loss, perplexity, encodings


class DictionaryLearning(nn.Module):
    """Dictionary Learning bottleneck using OMP sparse coding and backprop updates.

    Notes:
        - Only 'omp' sparse_solver is supported.
        - K-SVD/online updates and pattern quantization are not implemented.
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
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.sparsity_level = sparsity_level
        self.commitment_cost = commitment_cost
        # Retain config fields for logging/compatibility, but OMP + backprop is fixed.
        self.ksvd_iterations = ksvd_iterations
        self.use_online_learning = use_online_learning
        self.epsilon = epsilon
        self.omp_tolerance = tolerance
        self.omp_debug = omp_debug
        self.dict_learning_rate = dict_learning_rate
        self.sparse_solver = sparse_solver.lower()

        if self.sparse_solver != "omp":
            raise ValueError(f"Only sparse_solver='omp' is supported, got '{sparse_solver}'")

        if decay not in (None, 0):
            warnings.warn(
                "DictionaryLearning no longer uses the `decay` parameter; it is ignored.",
                stacklevel=2,
            )

        if use_pattern_quantizer:
            warnings.warn(
                "Pattern quantization is not implemented in DictionaryLearning; ignoring.",
                stacklevel=2,
            )

        if not use_backprop_only:
            warnings.warn(
                "Only backprop-based dictionary updates are supported; forcing use_backprop_only=True.",
                stacklevel=2,
            )
        self.use_backprop_only = True

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

        self.per_pixel_sparse_coding = per_pixel_sparse_coding
        self.patch_flatten_order = patch_flatten_order.lower()
        if self.patch_flatten_order not in ("channel_first", "spatial_first"):
            raise ValueError("patch_flatten_order must be 'channel_first' or 'spatial_first'")

        if self.per_pixel_sparse_coding and self.patch_size != (1, 1):
            warnings.warn(
                "per_pixel_sparse_coding ignores patch_size/patch_stride; coding per pixel.",
                stacklevel=2,
            )

        self.atom_dim = self.embedding_dim if self.per_pixel_sparse_coding else self.embedding_dim * self.patch_area

        self.dictionary = nn.Parameter(
            torch.randn(self.atom_dim, num_embeddings), requires_grad=True
        )
        self._normalize_dictionary()

        self.register_buffer("is_initialized", torch.tensor(False))
        self.register_buffer("atom_usage_ema", torch.ones(num_embeddings))
        self._last_diag = {}
        self._last_bottleneck_losses = {}
        self.enable_ksvd_update = False

    def _normalize_dictionary(self):
        with torch.no_grad():
            self.dictionary.copy_(F.normalize(self.dictionary, p=2, dim=0, eps=self.epsilon))

    def _maybe_initialize_dictionary(self, patches_flat):
        if not (self.training and not self.is_initialized.item()):
            return
        with torch.no_grad():
            n_patches = patches_flat.shape[0]
            if n_patches == 0:
                return
            # Seed the dictionary with random data patches to stabilize early OMP.
            if n_patches >= self.num_embeddings:
                indices = torch.randperm(n_patches, device=patches_flat.device)[:self.num_embeddings]
                init_atoms = patches_flat[indices]
            else:
                indices = torch.randint(0, n_patches, (self.num_embeddings,), device=patches_flat.device)
                init_atoms = patches_flat[indices] + 0.01 * torch.randn(
                    self.num_embeddings, self.atom_dim, device=patches_flat.device, dtype=patches_flat.dtype
                )
            self.dictionary.copy_(init_atoms.t())
            self._normalize_dictionary()
            self.is_initialized.fill_(True)

    def _patchify(self, x):
        B, C, H, W = x.shape
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_stride)

        if self.patch_flatten_order == "channel_first":
            # Each patch is [C, ph, pw] flattened with channel-major ordering.
            patches = patches.transpose(1, 2).reshape(-1, patches.shape[1])
        else:
            ph, pw = self.patch_size
            # Reorder to [ph, pw, C] per patch so spatial neighbors are contiguous.
            patches = patches.view(B, C, ph, pw, -1)
            patches = patches.permute(0, 4, 2, 3, 1).reshape(-1, self.atom_dim)

        h_out = (H - self.patch_size[0]) // self.patch_stride[0] + 1
        w_out = (W - self.patch_size[1]) // self.patch_stride[1] + 1
        return patches, (h_out, w_out)

    def _unpatchify(self, patches, output_size, original_size):
        B, C, H, W = original_size
        h_out, w_out = output_size
        L = h_out * w_out

        if self.patch_flatten_order == "channel_first":
            patches = patches.view(B, L, -1).transpose(1, 2)
        else:
            ph, pw = self.patch_size
            patches = patches.view(B, L, ph, pw, C)
            patches = patches.permute(0, 4, 2, 3, 1).reshape(B, C * ph * pw, L)

        x_recon = F.fold(
            patches,
            output_size=(H, W),
            kernel_size=self.patch_size,
            stride=self.patch_stride,
        )

        if self.patch_stride != self.patch_size:
            # Overlapping patches are summed by fold; divide by overlap count.
            ones = torch.ones_like(x_recon)
            count_patches = F.unfold(ones, kernel_size=self.patch_size, stride=self.patch_stride)
            count_map = F.fold(
                count_patches,
                output_size=(H, W),
                kernel_size=self.patch_size,
                stride=self.patch_stride,
            )
            x_recon = x_recon / count_map

        return x_recon

    def _sparse_code(self, patches_flat):
        self._maybe_initialize_dictionary(patches_flat)
        self._normalize_dictionary()

        # OMP expects signals as columns: X [atom_dim, N], D [atom_dim, K].
        signals = patches_flat.t()
        coeffs = self.batch_omp(signals, self.dictionary)
        recon_patches_flat = torch.matmul(self.dictionary, coeffs).t()
        return recon_patches_flat, coeffs

    def _maybe_revive_dead_atoms(self, patches_flat, recon_patches_flat, coeffs):
        with torch.no_grad():
            current_usage = (coeffs.abs() > 1e-5).float().mean(dim=1)
            self.atom_usage_ema.mul_(0.99).add_(current_usage, alpha=0.01)

            dead_mask = self.atom_usage_ema < 1e-3
            n_dead = int(dead_mask.sum().item())
            if n_dead == 0:
                return

            # Replace rarely-used atoms with high-error patches to re-activate them.
            n_candidates = min(n_dead * 4, patches_flat.size(0))
            if n_candidates == 0:
                return

            err = (patches_flat - recon_patches_flat).norm(dim=1)
            top_indices = torch.topk(err, k=n_candidates).indices
            rand_subset = torch.randperm(n_candidates, device=patches_flat.device)[:n_dead]
            replacement_indices = top_indices[rand_subset]

            replacements = patches_flat[replacement_indices]
            replacements = replacements / (replacements.norm(dim=1, keepdim=True) + 1e-8)

            self.dictionary[:, dead_mask].copy_(replacements.t())
            self.atom_usage_ema[dead_mask] = self.atom_usage_ema.mean()

    def _update_diag(self, coeffs):
        with torch.no_grad():
            dict_norms = self.dictionary.norm(dim=0)
            coeff_abs = coeffs.abs()
            self._last_diag = {
                "dict_norm_min": dict_norms.min(),
                "dict_norm_max": dict_norms.max(),
                "coeff_norm_mean": coeff_abs.mean(),
                "coeff_norm_max": coeff_abs.max(),
            }

    def orthogonality_loss(self):
        d = F.normalize(self.dictionary, p=2, dim=0, eps=self.epsilon)
        gram = d.t() @ d
        identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        return (gram - identity).pow(2).mean()

    def forward(self, z_e):
        B, C, H, W = z_e.shape
        if C != self.embedding_dim:
            raise ValueError(
                f"Expected channel dim {self.embedding_dim} but received {C}"
            )

        if self.per_pixel_sparse_coding:
            # Per-pixel mode: each spatial location is sparse-coded independently.
            patches_flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, C)
            recon_patches_flat, coeffs = self._sparse_code(patches_flat)
            z_dl = recon_patches_flat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        else:
            # Patch mode: each patch is a single sparse-coded signal.
            patches_flat, spatial_dims = self._patchify(z_e)
            recon_patches_flat, coeffs = self._sparse_code(patches_flat)
            z_dl = self._unpatchify(recon_patches_flat, spatial_dims, (B, C, H, W))

        if self.training:
            self._maybe_revive_dead_atoms(patches_flat, recon_patches_flat, coeffs)

        # Dictionary loss updates D; commitment loss keeps encoder outputs stable.
        dl_latent_loss = F.mse_loss(z_dl, z_e.detach())
        e_latent_loss = F.mse_loss(z_dl.detach(), z_e)
        loss = dl_latent_loss + self.commitment_cost * e_latent_loss

        self._last_bottleneck_losses = {
            "dl_latent_loss": dl_latent_loss.detach(),
            "e_latent_loss": e_latent_loss.detach(),
            "pattern_loss": z_e.new_tensor(0.0),
        }
        self._update_diag(coeffs)

        # Straight-through estimator keeps gradients for the encoder path.
        z_dl = z_e + (z_dl - z_e).detach()
        return z_dl, loss, coeffs

    def batch_omp(self, X, D):
        """
        Batched Orthogonal Matching Pursuit adapted from amzn/sparse-vqvae utils/pyomp.py.

        Args:
            X: Input signals of shape (M, B)
            D: Dictionary of shape (M, N) with normalized columns.

        Returns:
            coefficients: Sparse coefficient matrix of shape (N, B)
        """
        if X.dim() != 2 or D.dim() != 2:
            raise ValueError("X and D must be 2D tensors")
        M, B = X.shape
        _, N = D.shape
        k_max = self.sparsity_level
        tol = self.omp_tolerance if self.omp_tolerance is not None else 1e-7

        if k_max <= 0 or B == 0:
            return torch.zeros((N, B), device=X.device, dtype=X.dtype)

        dictionary_t = D.t()
        # Gram matrix with jitter to stabilize Cholesky updates.
        diag_eps = 1e-5
        G = dictionary_t @ D
        G = G + diag_eps * torch.eye(N, device=X.device, dtype=X.dtype)
        eps = torch.norm(X, dim=0)  # residual norms per signal
        h_bar = (dictionary_t @ X).t()  # (B, N) correlations

        h = h_bar.clone()
        x = torch.zeros_like(h_bar)
        # Progressive Cholesky factors per batch element (one per signal).
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

            # Select next atom per batch (highest residual correlation).
            index = (h * (~I_logic).float()).abs().argmax(dim=1)
            _update_logical(I_logic, index)
            expanded_batch_idx = batch_idx.unsqueeze(0).expand(k, B).t()

            if k > 1:
                # Rank-1 Cholesky update for each batch element.
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
