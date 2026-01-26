import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Simplified VQ-VAE baseline (lucidrains-style) without EMA."""

    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(
            -1 / num_embeddings, 1 / num_embeddings
        )

    def forward(self, z_e: torch.Tensor):
        """
        Args:
            z_e: input tensor of shape [B, C, H, W] from the encoder

        Returns:
            Tuple of (quantized tensor, loss, perplexity, one-hot encodings)
        """
        if z_e.dim() != 4:
            raise ValueError(
                f"Expected input [B, C, H, W], got {tuple(z_e.shape)}"
            )
        if z_e.shape[1] != self.embedding_dim:
            raise ValueError(
                (
                    "Expected channel dim "
                    f"{self.embedding_dim} but received {z_e.shape[1]}"
                )
            )

        # [B, C, H, W] -> [B, H, W, C] -> [N, C]
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        z_flat = z_e.view(-1, self.embedding_dim)

        # Squared L2 distance to each codebook vector.
        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * z_flat @ self.embedding.weight.t()
        )
        indices = torch.argmin(distances, dim=1)

        encodings = torch.zeros(
            indices.shape[0],
            self.num_embeddings,
            device=z_flat.device,
            dtype=z_flat.dtype,
        )
        encodings.scatter_(1, indices.unsqueeze(1), 1)

        quantized = self.embedding(indices).view_as(z_e)

        q_latent_loss = F.mse_loss(quantized, z_e.detach())
        e_latent_loss = F.mse_loss(quantized.detach(), z_e)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator: forward uses quantized, backward sees
        # identity.
        quantized = z_e + (quantized - z_e).detach()

        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        )

        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return quantized, loss, perplexity, encodings


class VectorQuantizerEMA(nn.Module):
    """Simplified VQ-VAE baseline with EMA updates (lucidrains-style)."""

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        commitment_cost=0.25,
        ema_decay=0.99,
        epsilon=1e-10,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.epsilon = epsilon

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.normal_()
        self.embedding.weight.requires_grad = False

        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer(
            "_ema_w", torch.randn(num_embeddings, embedding_dim)
        )

    def forward(self, z_e: torch.Tensor):
        """
        Args:
            z_e: input tensor of shape [B, C, H, W] from the encoder

        Returns:
            Tuple of (quantized tensor, loss, perplexity, one-hot encodings)
        """
        if z_e.dim() != 4:
            raise ValueError(
                f"Expected input [B, C, H, W], got {tuple(z_e.shape)}"
            )
        if z_e.shape[1] != self.embedding_dim:
            raise ValueError(
                (
                    "Expected channel dim "
                    f"{self.embedding_dim} but received {z_e.shape[1]}"
                )
            )

        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        z_flat = z_e.view(-1, self.embedding_dim)

        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * z_flat @ self.embedding.weight.t()
        )
        indices = torch.argmin(distances, dim=1)

        encodings = torch.zeros(
            indices.shape[0],
            self.num_embeddings,
            device=z_flat.device,
            dtype=z_flat.dtype,
        )
        encodings.scatter_(1, indices.unsqueeze(1), 1)

        quantized = self.embedding(indices).view_as(z_e)

        if self.training:
            with torch.no_grad():
                counts = encodings.sum(dim=0).to(
                    self.ema_cluster_size.dtype
                )
                self.ema_cluster_size.mul_(self.ema_decay).add_(
                    counts, alpha=1 - self.ema_decay
                )

                dw = (encodings.t() @ z_flat).to(self._ema_w.dtype)
                self._ema_w.mul_(self.ema_decay).add_(
                    dw, alpha=1 - self.ema_decay
                )

                n = self.ema_cluster_size.sum()
                cluster_size = (
                    (self.ema_cluster_size + self.epsilon)
                    / (n + self.num_embeddings * self.epsilon) * n
                )
                self.embedding.weight.copy_(
                    self._ema_w / cluster_size.unsqueeze(1)
                )

        e_latent_loss = F.mse_loss(quantized.detach(), z_e)
        loss = self.commitment_cost * e_latent_loss

        quantized = z_e + (quantized - z_e).detach()

        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        )

        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return quantized, loss, perplexity, encodings


class DictionaryLearning(nn.Module):
    """Dictionary Learning bottleneck using OMP sparse coding and backprop
    updates.

    Notes:
        - Only 'omp' sparse_solver is supported.
        - Online updates and pattern quantization are not implemented.
    """

    def __init__(
        self,
        num_embeddings=512,
        embedding_dim=64,
        sparsity_level=5,
        commitment_cost=0.25,
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
        **kwargs,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.sparsity_level = sparsity_level
        self.commitment_cost = commitment_cost
        # Retain config fields for logging/compatibility, but OMP + backprop
        # is fixed.
        self.use_online_learning = use_online_learning
        self.epsilon = epsilon
        self.omp_tolerance = tolerance
        self.omp_debug = omp_debug
        self.dict_learning_rate = dict_learning_rate
        self.sparse_solver = sparse_solver.lower()

        if self.sparse_solver != "omp":
            raise ValueError(
                f"Only sparse_solver='omp' is supported, got '{sparse_solver}'"
            )

        if decay not in (None, 0):
            warnings.warn(
                (
                    "DictionaryLearning no longer uses the `decay` parameter; "
                    "it is ignored."
                ),
                stacklevel=2,
            )

        if use_pattern_quantizer:
            warnings.warn(
                (
                    "Pattern quantization is not implemented in "
                    "DictionaryLearning; ignoring."
                ),
                stacklevel=2,
            )

        if not use_backprop_only:
            warnings.warn(
                (
                    "Only backprop-based dictionary updates are supported; "
                    "forcing use_backprop_only=True."
                ),
                stacklevel=2,
            )
        self.use_backprop_only = True

        self.patch_size = self._as_pair(patch_size, "patch_size")
        self.patch_area = self.patch_size[0] * self.patch_size[1]

        if patch_stride is None:
            stride_value = self.patch_size
        else:
            stride_value = patch_stride
        self.patch_stride = self._as_pair(stride_value, "patch_stride")

        self.per_pixel_sparse_coding = per_pixel_sparse_coding
        self.patch_flatten_order = patch_flatten_order.lower()
        if self.patch_flatten_order not in ("channel_first", "spatial_first"):
            raise ValueError(
                (
                    "patch_flatten_order must be 'channel_first' or "
                    "'spatial_first'"
                )
            )

        if self.per_pixel_sparse_coding and self.patch_size != (1, 1):
            warnings.warn(
                (
                    "per_pixel_sparse_coding ignores patch_size/patch_stride; "
                    "coding per pixel."
                ),
                stacklevel=2,
            )

        if self.per_pixel_sparse_coding:
            self.atom_dim = self.embedding_dim
        else:
            self.atom_dim = self.embedding_dim * self.patch_area

        self.dictionary = nn.Parameter(
            torch.randn(self.atom_dim, num_embeddings), requires_grad=True
        )
        self._normalize_dictionary()

        self.register_buffer("is_initialized", torch.tensor(False))
        self._last_diag = {}
        self._last_bottleneck_losses = {}

    @staticmethod
    def _as_pair(value, name):
        if isinstance(value, int):
            return (value, value)
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return (int(value[0]), int(value[1]))
        raise ValueError(f"{name} must be an int or a tuple of two ints")

    def _normalize_dictionary(self):
        with torch.no_grad():
            self.dictionary.copy_(
                F.normalize(self.dictionary, p=2, dim=0, eps=self.epsilon)
            )

    def _maybe_initialize_dictionary(self, patches_flat):
        if not (self.training and not self.is_initialized.item()):
            return
        with torch.no_grad():
            n_patches = patches_flat.shape[0]
            if n_patches == 0:
                return
            # Seed the dictionary with random data patches to stabilize early
            # OMP.
            if n_patches >= self.num_embeddings:
                indices = torch.randperm(
                    n_patches, device=patches_flat.device
                )[: self.num_embeddings]
                init_atoms = patches_flat[indices]
            else:
                indices = torch.randint(
                    0,
                    n_patches,
                    (self.num_embeddings,),
                    device=patches_flat.device,
                )
                init_atoms = patches_flat[indices] + 0.01 * torch.randn(
                    self.num_embeddings,
                    self.atom_dim,
                    device=patches_flat.device,
                    dtype=patches_flat.dtype,
                )
            self.dictionary.copy_(init_atoms.t())
            self._normalize_dictionary()
            self.is_initialized.fill_(True)

    def _patchify(self, x):
        B, C, H, W = x.shape
        patches = F.unfold(
            x, kernel_size=self.patch_size, stride=self.patch_stride
        )

        if self.patch_flatten_order == "channel_first":
            # Each patch is [C, ph, pw] flattened with channel-major ordering.
            patches = patches.transpose(1, 2).reshape(-1, patches.shape[1])
        else:
            ph, pw = self.patch_size
            # Reorder to [ph, pw, C] per patch so spatial neighbors are
            # contiguous.
            patches = patches.view(B, C, ph, pw, -1)
            patches = patches.permute(0, 4, 2, 3, 1).reshape(
                -1, self.atom_dim
            )

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
            patches = patches.permute(0, 4, 2, 3, 1).reshape(
                B, C * ph * pw, L
            )

        x_recon = F.fold(
            patches,
            output_size=(H, W),
            kernel_size=self.patch_size,
            stride=self.patch_stride,
        )

        if self.patch_stride != self.patch_size:
            # Overlapping patches are summed by fold; divide by overlap count.
            ones = torch.ones_like(x_recon)
            count_patches = F.unfold(
                ones, kernel_size=self.patch_size, stride=self.patch_stride
            )
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
        identity = torch.eye(
            gram.shape[0], device=gram.device, dtype=gram.dtype
        )
        return (gram - identity).pow(2).mean()

    def forward(self, z_e):
        B, C, H, W = z_e.shape
        if C != self.embedding_dim:
            raise ValueError(
                f"Expected channel dim {self.embedding_dim} but received {C}"
            )

        if self.per_pixel_sparse_coding:
            # Per-pixel mode: each spatial location is sparse-coded
            # independently.
            patches_flat = (
                z_e.permute(0, 2, 3, 1).contiguous().view(-1, C)
            )
            recon_patches_flat, coeffs = self._sparse_code(patches_flat)
            z_dl = (
                recon_patches_flat.view(B, H, W, C)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
        else:
            # Patch mode: each patch is a single sparse-coded signal.
            patches_flat, spatial_dims = self._patchify(z_e)
            recon_patches_flat, coeffs = self._sparse_code(patches_flat)
            z_dl = self._unpatchify(
                recon_patches_flat, spatial_dims, (B, C, H, W)
            )

        # Dictionary loss updates D; commitment loss keeps encoder outputs
        # stable.
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
        Batched Orthogonal Matching Pursuit adapted from
        amzn/sparse-vqvae utils/pyomp.py.

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
        tol = (
            self.omp_tolerance if self.omp_tolerance is not None else 1e-7
        )

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
        active_idx = torch.ones(B, 0, device=X.device, dtype=torch.long)
        active_mask = torch.zeros_like(h_bar, dtype=torch.bool)
        delta = torch.zeros(B, device=X.device, dtype=X.dtype)

        def _mark_active(mask, to_add):
            running_idx = torch.arange(to_add.shape[0], device=to_add.device)
            mask[running_idx, to_add] = True

        k = 0
        batch_idx = torch.arange(B, device=X.device)
        while k < k_max and eps.max() > tol:
            k += 1

            # Select next atom per batch (highest residual correlation).
            index = (h * (~active_mask).float()).abs().argmax(dim=1)
            _mark_active(active_mask, index)
            expanded_batch_idx = batch_idx.unsqueeze(0).expand(k, B).t()

            if k > 1:
                # Rank-1 Cholesky update for each batch element.
                G_stack = G[
                    active_idx[batch_idx, :],
                    index[expanded_batch_idx[..., :-1]],
                ].view(B, k - 1, 1)
                # Solve L w = G_stack for w (lower triangular)
                try:
                    w = torch.linalg.solve_triangular(L, G_stack, upper=False)
                except AttributeError:
                    w = torch.triangular_solve(
                        G_stack, L, upper=False
                    ).solution
                w = w.view(B, 1, k - 1)
                w_corner = torch.sqrt(
                    torch.clamp(
                        1 - (w**2).sum(dim=2, keepdim=True), min=diag_eps
                    )
                )

                # Build new L = [[L, 0], [w, w_corner]]
                k_zeros = torch.zeros(
                    B, k - 1, 1, device=X.device, dtype=X.dtype
                )
                L = torch.cat(
                    (
                        torch.cat((L, k_zeros), dim=2),
                        torch.cat((w, w_corner), dim=2),
                    ),
                    dim=1,
                )

            active_idx = torch.cat([active_idx, index.unsqueeze(1)], dim=1)

            # Solve for coefficients on active set via Cholesky solve
            h_stack = h_bar[
                expanded_batch_idx, active_idx[batch_idx, :]
            ].view(B, k, 1)
            try:
                x_stack = torch.cholesky_solve(h_stack, L)
            except AttributeError:
                x_stack = torch.linalg.cholesky_solve(h_stack, L)
            x[batch_idx.unsqueeze(1), active_idx[batch_idx]] = (
                x_stack.squeeze(-1)
            )

            beta = (
                x[batch_idx.unsqueeze(1), active_idx[batch_idx]]
                .unsqueeze(1)
                .bmm(G[active_idx[batch_idx], :])
                .squeeze(1)
            )
            h = h_bar - beta

            new_delta = (x * beta).sum(dim=1)
            eps = eps + delta - new_delta
            delta = new_delta

            # NaN/inf guard: break early if instability detected
            if (
                not torch.isfinite(x).all()
                or not torch.isfinite(L).all()
                or not torch.isfinite(eps).all()
            ):
                break

            if self.omp_debug and k % 1 == 0:
                print(
                    "OMP step "
                    f"{k}, residual max={eps.max().item():.4f}, "
                    "below tol="
                    f"{(eps < tol).float().mean().item():.4f}"
                )

        return x.t()
