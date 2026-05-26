import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


from .bottleneck_utils import (
    SparseCodes,
    _normalize_dictionary,
)


class DictionaryLearning(nn.Module):
    """Dictionary-learning bottleneck with OMP sparse coding."""

    def __init__(
        self,
        num_embeddings=512,
        embedding_dim=64,
        sparsity_level=5,
        commitment_cost=0.25,
        dict_learning_rate=None,
        patch_based=False,
        patch_size=1,
        patch_stride=None,
        patch_reconstruction="tile",
        coef_max=None,
        omp_residual_tolerance=None,
        dead_atom_revival_steps=None,
        dictionary_through_decoder=False,
        data_init_from_first_batch=False,
        epsilon=1e-10,
    ):
        super().__init__()

        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.sparsity_level = int(sparsity_level)
        self.commitment_cost = float(commitment_cost)
        self.epsilon = float(epsilon)
        self.dict_learning_rate = dict_learning_rate
        # When True, decoder gradients flow into the dictionary atoms via the
        # OMP-linear-combo path while retaining an identity gradient to the
        # encoder. Useful to break bottleneck collapse since atoms then get
        # shaped by reconstruction quality, not just MSE-to-z_e.
        self.dictionary_through_decoder = bool(dictionary_through_decoder)
        self.coef_max = None if coef_max is None else float(coef_max)
        if self.coef_max is not None:
            if not math.isfinite(self.coef_max) or self.coef_max <= 0.0:
                raise ValueError(f"coef_max must be finite and > 0, got {self.coef_max}")
        # Relative residual tolerance for OMP early termination. If set, OMP stops
        # picking atoms per sample once ||residual||^2 / ||signal||^2 <= tolerance.
        # Unused slots in support/coeffs stay at 0. None preserves fixed-k behavior.
        self.omp_residual_tolerance = (
            None if omp_residual_tolerance is None else float(omp_residual_tolerance)
        )
        if self.omp_residual_tolerance is not None:
            if not math.isfinite(self.omp_residual_tolerance) or self.omp_residual_tolerance < 0.0:
                raise ValueError(
                    "omp_residual_tolerance must be finite and >= 0, got "
                    f"{self.omp_residual_tolerance}"
                )
        # Dead-atom revival: every step, atoms unused for >= this many consecutive forwards
        # get resampled from current-batch latents (RQ-VAE's restart_unused_codes idea).
        # None disables. Recommended: ~100 for default training cadence.
        self.dead_atom_revival_steps = (
            None if dead_atom_revival_steps is None else int(dead_atom_revival_steps)
        )
        if self.dead_atom_revival_steps is not None and self.dead_atom_revival_steps < 1:
            raise ValueError(
                "dead_atom_revival_steps must be >= 1 or None, got "
                f"{self.dead_atom_revival_steps}"
            )
        self.patch_based = bool(patch_based)
        effective_patch_size = int(patch_size) if self.patch_based else 1
        self.patch_size = effective_patch_size
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {self.patch_size}")
        if patch_stride is None:
            patch_stride = self.patch_size
        effective_patch_stride = int(patch_stride) if self.patch_based else 1
        self.patch_stride = effective_patch_stride
        if self.patch_stride <= 0:
            raise ValueError(f"patch_stride must be positive, got {self.patch_stride}")
        if self.patch_stride > self.patch_size:
            raise ValueError(
                f"patch_stride ({self.patch_stride}) must be <= patch_size ({self.patch_size})"
            )
        if patch_reconstruction != "tile":
            raise ValueError(
                f"Only 'tile' patch_reconstruction is supported, got {patch_reconstruction!r}"
            )
        self.patch_reconstruction = "tile"
        self.patch_dim = self.embedding_dim * self.patch_size * self.patch_size

        self.dictionary = nn.Parameter(
            torch.randn(self.patch_dim, self.num_embeddings) * 0.02
        )
        self.register_buffer(
            "_steps_since_use",
            torch.zeros(self.num_embeddings, dtype=torch.long),
        )
        # When True, replace the random init with normalized samples from the first
        # training batch's encoder output. Sidesteps OMP's chicken-and-egg cold start
        # (random atoms produce random OMP picks → no meaningful gradient signal).
        self.data_init_from_first_batch = bool(data_init_from_first_batch)
        self.register_buffer(
            "_data_initialized",
            torch.zeros((), dtype=torch.bool),
        )
        self.normalize_dictionary_()
        self._last_diag = {}
        self._last_latent_loss = None
        self._last_dl_latent_loss = None
        self._last_e_latent_loss = None
        self._last_bottleneck_loss = torch.zeros(())

    def normalize_dictionary_(self):
        with torch.no_grad():
            self.dictionary.copy_(
                _normalize_dictionary(self.dictionary.detach(), eps=self.epsilon)
            )

    def is_dictionary_parameter(self, name: str) -> bool:
        return str(name) == "dictionary"

    @property
    def dictionary_dtype(self) -> torch.dtype:
        return self.dictionary.dtype

    def dictionary_for_visualization(self, max_vectors: int) -> torch.Tensor:
        atoms = self.dictionary.detach().t().cpu()
        max_vectors = max(1, int(max_vectors))
        if int(atoms.size(0)) <= max_vectors:
            return atoms
        indices = torch.linspace(
            0,
            int(atoms.size(0)) - 1,
            steps=max_vectors,
        ).round().to(torch.long)
        return atoms.index_select(0, indices)

    def project_dictionary_gradient_(self):
        if self.dictionary.grad is None:
            return
        with torch.no_grad():
            atoms = _normalize_dictionary(self.dictionary.detach(), eps=self.epsilon)
            grad = torch.nan_to_num(self.dictionary.grad)
            radial = (atoms * grad).sum(dim=0, keepdim=True)
            grad = grad - atoms * radial
            self.dictionary.grad.copy_(torch.nan_to_num(grad))

    def _validate_omp_inputs(self, X, D):
        if X.ndim != 2 or D.ndim != 2:
            raise ValueError(
                f"Expected 2D tensors, got X={tuple(X.shape)} D={tuple(D.shape)}"
            )
        if int(X.size(0)) != int(D.size(0)):
            raise ValueError(
                f"Signal dim ({int(X.size(0))}) must match dictionary dim ({int(D.size(0))})"
            )
        if self.sparsity_level > int(D.size(1)):
            raise ValueError(
                f"sparsity_level ({int(self.sparsity_level)}) must be <= num_atoms ({int(D.size(1))})"
            )

    def batch_omp_with_support(self, X, D):
        """Batch-OMP (Rubinstein-Zibulevsky-Elad 2008) with incremental Cholesky.

        Port of sparse-vqvae's ``Batch_OMP``
        (https://github.com/amzn/sparse-vqvae/blob/main/utils/pyomp.py).

        Precomputes ``G = D^T D`` and ``h_bar = D^T X`` once, then iteratively
        picks atoms and updates the Cholesky factor of ``G[I, I]`` incrementally.
        With ``omp_residual_tolerance=None``, always runs ``sparsity_level``
        selections. Otherwise each sample stops once its squared residual norm
        falls below ``omp_residual_tolerance * ||x||^2`` while other samples in
        the batch can keep selecting atoms.

        Args:
            X: ``[patch_dim, batch]`` signals.
            D: ``[patch_dim, num_atoms]`` dictionary (unit-norm columns).

        Returns:
            ``(support, coeffs_ordered)``, both ``[batch, sparsity_level]``.
        """
        self._validate_omp_inputs(X, D)
        X = torch.nan_to_num(X)
        D = torch.nan_to_num(D)

        _, batch_size = X.shape
        num_atoms = int(D.size(1))
        device = X.device
        dtype = X.dtype
        sparsity = int(self.sparsity_level)
        fixed_sparsity = self.omp_residual_tolerance is None
        tolerance = 0.0 if fixed_sparsity else float(self.omp_residual_tolerance)

        # Precompute Gram and initial correlations.
        G = D.t() @ D  # [num_atoms, num_atoms]
        h_bar = (D.t() @ X).t().contiguous()  # [batch_size, num_atoms]
        h = h_bar.clone()

        # eps tracks the per-sample residual squared L2 norm so the eps/delta
        # updates stay in consistent units (the upstream sparse-vqvae mixes
        # ``||x||`` and ``||recon||^2`` which makes eps go negative for any
        # signal with ``||x||>1``).
        x_norm_sq = (X * X).sum(dim=0)  # [batch_size]
        eps = x_norm_sq.clone()
        tolerance_sq = None if fixed_sparsity else x_norm_sq * tolerance

        # Outputs.
        support = torch.zeros(batch_size, sparsity, device=device, dtype=torch.long)
        coeffs_ordered = torch.zeros(batch_size, sparsity, device=device, dtype=dtype)
        sparse_code = torch.zeros(batch_size, num_atoms, device=device, dtype=dtype)
        I_logic = torch.zeros(batch_size, num_atoms, device=device, dtype=torch.bool)
        L = torch.ones(batch_size, 1, 1, device=device, dtype=dtype)
        batch_idx = torch.arange(batch_size, device=device)

        k = 0
        while k < sparsity:
            if fixed_sparsity:
                active = torch.ones(batch_size, device=device, dtype=torch.bool)
            else:
                active = eps > tolerance_sq
                if not bool(active.any().item()):
                    break

            # argmax |h| over not-yet-picked atoms. Multiplicative masking can
            # reselect an already-picked atom when all remaining correlations
            # are zero, so use -inf-style masking instead.
            masked_corr = h.abs().masked_fill(I_logic, torch.finfo(dtype).min)
            index = masked_corr.argmax(dim=1)
            index = torch.where(active, index, torch.zeros_like(index))
            I_logic[batch_idx[active], index[active]] = True

            if k > 0:
                # G_cross[b, i] = G[support[b, i], index[b]] for i = 0..k-1.
                I_prev = support[:, :k]
                idx_b = index.unsqueeze(1).expand(batch_size, k)
                G_cross = G[I_prev, idx_b].view(batch_size, k, 1)
                w = torch.linalg.solve_triangular(L, G_cross, upper=False)
                w_corner = torch.sqrt(
                    (1.0 - (w * w).sum(dim=(1, 2), keepdim=False)).clamp_min(self.epsilon)
                )
                # L_new = [[L, 0], [w^T, w_corner]]
                k_zeros = w.new_zeros(batch_size, k, 1)
                L = torch.cat(
                    (
                        torch.cat((L, k_zeros), dim=2),
                        torch.cat((w.transpose(1, 2), w_corner.view(batch_size, 1, 1)), dim=2),
                    ),
                    dim=1,
                )

            support[active, k] = index[active]

            # Solve L L^T x_I = h_bar[I].
            I_curr = support[:, : k + 1]
            x_stack = torch.zeros(batch_size, k + 1, device=device, dtype=dtype)
            h_stack = h_bar[active].gather(1, I_curr[active]).unsqueeze(-1)
            x_active = torch.cholesky_solve(h_stack, L[active]).squeeze(-1)
            if self.coef_max is not None:
                x_active = x_active.clamp(-self.coef_max, self.coef_max)
            x_stack[active] = x_active

            next_sparse_code = torch.zeros_like(sparse_code)
            next_sparse_code.scatter_(1, I_curr, x_stack)
            sparse_code[active] = next_sparse_code[active]
            coeffs_ordered[active, : k + 1] = x_active

            # beta = sparse_code @ G; h = h_bar - beta.
            beta = sparse_code @ G
            h = h_bar - beta

            # Track the actual residual norm. The simpler
            # ``||X||^2 - ||D alpha||^2`` expression is only valid for the
            # unconstrained orthogonal projection; coefficient clipping breaks
            # that identity.
            recon_norm_sq = (sparse_code * beta).sum(dim=1)
            cross = (sparse_code * h_bar).sum(dim=1)
            eps = (x_norm_sq - 2.0 * cross + recon_norm_sq).clamp_min(0.0)

            k += 1

        return support, coeffs_ordered

    @torch.no_grad()
    def _init_dictionary_from_signals_(self, signals: torch.Tensor):
        """Replace random init with normalized samples from current-batch latents.

        ``signals`` is ``[patch_dim, N]``. Samples N random columns (with replacement
        if N < num_embeddings, broadcast from rank 0 for DDP determinism), L2-normalizes,
        and copies into the dictionary parameter. Run exactly once, gated by
        ``_data_initialized``.
        """
        N = int(signals.shape[1])
        if N <= 0:
            return
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                idx = torch.randint(0, N, (self.num_embeddings,), device=signals.device)
                sampled = signals[:, idx]
            else:
                sampled = torch.empty(
                    self.patch_dim, self.num_embeddings,
                    device=signals.device, dtype=signals.dtype,
                )
            torch.distributed.broadcast(sampled, src=0)
        else:
            idx = torch.randint(0, N, (self.num_embeddings,), device=signals.device)
            sampled = signals[:, idx]
        sampled = F.normalize(sampled, p=2, dim=0, eps=self.epsilon)
        self.dictionary.data.copy_(sampled.to(dtype=self.dictionary.dtype))
        self._data_initialized.fill_(True)

    @torch.no_grad()
    def _track_atom_usage_and_revive_(self, support: torch.Tensor, signals: torch.Tensor):
        """Track per-atom usage; resample atoms unused for >= revival_steps consecutive forwards.

        Mirrors RQ-VAE's restart_unused_codes: a dead atom is replaced with a normalized
        random sample from the current-batch latents (``signals``: ``[patch_dim, N]``).
        """
        flat = support.detach().to(torch.long).reshape(-1).clamp(0, self.num_embeddings - 1)
        used_mask = torch.zeros(
            self.num_embeddings, dtype=torch.bool, device=self._steps_since_use.device
        )
        if flat.numel() > 0:
            used_mask.scatter_(0, flat.to(used_mask.device), True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            used_int = used_mask.to(torch.long)
            torch.distributed.all_reduce(used_int, op=torch.distributed.ReduceOp.SUM)
            used_mask = used_int > 0
        self._steps_since_use += 1
        self._steps_since_use[used_mask] = 0
        dead_mask = self._steps_since_use >= int(self.dead_atom_revival_steps)
        num_dead = int(dead_mask.sum().item())
        if num_dead <= 0:
            return
        N = int(signals.shape[1])
        if N <= 0:
            return
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                idx = torch.randint(0, N, (num_dead,), device=signals.device)
                sampled = signals[:, idx]
            else:
                sampled = torch.empty(self.patch_dim, num_dead, device=signals.device, dtype=signals.dtype)
            torch.distributed.broadcast(sampled, src=0)
        else:
            idx = torch.randint(0, N, (num_dead,), device=signals.device)
            sampled = signals[:, idx]
        sampled = F.normalize(sampled, p=2, dim=0, eps=self.epsilon)
        self.dictionary.data[:, dead_mask] = sampled.to(dtype=self.dictionary.dtype)
        self._steps_since_use[dead_mask] = 0

    def _is_patch_based(self):
        return self.patch_based

    def _patch_grid_geometry(self, height: int, width: int):
        height = int(height)
        width = int(width)
        center_pad = max(self.patch_size - self.patch_stride, 0) // 2
        nph = math.ceil(height / self.patch_stride)
        npw = math.ceil(width / self.patch_stride)
        height_padded = (nph - 1) * self.patch_stride + self.patch_size
        width_padded = (npw - 1) * self.patch_stride + self.patch_size
        pad_top = center_pad
        pad_left = center_pad
        pad_bottom = height_padded - height - center_pad
        pad_right = width_padded - width - center_pad
        return nph, npw, height_padded, width_padded, pad_top, pad_left, pad_bottom, pad_right

    def _extract_patches(self, z_e):
        _, _, height, width = z_e.shape
        (
            nph,
            npw,
            _height_padded,
            _width_padded,
            pad_top,
            pad_left,
            pad_bottom,
            pad_right,
        ) = self._patch_grid_geometry(height, width)
        pad_mode = "reflect"
        if pad_left >= width or pad_right >= width or pad_top >= height or pad_bottom >= height:
            pad_mode = "replicate"
        padded = F.pad(
            z_e,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode=pad_mode,
        )
        patches = F.unfold(
            padded,
            kernel_size=self.patch_size,
            stride=self.patch_stride,
        )
        return patches, nph, npw, height, width

    def _sparse_atom_sum(self, support, values, *, clamp_values=False):
        depth = int(support.shape[-1])
        dictionary = _normalize_dictionary(self.dictionary, eps=self.epsilon).t()
        support_flat = support.to(torch.long).clamp(0, self.num_embeddings - 1).reshape(-1, depth)
        values_dense = values.to(dictionary.dtype)
        if clamp_values and self.coef_max is not None:
            values_dense = values_dense.clamp(-self.coef_max, self.coef_max)
        values_flat = values_dense.reshape(-1, depth)
        atoms = dictionary[support_flat]
        return (atoms * values_flat.unsqueeze(-1)).sum(dim=1)

    def _reconstruct_patches_tile(self, support, values, height, width):
        """Fold-and-average patches; equivalent to tiling when patches do not overlap."""
        B, nph, npw, _depth = support.shape
        C = self.embedding_dim
        recon = self._sparse_atom_sum(support, values, clamp_values=True)

        (
            expected_nph,
            expected_npw,
            height_padded,
            width_padded,
            pad_top,
            pad_left,
            _pad_bottom,
            _pad_right,
        ) = self._patch_grid_geometry(height, width)
        if nph != expected_nph or npw != expected_npw:
            raise ValueError(
                "Patch token grid does not match latent shape: "
                f"got {(nph, npw)}, expected {(expected_nph, expected_npw)} for latent {(height, width)}"
            )

        recon = recon.view(B, nph * npw, self.patch_dim).transpose(1, 2).contiguous()
        recon = F.fold(
            recon,
            output_size=(height_padded, width_padded),
            kernel_size=self.patch_size,
            stride=self.patch_stride,
        )
        weights = recon.new_ones(B, C * self.patch_size * self.patch_size, nph * npw)
        weights = F.fold(
            weights,
            output_size=(height_padded, width_padded),
            kernel_size=self.patch_size,
            stride=self.patch_stride,
        ).clamp_min(self.epsilon)
        recon = recon / weights
        return recon[:, :, pad_top: pad_top + height, pad_left: pad_left + width]

    def _reconstruct_sparse(self, support, values, height, width):
        if self._is_patch_based():
            if self.patch_reconstruction == "tile":
                return self._reconstruct_patches_tile(support, values, height, width)
            raise ValueError(f"Unsupported or removed patch_reconstruction mode: {self.patch_reconstruction}")

        recon = self._sparse_atom_sum(support, values)
        batch_size = support.shape[0]
        return recon.view(batch_size, height, width, self.embedding_dim).permute(0, 3, 1, 2).contiguous()

    def clamp_sparse_coeffs(self, coeffs: torch.Tensor) -> torch.Tensor:
        coeffs = torch.nan_to_num(coeffs)
        if self.coef_max is None:
            return coeffs
        return coeffs.clamp(-self.coef_max, self.coef_max)

    def project_sparse_coeffs(self, support: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        return self.clamp_sparse_coeffs(coeffs)

    def _coeff_bin_values(
        self,
        *,
        coeff_vocab_size: int,
        coeff_bin_values: Optional[Sequence[float] | torch.Tensor] = None,
        coeff_max: Optional[float] = None,
        coeff_quantization: str = "uniform",
        coeff_mu: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if coeff_bin_values is not None:
            values = torch.as_tensor(coeff_bin_values, device=device, dtype=dtype).reshape(-1)
            if values.numel() != int(coeff_vocab_size):
                raise ValueError(
                    "coeff_bin_values length must match coeff_vocab_size, "
                    f"got {values.numel()} vs {int(coeff_vocab_size)}"
                )
            return values

        coeff_vocab_size = int(coeff_vocab_size)
        if coeff_vocab_size <= 0:
            raise ValueError(f"coeff_vocab_size must be positive, got {coeff_vocab_size}")
        if coeff_max is None:
            raise ValueError("coeff_max is required when coeff_bin_values is not provided")

        coeff_max = float(coeff_max)
        coeff_quantization = str(coeff_quantization).strip().lower()
        if coeff_quantization == "uniform":
            return torch.linspace(-coeff_max, coeff_max, steps=coeff_vocab_size, device=device, dtype=dtype)
        if coeff_quantization != "mu_law":
            raise ValueError(
                "coeff_quantization must be 'uniform' or 'mu_law', got "
                f"{coeff_quantization!r}"
            )

        coeff_mu = float(coeff_mu)
        if coeff_mu <= 0.0:
            raise ValueError(f"coeff_mu must be > 0 for mu-law quantization, got {coeff_mu}")
        if coeff_vocab_size == 1:
            return torch.zeros(1, device=device, dtype=dtype)
        z = torch.linspace(-1.0, 1.0, steps=coeff_vocab_size, device=device, dtype=dtype)
        decoded = torch.sign(z) * (torch.expm1(z.abs() * math.log1p(coeff_mu)) / coeff_mu)
        return decoded * coeff_max

    def _quantize_coefficients(
        self,
        coeffs: torch.Tensor,
        *,
        coeff_vocab_size: int,
        coeff_max: float,
        coeff_quantization: str = "uniform",
        coeff_mu: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        coeff_vocab_size = int(coeff_vocab_size)
        if coeff_vocab_size <= 0:
            raise ValueError(f"coeff_vocab_size must be positive, got {coeff_vocab_size}")
        coeff_max = float(coeff_max)
        if not math.isfinite(coeff_max) or coeff_max <= 0.0:
            raise ValueError(f"coeff_max must be finite and > 0, got {coeff_max}")

        coeff_quantization = str(coeff_quantization).strip().lower()
        coeffs = coeffs.to(torch.float32)
        clamped = coeffs.clamp(-coeff_max, coeff_max)

        if coeff_vocab_size == 1:
            bin_idx = torch.zeros_like(clamped, dtype=torch.long)
        elif coeff_quantization == "uniform":
            scaled = (clamped + coeff_max) / (2.0 * coeff_max)
            bin_float = scaled * float(coeff_vocab_size - 1)
            bin_idx = torch.round(bin_float).to(torch.long).clamp_(0, coeff_vocab_size - 1)
        elif coeff_quantization == "mu_law":
            coeff_mu = float(coeff_mu)
            if coeff_mu <= 0.0:
                raise ValueError(f"coeff_mu must be > 0 for mu-law quantization, got {coeff_mu}")
            normalized = clamped / coeff_max
            encoded = torch.sign(normalized) * torch.log1p(normalized.abs() * coeff_mu) / math.log1p(coeff_mu)
            bin_float = (encoded + 1.0) * ((coeff_vocab_size - 1) / 2.0)
            bin_idx = torch.round(bin_float).to(torch.long).clamp_(0, coeff_vocab_size - 1)
        else:
            raise ValueError(
                "coeff_quantization must be 'uniform' or 'mu_law', got "
                f"{coeff_quantization!r}"
            )

        bin_values = self._coeff_bin_values(
            coeff_vocab_size=coeff_vocab_size,
            coeff_max=coeff_max,
            coeff_quantization=coeff_quantization,
            coeff_mu=coeff_mu,
            device=coeffs.device,
            dtype=torch.float32,
        )
        coeff_q = bin_values[bin_idx]
        return bin_idx, coeff_q

    def sparse_codes_to_tokens(
        self,
        sparse_codes: SparseCodes,
        *,
        coeff_vocab_size: int,
        coeff_max: float,
        coeff_quantization: str = "uniform",
        coeff_mu: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pack sparse codes into interleaved [atom_id, coeff_bin] tokens."""
        support = sparse_codes.support.to(torch.long).clamp(0, self.num_embeddings - 1)
        bin_idx, coeff_q = self._quantize_coefficients(
            sparse_codes.values,
            coeff_vocab_size=coeff_vocab_size,
            coeff_max=coeff_max,
            coeff_quantization=coeff_quantization,
            coeff_mu=coeff_mu,
        )
        tokens = torch.empty(
            *support.shape[:-1],
            int(self.sparsity_level) * 2,
            device=support.device,
            dtype=torch.long,
        )
        tokens[..., 0::2] = support
        tokens[..., 1::2] = bin_idx + int(self.num_embeddings)
        return tokens, coeff_q

    def tokens_to_latent(
        self,
        tokens: torch.Tensor,
        *,
        latent_hw: Optional[Tuple[int, int]] = None,
        atom_vocab_size: Optional[int] = None,
        coeff_vocab_size: Optional[int] = None,
        coeff_bin_values: Optional[Sequence[float] | torch.Tensor] = None,
        coeff_max: Optional[float] = None,
        coeff_quantization: str = "uniform",
        coeff_mu: float = 0.0,
    ) -> torch.Tensor:
        """Decode quantized sparse tokens back to a latent map."""
        if tokens.dim() != 4:
            raise ValueError(f"Expected [B,H,W,D], got {tuple(tokens.shape)}")

        batch_size, token_h, token_w, token_depth = tokens.shape
        expected_token_depth = int(self.sparsity_level) * 2
        if int(token_depth) != expected_token_depth:
            raise ValueError(
                f"Expected token depth {expected_token_depth}, got {int(token_depth)}"
            )

        atom_vocab_size = int(atom_vocab_size or self.num_embeddings)
        if atom_vocab_size <= 0:
            raise ValueError(f"atom_vocab_size must be positive, got {atom_vocab_size}")

        if coeff_vocab_size is None:
            if coeff_bin_values is None:
                raise ValueError("coeff_vocab_size or coeff_bin_values is required for token decode")
            coeff_vocab_size = int(torch.as_tensor(coeff_bin_values).numel())
        coeff_vocab_size = int(coeff_vocab_size)
        if coeff_vocab_size <= 0:
            raise ValueError(f"coeff_vocab_size must be positive, got {coeff_vocab_size}")

        support = tokens[..., 0::2].to(torch.long)
        coeff_bins = tokens[..., 1::2].to(torch.long) - atom_vocab_size
        coeff_bins = coeff_bins.clamp(0, coeff_vocab_size - 1)
        coeff_values = self._coeff_bin_values(
            coeff_vocab_size=coeff_vocab_size,
            coeff_bin_values=coeff_bin_values,
            coeff_max=coeff_max,
            coeff_quantization=coeff_quantization,
            coeff_mu=coeff_mu,
            device=tokens.device,
            dtype=self.dictionary.dtype,
        )
        values = coeff_values[coeff_bins]

        if self._is_patch_based():
            if latent_hw is None:
                raise ValueError("latent_hw is required for patch-based token decoding")
            height, width = int(latent_hw[0]), int(latent_hw[1])
        else:
            height, width = int(token_h), int(token_w)

        z_q = self._reconstruct_sparse(support, values, height, width)
        if z_q.shape != (batch_size, self.embedding_dim, height, width):
            raise RuntimeError(
                "Decoded latent shape mismatch: "
                f"expected {(batch_size, self.embedding_dim, height, width)}, got {tuple(z_q.shape)}"
            )
        return z_q

    def forward(self, z_e):
        """Sparse-code ``z_e`` against the learned dictionary.

        OMP support and coefficient values are computed under ``no_grad``. The
        bottleneck loss is ``dl_latent_loss + commitment_cost * e_latent_loss``,
        and the straight-through estimator on ``z_dl`` lets decoder gradients
        reach the encoder while bypassing the dictionary.
        """
        if z_e.dim() != 4:
            raise ValueError(f"Expected input [B, C, H, W], got {tuple(z_e.shape)}")
        B, C, H, W = z_e.shape
        if C != self.embedding_dim:
            raise ValueError(
                f"Expected channel dim {self.embedding_dim} but received {C}"
            )

        if self._is_patch_based():
            patches, grid_h, grid_w, height, width = self._extract_patches(z_e)
            signals = patches.permute(0, 2, 1).contiguous().view(-1, self.patch_dim).t()
        else:
            grid_h, grid_w, height, width = H, W, H, W
            signals = z_e.permute(0, 2, 3, 1).contiguous().view(-1, C).t()

        if (
            self.training
            and self.data_init_from_first_batch
            and not bool(self._data_initialized.item())
        ):
            self._init_dictionary_from_signals_(signals)

        dictionary = _normalize_dictionary(self.dictionary, eps=self.epsilon)
        with torch.no_grad():
            support, values = self.batch_omp_with_support(signals, dictionary)
        support = support.view(B, grid_h, grid_w, self.sparsity_level)
        values = self.clamp_sparse_coeffs(
            values.view(B, grid_h, grid_w, self.sparsity_level)
        )
        if self.training and self.dead_atom_revival_steps is not None:
            self._track_atom_usage_and_revive_(support, signals)

        z_dl = self._reconstruct_sparse(support, values, height, width)

        dl_latent_loss = F.mse_loss(z_dl, z_e.detach())
        e_latent_loss = F.mse_loss(z_dl.detach(), z_e)
        latent_loss = dl_latent_loss + self.commitment_cost * e_latent_loss
        loss = latent_loss

        self._last_latent_loss = latent_loss.detach()
        self._last_dl_latent_loss = dl_latent_loss.detach()
        self._last_e_latent_loss = e_latent_loss.detach()
        self._last_bottleneck_loss = loss.detach()
        self._update_diagnostics(values)

        if self.dictionary_through_decoder:
            z_dl = z_dl + (z_e - z_e.detach())
        else:
            z_dl = z_e + (z_dl - z_e).detach()
        sparse_codes = SparseCodes(
            support=support,
            values=values,
            num_embeddings=self.num_embeddings,
        )
        return z_dl, loss, sparse_codes

    def _update_diagnostics(self, values):
        num_sites = max(int(values.shape[0] * values.shape[1] * values.shape[2]), 1)
        dict_norms = self.dictionary.norm(dim=0).detach()
        coeff_abs = values.abs()
        coeff_abs_max = coeff_abs.max().detach()
        if self.coef_max is None:
            coeff_clip_frac = coeff_abs.new_zeros(())
        else:
            clip_eps = max(float(self.epsilon), float(self.coef_max) * 1e-6)
            coeff_clip_frac = (
                (coeff_abs >= (float(self.coef_max) - clip_eps))
                .to(values.dtype)
                .mean()
                .detach()
            )
        self._last_diag = {
            "dict_norm_max": dict_norms.max(),
            "dict_norm_mean": dict_norms.mean(),
            "dict_norm_min": dict_norms.min(),
            "coeff_abs_mean": (
                coeff_abs.sum() / float(self.num_embeddings * num_sites)
            ).detach(),
            "coeff_abs_max": coeff_abs_max,
            "coeff_clip_frac": coeff_clip_frac,
        }
