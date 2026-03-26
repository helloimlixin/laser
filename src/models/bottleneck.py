import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SparseCodes:
    support: torch.Tensor
    values: torch.Tensor
    num_embeddings: int


def _normalize_dictionary(
    dictionary: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return F.normalize(torch.nan_to_num(dictionary), p=2, dim=0, eps=eps)


def _dictionary_abs_offdiag_cosines(
    dictionary: torch.Tensor,
    eps: float,
    detach: bool,
):
    atoms = dictionary.detach() if detach else dictionary
    atoms = _normalize_dictionary(atoms, eps=eps)
    gram = atoms.t() @ atoms
    gram = gram - torch.diag_embed(torch.diagonal(gram))
    return gram.abs(), int(gram.size(0) * max(gram.size(0) - 1, 0))


class VectorQuantizer(nn.Module):
    """VQ baseline aligned with lucidrains vector-quantize-pytorch (no EMA)."""

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

        # Commitment loss only (lucidrains-style); codebook learns via gradients.
        loss = self.commitment_cost * F.mse_loss(quantized, z_e)

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
    """VQ baseline aligned with lucidrains vector-quantize-pytorch (EMA)."""

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

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        expected = {
            prefix + "ema_cluster_size": self.ema_cluster_size.detach().clone(),
            prefix + "_ema_w": self._ema_w.detach().clone(),
        }
        for key, value in expected.items():
            loaded = state_dict.get(key)
            if loaded is None or tuple(loaded.shape) != tuple(value.shape):
                state_dict[key] = value
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
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

        loss = self.commitment_cost * F.mse_loss(quantized.detach(), z_e)

        quantized = z_e + (quantized - z_e).detach()

        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        )

        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return quantized, loss, perplexity, encodings


class DictionaryLearning(nn.Module):
    """Dictionary-learning bottleneck with optional latent patch sparse coding."""

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
        patch_reconstruction="hann",
        epsilon=1e-10,
        **legacy_kwargs,
    ):
        super().__init__()

        legacy_kwargs.pop("sparse_solver", None)
        legacy_kwargs.pop("tolerance", None)
        legacy_kwargs.pop("omp_debug", None)
        legacy_kwargs.pop("use_online_learning", None)
        legacy_kwargs.pop("fast_omp", None)
        legacy_kwargs.pop("omp_diag_eps", None)
        legacy_kwargs.pop("omp_cholesky_eps", None)
        legacy_kwargs.pop("sparse_coding_scheme", None)
        legacy_kwargs.pop("lista_steps", None)
        legacy_kwargs.pop("lista_step_size_init", None)
        legacy_kwargs.pop("lista_threshold_init", None)
        legacy_kwargs.pop("lista_layers", None)
        legacy_kwargs.pop("lista_tied_weights", None)
        legacy_kwargs.pop("lista_initial_threshold", None)
        legacy_kwargs.pop("dictionary_update_mode", None)
        legacy_kwargs.pop("dict_ema_decay", None)
        legacy_kwargs.pop("dict_ema_eps", None)
        if legacy_kwargs:
            unknown = ", ".join(sorted(legacy_kwargs))
            raise TypeError(f"Unsupported DictionaryLearning arguments: {unknown}")

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.sparsity_level = sparsity_level
        self.commitment_cost = commitment_cost
        self.epsilon = epsilon
        self.dict_learning_rate = dict_learning_rate
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
        if patch_reconstruction not in {"center_crop", "hann"}:
            raise ValueError(
                "patch_reconstruction must be 'center_crop' or 'hann', got "
                f"{patch_reconstruction!r}"
            )
        self.patch_reconstruction = str(patch_reconstruction)
        self.patch_dim = self.embedding_dim * self.patch_size * self.patch_size

        self.dictionary = nn.Parameter(
            torch.randn(self.patch_dim, self.num_embeddings) * 0.02
        )
        if self.patch_size > 1 or self.patch_stride > 1:
            hann_1d = torch.hann_window(self.patch_size, periodic=False)
            window_2d = hann_1d.unsqueeze(1) * hann_1d.unsqueeze(0)
            window_flat = window_2d.flatten().unsqueeze(0).expand(
                self.embedding_dim, -1
            ).reshape(-1)
        else:
            window_flat = torch.ones(self.patch_dim)
        self.register_buffer("_hann_win", window_flat)
        self.normalize_dictionary_()
        self._last_diag = {}
        self._last_dl_latent_loss = None
        self._last_e_latent_loss = None

    def normalize_dictionary_(self):
        with torch.no_grad():
            self.dictionary.copy_(
                _normalize_dictionary(self.dictionary.detach(), eps=self.epsilon)
            )

    def project_dictionary_gradient_(self):
        if self.dictionary.grad is None:
            return
        with torch.no_grad():
            atoms = _normalize_dictionary(self.dictionary.detach(), eps=self.epsilon)
            grad = torch.nan_to_num(self.dictionary.grad)
            radial = (atoms * grad).sum(dim=0, keepdim=True)
            self.dictionary.grad.copy_(torch.nan_to_num(grad - atoms * radial))

    def coherence_penalty(self, margin=0.0):
        abs_offdiag, pair_count = _dictionary_abs_offdiag_cosines(
            self.dictionary, eps=self.epsilon, detach=False
        )
        if pair_count <= 0:
            return torch.zeros(
                (), device=self.dictionary.device, dtype=self.dictionary.dtype
            )
        excess = F.relu(abs_offdiag - float(max(0.0, margin)))
        return excess.square().sum() / float(pair_count)

    def coherence_stats(self):
        abs_offdiag, pair_count = _dictionary_abs_offdiag_cosines(
            self.dictionary, eps=self.epsilon, detach=True
        )
        if pair_count <= 0:
            zero = torch.zeros(
                (), device=self.dictionary.device, dtype=self.dictionary.dtype
            )
            return zero, zero, zero
        mean_abs = abs_offdiag.sum() / float(pair_count)
        rms_abs = torch.sqrt(abs_offdiag.square().sum() / float(pair_count))
        return abs_offdiag.max(), mean_abs, rms_abs

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

    def batch_topk_with_support(self, X, D):
        """Select top-k atoms once, then solve coefficients on that fixed support."""
        self._validate_omp_inputs(X, D)
        X = torch.nan_to_num(X)
        D = torch.nan_to_num(D)

        correlations, gram = self._correlations_and_gram(X, D)
        support = self._scores_to_topk_support(correlations.abs())
        values = self._solve_support_coefficients(correlations, gram, support, D)
        return support, values

    def _correlations_and_gram(self, X, D):
        dictionary_t = D.t()
        return dictionary_t.mm(X).t(), dictionary_t.mm(D)

    def _scores_to_topk_support(self, scores):
        return scores.topk(
            int(self.sparsity_level), dim=1, largest=True, sorted=True
        ).indices

    def _solve_support_coefficients(self, correlations, gram, support, dictionary):
        rhs = correlations.gather(1, support).unsqueeze(-1)
        gram_support = gram[support.unsqueeze(-1), support.unsqueeze(-2)]
        reg_eye = torch.eye(
            int(self.sparsity_level), device=dictionary.device, dtype=dictionary.dtype
        ).unsqueeze(0)
        gram_support = gram_support + float(self.epsilon) * reg_eye

        try:
            values = torch.linalg.solve(gram_support, rhs).squeeze(-1)
        except RuntimeError:
            diag = gram_support.diagonal(dim1=-2, dim2=-1).clamp_min(self.epsilon)
            values = rhs.squeeze(-1) / diag
        return torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    def batch_omp_with_support(self, X, D):
        """Scratch-style batched OMP that returns support indices and ordered coefficients."""
        self._validate_omp_inputs(X, D)
        X = torch.nan_to_num(X)
        D = torch.nan_to_num(D)

        diag_eps = 1e-4
        cholesky_eps = 1e-6
        _, batch_size = X.size()
        device = D.device
        dtype = D.dtype
        batch_idx = torch.arange(batch_size, device=device)

        dictionary_t = D.t()
        gram = dictionary_t.mm(D)
        if diag_eps > 0.0:
            gram = gram + float(diag_eps) * torch.eye(
                gram.size(0),
                device=device,
                dtype=dtype,
            )
        corr_init = dictionary_t.mm(X).t()
        corr = corr_init.clone()
        dense_coeffs = torch.zeros_like(corr_init)
        cholesky = torch.empty(batch_size, 0, 0, device=device, dtype=dtype)
        support = torch.empty(batch_size, 0, device=device, dtype=torch.long)
        chosen = torch.zeros_like(corr_init, dtype=torch.bool)

        while support.size(1) < int(self.sparsity_level):
            scores = corr.abs().masked_fill(chosen, -1.0)
            index = scores.argmax(dim=1)
            chosen[batch_idx, index] = True

            selected = int(support.size(1))
            diag_g = gram[index, index].view(batch_size, 1, 1)
            if selected == 0:
                cholesky = torch.sqrt(torch.clamp(diag_g, min=cholesky_eps))
            else:
                expanded_batch_idx = batch_idx.unsqueeze(0).expand(selected, batch_size).t()
                gram_stack = gram[
                    support[batch_idx, :],
                    index[expanded_batch_idx],
                ].view(batch_size, selected, 1)
                w = torch.linalg.solve_triangular(cholesky, gram_stack, upper=False)
                w_t = w.transpose(1, 2)
                corner = torch.sqrt(
                    torch.clamp(
                        diag_g - (w_t ** 2).sum(dim=2, keepdim=True),
                        min=cholesky_eps,
                    )
                )
                zeros = torch.zeros(batch_size, selected, 1, device=device, dtype=dtype)
                cholesky = torch.cat(
                    (
                        torch.cat((cholesky, zeros), dim=2),
                        torch.cat((w_t, corner), dim=2),
                    ),
                    dim=1,
                )

            support = torch.cat([support, index.unsqueeze(1)], dim=1)
            support_size = int(support.size(1))
            expanded_batch_idx = batch_idx.unsqueeze(0).expand(support_size, batch_size).t()
            rhs = corr_init[expanded_batch_idx, support].view(batch_size, support_size, 1)
            try:
                coeff_stack = torch.cholesky_solve(rhs, cholesky)
            except RuntimeError:
                gram_support = torch.bmm(cholesky, cholesky.transpose(1, 2))
                reg_eye = torch.eye(
                    support_size,
                    device=device,
                    dtype=dtype,
                ).expand(batch_size, -1, -1)
                coeff_stack = torch.linalg.solve(
                    gram_support + cholesky_eps * reg_eye,
                    rhs,
                )
            coeff_stack = torch.nan_to_num(coeff_stack, nan=0.0, posinf=0.0, neginf=0.0)
            dense_coeffs[batch_idx.unsqueeze(1), support] = coeff_stack.squeeze(-1)
            coeffs_ordered = dense_coeffs[batch_idx.unsqueeze(1), support]
            coeffs_ordered = torch.nan_to_num(coeffs_ordered, nan=0.0, posinf=0.0, neginf=0.0)
            beta = coeffs_ordered.unsqueeze(1).bmm(gram[support[batch_idx], :]).squeeze(1)
            corr = torch.nan_to_num(corr_init - beta, nan=0.0, posinf=0.0, neginf=0.0)

        coeffs_ordered = dense_coeffs[batch_idx.unsqueeze(1), support]
        return support, torch.nan_to_num(coeffs_ordered, nan=0.0, posinf=0.0, neginf=0.0)

    def _is_patch_based(self):
        return self.patch_based

    def _extract_patches(self, z_e):
        _, _, height, width = z_e.shape
        center_pad = max(self.patch_size - self.patch_stride, 0) // 2
        nph = math.ceil(height / self.patch_stride)
        npw = math.ceil(width / self.patch_stride)
        height_padded = (nph - 1) * self.patch_stride + self.patch_size
        width_padded = (npw - 1) * self.patch_stride + self.patch_size
        pad_top = center_pad
        pad_left = center_pad
        pad_bottom = height_padded - height - center_pad
        pad_right = width_padded - width - center_pad
        padded = F.pad(
            z_e,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="reflect",
        )
        patches = F.unfold(
            padded,
            kernel_size=self.patch_size,
            stride=self.patch_stride,
        )
        return patches, nph, npw, height, width

    def _reconstruct_patches_center_crop(self, support, values, height, width):
        batch_size, nph, npw, depth = support.shape
        crop_start = max(self.patch_size - self.patch_stride, 0) // 2
        dictionary = _normalize_dictionary(self.dictionary, eps=self.epsilon).t()
        support_flat = support.to(torch.long).clamp(0, self.num_embeddings - 1).reshape(-1, depth)
        values_flat = values.to(dictionary.dtype).reshape(-1, depth)
        atoms = dictionary[support_flat]
        recon = (atoms * values_flat.unsqueeze(-1)).sum(dim=1)
        recon = recon.view(
            batch_size * nph * npw,
            self.embedding_dim,
            self.patch_size,
            self.patch_size,
        )
        recon = recon[
            :,
            :,
            crop_start:crop_start + self.patch_stride,
            crop_start:crop_start + self.patch_stride,
        ]
        recon = recon.view(
            batch_size,
            nph,
            npw,
            self.embedding_dim,
            self.patch_stride,
            self.patch_stride,
        )
        recon = recon.permute(0, 3, 1, 4, 2, 5).contiguous()
        recon = recon.view(
            batch_size,
            self.embedding_dim,
            nph * self.patch_stride,
            npw * self.patch_stride,
        )
        return recon[:, :, :height, :width]

    def _reconstruct_patches_hann(self, support, values, height, width):
        batch_size, nph, npw, depth = support.shape
        center_pad = max(self.patch_size - self.patch_stride, 0) // 2
        height_padded = (nph - 1) * self.patch_stride + self.patch_size
        width_padded = (npw - 1) * self.patch_stride + self.patch_size
        dictionary = _normalize_dictionary(self.dictionary, eps=self.epsilon).t()
        support_flat = support.to(torch.long).clamp(0, self.num_embeddings - 1).reshape(-1, depth)
        values_flat = values.to(dictionary.dtype).reshape(-1, depth)
        atoms = dictionary[support_flat]
        recon = (atoms * values_flat.unsqueeze(-1)).sum(dim=1)
        window = self._hann_win.to(recon.dtype)
        recon = recon * window.unsqueeze(0)
        recon = recon.view(batch_size, nph * npw, self.patch_dim).permute(0, 2, 1)
        weighted = F.fold(
            recon,
            output_size=(height_padded, width_padded),
            kernel_size=self.patch_size,
            stride=self.patch_stride,
        )
        window_map = F.fold(
            window.view(1, -1, 1).expand(batch_size, -1, nph * npw),
            output_size=(height_padded, width_padded),
            kernel_size=self.patch_size,
            stride=self.patch_stride,
        )
        recon = weighted / window_map.clamp_min(1e-8)
        recon = recon[
            :,
            :,
            center_pad:center_pad + nph * self.patch_stride,
            center_pad:center_pad + npw * self.patch_stride,
        ]
        return recon[:, :, :height, :width]

    def _reconstruct_sparse(self, support, values, height, width):
        if self.patch_reconstruction == "hann" and self._is_patch_based():
            return self._reconstruct_patches_hann(support, values, height, width)
        if self._is_patch_based():
            return self._reconstruct_patches_center_crop(support, values, height, width)

        dictionary = _normalize_dictionary(self.dictionary, eps=self.epsilon).t()
        support_flat = support.to(torch.long).clamp(0, self.num_embeddings - 1).reshape(-1, self.sparsity_level)
        values_flat = values.to(dictionary.dtype).reshape(-1, self.sparsity_level)
        atoms = dictionary[support_flat]
        recon = (atoms * values_flat.unsqueeze(-1)).sum(dim=1)
        batch_size = support.shape[0]
        return recon.view(batch_size, height, width, self.embedding_dim).permute(0, 3, 1, 2).contiguous()

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
        if z_e.dim() != 4:
            raise ValueError(
                f"Expected input [B, C, H, W], got {tuple(z_e.shape)}"
            )
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
        dictionary = _normalize_dictionary(self.dictionary, eps=self.epsilon)
        with torch.no_grad():
            if self._is_patch_based():
                support, values = self.batch_omp_with_support(signals, dictionary)
            else:
                support, values = self.batch_topk_with_support(signals, dictionary)
        support = support.view(B, grid_h, grid_w, self.sparsity_level)
        values = values.view(B, grid_h, grid_w, self.sparsity_level)
        z_dl = self._reconstruct_sparse(support, values, height, width)

        dl_latent_loss = F.mse_loss(z_dl, z_e.detach())
        e_latent_loss = F.mse_loss(z_dl.detach(), z_e)
        loss = dl_latent_loss + self.commitment_cost * e_latent_loss

        self._last_dl_latent_loss = dl_latent_loss.detach()
        self._last_e_latent_loss = e_latent_loss.detach()
        num_sites = max(int(values.shape[0] * values.shape[1] * values.shape[2]), 1)
        dict_norms = self.dictionary.norm(dim=0).detach()
        self._last_diag = {
            "dict_norm_max": dict_norms.max(),
            "dict_norm_mean": dict_norms.mean(),
            "dict_norm_min": dict_norms.min(),
            "coeff_abs_mean": (
                values.abs().sum() / float(self.num_embeddings * num_sites)
            ).detach(),
            "coeff_abs_max": values.abs().max().detach(),
        }

        z_dl = z_e + (z_dl - z_e).detach()
        sparse_codes = SparseCodes(
            support=support,
            values=values,
            num_embeddings=self.num_embeddings,
        )
        return z_dl, loss, sparse_codes
