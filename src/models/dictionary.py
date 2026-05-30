"""Dictionary-learning bottleneck: OMP sparse coding over a learned dictionary."""

import math
from contextlib import nullcontext
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bottleneck_utils import SparseCodes, _normalize_dictionary


def _gaussian_kl_to_fixed_mean(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: float,
) -> torch.Tensor:
    """KL(q || p) for diagonal Gaussians where p uses a fixed std and mean.

    Note: this is a mathematically-valid Gaussian KL, but inside ``DictionaryLearning``
    the ``target_mean`` is the *data-dependent* OMP coefficient (taken under no_grad).
    That makes the resulting loss a **refinement-around-OMP regularizer**, not a KL
    against a fixed prior. See ``DictionaryLearning.forward`` for the role this term
    plays in the objective. The historical name "coefficient KL" is preserved here
    for the math primitive, but the caller-side names have been changed to
    ``coeff_refine_loss`` / ``variational_coeff_refine_weight`` to match the
    behavior.
    """
    target_std = float(max(target_std, 1e-6))
    target_var = target_std * target_std
    var = logvar.exp().clamp_min(1e-8)
    sq_mean = (mu - target_mean).square()
    kl = 0.5 * ((var + sq_mean) / target_var - 1.0 + math.log(target_var) - logvar)
    return kl.mean()


def _dictionary_abs_offdiag_cosines(
    dictionary: torch.Tensor,
    eps: float,
    detach: bool,
    max_atoms: Optional[int] = None,
):
    atoms = dictionary.detach() if detach else dictionary
    atoms = _normalize_dictionary(atoms, eps=eps)
    if max_atoms is not None and int(atoms.size(1)) > int(max_atoms):
        num_atoms = int(atoms.size(1))
        count = max(1, int(max_atoms))
        step = float(num_atoms) / float(count)
        indices = (
            torch.arange(count, device=atoms.device, dtype=torch.float32)
            .mul(step)
            .floor()
            .to(torch.long)
        )
        atoms = atoms.index_select(1, indices)
    gram = atoms.t() @ atoms
    gram = gram - torch.diag_embed(torch.diagonal(gram))
    return gram.abs(), int(gram.size(0) * max(gram.size(0) - 1, 0))


def _disable_autocast_for(tensor: torch.Tensor):
    device_type = tensor.device.type
    if device_type in {"cpu", "cuda", "xpu", "mps"}:
        return torch.autocast(device_type=device_type, enabled=False)
    return nullcontext()


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
        coef_max=None,
        bounded_omp_refine_steps=8,
        dictionary_usage_ema_decay=0.99,
        dictionary_usage_grad_scale=0.0,
        dictionary_usage_grad_min=0.1,
        dictionary_usage_grad_max=10.0,
        variational_coeffs=False,
        variational_coeff_refine_weight=0.0,
        variational_coeff_target_std=0.25,
        variational_coeff_min_std=0.01,
        dictionary_through_decoder=False,
        data_init_from_first_batch=False,
        dead_atom_revival_steps=0,
        dictionary_update_mode=None,
        dictionary_ksvd_lr=0.2,
        dictionary_ksvd_update_every=1,
        dictionary_ksvd_min_usage=1,
        dictionary_ksvd_max_atoms_per_step=512,
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
        legacy_dictionary_update_mode = legacy_kwargs.pop("dictionary_update_mode", None)
        if dictionary_update_mode is None:
            dictionary_update_mode = legacy_dictionary_update_mode
        dict_ema_decay = legacy_kwargs.pop("dict_ema_decay", None)
        legacy_kwargs.pop("dict_ema_eps", None)
        # A3 rename (May 2026): renamed for honesty about what the term does
        # (refinement-around-OMP, not a KL-to-prior). Refuse the old names
        # explicitly rather than silently dropping them.
        for old_name, new_name in (
            ("variational_coeff_kl_weight", "variational_coeff_refine_weight"),
            ("variational_coeff_prior_std", "variational_coeff_target_std"),
        ):
            if old_name in legacy_kwargs:
                raise TypeError(
                    f"{old_name!r} was renamed to {new_name!r} in the May 2026 "
                    "A3 cleanup; update your configs/checkpoints."
                )
        if legacy_kwargs:
            unknown = ", ".join(sorted(legacy_kwargs))
            raise TypeError(f"Unsupported DictionaryLearning arguments: {unknown}")

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.sparsity_level = sparsity_level
        self.commitment_cost = commitment_cost
        self.epsilon = epsilon
        self.dict_learning_rate = dict_learning_rate
        self.coef_max = None if coef_max is None else float(coef_max)
        if self.coef_max is not None:
            if not math.isfinite(self.coef_max) or self.coef_max <= 0.0:
                raise ValueError(f"coef_max must be finite and > 0, got {self.coef_max}")
        self.bounded_omp_refine_steps = int(bounded_omp_refine_steps)
        if self.bounded_omp_refine_steps < 0:
            raise ValueError(
                "bounded_omp_refine_steps must be >= 0, got "
                f"{self.bounded_omp_refine_steps}"
            )
        self.variational_coeffs = bool(variational_coeffs)
        # ``variational_coeff_refine_weight`` weights the refinement-around-OMP term
        # (Gaussian-KL math, but the "prior" mean is the data-dependent OMP coefficient
        # so it acts as an L2 pull toward OMP plus a variance shrinkage). Renamed from
        # ``variational_coeff_kl_weight`` (May 2026, A3) to avoid implying a fixed prior.
        self.variational_coeff_refine_weight = float(variational_coeff_refine_weight)
        # ``variational_coeff_target_std`` is the std of the reference Gaussian in the
        # refinement term AND the upper bound on the learned posterior std. Renamed from
        # ``variational_coeff_prior_std`` (May 2026, A3).
        self.variational_coeff_target_std = float(variational_coeff_target_std)
        self.variational_coeff_min_std = float(variational_coeff_min_std)
        # A2 switch: when True, ``forward`` skips the straight-through estimator so
        # decoder gradients flow into the dictionary (and into coeff_variational_*).
        # Off by default to preserve the legacy proto/VQ-style training behavior where
        # the dictionary learns only from ``dl_latent_loss``.
        self.dictionary_through_decoder = bool(dictionary_through_decoder)
        self.data_init_from_first_batch = bool(data_init_from_first_batch)
        self.dead_atom_revival_steps = max(0, int(dead_atom_revival_steps or 0))
        if dictionary_update_mode is None:
            dictionary_update_mode = "gradient"
        if dictionary_update_mode not in ("gradient", "usage_ema", "online_ksvd"):
            # Tolerate removed/legacy values (e.g. "ema") by falling back to the default.
            dictionary_update_mode = "gradient"
        self.dictionary_update_mode = str(dictionary_update_mode)
        self.dictionary_ksvd_lr = float(dictionary_ksvd_lr)
        if not 0.0 < self.dictionary_ksvd_lr <= 1.0:
            raise ValueError(
                "dictionary_ksvd_lr must be in (0, 1], got "
                f"{self.dictionary_ksvd_lr}"
            )
        self.dictionary_ksvd_update_every = max(1, int(dictionary_ksvd_update_every))
        self.dictionary_ksvd_min_usage = max(1, int(dictionary_ksvd_min_usage))
        self.dictionary_ksvd_max_atoms_per_step = max(1, int(dictionary_ksvd_max_atoms_per_step))
        if dict_ema_decay is not None:
            dictionary_usage_ema_decay = dict_ema_decay
        self.dictionary_usage_ema_decay = float(dictionary_usage_ema_decay)
        if not 0.0 <= self.dictionary_usage_ema_decay < 1.0:
            raise ValueError(
                "dictionary_usage_ema_decay must be in [0, 1), got "
                f"{self.dictionary_usage_ema_decay}"
            )
        self.dictionary_usage_grad_scale = float(dictionary_usage_grad_scale)
        if self.dictionary_usage_grad_scale < 0.0:
            raise ValueError(
                "dictionary_usage_grad_scale must be >= 0, got "
                f"{self.dictionary_usage_grad_scale}"
            )
        self.dictionary_usage_grad_min = float(dictionary_usage_grad_min)
        self.dictionary_usage_grad_max = float(dictionary_usage_grad_max)
        if self.dictionary_usage_grad_min <= 0.0 or self.dictionary_usage_grad_max <= 0.0:
            raise ValueError("dictionary usage gradient clamps must be positive")
        if self.dictionary_usage_grad_min > self.dictionary_usage_grad_max:
            raise ValueError(
                "dictionary_usage_grad_min cannot exceed dictionary_usage_grad_max: "
                f"{self.dictionary_usage_grad_min} > {self.dictionary_usage_grad_max}"
            )
        if self.variational_coeff_refine_weight < 0.0:
            raise ValueError(
                "variational_coeff_refine_weight must be >= 0, got "
                f"{self.variational_coeff_refine_weight}"
            )
        if self.variational_coeff_target_std <= 0.0:
            raise ValueError(
                "variational_coeff_target_std must be > 0, got "
                f"{self.variational_coeff_target_std}"
            )
        if self.variational_coeff_min_std <= 0.0:
            raise ValueError(
                f"variational_coeff_min_std must be > 0, got {self.variational_coeff_min_std}"
            )
        if self.variational_coeff_min_std > self.variational_coeff_target_std:
            raise ValueError(
                "variational_coeff_min_std cannot exceed variational_coeff_target_std: "
                f"{self.variational_coeff_min_std} > {self.variational_coeff_target_std}"
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
        if patch_reconstruction not in {"center_crop", "hann", "tile"}:
            raise ValueError(
                "patch_reconstruction must be 'center_crop', 'hann', or 'tile', got "
                f"{patch_reconstruction!r}"
            )
        # Non-overlapping patches should use tile stitching.
        if self.patch_stride == self.patch_size and patch_reconstruction != "tile":
            patch_reconstruction = "tile"
        self.patch_reconstruction = str(patch_reconstruction)
        self.patch_dim = self.embedding_dim * self.patch_size * self.patch_size

        self.dictionary = nn.Parameter(
            torch.randn(self.patch_dim, self.num_embeddings) * 0.02
        )
        if self.dictionary_update_mode == "online_ksvd":
            self.dictionary.requires_grad_(False)
        if self.variational_coeffs:
            hidden_dim = max(64, min(256, self.embedding_dim * self.patch_size))
            self.coeff_variational_atom_emb = nn.Embedding(self.num_embeddings, hidden_dim)
            self.coeff_variational_posterior = nn.Sequential(
                nn.Linear(hidden_dim + 1, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2),
            )
        else:
            self.coeff_variational_atom_emb = None
            self.coeff_variational_posterior = None
        if self.patch_size > 1 or self.patch_stride > 1:
            hann_1d = torch.hann_window(self.patch_size, periodic=False)
            window_2d = hann_1d.unsqueeze(1) * hann_1d.unsqueeze(0)
            window_flat = window_2d.flatten().unsqueeze(0).expand(
                self.embedding_dim, -1
            ).reshape(-1)
        else:
            window_flat = torch.ones(self.patch_dim)
        self.register_buffer("_hann_win", window_flat)
        self.register_buffer("dictionary_usage_ema", torch.zeros(self.num_embeddings))
        self.register_buffer("dictionary_usage_steps", torch.zeros((), dtype=torch.long))
        self.register_buffer("_data_initialized", torch.tensor(False, dtype=torch.bool))
        self.normalize_dictionary_()
        self._last_diag = {}
        self._last_dl_latent_loss = None
        self._last_e_latent_loss = None
        self._last_coeff_refine_loss = torch.zeros(())
        self._last_weighted_coeff_refine_loss = torch.zeros(())
        self._last_extra_bottleneck_loss = torch.zeros(())
        self._last_coeff_posterior_std = torch.zeros(())
        self._last_coeff_target_std = torch.tensor(self.variational_coeff_target_std)
        self._last_bottleneck_loss = torch.zeros(())
        self._last_ksvd_batch = None
        self._online_ksvd_steps = 0

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
        variational_prefixes = (
            prefix + "coeff_variational_atom_emb.",
            prefix + "coeff_variational_posterior.",
        )
        if (
            self.variational_coeffs
            and self.coeff_variational_atom_emb is not None
            and self.coeff_variational_posterior is not None
        ):
            expected_state = {
                prefix + "coeff_variational_atom_emb.weight": self.coeff_variational_atom_emb.weight.detach().clone(),
            }
            for subkey, value in self.coeff_variational_posterior.state_dict().items():
                expected_state[prefix + "coeff_variational_posterior." + subkey] = value.detach().clone()
            for key, expected in expected_state.items():
                loaded = state_dict.get(key)
                if loaded is None or tuple(loaded.shape) != tuple(expected.shape):
                    state_dict[key] = expected
        else:
            for key in list(state_dict.keys()):
                if key.startswith(variational_prefixes):
                    state_dict.pop(key, None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

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
            grad = grad - atoms * radial
            if self.dictionary_usage_grad_scale > 0.0:
                usage = self.dictionary_usage_ema.to(device=grad.device, dtype=grad.dtype)
                active = usage > 0
                if bool(active.any()):
                    reference = usage[active].mean().clamp_min(float(self.epsilon))
                    usage = usage.clamp_min(float(self.epsilon))
                    scale = (reference / usage).pow(float(self.dictionary_usage_grad_scale))
                    scale = scale.clamp(
                        float(self.dictionary_usage_grad_min),
                        float(self.dictionary_usage_grad_max),
                    )
                    grad = grad * scale.unsqueeze(0)
            self.dictionary.grad.copy_(torch.nan_to_num(grad))

    def coherence_penalty(self, margin=0.0):
        abs_offdiag, pair_count = _dictionary_abs_offdiag_cosines(
            self.dictionary, eps=self.epsilon, detach=False, max_atoms=4096
        )
        if pair_count <= 0:
            return torch.zeros(
                (), device=self.dictionary.device, dtype=self.dictionary.dtype
            )
        excess = F.relu(abs_offdiag - float(max(0.0, margin)))
        return excess.square().sum() / float(pair_count)

    def coherence_stats(self, max_exact_atoms=4096, sample_atoms=1024):
        num_atoms = int(self.dictionary.size(1))
        max_atoms = None
        if num_atoms > int(max_exact_atoms):
            max_atoms = max(1, int(sample_atoms))
        abs_offdiag, pair_count = _dictionary_abs_offdiag_cosines(
            self.dictionary, eps=self.epsilon, detach=True, max_atoms=max_atoms
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

    def _support_gram(self, gram, support):
        return gram[support.unsqueeze(-1), support.unsqueeze(-2)]

    def _solve_regularized_support_system(
        self,
        gram_support: torch.Tensor,
        rhs: torch.Tensor,
        *,
        reg_eps: Optional[float] = None,
    ) -> torch.Tensor:
        support_size = int(rhs.size(1))
        reg_eps = float(self.epsilon if reg_eps is None else reg_eps)
        reg_eye = torch.eye(
            support_size,
            device=gram_support.device,
            dtype=gram_support.dtype,
        ).unsqueeze(0)
        gram_support = gram_support + reg_eps * reg_eye
        rhs = rhs.unsqueeze(-1)
        try:
            values = torch.linalg.solve(gram_support, rhs).squeeze(-1)
        except RuntimeError:
            diag = gram_support.diagonal(dim1=-2, dim2=-1).clamp_min(
                max(reg_eps, float(self.epsilon))
            )
            values = rhs.squeeze(-1) / diag
        return torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    def _refine_support_coefficients_bounded(
        self,
        gram_support: torch.Tensor,
        rhs: torch.Tensor,
        *,
        init: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.coef_max is None:
            return self._solve_regularized_support_system(gram_support, rhs)

        coeffs = init
        if coeffs is None:
            coeffs = self._solve_regularized_support_system(gram_support, rhs)
        coeffs = torch.nan_to_num(coeffs, nan=0.0, posinf=0.0, neginf=0.0)
        coeffs = coeffs.clamp(-self.coef_max, self.coef_max)
        if self.bounded_omp_refine_steps <= 0:
            return coeffs

        lipschitz = gram_support.abs().sum(dim=-1).amax(dim=-1)
        lipschitz = lipschitz.clamp_min(max(float(self.epsilon), 1e-6))
        step_size = lipschitz.reciprocal().unsqueeze(-1)
        for _ in range(int(self.bounded_omp_refine_steps)):
            grad = torch.bmm(gram_support, coeffs.unsqueeze(-1)).squeeze(-1) - rhs
            coeffs = (coeffs - step_size * grad).clamp(-self.coef_max, self.coef_max)
        return torch.nan_to_num(coeffs, nan=0.0, posinf=0.0, neginf=0.0)

    def _solve_support_coefficients(self, correlations, gram, support, dictionary):
        del dictionary
        rhs = correlations.gather(1, support)
        gram_support = self._support_gram(gram, support)
        return self._refine_support_coefficients_bounded(gram_support, rhs)

    def _support_atoms_rhs_and_gram(self, X_t, D, support):
        atoms = D.t()[support.to(torch.long)]
        rhs = torch.bmm(atoms, X_t.unsqueeze(-1)).squeeze(-1)
        gram_support = torch.bmm(atoms, atoms.transpose(1, 2))
        return atoms, rhs, gram_support

    def _solve_residual_support(self, X_t, D, support):
        atoms, rhs, gram_support = self._support_atoms_rhs_and_gram(X_t, D, support)
        support_size = int(support.size(1))
        diag_eps = 1e-4
        cholesky_eps = 1e-6
        reg_eye = torch.eye(
            support_size,
            device=gram_support.device,
            dtype=gram_support.dtype,
        ).unsqueeze(0)
        gram_reg = gram_support + diag_eps * reg_eye
        init = self._solve_regularized_support_system(
            gram_reg,
            rhs,
            reg_eps=cholesky_eps,
        )
        coeffs = self._refine_support_coefficients_bounded(
            gram_support,
            rhs,
            init=init,
        )
        recon = torch.bmm(coeffs.unsqueeze(1), atoms).squeeze(1)
        return coeffs, recon

    def _select_residual_atom(self, residual, D, chosen):
        """Choose the best atom per signal without materializing a full KxK Gram."""
        _, num_atoms = D.shape
        chunk_size = 16384
        best_scores = residual.new_full((residual.size(0),), -1.0)
        best_indices = torch.zeros(residual.size(0), device=residual.device, dtype=torch.long)
        for start in range(0, int(num_atoms), chunk_size):
            end = min(start + chunk_size, int(num_atoms))
            scores = residual.mm(D[:, start:end]).abs()
            scores.masked_fill_(chosen[:, start:end], -1.0)
            values, local_indices = scores.max(dim=1)
            update = values > best_scores
            best_scores = torch.where(update, values, best_scores)
            best_indices = torch.where(
                update,
                local_indices.to(best_indices.dtype) + int(start),
                best_indices,
            )
        return best_indices

    def batch_omp_with_support(self, X, D):
        """Batched OMP that scales to large dictionaries and low sparsity.

        The old scratch-style path formed a dense KxK dictionary Gram matrix
        and updated correlations through it. That is fast for small K but is the
        wrong shape for large patch dictionaries. This residual OMP path
        recomputes D^T residual in chunks and only solves each sample's small
        support Gram, so memory scales with signal_dim*K and batch*K rather than
        K^2.
        """
        self._validate_omp_inputs(X, D)
        X = torch.nan_to_num(X)
        D = torch.nan_to_num(D)

        _, batch_size = X.size()
        device = X.device
        batch_idx = torch.arange(batch_size, device=device)
        X_t = X.t().contiguous()
        residual = X_t.clone()

        support = torch.empty(batch_size, 0, device=device, dtype=torch.long)
        chosen = torch.zeros(batch_size, int(D.size(1)), device=device, dtype=torch.bool)
        coeffs_ordered = X.new_empty(batch_size, 0)

        while support.size(1) < int(self.sparsity_level):
            index = self._select_residual_atom(residual, D, chosen)
            chosen[batch_idx, index] = True

            support = torch.cat([support, index.unsqueeze(1)], dim=1)
            coeffs_ordered = torch.nan_to_num(coeffs_ordered, nan=0.0, posinf=0.0, neginf=0.0)
            coeffs_ordered, recon = self._solve_residual_support(X_t, D, support)
            residual = torch.nan_to_num(X_t - recon, nan=0.0, posinf=0.0, neginf=0.0)

        return support, torch.nan_to_num(coeffs_ordered, nan=0.0, posinf=0.0, neginf=0.0)

    def update_dictionary_usage_ema_(self, support):
        if self.dictionary_usage_ema_decay <= 0.0 and self.dictionary_usage_grad_scale <= 0.0:
            return
        with torch.no_grad():
            flat = support.detach().to(torch.long).reshape(-1)
            if flat.numel() == 0:
                return
            counts = torch.zeros(
                self.num_embeddings,
                device=flat.device,
                dtype=self.dictionary_usage_ema.dtype,
            )
            counts.scatter_add_(0, flat.clamp(0, self.num_embeddings - 1), torch.ones_like(flat, dtype=counts.dtype))
            site_count = torch.as_tensor(
                max(flat.numel() // max(int(self.sparsity_level), 1), 1),
                device=counts.device,
                dtype=counts.dtype,
            )
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(site_count, op=torch.distributed.ReduceOp.SUM)
            counts = counts / site_count.clamp_min(1.0)
            counts = counts.to(device=self.dictionary_usage_ema.device, dtype=self.dictionary_usage_ema.dtype)
            decay = float(self.dictionary_usage_ema_decay)
            if int(self.dictionary_usage_steps.item()) == 0:
                self.dictionary_usage_ema.copy_(counts)
            else:
                self.dictionary_usage_ema.mul_(decay).add_(counts, alpha=1.0 - decay)
            self.dictionary_usage_steps.add_(1)

    def _distributed_is_initialized(self):
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    @torch.no_grad()
    def _broadcast_dictionary_(self):
        if self._distributed_is_initialized():
            torch.distributed.broadcast(self.dictionary.data, src=0)

    @torch.no_grad()
    def _all_gather_signal_columns(self, signals, *, max_local_columns=2048):
        """Gather a bounded set of latent signal columns across DDP ranks."""
        if signals.ndim != 2 or signals.numel() == 0:
            return signals
        local = signals.detach()
        if int(local.size(1)) > int(max_local_columns):
            idx = torch.linspace(
                0,
                int(local.size(1)) - 1,
                steps=int(max_local_columns),
                device=local.device,
            ).round().to(torch.long)
            local = local.index_select(1, idx)
        if not self._distributed_is_initialized():
            return local

        count = torch.tensor([int(local.size(1))], device=local.device, dtype=torch.long)
        counts = [torch.zeros_like(count) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(counts, count)
        counts = torch.cat(counts, dim=0)
        max_count = int(counts.max().item())
        if max_count <= 0:
            return local[:, :0]
        if int(local.size(1)) < max_count:
            pad = local.new_zeros((int(local.size(0)), max_count - int(local.size(1))))
            local = torch.cat([local, pad], dim=1)

        gathered = [torch.empty_like(local) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered, local.contiguous())
        parts = [part[:, : int(n.item())] for part, n in zip(gathered, counts)]
        return torch.cat(parts, dim=1) if parts else local[:, :0]

    @torch.no_grad()
    def _signal_atoms(self, signals, count):
        if int(count) <= 0 or signals.ndim != 2 or signals.numel() == 0:
            return None
        signals = torch.nan_to_num(
            signals.detach().to(device=self.dictionary.device, dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        valid = signals.norm(dim=0) > max(float(self.epsilon), 1e-8)
        if not bool(valid.any()):
            return None
        signals = signals[:, valid]
        num_signals = int(signals.size(1))
        if num_signals >= int(count):
            idx = torch.linspace(
                0,
                num_signals - 1,
                steps=int(count),
                device=signals.device,
            ).round().to(torch.long)
        else:
            idx = torch.arange(int(count), device=signals.device, dtype=torch.long) % num_signals
        atoms = signals.index_select(1, idx)
        return _normalize_dictionary(atoms.to(dtype=self.dictionary.dtype), eps=self.epsilon)

    @torch.no_grad()
    def _maybe_data_initialize_dictionary_(self, signals):
        if not self.training or not self.data_init_from_first_batch:
            return
        if bool(self._data_initialized.item()):
            return
        atoms = self._signal_atoms(self._all_gather_signal_columns(signals), self.num_embeddings)
        if atoms is not None:
            self.dictionary.copy_(atoms)
            self.normalize_dictionary_()
        self._data_initialized.fill_(True)
        self._broadcast_dictionary_()

    @torch.no_grad()
    def _maybe_revive_dead_atoms_(self, signals):
        if not self.training or self.dead_atom_revival_steps <= 0:
            return
        usage_steps = int(self.dictionary_usage_steps.item())
        if usage_steps <= 0 or usage_steps % int(self.dead_atom_revival_steps) != 0:
            return
        usage = self.dictionary_usage_ema.to(device=self.dictionary.device)
        dead = torch.nonzero(usage <= 0, as_tuple=False).flatten()
        if int(dead.numel()) <= 0:
            return
        atoms = self._signal_atoms(self._all_gather_signal_columns(signals), int(dead.numel()))
        if atoms is None:
            return
        self.dictionary[:, dead].copy_(atoms.to(dtype=self.dictionary.dtype))
        self.normalize_dictionary_()
        self._broadcast_dictionary_()

    @torch.no_grad()
    def _all_gather_ksvd_stats(self, indices, numerator, denominator):
        if not self._distributed_is_initialized():
            return indices, numerator, denominator
        gathered_indices = [torch.empty_like(indices) for _ in range(torch.distributed.get_world_size())]
        gathered_numerators = [torch.empty_like(numerator) for _ in range(torch.distributed.get_world_size())]
        gathered_denominators = [torch.empty_like(denominator) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered_indices, indices)
        torch.distributed.all_gather(gathered_numerators, numerator)
        torch.distributed.all_gather(gathered_denominators, denominator)
        return (
            torch.cat(gathered_indices, dim=0),
            torch.cat(gathered_numerators, dim=1),
            torch.cat(gathered_denominators, dim=0),
        )

    @torch.no_grad()
    def online_ksvd_update_(self):
        """Mini-batch K-SVD-style atom update on the last sparse coding batch.

        Coefficients and support are fixed. For each active atom, the update is
        the coordinate-descent least-squares solution against the residual with
        that atom added back, blended into the current atom and renormalized.
        """
        if self.dictionary_update_mode != "online_ksvd":
            return
        self._online_ksvd_steps += 1
        if self._online_ksvd_steps % self.dictionary_ksvd_update_every != 0:
            return
        batch = self._last_ksvd_batch
        if not batch:
            return

        signals = batch["signals"].to(device=self.dictionary.device, dtype=self.dictionary.dtype)
        support = batch["support"].to(device=self.dictionary.device, dtype=torch.long)
        values = batch["values"].to(device=self.dictionary.device, dtype=self.dictionary.dtype)
        if signals.ndim != 2 or support.ndim != 2 or values.shape != support.shape:
            return
        if int(support.size(0)) <= 0:
            return

        dictionary_t = _normalize_dictionary(self.dictionary.detach(), eps=self.epsilon).t()
        atoms = dictionary_t[support.clamp(0, self.num_embeddings - 1)]
        recon = (atoms * values.unsqueeze(-1)).sum(dim=1).t().contiguous()
        residual = torch.nan_to_num(signals - recon, nan=0.0, posinf=0.0, neginf=0.0)

        flat_support = support.reshape(-1)
        flat_values = values.reshape(-1)
        counts = torch.bincount(
            flat_support.clamp(0, self.num_embeddings - 1),
            minlength=self.num_embeddings,
        )
        active = torch.nonzero(counts >= self.dictionary_ksvd_min_usage, as_tuple=False).flatten()
        if active.numel() == 0:
            return
        active_counts = counts.index_select(0, active)
        order = torch.argsort(active_counts, descending=True)
        active = active.index_select(0, order[: self.dictionary_ksvd_max_atoms_per_step])

        slot_to_signal = torch.arange(
            int(support.size(0)),
            device=support.device,
            dtype=torch.long,
        ).unsqueeze(1).expand_as(support).reshape(-1)
        old_dictionary = dictionary_t.t().contiguous()

        local_count = int(active.numel())
        stat_count = self.dictionary_ksvd_max_atoms_per_step
        indices = torch.full((stat_count,), -1, device=self.dictionary.device, dtype=torch.long)
        numerator = torch.zeros(
            self.patch_dim,
            stat_count,
            device=self.dictionary.device,
            dtype=self.dictionary.dtype,
        )
        denominator = torch.zeros(stat_count, device=self.dictionary.device, dtype=self.dictionary.dtype)
        if local_count > 0:
            indices[:local_count] = active[:local_count]

        for out_col, atom_idx in enumerate(active.tolist()):
            mask = flat_support == int(atom_idx)
            if not bool(mask.any()):
                continue
            coeff = flat_values[mask]
            denom = coeff.square().sum()
            if float(denom.detach().item()) <= float(self.epsilon):
                continue
            signal_ids = slot_to_signal[mask]
            residual_plus = residual.index_select(1, signal_ids) + old_dictionary[:, atom_idx : atom_idx + 1] * coeff.unsqueeze(0)
            numerator[:, out_col] = (residual_plus * coeff.unsqueeze(0)).sum(dim=1)
            denominator[out_col] = denom

        indices, numerator, denominator = self._all_gather_ksvd_stats(indices, numerator, denominator)
        valid = (indices >= 0) & (denominator > float(self.epsilon))
        if not bool(valid.any()):
            return
        valid_indices = indices[valid]
        for atom_idx in valid_indices.unique(sorted=True).tolist():
            mask = (indices == int(atom_idx)) & valid
            denom = denominator[mask].sum()
            if float(denom.detach().item()) <= float(self.epsilon):
                continue
            update = numerator[:, mask].sum(dim=1) / denom
            update = torch.nan_to_num(update, nan=0.0, posinf=0.0, neginf=0.0)
            update = F.normalize(update, p=2, dim=0, eps=self.epsilon)
            current = old_dictionary[:, atom_idx]
            if torch.dot(update, current) < 0:
                update = -update
            blended = (1.0 - self.dictionary_ksvd_lr) * current + self.dictionary_ksvd_lr * update
            self.dictionary[:, atom_idx].copy_(F.normalize(blended, p=2, dim=0, eps=self.epsilon))

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
        pad = (pad_left, pad_right, pad_top, pad_bottom)
        # ``reflect`` padding requires each pad to be smaller than the
        # corresponding spatial dimension; fall back to ``replicate`` for
        # latents smaller than the patch (e.g. a 1x1 latent with patch_size > 1).
        if pad_left < width and pad_right < width and pad_top < height and pad_bottom < height:
            padded = F.pad(z_e, pad, mode="reflect")
        else:
            padded = F.pad(z_e, pad, mode="replicate")
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

    def _reconstruct_patches_center_crop(self, support, values, height, width):
        batch_size, nph, npw, _ = support.shape
        crop_start = max(self.patch_size - self.patch_stride, 0) // 2
        recon = self._sparse_atom_sum(support, values)
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
        batch_size, nph, npw, _ = support.shape
        center_pad = max(self.patch_size - self.patch_stride, 0) // 2
        height_padded = (nph - 1) * self.patch_stride + self.patch_size
        width_padded = (npw - 1) * self.patch_stride + self.patch_size
        recon = self._sparse_atom_sum(support, values)
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

    def _reconstruct_patches_tile(self, support, values, height, width):
        """Direct patch tiling for non-overlapping patches (stride == size)."""
        B, nph, npw, D = support.shape
        C = self.embedding_dim
        recon = self._sparse_atom_sum(support, values, clamp_values=True)

        recon = recon.view(B, nph, npw, C, self.patch_size, self.patch_size)
        recon = recon.permute(0, 3, 1, 4, 2, 5).contiguous()
        recon = recon.view(B, C, nph * self.patch_size, npw * self.patch_size)
        return recon[:, :, :height, :width]

    def _reconstruct_sparse(self, support, values, height, width):
        if self.patch_reconstruction == "tile" and self._is_patch_based():
            return self._reconstruct_patches_tile(support, values, height, width)
        if self.patch_reconstruction == "hann" and self._is_patch_based():
            return self._reconstruct_patches_hann(support, values, height, width)
        if self._is_patch_based():
            return self._reconstruct_patches_center_crop(support, values, height, width)

        recon = self._sparse_atom_sum(support, values)
        batch_size = support.shape[0]
        return recon.view(batch_size, height, width, self.embedding_dim).permute(0, 3, 1, 2).contiguous()

    def clamp_sparse_coeffs(self, coeffs: torch.Tensor) -> torch.Tensor:
        coeffs = torch.nan_to_num(coeffs)
        if self.coef_max is None:
            return coeffs
        return coeffs.clamp(-self.coef_max, self.coef_max)

    def _coeff_posterior_stats(
        self,
        support: torch.Tensor,
        coeffs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.variational_coeffs:
            raise RuntimeError("_coeff_posterior_stats requires variational_coeffs=True")
        if self.coeff_variational_atom_emb is None or self.coeff_variational_posterior is None:
            raise RuntimeError("variational coefficient modules were not initialized")
        if support.shape != coeffs.shape:
            raise ValueError(f"support and coeffs shape mismatch: {support.shape} vs {coeffs.shape}")

        support_clamped = support.to(torch.long).clamp(0, self.num_embeddings - 1)
        coeffs_base = self.clamp_sparse_coeffs(coeffs.to(torch.float32))
        atom_emb = self.coeff_variational_atom_emb(support_clamped)
        posterior_in = torch.cat([atom_emb, coeffs_base.unsqueeze(-1)], dim=-1)
        posterior_raw = self.coeff_variational_posterior(posterior_in)

        mean_offset = self.variational_coeff_target_std * torch.tanh(posterior_raw[..., 0])
        posterior_mu = self.clamp_sparse_coeffs(coeffs_base + mean_offset)

        std_range = max(self.variational_coeff_target_std - self.variational_coeff_min_std, 0.0)
        posterior_std = self.variational_coeff_min_std + std_range * torch.sigmoid(posterior_raw[..., 1])
        posterior_logvar = 2.0 * torch.log(posterior_std.clamp_min(1e-6))
        return posterior_mu, posterior_logvar

    def project_sparse_coeffs(self, support: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        coeffs_clamped = self.clamp_sparse_coeffs(coeffs)
        if not self.variational_coeffs:
            return coeffs_clamped
        coeff_mu, _ = self._coeff_posterior_stats(support, coeffs_clamped)
        return coeff_mu

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

        Gradient flow (important — design choice, see A2 in the May 2026 review):

        * OMP support selection and (non-variational) coefficient values are
          computed under ``no_grad``. They are not part of the autograd graph.
        * ``dl_latent_loss = MSE(z_dl, z_e.detach())`` is the **only** training
          signal the dictionary atoms receive in the legacy path. Pixel /
          perceptual losses do **not** backprop into the dictionary because of
          the straight-through estimator on the final line below.
        * ``e_latent_loss = MSE(z_dl.detach(), z_e)`` trains the encoder to sit
          near the dictionary span.
        * ``z_dl = z_e + (z_dl - z_e).detach()`` is the standard VQ-VAE-style
          straight-through estimator: the decoder consumes ``z_dl``'s value but
          backprops into ``z_e`` (encoder), not into ``self.dictionary``.

        If ``self.dictionary_through_decoder`` is True (A2 switch), a zero-valued
        dictionary-gradient term is added to the STE. Decoder gradients then flow
        into both ``z_e`` and the dictionary atoms (and into the
        ``coeff_variational_*`` modules in the variational branch).

        The ``coeff_refine_loss`` term (when ``variational_coeffs=True``) is a
        Gaussian KL whose reference mean is the *data-dependent* OMP
        coefficient — it acts as a refinement-around-OMP regularizer, not a
        KL against a fixed prior. Renamed from ``coeff_kl_loss`` (May 2026, A3)
        for honesty.
        """
        if z_e.dim() != 4:
            raise ValueError(
                f"Expected input [B, C, H, W], got {tuple(z_e.shape)}"
            )
        B, C, H, W = z_e.shape
        if C != self.embedding_dim:
            raise ValueError(
                f"Expected channel dim {self.embedding_dim} but received {C}"
            )

        # OMP support selection and least-squares coefficient solves are fragile in
        # FP16. Keep this bottleneck math in FP32 under AMP, then cast only the
        # returned latent value back to the encoder dtype.
        with _disable_autocast_for(z_e):
            z_e_work = torch.nan_to_num(z_e.float(), nan=0.0, posinf=0.0, neginf=0.0)
            if self._is_patch_based():
                patches, grid_h, grid_w, height, width = self._extract_patches(z_e_work)
                signals = patches.permute(0, 2, 1).contiguous().view(-1, self.patch_dim).t()
            else:
                grid_h, grid_w, height, width = H, W, H, W
                signals = z_e_work.permute(0, 2, 3, 1).contiguous().view(-1, C).t()
            self._maybe_data_initialize_dictionary_(signals)
            self._maybe_revive_dead_atoms_(signals)
            dictionary = _normalize_dictionary(self.dictionary.float(), eps=max(float(self.epsilon), 1e-8))
            with torch.no_grad():
                support, values = self.batch_omp_with_support(signals, dictionary)
            support = support.view(B, grid_h, grid_w, self.sparsity_level)
            values = self.clamp_sparse_coeffs(
                values.view(B, grid_h, grid_w, self.sparsity_level)
            ).float()
            if self.training and self.dictionary_update_mode == "online_ksvd":
                self._last_ksvd_batch = {
                    "signals": signals.detach(),
                    "support": support.detach().reshape(-1, self.sparsity_level),
                    "values": values.detach().reshape(-1, self.sparsity_level),
                }
            if self.training:
                self.update_dictionary_usage_ema_(support)

            coeff_refine_loss = z_e_work.new_zeros(())
            weighted_coeff_refine_loss = z_e_work.new_zeros(())
            coeffs_for_recon = values
            if self.variational_coeffs:
                coeff_mu, coeff_logvar = self._coeff_posterior_stats(support, values)
                if self.training:
                    coeff_eps = torch.randn_like(coeff_mu)
                    coeff_std = (0.5 * coeff_logvar).exp()
                    coeffs_for_recon = self.clamp_sparse_coeffs(coeff_mu + coeff_std * coeff_eps)
                else:
                    coeffs_for_recon = coeff_mu
                # Gaussian-KL math, but ``values`` (the OMP coefficient) is data-dependent and
                # detached, so this acts as a refinement-around-OMP regularizer. Renamed from
                # ``coeff_kl_loss`` (May 2026, A3); see ``_gaussian_kl_to_fixed_mean`` docstring.
                coeff_refine_loss = _gaussian_kl_to_fixed_mean(
                    coeff_mu,
                    coeff_logvar,
                    values,
                    target_std=self.variational_coeff_target_std,
                )
                weighted_coeff_refine_loss = (
                    float(self.variational_coeff_refine_weight) * coeff_refine_loss
                )
                self._last_coeff_posterior_std = (0.5 * coeff_logvar).exp().mean().detach()
                self._last_coeff_target_std = torch.as_tensor(
                    self.variational_coeff_target_std,
                    device=z_e.device,
                    dtype=torch.float32,
                )
            else:
                self._last_coeff_posterior_std = z_e_work.new_zeros(())
                self._last_coeff_target_std = torch.as_tensor(
                    self.variational_coeff_target_std,
                    device=z_e.device,
                    dtype=torch.float32,
                )

            z_dl = self._reconstruct_sparse(support, coeffs_for_recon, height, width).float()

            dl_latent_loss = F.mse_loss(z_dl, z_e_work.detach())
            e_latent_loss = F.mse_loss(z_dl.detach(), z_e_work)
            loss = dl_latent_loss + self.commitment_cost * e_latent_loss + weighted_coeff_refine_loss

        self._last_dl_latent_loss = dl_latent_loss.detach()
        self._last_e_latent_loss = e_latent_loss.detach()
        self._last_coeff_refine_loss = coeff_refine_loss.detach()
        self._last_weighted_coeff_refine_loss = weighted_coeff_refine_loss.detach()
        self._last_extra_bottleneck_loss = weighted_coeff_refine_loss.detach()
        self._last_bottleneck_loss = loss.detach()
        num_sites = max(int(values.shape[0] * values.shape[1] * values.shape[2]), 1)
        dict_norms = self.dictionary.norm(dim=0).detach()
        coeff_abs = coeffs_for_recon.abs()
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
            "dict_usage_ema_max": self.dictionary_usage_ema.max().detach(),
            "dict_usage_ema_mean": self.dictionary_usage_ema.mean().detach(),
            "dict_usage_ema_min": self.dictionary_usage_ema.min().detach(),
            "coeff_abs_mean": (
                coeff_abs.sum() / float(self.num_embeddings * num_sites)
            ).detach(),
            "coeff_active_abs_mean": coeff_abs.mean().detach(),
            "coeff_abs_max": coeff_abs_max,
            "coeff_clip_frac": coeff_clip_frac,
        }

        if not self.dictionary_through_decoder:
            # Legacy VQ-style straight-through estimator: decoder loss reaches the
            # encoder but bypasses the dictionary and (in variational mode) the
            # coeff_variational_* networks.
            z_dl_value = z_dl.to(dtype=z_e.dtype)
            z_dl = z_e + (z_dl_value - z_e).detach()
        else:
            # Preserve the STE encoder path and add a zero-valued dictionary path
            # so decoder losses train both encoder and dictionary.
            z_dl_value = z_dl.to(dtype=z_e.dtype)
            z_dl = z_e + (z_dl_value - z_e).detach() + (z_dl_value - z_dl_value.detach())
        sparse_codes = SparseCodes(
            support=support,
            values=coeffs_for_recon,
            num_embeddings=self.num_embeddings,
        )
        return z_dl, loss, sparse_codes
