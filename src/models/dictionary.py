"""Dictionary-learning bottleneck: OMP sparse coding over a learned dictionary."""

import math
from contextlib import nullcontext
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bottleneck_utils import SparseCodes, _normalize_dictionary


def _prime_factors(value: int) -> list[int]:
    factors: list[int] = []
    divisor = 2
    while divisor * divisor <= value:
        while value % divisor == 0:
            factors.append(divisor)
            value //= divisor
        divisor += 1
    if value > 1:
        factors.append(value)
    return factors


def _balanced_factor_dims(num_embeddings: int, mode_dims: Sequence[int]) -> tuple[int, int, int]:
    """Factor ``num_embeddings`` across tensor modes with balanced overcompleteness."""
    dims = [max(int(dim), 1) for dim in mode_dims]
    factors = [1, 1, 1]
    for prime in sorted(_prime_factors(int(num_embeddings)), reverse=True):
        mode = min(range(3), key=lambda idx: (factors[idx] / float(dims[idx]), factors[idx]))
        factors[mode] *= int(prime)
    return int(factors[0]), int(factors[1]), int(factors[2])


def _coerce_factor_dims(
    raw: Optional[Sequence[int] | str],
    *,
    num_embeddings: int,
    mode_dims: Sequence[int],
) -> tuple[int, int, int]:
    if raw is None:
        dims = _balanced_factor_dims(int(num_embeddings), mode_dims)
    elif isinstance(raw, str):
        parts = [part.strip() for part in raw.replace("x", ",").split(",") if part.strip()]
        dims = tuple(int(part) for part in parts)
    else:
        dims = tuple(int(part) for part in raw)
    if len(dims) != 3:
        raise ValueError(
            "separable_dictionary_factor_dims must contain exactly three values "
            f"(channel, patch_y, patch_x), got {dims}"
        )
    if any(dim <= 0 for dim in dims):
        raise ValueError(f"separable dictionary factor dims must be positive, got {dims}")
    product = int(dims[0] * dims[1] * dims[2])
    if product != int(num_embeddings):
        raise ValueError(
            "separable_dictionary_factor_dims must multiply to num_embeddings, "
            f"got {dims} -> {product} vs {int(num_embeddings)}"
        )
    return int(dims[0]), int(dims[1]), int(dims[2])


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
        variational_coeffs=False,
        variational_coeff_refine_weight=0.0,
        variational_coeff_target_std=0.25,
        variational_coeff_min_std=0.01,
        data_init_from_first_batch=False,
        separable_dictionary_rank=0,
        separable_dictionary_factor_dims=None,
        epsilon=1e-10,
        **legacy_kwargs,
    ):
        super().__init__()

        legacy_kwargs.pop("sparse_solver", None)
        legacy_kwargs.pop("tolerance", None)
        legacy_kwargs.pop("omp_debug", None)
        omp_residual_tolerance = legacy_kwargs.pop("omp_residual_tolerance", None)
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
        # Retired (June 2026 simplification): the bottleneck is now a single plain
        # gradient-trained nn.Parameter. Accept-and-ignore the removed update-mode /
        # online-K-SVD / usage-EMA / dead-atom knobs so old configs and checkpoint
        # hparams keep loading.
        for _retired in (
            "dictionary_update_mode",
            "dictionary_through_decoder",
            "dead_atom_revival_steps",
            "dictionary_usage_ema_decay",
            "dictionary_usage_grad_scale",
            "dictionary_usage_grad_min",
            "dictionary_usage_grad_max",
            "dictionary_ksvd_lr",
            "dictionary_ksvd_update_every",
            "dictionary_ksvd_min_usage",
            "dictionary_ksvd_max_atoms_per_step",
            "online_ksvd_enabled",
            "online_ksvd_start_step",
            "online_ksvd_interval_steps",
            "online_ksvd_stop_step",
            "online_ksvd_max_samples",
            "online_ksvd_max_atoms",
            "online_ksvd_blend",
            "online_ksvd_min_coeff",
            "dict_ema_decay",
            "dict_ema_eps",
        ):
            legacy_kwargs.pop(_retired, None)
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
        self.omp_residual_tolerance = (
            None if omp_residual_tolerance is None else float(omp_residual_tolerance)
        )
        if self.omp_residual_tolerance is not None:
            if not math.isfinite(self.omp_residual_tolerance) or self.omp_residual_tolerance < 0.0:
                raise ValueError(
                    "omp_residual_tolerance must be finite and >= 0, got "
                    f"{self.omp_residual_tolerance}"
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
        self.data_init_from_first_batch = bool(data_init_from_first_batch)
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
        if patch_reconstruction == "hann" and self.patch_size <= 2:
            patch_reconstruction = "center_crop"
        # Non-overlapping patches should use tile stitching.
        if self.patch_stride == self.patch_size and patch_reconstruction != "tile":
            patch_reconstruction = "tile"
        self.patch_reconstruction = str(patch_reconstruction)
        self.patch_dim = self.embedding_dim * self.patch_size * self.patch_size
        self.separable_dictionary_rank = int(separable_dictionary_rank or 0)
        if self.separable_dictionary_rank < 0:
            raise ValueError(
                "separable_dictionary_rank must be >= 0, got "
                f"{self.separable_dictionary_rank}"
            )
        self.separable_dictionary_enabled = self.separable_dictionary_rank > 0
        self.separable_dictionary_mode_dims = (
            int(self.embedding_dim),
            int(self.patch_size),
            int(self.patch_size),
        )
        self.separable_dictionary_factor_dims = _coerce_factor_dims(
            separable_dictionary_factor_dims,
            num_embeddings=self.num_embeddings,
            mode_dims=self.separable_dictionary_mode_dims,
        )
        if self.separable_dictionary_enabled:
            self.dictionary = nn.Parameter(
                torch.empty(self.patch_dim, self.num_embeddings),
                requires_grad=False,
            )
            self.separable_dictionary_factors = nn.ParameterList()
            for mode_dim, factor_dim in zip(
                self.separable_dictionary_mode_dims,
                self.separable_dictionary_factor_dims,
            ):
                self.separable_dictionary_factors.append(
                    nn.Parameter(
                        torch.randn(
                            self.separable_dictionary_rank,
                            int(mode_dim),
                            int(factor_dim),
                        )
                        * 0.02
                    )
                )
        else:
            self.dictionary = nn.Parameter(
                torch.randn(self.patch_dim, self.num_embeddings) * 0.02
            )
            self.separable_dictionary_factors = nn.ParameterList()
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

    def _compose_separable_dictionary(self) -> torch.Tensor:
        if not self.separable_dictionary_enabled:
            return self.dictionary
        factors = list(self.separable_dictionary_factors)
        if len(factors) != 3:
            raise RuntimeError(
                "separable dictionary expects three factor tensors, got "
                f"{len(factors)}"
            )
        dictionary = None
        rank = int(self.separable_dictionary_rank)
        for rank_idx in range(rank):
            channel = factors[0][rank_idx]
            patch_y = factors[1][rank_idx]
            patch_x = factors[2][rank_idx]
            term = torch.einsum(
                "ap,bq,cr->abcpqr",
                channel,
                patch_y,
                patch_x,
            ).reshape(self.patch_dim, self.num_embeddings)
            dictionary = term if dictionary is None else dictionary + term
        if dictionary is None:
            dictionary = self.dictionary
        return _normalize_dictionary(dictionary, eps=self.epsilon)

    def effective_dictionary(self) -> torch.Tensor:
        """Return the dense dictionary used by sparse coding.

        Dense dictionaries return the learned table directly. Separable
        dictionaries compose the full table as a sum of Kronecker-factor terms,
        matching the low-separation-rank structure from LSR dictionary learning.
        """
        if self.separable_dictionary_enabled:
            return self._compose_separable_dictionary()
        return self.dictionary

    def normalize_dictionary_(self):
        with torch.no_grad():
            if self.separable_dictionary_enabled:
                for factor in self.separable_dictionary_factors:
                    factor.copy_(F.normalize(torch.nan_to_num(factor), p=2, dim=1, eps=self.epsilon))
                self.dictionary.copy_(self._compose_separable_dictionary().detach())
            else:
                self.dictionary.copy_(
                    _normalize_dictionary(self.dictionary.detach(), eps=self.epsilon)
                )

    def project_dictionary_gradient_(self):
        if self.separable_dictionary_enabled:
            with torch.no_grad():
                for factor in self.separable_dictionary_factors:
                    if factor.grad is None:
                        continue
                    atoms = F.normalize(
                        torch.nan_to_num(factor.detach()),
                        p=2,
                        dim=1,
                        eps=self.epsilon,
                    )
                    grad = torch.nan_to_num(factor.grad)
                    radial = (atoms * grad).sum(dim=1, keepdim=True)
                    factor.grad.copy_(torch.nan_to_num(grad - atoms * radial))
            return
        if self.dictionary.grad is None:
            return
        with torch.no_grad():
            atoms = _normalize_dictionary(self.dictionary.detach(), eps=self.epsilon)
            grad = torch.nan_to_num(self.dictionary.grad)
            radial = (atoms * grad).sum(dim=0, keepdim=True)
            grad = grad - atoms * radial
            self.dictionary.grad.copy_(torch.nan_to_num(grad))

    def coherence_penalty(self, margin=0.0):
        abs_offdiag, pair_count = _dictionary_abs_offdiag_cosines(
            self.effective_dictionary(), eps=self.epsilon, detach=False, max_atoms=4096
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
            self.effective_dictionary(), eps=self.epsilon, detach=True, max_atoms=max_atoms
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
        active = torch.ones(batch_size, device=device, dtype=torch.bool)
        tolerance_sq = None
        if self.omp_residual_tolerance is not None:
            tolerance_sq = (X_t * X_t).sum(dim=1) * float(self.omp_residual_tolerance)

        while support.size(1) < int(self.sparsity_level):
            if tolerance_sq is not None and not bool(active.any().item()):
                break
            prev_k = int(support.size(1))
            prev_coeffs = coeffs_ordered
            prev_residual = residual
            index = self._select_residual_atom(residual, D, chosen)
            if tolerance_sq is not None:
                index = torch.where(active, index, torch.zeros_like(index))
                chosen[batch_idx[active], index[active]] = True
            else:
                chosen[batch_idx, index] = True

            support = torch.cat([support, index.unsqueeze(1)], dim=1)
            coeffs_ordered = torch.nan_to_num(coeffs_ordered, nan=0.0, posinf=0.0, neginf=0.0)
            coeffs_ordered, recon = self._solve_residual_support(X_t, D, support)
            residual = torch.nan_to_num(X_t - recon, nan=0.0, posinf=0.0, neginf=0.0)
            if tolerance_sq is not None:
                inactive = ~active
                if bool(inactive.any().item()):
                    if prev_k > 0:
                        coeffs_ordered[inactive, :prev_k] = prev_coeffs[inactive]
                    coeffs_ordered[inactive, prev_k:] = 0.0
                    residual[inactive] = prev_residual[inactive]
                residual_sq = (residual * residual).sum(dim=1)
                active = active & (residual_sq > tolerance_sq)

        if support.size(1) < int(self.sparsity_level):
            pad = int(self.sparsity_level) - int(support.size(1))
            support = torch.cat(
                [support, torch.zeros(batch_size, pad, device=device, dtype=torch.long)],
                dim=1,
            )
            coeffs_ordered = torch.cat(
                [coeffs_ordered, X.new_zeros(batch_size, pad)],
                dim=1,
            )
        return support, torch.nan_to_num(coeffs_ordered, nan=0.0, posinf=0.0, neginf=0.0)

    def _distributed_is_initialized(self):
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    @torch.no_grad()
    def _broadcast_dictionary_(self):
        if self._distributed_is_initialized():
            if self.separable_dictionary_enabled:
                for factor in self.separable_dictionary_factors:
                    torch.distributed.broadcast(factor.data, src=0)
                self.normalize_dictionary_()
            else:
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
        signals = F.normalize(signals[:, valid], p=2, dim=0, eps=max(float(self.epsilon), 1e-8))
        num_signals = int(signals.size(1))
        if num_signals >= int(count):
            idx = torch.randperm(num_signals, device=signals.device)[: int(count)]
            atoms = signals.index_select(1, idx)
        else:
            atoms = torch.empty(
                int(signals.size(0)),
                int(count),
                device=signals.device,
                dtype=signals.dtype,
            )
            idx = torch.randperm(num_signals, device=signals.device)
            atoms[:, :num_signals] = signals.index_select(1, idx)
            remaining = int(count) - num_signals
            if remaining > 0:
                base_idx = torch.randint(num_signals, (remaining,), device=signals.device)
                base = signals.index_select(1, base_idx)
                noise = F.normalize(
                    torch.randn_like(base),
                    p=2,
                    dim=0,
                    eps=max(float(self.epsilon), 1e-8),
                )
                atoms[:, num_signals:] = base + 0.25 * noise
        return _normalize_dictionary(atoms.to(dtype=self.dictionary.dtype), eps=self.epsilon)

    @torch.no_grad()
    def _sample_atoms_from_signals(self, signals: torch.Tensor, count: int) -> torch.Tensor:
        atoms = self._signal_atoms(signals, count)
        if atoms is None:
            return torch.empty(self.patch_dim, 0, device=self.dictionary.device, dtype=self.dictionary.dtype)
        return atoms

    @torch.no_grad()
    def _maybe_data_initialize_dictionary_(self, signals):
        if not self.training or not self.data_init_from_first_batch:
            return
        if bool(self._data_initialized.item()):
            return
        if self.separable_dictionary_enabled:
            self.normalize_dictionary_()
            self._data_initialized.fill_(True)
            self._broadcast_dictionary_()
            return
        atoms = self._signal_atoms(self._all_gather_signal_columns(signals), self.num_embeddings)
        if atoms is not None:
            self.dictionary.copy_(atoms)
            self.normalize_dictionary_()
        self._data_initialized.fill_(True)
        self._broadcast_dictionary_()

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
        dictionary = _normalize_dictionary(self.effective_dictionary(), eps=self.epsilon).t()
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
            dictionary = _normalize_dictionary(
                self.effective_dictionary().float(),
                eps=max(float(self.epsilon), 1e-8),
            )
            with torch.no_grad():
                support, values = self.batch_omp_with_support(signals, dictionary)
            support = support.view(B, grid_h, grid_w, self.sparsity_level)
            values = self.clamp_sparse_coeffs(
                values.view(B, grid_h, grid_w, self.sparsity_level)
            ).float()

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
        dict_norms = self.effective_dictionary().norm(dim=0).detach()
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
            "coeff_abs_mean": (
                coeff_abs.sum() / float(self.num_embeddings * num_sites)
            ).detach(),
            "coeff_active_abs_mean": coeff_abs.mean().detach(),
            "coeff_abs_max": coeff_abs_max,
            "coeff_clip_frac": coeff_clip_frac,
        }

        # VQ-style straight-through estimator: the decoder consumes ``z_dl``'s value
        # but backprops into ``z_e`` (encoder); the dictionary learns from the
        # bottleneck losses, not through the decoder.
        z_dl_value = z_dl.to(dtype=z_e.dtype)
        z_dl = z_e + (z_dl_value - z_e).detach()
        sparse_codes = SparseCodes(
            support=support,
            values=coeffs_for_recon,
            num_embeddings=self.num_embeddings,
        )
        return z_dl, loss, sparse_codes
