"""Dictionary-learning bottleneck: Batch OMP over one learned dense dictionary."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bottleneck_utils import SparseCodes, _normalize_dictionary


class DictionaryLearning(nn.Module):
    """Sparse latent bottleneck with plain Batch OMP and a learned dictionary."""

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
        data_init_from_first_batch=False,
        dead_atom_revival=False,
        dead_atom_revival_interval=500,
        dead_atom_revival_max_fraction=0.05,
        dead_atom_revival_noise=0.05,
        dead_atom_revival_patience=5,
        epsilon=1e-10,
    ):
        super().__init__()

        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.sparsity_level = int(sparsity_level)
        self.commitment_cost = float(commitment_cost)
        self.epsilon = float(epsilon)
        self.dict_learning_rate = dict_learning_rate
        self.data_init_from_first_batch = bool(data_init_from_first_batch)
        self.dead_atom_revival = bool(dead_atom_revival)
        self.dead_atom_revival_interval = max(1, int(dead_atom_revival_interval))
        self.dead_atom_revival_max_fraction = float(dead_atom_revival_max_fraction)
        self.dead_atom_revival_noise = max(0.0, float(dead_atom_revival_noise))
        self.dead_atom_revival_patience = max(1, int(dead_atom_revival_patience))

        if self.num_embeddings <= 0:
            raise ValueError(f"num_embeddings must be positive, got {self.num_embeddings}")
        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {self.embedding_dim}")
        if self.sparsity_level <= 0:
            raise ValueError(f"sparsity_level must be positive, got {self.sparsity_level}")
        if self.sparsity_level > self.num_embeddings:
            raise ValueError(
                f"sparsity_level ({self.sparsity_level}) must be <= "
                f"num_embeddings ({self.num_embeddings})"
            )
        if self.commitment_cost < 0.0:
            raise ValueError(f"commitment_cost must be >= 0, got {self.commitment_cost}")
        if self.dead_atom_revival_max_fraction < 0.0:
            raise ValueError(
                "dead_atom_revival_max_fraction must be >= 0, got "
                f"{self.dead_atom_revival_max_fraction}"
            )
        self.patch_based = bool(patch_based)
        self.patch_size = int(patch_size) if self.patch_based else 1
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {self.patch_size}")
        if patch_stride is None:
            patch_stride = self.patch_size
        self.patch_stride = int(patch_stride) if self.patch_based else 1
        if self.patch_stride <= 0:
            raise ValueError(f"patch_stride must be positive, got {self.patch_stride}")
        if self.patch_based and self.patch_stride != self.patch_size:
            raise ValueError(
                "patch-based dictionary learning only supports non-overlapping patches; "
                f"patch_stride ({self.patch_stride}) must equal patch_size ({self.patch_size})"
            )

        patch_reconstruction = str(patch_reconstruction).strip().lower()
        if self.patch_based and patch_reconstruction != "tile":
            raise ValueError(
                "patch_reconstruction must be 'tile' for non-overlapping patch dictionary "
                "learning, got "
                f"{patch_reconstruction!r}"
            )
        self.patch_reconstruction = "tile"

        self.patch_dim = self.embedding_dim * self.patch_size * self.patch_size
        self.dictionary = nn.Parameter(torch.randn(self.patch_dim, self.num_embeddings) * 0.02)

        self.register_buffer("_data_initialized", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("_atom_usage_window", torch.zeros(self.num_embeddings))
        self.register_buffer(
            "_atom_unused_intervals",
            torch.zeros(self.num_embeddings, dtype=torch.long),
        )
        self.register_buffer("_revival_step", torch.zeros((), dtype=torch.long))
        self.register_buffer("_last_dead_atom_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("_last_revived_atom_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("_last_active_atom_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("_last_revival_check_step", torch.zeros((), dtype=torch.long))
        self._revival_candidate_atoms = None

        self.normalize_dictionary_()
        self._last_dl_latent_loss = None
        self._last_e_latent_loss = None
        self._last_dictionary_loss = torch.zeros(())
        self._last_commitment_loss = torch.zeros(())
        self._last_dictionary_loss_for_backward = None
        self._last_bottleneck_objective_for_backward = None
        self._last_extra_bottleneck_loss = torch.zeros(())
        self._last_bottleneck_objective = torch.zeros(())
        self._last_bottleneck_loss = torch.zeros(())

    def effective_dictionary(self) -> torch.Tensor:
        return self.dictionary

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

    def _batch_omp_cholesky_with_support(self, signals, dictionary, debug=False):
        """Batch OMP using a shared Gram matrix and progressive Cholesky solves."""
        embedding_dim, num_signals = signals.shape
        if int(embedding_dim) != int(dictionary.size(0)):
            raise ValueError(
                f"Signal dim ({int(embedding_dim)}) must match dictionary dim "
                f"({int(dictionary.size(0))})"
            )
        dictionary_t = dictionary.t()
        gram_matrix = dictionary_t.mm(dictionary)
        corr_init = dictionary_t.mm(signals).t()
        gamma = torch.zeros_like(corr_init)

        corr = corr_init
        L = torch.ones(num_signals, 1, 1, device=signals.device, dtype=signals.dtype)
        support = torch.zeros(num_signals, 0, dtype=torch.long, device=signals.device)
        omega = torch.ones_like(corr_init, dtype=torch.bool)
        signal_idx = torch.arange(num_signals, device=signals.device)

        for k in range(1, int(self.sparsity_level) + 1):
            scores = corr.abs().masked_fill(~omega, -1.0)
            next_atoms = torch.argmax(scores, dim=1)
            omega[signal_idx, next_atoms] = False
            expanded_signal_idx = signal_idx.unsqueeze(0).expand(k, num_signals).t()

            if k > 1:
                prev_support = support[signal_idx, :]
                new_atoms = next_atoms[expanded_signal_idx[..., :-1]]
                gram_cross = gram_matrix[prev_support, new_atoms].view(num_signals, k - 1, 1)
                w = torch.linalg.solve_triangular(L, gram_cross, upper=False).view(
                    num_signals,
                    1,
                    k - 1,
                )
                bottom_right = (1.0 - (w**2).sum(dim=2, keepdim=True)).clamp_min(
                    max(float(self.epsilon), 1e-10)
                ).sqrt()
                zeros = torch.zeros(
                    num_signals,
                    k - 1,
                    1,
                    device=signals.device,
                    dtype=signals.dtype,
                )
                L = torch.cat(
                    (
                        torch.cat((L, zeros), dim=2),
                        torch.cat((w, bottom_right), dim=2),
                    ),
                    dim=1,
                )

            support = torch.cat([support, next_atoms.unsqueeze(1)], dim=1)
            corr_active = corr_init[expanded_signal_idx, support[signal_idx, :]].view(
                num_signals,
                k,
                1,
            )
            gamma_active = torch.cholesky_solve(corr_active, L).squeeze(-1)
            gamma[signal_idx.unsqueeze(1), support[signal_idx]] = gamma_active

            active = gamma[signal_idx.unsqueeze(1), support[signal_idx]]
            beta = active.unsqueeze(1).bmm(gram_matrix[support[signal_idx], :]).squeeze(1)
            corr = corr_init - beta

            if debug:
                residual_proxy = corr.abs().amax(dim=1).max()
                print(f"Step {k}, max residual correlation: {float(residual_proxy):.4f}")

        values = gamma[signal_idx.unsqueeze(1), support[signal_idx]]
        return support, values, gamma.t()

    def update_gamma(self, signals, dictionary, debug=False):
        """Return the full sparse-code matrix produced by Batch OMP."""
        self._validate_omp_inputs(signals, dictionary)
        signals = torch.nan_to_num(signals)
        dictionary = torch.nan_to_num(dictionary)
        _, _, gamma = self._batch_omp_cholesky_with_support(
            signals,
            dictionary,
            debug=debug,
        )
        return gamma

    def batch_omp_with_support(self, X, D):
        """Return OMP support and coefficients with exactly ``sparsity_level`` atoms."""
        self._validate_omp_inputs(X, D)
        X = torch.nan_to_num(X)
        D = torch.nan_to_num(D)
        support, values, _ = self._batch_omp_cholesky_with_support(X, D)
        return support, torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    def _distributed_is_initialized(self):
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    def _distributed_rank(self):
        if not self._distributed_is_initialized():
            return 0
        return int(torch.distributed.get_rank())

    @torch.no_grad()
    def _broadcast_dictionary_(self):
        if self._distributed_is_initialized():
            torch.distributed.broadcast(self.dictionary.data, src=0)

    @torch.no_grad()
    def _all_gather_signal_columns(self, signals, *, max_local_columns=2048):
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
            return torch.empty(
                self.patch_dim,
                0,
                device=self.dictionary.device,
                dtype=self.dictionary.dtype,
        )
        return atoms

    def _max_revival_count(self) -> int:
        if not self.dead_atom_revival:
            return 0
        if self.dead_atom_revival_max_fraction <= 0.0:
            return 0
        max_fraction = min(float(self.dead_atom_revival_max_fraction), 1.0)
        return max(1, int(math.ceil(float(self.num_embeddings) * max_fraction)))

    @torch.no_grad()
    def _with_revival_noise(self, atoms: torch.Tensor) -> torch.Tensor:
        if atoms.numel() == 0 or self.dead_atom_revival_noise <= 0.0:
            return atoms
        noise = F.normalize(
            torch.randn_like(atoms.float()),
            p=2,
            dim=0,
            eps=max(float(self.epsilon), 1e-8),
        ).to(dtype=atoms.dtype)
        return _normalize_dictionary(
            atoms + float(self.dead_atom_revival_noise) * noise,
            eps=self.epsilon,
        )

    @torch.no_grad()
    def _record_atom_usage_(self, support: torch.Tensor, signals: torch.Tensor) -> None:
        if not self.training or not self.dead_atom_revival:
            return
        if support.numel() == 0:
            return

        support_flat = support.detach().reshape(-1).to(torch.long)
        counts = torch.bincount(support_flat, minlength=self.num_embeddings).to(
            device=self._atom_usage_window.device,
            dtype=self._atom_usage_window.dtype,
        )
        if self._distributed_is_initialized():
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
        self._atom_usage_window.add_(counts)

        next_step = int(self._revival_step.item()) + 1
        should_prepare_candidates = (
            self.dead_atom_revival_max_fraction > 0.0
            and next_step % int(self.dead_atom_revival_interval) == 0
        )
        if not should_prepare_candidates:
            return

        max_count = self._max_revival_count()
        if max_count <= 0:
            self._revival_candidate_atoms = None
            return
        max_columns = max(2048, max_count * 16)
        candidate_signals = self._all_gather_signal_columns(
            signals,
            max_local_columns=max_columns,
        )
        atoms = self._signal_atoms(candidate_signals, max_count)
        if atoms is None:
            self._revival_candidate_atoms = None
            return
        self._revival_candidate_atoms = self._with_revival_noise(atoms.detach())

    @torch.no_grad()
    def _fallback_revival_atoms(self, count: int) -> torch.Tensor:
        atoms = torch.randn(
            self.patch_dim,
            int(count),
            device=self.dictionary.device,
            dtype=self.dictionary.dtype,
        )
        return _normalize_dictionary(atoms, eps=self.epsilon)

    @torch.no_grad()
    def _reset_optimizer_state_for_atoms_(self, optimizer, atom_ids: torch.Tensor) -> None:
        if optimizer is None or atom_ids.numel() == 0:
            return
        state = getattr(optimizer, "state", {}).get(self.dictionary, None)
        if not isinstance(state, dict):
            return
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = state.get(name, None)
            if torch.is_tensor(value) and value.shape == self.dictionary.shape:
                value.index_fill_(1, atom_ids.to(device=value.device), 0.0)

    @torch.no_grad()
    def revive_dead_atoms_after_step_(self, optimizer=None) -> int:
        if not self.training or not self.dead_atom_revival:
            return 0
        self._revival_step.add_(1)
        self._last_revived_atom_count.zero_()
        if self.dead_atom_revival_max_fraction <= 0.0:
            return 0
        if int(self._revival_step.item()) % int(self.dead_atom_revival_interval) != 0:
            return 0

        used_this_window = self._atom_usage_window > 0
        self._last_active_atom_count.copy_(used_this_window.sum().to(torch.long))
        self._atom_unused_intervals[used_this_window] = 0
        self._atom_unused_intervals[~used_this_window] += 1

        dead_mask = self._atom_unused_intervals >= int(self.dead_atom_revival_patience)
        dead_ids = torch.nonzero(dead_mask, as_tuple=False).flatten()
        dead_count = int(dead_ids.numel())
        self._last_dead_atom_count.fill_(dead_count)
        self._last_revival_check_step.copy_(self._revival_step)
        self._atom_usage_window.zero_()

        max_count = self._max_revival_count()
        if dead_count <= 0 or max_count <= 0:
            self._revival_candidate_atoms = None
            return 0

        if dead_count > max_count:
            idle = self._atom_unused_intervals.index_select(0, dead_ids)
            order = torch.argsort(idle, descending=True)
            dead_ids = dead_ids.index_select(0, order[:max_count])
        revive_count = int(dead_ids.numel())
        if revive_count <= 0:
            self._revival_candidate_atoms = None
            return 0

        atoms = self._revival_candidate_atoms
        if atoms is None or int(atoms.size(1)) < revive_count:
            atoms = self._fallback_revival_atoms(revive_count)
        else:
            atoms = atoms[:, :revive_count].to(
                device=self.dictionary.device,
                dtype=self.dictionary.dtype,
            )

        if self._distributed_rank() == 0:
            self.dictionary.data.index_copy_(1, dead_ids.to(self.dictionary.device), atoms)
        self._reset_optimizer_state_for_atoms_(optimizer, dead_ids)
        self._broadcast_dictionary_()
        self.normalize_dictionary_()
        self._atom_unused_intervals[dead_ids] = 0
        self._last_revived_atom_count.fill_(revive_count)
        self._revival_candidate_atoms = None
        return revive_count

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

    def _is_patch_based(self):
        return self.patch_based

    def _extract_patches(self, z_e):
        _, _, height, width = z_e.shape
        nph = math.ceil(height / self.patch_size)
        npw = math.ceil(width / self.patch_size)
        height_padded = nph * self.patch_size
        width_padded = npw * self.patch_size
        pad_bottom = height_padded - height
        pad_right = width_padded - width
        pad = (0, pad_right, 0, pad_bottom)
        padded = F.pad(z_e, pad, mode="replicate") if pad_right or pad_bottom else z_e
        patches = F.unfold(
            padded,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        return patches, nph, npw, height, width

    def _sparse_atom_sum(self, support, values):
        depth = int(support.shape[-1])
        dictionary = _normalize_dictionary(self.effective_dictionary(), eps=self.epsilon).t()
        support_flat = support.to(torch.long).clamp(0, self.num_embeddings - 1).reshape(-1, depth)
        values_flat = values.to(dictionary.dtype).reshape(-1, depth)
        atoms = dictionary[support_flat]
        return (atoms * values_flat.unsqueeze(-1)).sum(dim=1)

    def _reconstruct_patches(self, support, values, height, width):
        batch_size, nph, npw, _ = support.shape
        recon = self._sparse_atom_sum(support, values)
        recon = recon.view(
            batch_size,
            nph,
            npw,
            self.embedding_dim,
            self.patch_size,
            self.patch_size,
        )
        recon = recon.permute(0, 3, 1, 4, 2, 5).contiguous()
        recon = recon.view(
            batch_size,
            self.embedding_dim,
            nph * self.patch_size,
            npw * self.patch_size,
        )
        return recon[:, :, :height, :width]

    def _reconstruct_sparse(self, support, values, height, width):
        if self._is_patch_based():
            return self._reconstruct_patches(support, values, height, width)

        recon = self._sparse_atom_sum(support, values)
        batch_size = support.shape[0]
        return recon.view(
            batch_size,
            height,
            width,
            self.embedding_dim,
        ).permute(0, 3, 1, 2).contiguous()

    def forward(self, z_e):
        """Sparse-code ``z_e`` and return a straight-through reconstructed latent."""
        if z_e.dim() != 4:
            raise ValueError(f"Expected input [B, C, H, W], got {tuple(z_e.shape)}")
        batch_size, channels, height, width = z_e.shape
        if channels != self.embedding_dim:
            raise ValueError(
                f"Expected channel dim {self.embedding_dim} but received {channels}"
            )

        z_e_work = torch.nan_to_num(z_e.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if self._is_patch_based():
            patches, grid_h, grid_w, latent_h, latent_w = self._extract_patches(z_e_work)
            signals = patches.permute(0, 2, 1).contiguous().view(-1, self.patch_dim).t()
        else:
            grid_h, grid_w, latent_h, latent_w = height, width, height, width
            signals = z_e_work.permute(0, 2, 3, 1).contiguous().view(-1, channels).t()

        self._maybe_data_initialize_dictionary_(signals)
        dictionary = _normalize_dictionary(
            self.effective_dictionary().float(),
            eps=max(float(self.epsilon), 1e-8),
        )
        with torch.no_grad():
            support, values = self.batch_omp_with_support(signals, dictionary)
        support = support.view(batch_size, grid_h, grid_w, self.sparsity_level)
        values = values.view(batch_size, grid_h, grid_w, self.sparsity_level).float()
        self._record_atom_usage_(support, signals)

        z_dl = self._reconstruct_sparse(support, values, latent_h, latent_w).float()
        dl_latent_loss = F.mse_loss(z_dl, z_e_work.detach())
        e_latent_loss = F.mse_loss(z_dl.detach(), z_e_work)
        dictionary_loss = dl_latent_loss
        commitment_loss = float(self.commitment_cost) * e_latent_loss
        bottleneck_loss = commitment_loss
        objective = dictionary_loss + bottleneck_loss

        self._last_dl_latent_loss = dl_latent_loss.detach()
        self._last_e_latent_loss = e_latent_loss.detach()
        self._last_dictionary_loss = dictionary_loss.detach()
        self._last_commitment_loss = commitment_loss.detach()
        self._last_dictionary_loss_for_backward = dictionary_loss
        self._last_bottleneck_objective_for_backward = objective
        self._last_extra_bottleneck_loss = z_e_work.new_zeros(())
        self._last_bottleneck_loss = bottleneck_loss.detach()
        self._last_bottleneck_objective = objective.detach()

        z_dl_value = z_dl.to(dtype=z_e.dtype)
        z_dl = z_e + (z_dl_value - z_e).detach()
        sparse_codes = SparseCodes(
            support=support,
            values=values,
            num_embeddings=self.num_embeddings,
        )
        return z_dl, bottleneck_loss, sparse_codes
