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
        progressive_loss=False,
        patch_based=False,
        patch_size=1,
        patch_stride=None,
        patch_reconstruction="tile",
        data_init_from_first_batch=False,
        data_init_start_step=0,
        data_init_accumulation_steps=1,
        dead_atom_revival=False,
        dead_atom_revival_interval=500,
        dead_atom_revival_max_fraction=0.05,
        dead_atom_revival_noise=0.05,
        dead_atom_revival_patience=5,
        omp_compute_precision="float32",
        token_noise_rate=0.0,
        omp_ridge=0.0,
        omp_max_support_coherence=1.0,
        dictionary_update_mode="gradient",
        dictionary_update_relaxation=0.25,
        dictionary_update_max_atoms_per_step=512,
        dictionary_update_min_usage=2,
        dictionary_update_accumulation_steps=1,
        dictionary_update_max_backtracks=6,
        dictionary_collective_backend="gloo",
        coefficient_quantization_bits=0,
        coefficient_quantization_max=None,
        coefficient_quantization_start_step=0,
        coefficient_quantization_warmup_steps=0,
        commitment_normalize_by_variance=False,
        epsilon=1e-10,
    ):
        super().__init__()

        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.sparsity_level = int(sparsity_level)
        self.commitment_cost = float(commitment_cost)
        self.epsilon = float(epsilon)
        self.dict_learning_rate = dict_learning_rate
        self.progressive_loss = bool(progressive_loss)
        self.data_init_from_first_batch = bool(data_init_from_first_batch)
        self.data_init_start_step = int(data_init_start_step)
        self.data_init_accumulation_steps = int(data_init_accumulation_steps)
        self.dead_atom_revival = bool(dead_atom_revival)
        self.dead_atom_revival_interval = max(1, int(dead_atom_revival_interval))
        self.dead_atom_revival_max_fraction = float(dead_atom_revival_max_fraction)
        self.dead_atom_revival_noise = max(0.0, float(dead_atom_revival_noise))
        self.dead_atom_revival_patience = max(1, int(dead_atom_revival_patience))
        self.omp_compute_precision = str(omp_compute_precision).strip().lower()
        # Fraction of atoms/coefficients randomly replaced before the decoder,
        # training only.  Zero reproduces the previous behaviour exactly.
        self.token_noise_rate = float(token_noise_rate)
        if not 0.0 <= self.token_noise_rate < 1.0:
            raise ValueError(
                f"token_noise_rate must be in [0, 1), got {self.token_noise_rate}"
            )
        if self.omp_compute_precision not in {"float32", "bfloat16"}:
            raise ValueError(
                "omp_compute_precision must be 'float32' or 'bfloat16', got "
                f"{omp_compute_precision!r}"
            )
        self.omp_ridge = float(omp_ridge)
        if not math.isfinite(self.omp_ridge) or self.omp_ridge < 0.0:
            raise ValueError(
                f"omp_ridge must be finite and non-negative, got {omp_ridge!r}"
            )
        self.omp_max_support_coherence = float(omp_max_support_coherence)
        if (
            not math.isfinite(self.omp_max_support_coherence)
            or not 0.0 < self.omp_max_support_coherence <= 1.0
        ):
            raise ValueError(
                "omp_max_support_coherence must be finite and in (0, 1], got "
                f"{omp_max_support_coherence!r}"
            )
        self.dictionary_update_mode = str(dictionary_update_mode).strip().lower()
        if self.dictionary_update_mode not in {"gradient", "alternating_residual"}:
            raise ValueError(
                "dictionary_update_mode must be 'gradient' or "
                f"'alternating_residual', got {dictionary_update_mode!r}"
            )
        self.dictionary_update_relaxation = float(dictionary_update_relaxation)
        self.dictionary_update_max_atoms_per_step = int(
            dictionary_update_max_atoms_per_step
        )
        self.dictionary_update_min_usage = int(dictionary_update_min_usage)
        self.dictionary_update_accumulation_steps = int(
            dictionary_update_accumulation_steps
        )
        self.dictionary_update_max_backtracks = int(dictionary_update_max_backtracks)
        self.dictionary_collective_backend = str(
            dictionary_collective_backend
        ).strip().lower()
        if self.dictionary_collective_backend not in {
            "gloo",
            "default",
            "ddp_buffer",
        }:
            raise ValueError(
                "dictionary_collective_backend must be 'gloo', 'default', or "
                "'ddp_buffer', got "
                f"{dictionary_collective_backend!r}"
            )
        self.coefficient_quantization_bits = int(coefficient_quantization_bits or 0)
        self.coefficient_quantization_max = (
            None
            if coefficient_quantization_max is None
            else float(coefficient_quantization_max)
        )
        self.coefficient_quantization_start_step = int(
            coefficient_quantization_start_step
        )
        self.coefficient_quantization_warmup_steps = int(
            coefficient_quantization_warmup_steps
        )
        self.commitment_normalize_by_variance = bool(
            commitment_normalize_by_variance
        )
        if not 0.0 < self.dictionary_update_relaxation <= 1.0:
            raise ValueError(
                "dictionary_update_relaxation must be in (0, 1], got "
                f"{self.dictionary_update_relaxation}"
            )
        if self.dictionary_update_max_atoms_per_step <= 0:
            raise ValueError(
                "dictionary_update_max_atoms_per_step must be positive, got "
                f"{self.dictionary_update_max_atoms_per_step}"
            )
        if self.dictionary_update_min_usage <= 0:
            raise ValueError(
                "dictionary_update_min_usage must be positive, got "
                f"{self.dictionary_update_min_usage}"
            )
        if self.dictionary_update_accumulation_steps <= 0:
            raise ValueError(
                "dictionary_update_accumulation_steps must be positive, got "
                f"{self.dictionary_update_accumulation_steps}"
            )
        if self.dictionary_update_max_backtracks < 0:
            raise ValueError(
                "dictionary_update_max_backtracks must be non-negative, got "
                f"{self.dictionary_update_max_backtracks}"
            )
        if self.coefficient_quantization_bits < 0 or self.coefficient_quantization_bits == 1:
            raise ValueError(
                "coefficient_quantization_bits must be 0 (disabled) or at least 2, got "
                f"{self.coefficient_quantization_bits}"
            )
        if (
            self.coefficient_quantization_bits > 0
            and (
                self.coefficient_quantization_max is None
                or not math.isfinite(self.coefficient_quantization_max)
                or self.coefficient_quantization_max <= 0.0
            )
        ):
            raise ValueError(
                "coefficient_quantization_max must be finite and positive when coefficient "
                "quantization is enabled"
            )
        if self.coefficient_quantization_start_step < 0:
            raise ValueError(
                "coefficient_quantization_start_step must be non-negative, got "
                f"{self.coefficient_quantization_start_step}"
            )
        if self.coefficient_quantization_warmup_steps < 0:
            raise ValueError(
                "coefficient_quantization_warmup_steps must be non-negative, got "
                f"{self.coefficient_quantization_warmup_steps}"
            )
        if self.data_init_start_step < 0:
            raise ValueError(
                "data_init_start_step must be non-negative, got "
                f"{self.data_init_start_step}"
            )
        if self.data_init_accumulation_steps <= 0:
            raise ValueError(
                "data_init_accumulation_steps must be positive, got "
                f"{self.data_init_accumulation_steps}"
            )

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
        dictionary = torch.randn(self.patch_dim, self.num_embeddings) * 0.02
        if (
            self.dictionary_update_mode == "alternating_residual"
            and self.dictionary_collective_backend == "ddp_buffer"
        ):
            # The alternating update is not differentiated.  Registering its
            # dictionary as a buffer lets DDP broadcast rank zero's completed
            # update at the beginning of the next forward, in DDP's own
            # collective order.  This avoids interleaving ad-hoc collectives
            # with asynchronous gradient reductions after backward.
            self.register_buffer("dictionary", dictionary)
        else:
            self.dictionary = nn.Parameter(
                dictionary,
                requires_grad=self.dictionary_update_mode == "gradient",
            )

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
        self.register_buffer("_dictionary_update_step", torch.zeros((), dtype=torch.long))
        self.register_buffer(
            "_last_dictionary_updated_atom_count", torch.zeros((), dtype=torch.long)
        )
        self.register_buffer(
            "_last_dictionary_update_relaxation", torch.zeros((), dtype=torch.float32)
        )
        self.register_buffer(
            "_last_dictionary_update_relative_improvement",
            torch.zeros((), dtype=torch.float32),
        )
        self.register_buffer(
            "_last_dictionary_update_accumulated_step_count",
            torch.zeros((), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_last_dictionary_update_step",
            torch.zeros((), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_last_coefficient_saturation_fraction",
            torch.zeros((), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_last_coefficient_abs_p99",
            torch.zeros((), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_last_coefficient_abs_p999",
            torch.zeros((), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_last_coefficient_abs_max",
            torch.zeros((), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_last_coefficient_quantization_fraction",
            torch.zeros((), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_last_support_coherence_mean",
            torch.zeros((), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_last_support_coherence_max",
            torch.zeros((), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_last_support_coherence_fallback_fraction",
            torch.zeros((), dtype=torch.float32),
            persistent=False,
        )
        self._revival_candidate_atoms = None
        self._last_dictionary_update_batch = None
        self._dictionary_update_microbatches = []
        # Delayed scratch initialization can pool several latent batches before
        # sampling atoms. This is ordinary process state on purpose: a partial
        # initialization window is cheap to rebuild after a restart.
        self._data_init_accumulator = []
        # Detached per-step batches are accumulated in ordinary Python state.
        # A partial window is intentionally not checkpointed: after resume the
        # first update waits for a complete fresh window instead of applying
        # incomplete sufficient statistics.
        self._dictionary_update_accumulator = []
        # Dictionary statistics are staged through CPU and reduced on Gloo.
        # Keeping this as ordinary Python state prevents checkpoint pickling.
        self._dictionary_process_group = None
        self.normalize_dictionary_()
        self._last_dl_latent_loss = None
        self._last_e_latent_loss = None
        self._last_dictionary_loss = torch.zeros(())
        self._last_final_dictionary_loss = torch.zeros(())
        self._last_commitment_loss = torch.zeros(())
        self._last_dictionary_loss_for_backward = None
        self._last_bottleneck_objective_for_backward = None
        self._last_extra_bottleneck_loss = torch.zeros(())
        self._last_bottleneck_objective = torch.zeros(())
        self._last_bottleneck_loss = torch.zeros(())
        self._last_latent_rms_for_backward = None
        self._last_bottleneck_explained_variance = torch.zeros(())

    def effective_dictionary(self) -> torch.Tensor:
        return self.dictionary

    def _coefficient_quantization_fraction(self) -> float:
        """Return the continuous-to-quantized curriculum fraction."""
        if self.coefficient_quantization_bits <= 0:
            return 0.0
        step = int(self._dictionary_update_step.item())
        start = int(self.coefficient_quantization_start_step)
        if step < start:
            return 0.0
        warmup = int(self.coefficient_quantization_warmup_steps)
        if warmup == 0:
            return 1.0
        return min(max(float(step - start) / float(warmup), 0.0), 1.0)

    def _quantize_coefficients(
        self,
        values: torch.Tensor,
        *,
        record_stats: bool = True,
    ) -> torch.Tensor:
        """Apply the codec's signed coefficient quantizer during stage-one training.

        OMP itself is non-differentiable and runs under ``no_grad``.  Quantizing its
        fitted values here makes both waveform reconstruction and the alternating
        dictionary update see the same coefficients. A curriculum is available for
        scratch training, where an initially low-amplitude random encoder would
        otherwise have every coefficient rounded to zero before it can learn.
        """
        quantization_fraction = self._coefficient_quantization_fraction()
        if record_stats:
            self._last_coefficient_quantization_fraction.fill_(
                quantization_fraction
            )
            finite_for_stats = torch.nan_to_num(
                values.detach().float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            abs_values = finite_for_stats.abs().reshape(-1)
            if abs_values.numel() > 0:
                percentiles = torch.quantile(
                    abs_values, abs_values.new_tensor([0.99, 0.999])
                )
                self._last_coefficient_abs_p99.copy_(
                    percentiles[0].to(
                        device=self._last_coefficient_abs_p99.device,
                        dtype=self._last_coefficient_abs_p99.dtype,
                    )
                )
                self._last_coefficient_abs_p999.copy_(percentiles[1])
                self._last_coefficient_abs_max.copy_(
                    abs_values.max().to(
                        device=self._last_coefficient_abs_max.device,
                        dtype=self._last_coefficient_abs_max.dtype,
                    )
                )
        if self.coefficient_quantization_bits <= 0:
            if record_stats:
                self._last_coefficient_saturation_fraction.zero_()
            return values
        bound = float(self.coefficient_quantization_max)
        qmax = (1 << (int(self.coefficient_quantization_bits) - 1)) - 1
        step = bound / float(qmax)
        finite = torch.nan_to_num(values.float(), nan=0.0, posinf=bound, neginf=-bound)
        if record_stats:
            abs_values = finite.abs().reshape(-1)
            self._last_coefficient_saturation_fraction.copy_(
                (abs_values > bound).float().mean().detach().to(
                    device=self._last_coefficient_saturation_fraction.device,
                    dtype=self._last_coefficient_saturation_fraction.dtype,
                )
            )
        quantized = torch.round(finite.clamp(-bound, bound) / step).mul(step)
        if quantization_fraction <= 0.0:
            return finite
        if quantization_fraction >= 1.0:
            return quantized
        return torch.lerp(finite, quantized, quantization_fraction)

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

    @torch.no_grad()
    def alternating_dictionary_update_after_step_(self, *, optimizer_updated=True) -> int:
        """Update active atoms with accumulated fixed codes after an Adam step.

        Each preceding forward caches detached encoder signals and their OMP
        codes. Several optimizer steps may be accumulated before applying the
        update so large audio dictionaries do not estimate an atom from only a
        handful of frame selections. With those codes fixed, each selected atom
        has a closed-form residual least-squares target. Active atoms are moved
        toward those targets together; a global fixed-code reconstruction check
        backtracks the relaxation when cross-atom interactions increase error.
        """
        # This counter also drives initialization and coefficient curricula in
        # gradient mode. A skipped optimizer step must not advance any schedule
        # or leave stale fixed codes queued for a later dictionary update.
        if self._distributed_is_initialized() and self.dictionary_collective_backend != "ddp_buffer":
            updates = torch.tensor(int(optimizer_updated), device=self.dictionary.device)
            self._dictionary_all_reduce_(updates, op=torch.distributed.ReduceOp.SUM)
            if int(updates) not in (0, torch.distributed.get_world_size()):
                raise RuntimeError("Ranks disagree on whether the optimizer updated")
        if not optimizer_updated:
            self._last_dictionary_update_batch = None
            self._dictionary_update_microbatches.clear()
            self._dictionary_update_accumulator.clear()
            self._last_dictionary_update_accumulated_step_count.zero_()
            return 0
        self._dictionary_update_step.add_(1)
        if self.dictionary_update_mode != "alternating_residual":
            return 0

        # Distributed updates gather every rank's fixed-code batch first, then
        # rank zero performs the closed-form update locally and broadcasts the
        # result.  This deliberately keeps the collective sequence fixed:
        # data-dependent active sets and backtracking must never determine how
        # many collectives an individual rank executes.

        batch = self._last_dictionary_update_batch
        self._last_dictionary_update_batch = None
        microbatches = self._dictionary_update_microbatches
        self._dictionary_update_microbatches = []
        if microbatches:
            batch = {
                key: torch.cat([item[key] for item in microbatches], dim=dim)
                for key, dim in (("signals", 1), ("support", 0), ("values", 0))
            }
        # When scratch training requests delayed data initialization, do not fit
        # the temporary random dictionary to the encoder's nearly constant
        # initialization-time latents. The dictionary is replaced once the
        # continuous reconstruction warmup has made those latents informative.
        if self.data_init_from_first_batch and not bool(self._data_initialized.item()):
            self._dictionary_update_accumulator.clear()
            self._last_dictionary_update_accumulated_step_count.zero_()
            return 0
        # Every rank must make the same decision before the first statistics
        # all-reduce. A missing/malformed local cache used to return here on only
        # that rank, leaving all peers spinning forever in NCCL.
        local_batch_valid = bool(
            isinstance(batch, dict)
            and torch.is_tensor(batch.get("signals"))
            and torch.is_tensor(batch.get("support"))
            and torch.is_tensor(batch.get("values"))
        )
        if local_batch_valid:
            signals = batch["signals"].to(
                device=self.dictionary.device,
                dtype=torch.float32,
            )
            support = batch["support"].to(
                device=self.dictionary.device,
                dtype=torch.long,
            )
            values = batch["values"].to(
                device=self.dictionary.device,
                dtype=torch.float32,
            )
            local_batch_valid = bool(
                signals.ndim == 2
                and support.ndim == 2
                and values.shape == support.shape
                and int(signals.size(0)) == int(self.patch_dim)
                and int(signals.size(1)) == int(support.size(0))
            )

        valid_batch = torch.tensor(
            int(local_batch_valid),
            device=self.dictionary.device,
            dtype=torch.long,
        )
        distributed_buffer_update = bool(
            self._distributed_is_initialized()
            and self.dictionary_collective_backend == "ddp_buffer"
        )
        if self._distributed_is_initialized() and not distributed_buffer_update:
            self._dictionary_all_reduce_(
                valid_batch,
                op=torch.distributed.ReduceOp.MIN,
            )
        if not bool(valid_batch.item()):
            self._dictionary_update_accumulator.clear()
            self._last_dictionary_update_accumulated_step_count.zero_()
            return 0

        self._dictionary_update_accumulator.append(
            {
                "signals": signals,
                "support": support,
                "values": values,
            }
        )
        accumulated_steps = len(self._dictionary_update_accumulator)
        self._last_dictionary_update_accumulated_step_count.fill_(accumulated_steps)
        if accumulated_steps < int(self.dictionary_update_accumulation_steps):
            return 0

        accumulated = self._dictionary_update_accumulator
        self._dictionary_update_accumulator = []
        signals = torch.cat([item["signals"] for item in accumulated], dim=1)
        support = torch.cat([item["support"] for item in accumulated], dim=0)
        values = torch.cat([item["values"] for item in accumulated], dim=0)
        self._last_dictionary_update_accumulated_step_count.zero_()
        # Retain the outcome of this complete update window until the next one.
        # Scalar logging usually has a different cadence from dictionary
        # updates, so clearing these fields on intervening accumulation steps
        # would make every W&B sample incorrectly report zero updates.
        self._last_dictionary_update_step.copy_(self._dictionary_update_step)
        self._last_dictionary_updated_atom_count.zero_()
        self._last_dictionary_update_relaxation.zero_()
        self._last_dictionary_update_relative_improvement.zero_()

        distributed_update = bool(
            self._distributed_is_initialized()
            and self.dictionary_collective_backend != "ddp_buffer"
        )
        update_rank = 0
        if distributed_update:
            signals, support, values = self._gather_dictionary_update_batch_(
                signals,
                support,
                values,
            )

        def finish_update(updated_count: int) -> int:
            if not distributed_update:
                return int(updated_count)
            result = torch.tensor(
                [
                    float(updated_count),
                    float(self._last_dictionary_update_relaxation.item()),
                    float(self._last_dictionary_update_relative_improvement.item()),
                ],
                device=self.dictionary.device,
                dtype=torch.float32,
            )
            if self._distributed_rank() != update_rank:
                result.zero_()
            # Every distributed update window has exactly these two terminal
            # broadcasts, including no-op windows.
            self._dictionary_broadcast_(self.dictionary.data, src=update_rank)
            self._dictionary_broadcast_(result, src=update_rank)
            self._last_dictionary_updated_atom_count.fill_(int(result[0].item()))
            self._last_dictionary_update_relaxation.fill_(float(result[1].item()))
            self._last_dictionary_update_relative_improvement.fill_(
                float(result[2].item())
            )
            return int(result[0].item())

        if (
            self._distributed_is_initialized()
            and self._distributed_rank() != update_rank
        ):
            return finish_update(0)

        dictionary = _normalize_dictionary(
            self.dictionary.detach().float(),
            eps=max(float(self.epsilon), 1e-8),
        )
        support = support.clamp(0, self.num_embeddings - 1)
        atoms = dictionary.t()[support]
        reconstruction = (atoms * values.unsqueeze(-1)).sum(dim=1).t().contiguous()
        residual = torch.nan_to_num(
            signals - reconstruction,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        flat_support = support.reshape(-1)
        flat_values = values.reshape(-1)
        counts = torch.bincount(flat_support[flat_values != 0], minlength=self.num_embeddings).to(
            device=self.dictionary.device,
            dtype=torch.long,
        )
        active = torch.nonzero(
            counts >= int(self.dictionary_update_min_usage),
            as_tuple=False,
        ).flatten()
        if active.numel() == 0:
            return finish_update(0)
        active_counts = counts.index_select(0, active)
        order = torch.argsort(active_counts, descending=True, stable=True)
        active = active.index_select(
            0,
            order[: int(self.dictionary_update_max_atoms_per_step)],
        )

        lookup = torch.full(
            (self.num_embeddings,),
            -1,
            device=self.dictionary.device,
            dtype=torch.long,
        )
        lookup[active] = torch.arange(
            int(active.numel()),
            device=self.dictionary.device,
            dtype=torch.long,
        )
        active_columns = lookup[flat_support]
        selected = active_columns >= 0
        active_columns = active_columns[selected]
        selected_values = flat_values[selected]
        signal_ids = (
            torch.arange(
                int(support.size(0)),
                device=self.dictionary.device,
                dtype=torch.long,
            )
            .unsqueeze(1)
            .expand_as(support)
            .reshape(-1)[selected]
        )

        residual_correlation = torch.zeros(
            self.patch_dim,
            int(active.numel()),
            device=self.dictionary.device,
            dtype=torch.float32,
        )
        coefficient_energy = torch.zeros(
            int(active.numel()),
            device=self.dictionary.device,
            dtype=torch.float32,
        )
        if active_columns.numel() > 0:
            residual_correlation.index_add_(
                1,
                active_columns,
                residual.index_select(1, signal_ids) * selected_values.unsqueeze(0),
            )
            coefficient_energy.index_add_(
                0,
                active_columns,
                selected_values.square(),
            )
        old_atoms = dictionary.index_select(1, active)
        sufficient_statistic = (
            residual_correlation
            + old_atoms * coefficient_energy.unsqueeze(0)
        )
        valid_atoms = (
            coefficient_energy > max(float(self.epsilon), 1e-8)
        ) & (
            sufficient_statistic.norm(dim=0) > max(float(self.epsilon), 1e-8)
        )
        if not bool(valid_atoms.any()):
            return finish_update(0)
        targets = old_atoms.clone()
        targets[:, valid_atoms] = F.normalize(
            sufficient_statistic[:, valid_atoms],
            p=2,
            dim=0,
            eps=max(float(self.epsilon), 1e-8),
        )

        baseline_error = residual.square().sum()
        accepted_atoms = None
        accepted_error = None
        accepted_relaxation = 0.0
        relaxation = float(self.dictionary_update_relaxation)
        tolerance = max(float(baseline_error.item()), 1.0) * 1e-7
        for _ in range(int(self.dictionary_update_max_backtracks) + 1):
            trial_atoms = F.normalize(
                old_atoms.lerp(targets, relaxation),
                p=2,
                dim=0,
                eps=max(float(self.epsilon), 1e-8),
            )
            delta_atoms = trial_atoms - old_atoms
            delta_reconstruction = torch.zeros_like(residual)
            delta_reconstruction.index_add_(
                1,
                signal_ids,
                delta_atoms[:, active_columns] * selected_values.unsqueeze(0),
            )
            trial_error = (residual - delta_reconstruction).square().sum()
            if float(trial_error.item()) <= float(baseline_error.item()) + tolerance:
                accepted_atoms = trial_atoms
                accepted_error = trial_error
                accepted_relaxation = relaxation
                break
            relaxation *= 0.5

        if accepted_atoms is None:
            return finish_update(0)
        self.dictionary.data.index_copy_(
            1,
            active,
            accepted_atoms.to(dtype=self.dictionary.dtype),
        )
        self.normalize_dictionary_()
        updated_count = int(valid_atoms.sum().item())
        self._last_dictionary_updated_atom_count.fill_(updated_count)
        self._last_dictionary_update_relaxation.fill_(accepted_relaxation)
        relative_improvement = (
            (baseline_error - accepted_error)
            / baseline_error.clamp_min(max(float(self.epsilon), 1e-8))
        )
        self._last_dictionary_update_relative_improvement.copy_(
            relative_improvement.to(
                device=self._last_dictionary_update_relative_improvement.device,
                dtype=self._last_dictionary_update_relative_improvement.dtype,
            )
        )
        return finish_update(updated_count)

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

    def _batch_omp_cholesky_with_support(
        self, signals, dictionary, debug=False, return_prefix_values=False,
    ):
        # Respect omp_compute_precision even inside an AMP encoder forward.
        with torch.autocast(device_type=signals.device.type, enabled=False):
            return self._batch_omp_cholesky_impl(
                signals, dictionary, debug, return_prefix_values,
            )

    def _batch_omp_cholesky_impl(
        self,
        signals,
        dictionary,
        debug=False,
        return_prefix_values=False,
    ):
        """Batch OMP using a shared Gram matrix and progressive Cholesky solves."""
        embedding_dim, num_signals = signals.shape
        if int(embedding_dim) != int(dictionary.size(0)):
            raise ValueError(
                f"Signal dim ({int(embedding_dim)}) must match dictionary dim "
                f"({int(dictionary.size(0))})"
            )
        # CUDA does not implement BF16 triangular_solve or cholesky_solve.  In
        # BF16 mode only the large OMP matrix products use BF16; the tiny
        # per-signal Cholesky systems (at most sparsity_level square) and the
        # returned coefficients stay in FP32.
        if self.omp_compute_precision == "bfloat16":
            if signals.is_cuda and not torch.cuda.is_bf16_supported():
                raise RuntimeError(
                    "omp_compute_precision='bfloat16' requires CUDA BF16 support"
                )
            matrix_dtype = torch.bfloat16
            solve_dtype = torch.float32
        else:
            matrix_dtype = signals.dtype
            solve_dtype = signals.dtype

        matrix_dictionary = dictionary.to(dtype=matrix_dtype)
        matrix_signals = signals.to(dtype=matrix_dtype)
        dictionary_t = matrix_dictionary.t()
        gram_matrix = dictionary_t.mm(matrix_dictionary)
        corr_init = dictionary_t.mm(matrix_signals).t()
        gamma = torch.zeros(
            corr_init.shape,
            device=corr_init.device,
            dtype=solve_dtype,
        )

        corr = corr_init
        # Tikhonov regularization and coherence-gated support selection keep the
        # selected Gram systems conditioned. This avoids large cancelling OMP
        # coefficients without imposing any bound on the coefficients themselves.
        ridge = float(self.omp_ridge)
        diagonal = gram_matrix.diagonal().to(solve_dtype) + ridge
        # Gram formation loses precision before triangular solves do. Once a
        # row needs a stable solve, keep it on that path for later prefixes.
        pivot_rtol = 64 * torch.finfo(matrix_dtype).eps
        stable_rows = torch.zeros(num_signals, dtype=torch.bool, device=signals.device)
        previous_values = signals.new_zeros(num_signals, 0, dtype=solve_dtype)
        previous_objective = signals.t().double().square().sum(-1)

        def reconstruction_objective(atoms, coefficients):
            reconstructed = (atoms * coefficients[..., None]).sum(1)
            error = (signals.t().double() - reconstructed.double()).square().sum(-1)
            return error + ridge * coefficients.double().square().sum(-1)

        def stable_solve(atoms, targets):
            # CUDA lstsq only supplies a full-rank solver. SVD also handles
            # repeated/linearly dependent columns and overcomplete supports.
            design = atoms.transpose(1, 2).double()
            u, singular, vh = torch.linalg.svd(design, full_matrices=False)
            if ridge:
                inverse = singular / (singular.square() + ridge)
            else:
                cutoff = singular[..., :1] * max(design.shape[-2:]) * torch.finfo(solve_dtype).eps
                inverse = torch.where(singular > cutoff, singular.reciprocal(), 0.)
            projected = u.transpose(1, 2) @ targets.double().unsqueeze(-1)
            return (vh.transpose(1, 2) @ (inverse[..., None] * projected)).squeeze(-1).to(solve_dtype)
        support = torch.zeros(num_signals, 0, dtype=torch.long, device=signals.device)
        omega = torch.ones_like(corr_init, dtype=torch.bool)
        coherence_eligible = (
            torch.ones_like(omega)
            if self.omp_max_support_coherence < 1.0
            else None
        )
        coherence_fallbacks = 0
        signal_idx = torch.arange(num_signals, device=signals.device)
        prefix_values = [] if return_prefix_values else None

        for k in range(1, int(self.sparsity_level) + 1):
            scores = corr.abs().masked_fill(~omega, -1.0)
            if coherence_eligible is not None and k > 1:
                allowed = omega & coherence_eligible
                has_allowed = allowed.any(dim=1)
                constrained_scores = scores.masked_fill(~allowed, -1.0)
                scores = torch.where(
                    has_allowed.unsqueeze(1),
                    constrained_scores,
                    scores,
                )
                coherence_fallbacks += int((~has_allowed).sum().item())
            next_atoms = torch.argmax(scores, dim=1)
            omega[signal_idx, next_atoms] = False
            if coherence_eligible is not None and k < int(self.sparsity_level):
                selected_correlations = gram_matrix[next_atoms].abs()
                coherence_eligible.logical_and_(
                    selected_correlations
                    <= float(self.omp_max_support_coherence) + 1.0e-6
                )
            expanded_signal_idx = signal_idx.unsqueeze(0).expand(k, num_signals).t()

            candidate_diagonal = diagonal[next_atoms].view(num_signals, 1, 1)
            if k == 1:
                stable_rows |= candidate_diagonal.flatten() <= 0
                L = candidate_diagonal.clamp_min(max(float(self.epsilon), 1e-10)).sqrt()
            else:
                prev_support = support[signal_idx, :]
                new_atoms = next_atoms[expanded_signal_idx[..., :-1]]
                gram_cross = gram_matrix[prev_support, new_atoms].view(
                    num_signals,
                    k - 1,
                    1,
                ).to(dtype=solve_dtype)
                w = torch.linalg.solve_triangular(L, gram_cross, upper=False).view(
                    num_signals,
                    1,
                    k - 1,
                )
                pivot = candidate_diagonal - (w**2).sum(dim=2, keepdim=True)
                stable_rows |= (~torch.isfinite(pivot) | (pivot <= pivot_rtol * candidate_diagonal)).flatten()
                bottom_right = pivot.clamp_min(max(float(self.epsilon), 1e-10)).sqrt()
                zeros = torch.zeros(
                    num_signals,
                    k - 1,
                    1,
                    device=signals.device,
                    dtype=solve_dtype,
                )
                L = torch.cat(
                    (
                        torch.cat((L, zeros), dim=2),
                        torch.cat((w, bottom_right), dim=2),
                    ),
                    dim=1,
                )

            # Invalid Cholesky factors must never reach a solve, even for rows
            # whose result will be replaced. Their actual solve uses SVD below.
            L = torch.where(stable_rows[:, None, None], torch.eye(k, device=L.device, dtype=L.dtype), L)

            support = torch.cat([support, next_atoms.unsqueeze(1)], dim=1)
            corr_active = corr_init[expanded_signal_idx, support[signal_idx, :]].view(
                num_signals,
                k,
                1,
            ).to(dtype=solve_dtype)
            gamma_active = torch.cholesky_solve(corr_active, L).squeeze(-1)
            atoms = dictionary.t()[support].to(solve_dtype)
            objective = reconstruction_objective(atoms, gamma_active)
            tolerance = previous_objective.clamp_min(1.) * (32 * torch.finfo(solve_dtype).eps)
            stable_rows |= (~torch.isfinite(objective)) | (objective > previous_objective + tolerance)
            if bool(stable_rows.any()):
                gamma_active[stable_rows] = stable_solve(atoms[stable_rows], signals.t()[stable_rows])
                objective = reconstruction_objective(atoms, gamma_active)
            rejected = (~torch.isfinite(objective)) | (objective > previous_objective + tolerance)
            # A rejected candidate is an inactive slot (zero coefficient).
            # Keeping the prior prefix also handles precision loss on casting
            # a stable double-precision solution back to the training dtype.
            padded_previous = F.pad(previous_values, (0, 1))
            gamma_active = torch.where(rejected[:, None], padded_previous, gamma_active)
            previous_values = gamma_active.clone()
            previous_objective = reconstruction_objective(atoms, gamma_active)
            gamma[signal_idx.unsqueeze(1), support[signal_idx]] = gamma_active
            if prefix_values is not None:
                prefix_values.append(gamma_active.clone())

            active = gamma[signal_idx.unsqueeze(1), support[signal_idx]]
            beta = active.to(dtype=matrix_dtype).unsqueeze(1).bmm(
                gram_matrix[support[signal_idx], :]
            ).squeeze(1)
            corr = corr_init - beta
            if bool(stable_rows.any()):
                residual = signals.t()[stable_rows].double() - (
                    atoms[stable_rows].double() * gamma_active[stable_rows, :, None].double()
                ).sum(1)
                corr[stable_rows] = (residual @ dictionary.double()).to(corr.dtype)

            if debug:
                residual_proxy = corr.abs().amax(dim=1).max()
                print(f"Step {k}, max residual correlation: {float(residual_proxy):.4f}")

        values = gamma[signal_idx.unsqueeze(1), support[signal_idx]]
        if int(self.sparsity_level) > 1:
            depth = int(self.sparsity_level)
            rows = support.unsqueeze(2).expand(-1, -1, depth)
            cols = support.unsqueeze(1).expand(-1, depth, -1)
            support_gram = gram_matrix[rows, cols].abs().float()
            pair_mask = torch.triu(
                torch.ones(
                    depth,
                    depth,
                    device=support.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            pair_coherence = support_gram[:, pair_mask]
            self._last_support_coherence_mean.copy_(
                pair_coherence.mean().detach().to(
                    device=self._last_support_coherence_mean.device,
                    dtype=self._last_support_coherence_mean.dtype,
                )
            )
            self._last_support_coherence_max.copy_(
                pair_coherence.max().detach().to(
                    device=self._last_support_coherence_max.device,
                    dtype=self._last_support_coherence_max.dtype,
                )
            )
            denominator = max(num_signals * (depth - 1), 1)
            self._last_support_coherence_fallback_fraction.fill_(
                float(coherence_fallbacks) / float(denominator)
            )
        else:
            self._last_support_coherence_mean.zero_()
            self._last_support_coherence_max.zero_()
            self._last_support_coherence_fallback_fraction.zero_()
        if prefix_values is not None:
            return support, values, gamma.t(), tuple(prefix_values)
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

    def batch_omp_with_support_and_prefixes(self, X, D):
        """Return final OMP codes and re-fitted coefficients at every depth."""
        self._validate_omp_inputs(X, D)
        X = torch.nan_to_num(X)
        D = torch.nan_to_num(D)
        support, values, _, prefix_values = self._batch_omp_cholesky_with_support(
            X,
            D,
            return_prefix_values=True,
        )
        values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        prefix_values = tuple(
            torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            for value in prefix_values
        )
        return support, values, prefix_values

    def _distributed_is_initialized(self):
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    def prepare_distributed_process_group_(self):
        """Create the CPU/Gloo group used for dictionary-maintenance statistics."""
        # Check torch.distributed directly here. Some unit tests deliberately
        # replace ``_distributed_is_initialized`` while recording collectives.
        if not (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            return None
        if self.dictionary_collective_backend == "ddp_buffer":
            return None
        if self.dictionary_collective_backend == "default":
            return torch.distributed.group.WORLD
        if self._dictionary_process_group is None:
            world_size = int(torch.distributed.get_world_size())
            self._dictionary_process_group = torch.distributed.new_group(
                ranks=list(range(world_size)),
                backend="gloo",
            )
        return self._dictionary_process_group

    def _dictionary_group(self):
        if not self._distributed_is_initialized():
            return None
        return self.prepare_distributed_process_group_()

    @staticmethod
    def _dictionary_collective_fence_(tensor):
        """Finish local CUDA work after every rank reaches the Gloo gate.

        The gate is intentionally in the caller.  Synchronizing CUDA before all
        ranks have returned from DDP backward can deadlock: one rank waits for a
        pending NCCL reduction while another has already entered a CPU
        collective.  Once the Gloo gate has been crossed, every rank has
        enqueued its DDP work and a local CUDA synchronization is safe.
        """
        if torch.is_tensor(tensor) and tensor.is_cuda:
            torch.cuda.synchronize(device=tensor.device)

    @staticmethod
    def _dictionary_collective_entry_gate_(group):
        """Make reaching dictionary maintenance a CPU-only rendezvous."""
        torch.distributed.barrier(group=group)

    def _dictionary_collective_exit_fence_(self, tensor, group):
        """Keep every rank out of NCCL until staged Gloo results reach CUDA.

        A Gloo collective may return on one rank while another rank is still
        copying its reduced CPU tensor back to the GPU.  If the faster rank
        immediately enters DDP's next-forward NCCL broadcast, that broadcast
        can occupy the CUDA stream needed by the slower rank's copy and wedge
        both process groups.  Synchronize the copy locally, then use the Gloo
        group as an exit gate before returning to DDP.
        """
        self._dictionary_collective_fence_(tensor)
        torch.distributed.barrier(group=group)

    def _dictionary_all_reduce_(self, tensor, *, op):
        if self.dictionary_collective_backend == "ddp_buffer":
            return
        group = self._dictionary_group()
        if group is None or self.dictionary_collective_backend == "default":
            torch.distributed.all_reduce(tensor, op=op)
            return
        self._dictionary_collective_entry_gate_(group)
        self._dictionary_collective_fence_(tensor)
        staged = tensor.detach().to(device="cpu")
        torch.distributed.all_reduce(staged, op=op, group=group)
        tensor.copy_(staged.to(device=tensor.device))
        self._dictionary_collective_exit_fence_(tensor, group)

    def _dictionary_all_gather_(self, outputs, tensor):
        if self.dictionary_collective_backend == "ddp_buffer":
            return
        group = self._dictionary_group()
        if group is None or self.dictionary_collective_backend == "default":
            torch.distributed.all_gather(outputs, tensor)
            return
        self._dictionary_collective_entry_gate_(group)
        self._dictionary_collective_fence_(tensor)
        staged = tensor.detach().to(device="cpu")
        staged_outputs = [torch.empty_like(staged) for _ in outputs]
        torch.distributed.all_gather(staged_outputs, staged, group=group)
        for output, gathered in zip(outputs, staged_outputs):
            output.copy_(gathered.to(device=output.device))
        self._dictionary_collective_exit_fence_(tensor, group)

    def _dictionary_broadcast_(self, tensor, *, src):
        if self.dictionary_collective_backend == "ddp_buffer":
            return
        group = self._dictionary_group()
        if group is None or self.dictionary_collective_backend == "default":
            torch.distributed.broadcast(tensor, src=src)
            return
        self._dictionary_collective_entry_gate_(group)
        self._dictionary_collective_fence_(tensor)
        staged = tensor.detach().to(device="cpu")
        torch.distributed.broadcast(staged, src=src, group=group)
        tensor.copy_(staged.to(device=tensor.device))
        self._dictionary_collective_exit_fence_(tensor, group)

    @torch.no_grad()
    def _gather_dictionary_update_batch_(self, signals, support, values):
        """Gather a variable-width fixed-code batch in a fixed collective order."""
        if not self._distributed_is_initialized():
            return signals, support, values

        world_size = int(torch.distributed.get_world_size())
        local_columns = torch.tensor(
            [int(signals.size(1))],
            device=signals.device,
            dtype=torch.long,
        )
        gathered_sizes = [torch.empty_like(local_columns) for _ in range(world_size)]
        self._dictionary_all_gather_(gathered_sizes, local_columns)
        sizes = [int(item.item()) for item in gathered_sizes]
        max_columns = max(sizes, default=0)
        if max_columns <= 0:
            raise RuntimeError("distributed dictionary update gathered no signals")

        pad_columns = max_columns - int(signals.size(1))
        if pad_columns > 0:
            signals = F.pad(signals, (0, pad_columns))
            support = F.pad(support, (0, 0, 0, pad_columns))
            values = F.pad(values, (0, 0, 0, pad_columns))

        gathered_signals = [torch.empty_like(signals) for _ in range(world_size)]
        gathered_support = [torch.empty_like(support) for _ in range(world_size)]
        gathered_values = [torch.empty_like(values) for _ in range(world_size)]
        self._dictionary_all_gather_(gathered_signals, signals)
        self._dictionary_all_gather_(gathered_support, support)
        self._dictionary_all_gather_(gathered_values, values)

        global_signals = torch.cat(
            [item[:, :size] for item, size in zip(gathered_signals, sizes)],
            dim=1,
        )
        global_support = torch.cat(
            [item[:size] for item, size in zip(gathered_support, sizes)],
            dim=0,
        )
        global_values = torch.cat(
            [item[:size] for item, size in zip(gathered_values, sizes)],
            dim=0,
        )
        return global_signals, global_support, global_values

    def _distributed_rank(self):
        if not self._distributed_is_initialized():
            return 0
        return int(torch.distributed.get_rank())

    @torch.no_grad()
    def _broadcast_dictionary_(self):
        if self._distributed_is_initialized():
            self._dictionary_broadcast_(self.dictionary.data, src=0)

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
        self._dictionary_all_gather_(counts, count)
        counts = torch.cat(counts, dim=0)
        max_count = int(counts.max().item())
        if max_count <= 0:
            return local[:, :0]
        if int(local.size(1)) < max_count:
            pad = local.new_zeros((int(local.size(0)), max_count - int(local.size(1))))
            local = torch.cat([local, pad], dim=1)

        gathered = [torch.empty_like(local) for _ in range(torch.distributed.get_world_size())]
        self._dictionary_all_gather_(gathered, local.contiguous())
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
    def _record_atom_usage_(self, support: torch.Tensor, signals: torch.Tensor, values=None) -> None:
        if not self.training or not self.dead_atom_revival:
            return
        if support.numel() == 0:
            return

        support_flat = support.detach().reshape(-1).to(torch.long)
        if values is not None:
            support_flat = support_flat[values.detach().reshape(-1) != 0]
        counts = torch.bincount(support_flat, minlength=self.num_embeddings).to(
            device=self._atom_usage_window.device,
            dtype=self._atom_usage_window.dtype,
        )
        if self._distributed_is_initialized():
            self._dictionary_all_reduce_(counts, op=torch.distributed.ReduceOp.SUM)
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
        if int(self._dictionary_update_step.item()) < int(self.data_init_start_step):
            return
        world_size = (
            int(torch.distributed.get_world_size())
            if self._distributed_is_initialized()
            else 1
        )
        max_local_columns = max(
            2048,
            int(math.ceil(float(self.num_embeddings) / float(world_size))),
        )
        local = signals.detach()
        if int(local.size(1)) > max_local_columns:
            idx = torch.linspace(
                0,
                int(local.size(1)) - 1,
                steps=max_local_columns,
                device=local.device,
            ).round().to(torch.long)
            local = local.index_select(1, idx)
        self._data_init_accumulator.append(local)
        if len(self._data_init_accumulator) < int(self.data_init_accumulation_steps):
            return
        local = torch.cat(self._data_init_accumulator, dim=1)
        self._data_init_accumulator = []
        # The cap applies to the pooled window, not each individual forward.
        # Scale it with the requested window so a 2-step, 8-GPU audio launch can
        # contribute more than 8,192 distinct 75 Hz latent frames globally.
        pooled_local_columns = max_local_columns * int(self.data_init_accumulation_steps)
        atoms = self._signal_atoms(
            self._all_gather_signal_columns(
                local,
                max_local_columns=pooled_local_columns,
            ),
            self.num_embeddings,
        )
        if atoms is not None:
            self._last_dictionary_update_batch = None
            self._dictionary_update_microbatches.clear()
            self._dictionary_update_accumulator.clear()
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

    def _corrupt_tokens(self, support, values):
        """Replace a fraction of atoms and coefficients with random valid ones."""
        rate = float(self.token_noise_rate)
        support_out, values_out = support.clone(), values.clone()
        atom_mask = torch.rand(support.shape, device=support.device) < rate
        if atom_mask.any():
            replacement = torch.randint(
                0, int(self.num_embeddings), (int(atom_mask.sum()),),
                device=support.device, dtype=support.dtype,
            )
            support_out[atom_mask] = replacement
        value_mask = torch.rand(values.shape, device=values.device) < rate
        if value_mask.any():
            # Uniform over each depth's observed coefficient range, which is the
            # value-space equivalent of picking a random coefficient bin.
            scale = values.abs().amax(dim=(0, 1, 2), keepdim=True).clamp_min(1e-6)
            draw = (torch.rand(values.shape, device=values.device) * 2.0 - 1.0) * scale
            values_out = torch.where(value_mask, draw.to(values.dtype), values_out)
        return support_out, values_out

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
            if self.progressive_loss:
                support_flat, values, prefix_values = (
                    self.batch_omp_with_support_and_prefixes(signals, dictionary)
                )
            else:
                support_flat, values = self.batch_omp_with_support(signals, dictionary)
                prefix_values = None
        values = self._quantize_coefficients(values)
        if prefix_values is not None:
            prefix_values = tuple(
                self._quantize_coefficients(prefix_value, record_stats=False)
                for prefix_value in prefix_values
            )
        support = support_flat.view(batch_size, grid_h, grid_w, self.sparsity_level)
        values = values.view(batch_size, grid_h, grid_w, self.sparsity_level).float()
        if self.training and self.dictionary_update_mode == "alternating_residual":
            self._last_dictionary_update_batch = {
                "signals": signals.detach(),
                "support": support_flat.detach(),
                "values": values.detach().reshape(-1, self.sparsity_level),
            }
            self._dictionary_update_microbatches.append(self._last_dictionary_update_batch)
        self._record_atom_usage_(support, signals, values)

        # Token-error augmentation.  Corrupting 10% of tokens costs this
        # tokenizer +27.4 rFID where the reference RQ-VAE loses only +1.0: its
        # residual codes let later depths correct an earlier mistake, while a
        # wrong LASER atom swaps a dictionary element outright.  Training the
        # decoder on corrupted supports and coefficients -- the same uniform
        # random replacement the sensitivity measurement uses -- asks it to
        # tolerate the errors the stage-2 prior will actually make.  Applied
        # only in training, and only to what the decoder sees: the dictionary
        # and commitment objectives below keep the clean codes.
        decoder_support, decoder_values = support, values
        if self.training and self.token_noise_rate > 0:
            decoder_support, decoder_values = self._corrupt_tokens(support, values)

        z_dl = self._reconstruct_sparse(support, values, latent_h, latent_w).float()
        z_decoder = (
            z_dl
            if decoder_support is support
            else self._reconstruct_sparse(
                decoder_support, decoder_values, latent_h, latent_w
            ).float()
        )
        final_dictionary_loss = F.mse_loss(z_dl, z_e_work.detach())
        final_commitment_loss = F.mse_loss(z_dl.detach(), z_e_work)
        self._last_latent_rms_for_backward = z_e_work.square().mean().clamp_min(1e-12).sqrt()
        target_variance = z_e_work.detach().var(unbiased=False).clamp_min(1e-12)
        if self.commitment_normalize_by_variance:
            final_commitment_loss = final_commitment_loss / target_variance
        self._last_bottleneck_explained_variance = (
            1.0 - final_dictionary_loss.detach() / target_variance
        )
        if self.progressive_loss:
            signal_targets = signals.t()
            dictionary_t = dictionary.t()
            dictionary_losses = []
            commitment_losses = []
            for depth, prefix_coefficients in enumerate(prefix_values, start=1):
                prefix_atoms = dictionary_t[support_flat[:, :depth]]
                prefix_reconstruction = (
                    prefix_atoms * prefix_coefficients.unsqueeze(-1)
                ).sum(dim=1)
                dictionary_losses.append(
                    F.mse_loss(prefix_reconstruction, signal_targets.detach())
                )
                commitment_losses.append(
                    F.mse_loss(prefix_reconstruction.detach(), signal_targets)
                    / (
                        target_variance
                        if self.commitment_normalize_by_variance
                        else target_variance.new_ones(())
                    )
                )
            dl_latent_loss = torch.stack(dictionary_losses).mean()
            e_latent_loss = torch.stack(commitment_losses).mean()
        else:
            dl_latent_loss = final_dictionary_loss
            e_latent_loss = final_commitment_loss
        dictionary_loss = dl_latent_loss
        commitment_loss = float(self.commitment_cost) * e_latent_loss
        bottleneck_loss = commitment_loss
        # In alternating mode the dictionary objective is optimized by the
        # explicit fixed-code residual step, not autograd. Keeping it out of
        # the backbone loss also makes LASER's encoder commitment numerically
        # match RQ-VAE's depth-averaged commitment objective.
        if self.dictionary_update_mode == "alternating_residual":
            objective = bottleneck_loss
        else:
            objective = dictionary_loss + bottleneck_loss

        self._last_dl_latent_loss = dl_latent_loss.detach()
        self._last_e_latent_loss = e_latent_loss.detach()
        self._last_dictionary_loss = dictionary_loss.detach()
        self._last_final_dictionary_loss = final_dictionary_loss.detach()
        self._last_commitment_loss = commitment_loss.detach()
        self._last_dictionary_loss_for_backward = dictionary_loss
        self._last_bottleneck_objective_for_backward = objective
        self._last_extra_bottleneck_loss = z_e_work.new_zeros(())
        self._last_bottleneck_loss = bottleneck_loss.detach()
        self._last_bottleneck_objective = objective.detach()

        z_dl_value = z_dl.to(dtype=z_e.dtype)
        z_dl = z_e + (z_dl_value - z_e).detach()
        # Preserve the sparse (and optionally token-corrupted) forward value
        # while routing reconstruction/perceptual gradients back to the
        # encoder.  Returning the raw ``z_decoder`` here silently detached the
        # encoder from every decoder-side objective because Batch OMP runs
        # under no_grad; only the commitment loss could then train it.
        z_decoder_value = z_decoder.to(dtype=z_e.dtype)
        z_decoder = z_e + (z_decoder_value - z_e).detach()
        sparse_codes = SparseCodes(
            support=support,
            values=values,
            num_embeddings=self.num_embeddings,
        )
        self._last_sparse_codes_for_visualization = SparseCodes(
            support=support.detach(),
            values=values.detach(),
            num_embeddings=self.num_embeddings,
        )
        # The decoder sees the (optionally corrupted) latent; every objective
        # above was computed against the clean reconstruction.
        return z_decoder, bottleneck_loss, sparse_codes
