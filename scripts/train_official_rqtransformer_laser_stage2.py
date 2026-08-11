#!/usr/bin/env python3
"""Train the vendored KakaoBrain RQ-Transformer on LASER sparse pairs.

The transformer implementation lives in ``src.models.rqtransformer``.
Only the stage-1 auxiliary embedding is adapted: OMP atom ids use the learned
LASER dictionary supports remain discrete while real coefficients are mapped
through either a uniform grid or cache-provided nonuniform scalar centers.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import make_dataclass
from datetime import timedelta
from functools import partial
import json
import math
import os
from pathlib import Path
import sys
import shutil
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    OptimStateKeyType,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "rq-vae-transformer"))

from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.nodes import AnyNode, BooleanNode, BytesNode, FloatNode, IntegerNode, StringNode
from src.models.rqtransformer.configs import RQTransformerConfig
from src.models.rqtransformer.attentions import AttentionBlock, AttentionStack
from src.models.rqtransformer.transformers import RQTransformer, sample_from_logits
from rqvae.models.rqvae.rqvae import RQVAE
from src.data.imagenet_labels import class_names_for_dataset


def load_stage1_checkpoint(path: Path):
    """Load trusted tensor/config state without importing the legacy rqvae package."""
    from typing import Any

    dist_env = make_dataclass(
        "DistEnv",
        [
            ("world_size", int), ("world_rank", int), ("local_rank", int),
            ("num_gpus", int), ("master", bool), ("device_name", str),
        ],
    )
    dist_env.__module__ = "rqvae.utils.dist"
    torch.serialization.add_safe_globals([
        Any, list, dict, tuple, set, int, str, float, bool, bytes, defaultdict,
        DictConfig, ListConfig, ContainerMetadata, Metadata, AnyNode, BooleanNode,
        BytesNode, FloatNode, IntegerNode, StringNode, dist_env,
    ])
    return torch.load(path, map_location="cpu", weights_only=True)


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def uses_inception_score(dataset: str) -> bool:
    """Inception Score is only part of the ImageNet generation protocol."""
    return dataset == "imagenet"


def unwrap(model):
    return model.module if isinstance(model, (DDP, FSDP)) else model


def is_fsdp_model(model) -> bool:
    return isinstance(model, FSDP)


def wrap_distributed_model(model, backend: str, device: torch.device, world_size: int):
    """Wrap the trainable stage-2 model without touching the frozen stage-1 model."""
    if world_size <= 1:
        if backend == "fsdp":
            raise ValueError("--distributed-backend fsdp requires more than one process")
        return model
    if backend == "ddp":
        return DDP(model, device_ids=[device.index], broadcast_buffers=False)
    if backend != "fsdp":
        raise ValueError(f"unsupported distributed backend: {backend}")
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={AttentionBlock},
    )
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=device,
        sync_module_states=True,
        limit_all_gathers=True,
    )


def optimizer_state_uses_names(optimizer_state) -> bool:
    state = optimizer_state.get("state", {})
    if state:
        return isinstance(next(iter(state)), str)
    groups = optimizer_state.get("param_groups", [])
    return bool(groups and groups[0].get("params") and isinstance(groups[0]["params"][0], str))


def optimizer_state_to_names(optimizer_state, unwrapped_model):
    """Convert a legacy DDP optimizer checkpoint to FSDP's full named format."""
    if optimizer_state_uses_names(optimizer_state):
        return optimizer_state
    return FSDP.rekey_optim_state_dict(
        optimizer_state,
        OptimStateKeyType.PARAM_NAME,
        unwrapped_model,
    )


def optimizer_state_to_ids(optimizer_state, parameter_names):
    """Return the standard ID-keyed optimizer format consumed by non-FSDP AdamW."""
    if not optimizer_state_uses_names(optimizer_state):
        return optimizer_state
    name_to_id = {name: index for index, name in enumerate(parameter_names)}
    unknown = set(optimizer_state.get("state", {})) - set(name_to_id)
    for group in optimizer_state.get("param_groups", []):
        unknown.update(name for name in group.get("params", []) if name not in name_to_id)
    if unknown:
        preview = ", ".join(sorted(unknown)[:5])
        raise KeyError(f"optimizer checkpoint contains unknown parameters: {preview}")
    converted = dict(optimizer_state)
    converted["state"] = {
        name_to_id[name]: value for name, value in optimizer_state.get("state", {}).items()
    }
    converted_groups = []
    for group in optimizer_state.get("param_groups", []):
        group_names = set(group.get("params", []))
        ordered_ids = [name_to_id[name] for name in parameter_names if name in group_names]
        converted_groups.append({**group, "params": ordered_ids})
    converted["param_groups"] = converted_groups
    return converted


def optimizer_state_for_unwrapped_load(optimizer_state, unwrapped_model):
    if not optimizer_state_uses_names(optimizer_state):
        return optimizer_state
    return FSDP.rekey_optim_state_dict(
        optimizer_state,
        OptimStateKeyType.PARAM_ID,
        unwrapped_model,
    )


def full_checkpoint_states(model, optimizer, parameter_names):
    """Gather one DDP-compatible full checkpoint on rank zero only."""
    if not is_fsdp_model(model):
        if rank() != 0:
            return None, None
        return unwrap(model).state_dict(), optimizer.state_dict()
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        model_state = model.state_dict()
        optimizer_state = FSDP.optim_state_dict(model, optimizer)
    if rank() == 0:
        optimizer_state = optimizer_state_to_ids(optimizer_state, parameter_names)
    return model_state, optimizer_state


@contextmanager
def model_for_custom_methods(model):
    """Expose sampling methods safely while FSDP parameters are sharded."""
    if is_fsdp_model(model):
        with FSDP.summon_full_params(
            model, recurse=True, writeback=False, rank0_only=False, offload_to_cpu=False
        ):
            # Cached autoregressive methods are not routed through FSDP.forward().
            # Once all parameters are summoned, temporarily expose each wrapped
            # block's underlying module. Otherwise ordinary calls made by the
            # coefficient micro-transformer reshard a nested block mid-context.
            replacements = []

            def expose_unwrapped_children(module):
                for name, child in list(module.named_children()):
                    if isinstance(child, FSDP):
                        replacements.append((module, name, child))
                        module._modules[name] = unwrap(child)
                        expose_unwrapped_children(module._modules[name])
                    else:
                        expose_unwrapped_children(child)

            exposed_model = unwrap(model)
            expose_unwrapped_children(exposed_model)
            try:
                yield exposed_model
            finally:
                for parent, name, wrapper in reversed(replacements):
                    parent._modules[name] = wrapper
    else:
        yield unwrap(model)


def cuda_memory_report(device: torch.device, phase: str):
    """Collect peak CUDA usage from every rank and print it once on rank zero."""
    torch.cuda.synchronize(device)
    local = torch.tensor(
        [torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_initialized():
        gathered = [torch.zeros_like(local) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, local)
    else:
        gathered = [local]
    values = [(float(item[0]) / 2**30, float(item[1]) / 2**30) for item in gathered]
    if rank() == 0:
        summary = "; ".join(
            f"rank {index}: allocated={allocated:.2f} GiB, reserved={reserved:.2f} GiB"
            for index, (allocated, reserved) in enumerate(values)
        )
        print(f"CUDA memory ({phase}) — {summary}", flush=True)
    return values


def move_optimizer_state(optimizer, device):
    """Move Adam state tensors without changing parameter or group identity."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


@contextmanager
def optimizer_state_offloaded_for_generation(model, optimizer, device):
    """Free optimizer memory while autoregressive sampling needs a large batch."""
    move_optimizer_state(optimizer, torch.device("cpu"))
    torch.cuda.empty_cache()
    try:
        yield
    finally:
        move_optimizer_state(optimizer, device)


def persistent_checkpoint_dir(output: Path, configured: Path | None) -> Path:
    """Resolve checkpoint storage and reject ephemeral/out-of-workspace targets."""
    target = (configured or (output / "checkpoints")).expanduser().resolve()
    workspace = Path("/workspace")
    resolved_workspace = workspace.resolve()
    output_uses_workspace = output.expanduser().resolve().is_relative_to(resolved_workspace)
    if (workspace.is_dir() or output_uses_workspace) and not target.is_relative_to(resolved_workspace):
        raise ValueError(
            f"Checkpoint directory must be under /workspace so it survives restarts; got {target}"
        )
    return target


def create_cosine_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    initial_lr: float,
    min_lr: float,
    total_steps: int,
    completed_steps: int = 0,
    state_dict=None,
):
    """Create the original RQ-Transformer stepwise cosine LR schedule.

    Legacy checkpoints do not contain scheduler state.  In that case,
    ``completed_steps`` places the scheduler directly at the matching point on
    the global curve, so relaunching for another few epochs cannot restart it.
    """
    if total_steps <= 0:
        raise ValueError("cosine schedule requires a positive number of total steps")
    if not 0 <= completed_steps <= total_steps:
        raise ValueError(
            f"completed optimizer steps ({completed_steps}) must be in [0, {total_steps}]"
        )
    for param_group in optimizer.param_groups:
        param_group["initial_lr"] = float(initial_lr)

    if state_dict is None:
        # _LRScheduler performs its initial step in the constructor.  Seeding
        # last_epoch one step behind lands exactly on ``completed_steps``.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(total_steps),
            eta_min=float(min_lr),
            last_epoch=int(completed_steps) - 1,
        )
        # PyTorch releases differ on whether the constructor immediately
        # writes the closed-form LR for a nonzero last_epoch. Make the resumed
        # legacy-checkpoint position explicit and version-independent.
        progress = float(completed_steps) / float(total_steps)
        resumed_lr = float(min_lr) + 0.5 * (float(initial_lr) - float(min_lr)) * (
            1.0 + math.cos(math.pi * progress)
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = resumed_lr
        scheduler._last_lr = [resumed_lr for _ in optimizer.param_groups]
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(total_steps), eta_min=float(min_lr)
        )
        scheduler.load_state_dict(state_dict)
        if scheduler.last_epoch != completed_steps:
            raise ValueError(
                "scheduler/checkpoint step mismatch: "
                f"scheduler={scheduler.last_epoch}, checkpoint={completed_steps}"
            )
        if scheduler.T_max != total_steps or scheduler.eta_min != min_lr:
            raise ValueError(
                "resumed scheduler settings differ from this launch: "
                f"checkpoint T_max={scheduler.T_max}, eta_min={scheduler.eta_min}; "
                f"launch T_max={total_steps}, eta_min={min_lr}"
            )
    return scheduler


class ResumableDistributedSampler(DistributedSampler):
    """Distributed sampler that starts at a saved per-rank sample cursor."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_index = 0

    def set_start_index(self, start_index: int):
        self.start_index = max(0, int(start_index))

    def __iter__(self):
        return iter(list(super().__iter__())[self.start_index:])

    def __len__(self):
        return max(0, super().__len__() - self.start_index)


class LaserAux(nn.Module):
    """Frozen stage-1 encoder and sparse dictionary expected by RQTransformer."""

    def __init__(self, checkpoint: Path, num_atoms: int, coeff_vocab_size: int,
                 coeff_max: float, coeff_scale: float = 1.0,
                 attn_resolutions=(8,), coeff_scales=None,
                 soft_target_physical=False, clamp_coeffs=True,
                 coeff_bin_centers=None, sparsity_level: int = 2):
        super().__init__()
        self.sparsity_level = int(sparsity_level)
        if self.sparsity_level <= 0:
            raise ValueError("sparsity_level must be positive")
        stage1 = RQVAE(
            embed_dim=256,
            n_embed=num_atoms,
            decay=0.99,
            loss_type="mse",
            latent_loss_weight=0.25,
            bottleneck_type="rq",
            ddconfig=dict(
                double_z=False, z_channels=256, resolution=256, in_channels=3,
                out_ch=3, ch=128, ch_mult=[1, 1, 2, 2, 4, 4],
                num_res_blocks=2, attn_resolutions=list(attn_resolutions), dropout=0.0,
            ),
            latent_shape=[8, 8, 256], code_shape=[8, 8, self.sparsity_level],
            shared_codebook=True, restart_unused_codes=True,
        )
        payload = load_stage1_checkpoint(checkpoint)
        state = payload["state_dict"]
        # Accept both native RQ-VAE checkpoints and maintained LASER Lightning
        # checkpoints.  The latter uses descriptive projection/bottleneck names.
        filtered = {}
        for key, value in state.items():
            if key.startswith(("encoder.", "decoder.", "quant_conv.", "post_quant_conv.")):
                filtered[key] = value
            elif key.startswith("pre_bottleneck."):
                filtered["quant_conv." + key.removeprefix("pre_bottleneck.")] = value
            elif key.startswith("post_bottleneck."):
                filtered["post_quant_conv." + key.removeprefix("post_bottleneck.")] = value
        missing, unexpected = stage1.load_state_dict(filtered, strict=False)
        bad_missing = [k for k in missing if not k.startswith("quantizer.")]
        if bad_missing or unexpected:
            raise RuntimeError(f"stage-1 mismatch: missing={bad_missing}, unexpected={unexpected}")
        self.encoder = stage1.encoder
        self.quant_conv = stage1.quant_conv
        self.post_quant_conv = stage1.post_quant_conv
        self.decoder = stage1.decoder
        dictionary_key = (
            "quantizer.dictionary"
            if "quantizer.dictionary" in state
            else "bottleneck.dictionary"
        )
        if dictionary_key not in state:
            raise RuntimeError("stage-1 checkpoint has no sparse dictionary")
        self.register_buffer("dictionary", F.normalize(state[dictionary_key].float(), dim=0))
        if coeff_bin_centers is None:
            coefficient_bins = torch.linspace(-coeff_max, coeff_max, coeff_vocab_size)
        else:
            coefficient_bins = torch.as_tensor(coeff_bin_centers, dtype=torch.float32)
            if coefficient_bins.ndim != 1 or coefficient_bins.numel() != coeff_vocab_size:
                raise ValueError(
                    "custom coefficient-bin centers must be a one-dimensional "
                    f"sequence of length {coeff_vocab_size}"
                )
            if not torch.isfinite(coefficient_bins).all():
                raise ValueError("custom coefficient-bin centers must be finite")
            if not torch.all(coefficient_bins[1:] > coefficient_bins[:-1]):
                raise ValueError("custom coefficient-bin centers must be strictly increasing")
        self.register_buffer("coeff_bins", coefficient_bins)
        self.num_atoms = int(num_atoms)
        self.coeff_vocab_size = int(coeff_vocab_size)
        self.vocab_size = self.num_atoms + self.coeff_vocab_size
        self.coeff_max = float(coeff_max)
        scales = (
            coeff_scales
            if coeff_scales is not None
            else [coeff_scale] * self.sparsity_level
        )
        if len(scales) != self.sparsity_level:
            raise ValueError(
                f"LASER k={self.sparsity_level} requires exactly "
                f"{self.sparsity_level} coefficient scales"
            )
        self.register_buffer("coeff_scales", torch.tensor(scales, dtype=torch.float32))
        self.coeff_scale = float(coeff_scale)  # Backward-compatible metadata.
        self.soft_target_physical = bool(soft_target_physical)
        self.clamp_coeffs = bool(clamp_coeffs)
        if (self.coeff_scales <= 0).any():
            raise ValueError("coeff_scale must be positive")
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def encode_sparse_components(self, images: torch.Tensor):
        z = self.quant_conv(self.encoder(images)).permute(0, 2, 3, 1).float()
        b, h, w, c = z.shape
        signals = z.reshape(-1, c)
        # Cache extraction runs the encoder under BF16 autocast, but OMP's
        # Cholesky solve must remain FP32 for dtype consistency and stability.
        with torch.autocast(device_type=signals.device.type, enabled=False):
            signals = signals.float()
            dictionary = self.dictionary.float()
            gram = dictionary.t() @ dictionary
            corr_init = signals @ dictionary
            corr = corr_init
            num_signals = signals.shape[0]
            signal_idx = torch.arange(num_signals, device=signals.device)
            available = torch.ones_like(corr_init, dtype=torch.bool)
            support = torch.empty(num_signals, 0, dtype=torch.long, device=signals.device)
            chol = torch.ones(num_signals, 1, 1, device=signals.device, dtype=signals.dtype)
            active_coeffs = None
            for depth in range(1, self.sparsity_level + 1):
                scores = corr.abs().masked_fill(~available, -1.0)
                next_atoms = scores.argmax(dim=1)
                available[signal_idx, next_atoms] = False
                expanded_idx = signal_idx[:, None].expand(num_signals, depth)
                if depth > 1:
                    previous = support
                    repeated_new = next_atoms[:, None].expand(num_signals, depth - 1)
                    gram_cross = gram[previous, repeated_new].unsqueeze(-1)
                    solved = torch.linalg.solve_triangular(
                        chol, gram_cross, upper=False
                    ).transpose(1, 2)
                    bottom_right = (
                        1.0 - solved.square().sum(dim=2, keepdim=True)
                    ).clamp_min(1e-10).sqrt()
                    zeros = torch.zeros(
                        num_signals, depth - 1, 1,
                        device=signals.device, dtype=signals.dtype,
                    )
                    chol = torch.cat(
                        (
                            torch.cat((chol, zeros), dim=2),
                            torch.cat((solved, bottom_right), dim=2),
                        ),
                        dim=1,
                    )
                support = torch.cat((support, next_atoms[:, None]), dim=1)
                active_corr = corr_init[expanded_idx, support]
                active_coeffs = torch.cholesky_solve(
                    active_corr.unsqueeze(-1), chol
                ).squeeze(-1)
                corr = corr_init - active_coeffs.unsqueeze(1).bmm(gram[support]).squeeze(1)
        atoms = support.view(b, h, w, self.sparsity_level)
        physical_coeffs = active_coeffs.view(b, h, w, self.sparsity_level)
        scales = self.coeff_scales.view(1, 1, 1, self.sparsity_level)
        coeffs = physical_coeffs / scales
        if self.clamp_coeffs:
            coeffs = coeffs.clamp(-self.coeff_max, self.coeff_max)
        return atoms, coeffs

    @torch.no_grad()
    def sparse_targets(self, atoms: torch.Tensor, coeffs: torch.Tensor, *, temp: float = 0.5,
                       stochastic: bool = True, compact: bool = False,
                       hard: bool = False):
        atoms = atoms.long()
        coeffs = coeffs.float()
        coeff_tokens = (coeffs[..., None] - self.coeff_bins).abs().argmin(dim=-1)
        # The official stage-2 recipe trains against stage-1 soft codes.  For
        # LASER, atom support is discrete OMP while the continuous coefficient
        # posterior is discretized into a temperature-controlled 16K density.
        if self.soft_target_physical:
            target_values = (
                coeffs * self.coeff_scales.view(1, 1, 1, self.sparsity_level)
            )[..., None]
            bin_values = (
                self.coeff_bins.view(1, 1, 1, 1, -1)
                * self.coeff_scales.view(1, 1, 1, self.sparsity_level, 1)
            )
        else:
            target_values = coeffs[..., None]
            bin_values = self.coeff_bins
        coeff_logits = -(target_values - bin_values).square() / max(float(temp), 1e-6)
        if hard:
            coeff_probs = F.one_hot(
                coeff_tokens, num_classes=self.coeff_vocab_size
            ).to(coeff_logits.dtype)
        else:
            coeff_probs = coeff_logits.softmax(dim=-1)
        if stochastic and not hard:
            coeff_tokens = torch.multinomial(
                coeff_probs.reshape(-1, self.coeff_vocab_size), 1
            ).reshape_as(coeff_tokens)
        b, h, w, _ = atoms.shape
        scalar_depth = 2 * self.sparsity_level
        tokens = torch.empty(b, h, w, scalar_depth, device=atoms.device, dtype=torch.long)
        tokens[..., 0::2] = atoms
        tokens[..., 1::2] = coeff_tokens + self.num_atoms
        if compact:
            return tokens, (atoms, coeff_probs)
        soft_targets = torch.zeros(b, h, w, scalar_depth, self.vocab_size,
                                   device=atoms.device, dtype=coeff_probs.dtype)
        soft_targets[..., 0::2, :self.num_atoms].scatter_(-1, atoms[..., None], 1.0)
        soft_targets[..., 1::2, self.num_atoms:] = coeff_probs
        return tokens, soft_targets

    @torch.no_grad()
    def encode_sparse(self, images: torch.Tensor, *, temp: float = 0.5,
                      stochastic: bool = True, hard: bool = False):
        atoms, coeffs = self.encode_sparse_components(images)
        return self.sparse_targets(
            atoms, coeffs, temp=temp, stochastic=stochastic, hard=hard
        )

    @torch.no_grad()
    def get_code_emb_with_depth(self, tokens: torch.Tensor):
        out = torch.empty(*tokens.shape, 256, device=tokens.device, dtype=self.dictionary.dtype)
        atom_vectors = self.dictionary.t()[tokens[..., 0::2]]
        out[..., 0::2, :] = atom_vectors
        coeff_ids = (tokens[..., 1::2] - self.num_atoms).clamp(0, self.coeff_vocab_size - 1)
        coeff = self.coeff_bins[coeff_ids] * self.coeff_scales.view(
            1, 1, 1, self.sparsity_level
        )
        out[..., 1::2, :] = (coeff[..., None] - 1.0) * atom_vectors
        return out, None

    @torch.no_grad()
    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        atoms = tokens[..., 0::2].long()
        coeff_ids = (tokens[..., 1::2].long() - self.num_atoms).clamp(0, self.coeff_vocab_size - 1)
        coeffs = self.coeff_bins[coeff_ids] * self.coeff_scales.view(
            1, 1, 1, self.sparsity_level
        )
        atom_vectors = self.dictionary.t()[atoms]
        z_q = (atom_vectors * coeffs[..., None]).sum(dim=-2)
        z_q = self.post_quant_conv(z_q.permute(0, 3, 1, 2).contiguous())
        return self.decoder(z_q).clamp(-1.0, 1.0)

    @torch.no_grad()
    def compound_coeff_ids(self, coeffs: torch.Tensor, *, stochastic: bool = True,
                           temp: float = 0.5, hard: bool = False):
        """Quantize real coefficients for compound (atom, coefficient) events."""
        target_values = coeffs.float()[..., None]
        logits = -(target_values - self.coeff_bins).square() / max(float(temp), 1e-6)
        nearest = logits.argmax(dim=-1)
        if hard:
            probs = F.one_hot(nearest, num_classes=self.coeff_vocab_size).to(logits.dtype)
        else:
            probs = logits.softmax(dim=-1)
        if stochastic and not hard:
            ids = torch.multinomial(probs.reshape(-1, self.coeff_vocab_size), 1)
            ids = ids.reshape(coeffs.shape)
        else:
            ids = nearest
        return ids.long(), probs

    @torch.no_grad()
    def compound_embeddings(self, atoms: torch.Tensor, coeff_ids: torch.Tensor):
        """Physical latent contribution of every compound sparse event."""
        atom_vectors = self.dictionary.t()[atoms.long()]
        coeffs = self.coeff_bins[coeff_ids.long().clamp(0, self.coeff_vocab_size - 1)]
        scale_shape = [1] * (coeffs.ndim - 1) + [self.sparsity_level]
        coeffs = coeffs * self.coeff_scales.view(*scale_shape)
        return atom_vectors * coeffs[..., None]

    @torch.no_grad()
    def physical_contributions(self, atoms: torch.Tensor, coeffs: torch.Tensor):
        """Continuous physical LASER contribution c_i d_{a_i} for each pair."""
        atom_vectors = self.dictionary.t()[atoms.long()]
        scale_shape = [1] * (coeffs.ndim - 1) + [self.sparsity_level]
        physical_coeffs = coeffs.float() * self.coeff_scales.view(*scale_shape)
        return atom_vectors * physical_coeffs[..., None]

    @torch.no_grad()
    def decode_compound(self, atoms: torch.Tensor, coeff_ids: torch.Tensor) -> torch.Tensor:
        z_q = self.compound_embeddings(atoms, coeff_ids).sum(dim=-2)
        z_q = self.post_quant_conv(z_q.permute(0, 3, 1, 2).contiguous())
        return self.decoder(z_q).clamp(-1.0, 1.0)


class SparseTokenCacheDataset(torch.utils.data.Dataset):
    """Memory-mapped-at-load sparse components; no source image access in stage 2."""

    def __init__(self, path: Path):
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        self.atoms = payload["atoms"]
        self.coeffs = payload["coeffs"]
        self.labels = payload["labels"]
        self.meta = payload["meta"]
        if not (len(self.atoms) == len(self.coeffs) == len(self.labels)):
            raise ValueError("token-cache tensors have inconsistent row counts")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.atoms[index], self.coeffs[index], self.labels[index]


class FlatImages(Dataset):
    """Recursively load an unconditional image dataset from flat files."""

    def __init__(self, root: Path, transform):
        self.files = sorted(
            path
            for path in root.rglob("*")
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        if not self.files:
            raise ValueError(f"no images found below {root}")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.files[index]) as image:
            return self.transform(image.convert("RGB")), 0

@torch.no_grad()
def sample_class_grid(model, aux, class_names, output_dir: Path, step: int, wb=None,
                      num_condition_classes=1000,
                      num_samples=64,
                      sample_batch_size=8,
                      setting_name="default",
                      atom_temperature=1.0, atom_top_k=0, atom_top_p=0.92,
                      coeff_temperature=1.0, coeff_top_k=0, coeff_top_p=0.92):
    device = next(model.parameters()).device
    grid_side = math.isqrt(int(num_samples))
    if grid_side * grid_side != int(num_samples):
        raise ValueError("preview sample count must be a perfect square")
    if num_condition_classes == 1:
        chosen = torch.zeros(grid_side, device=device, dtype=torch.long)
    else:
        chosen = torch.randperm(num_condition_classes, device=device)[:grid_side]
    labels = chosen.repeat_interleave(grid_side)
    was_training = model.training
    model.eval()
    image_batches = []
    for start in range(0, num_samples, sample_batch_size):
        stop = min(start + sample_batch_size, num_samples)
        batch_labels = labels[start:stop]
        current = stop - start
        if isinstance(model, CompoundLaserRQTransformer):
            atoms, coeff_ids = model.sample_compound(
                current, aux, cond=batch_labels,
                atom_temperature=atom_temperature,
                atom_top_k=atom_top_k or aux.num_atoms, atom_top_p=atom_top_p,
                coeff_temperature=coeff_temperature,
                coeff_top_k=coeff_top_k or aux.coeff_vocab_size,
                coeff_top_p=coeff_top_p,
                amp=True,
            )
            batch_images = aux.decode_compound(atoms, coeff_ids)
        else:
            partial = torch.zeros(current, 8, 8, 4, device=device, dtype=torch.long)
            batch_images = aux.decode_tokens(model.sample(
                partial, model_aux=aux, cond=batch_labels, temperature=1.0,
                top_k=atom_top_k or aux.num_atoms, top_p=atom_top_p,
                amp=True, cached=True, is_tqdm=False,
            ))
        image_batches.append(((batch_images.float().cpu() + 1.0) * 0.5).clamp(0, 1))
    images = torch.cat(image_batches)
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    target = sample_dir / f"step_{step:07d}_{setting_name}.png"
    # A raw padding-free mosaic: no figure canvas, margins, labels, or titles.
    save_image(images, target, nrow=grid_side, padding=0)
    if wb is not None:
        import wandb
        wb.log({
            f"samples/{setting_name}": wandb.Image(str(target)),
            "train/global_step": step,
        })
    if was_training:
        model.train()
    return target


def preview_sampling_settings(args):
    original = {
        "setting_name": "at1_k250_p1__ct1_k250_p1",
        "atom_temperature": 1.0, "atom_top_k": 250, "atom_top_p": 1.0,
        "coeff_temperature": 1.0, "coeff_top_k": 250, "coeff_top_p": 1.0,
    }
    if not args.sample_grid_sweep:
        return [{
            "setting_name": (
                f"at{args.atom_temperature:g}_k{args.atom_top_k}_p{args.atom_top_p:g}"
                f"__ct{args.coeff_temperature:g}_k{args.coeff_top_k}_p{args.coeff_top_p:g}"
            ),
            "atom_temperature": args.atom_temperature,
            "atom_top_k": args.atom_top_k,
            "atom_top_p": args.atom_top_p,
            "coeff_temperature": args.coeff_temperature,
            "coeff_top_k": args.coeff_top_k,
            "coeff_top_p": args.coeff_top_p,
        }]
    return [
        original,
        {
            "setting_name": "at0.85_k250_p1__ct0.85_k250_p1",
            "atom_temperature": 0.85, "atom_top_k": 250, "atom_top_p": 1.0,
            "coeff_temperature": 0.85, "coeff_top_k": 250, "coeff_top_p": 1.0,
        },
        {
            "setting_name": "at0.9_k0_p0.92__ct1_k0_p0.85",
            "atom_temperature": 0.9, "atom_top_k": 0, "atom_top_p": 0.92,
            "coeff_temperature": 1.0, "coeff_top_k": 0, "coeff_top_p": 0.85,
        },
        {
            "setting_name": "at0.95_k250_p0.95__ct0.9_k250_p0.95",
            "atom_temperature": 0.95, "atom_top_k": 250, "atom_top_p": 0.95,
            "coeff_temperature": 0.9, "coeff_top_k": 250, "coeff_top_p": 0.95,
        },
    ]


@torch.no_grad()
def evaluate_generation_metrics(model, aux, val_loader, num_samples: int, batch_size: int = 64,
                                num_condition_classes=1000,
                                atom_temperature=1.0, atom_top_k=0, atom_top_p=0.92,
                                coeff_temperature=1.0, coeff_top_k=0, coeff_top_p=0.92,
                                compute_inception_score=True):
    from torchmetrics.image.fid import FrechetInceptionDistance

    device = next(model.parameters()).device
    world = dist.get_world_size() if dist.is_initialized() else 1
    process_rank = dist.get_rank() if dist.is_initialized() else 0
    local_samples = int(num_samples) // world + (process_rank < int(num_samples) % world)
    # Every rank accumulates its shard; compute() merges the sufficient
    # statistics, avoiding a costly image all-gather.
    fid_metric = FrechetInceptionDistance(
        feature=2048, normalize=True, sync_on_compute=dist.is_initialized()
    ).to(device)
    inception_metric = None
    if compute_inception_score:
        from torchmetrics.image.inception import InceptionScore
        inception_metric = InceptionScore(
            normalize=True, splits=10, sync_on_compute=dist.is_initialized()
        ).to(device)
    seen = 0
    for images, _ in val_loader:
        images = ((images.to(device, non_blocking=True).float() + 1.0) * 0.5).clamp(0, 1)
        keep = min(images.size(0), local_samples - seen)
        fid_metric.update(images[:keep], real=True)
        seen += keep
        if seen >= local_samples:
            break
    generated = 0
    was_training = model.training
    model.eval()
    while generated < local_samples:
        current = min(int(batch_size), local_samples - generated)
        # ImageNet uses the exact uniform class prior. Face models have a
        # single unconditional embedding and therefore receive all-zero ids.
        local_indices = torch.arange(
            generated, generated + current, device=device, dtype=torch.long
        )
        labels = (local_indices * world + process_rank).remainder(num_condition_classes)
        if isinstance(model, CompoundLaserRQTransformer):
            atoms, coeff_ids = model.sample_compound(
                current, aux, cond=labels,
                atom_temperature=atom_temperature,
                atom_top_k=atom_top_k or aux.num_atoms, atom_top_p=atom_top_p,
                coeff_temperature=coeff_temperature,
                coeff_top_k=coeff_top_k or aux.coeff_vocab_size,
                coeff_top_p=coeff_top_p,
                amp=True,
            )
            images = ((aux.decode_compound(atoms, coeff_ids).float() + 1.0) * 0.5).clamp(0, 1)
        else:
            partial = torch.zeros(current, 8, 8, 4, device=device, dtype=torch.long)
            tokens = model.sample(
                partial, model_aux=aux, cond=labels, temperature=1.0,
                top_k=aux.num_atoms, top_p=0.92, amp=True, cached=True, is_tqdm=False,
            )
            images = ((aux.decode_tokens(tokens).float() + 1.0) * 0.5).clamp(0, 1)
        fid_metric.update(images, real=False)
        if inception_metric is not None:
            inception_metric.update(images)
        generated += current
    fid = float(fid_metric.compute().item())
    if inception_metric is None:
        inception_mean = inception_std = None
    else:
        inception_mean, inception_std = (
            float(value.item()) for value in inception_metric.compute()
        )
    if was_training:
        model.train()
    return fid, inception_mean, inception_std


def atomic_torch_save(payload, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)


def snapshot_checkpoint(source: Path, target: Path):
    """Create an atomic, space-efficient snapshot of an immutable checkpoint."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".snapshot.tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        # Hard links can fail when a custom checkpoint directory crosses filesystems.
        shutil.copy2(source, temporary)
    os.replace(temporary, target)


def upload_checkpoints(wb, paths: list[Path], *, artifact_name: str, aliases, metadata):
    import wandb
    artifact = wandb.Artifact(artifact_name, type="model", metadata=metadata)
    for path in paths:
        artifact.add_file(str(path), name=path.name)
    logged = wb.log_artifact(artifact, aliases=list(aliases))
    logged.wait()
    print(f"Uploaded checkpoint artifact {logged.name}", flush=True)


def upload_checkpoint(wb, path: Path, *, artifact_name: str, aliases, metadata):
    """Backward-compatible single-checkpoint artifact upload."""
    upload_checkpoints(
        wb, [path], artifact_name=artifact_name, aliases=aliases, metadata=metadata
    )


def _replace_hard_link(source: Path, destination: Path):
    """Atomically point a fixed upload slot at an immutable checkpoint inode."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def upload_selected_checkpoint_files(
    wb,
    *,
    last_checkpoint: Path,
    best_fid,
    upload_dir: Path,
):
    """Replace fixed W&B run-file slots with last plus the best three FIDs.

    Artifact versions are immutable and therefore accumulate indefinitely.
    Run files keep stable online names, matching the Stage-1 retention policy,
    while the local hard links avoid another multi-gigabyte checkpoint copy.
    """
    sources = [("last.pt", last_checkpoint)]
    sources.extend(
        (f"best-fid-{rank_index:02d}.pt", Path(saved_path))
        for rank_index, (_, saved_path) in enumerate(
            sorted(best_fid, key=lambda item: float(item[0]))[:3], start=1
        )
    )
    sources = [(slot, source.resolve()) for slot, source in sources if source.is_file()]
    if not sources:
        return []

    upload_dir = upload_dir.expanduser().resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)
    active_names = {slot for slot, _ in sources}
    for stale in upload_dir.glob("best-fid-*.pt"):
        if stale.name not in active_names:
            stale.unlink(missing_ok=True)

    uploaded = []
    for slot, source in sources:
        destination = upload_dir / slot
        _replace_hard_link(source, destination)
        wb.save(str(destination), base_path=str(upload_dir), policy="now")
        uploaded.append(destination)
    print(
        "Queued fixed W&B checkpoint files: "
        + ", ".join(path.name for path in uploaded),
        flush=True,
    )
    return uploaded


def upload_token_cache_once(wb, token_cache: Path, output_dir: Path):
    """Upload the immutable cache once, with a persistent local receipt."""
    import wandb

    receipt = output_dir / "token_cache_artifact.json"
    if receipt.is_file():
        return
    payload = torch.load(token_cache, map_location="cpu", weights_only=True, mmap=True)
    metadata = dict(payload.get("meta", {}))
    artifact = wandb.Artifact(f"{wb.id}-token-cache", type="dataset", metadata=metadata)
    artifact.add_file(str(token_cache), name=token_cache.name)
    logged = wb.log_artifact(artifact, aliases=["latest", "training-cache"])
    logged.wait()
    receipt.write_text(json.dumps({
        "artifact": f"{wb.id}-token-cache",
        "version": logged.version,
        "file": str(token_cache.resolve()),
    }, indent=2) + "\n")
    print(f"Uploaded token cache artifact {logged.name}", flush=True)


def image_transform():
    return transforms.Compose([
        transforms.Resize(256), transforms.RandomCrop(256), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])


def val_image_transform():
    return transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])


class LaserRQTransformer(RQTransformer):
    """Upstream model with depth-wise validity masks for the combined vocabulary."""

    def __init__(self, config, num_atoms: int):
        super().__init__(config)
        self.num_atoms = int(num_atoms)

    def _mask_depth_vocab(self, logits):
        logits = logits.clone()
        mask_value = torch.finfo(logits.dtype).min
        logits[..., 0::2, self.num_atoms:] = mask_value
        logits[..., 1::2, :self.num_atoms] = mask_value
        return logits

    def classify_head_outputs(self, head_outputs):
        """Avoid computing invalid vocabulary halves at alternating depths."""
        normalized = self.classifier.layer_norm(head_outputs)
        linear = self.classifier.linear
        atom_logits = F.linear(
            normalized[..., 0::2, :], linear.weight[: self.num_atoms],
            None if linear.bias is None else linear.bias[: self.num_atoms],
        )
        coeff_logits = F.linear(
            normalized[..., 1::2, :], linear.weight[self.num_atoms :],
            None if linear.bias is None else linear.bias[self.num_atoms :],
        )
        return {"atom_logits": atom_logits, "coeff_logits": coeff_logits}

    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        if isinstance(out, dict):
            return out
        if isinstance(out, tuple) and out and isinstance(out[0], dict):
            return out
        if isinstance(out, tuple):
            return self._mask_depth_vocab(out[0]), out[1]
        return self._mask_depth_vocab(out)

    def cached_forward(self, *args, sample_loc=(0, 0, 0), **kwargs):
        logits = super().cached_forward(*args, sample_loc=sample_loc, **kwargs)
        mask_value = torch.finfo(logits.dtype).min
        d = int(sample_loc[2])
        if d % 2 == 0:
            logits[:, self.num_atoms:] = mask_value
        else:
            logits[:, :self.num_atoms] = mask_value
        return logits


class AtomConditionedRefinerBlock(nn.Module):
    """Residual pair-local refinement after an atom identity is known."""

    def __init__(self, embed_dim: int):
        super().__init__()
        hidden_dim = 2 * int(embed_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(3 * embed_dim),
            nn.Linear(3 * embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, hidden, atom_context):
        fused = torch.cat((hidden, atom_context, hidden * atom_context), dim=-1)
        return hidden + self.net(fused)


class CompoundLaserRQTransformer(RQTransformer):
    """AR model with one depth position per (atom, coefficient) pair.

    The joint distribution is factorized without expanding the vocabulary:
    p(atom, coeff | history) = p(atom | history) p(coeff | history, atom).
    Packed integer inputs are only a transport representation; embeddings are
    reconstructed from the frozen LASER dictionary and coefficient bins.
    """

    def __init__(self, config, num_atoms: int, coeff_vocab_size: int,
                 refiner_layers: int = 0, geometry_head: bool = False,
                 micro_transformer_layers: int = 0,
                 depth_specific_coeff_heads: bool = False):
        super().__init__(config)
        self.num_atoms = int(num_atoms)
        self.coeff_vocab_size = int(coeff_vocab_size)
        input_dim = int(config.input_embed_dim)
        embed_dim = int(config.embed_dim)
        self.coeff_token_embedding = nn.Embedding(self.coeff_vocab_size, input_dim)
        self.pair_embedding_adapter = nn.Sequential(
            nn.LayerNorm(3 * input_dim),
            nn.Linear(3 * input_dim, input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, input_dim),
        )
        self.coeff_atom_proj = nn.Linear(input_dim, embed_dim, bias=False)
        self.refiner_layers = int(refiner_layers)
        self.micro_transformer_layers = int(micro_transformer_layers)
        if self.refiner_layers > 0 and self.micro_transformer_layers > 0:
            raise ValueError("MLP refiner and micro-transformer are mutually exclusive")
        if self.micro_transformer_layers > 0:
            micro_config = config.head.copy()
            micro_config.n_layer = self.micro_transformer_layers
            self.coeff_micro_transformer = AttentionStack(micro_config)
            self.coeff_micro_pos = nn.Parameter(torch.zeros(1, 2, embed_dim))
            self.coeff_micro_pos.data.normal_(mean=0.0, std=0.02)
        elif self.refiner_layers > 0:
            self.coeff_refiner = nn.ModuleList([
                AtomConditionedRefinerBlock(embed_dim)
                for _ in range(self.refiner_layers)
            ])
        else:
            self.coeff_fusion = nn.Sequential(
                nn.LayerNorm(3 * embed_dim),
                nn.Linear(3 * embed_dim, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim),
            )
        self.depth_specific_coeff_heads = bool(depth_specific_coeff_heads)

        def make_coeff_classifier():
            return nn.Sequential(
                nn.LayerNorm(config.embed_dim),
                nn.Linear(config.embed_dim, self.coeff_vocab_size),
            )

        if self.depth_specific_coeff_heads:
            self.coeff_classifier = nn.ModuleList([
                make_coeff_classifier() for _ in range(self.block_size[-1])
            ])
        else:
            self.coeff_classifier = make_coeff_classifier()
        self.contribution_head = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.SiLU(),
            nn.Linear(config.embed_dim, input_dim),
        ) if geometry_head else None
        self._teacher_atoms = None

    def unpack(self, packed):
        return packed.div(self.coeff_vocab_size, rounding_mode="floor"), packed.remainder(self.coeff_vocab_size)

    def embed_with_model_aux(self, packed, model_aux):
        atoms, coeff_ids = self.unpack(packed)
        atom_vectors = model_aux.dictionary.t()[atoms.long()]
        coeff_embedding = self.coeff_token_embedding(coeff_ids.long())
        contribution = model_aux.compound_embeddings(atoms, coeff_ids)
        features = torch.cat((atom_vectors, coeff_embedding, contribution), dim=-1)
        # Preserve the physical latent contribution while adding a learned
        # representation that keeps atom identity and coefficient identity.
        return contribution + self.pair_embedding_adapter(features)

    def refine_coefficient_hidden(self, hidden, atom_vectors):
        atom_context = self.coeff_atom_proj(atom_vectors)
        if self.micro_transformer_layers > 0:
            pair = torch.stack((hidden, atom_context), dim=-2)
            pair = pair + self.coeff_micro_pos
            micro_output = self.coeff_micro_transformer(
                pair.reshape(-1, 2, pair.shape[-1])
            ).reshape_as(pair)
            return hidden + micro_output[..., -1, :]
        if self.refiner_layers > 0:
            for block in self.coeff_refiner:
                hidden = block(hidden, atom_context)
            return hidden
        fused = torch.cat((hidden, atom_context, hidden * atom_context), dim=-1)
        return hidden + self.coeff_fusion(fused)

    def classify_coefficients(self, refined, depth_index=None):
        if not self.depth_specific_coeff_heads:
            return self.coeff_classifier(refined)
        if depth_index is not None:
            return self.coeff_classifier[int(depth_index)](refined)
        if refined.shape[-2] != len(self.coeff_classifier):
            raise ValueError("depth-specific coefficient heads require an explicit depth axis")
        return torch.stack([
            head(refined[..., depth, :])
            for depth, head in enumerate(self.coeff_classifier)
        ], dim=-2)

    def coefficient_logits(self, hidden, atom_vectors, depth_index=None):
        refined = self.refine_coefficient_hidden(hidden, atom_vectors)
        return self.classify_coefficients(refined, depth_index=depth_index)

    def classify_head_outputs(self, head_outputs):
        atoms = self._teacher_atoms
        if atoms is None:
            raise RuntimeError("compound teacher atoms were not set")
        atom_logits = self.classifier(head_outputs)
        atom_vectors = self._model_aux.dictionary.t()[atoms.long()]
        refined = self.refine_coefficient_hidden(head_outputs, atom_vectors)
        outputs = {
            "atom_logits": atom_logits,
            "coeff_logits": self.classify_coefficients(refined),
        }
        if self.contribution_head is not None:
            outputs["physical_contribution"] = self.contribution_head(refined)
        return outputs

    def forward(self, packed, model_aux=None, cond=None, amp=False):
        self._teacher_atoms, _ = self.unpack(packed)
        self._model_aux = model_aux
        try:
            return super().forward(packed, model_aux=model_aux, cond=cond, amp=amp)
        finally:
            self._teacher_atoms = None
            self._model_aux = None

    @torch.no_grad()
    def cached_head_output(self, packed, model_aux, cond, sample_loc, amp=True):
        h, w, d = sample_loc
        B, H, W, D = packed.shape
        sampling_idx = h * W + w
        xs = packed.reshape(B, -1, D)[:, :sampling_idx + 1]
        with torch.amp.autocast("cuda", enabled=amp):
            if cond is None:
                cond = torch.zeros(B, self.block_size_cond, device=xs.device, dtype=torch.long)
            else:
                cond = cond.reshape(B, self.block_size_cond)
            seq_len, cond_len = xs.shape[1], cond.shape[1]
            if d == 0:
                xs_emb = self.input_mlp(self.embed_with_model_aux(xs, model_aux))
                conds_emb = self.cond_emb(cond) + self.pos_emb_cond[:, :cond_len]
                spatial_inputs = xs_emb.sum(dim=-2) + self.pos_emb_hw[:, :seq_len]
                latents = torch.cat((conds_emb, spatial_inputs[:, :-1]), dim=1)
                latents = self.embed_drop(latents)[:, :cond_len + sampling_idx]
                if self._cache["spatial_ctx_hw"] is None:
                    spatial_ctx = self.body_transformer.cached_forward(latents)[:, -1:].contiguous()
                else:
                    spatial_ctx = self.body_transformer.cached_forward(latents[:, -1:])
                self._cache["spatial_ctx_hw"] = spatial_ctx
            spatial_ctx = self._cache["spatial_ctx_hw"]
            depth_ctx = self.embed_with_model_aux(xs, model_aux)
            if self.config.cumsum_depth_ctx:
                depth_ctx = torch.cumsum(depth_ctx, dim=-2)
            depth_ctx = self.head_mlp(depth_ctx)[:, sampling_idx]
            full = torch.cat((spatial_ctx, depth_ctx[:, :-1]), dim=1)
            full = full + self.pos_emb_d[:, :D]
            if d == 0:
                self.head_transformer.init_cache()
            return self.head_transformer.cached_forward(full[:, d:d + 1]).reshape(B, -1)

    @torch.no_grad()
    def sample_compound(self, batch_size, model_aux, cond=None, temperature=1.0,
                        atom_top_k=16384, atom_top_p=0.92, coeff_top_p=0.92,
                        coeff_top_k=0, atom_temperature=None,
                        coeff_temperature=None, amp=True):
        H, W, D = self.block_size
        device = next(self.parameters()).device
        atoms = torch.zeros(batch_size, H, W, D, device=device, dtype=torch.long)
        coeff_ids = torch.full_like(atoms, self.coeff_vocab_size // 2)
        packed = atoms * self.coeff_vocab_size + coeff_ids
        atom_temperature = float(temperature if atom_temperature is None else atom_temperature)
        coeff_temperature = float(temperature if coeff_temperature is None else coeff_temperature)
        self.init_cache()
        for h in range(H):
            for w in range(W):
                for d in range(D):
                    hidden = self.cached_head_output(packed, model_aux, cond, (h, w, d), amp=amp)
                    atom_logits = self.classifier(hidden)
                    if d > 0:
                        # OMP supports contain distinct atoms; enforce the same
                        # invariant during generation instead of wasting a pair.
                        atom_logits = atom_logits.clone()
                        atom_logits.scatter_(1, atoms[:, h, w, :d], -float("inf"))
                    atom = sample_from_logits(atom_logits, temperature=atom_temperature,
                                              top_k=min(atom_top_k, self.num_atoms), top_p=atom_top_p)
                    atom_vec = model_aux.dictionary.t()[atom.long()]
                    coeff_logits = self.coefficient_logits(hidden, atom_vec, depth_index=d)
                    coeff_id = sample_from_logits(coeff_logits, temperature=coeff_temperature,
                                                  top_k=coeff_top_k or self.coeff_vocab_size,
                                                  top_p=coeff_top_p)
                    atoms[:, h, w, d] = atom
                    coeff_ids[:, h, w, d] = coeff_id
                    packed[:, h, w, d] = atom * self.coeff_vocab_size + coeff_id
        self.init_cache()
        return atoms, coeff_ids

def build_model(total_vocab_size: int, num_atoms: int, *, compound=False,
                coeff_vocab_size=2048, compound_refiner_layers=0,
                compound_geometry_head=False,
                compound_micro_transformer_layers=0,
                compound_depth_specific_coeff_heads=False,
                sparsity_level=2,
                model_preset="imagenet-1400m"):
    presets = {
        "imagenet-1400m": {
            "embed_dim": 1536, "vocab_size_cond": 1000,
            "body_layers": 42, "body_heads": 24,
            "head_layers": 6, "head_heads": 24,
        },
        # Exact body/head geometry from KakaoBrain's
        # ffhq256-rqtransformer-8x8x4-350M.yaml. Compound mode changes only
        # the depth axis from scalar slots to sparse compound events.
        "ffhq-350m": {
            "embed_dim": 1024, "vocab_size_cond": 1,
            "body_layers": 24, "body_heads": 16,
            "head_layers": 4, "head_heads": 16,
        },
    }
    try:
        preset = presets[model_preset]
    except KeyError as error:
        raise ValueError(f"unknown model preset: {model_preset}") from error
    cfg = OmegaConf.create({
        "type": "rq-transformer",
        "block_size": [8, 8, sparsity_level if compound else 2 * sparsity_level],
        "embed_dim": preset["embed_dim"],
        "input_embed_dim": 256, "shared_tok_emb": True, "shared_cls_emb": True,
        "input_emb_vqvae": True, "head_emb_vqvae": True, "cumsum_depth_ctx": True,
        "vocab_size": num_atoms if compound else total_vocab_size,
        "vocab_size_cond": preset["vocab_size_cond"], "block_size_cond": 1,
        "body": {"n_layer": preset["body_layers"], "block": {"n_head": preset["body_heads"]}},
        "head": {"n_layer": preset["head_layers"], "block": {"n_head": preset["head_heads"]}},
    })
    if compound:
        return CompoundLaserRQTransformer(
            RQTransformerConfig.create(cfg), num_atoms=num_atoms,
            coeff_vocab_size=coeff_vocab_size,
            refiner_layers=compound_refiner_layers,
            geometry_head=compound_geometry_head,
            micro_transformer_layers=compound_micro_transformer_layers,
            depth_specific_coeff_heads=compound_depth_specific_coeff_heads,
        )
    return LaserRQTransformer(RQTransformerConfig.create(cfg), num_atoms=num_atoms)


def compound_objective(
    atom_logits,
    coeff_logits,
    physical_prediction,
    target_atoms,
    target_coeff_probs,
    target_physical,
    *,
    atom_weight: float,
    geometry_weight: float,
    accumulation: int,
    distribution_geometry: bool = False,
    geometry_dictionary=None,
    geometry_coeff_bins=None,
    geometry_coeff_scales=None,
    geometry_top_k: int = 4,
):
    """Weighted token objective plus normalized physical latent geometry.

    V3 learns geometry through an auxiliary contribution head.  V4 instead
    constructs the physical contribution from the atom/coeff distributions
    used at sampling time, so the auxiliary objective cannot be solved by an
    inference-time-unused shortcut.
    """
    atom_log_probs = F.log_softmax(atom_logits.float(), dim=-1)
    coeff_log_probs = F.log_softmax(coeff_logits.float(), dim=-1)
    atom_nll = -atom_log_probs.gather(
        -1, target_atoms.long().unsqueeze(-1)
    ).squeeze(-1)
    coeff_cross_entropy = -(target_coeff_probs * coeff_log_probs).sum(dim=-1)
    depth = target_atoms.shape[-1]
    classification = (
        atom_weight * atom_nll.sum(dim=-1) + coeff_cross_entropy.sum(dim=-1)
    ).mean() / ((atom_weight + 1.0) * depth)

    geometry = atom_logits.new_zeros((), dtype=torch.float32)
    pair_mse = atom_logits.new_zeros((), dtype=torch.float32)
    spatial_mse = atom_logits.new_zeros((), dtype=torch.float32)
    if geometry_weight > 0:
        if target_physical is None:
            raise ValueError("physical contribution target required for geometry loss")
        if distribution_geometry:
            if geometry_dictionary is None or geometry_coeff_bins is None or geometry_coeff_scales is None:
                raise ValueError("distribution geometry requires dictionary, bins, and depth scales")
            candidate_count = min(max(int(geometry_top_k), 1), atom_logits.shape[-1])
            candidate_logits, candidate_atoms = atom_logits.float().topk(candidate_count, dim=-1)
            target_atom_logits = atom_logits.float().gather(
                -1, target_atoms.long().unsqueeze(-1)
            )
            target_is_candidate = (candidate_atoms == target_atoms.long().unsqueeze(-1)).any(
                dim=-1, keepdim=True
            )
            target_atom_logits = target_atom_logits.masked_fill(
                target_is_candidate, -float("inf")
            )
            candidate_logits = torch.cat((candidate_logits, target_atom_logits), dim=-1)
            candidate_atoms = torch.cat(
                (candidate_atoms, target_atoms.long().unsqueeze(-1)), dim=-1
            )
            candidate_weights = candidate_logits.softmax(dim=-1)
            dictionary = geometry_dictionary.float().t()
            candidate_vectors = dictionary[candidate_atoms]
            expected_atom = (
                candidate_weights.unsqueeze(-1) * candidate_vectors
            ).sum(dim=-2)
            coeff_bins = geometry_coeff_bins.float()
            expected_coeff = (coeff_log_probs.exp() * coeff_bins).sum(dim=-1)
            scales = geometry_coeff_scales.float().view(
                *([1] * (expected_coeff.ndim - 1)), -1
            )
            prediction = expected_atom * (expected_coeff * scales).unsqueeze(-1)
        else:
            if physical_prediction is None:
                raise ValueError("auxiliary-head geometry requires physical prediction")
            prediction = physical_prediction.float()
        target = target_physical.float()
        pair_mse = F.mse_loss(prediction, target)
        predicted_spatial = prediction.sum(dim=-2)
        target_spatial = target.sum(dim=-2)
        spatial_mse = F.mse_loss(predicted_spatial, target_spatial)
        pair_energy = target.square().mean().detach().clamp_min(1e-6)
        spatial_energy = target_spatial.square().mean().detach().clamp_min(1e-6)
        geometry = 0.5 * (pair_mse / pair_energy + spatial_mse / spatial_energy)

    total = (classification + geometry_weight * geometry) / accumulation
    return total, {
        "atom_nll": atom_nll,
        "coeff_cross_entropy": coeff_cross_entropy,
        "classification": classification,
        "geometry": geometry,
        "geometry_pair_mse": pair_mse,
        "geometry_spatial_mse": spatial_mse,
    }


def scheduled_geometry_weight(
    target_weight: float,
    progress_epochs: float,
    start_epoch: float = 0.0,
    warmup_epochs: float = 0.0,
) -> float:
    """Delay distribution geometry until token predictions have useful support."""
    if target_weight <= 0 or progress_epochs <= start_epoch:
        return 0.0
    if warmup_epochs <= 0:
        return float(target_weight)
    fraction = min(1.0, (progress_epochs - start_epoch) / warmup_epochs)
    return float(target_weight) * max(0.0, fraction)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--token-cache", type=Path, default=None,
                   help="Precomputed atoms/coefficients/labels; disables image loading and encoding")
    p.add_argument("--resume-checkpoint", type=Path, default=None,
                   help="Explicit source stage-2 checkpoint (allows a new output/run)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", choices=("imagenet", "celebahq", "ffhq"), default="imagenet")
    p.add_argument(
        "--model-preset", choices=("imagenet-1400m", "ffhq-350m"),
        default="imagenet-1400m",
    )
    p.add_argument(
        "--fid-real-split", choices=("train", "val"), default=None,
        help="Reference split; defaults to val for ImageNet and train for CelebA-HQ/FFHQ",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--total-batch-size", type=int, default=2048)
    p.add_argument(
        "--distributed-backend", choices=("ddp", "fsdp"), default="ddp",
        help="Multi-GPU wrapper; FSDP uses transformer-block FULL_SHARD",
    )
    p.add_argument("--num-atoms", type=int, default=16384)
    p.add_argument("--sparsity-level", type=int, default=2)
    p.add_argument("--coeff-vocab-size", type=int, default=2048)
    p.add_argument("--coeff-max", type=float, default=20.0)
    p.add_argument("--coeff-scale", type=float, default=6.4)
    p.add_argument("--coeff-scales", type=float, nargs="+")
    p.add_argument(
        "--coeff-target-mode", choices=("soft", "hard"), default="soft",
        help="Use stochastic soft coefficient targets or deterministic nearest-bin targets",
    )
    p.add_argument(
        "--coeff-target-temperature", type=float, default=0.5,
        help="Squared-distance soft-target temperature; ignored in hard mode",
    )
    p.add_argument(
        "--compound-tokens", action=argparse.BooleanOptionalAction, default=False,
        help="Use one compound (atom, coefficient) AR event per sparse component",
    )
    p.add_argument("--compound-refiner-layers", type=int, default=0)
    p.add_argument("--compound-micro-transformer-layers", type=int, default=0)
    p.add_argument(
        "--compound-depth-specific-coeff-heads",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument(
        "--compound-distribution-geometry",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--geometry-top-k", type=int, default=4)
    p.add_argument("--atom-loss-weight", type=float, default=1.0)
    p.add_argument("--geometry-loss-weight", type=float, default=0.0)
    p.add_argument("--geometry-start-epoch", type=float, default=0.0)
    p.add_argument("--geometry-warmup-epochs", type=float, default=0.0)
    p.add_argument("--atom-temperature", type=float, default=1.0)
    p.add_argument("--atom-top-k", type=int, default=0,
                   help="0 keeps all atom logits")
    p.add_argument("--atom-top-p", type=float, default=0.92)
    p.add_argument("--coeff-temperature", type=float, default=1.0)
    p.add_argument("--coeff-top-k", type=int, default=0,
                   help="0 keeps all coefficient logits")
    p.add_argument("--coeff-top-p", type=float, default=0.92)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument(
        "--lr-schedule", choices=("cosine", "constant"), default="cosine",
        help="Optimizer-step LR schedule; cosine matches the original RQ-Transformer recipe",
    )
    p.add_argument(
        "--lr-schedule-epochs", type=int, default=100,
        help="Global cosine horizon; independent of the target epoch for this relaunch",
    )
    p.add_argument("--min-lr", type=float, default=0.0)
    p.add_argument("--wandb-project", default="laser")
    p.add_argument("--wandb-entity", default="helloimlixin-rutgers")
    p.add_argument("--wandb-name", default="imagenet-rqtransformer-laser-a16384-k2-stage2")
    p.add_argument("--wandb-id", default=None)
    p.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online",
        help="Disable W&B for local memory smoke tests with --wandb-mode disabled",
    )
    p.add_argument("--fid-num-samples", type=int, default=2048)
    p.add_argument("--fid-batch-size", type=int, default=64)
    p.add_argument("--fid-every", type=int, default=1,
                   help="Run full FID every N epochs; 0 disables evaluation")
    p.add_argument("--save-ckpt-freq", type=int, default=2,
                   help="Save the full training checkpoint every N epochs")
    p.add_argument("--save-step-freq", type=int, default=0,
                   help="Atomically overwrite last.pt every N optimizer steps; 0 disables")
    p.add_argument("--checkpoint-dir", type=Path, default=None,
                   help="Persistent checkpoint directory under /workspace")
    p.add_argument("--sample-grid-every", type=int, default=100,
                   help="Generate the expensive 64-image class grid every N optimizer steps; 0 disables it")
    p.add_argument("--sample-grid-size", type=int, default=64,
                   help="Perfect-square preview batch size")
    p.add_argument("--sample-grid-batch-size", type=int, default=8,
                   help="Memory-safe generation minibatch used to assemble a preview grid")
    p.add_argument(
        "--sample-grid-sweep", action=argparse.BooleanOptionalAction, default=False,
        help="Write one grid for each built-in atom/coefficient sampling preset",
    )
    p.add_argument(
        "--sample-grid-on-start", action=argparse.BooleanOptionalAction, default=False,
        help="Generate one preview after initialization/resume, before training",
    )
    p.add_argument(
        "--upload-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Upload 16+ GB checkpoints as W&B artifacts (disabled by default)",
    )
    p.add_argument(
        "--checkpoint-upload-mode",
        choices=("artifact", "files"),
        default="artifact",
        help=(
            "artifact creates immutable versions; files replaces fixed last and "
            "best-FID slots on the W&B run"
        ),
    )
    p.add_argument(
        "--upload-token-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Upload the immutable token cache once as a W&B dataset artifact",
    )
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--max-optimizer-steps", type=int, default=0,
        help="Stop after this many optimizer steps in this launch; 0 runs all epochs",
    )
    p.add_argument(
        "--smoke-test", action="store_true",
        help="Synchronize/step after one microbatch while preserving the configured global schedule",
    )
    p.add_argument(
        "--generation-smoke-test", action="store_true",
        help="Generate and decode one FID-sized batch, report peak memory, and exit",
    )
    args = p.parse_args()
    if args.sparsity_level <= 0:
        raise ValueError("--sparsity-level must be positive")
    if args.coeff_scales is not None and len(args.coeff_scales) != args.sparsity_level:
        raise ValueError(
            f"--coeff-scales requires {args.sparsity_level} values for "
            f"k={args.sparsity_level}"
        )
    if args.compound_refiner_layers < 0:
        raise ValueError("--compound-refiner-layers cannot be negative")
    if args.compound_micro_transformer_layers < 0:
        raise ValueError("--compound-micro-transformer-layers cannot be negative")
    if args.compound_refiner_layers > 0 and args.compound_micro_transformer_layers > 0:
        raise ValueError("compound MLP refiner and micro-transformer are mutually exclusive")
    if args.geometry_top_k <= 0:
        raise ValueError("--geometry-top-k must be positive")
    if args.atom_loss_weight <= 0:
        raise ValueError("--atom-loss-weight must be positive")
    if args.geometry_loss_weight < 0:
        raise ValueError("--geometry-loss-weight cannot be negative")
    if args.geometry_start_epoch < 0:
        raise ValueError("--geometry-start-epoch cannot be negative")
    if args.geometry_warmup_epochs < 0:
        raise ValueError("--geometry-warmup-epochs cannot be negative")
    if args.save_step_freq < 0:
        raise ValueError("--save-step-freq cannot be negative")
    if (args.sample_grid_size <= 0 or
            math.isqrt(args.sample_grid_size) ** 2 != args.sample_grid_size):
        raise ValueError("--sample-grid-size must be a positive perfect square")
    if not 0 < args.sample_grid_batch_size <= args.sample_grid_size:
        raise ValueError("--sample-grid-batch-size must be in [1, --sample-grid-size]")
    if args.fid_every < 0:
        raise ValueError("--fid-every cannot be negative")
    if args.max_optimizer_steps < 0:
        raise ValueError("--max-optimizer-steps cannot be negative")
    if args.smoke_test and args.max_optimizer_steps <= 0:
        raise ValueError("--smoke-test requires --max-optimizer-steps")
    if args.smoke_test and args.generation_smoke_test:
        raise ValueError("training and generation smoke modes are mutually exclusive")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    if args.lr_schedule_epochs <= 0:
        raise ValueError("--lr-schedule-epochs must be positive")
    if not 0 <= args.min_lr <= args.lr:
        raise ValueError("--min-lr must be between zero and --lr")
    if args.lr_schedule == "cosine" and args.epochs > args.lr_schedule_epochs:
        raise ValueError("--epochs cannot exceed the global cosine schedule horizon")
    face_datasets = {"celebahq", "ffhq"}
    if args.dataset in face_datasets and args.model_preset != "ffhq-350m":
        raise ValueError("CelebA-HQ and FFHQ require --model-preset ffhq-350m")
    if args.model_preset == "ffhq-350m" and args.dataset not in face_datasets:
        raise ValueError("--model-preset ffhq-350m is defined for CelebA-HQ and FFHQ")
    if args.geometry_loss_weight > 0 and not args.compound_tokens:
        raise ValueError("geometry contribution loss requires --compound-tokens")
    if args.compound_distribution_geometry and args.geometry_loss_weight <= 0:
        raise ValueError("distribution geometry requires a positive geometry loss weight")
    if args.compound_depth_specific_coeff_heads and not args.compound_tokens:
        raise ValueError("depth-specific coefficient heads require --compound-tokens")
    if args.atom_temperature <= 0 or args.coeff_temperature <= 0:
        raise ValueError("sampling temperatures must be positive")
    if args.coeff_target_temperature <= 0:
        raise ValueError("coefficient target temperature must be positive")
    if args.atom_top_k < 0 or args.coeff_top_k < 0:
        raise ValueError("sampling top-k values cannot be negative")
    if not 0 < args.atom_top_p <= 1 or not 0 < args.coeff_top_p <= 1:
        raise ValueError("sampling top-p values must be in (0, 1]")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        # Some H100 pod allocations expose NVSwitch P2P but do not permit
        # NCCL's multicast NVLS transport. Keep regular NVLink collectives.
        os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
        # Rank 0 periodically writes a ~16 GB recovery checkpoint to network
        # storage.  Allow that I/O to outlive NCCL's short default watchdog.
        dist.init_process_group("nccl", timeout=timedelta(minutes=45))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = persistent_checkpoint_dir(args.output, args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint = checkpoint_dir / "last.pt"
    resume_checkpoint = args.resume_checkpoint or last_checkpoint

    cache_meta = None
    cached_bin_centers = None
    if args.token_cache:
        dataset = SparseTokenCacheDataset(args.token_cache)
        cache_meta = dataset.meta
        expected_cache = {
            "dataset": args.dataset,
            "num_atoms": args.num_atoms,
            "coeff_vocab_size": args.coeff_vocab_size,
            "shape": [8, 8, args.sparsity_level],
        }
        for key, expected in expected_cache.items():
            actual = cache_meta.get(key)
            if actual != expected:
                raise ValueError(
                    f"token cache {key} mismatch: cache={actual!r}, launch={expected!r}"
                )
        cached_coeff_max = float(cache_meta.get("coeff_max", args.coeff_max))
        if not math.isclose(cached_coeff_max, args.coeff_max, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(
                "token cache coeff_max mismatch: "
                f"cache={cached_coeff_max}, launch={args.coeff_max}"
            )
        cached_scales = cache_meta.get("coeff_scales")
        if cached_scales is not None:
            cached_scales = [float(value) for value in cached_scales]
            if len(cached_scales) != args.sparsity_level:
                raise ValueError(
                    "token cache coefficient scale count mismatch: "
                    f"cache={len(cached_scales)}, k={args.sparsity_level}"
                )
            if args.coeff_scales is None:
                args.coeff_scales = cached_scales
            elif any(
                not math.isclose(float(given), cached, rel_tol=1e-6, abs_tol=1e-8)
                for given, cached in zip(args.coeff_scales, cached_scales)
            ):
                raise ValueError(
                    "token cache coeff_scales mismatch: "
                    f"cache={cached_scales}, launch={args.coeff_scales}"
                )
        elif not math.isclose(
            float(cache_meta.get("coeff_scale", args.coeff_scale)),
            args.coeff_scale, rel_tol=1e-6, abs_tol=1e-8,
        ):
            raise ValueError("token cache coeff_scale does not match the launch")
        cached_bin_centers = cache_meta.get("coeff_bin_centers")
        if cached_bin_centers is not None:
            if len(cached_bin_centers) != args.coeff_vocab_size:
                raise ValueError(
                    "token cache coefficient-bin count mismatch: "
                    f"cache={len(cached_bin_centers)}, launch={args.coeff_vocab_size}"
                )
            cached_bin_centers = [float(value) for value in cached_bin_centers]
        class_names = (
            ["unconditional"] if args.dataset in face_datasets
            else class_names_for_dataset("imagenet")
        )
    else:
        image_dataset = (
            FlatImages(args.data, transform=image_transform())
            if args.dataset == "ffhq"
            else datasets.ImageFolder(args.data / "train", transform=image_transform())
        )
        class_names = (
            ["unconditional"] if args.dataset in face_datasets
            else class_names_for_dataset("imagenet", image_dataset.classes)
        )
        dataset = image_dataset
    sampler = ResumableDistributedSampler(dataset, shuffle=True) if world > 1 else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        shuffle=sampler is None, num_workers=8, pin_memory=True,
                        persistent_workers=True, drop_last=True)
    val_loader = None
    if args.fid_every > 0:
        if args.dataset == "ffhq":
            val_dataset = FlatImages(args.data, transform=val_image_transform())
            # A full-dataset cache defines the matching real-reference corpus.
            if args.token_cache is not None:
                if len(val_dataset) < len(dataset):
                    raise ValueError(
                        f"FFHQ image corpus ({len(val_dataset):,}) is smaller than "
                        f"the token cache ({len(dataset):,})"
                    )
                val_dataset = Subset(val_dataset, range(len(dataset)))
        else:
            fid_real_split = args.fid_real_split or (
                "train" if args.dataset == "celebahq" else "val"
            )
            val_dataset = datasets.ImageFolder(
                args.data / fid_real_split, transform=val_image_transform()
            )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world, rank=rank(), shuffle=False, drop_last=False
        ) if world > 1 else None
        val_loader = DataLoader(
            val_dataset, batch_size=args.fid_batch_size, sampler=val_sampler,
            shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True,
        )
    total_vocab_size = args.num_atoms + args.coeff_vocab_size
    num_condition_classes = 1 if args.model_preset == "ffhq-350m" else 1000
    aux = LaserAux(args.checkpoint, args.num_atoms, args.coeff_vocab_size,
                   args.coeff_max, args.coeff_scale,
                   attn_resolutions=((16,) if args.dataset in face_datasets else (8,)),
                   coeff_scales=args.coeff_scales,
                   soft_target_physical=args.coeff_scales is not None,
                   coeff_bin_centers=cached_bin_centers,
                   sparsity_level=args.sparsity_level).to(device)
    unwrapped_model = build_model(
        total_vocab_size, args.num_atoms, compound=args.compound_tokens,
        coeff_vocab_size=args.coeff_vocab_size,
        compound_refiner_layers=args.compound_refiner_layers,
        compound_geometry_head=(
            args.geometry_loss_weight > 0 and not args.compound_distribution_geometry
        ),
        compound_micro_transformer_layers=args.compound_micro_transformer_layers,
        compound_depth_specific_coeff_heads=args.compound_depth_specific_coeff_heads,
        sparsity_level=args.sparsity_level,
        model_preset=args.model_preset,
    )
    parameter_names = [name for name, _ in unwrapped_model.named_parameters()]

    resume_payload = None
    resume_optimizer_state = None
    checkpoint_exists = args.resume and resume_checkpoint.is_file()
    if checkpoint_exists:
        should_load = args.distributed_backend != "fsdp" or rank() == 0
        if should_load:
            raw_payload = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
            unwrapped_model.load_state_dict(raw_payload["state_dict"], strict=True)
            resume_optimizer_state = raw_payload["optimizer"]
            resume_payload = {
                key: value for key, value in raw_payload.items()
                if key not in ("state_dict", "optimizer")
            }
            del raw_payload
            if args.distributed_backend == "fsdp":
                resume_optimizer_state = optimizer_state_to_names(
                    resume_optimizer_state, unwrapped_model
                )
        if args.distributed_backend == "fsdp":
            metadata = [resume_payload]
            dist.broadcast_object_list(metadata, src=0)
            resume_payload = metadata[0]

    unwrapped_model = unwrapped_model.to(device)
    model = wrap_distributed_model(
        unwrapped_model, args.distributed_backend, device, world
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4,
        betas=(0.9, 0.95), fused=True,
    )
    if checkpoint_exists:
        if is_fsdp_model(model):
            sharded_optimizer_state = FSDP.scatter_full_optim_state_dict(
                resume_optimizer_state if rank() == 0 else None,
                model,
                optim=optimizer,
            )
            optimizer.load_state_dict(sharded_optimizer_state)
            del sharded_optimizer_state
        else:
            optimizer.load_state_dict(
                optimizer_state_for_unwrapped_load(resume_optimizer_state, unwrap(model))
            )
        del resume_optimizer_state
    accumulation = args.total_batch_size // (args.batch_size * world)
    if accumulation * args.batch_size * world != args.total_batch_size:
        raise ValueError("total batch size must be divisible by per-step global batch size")
    complete_microbatches = (len(loader) // accumulation) * accumulation
    optimizer_steps_per_epoch = complete_microbatches // accumulation
    if optimizer_steps_per_epoch <= 0:
        raise ValueError("training loader does not contain a complete optimizer step")
    use_wandb = rank() == 0 and args.wandb_mode != "disabled"
    wb = None
    if use_wandb:
        import wandb
        wb = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                        name=args.wandb_name,
                        id=args.wandb_id, resume="allow" if args.wandb_id else None,
                        mode=args.wandb_mode,
                        config={**vars(args), "architecture": (
                            f"compound-v4-micro{args.compound_micro_transformer_layers}-rqtransformer-{args.model_preset}"
                            if args.compound_tokens and args.compound_micro_transformer_layers > 0
                            else f"compound-v3-refiner{args.compound_refiner_layers}-rqtransformer-{args.model_preset}"
                            if args.compound_tokens and args.compound_refiner_layers > 0
                            else f"compound-rqtransformer-{args.model_preset}" if args.compound_tokens
                            else f"official-rqtransformer-{args.model_preset}"
                        ),
                                "stochastic_codes": args.coeff_target_mode == "soft",
                                "temp": args.coeff_target_temperature, "top_p": 0.92,
                                "coefficient_quantizer": (
                                    None if cache_meta is None
                                    else cache_meta.get("coeff_quantization", "uniform")
                                ),
                                "preview_sampling_settings": preview_sampling_settings(args)})
        wb.define_metric("train/global_step")
        metric_names = (
            "train/loss", "train/atom_nll", "train/coeff_cross_entropy",
            "train/coeff_target_entropy", "train/coeff_kl",
            "train/classification_loss", "train/geometry_loss",
            "train/geometry_weight",
            "train/geometry_pair_mse", "train/geometry_spatial_mse",
            "train/atom_top1", "train/coeff_bin_mae", "train/grad_norm",
            "train/images_per_second", "train/lr", "train/epoch", "val/fid",
        )
        for depth_index in range(args.sparsity_level):
            metric_names += (
                f"train/atom_nll_depth{depth_index}",
                f"train/atom_top1_depth{depth_index}",
                f"train/coeff_cross_entropy_depth{depth_index}",
                f"train/coeff_bin_mae_depth{depth_index}",
            )
        if uses_inception_score(args.dataset):
            metric_names += ("val/inception_score", "val/inception_score_std")
        for metric_name in metric_names:
            wb.define_metric(metric_name, step_metric="train/global_step")
        (args.output / "launch_config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
        if args.upload_token_cache:
            if args.token_cache is None:
                raise ValueError("--upload-token-cache requires --token-cache")
            upload_token_cache_once(wb, args.token_cache, args.output)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    global_step = 0
    start_epoch = 0
    resume_batch_idx = 0
    best_fid = []
    best_inception = []
    if resume_payload is not None:
        global_step = int(resume_payload.get("global_step", 0))
        start_epoch = int(resume_payload.get("epoch", 0))
        resume_batch_idx = int(resume_payload.get("batch_idx", 0))
        best_fid = [(float(x[0]), str(x[1])) for x in resume_payload.get("best_fid", [])]
        best_inception = [
            (float(x[0]), str(x[1])) for x in resume_payload.get("best_inception", [])
        ]
        if not uses_inception_score(args.dataset):
            best_inception = []
        saved_config = resume_payload.get("config", {})
        old_batch_size = int(saved_config.get("batch_size", args.batch_size))
        if resume_batch_idx and old_batch_size != args.batch_size:
            resume_batch_idx = (resume_batch_idx * old_batch_size) // args.batch_size
        if int(saved_config.get("fid_num_samples", -1)) != args.fid_num_samples:
            # FIDs from different real/fake sample counts are not comparable.
            best_fid = []
            best_inception = []
            if rank() == 0:
                print("Reset prior best-FID history because the evaluation "
                      f"protocol changed from {saved_config.get('fid_num_samples')} "
                      f"to {args.fid_num_samples} samples", flush=True)
        if rank() == 0:
            print(f"Resumed from {resume_checkpoint}: epoch={start_epoch}, "
                  f"batch={resume_batch_idx}, step={global_step}", flush=True)
    scheduler = None
    if args.lr_schedule == "cosine":
        schedule_steps = args.lr_schedule_epochs * optimizer_steps_per_epoch
        scheduler_state = None if resume_payload is None else resume_payload.get("scheduler")
        if scheduler_state is None:
            # Fresh runs start at the requested base LR. Legacy resumptions are
            # mapped onto the curve using their already-completed global steps.
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr
        scheduler = create_cosine_lr_scheduler(
            optimizer,
            initial_lr=args.lr,
            min_lr=args.min_lr,
            total_steps=schedule_steps,
            completed_steps=global_step,
            state_dict=scheduler_state,
        )
        if rank() == 0:
            source = "checkpointed" if scheduler_state is not None else (
                "legacy-backfilled" if global_step else "fresh"
            )
            print(
                f"Cosine LR schedule ({source}): step={global_step}/{schedule_steps}, "
                f"lr={optimizer.param_groups[0]['lr']:.8g}, min_lr={args.min_lr:.8g}",
                flush=True,
            )
    elif resume_payload is not None:
        # Constant-LR resumes intentionally honor the new launch value rather
        # than whatever value was serialized by a previous schedule.
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr
    optimizer.zero_grad(set_to_none=True)
    if args.sample_grid_on_start:
        if dist.is_initialized():
            dist.barrier()
        with optimizer_state_offloaded_for_generation(model, optimizer, device):
            with model_for_custom_methods(model) as sampling_model:
                if rank() == 0:
                    for setting in preview_sampling_settings(args):
                        target = sample_class_grid(
                            sampling_model, aux, class_names, args.output,
                            global_step, wb=wb,
                            num_condition_classes=num_condition_classes,
                            num_samples=args.sample_grid_size,
                            sample_batch_size=args.sample_grid_batch_size,
                            **setting,
                        )
                        print(f"Saved startup samples: {target}", flush=True)
                if is_fsdp_model(model):
                    dist.barrier()
        if dist.is_initialized():
            dist.barrier()
    if args.generation_smoke_test:
        with optimizer_state_offloaded_for_generation(model, optimizer, device):
            torch.cuda.reset_peak_memory_stats(device)
            with model_for_custom_methods(model) as generation_model:
                generation_model.eval()
                labels = torch.arange(args.fid_batch_size, device=device).remainder(
                    num_condition_classes
                )
                if isinstance(generation_model, CompoundLaserRQTransformer):
                    atoms, coeff_ids = generation_model.sample_compound(
                        args.fid_batch_size,
                        aux,
                        cond=labels,
                        atom_temperature=args.atom_temperature,
                        atom_top_k=args.atom_top_k or aux.num_atoms,
                        atom_top_p=args.atom_top_p,
                        coeff_temperature=args.coeff_temperature,
                        coeff_top_k=args.coeff_top_k or aux.coeff_vocab_size,
                        coeff_top_p=args.coeff_top_p,
                        amp=True,
                    )
                    images = aux.decode_compound(atoms, coeff_ids)
                else:
                    partial = torch.zeros(
                        args.fid_batch_size, 8, 8, 4, device=device, dtype=torch.long
                    )
                    tokens = generation_model.sample(
                        partial,
                        model_aux=aux,
                        cond=labels,
                        temperature=1.0,
                        top_k=aux.num_atoms,
                        top_p=args.atom_top_p,
                        amp=True,
                        cached=True,
                        is_tqdm=False,
                    )
                    images = aux.decode_tokens(tokens)
                if not torch.isfinite(images).all():
                    raise RuntimeError("generation smoke test produced non-finite pixels")
                del images
            cuda_memory_report(device, f"generation smoke batch {args.fid_batch_size}")
        if rank() == 0:
            print("Generation smoke test passed", flush=True)
        if wb is not None:
            wb.finish()
        return
    last_perf_step = global_step
    last_perf_time = time.monotonic()
    launch_start_step = global_step
    memory_reported = False
    stop_training = False
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, args.epochs):
        batch_offset = resume_batch_idx if epoch == start_epoch else 0
        if sampler is not None:
            sampler.set_epoch(epoch)
            sampler.set_start_index(batch_offset * args.batch_size)
        model.train()
        for batch_idx, batch in enumerate(loader):
            absolute_batch_idx = batch_idx + batch_offset
            if batch_idx >= complete_microbatches:
                break
            if args.token_cache:
                atoms, coeffs, labels = batch
                atoms = atoms.to(device, non_blocking=True)
                coeffs = coeffs.to(device, non_blocking=True)
                labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
                if num_condition_classes == 1:
                    labels.zero_()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    if args.compound_tokens:
                        coeff_ids, target_coeff_probs = aux.compound_coeff_ids(
                            coeffs,
                            temp=args.coeff_target_temperature,
                            stochastic=args.coeff_target_mode == "soft",
                            hard=args.coeff_target_mode == "hard",
                        )
                        tokens = atoms.long() * args.coeff_vocab_size + coeff_ids
                        target_physical = aux.physical_contributions(atoms, coeffs)
                        compact_targets = (atoms.long(), target_coeff_probs, target_physical)
                    else:
                        tokens, compact_targets = aux.sparse_targets(
                            atoms,
                            coeffs,
                            temp=args.coeff_target_temperature,
                            stochastic=args.coeff_target_mode == "soft",
                            compact=True,
                            hard=args.coeff_target_mode == "hard",
                        )
            else:
                images, labels = batch
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                if num_condition_classes == 1:
                    labels.zero_()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    if args.compound_tokens:
                        atoms, coeffs = aux.encode_sparse_components(images)
                        coeff_ids, target_coeff_probs = aux.compound_coeff_ids(
                            coeffs,
                            temp=args.coeff_target_temperature,
                            stochastic=args.coeff_target_mode == "soft",
                            hard=args.coeff_target_mode == "hard",
                        )
                        tokens = atoms.long() * args.coeff_vocab_size + coeff_ids
                        target_physical = aux.physical_contributions(atoms, coeffs)
                        compact_targets = (atoms.long(), target_coeff_probs, target_physical)
                    else:
                        tokens, soft_targets = aux.encode_sparse(
                            images,
                            temp=args.coeff_target_temperature,
                            stochastic=args.coeff_target_mode == "soft",
                            hard=args.coeff_target_mode == "hard",
                        )
            sync = args.smoke_test or ((absolute_batch_idx + 1) % accumulation == 0)
            # FSDP deliberately reduce-scatters every microbatch. Its no_sync()
            # retains full, unsharded gradients and is too memory-hungry for a
            # 1.45B-parameter model on these 20 GB cards.
            ctx = model.no_sync() if isinstance(model, DDP) and not sync else torch.enable_grad()
            with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(tokens, model_aux=aux, cond=labels, amp=False)
                diagnostic_metrics = None
                if args.token_cache or args.compound_tokens:
                    if isinstance(logits, dict):
                        atom_logits = logits["atom_logits"]
                        coeff_logits = logits["coeff_logits"]
                    else:
                        atom_logits = logits[..., 0::2, :args.num_atoms]
                        coeff_logits = logits[..., 1::2, args.num_atoms:]
                    if args.compound_tokens:
                        target_atoms, target_coeff_probs, target_physical = compact_targets
                        progress_epochs = epoch + (
                            min(absolute_batch_idx + 1, complete_microbatches)
                            / complete_microbatches
                        )
                        geometry_weight = scheduled_geometry_weight(
                            args.geometry_loss_weight,
                            progress_epochs,
                            args.geometry_start_epoch,
                            args.geometry_warmup_epochs,
                        )
                        loss, objective = compound_objective(
                            atom_logits,
                            coeff_logits,
                            logits.get("physical_contribution") if isinstance(logits, dict) else None,
                            target_atoms,
                            target_coeff_probs,
                            target_physical,
                            atom_weight=args.atom_loss_weight,
                            geometry_weight=geometry_weight,
                            accumulation=accumulation,
                            distribution_geometry=args.compound_distribution_geometry,
                            geometry_dictionary=aux.dictionary,
                            geometry_coeff_bins=aux.coeff_bins,
                            geometry_coeff_scales=aux.coeff_scales,
                            geometry_top_k=args.geometry_top_k,
                        )
                        atom_loss = objective["atom_nll"]
                        coeff_loss = objective["coeff_cross_entropy"]
                        with torch.no_grad():
                            coeff_entropy = -(
                                target_coeff_probs * target_coeff_probs.clamp_min(1e-30).log()
                            ).sum(dim=-1)
                            pred_coeff_ids = coeff_logits.argmax(dim=-1)
                            target_coeff_ids = target_coeff_probs.argmax(dim=-1)
                            pair_scales = aux.coeff_scales.view(
                                1, 1, 1, args.sparsity_level
                            )
                            pred_values = aux.coeff_bins[pred_coeff_ids] * pair_scales
                            target_values = aux.coeff_bins[target_coeff_ids] * pair_scales
                            diagnostic_metrics = {
                                "train/atom_nll": float(atom_loss.mean()),
                                "train/coeff_cross_entropy": float(coeff_loss.mean()),
                                "train/coeff_target_entropy": float(coeff_entropy.mean()),
                                "train/coeff_kl": float((coeff_loss - coeff_entropy).mean()),
                                "train/classification_loss": float(objective["classification"]),
                                "train/geometry_loss": float(objective["geometry"]),
                                "train/geometry_weight": geometry_weight,
                                "train/geometry_pair_mse": float(objective["geometry_pair_mse"]),
                                "train/geometry_spatial_mse": float(objective["geometry_spatial_mse"]),
                                "train/atom_top1": float(
                                    (atom_logits.argmax(dim=-1) == target_atoms.long()).float().mean()
                                ),
                                "train/coeff_bin_mae": float((pred_values - target_values).abs().mean()),
                            }
                            for depth_index in range(target_atoms.shape[-1]):
                                diagnostic_metrics.update({
                                    f"train/atom_nll_depth{depth_index}": float(
                                        atom_loss[..., depth_index].mean()
                                    ),
                                    f"train/atom_top1_depth{depth_index}": float(
                                        (
                                            atom_logits[..., depth_index, :].argmax(dim=-1)
                                            == target_atoms[..., depth_index].long()
                                        ).float().mean()
                                    ),
                                    f"train/coeff_cross_entropy_depth{depth_index}": float(
                                        coeff_loss[..., depth_index].mean()
                                    ),
                                    f"train/coeff_bin_mae_depth{depth_index}": float(
                                        (
                                            pred_values[..., depth_index]
                                            - target_values[..., depth_index]
                                        ).abs().mean()
                                    ),
                                })
                    else:
                        target_atoms, target_coeff_probs = compact_targets
                        atom_log_probs = F.log_softmax(atom_logits.float(), dim=-1)
                        coeff_log_probs = F.log_softmax(coeff_logits.float(), dim=-1)
                        atom_loss = -atom_log_probs.gather(
                            -1, target_atoms.long().unsqueeze(-1)
                        ).squeeze(-1)
                        coeff_loss = -(target_coeff_probs * coeff_log_probs).sum(dim=-1)
                        depth = target_atoms.shape[-1]
                        loss = (
                            atom_loss.sum(dim=-1) + coeff_loss.sum(dim=-1)
                        ).mean() / (2 * depth * accumulation)
                else:
                    log_probs = F.log_softmax(logits.float(), dim=-1)
                    loss = -(soft_targets * log_probs).sum(dim=-1).mean() / accumulation
            loss.backward()
            if sync:
                grad_norm = (
                    model.clip_grad_norm_(1.0)
                    if is_fsdp_model(model)
                    else torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if not memory_reported:
                    memory_values = cuda_memory_report(
                        device, f"first optimizer step {global_step}"
                    )
                    if wb is not None:
                        memory_payload = {"train/global_step": global_step}
                        for memory_rank, (allocated, reserved) in enumerate(memory_values):
                            memory_payload.update({
                                f"system/rank{memory_rank}_peak_allocated_gib": allocated,
                                f"system/rank{memory_rank}_peak_reserved_gib": reserved,
                            })
                        wb.log(memory_payload)
                    memory_reported = True
                if wb is not None and global_step % 10 == 0:
                    now = time.monotonic()
                    elapsed = max(now - last_perf_time, 1e-6)
                    completed_steps = max(global_step - last_perf_step, 1)
                    payload = {
                        "train/loss": float(loss.detach()) * accumulation,
                        "train/epoch": epoch, "train/global_step": global_step,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/grad_norm": float(grad_norm),
                        "train/images_per_second": completed_steps * args.total_batch_size / elapsed,
                    }
                    if diagnostic_metrics is not None:
                        payload.update(diagnostic_metrics)
                    wb.log(payload)
                    last_perf_step, last_perf_time = global_step, now
                if args.save_step_freq > 0 and global_step % args.save_step_freq == 0:
                    if dist.is_initialized():
                        dist.barrier()
                    model_state, optimizer_state = full_checkpoint_states(
                        model, optimizer, parameter_names
                    )
                    if rank() == 0:
                        recovery_snapshot = {
                            "epoch": epoch,
                            "batch_idx": absolute_batch_idx + 1,
                            "global_step": global_step,
                            "fid": None,
                            "inception_score": None,
                            "inception_score_std": None,
                            "state_dict": model_state,
                            "optimizer": optimizer_state,
                            "scheduler": None if scheduler is None else scheduler.state_dict(),
                            "config": vars(args),
                            "best_fid": best_fid,
                            "best_inception": best_inception,
                        }
                        atomic_torch_save(recovery_snapshot, last_checkpoint)
                        print(
                            f"Step {global_step}: saved recovery checkpoint {last_checkpoint}",
                            flush=True,
                        )
                        if (
                            args.upload_checkpoints
                            and args.checkpoint_upload_mode == "files"
                        ):
                            upload_selected_checkpoint_files(
                                wb,
                                last_checkpoint=last_checkpoint,
                                best_fid=best_fid,
                                upload_dir=args.output / "wandb_checkpoints",
                            )
                        del recovery_snapshot
                    del model_state, optimizer_state
                    if dist.is_initialized():
                        dist.barrier()
                if args.sample_grid_every > 0 and global_step % args.sample_grid_every == 0:
                    if dist.is_initialized():
                        dist.barrier()
                    with optimizer_state_offloaded_for_generation(model, optimizer, device):
                        with model_for_custom_methods(model) as sampling_model:
                            if rank() == 0:
                                for setting in preview_sampling_settings(args):
                                    target = sample_class_grid(
                                        sampling_model, aux, class_names, args.output,
                                        global_step, wb=wb,
                                        num_condition_classes=num_condition_classes,
                                        num_samples=args.sample_grid_size,
                                        sample_batch_size=args.sample_grid_batch_size,
                                        **setting,
                                    )
                                    print(f"Saved preview samples: {target}", flush=True)
                            if is_fsdp_model(model):
                                dist.barrier()
                    if dist.is_initialized():
                        dist.barrier()
                if (
                    args.max_optimizer_steps > 0
                    and global_step - launch_start_step >= args.max_optimizer_steps
                ):
                    stop_training = True
                    break
        if stop_training:
            if rank() == 0:
                print(
                    f"Stopped after {global_step - launch_start_step} optimizer step(s) "
                    "as requested; skipped epoch evaluation and checkpointing",
                    flush=True,
                )
            break
        resume_batch_idx = 0
        if sampler is not None:
            sampler.set_start_index(0)
        if dist.is_initialized():
            dist.barrier()
        run_fid = args.fid_every > 0 and (epoch + 1) % args.fid_every == 0
        # Every evaluated model must be recoverable, even when the FID cadence
        # does not coincide with the periodic recovery-checkpoint cadence.
        save_epoch = (
            run_fid
            or (epoch + 1) % args.save_ckpt_freq == 0
            or epoch + 1 == args.epochs
        )
        fid = None
        inception_score = None
        inception_score_std = None
        if run_fid:
            with optimizer_state_offloaded_for_generation(model, optimizer, device):
                torch.cuda.reset_peak_memory_stats(device)
                with model_for_custom_methods(model) as generation_model:
                    fid, inception_score, inception_score_std = evaluate_generation_metrics(
                        generation_model, aux, val_loader, args.fid_num_samples,
                        args.fid_batch_size,
                        num_condition_classes=num_condition_classes,
                        atom_temperature=args.atom_temperature,
                        atom_top_k=args.atom_top_k,
                        atom_top_p=args.atom_top_p,
                        coeff_temperature=args.coeff_temperature,
                        coeff_top_k=args.coeff_top_k,
                        coeff_top_p=args.coeff_top_p,
                        compute_inception_score=uses_inception_score(args.dataset),
                    )
                cuda_memory_report(device, f"epoch {epoch + 1} FID generation")
        best_path = None
        best_inception_path = None
        if rank() == 0:
            if wb is not None and fid is not None:
                evaluation_payload = {
                    "val/fid": fid,
                    "train/epoch": epoch + 1,
                    "train/global_step": global_step,
                }
                if inception_score is not None:
                    evaluation_payload.update({
                        "val/inception_score": inception_score,
                        "val/inception_score_std": inception_score_std,
                    })
                wb.log(evaluation_payload)
            qualifies = fid is not None and (
                len(best_fid) < 3 or fid < max(x[0] for x in best_fid)
            )
            if qualifies:
                best_path = checkpoint_dir / f"best_fid_{fid:.4f}_epoch_{epoch + 1:03d}.pt"
                best_fid.append((fid, str(best_path)))
                best_fid.sort(key=lambda item: item[0])
                while len(best_fid) > 3:
                    _, stale = best_fid.pop()
                    stale_path = Path(stale)
                    if stale_path.is_file():
                        stale_path.unlink()
            qualifies_inception = inception_score is not None and (
                len(best_inception) < 3
                or inception_score > min(x[0] for x in best_inception)
            )
            if qualifies_inception:
                best_inception_path = checkpoint_dir / (
                    f"best_is_{inception_score:.4f}_epoch_{epoch + 1:03d}.pt"
                )
                best_inception.append((inception_score, str(best_inception_path)))
                best_inception.sort(key=lambda item: item[0], reverse=True)
                while len(best_inception) > 3:
                    _, stale = best_inception.pop()
                    stale_path = Path(stale)
                    if stale_path.is_file():
                        stale_path.unlink()
        if save_epoch:
            model_state, optimizer_state = full_checkpoint_states(
                model, optimizer, parameter_names
            )
            if rank() == 0:
                snapshot = {
                    "epoch": epoch + 1, "global_step": global_step, "fid": fid,
                    "inception_score": inception_score,
                    "inception_score_std": inception_score_std,
                    "state_dict": model_state, "optimizer": optimizer_state,
                    "scheduler": None if scheduler is None else scheduler.state_dict(),
                    "config": vars(args), "best_fid": best_fid,
                    "best_inception": best_inception,
                }
                atomic_torch_save(snapshot, last_checkpoint)
                if best_path is not None:
                    snapshot_checkpoint(last_checkpoint, best_path)
                if best_inception_path is not None:
                    snapshot_checkpoint(last_checkpoint, best_inception_path)
                # Every scheduled full save uploads a recoverable `last` plus
                # every locally retained top-three FID and Inception member.
                if args.upload_checkpoints and args.checkpoint_upload_mode == "artifact":
                    artifact_aliases = ["latest"]
                    if best_path is not None:
                        artifact_aliases.extend(["best-fid", f"fid-epoch-{epoch + 1}"])
                    if best_inception_path is not None:
                        artifact_aliases.extend(["best-is", f"is-epoch-{epoch + 1}"])
                    selected_paths = [last_checkpoint]
                    selected_paths.extend(
                        path for _, saved_path in best_fid
                        if (path := Path(saved_path)).is_file() and path != last_checkpoint
                    )
                    selected_paths.extend(
                        path for _, saved_path in best_inception
                        if (path := Path(saved_path)).is_file()
                        and path not in selected_paths
                    )
                    upload_checkpoints(
                        wb, selected_paths, artifact_name=f"{wb.id}-checkpoint",
                        aliases=artifact_aliases,
                        metadata={
                            "epoch": epoch + 1,
                            "step": global_step,
                            "fid": fid,
                            "inception_score": inception_score,
                            "inception_score_std": inception_score_std,
                            "selected_checkpoints": [path.name for path in selected_paths],
                        },
                    )
                elif args.upload_checkpoints:
                    upload_selected_checkpoint_files(
                        wb,
                        last_checkpoint=last_checkpoint,
                        best_fid=best_fid,
                        upload_dir=args.output / "wandb_checkpoints",
                    )
                del snapshot
            del model_state, optimizer_state
        if rank() == 0:
            if not save_epoch:
                print(f"Epoch {epoch + 1}: checkpoint skipped", flush=True)
            elif fid is None:
                print(f"Epoch {epoch + 1}: FID skipped; saved {last_checkpoint}", flush=True)
            else:
                metrics = f"FID={fid:.4f}"
                if inception_score is not None:
                    metrics += f"; IS={inception_score:.4f}+/-{inception_score_std:.4f}"
                print(f"Epoch {epoch + 1}: {metrics}; saved {last_checkpoint}", flush=True)
        if dist.is_initialized():
            dist.barrier()
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
