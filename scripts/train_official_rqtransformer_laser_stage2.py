#!/usr/bin/env python3
"""Train the vendored KakaoBrain RQ-Transformer on LASER sparse pairs.

The transformer implementation lives in ``src.models.rqtransformer``.
Only the stage-1 auxiliary embedding is adapted: OMP atom ids use the learned
LASER dictionary and real coefficients are uniformly discretized into the same
16K per-depth vocabulary used by the official shared classifier.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import make_dataclass
from datetime import timedelta
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
import matplotlib.pyplot as plt
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.nodes import AnyNode, BooleanNode, BytesNode, FloatNode, IntegerNode, StringNode
from src.models.rqtransformer.configs import RQTransformerConfig
from src.models.rqtransformer.attentions import AttentionStack
from src.models.rqtransformer.transformers import RQTransformer, sample_from_logits
from src.models.rqvae.rqvae import RQVAE
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


def unwrap(model):
    return model.module if isinstance(model, DDP) else model


def persistent_checkpoint_dir(output: Path, configured: Path | None) -> Path:
    """Resolve checkpoint storage and reject ephemeral/out-of-workspace targets."""
    target = (configured or (output / "checkpoints")).expanduser().resolve()
    workspace = Path("/workspace")
    if workspace.is_dir() and not target.is_relative_to(workspace.resolve()):
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
                 soft_target_physical=False, clamp_coeffs=True):
        super().__init__()
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
            latent_shape=[8, 8, 256], code_shape=[8, 8, 2],
            shared_codebook=True, restart_unused_codes=True,
        )
        payload = load_stage1_checkpoint(checkpoint)
        state = payload["state_dict"]
        filtered = {k: v for k, v in state.items() if not k.startswith("quantizer.")}
        missing, unexpected = stage1.load_state_dict(filtered, strict=False)
        bad_missing = [k for k in missing if not k.startswith("quantizer.")]
        if bad_missing or unexpected:
            raise RuntimeError(f"stage-1 mismatch: missing={bad_missing}, unexpected={unexpected}")
        self.encoder = stage1.encoder
        self.quant_conv = stage1.quant_conv
        self.post_quant_conv = stage1.post_quant_conv
        self.decoder = stage1.decoder
        self.register_buffer("dictionary", F.normalize(state["quantizer.dictionary"].float(), dim=0))
        self.register_buffer("coeff_bins", torch.linspace(-coeff_max, coeff_max, coeff_vocab_size))
        self.num_atoms = int(num_atoms)
        self.coeff_vocab_size = int(coeff_vocab_size)
        self.vocab_size = self.num_atoms + self.coeff_vocab_size
        self.coeff_max = float(coeff_max)
        scales = coeff_scales if coeff_scales is not None else [coeff_scale, coeff_scale]
        if len(scales) != 2:
            raise ValueError("LASER k=2 requires exactly two coefficient scales")
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
        dictionary = self.dictionary
        gram = dictionary.t() @ dictionary
        corr0 = signals @ dictionary
        first = corr0.abs().argmax(dim=1)
        a = gram[first, first].clamp_min(1e-6)
        c1 = corr0.gather(1, first[:, None]).squeeze(1) / a
        residual_corr = corr0 - c1[:, None] * gram[first]
        residual_corr.scatter_(1, first[:, None], 0.0)
        second = residual_corr.abs().argmax(dim=1)
        g11, g22, g12 = gram[first, first], gram[second, second], gram[first, second]
        y1 = corr0.gather(1, first[:, None]).squeeze(1)
        y2 = corr0.gather(1, second[:, None]).squeeze(1)
        det = (g11 * g22 - g12.square()).clamp_min(1e-6)
        v1 = (g22 * y1 - g12 * y2) / det
        v2 = (g11 * y2 - g12 * y1) / det
        atoms = torch.stack((first, second), dim=-1).view(b, h, w, 2)
        physical_coeffs = torch.stack((v1, v2), dim=-1).view(b, h, w, 2)
        scales = self.coeff_scales.view(1, 1, 1, 2)
        coeffs = physical_coeffs / scales
        if self.clamp_coeffs:
            coeffs = coeffs.clamp(-self.coeff_max, self.coeff_max)
        return atoms, coeffs

    @torch.no_grad()
    def sparse_targets(self, atoms: torch.Tensor, coeffs: torch.Tensor, *, temp: float = 0.5,
                       stochastic: bool = True, compact: bool = False):
        atoms = atoms.long()
        coeffs = coeffs.float()
        scaled = (coeffs + self.coeff_max) * ((self.coeff_vocab_size - 1) / (2 * self.coeff_max))
        coeff_tokens = scaled.round().long().clamp(0, self.coeff_vocab_size - 1)
        # The official stage-2 recipe trains against stage-1 soft codes.  For
        # LASER, atom support is discrete OMP while the continuous coefficient
        # posterior is discretized into a temperature-controlled 16K density.
        if self.soft_target_physical:
            target_values = (coeffs * self.coeff_scales.view(1, 1, 1, 2))[..., None]
            bin_values = self.coeff_bins.view(1, 1, 1, 1, -1) * self.coeff_scales.view(1, 1, 1, 2, 1)
        else:
            target_values = coeffs[..., None]
            bin_values = self.coeff_bins
        coeff_logits = -(target_values - bin_values).square() / max(float(temp), 1e-6)
        coeff_probs = coeff_logits.softmax(dim=-1)
        if stochastic:
            coeff_tokens = torch.multinomial(
                coeff_probs.reshape(-1, self.coeff_vocab_size), 1
            ).reshape_as(coeff_tokens)
        b, h, w, _ = atoms.shape
        tokens = torch.empty(b, h, w, 4, device=atoms.device, dtype=torch.long)
        tokens[..., 0::2] = atoms
        tokens[..., 1::2] = coeff_tokens + self.num_atoms
        if compact:
            return tokens, (atoms, coeff_probs)
        soft_targets = torch.zeros(b, h, w, 4, self.vocab_size,
                                   device=atoms.device, dtype=coeff_probs.dtype)
        soft_targets[..., 0, :].scatter_(-1, atoms[..., 0, None], 1.0)
        soft_targets[..., 2, :].scatter_(-1, atoms[..., 1, None], 1.0)
        soft_targets[..., 1::2, self.num_atoms:] = coeff_probs
        return tokens, soft_targets

    @torch.no_grad()
    def encode_sparse(self, images: torch.Tensor, *, temp: float = 0.5, stochastic: bool = True):
        atoms, coeffs = self.encode_sparse_components(images)
        return self.sparse_targets(atoms, coeffs, temp=temp, stochastic=stochastic)

    @torch.no_grad()
    def get_code_emb_with_depth(self, tokens: torch.Tensor):
        out = torch.empty(*tokens.shape, 256, device=tokens.device, dtype=self.dictionary.dtype)
        atom_vectors = self.dictionary.t()[tokens[..., 0::2]]
        out[..., 0::2, :] = atom_vectors
        coeff_ids = (tokens[..., 1::2] - self.num_atoms).clamp(0, self.coeff_vocab_size - 1)
        coeff = self.coeff_bins[coeff_ids] * self.coeff_scales.view(1, 1, 1, 2)
        out[..., 1::2, :] = (coeff[..., None] - 1.0) * atom_vectors
        return out, None

    @torch.no_grad()
    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        atoms = tokens[..., 0::2].long()
        coeff_ids = (tokens[..., 1::2].long() - self.num_atoms).clamp(0, self.coeff_vocab_size - 1)
        coeffs = self.coeff_bins[coeff_ids] * self.coeff_scales.view(1, 1, 1, 2)
        atom_vectors = self.dictionary.t()[atoms]
        z_q = (atom_vectors * coeffs[..., None]).sum(dim=-2)
        z_q = self.post_quant_conv(z_q.permute(0, 3, 1, 2).contiguous())
        return self.decoder(z_q).clamp(-1.0, 1.0)

    @torch.no_grad()
    def compound_coeff_ids(self, coeffs: torch.Tensor, *, stochastic: bool = True,
                           temp: float = 0.5):
        """Quantize real coefficients for compound (atom, coefficient) events."""
        target_values = coeffs.float()[..., None]
        logits = -(target_values - self.coeff_bins).square() / max(float(temp), 1e-6)
        probs = logits.softmax(dim=-1)
        if stochastic:
            ids = torch.multinomial(probs.reshape(-1, self.coeff_vocab_size), 1)
            ids = ids.reshape(coeffs.shape)
        else:
            ids = probs.argmax(dim=-1)
        return ids.long(), probs

    @torch.no_grad()
    def compound_embeddings(self, atoms: torch.Tensor, coeff_ids: torch.Tensor):
        """Physical latent contribution of every compound sparse event."""
        atom_vectors = self.dictionary.t()[atoms.long()]
        coeffs = self.coeff_bins[coeff_ids.long().clamp(0, self.coeff_vocab_size - 1)]
        scale_shape = [1] * (coeffs.ndim - 1) + [2]
        coeffs = coeffs * self.coeff_scales.view(*scale_shape)
        return atom_vectors * coeffs[..., None]

    @torch.no_grad()
    def physical_contributions(self, atoms: torch.Tensor, coeffs: torch.Tensor):
        """Continuous physical LASER contribution c_i d_{a_i} for each pair."""
        atom_vectors = self.dictionary.t()[atoms.long()]
        scale_shape = [1] * (coeffs.ndim - 1) + [2]
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

@torch.no_grad()
def sample_class_grid(model, aux, class_names, output_dir: Path, step: int, wb=None,
                      atom_temperature=1.0, atom_top_p=0.92,
                      coeff_temperature=1.0, coeff_top_p=0.92):
    device = next(model.parameters()).device
    chosen = torch.randperm(1000, device=device)[:8]
    labels = chosen.repeat_interleave(8)
    was_training = model.training
    model.eval()
    if isinstance(model, CompoundLaserRQTransformer):
        atoms, coeff_ids = model.sample_compound(
            64, aux, cond=labels,
            atom_temperature=atom_temperature,
            atom_top_k=aux.num_atoms, atom_top_p=atom_top_p,
            coeff_temperature=coeff_temperature, coeff_top_p=coeff_top_p,
            amp=True,
        )
        images = (aux.decode_compound(atoms, coeff_ids).float().cpu() + 1.0) * 0.5
    else:
        partial = torch.zeros(64, 8, 8, 4, device=device, dtype=torch.long)
        tokens = model.sample(
            partial, model_aux=aux, cond=labels, temperature=1.0,
            top_k=16384, top_p=0.92, amp=True, cached=True, is_tqdm=False,
        )
        images = (aux.decode_tokens(tokens).float().cpu() + 1.0) * 0.5
    labels_cpu = labels.cpu().tolist()
    fig, axes = plt.subplots(8, 8, figsize=(20, 22))
    for index, axis in enumerate(axes.flat):
        axis.imshow(images[index].permute(1, 2, 0).clamp(0, 1).numpy())
        class_id = int(labels_cpu[index])
        label = class_names[class_id] if class_id < len(class_names) else f"class {class_id}"
        axis.set_title(f"{class_id}: {label}", fontsize=7)
        axis.axis("off")
    fig.suptitle(f"Class-conditional samples — optimizer step {step}", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    target = sample_dir / f"step_{step:07d}.png"
    fig.savefig(target, dpi=140)
    if wb is not None:
        import wandb
        wb.log({
            "samples/class_conditional_8x8": wandb.Image(str(target)),
            "train/global_step": step,
        })
    plt.close(fig)
    if was_training:
        model.train()
    return target


@torch.no_grad()
def evaluate_generation_metrics(model, aux, val_loader, num_samples: int, batch_size: int = 64,
                                atom_temperature=1.0, atom_top_p=0.92,
                                coeff_temperature=1.0, coeff_top_p=0.92):
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore

    device = next(model.parameters()).device
    world = dist.get_world_size() if dist.is_initialized() else 1
    process_rank = dist.get_rank() if dist.is_initialized() else 0
    local_samples = int(num_samples) // world + (process_rank < int(num_samples) % world)
    # Every rank accumulates its shard; compute() merges the sufficient
    # statistics, avoiding a costly image all-gather.
    fid_metric = FrechetInceptionDistance(
        feature=2048, normalize=True, sync_on_compute=dist.is_initialized()
    ).to(device)
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
        # Use the exact uniform ImageNet class prior. At 50K samples this emits
        # exactly 50 generations for each of the 1,000 classes.
        local_indices = torch.arange(
            generated, generated + current, device=device, dtype=torch.long
        )
        labels = (local_indices * world + process_rank).remainder(1000)
        if isinstance(model, CompoundLaserRQTransformer):
            atoms, coeff_ids = model.sample_compound(
                current, aux, cond=labels,
                atom_temperature=atom_temperature,
                atom_top_k=aux.num_atoms, atom_top_p=atom_top_p,
                coeff_temperature=coeff_temperature, coeff_top_p=coeff_top_p,
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
        inception_metric.update(images)
        generated += current
    fid = float(fid_metric.compute().item())
    inception_mean, inception_std = inception_metric.compute()
    if was_training:
        model.train()
    return fid, float(inception_mean.item()), float(inception_std.item())


def atomic_torch_save(payload, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)


def upload_checkpoints(wb, paths: list[Path], *, artifact_name: str, aliases, metadata):
    import wandb
    artifact = wandb.Artifact(artifact_name, type="model", metadata=metadata)
    for path in paths:
        artifact.add_file(str(path), name=path.name)
    wb.log_artifact(artifact, aliases=list(aliases))


def upload_checkpoint(wb, path: Path, *, artifact_name: str, aliases, metadata):
    """Backward-compatible single-checkpoint artifact upload."""
    upload_checkpoints(
        wb, [path], artifact_name=artifact_name, aliases=aliases, metadata=metadata
    )


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
                        atom_temperature=None, coeff_temperature=None, amp=True):
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
                                                  top_k=self.coeff_vocab_size, top_p=coeff_top_p)
                    atoms[:, h, w, d] = atom
                    coeff_ids[:, h, w, d] = coeff_id
                    packed[:, h, w, d] = atom * self.coeff_vocab_size + coeff_id
        self.init_cache()
        return atoms, coeff_ids

def build_model(total_vocab_size: int, num_atoms: int, *, compound=False,
                coeff_vocab_size=2048, compound_refiner_layers=0,
                compound_geometry_head=False,
                compound_micro_transformer_layers=0,
                compound_depth_specific_coeff_heads=False):
    cfg = OmegaConf.create({
        "type": "rq-transformer", "block_size": [8, 8, 2 if compound else 4], "embed_dim": 1536,
        "input_embed_dim": 256, "shared_tok_emb": True, "shared_cls_emb": True,
        "input_emb_vqvae": True, "head_emb_vqvae": True, "cumsum_depth_ctx": True,
        "vocab_size": num_atoms if compound else total_vocab_size,
        "vocab_size_cond": 1000, "block_size_cond": 1,
        "body": {"n_layer": 42, "block": {"n_head": 24}},
        "head": {"n_layer": 6, "block": {"n_head": 24}},
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
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--total-batch-size", type=int, default=2048)
    p.add_argument("--num-atoms", type=int, default=16384)
    p.add_argument("--coeff-vocab-size", type=int, default=2048)
    p.add_argument("--coeff-max", type=float, default=20.0)
    p.add_argument("--coeff-scale", type=float, default=6.4)
    p.add_argument(
        "--compound-tokens", action=argparse.BooleanOptionalAction, default=False,
        help="Use 128 compound (atom, coefficient) AR events instead of 256 scalar events",
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
    p.add_argument("--atom-top-p", type=float, default=0.92)
    p.add_argument("--coeff-temperature", type=float, default=1.0)
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
    p.add_argument("--wandb-name", default="imagenet-rqtransformer-laser-a16384-k2-stage2")
    p.add_argument("--wandb-id", default=None)
    p.add_argument("--fid-num-samples", type=int, default=2048)
    p.add_argument("--fid-batch-size", type=int, default=64)
    p.add_argument("--fid-every", type=int, default=1,
                   help="Run full FID every N epochs")
    p.add_argument("--save-ckpt-freq", type=int, default=2,
                   help="Save the full training checkpoint every N epochs")
    p.add_argument("--save-step-freq", type=int, default=0,
                   help="Atomically overwrite last.pt every N optimizer steps; 0 disables")
    p.add_argument("--checkpoint-dir", type=Path, default=None,
                   help="Persistent checkpoint directory under /workspace")
    p.add_argument("--sample-grid-every", type=int, default=100,
                   help="Generate the expensive 64-image class grid every N optimizer steps; 0 disables it")
    p.add_argument(
        "--upload-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Upload 16+ GB checkpoints as W&B artifacts (disabled by default)",
    )
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
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
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    if args.lr_schedule_epochs <= 0:
        raise ValueError("--lr-schedule-epochs must be positive")
    if not 0 <= args.min_lr <= args.lr:
        raise ValueError("--min-lr must be between zero and --lr")
    if args.lr_schedule == "cosine" and args.epochs > args.lr_schedule_epochs:
        raise ValueError("--epochs cannot exceed the global cosine schedule horizon")
    if args.geometry_loss_weight > 0 and not args.compound_tokens:
        raise ValueError("geometry contribution loss requires --compound-tokens")
    if args.compound_distribution_geometry and args.geometry_loss_weight <= 0:
        raise ValueError("distribution geometry requires a positive geometry loss weight")
    if args.compound_depth_specific_coeff_heads and not args.compound_tokens:
        raise ValueError("depth-specific coefficient heads require --compound-tokens")
    if args.atom_temperature <= 0 or args.coeff_temperature <= 0:
        raise ValueError("sampling temperatures must be positive")
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

    image_dataset = datasets.ImageFolder(args.data / "train", transform=image_transform())
    class_names = class_names_for_dataset("imagenet", image_dataset.classes)
    dataset = SparseTokenCacheDataset(args.token_cache) if args.token_cache else image_dataset
    sampler = ResumableDistributedSampler(dataset, shuffle=True) if world > 1 else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        shuffle=sampler is None, num_workers=8, pin_memory=True,
                        persistent_workers=True, drop_last=True)
    val_dataset = datasets.ImageFolder(args.data / "val", transform=val_image_transform())
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world, rank=rank(), shuffle=False, drop_last=False
    ) if world > 1 else None
    val_loader = DataLoader(
        val_dataset, batch_size=args.fid_batch_size, sampler=val_sampler,
        shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True,
    )
    total_vocab_size = args.num_atoms + args.coeff_vocab_size
    aux = LaserAux(args.checkpoint, args.num_atoms, args.coeff_vocab_size,
                   args.coeff_max, args.coeff_scale).to(device)
    model = build_model(
        total_vocab_size, args.num_atoms, compound=args.compound_tokens,
        coeff_vocab_size=args.coeff_vocab_size,
        compound_refiner_layers=args.compound_refiner_layers,
        compound_geometry_head=(
            args.geometry_loss_weight > 0 and not args.compound_distribution_geometry
        ),
        compound_micro_transformer_layers=args.compound_micro_transformer_layers,
        compound_depth_specific_coeff_heads=args.compound_depth_specific_coeff_heads,
    ).to(device)
    if world > 1:
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4,
        betas=(0.9, 0.95), fused=True,
    )
    accumulation = args.total_batch_size // (args.batch_size * world)
    if accumulation * args.batch_size * world != args.total_batch_size:
        raise ValueError("total batch size must be divisible by per-step global batch size")
    complete_microbatches = (len(loader) // accumulation) * accumulation
    optimizer_steps_per_epoch = complete_microbatches // accumulation
    if optimizer_steps_per_epoch <= 0:
        raise ValueError("training loader does not contain a complete optimizer step")
    use_wandb = rank() == 0
    wb = None
    if use_wandb:
        import wandb
        wb = wandb.init(project=args.wandb_project, name=args.wandb_name,
                        id=args.wandb_id, resume="allow" if args.wandb_id else None,
                        config={**vars(args), "architecture": (
                            f"compound-v4-micro{args.compound_micro_transformer_layers}-rqtransformer-1400M"
                            if args.compound_tokens and args.compound_micro_transformer_layers > 0
                            else f"compound-v3-refiner{args.compound_refiner_layers}-rqtransformer-1400M"
                            if args.compound_tokens and args.compound_refiner_layers > 0
                            else "compound-rqtransformer-1400M" if args.compound_tokens
                            else "official-rqtransformer-1400M"
                        ),
                                "stochastic_codes": True, "temp": 0.5, "top_p": 0.92})
        wb.define_metric("train/global_step")
        for metric_name in (
            "train/loss", "train/atom_nll", "train/coeff_cross_entropy",
            "train/coeff_target_entropy", "train/coeff_kl",
            "train/classification_loss", "train/geometry_loss",
            "train/geometry_weight",
            "train/geometry_pair_mse", "train/geometry_spatial_mse",
            "train/atom_top1", "train/coeff_bin_mae", "train/grad_norm",
            "train/atom_nll_depth0", "train/atom_nll_depth1",
            "train/atom_top1_depth0", "train/atom_top1_depth1",
            "train/coeff_cross_entropy_depth0", "train/coeff_cross_entropy_depth1",
            "train/coeff_bin_mae_depth0", "train/coeff_bin_mae_depth1",
            "train/images_per_second", "train/lr", "train/epoch", "val/fid",
            "val/inception_score", "val/inception_score_std",
        ):
            wb.define_metric(metric_name, step_metric="train/global_step")
        (args.output / "launch_config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    checkpoint_dir = persistent_checkpoint_dir(args.output, args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint = checkpoint_dir / "last.pt"
    resume_checkpoint = args.resume_checkpoint or last_checkpoint
    global_step = 0
    start_epoch = 0
    resume_batch_idx = 0
    best_fid = []
    best_inception = []
    resume_payload = None
    if args.resume and resume_checkpoint.is_file():
        resume_payload = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        unwrap(model).load_state_dict(resume_payload["state_dict"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer"])
        global_step = int(resume_payload.get("global_step", 0))
        start_epoch = int(resume_payload.get("epoch", 0))
        resume_batch_idx = int(resume_payload.get("batch_idx", 0))
        best_fid = [(float(x[0]), str(x[1])) for x in resume_payload.get("best_fid", [])]
        best_inception = [
            (float(x[0]), str(x[1])) for x in resume_payload.get("best_inception", [])
        ]
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
    last_perf_step = global_step
    last_perf_time = time.monotonic()
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
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    if args.compound_tokens:
                        coeff_ids, target_coeff_probs = aux.compound_coeff_ids(
                            coeffs, temp=0.5, stochastic=True
                        )
                        tokens = atoms.long() * args.coeff_vocab_size + coeff_ids
                        target_physical = aux.physical_contributions(atoms, coeffs)
                        compact_targets = (atoms.long(), target_coeff_probs, target_physical)
                    else:
                        tokens, compact_targets = aux.sparse_targets(
                            atoms, coeffs, temp=0.5, stochastic=True, compact=True
                        )
            else:
                images, labels = batch
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    if args.compound_tokens:
                        atoms, coeffs = aux.encode_sparse_components(images)
                        coeff_ids, target_coeff_probs = aux.compound_coeff_ids(
                            coeffs, temp=0.5, stochastic=True
                        )
                        tokens = atoms.long() * args.coeff_vocab_size + coeff_ids
                        target_physical = aux.physical_contributions(atoms, coeffs)
                        compact_targets = (atoms.long(), target_coeff_probs, target_physical)
                    else:
                        tokens, soft_targets = aux.encode_sparse(images, temp=0.5, stochastic=True)
            sync = ((absolute_batch_idx + 1) % accumulation == 0)
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
                            pair_scales = aux.coeff_scales.view(1, 1, 1, 2)
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
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
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
                    if rank() == 0:
                        recovery_snapshot = {
                            "epoch": epoch,
                            "batch_idx": absolute_batch_idx + 1,
                            "global_step": global_step,
                            "fid": None,
                            "inception_score": None,
                            "inception_score_std": None,
                            "state_dict": unwrap(model).state_dict(),
                            "optimizer": optimizer.state_dict(),
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
                    if dist.is_initialized():
                        dist.barrier()
                if args.sample_grid_every > 0 and global_step % args.sample_grid_every == 0:
                    if dist.is_initialized():
                        dist.barrier()
                    if rank() == 0:
                        target = sample_class_grid(
                            unwrap(model), aux, class_names, args.output, global_step, wb=wb,
                            atom_temperature=args.atom_temperature,
                            atom_top_p=args.atom_top_p,
                            coeff_temperature=args.coeff_temperature,
                            coeff_top_p=args.coeff_top_p,
                        )
                        print(f"Saved class-conditional samples: {target}", flush=True)
                    if dist.is_initialized():
                        dist.barrier()
        resume_batch_idx = 0
        if sampler is not None:
            sampler.set_start_index(0)
        if dist.is_initialized():
            dist.barrier()
        run_fid = (epoch + 1) % args.fid_every == 0
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
            fid, inception_score, inception_score_std = evaluate_generation_metrics(
                unwrap(model), aux, val_loader, args.fid_num_samples, args.fid_batch_size,
                atom_temperature=args.atom_temperature,
                atom_top_p=args.atom_top_p,
                coeff_temperature=args.coeff_temperature,
                coeff_top_p=args.coeff_top_p,
            )
        if rank() == 0:
            if wb is not None and fid is not None:
                wb.log({
                    "val/fid": fid,
                    "val/inception_score": inception_score,
                    "val/inception_score_std": inception_score_std,
                    "train/epoch": epoch + 1,
                    "train/global_step": global_step,
                })
            qualifies = fid is not None and (
                len(best_fid) < 3 or fid < max(x[0] for x in best_fid)
            )
            best_path = None
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
            best_inception_path = None
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
                snapshot = {
                    "epoch": epoch + 1, "global_step": global_step, "fid": fid,
                    "inception_score": inception_score,
                    "inception_score_std": inception_score_std,
                    "state_dict": unwrap(model).state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": None if scheduler is None else scheduler.state_dict(),
                    "config": vars(args), "best_fid": best_fid,
                    "best_inception": best_inception,
                }
                atomic_torch_save(snapshot, last_checkpoint)
                if best_path is not None:
                    shutil.copy2(last_checkpoint, best_path)
                if best_inception_path is not None:
                    shutil.copy2(last_checkpoint, best_inception_path)
                # Artifact cadence follows full FID evaluation cadence. Each
                # version contains the recoverable last checkpoint and every
                # locally available member of the top-three FID set.
                if args.upload_checkpoints and run_fid:
                    artifact_aliases = ["latest"]
                    if best_path is not None:
                        artifact_aliases.extend(["best", f"epoch-{epoch + 1}"])
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
            if not save_epoch:
                print(f"Epoch {epoch + 1}: checkpoint skipped", flush=True)
            elif fid is None:
                print(f"Epoch {epoch + 1}: FID skipped; saved {last_checkpoint}", flush=True)
            else:
                print(
                    f"Epoch {epoch + 1}: FID={fid:.4f}; "
                    f"IS={inception_score:.4f}+/-{inception_score_std:.4f}; "
                    f"saved {last_checkpoint}",
                    flush=True,
                )
        if dist.is_initialized():
            dist.barrier()
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
