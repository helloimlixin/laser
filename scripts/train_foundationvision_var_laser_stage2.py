#!/usr/bin/env python3
"""Train the official FoundationVision VAR backbone on LASER sparse levels.

The generative backbone, block-causal mask, initialization, optimizer parameter
groups, and LR/weight-decay schedule come from FoundationVision/VAR.  The only
model adaptation is the tractable compound LASER boundary

    p(atom, coefficient | history)
      = p(atom | history) p(coefficient | history, atom).

A literal joint head would contain 2048 * 2048 classes.  Each 8x8 OMP depth map
is one VAR stage, so production uses ``patch_nums=(8, 8)``.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as torch_dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
FV_VAR_ROOT = ROOT / "third_party" / "FoundationVision_VAR"
sys.path.insert(0, str(ROOT))

from scripts.train_official_rqtransformer_laser_stage2 import (
    FlatImages,
    LaserAux,
    SparseTokenCacheDataset,
    atomic_torch_save,
    val_image_transform,
)

# FoundationVision uses absolute imports (``import dist``, ``from models``).
sys.path.insert(0, str(FV_VAR_ROOT))
import dist as fv_dist
from models.helpers import sample_with_top_k_top_p_
from models.var import VAR as FoundationVisionVAR
from utils.amp_sc import AmpOptimizer
from utils.lr_control import lr_wd_annealing


def rank() -> int:
    return torch_dist.get_rank() if torch_dist.is_initialized() else 0


def foundationvision_parameter_groups(model: nn.Module):
    """Upstream ``filter_params`` grouping without its custom print wrapper."""
    no_weight_decay_keys = {
        "cls_token", "start_token", "task_token", "cfg_uncond",
        "pos_embed", "pos_1LC", "pos_start", "start_pos", "lvl_embed",
        "gamma", "beta", "ada_gss", "moe_bias", "scale_mul",
    }
    names, parameters = [], []
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            raise ValueError(f"unexpected frozen FoundationVision parameter: {name}")
        names.append(name)
        parameters.append(parameter)
        if (
            parameter.ndim == 1
            or name.endswith("bias")
            or any(key in name for key in no_weight_decay_keys)
        ):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = [
        {"params": no_decay, "wd_sc": 0.0, "lr_sc": 1.0},
        {"params": decay, "wd_sc": 1.0, "lr_sc": 1.0},
    ]
    return names, parameters, groups


class _LaserVAEContract(nn.Module):
    """The two attributes consumed by the official VAR constructor."""

    def __init__(self, input_dim: int, atom_vocab_size: int):
        super().__init__()
        self.Cvae = int(input_dim)
        self.vocab_size = int(atom_vocab_size)
        self.quantize = nn.Identity()


class FoundationVisionLaserVAR(FoundationVisionVAR):
    """Official VAR with a factorized compound LASER output boundary."""

    def __init__(
        self,
        *,
        input_dim: int = 256,
        num_atoms: int = 2048,
        coeff_vocab_size: int = 2048,
        num_classes: int = 1,
        depth: int = 16,
        embed_dim: int = 1024,
        num_heads: int = 16,
        patch_nums=(8, 8),
        shared_aln: bool = False,
        attn_l2_norm: bool = True,
        init_adaln: float = 0.5,
        init_adaln_gamma: float = 1e-3,
        init_head: float = 0.02,
        init_std: float = -1.0,
        flash_if_available: bool = True,
        fused_if_available: bool = True,
    ):
        if len(patch_nums) != 2 or patch_nums[0] != patch_nums[1]:
            raise ValueError("LASER k=2 requires two equal-resolution VAR stages")
        proxy = _LaserVAEContract(input_dim, num_atoms)
        super().__init__(
            vae_local=proxy,
            num_classes=num_classes,
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1 * depth / 24,
            norm_eps=1e-6,
            shared_aln=shared_aln,
            cond_drop_rate=0.1,
            attn_l2_norm=attn_l2_norm,
            patch_nums=tuple(patch_nums),
            flash_if_available=flash_if_available,
            fused_if_available=fused_if_available,
        )
        self.num_atoms = int(num_atoms)
        self.coeff_vocab_size = int(coeff_vocab_size)
        self.input_dim = int(input_dim)
        self.coeff_head = nn.Linear(self.C, self.coeff_vocab_size)
        self._coefficient_atom_vectors = None

        # Exact upstream initialization followed by the same head scaling for
        # the additional factorized coefficient classifier.
        self.init_weights(
            init_adaln=init_adaln,
            init_adaln_gamma=init_adaln_gamma,
            init_head=init_head,
            init_std=init_std,
        )
        coefficient_init_std = (
            (1 / self.C / 3) ** 0.5 if init_std < 0 else float(init_std)
        )
        nn.init.trunc_normal_(self.coeff_head.weight, std=coefficient_init_std)
        self.coeff_head.weight.data.mul_(init_head)
        self.coeff_head.bias.data.zero_()

    def _resolve_hidden(self, h_or_h_and_residual):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            hidden, residual = h_or_h_and_residual
            return residual + self.blocks[-1].drop_path(hidden)
        return h_or_h_and_residual

    def get_logits(self, h_or_h_and_residual, cond_BD):
        """Called from the unmodified upstream ``VAR.forward`` method."""
        hidden = self._resolve_hidden(h_or_h_and_residual)
        head_features = self.head_nm(hidden.float(), cond_BD).float()
        atom_logits = self.head(head_features)
        if self._coefficient_atom_vectors is None:
            raise RuntimeError("compound coefficient atoms were not supplied")
        atom_context = self.word_embed(self._coefficient_atom_vectors.float())
        coeff_logits = self.coeff_head(head_features + atom_context)
        return torch.cat((atom_logits, coeff_logits), dim=-1)

    def forward(
        self,
        label_B: torch.LongTensor,
        previous_level_BLC: torch.Tensor,
        target_atom_vectors_BLCv: torch.Tensor,
    ):
        self._coefficient_atom_vectors = target_atom_vectors_BLCv
        try:
            combined = super().forward(label_B, previous_level_BLC)
        finally:
            self._coefficient_atom_vectors = None
        return {
            "atom_logits": combined[..., :self.num_atoms],
            "coeff_logits": combined[..., self.num_atoms:],
        }

    def _inference_head_features(self, hidden, cond_BD):
        return self.head_nm(self._resolve_hidden(hidden).float(), cond_BD).float()

    @torch.no_grad()
    def sample_compound(
        self,
        batch_size: int,
        model_aux: LaserAux,
        cond=None,
        *,
        cfg: float = 1.5,
        atom_top_k: int = 900,
        atom_top_p: float = 0.96,
        coeff_top_k: int = 900,
        coeff_top_p: float = 0.96,
        seed: int | None = None,
        amp: bool = True,
    ):
        """Official KV-cache/CFG sampling with the compound LASER boundary."""
        B = int(batch_size)
        device = self.lvl_1L.device
        rng = None
        if seed is not None:
            self.rng.manual_seed(int(seed))
            rng = self.rng
        if cond is None:
            cond = torch.zeros(B, device=device, dtype=torch.long)
        cond = cond.to(device=device, dtype=torch.long).reshape(B)

        cond_BD = self.class_emb(
            torch.cat(
                (cond, torch.full_like(cond, fill_value=self.num_classes)), dim=0
            )
        )
        sos = cond_BD
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = (
            sos.unsqueeze(1).expand(2 * B, self.first_l, -1)
            + self.pos_start.expand(2 * B, self.first_l, -1)
            + lvl_pos[:, :self.first_l]
        )
        height = width = self.patch_nums[-1]
        levels = len(self.patch_nums)
        atoms = torch.empty(B, height, width, levels, device=device, dtype=torch.long)
        coeff_ids = torch.empty_like(atoms)
        accumulated = model_aux.dictionary.new_zeros(B, height, width, self.input_dim)
        cur_L = 0

        for block in self.blocks:
            block.attn.kv_caching(True)
        try:
            for level, pn in enumerate(self.patch_nums):
                ratio = level / self.num_stages_minus_1
                cur_L += pn * pn
                cond_or_shared = self.shared_ada_lin(cond_BD)
                hidden = next_token_map
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    for block in self.blocks:
                        hidden = block(
                            x=hidden, cond_BD=cond_or_shared, attn_bias=None
                        )
                head_features = self._inference_head_features(hidden, cond_BD)
                atom_logits = self.head(head_features)
                guidance = float(cfg) * ratio
                guided_atom_logits = (
                    (1 + guidance) * atom_logits[:B]
                    - guidance * atom_logits[B:]
                )
                if level > 0:
                    previous_atoms = atoms[..., :level].reshape(B, pn * pn, level)
                    guided_atom_logits.scatter_(
                        -1, previous_atoms, -float("inf"),
                    )
                sampled_atoms = sample_with_top_k_top_p_(
                    guided_atom_logits,
                    rng=rng,
                    top_k=min(int(atom_top_k), self.num_atoms),
                    top_p=float(atom_top_p),
                    num_samples=1,
                )[..., 0]
                atom_vectors = model_aux.dictionary.t()[sampled_atoms]
                compound_features = head_features + self.word_embed(
                    atom_vectors.repeat(2, 1, 1).float()
                )
                coeff_logits = self.coeff_head(compound_features)
                guided_coeff_logits = (
                    (1 + guidance) * coeff_logits[:B]
                    - guidance * coeff_logits[B:]
                )
                sampled_coeffs = sample_with_top_k_top_p_(
                    guided_coeff_logits,
                    rng=rng,
                    top_k=min(int(coeff_top_k), self.coeff_vocab_size),
                    top_p=float(coeff_top_p),
                    num_samples=1,
                )[..., 0]

                atoms[..., level] = sampled_atoms.reshape(B, pn, pn)
                coeff_ids[..., level] = sampled_coeffs.reshape(B, pn, pn)
                physical_coeffs = (
                    model_aux.coeff_bins[sampled_coeffs]
                    * model_aux.coeff_scales[level]
                )
                contribution = atom_vectors * physical_coeffs.unsqueeze(-1)
                accumulated.add_(contribution.reshape(B, pn, pn, self.input_dim))

                if level != levels - 1:
                    next_tokens = accumulated.reshape(B, pn * pn, self.input_dim)
                    next_token_map = (
                        self.word_embed(next_tokens.float())
                        + lvl_pos[:, cur_L:cur_L + self.patch_nums[level + 1] ** 2]
                    ).repeat(2, 1, 1)
        finally:
            for block in self.blocks:
                block.attn.kv_caching(False)
        return atoms, coeff_ids


def compound_batch(aux: LaserAux, atoms, coeffs):
    """Build upstream teacher forcing inputs and level-major hard targets."""
    coeff_ids, _ = aux.compound_coeff_ids(coeffs, stochastic=False, hard=True)
    quantized_contributions = aux.compound_embeddings(atoms, coeff_ids)
    B, H, W, K = atoms.shape
    level_atoms = atoms.permute(0, 3, 1, 2).reshape(B, K * H * W).long()
    level_coeffs = coeff_ids.permute(0, 3, 1, 2).reshape(B, K * H * W).long()
    previous_level = quantized_contributions[..., 0, :].reshape(B, H * W, -1)
    atom_vectors = aux.dictionary.t()[level_atoms]
    return previous_level, level_atoms, level_coeffs, atom_vectors


def exact_joint_objective(atom_logits, coeff_logits, atom_targets, coeff_targets):
    """Uniform per-event NLL, equivalent to CE on a factored joint token."""
    atom_nll = F.cross_entropy(
        atom_logits.float().reshape(-1, atom_logits.shape[-1]),
        atom_targets.reshape(-1), reduction="none",
    ).view_as(atom_targets)
    coeff_nll = F.cross_entropy(
        coeff_logits.float().reshape(-1, coeff_logits.shape[-1]),
        coeff_targets.reshape(-1), reduction="none",
    ).view_as(coeff_targets)
    return (atom_nll + coeff_nll).mean(), atom_nll, coeff_nll


@torch.no_grad()
def save_sample_grid(model, aux, output: Path, step: int, wb, batch_size=16):
    module = model.module if isinstance(model, DDP) else model
    was_training = module.training
    module.eval()
    batches = []
    for start in range(0, 64, batch_size):
        current = min(batch_size, 64 - start)
        atoms, coeff_ids = module.sample_compound(
            current, aux, cond=torch.zeros(current, device=aux.dictionary.device),
            cfg=1.5, atom_top_k=900, atom_top_p=0.96,
            coeff_top_k=900, coeff_top_p=0.96,
            seed=42 + step + start,
        )
        images = ((aux.decode_compound(atoms, coeff_ids).float() + 1) * 0.5).clamp(0, 1)
        batches.append(images.cpu())
    grid = torch.cat(batches)
    target = output / "samples" / f"step_{step:07d}_cfg1.5_k900_p0.96.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, target, nrow=8, padding=0)
    if wb is not None:
        import wandb
        wb.log({
            "samples/foundationvision_cfg1.5_k900_p0.96": wandb.Image(str(target)),
            "train/global_step": step,
        })
    if was_training:
        module.train()
    return target


@torch.no_grad()
def generation_fid(model, aux, real_loader, num_samples: int, batch_size: int):
    from torchmetrics.image.fid import FrechetInceptionDistance

    device = aux.dictionary.device
    metric = FrechetInceptionDistance(
        feature=2048, normalize=True, sync_on_compute=torch_dist.is_initialized()
    ).to(device)
    for images, _ in real_loader:
        real = ((images.to(device, non_blocking=True).float() + 1) * 0.5).clamp(0, 1)
        metric.update(real, real=True)

    module = model.module if isinstance(model, DDP) else model
    local_count = num_samples // fv_dist.get_world_size()
    local_count += int(rank() < num_samples % fv_dist.get_world_size())
    generated = 0
    was_training = module.training
    module.eval()
    while generated < local_count:
        current = min(batch_size, local_count - generated)
        atoms, coeff_ids = module.sample_compound(
            current, aux, cond=torch.zeros(current, device=device),
            cfg=1.5, atom_top_k=900, atom_top_p=0.96,
            coeff_top_k=900, coeff_top_p=0.96,
            seed=100_000 * rank() + generated,
        )
        fake = ((aux.decode_compound(atoms, coeff_ids).float() + 1) * 0.5).clamp(0, 1)
        metric.update(fake, real=False)
        generated += current
    value = float(metric.compute().item())
    if was_training:
        module.train()
    return value


def validate_rfid_preflight(paths, cache_path: Path):
    values = {}
    for mode, path in zip(("continuous", "quantized"), paths):
        payload = json.loads(path.read_text())
        if payload.get("cache_coeff_mode") != mode or int(payload.get("num_images", 0)) != 70_000:
            raise ValueError(f"invalid {mode} cache preflight: {path}")
        if Path(payload["token_cache"]).resolve() != cache_path.resolve():
            raise ValueError(f"{mode} preflight was computed from another cache")
        values[mode] = float(payload["rfid"])
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--token-cache", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-rfid-preflight", type=Path, nargs=2, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--base-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=float, default=4.0)
    parser.add_argument("--end-lr-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--save-step-freq", type=int, default=250)
    parser.add_argument("--save-epoch-freq", type=int, default=10)
    parser.add_argument("--sample-grid-every", type=int, default=500)
    parser.add_argument("--sample-grid-on-start", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fid-every", type=int, default=50)
    parser.add_argument("--fid-num-samples", type=int, default=50_000)
    parser.add_argument("--fid-batch-size", type=int, default=64)
    parser.add_argument("--max-optimizer-steps", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-entity", default="helloimlixin-rutgers")
    parser.add_argument("--wandb-project", default="laser")
    parser.add_argument("--wandb-id", required=True)
    parser.add_argument("--wandb-name", required=True)
    parser.add_argument("--wandb-mode", choices=("online", "disabled"), default="online")
    args = parser.parse_args()

    fv_dist.initialize(timeout=45)
    device = fv_dist.get_device()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if args.depth != 16:
        raise ValueError("this run is pinned to FoundationVision's published d16 config")
    if args.batch_size <= 0 or args.accumulation <= 0:
        raise ValueError("batch size and accumulation must be positive")
    effective_batch = args.batch_size * fv_dist.get_world_size() * args.accumulation
    if effective_batch != 768:
        raise ValueError(
            f"official d16 config requires effective batch 768, got {effective_batch}"
        )
    peak_lr = args.base_lr * effective_batch / 256
    if not math.isclose(peak_lr, 3e-4):
        raise ValueError(f"official batch-scaled d16 peak LR must be 3e-4, got {peak_lr}")

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint = checkpoint_dir / "last.pt"
    preflight = validate_rfid_preflight(args.cache_rfid_preflight, args.token_cache)

    cache_dataset = SparseTokenCacheDataset(args.token_cache)
    meta = dict(cache_dataset.meta)
    expected = {
        "dataset": "ffhq", "num_atoms": 2048,
        "coeff_vocab_size": 2048, "shape": [8, 8, 2],
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"token cache {key}: {meta.get(key)!r} != {value!r}")
    sampler = DistributedSampler(cache_dataset, shuffle=True, drop_last=True)
    loader = DataLoader(
        cache_dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=8, pin_memory=True, persistent_workers=True, drop_last=True,
    )
    usable_microbatches = (len(loader) // args.accumulation) * args.accumulation
    if usable_microbatches <= 0:
        raise ValueError("not enough cache rows for one optimizer step")

    aux = LaserAux(
        args.checkpoint, 2048, 2048, float(meta["coeff_max"]),
        float(meta.get("coeff_scale", 1.0)), attn_resolutions=(16,),
        coeff_scales=meta["coeff_scales"], clamp_coeffs=True,
        coeff_bin_centers=meta.get("coeff_bin_centers"), sparsity_level=2,
    ).to(device).eval()
    model = FoundationVisionLaserVAR(
        input_dim=256, num_atoms=2048, coeff_vocab_size=2048,
        num_classes=1, depth=16, embed_dim=1024, num_heads=16,
        patch_nums=(8, 8), shared_aln=False, attn_l2_norm=True,
        init_adaln=0.5, init_adaln_gamma=1e-3,
        init_head=0.02, init_std=-1,
        flash_if_available=True, fused_if_available=True,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    names, parameters, groups = foundationvision_parameter_groups(model)
    optimizer = torch.optim.AdamW(
        groups, lr=peak_lr, weight_decay=0,
        betas=(0.9, 0.95), fused=True,
    )
    amp_optimizer = AmpOptimizer(
        mixed_precision=1, optimizer=optimizer, names=names, paras=parameters,
        grad_clip=args.grad_clip, n_gradient_accumulation=args.accumulation,
    )

    start_epoch = start_batch = global_step = 0
    best_fid = math.inf
    if args.resume and last_checkpoint.is_file():
        payload = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["state_dict"], strict=True)
        amp_optimizer.load_state_dict(payload["amp_optimizer"])
        start_epoch = int(payload["epoch"])
        start_batch = int(payload.get("batch_idx", 0))
        global_step = int(payload["global_step"])
        best_fid = float(payload.get("best_fid", math.inf))
        del payload
        print(
            f"Resumed {last_checkpoint}: epoch={start_epoch}, "
            f"batch={start_batch}, optimizer_step={global_step}", flush=True,
        )

    model = DDP(
        model, device_ids=[fv_dist.get_local_rank()], broadcast_buffers=False,
        find_unused_parameters=False,
    )
    real_dataset = FlatImages(args.data, transform=val_image_transform())
    if len(real_dataset) != len(cache_dataset):
        raise ValueError("FFHQ source/cache row count mismatch")
    real_sampler = DistributedSampler(real_dataset, shuffle=False, drop_last=False)
    real_loader = DataLoader(
        real_dataset, batch_size=args.fid_batch_size, sampler=real_sampler,
        num_workers=8, pin_memory=True, persistent_workers=True,
    )

    wb = None
    if rank() == 0 and args.wandb_mode == "online":
        import wandb
        wb = wandb.init(
            entity=args.wandb_entity, project=args.wandb_project,
            id=args.wandb_id, name=args.wandb_name, resume="allow",
            config={
                **vars(args),
                "architecture": "FoundationVision-VAR-d16-LASER-compound-adapter",
                "foundationvision_var_commit": "78b95394fc5896192e3a003e4b295f8ea743c48f",
                "foundationvision_patch_nums": [8, 8],
                "foundationvision_embed_dim": 1024,
                "foundationvision_num_heads": 16,
                "foundationvision_drop_path_rate": 0.1 * 16 / 24,
                "foundationvision_attn_l2_norm": True,
                "foundationvision_shared_adaln": False,
                "foundationvision_cond_drop_rate": 0.1,
                "foundationvision_init_adaln": 0.5,
                "foundationvision_init_adaln_gamma": 1e-3,
                "foundationvision_init_head": 0.02,
                "foundationvision_schedule": "lin0",
                "foundationvision_fp16": True,
                "effective_batch_size": effective_batch,
                "peak_lr": peak_lr,
                "model_parameters": parameter_count,
                "compound_objective": "atom_ce_plus_coefficient_ce",
            },
        )
        wb.summary["diagnostics/continuous_cache_reconstruction_rfid"] = preflight["continuous"]
        wb.summary["diagnostics/quantized_cache_reconstruction_rfid"] = preflight["quantized"]
        for metric in (
            "train/loss", "train/atom_nll", "train/coeff_nll",
            "train/atom_top1", "train/coeff_top1",
            "train/atom_top1_depth0", "train/atom_top1_depth1",
            "train/coeff_top1_depth0", "train/coeff_top1_depth1",
            "train/coeff_physical_mae", "train/coeff_physical_mae_depth0",
            "train/coeff_physical_mae_depth1", "train/lr", "train/weight_decay",
            "train/grad_norm", "train/images_per_second", "train/epoch", "val/fid",
        ):
            wb.define_metric(metric, step_metric="train/global_step")
        (args.output / "launch_config.json").write_text(
            json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2) + "\n"
        )

    def save_checkpoint(epoch, batch_idx):
        if rank() != 0:
            return
        snapshot = {
            "epoch": int(epoch), "batch_idx": int(batch_idx),
            "global_step": int(global_step),
            "state_dict": model.module.state_dict(),
            "amp_optimizer": amp_optimizer.state_dict(),
            "best_fid": float(best_fid),
            "config": vars(args),
            "foundationvision_var_commit": "78b95394fc5896192e3a003e4b295f8ea743c48f",
        }
        atomic_torch_save(snapshot, last_checkpoint)
        print(f"Saved checkpoint at optimizer step {global_step}: {last_checkpoint}", flush=True)

    if args.sample_grid_on_start:
        if torch_dist.is_initialized():
            torch_dist.barrier()
        if rank() == 0:
            save_sample_grid(model, aux, args.output, global_step, wb)
        if torch_dist.is_initialized():
            torch_dist.barrier()

    total_micro_iterations = args.epochs * usable_microbatches
    warmup_iterations = args.warmup_epochs * usable_microbatches
    launch_step = global_step
    last_log_step = global_step
    last_log_time = time.monotonic()
    memory_logged = False
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    stop = False
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for batch_idx, (atoms, coeffs, labels) in enumerate(loader):
            if batch_idx >= usable_microbatches:
                break
            if epoch == start_epoch and batch_idx < start_batch:
                continue
            micro_iteration = epoch * usable_microbatches + batch_idx
            _, current_lr, _, current_wd = lr_wd_annealing(
                "lin0", optimizer, peak_lr, args.weight_decay, args.weight_decay,
                micro_iteration, warmup_iterations, total_micro_iterations,
                wp0=0.005, wpe=args.end_lr_ratio,
            )
            atoms = atoms.to(device, dtype=torch.long, non_blocking=True)
            coeffs = coeffs.to(device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            labels.zero_()
            with torch.no_grad():
                previous, atom_targets, coeff_targets, atom_vectors = compound_batch(
                    aux, atoms, coeffs
                )
            stepping = (batch_idx + 1) % args.accumulation == 0
            model.require_backward_grad_sync = stepping
            sync_context = nullcontext() if stepping else model.no_sync()
            with sync_context, amp_optimizer.amp_ctx:
                outputs = model(labels, previous, atom_vectors)
                loss, atom_nll, coeff_nll = exact_joint_objective(
                    outputs["atom_logits"], outputs["coeff_logits"],
                    atom_targets, coeff_targets,
                )
            grad_norm, _ = amp_optimizer.backward_clip_step(stepping=stepping, loss=loss)
            if not stepping:
                continue

            global_step += 1
            if not memory_logged:
                torch.cuda.synchronize(device)
                local_memory = torch.tensor(
                    [
                        torch.cuda.max_memory_allocated(device) / 2**30,
                        torch.cuda.max_memory_reserved(device) / 2**30,
                    ],
                    device=device,
                )
                gathered_memory = [
                    torch.zeros_like(local_memory)
                    for _ in range(fv_dist.get_world_size())
                ]
                torch_dist.all_gather(gathered_memory, local_memory)
                if rank() == 0:
                    description = "; ".join(
                        f"rank {memory_rank}: allocated={float(values[0]):.2f} GiB, "
                        f"reserved={float(values[1]):.2f} GiB"
                        for memory_rank, values in enumerate(gathered_memory)
                    )
                    print(f"First-step CUDA peak — {description}", flush=True)
                    if wb is not None:
                        memory_payload = {"train/global_step": global_step}
                        for memory_rank, values in enumerate(gathered_memory):
                            memory_payload.update({
                                f"system/rank{memory_rank}_peak_allocated_gib": float(values[0]),
                                f"system/rank{memory_rank}_peak_reserved_gib": float(values[1]),
                            })
                        wb.log(memory_payload)
                memory_logged = True
            if rank() == 0 and global_step % 10 == 0:
                now = time.monotonic()
                elapsed = max(now - last_log_time, 1e-6)
                steps_elapsed = max(global_step - last_log_step, 1)
                with torch.no_grad():
                    atom_pred = outputs["atom_logits"].argmax(dim=-1)
                    coeff_pred = outputs["coeff_logits"].argmax(dim=-1)
                    coeff_scale = aux.coeff_scales.repeat_interleave(64).view(1, -1)
                    pred_physical = aux.coeff_bins[coeff_pred] * coeff_scale
                    target_physical = aux.coeff_bins[coeff_targets] * coeff_scale
                    physical_mae = (pred_physical - target_physical).abs()
                    payload = {
                        "train/global_step": global_step,
                        "train/epoch": epoch,
                        "train/loss": float(loss),
                        "train/atom_nll": float(atom_nll.mean()),
                        "train/coeff_nll": float(coeff_nll.mean()),
                        "train/atom_top1": float((atom_pred == atom_targets).float().mean()),
                        "train/coeff_top1": float((coeff_pred == coeff_targets).float().mean()),
                        "train/coeff_physical_mae": float(physical_mae.mean()),
                        "train/lr": current_lr,
                        "train/weight_decay": current_wd,
                        "train/grad_norm": float(grad_norm),
                        "train/images_per_second": (
                            steps_elapsed * effective_batch / elapsed
                        ),
                    }
                    for depth_index, (begin, end) in enumerate(((0, 64), (64, 128))):
                        payload.update({
                            f"train/atom_top1_depth{depth_index}": float(
                                (atom_pred[:, begin:end] == atom_targets[:, begin:end]).float().mean()
                            ),
                            f"train/coeff_top1_depth{depth_index}": float(
                                (coeff_pred[:, begin:end] == coeff_targets[:, begin:end]).float().mean()
                            ),
                            f"train/coeff_physical_mae_depth{depth_index}": float(
                                physical_mae[:, begin:end].mean()
                            ),
                        })
                if wb is not None:
                    wb.log(payload)
                last_log_step, last_log_time = global_step, now

            if args.save_step_freq > 0 and global_step % args.save_step_freq == 0:
                if torch_dist.is_initialized():
                    torch_dist.barrier()
                save_checkpoint(epoch, batch_idx + 1)
                if torch_dist.is_initialized():
                    torch_dist.barrier()
            if args.sample_grid_every > 0 and global_step % args.sample_grid_every == 0:
                if torch_dist.is_initialized():
                    torch_dist.barrier()
                if rank() == 0:
                    save_sample_grid(model, aux, args.output, global_step, wb)
                if torch_dist.is_initialized():
                    torch_dist.barrier()
            if args.max_optimizer_steps > 0 and global_step - launch_step >= args.max_optimizer_steps:
                stop = True
                break

        start_batch = 0
        if stop:
            if torch_dist.is_initialized():
                torch_dist.barrier()
            save_checkpoint(epoch, batch_idx + 1)
            break

        run_fid = args.fid_every > 0 and (epoch + 1) % args.fid_every == 0
        if run_fid:
            if torch_dist.is_initialized():
                torch_dist.barrier()
            fid = generation_fid(
                model, aux, real_loader, args.fid_num_samples, args.fid_batch_size
            )
            if rank() == 0:
                best_fid = min(best_fid, fid)
                if wb is not None:
                    wb.log({
                        "val/fid": fid, "train/epoch": epoch + 1,
                        "train/global_step": global_step,
                    })
                print(f"Epoch {epoch + 1}: FID={fid:.6f}", flush=True)
        if (epoch + 1) % args.save_epoch_freq == 0 or run_fid:
            if torch_dist.is_initialized():
                torch_dist.barrier()
            save_checkpoint(epoch + 1, 0)
            if torch_dist.is_initialized():
                torch_dist.barrier()

    if wb is not None:
        wb.finish()
    fv_dist.finalize()


if __name__ == "__main__":
    main()
