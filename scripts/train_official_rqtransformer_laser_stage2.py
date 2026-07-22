#!/usr/bin/env python3
"""Train upstream KakaoBrain RQ-Transformer on discretized LASER sparse pairs.

The transformer is imported unchanged from ``third_party/rq-vae-transformer``.
Only the stage-1 auxiliary embedding is adapted: OMP atom ids use the learned
LASER dictionary and real coefficients are uniformly discretized into the same
16K per-depth vocabulary used by the official shared classifier.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import shutil

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party" / "rq-vae-transformer"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UPSTREAM))

from omegaconf import OmegaConf
from rqvae.models.rqtransformer.configs import RQTransformerConfig
from rqvae.models.rqtransformer.transformers import RQTransformer
from rqvae.models.rqvae.rqvae import RQVAE
from src.data.imagenet_labels import class_names_for_dataset


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def unwrap(model):
    return model.module if isinstance(model, DDP) else model


class LaserAux(nn.Module):
    """Frozen stage-1 encoder and sparse dictionary expected by RQTransformer."""

    def __init__(self, checkpoint: Path, num_atoms: int, coeff_vocab_size: int,
                 coeff_max: float, coeff_scale: float = 1.0):
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
                num_res_blocks=2, attn_resolutions=[8], dropout=0.0,
            ),
            latent_shape=[8, 8, 256], code_shape=[8, 8, 2],
            shared_codebook=True, restart_unused_codes=True,
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
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
        self.coeff_scale = float(coeff_scale)
        if self.coeff_scale <= 0:
            raise ValueError("coeff_scale must be positive")
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def encode_sparse(self, images: torch.Tensor, *, temp: float = 0.5, stochastic: bool = True):
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
        coeffs = (physical_coeffs / self.coeff_scale).clamp(-self.coeff_max, self.coeff_max)
        scaled = (coeffs + self.coeff_max) * ((self.coeff_vocab_size - 1) / (2 * self.coeff_max))
        coeff_tokens = scaled.round().long().clamp(0, self.coeff_vocab_size - 1)
        # The official stage-2 recipe trains against stage-1 soft codes.  For
        # LASER, atom support is discrete OMP while the continuous coefficient
        # posterior is discretized into a temperature-controlled 16K density.
        coeff_logits = -(coeffs[..., None] - self.coeff_bins).square() / max(float(temp), 1e-6)
        coeff_probs = coeff_logits.softmax(dim=-1)
        if stochastic:
            coeff_tokens = torch.multinomial(
                coeff_probs.reshape(-1, self.coeff_vocab_size), 1
            ).reshape_as(coeff_tokens)
        tokens = torch.empty(b, h, w, 4, device=images.device, dtype=torch.long)
        tokens[..., 0::2] = atoms
        tokens[..., 1::2] = coeff_tokens + self.num_atoms
        soft_targets = torch.zeros(b, h, w, 4, self.vocab_size,
                                   device=images.device, dtype=coeff_probs.dtype)
        soft_targets[..., 0, :].scatter_(-1, atoms[..., 0, None], 1.0)
        soft_targets[..., 2, :].scatter_(-1, atoms[..., 1, None], 1.0)
        soft_targets[..., 1::2, self.num_atoms:] = coeff_probs
        return tokens, soft_targets

    @torch.no_grad()
    def get_code_emb_with_depth(self, tokens: torch.Tensor):
        out = torch.empty(*tokens.shape, 256, device=tokens.device, dtype=self.dictionary.dtype)
        atom_vectors = self.dictionary.t()[tokens[..., 0::2]]
        out[..., 0::2, :] = atom_vectors
        coeff_ids = (tokens[..., 1::2] - self.num_atoms).clamp(0, self.coeff_vocab_size - 1)
        coeff = self.coeff_bins[coeff_ids] * self.coeff_scale
        # Each pair must cumulatively equal c_i * D_i.  RQTransformer cumsums
        # depth embeddings, so emit D_i followed by (c_i - 1) * D_i.
        out[..., 1::2, :] = (coeff[..., None] - 1.0) * atom_vectors
        return out, None

    @torch.no_grad()
    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        atoms = tokens[..., 0::2].long()
        coeff_ids = (tokens[..., 1::2].long() - self.num_atoms).clamp(0, self.coeff_vocab_size - 1)
        coeffs = self.coeff_bins[coeff_ids] * self.coeff_scale
        atom_vectors = self.dictionary.t()[atoms]
        z_q = (atom_vectors * coeffs[..., None]).sum(dim=-2)
        z_q = self.post_quant_conv(z_q.permute(0, 3, 1, 2).contiguous())
        return self.decoder(z_q).clamp(-1.0, 1.0)


@torch.no_grad()
def sample_class_grid(model, aux, class_names, output_dir: Path, step: int, wb=None):
    device = next(model.parameters()).device
    chosen = torch.randperm(1000, device=device)[:8]
    labels = chosen.repeat_interleave(8)
    partial = torch.zeros(64, 8, 8, 4, device=device, dtype=torch.long)
    was_training = model.training
    model.eval()
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
        wb.log({"samples/class_conditional_8x8": wandb.Image(str(target))}, step=step)
    plt.close(fig)
    if was_training:
        model.train()
    return target


@torch.no_grad()
def evaluate_generation_fid(model, aux, val_loader, num_samples: int, batch_size: int = 64):
    from torchmetrics.image.fid import FrechetInceptionDistance

    device = next(model.parameters()).device
    metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    seen = 0
    for images, _ in val_loader:
        images = ((images.to(device, non_blocking=True).float() + 1.0) * 0.5).clamp(0, 1)
        keep = min(images.size(0), int(num_samples) - seen)
        metric.update(images[:keep], real=True)
        seen += keep
        if seen >= int(num_samples):
            break
    generated = 0
    was_training = model.training
    model.eval()
    while generated < int(num_samples):
        current = min(int(batch_size), int(num_samples) - generated)
        labels = torch.randint(0, 1000, (current,), device=device)
        partial = torch.zeros(current, 8, 8, 4, device=device, dtype=torch.long)
        tokens = model.sample(
            partial, model_aux=aux, cond=labels, temperature=1.0,
            top_k=aux.num_atoms, top_p=0.92, amp=True, cached=True, is_tqdm=False,
        )
        images = ((aux.decode_tokens(tokens).float() + 1.0) * 0.5).clamp(0, 1)
        metric.update(images, real=False)
        generated += current
    value = float(metric.compute().item())
    if was_training:
        model.train()
    return value


def atomic_torch_save(payload, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)


def upload_checkpoint(wb, path: Path, *, artifact_name: str, aliases, metadata):
    import wandb
    artifact = wandb.Artifact(artifact_name, type="model", metadata=metadata)
    artifact.add_file(str(path), name=path.name)
    wb.log_artifact(artifact, aliases=list(aliases))


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

    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
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


def build_model(total_vocab_size: int, num_atoms: int):
    cfg = OmegaConf.create({
        "type": "rq-transformer", "block_size": [8, 8, 4], "embed_dim": 1536,
        "input_embed_dim": 256, "shared_tok_emb": True, "shared_cls_emb": True,
        "input_emb_vqvae": True, "head_emb_vqvae": True, "cumsum_depth_ctx": True,
        "vocab_size": total_vocab_size, "vocab_size_cond": 1000, "block_size_cond": 1,
        "body": {"n_layer": 42, "block": {"n_head": 24}},
        "head": {"n_layer": 6, "block": {"n_head": 24}},
    })
    return LaserRQTransformer(RQTransformerConfig.create(cfg), num_atoms=num_atoms)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--total-batch-size", type=int, default=2048)
    p.add_argument("--num-atoms", type=int, default=16384)
    p.add_argument("--coeff-vocab-size", type=int, default=2048)
    p.add_argument("--coeff-max", type=float, default=20.0)
    p.add_argument("--coeff-scale", type=float, default=6.4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wandb-project", default="laser")
    p.add_argument("--wandb-name", default="imagenet-rqtransformer-laser-a16384-k2-stage2")
    p.add_argument("--fid-num-samples", type=int, default=2048)
    p.add_argument("--fid-batch-size", type=int, default=64)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    args.output.mkdir(parents=True, exist_ok=True)

    dataset = datasets.ImageFolder(args.data / "train", transform=image_transform())
    class_names = class_names_for_dataset("imagenet", dataset.classes)
    sampler = DistributedSampler(dataset, shuffle=True) if world > 1 else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        shuffle=sampler is None, num_workers=8, pin_memory=True,
                        persistent_workers=True, drop_last=True)
    val_loader = None
    if rank() == 0:
        val_dataset = datasets.ImageFolder(args.data / "val", transform=val_image_transform())
        val_loader = DataLoader(val_dataset, batch_size=args.fid_batch_size, shuffle=False,
                                num_workers=8, pin_memory=True, persistent_workers=True)
    total_vocab_size = args.num_atoms + args.coeff_vocab_size
    aux = LaserAux(args.checkpoint, args.num_atoms, args.coeff_vocab_size,
                   args.coeff_max, args.coeff_scale).to(device)
    model = build_model(total_vocab_size, args.num_atoms).to(device)
    if world > 1:
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4,
                                  betas=(0.9, 0.95))
    accumulation = args.total_batch_size // (args.batch_size * world)
    if accumulation * args.batch_size * world != args.total_batch_size:
        raise ValueError("total batch size must be divisible by per-step global batch size")
    use_wandb = rank() == 0
    wb = None
    if use_wandb:
        import wandb
        wb = wandb.init(project=args.wandb_project, name=args.wandb_name,
                        config={**vars(args), "architecture": "official-rqtransformer-1400M",
                                "stochastic_codes": True, "temp": 0.5, "top_p": 0.92})
        (args.output / "launch_config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    checkpoint_dir = args.output / "checkpoints"
    last_checkpoint = checkpoint_dir / "last.pt"
    global_step = 0
    start_epoch = 0
    best_fid = []
    if args.resume and last_checkpoint.is_file():
        resume_payload = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        unwrap(model).load_state_dict(resume_payload["state_dict"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer"])
        global_step = int(resume_payload.get("global_step", 0))
        start_epoch = int(resume_payload.get("epoch", 0))
        best_fid = [(float(x[0]), str(x[1])) for x in resume_payload.get("best_fid", [])]
        if rank() == 0:
            print(f"Resumed from {last_checkpoint}: epoch={start_epoch}, step={global_step}", flush=True)
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        complete_microbatches = (len(loader) // accumulation) * accumulation
        for batch_idx, (images, labels) in enumerate(loader):
            if batch_idx >= complete_microbatches:
                break
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                tokens, soft_targets = aux.encode_sparse(images, temp=0.5, stochastic=True)
            sync = ((batch_idx + 1) % accumulation == 0)
            ctx = model.no_sync() if isinstance(model, DDP) and not sync else torch.enable_grad()
            with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(tokens, model_aux=aux, cond=labels, amp=False)
                log_probs = F.log_softmax(logits.float(), dim=-1)
                loss = -(soft_targets * log_probs).sum(dim=-1).mean() / accumulation
            loss.backward()
            if sync:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); global_step += 1
                if wb is not None and global_step % 10 == 0:
                    wb.log({"train/loss": float(loss.detach()) * accumulation,
                            "train/epoch": epoch, "train/lr": optimizer.param_groups[0]["lr"]}, step=global_step)
                if global_step % 100 == 0:
                    if dist.is_initialized():
                        dist.barrier()
                    if rank() == 0:
                        target = sample_class_grid(
                            unwrap(model), aux, class_names, args.output, global_step, wb=wb
                        )
                        print(f"Saved class-conditional samples: {target}", flush=True)
                    if dist.is_initialized():
                        dist.barrier()
        if dist.is_initialized():
            dist.barrier()
        if rank() == 0:
            fid = evaluate_generation_fid(
                unwrap(model), aux, val_loader, args.fid_num_samples, args.fid_batch_size
            )
            if wb is not None:
                wb.log({"val/fid": fid, "train/epoch": epoch + 1}, step=global_step)
            qualifies = len(best_fid) < 3 or fid < max(x[0] for x in best_fid)
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
            snapshot = {
                "epoch": epoch + 1, "global_step": global_step, "fid": fid,
                "state_dict": unwrap(model).state_dict(), "optimizer": optimizer.state_dict(),
                "config": vars(args), "best_fid": best_fid,
            }
            atomic_torch_save(snapshot, last_checkpoint)
            upload_checkpoint(
                wb, last_checkpoint, artifact_name=f"{wb.id}-last",
                aliases=["latest"], metadata={"epoch": epoch + 1, "step": global_step, "fid": fid},
            )
            if best_path is not None:
                shutil.copy2(last_checkpoint, best_path)
                upload_checkpoint(
                    wb, best_path, artifact_name=f"{wb.id}-best-fid",
                    aliases=["best", f"epoch-{epoch + 1}"],
                    metadata={"epoch": epoch + 1, "step": global_step, "fid": fid},
                )
            print(f"Epoch {epoch + 1}: FID={fid:.4f}; saved {last_checkpoint}", flush=True)
        if dist.is_initialized():
            dist.barrier()
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
