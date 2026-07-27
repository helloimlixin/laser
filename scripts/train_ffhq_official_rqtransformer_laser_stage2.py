#!/usr/bin/env python3
"""Train the upstream FFHQ 350M RQ-Transformer on LASER sparse codes."""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import shutil
import sys

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf
from src.models.rqtransformer.configs import RQTransformerConfig
from scripts.train_official_rqtransformer_laser_stage2 import (
    LaserAux,
    LaserRQTransformer,
    atomic_torch_save,
    rank,
    unwrap,
    upload_checkpoint,
)


class FlatImages(Dataset):
    def __init__(self, root: Path, transform):
        self.files = sorted(
            p for p in root.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if not self.files:
            raise ValueError(f"no images found below {root}")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.files[index]) as image:
            return self.transform(image.convert("RGB")), 0


def image_transform(train: bool):
    ops = [transforms.Resize(256)]
    ops.append(transforms.RandomCrop(256) if train else transforms.CenterCrop(256))
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops.extend([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
    return transforms.Compose(ops)


def build_model(vocab_size: int, num_atoms: int):
    # Exact architecture from configs/ffhq/stage2/ffhq256-rqtransformer-8x8x4-350M.yaml.
    cfg = OmegaConf.create({
        "type": "rq-transformer", "block_size": [8, 8, 4], "embed_dim": 1024,
        "input_embed_dim": 256, "shared_tok_emb": True, "shared_cls_emb": True,
        "input_emb_vqvae": True, "head_emb_vqvae": True, "cumsum_depth_ctx": True,
        "vocab_size": vocab_size, "vocab_size_cond": 1, "block_size_cond": 1,
        "body": {"n_layer": 24, "block": {"n_head": 16}},
        "head": {"n_layer": 4, "block": {"n_head": 16}},
    })
    return LaserRQTransformer(RQTransformerConfig.create(cfg), num_atoms=num_atoms)


@torch.no_grad()
def sample_grid(model, aux, output_dir: Path, step: int, wb=None):
    device = next(model.parameters()).device
    partial = torch.zeros(64, 8, 8, 4, device=device, dtype=torch.long)
    cond = torch.zeros(64, device=device, dtype=torch.long)
    was_training = model.training
    model.eval()
    tokens = model.sample(partial, model_aux=aux, cond=cond, temperature=1.0,
                          top_k=250, top_p=1.0, amp=True, cached=True, is_tqdm=False)
    images = ((aux.decode_tokens(tokens).float().cpu() + 1.0) * 0.5).clamp(0, 1)
    fig, axes = plt.subplots(8, 8, figsize=(16, 16))
    for image, axis in zip(images, axes.flat):
        axis.imshow(image.permute(1, 2, 0).numpy()); axis.axis("off")
    fig.tight_layout(pad=0.05)
    sample_dir = output_dir / "samples"; sample_dir.mkdir(parents=True, exist_ok=True)
    target = sample_dir / f"step_{step:07d}.png"
    fig.savefig(target, dpi=120); plt.close(fig)
    if wb is not None:
        import wandb
        wb.log({"samples/unconditional_8x8": wandb.Image(str(target))}, step=step)
    if was_training:
        model.train()


@torch.no_grad()
def evaluate_fid(
    model, aux, loader, num_samples: int, batch_size: int, top_k: int, top_p: float
):
    from torchmetrics.image.fid import FrechetInceptionDistance
    from tqdm.auto import tqdm
    device = next(model.parameters()).device
    # FID runs entirely on rank 0 while the other ranks wait at a barrier.
    # TorchMetrics otherwise attempts a distributed all-gather in compute(),
    # which cannot match that barrier and also duplicates the feature buffers.
    metric = FrechetInceptionDistance(
        feature=2048, normalize=True, sync_on_compute=False
    ).to(device)
    # The reported FFHQ protocol compares generated samples with every image in
    # the training set, rather than matching the number of real and fake images.
    num_real = 0
    real_progress = tqdm(loader, desc="FID real features", unit="batch",
                         dynamic_ncols=True)
    for images, _ in real_progress:
        metric.update(((images.to(device) + 1) * 0.5).clamp(0, 1), real=True)
        num_real += images.size(0)
        real_progress.set_postfix(images=num_real)
    generated = 0
    was_training = model.training; model.eval()
    with tqdm(total=num_samples, desc="FID generated samples", unit="image",
              dynamic_ncols=True) as generated_progress:
        while generated < num_samples:
            current = min(batch_size, num_samples - generated)
            partial = torch.zeros(current, 8, 8, 4, device=device, dtype=torch.long)
            cond = torch.zeros(current, device=device, dtype=torch.long)
            tokens = model.sample(partial, model_aux=aux, cond=cond, temperature=1.0,
                                  top_k=top_k, top_p=top_p, amp=True, cached=True,
                                  is_tqdm=False)
            metric.update(((aux.decode_tokens(tokens).float() + 1) * 0.5).clamp(0, 1), real=False)
            generated += current
            generated_progress.update(current)
    value = float(metric.compute().item())
    if was_training:
        model.train()
    return value, num_real


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--total-batch-size", type=int, default=128)
    p.add_argument("--num-atoms", type=int, default=1024)
    p.add_argument("--coeff-vocab-size", type=int, default=1024)
    p.add_argument("--coeff-max", type=float, default=20.0)
    p.add_argument("--coeff-scale", type=float, default=1.0)
    p.add_argument("--coeff-scales", type=float, nargs=2)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wandb-project", default="laser")
    p.add_argument("--wandb-entity", default="helloimlixin-rutgers")
    p.add_argument("--wandb-name", required=True)
    p.add_argument("--wandb-id", required=True)
    p.add_argument("--fid-num-samples", type=int, default=50_000)
    p.add_argument("--fid-batch-size", type=int, default=64)
    p.add_argument("--fid-every", type=int, default=50)
    p.add_argument("--fid-top-k", type=int, default=250)
    p.add_argument("--fid-top-p", type=float, default=1.0)
    p.add_argument("--resume-checkpoint", type=Path)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--upload-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Upload multi-GB checkpoints as W&B artifacts (disabled by default)",
    )
    args = p.parse_args()
    if args.fid_num_samples <= 0 or args.fid_batch_size <= 0 or args.fid_every <= 0:
        p.error("--fid-num-samples, --fid-batch-size, and --fid-every must be positive")
    if args.fid_top_k <= 0:
        p.error("--fid-top-k must be positive")
    if not 0.0 < args.fid_top_p <= 1.0:
        p.error("--fid-top-p must be in (0, 1]")

    local_rank = int(os.environ.get("LOCAL_RANK", "0")); world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        # Rank 0 performs 50k-sample FID evaluation while the remaining ranks
        # wait at the epoch barrier.  That pass legitimately takes much longer
        # than NCCL's 10-minute default timeout.
        dist.init_process_group("nccl", timeout=timedelta(hours=24))
    torch.cuda.set_device(local_rank); device = torch.device("cuda", local_rank)
    args.output.mkdir(parents=True, exist_ok=True)
    train_set = FlatImages(args.data / "images1024x1024", image_transform(True))
    val_set = FlatImages(args.data / "images1024x1024", image_transform(False))
    sampler = DistributedSampler(train_set, shuffle=True) if world > 1 else None
    loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler,
                        shuffle=sampler is None, num_workers=8, pin_memory=True,
                        persistent_workers=True, drop_last=True)
    val_loader = None
    if rank() == 0:
        val_loader = DataLoader(val_set, batch_size=args.fid_batch_size, shuffle=False,
                                num_workers=8, pin_memory=True, persistent_workers=True)
    aux = LaserAux(args.checkpoint, args.num_atoms, args.coeff_vocab_size,
                   args.coeff_max, args.coeff_scale, attn_resolutions=(16,),
                   coeff_scales=args.coeff_scales,
                   soft_target_physical=args.coeff_scales is not None).to(device)
    model = build_model(args.num_atoms + args.coeff_vocab_size, args.num_atoms).to(device)
    if world > 1:
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.95))
    accumulation = args.total_batch_size // (args.batch_size * world)
    if accumulation < 1 or accumulation * args.batch_size * world != args.total_batch_size:
        raise ValueError("total batch size must be divisible by per-step global batch size")
    wb = None
    if rank() == 0:
        import wandb
        wb = wandb.init(entity=args.wandb_entity, project=args.wandb_project, id=args.wandb_id,
                        resume="allow", name=args.wandb_name,
                        config={**vars(args), "architecture": "official-rqtransformer-350M",
                                "source_config": "configs/ffhq/stage2/ffhq256-rqtransformer-8x8x4-350M.yaml",
                                "stochastic_codes": True, "temp": 0.5,
                                "fid_top_k": args.fid_top_k, "fid_top_p": args.fid_top_p})
        (args.output / "launch_config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    checkpoint_dir = args.output / "checkpoints"; last_path = checkpoint_dir / "last.pt"
    start_epoch = global_step = 0; best_fid = []
    resume_path = args.resume_checkpoint or last_path
    if args.resume and resume_path.is_file():
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        unwrap(model).load_state_dict(payload["state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch, global_step = int(payload["epoch"]), int(payload["global_step"])
        best_fid = [(float(x[0]), str(x[1])) for x in payload.get("best_fid", [])]
        saved_config = payload.get("config", {})
        if (int(saved_config.get("fid_num_samples", -1)) != args.fid_num_samples or
                int(saved_config.get("fid_every", -1)) != args.fid_every):
            best_fid = []
        # A pre-FID recovery snapshot records the completed epoch with fid=None.
        # Re-run that epoch after an evaluation crash so its scheduled FID is
        # not silently skipped on resume.
        if (payload.get("fid") is None and start_epoch > 0 and
                start_epoch % args.fid_every == 0):
            start_epoch -= 1
        if rank() == 0:
            print(f"Resumed epoch {start_epoch} at step {global_step} from {resume_path}",
                  flush=True)
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None: sampler.set_epoch(epoch)
        model.train(); complete = (len(loader) // accumulation) * accumulation
        for batch_idx, (images, _) in enumerate(loader):
            if batch_idx >= complete: break
            images = images.to(device, non_blocking=True)
            cond = torch.zeros(images.size(0), device=device, dtype=torch.long)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                tokens, targets = aux.encode_sparse(images, temp=0.5, stochastic=True)
            sync = (batch_idx + 1) % accumulation == 0
            ctx = model.no_sync() if isinstance(model, DDP) and not sync else torch.enable_grad()
            with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(tokens, model_aux=aux, cond=cond, amp=False)
                loss = -(targets * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1).mean() / accumulation
            loss.backward()
            if sync:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); global_step += 1
                if wb is not None and global_step % 10 == 0:
                    wb.log({"train/loss": float(loss.detach()) * accumulation,
                            "train/epoch": epoch, "train/lr": args.lr}, step=global_step)
                if global_step % 100 == 0:
                    if dist.is_initialized(): dist.barrier()
                    if rank() == 0: sample_grid(unwrap(model), aux, args.output, global_step, wb)
                    if dist.is_initialized(): dist.barrier()
        if dist.is_initialized(): dist.barrier()
        if rank() == 0:
            # Persist the completed epoch before the comparatively long FID pass.
            # This also gives a local recovery point if evaluation or artifact
            # upload fails after training has finished.
            pre_fid_snapshot = {
                "epoch": epoch + 1, "global_step": global_step, "fid": None,
                "state_dict": unwrap(model).state_dict(),
                "optimizer": optimizer.state_dict(), "config": vars(args),
                "best_fid": best_fid,
            }
            atomic_torch_save(pre_fid_snapshot, last_path)
            fid = None
            best_path = None
            if (epoch + 1) % args.fid_every == 0 or epoch + 1 == args.epochs:
                fid, fid_num_real = evaluate_fid(
                    unwrap(model), aux, val_loader, args.fid_num_samples,
                    args.fid_batch_size, args.fid_top_k, args.fid_top_p,
                )
                wb.log({"val/fid": fid, "val/fid_num_generated": args.fid_num_samples,
                        "val/fid_num_real": fid_num_real, "val/fid_top_k": args.fid_top_k,
                        "val/fid_top_p": args.fid_top_p, "train/epoch": epoch + 1},
                       step=global_step)
                qualifies = len(best_fid) < 3 or fid < max(x[0] for x in best_fid)
                if qualifies:
                    best_path = checkpoint_dir / f"best_fid_{fid:.4f}_epoch_{epoch + 1:03d}.pt"
                    best_fid.append((fid, str(best_path))); best_fid.sort(key=lambda x: x[0])
                    while len(best_fid) > 3:
                        _, stale = best_fid.pop()
                        Path(stale).unlink(missing_ok=True)
                snapshot = {"epoch": epoch + 1, "global_step": global_step, "fid": fid,
                            "state_dict": unwrap(model).state_dict(),
                            "optimizer": optimizer.state_dict(), "config": vars(args),
                            "best_fid": best_fid}
                atomic_torch_save(snapshot, last_path)
            if args.upload_checkpoints:
                upload_checkpoint(wb, last_path, artifact_name=f"{wb.id}-last", aliases=["latest"],
                                  metadata={"epoch": epoch + 1, "step": global_step, "fid": fid})
            if best_path is not None:
                shutil.copy2(last_path, best_path)
                if args.upload_checkpoints:
                    upload_checkpoint(wb, best_path, artifact_name=f"{wb.id}-best-fid",
                                      aliases=["best", f"epoch-{epoch + 1}"],
                                      metadata={"epoch": epoch + 1, "step": global_step, "fid": fid})
            if fid is None:
                print(f"Epoch {epoch + 1}: checkpoint saved; next FID at epoch "
                      f"{min(((epoch + 1) // args.fid_every + 1) * args.fid_every, args.epochs)}",
                      flush=True)
            else:
                print(f"Epoch {epoch + 1}: FID={fid:.4f}; retained "
                      f"{len(best_fid)} best + last", flush=True)
        if dist.is_initialized(): dist.barrier()
    if wb is not None: wb.finish()


if __name__ == "__main__":
    main()
