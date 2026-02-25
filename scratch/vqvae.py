"""
VQ-VAE + decoder-only transformer prior on CelebA 64x64.

Stage 1:
  Encoder -> EMA VQ bottleneck -> Decoder

Stage 2:
  Decoder-only transformer prior over flattened VQ token grids.

Default data path assumes running from `scratch/`:
  ../../data/celeba/img_align_celeba
"""

import argparse
import math
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import transforms, utils
from tqdm import tqdm

try:
    import wandb
except Exception:
    wandb = None

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except Exception:
    FrechetInceptionDistance = None


def _disable_lightning_cuda_matmul_capability_probe():
    """
    Lightning may probe CUDA capability at startup in a way that can fail on
    some systems despite CUDA training working fine.
    """
    try:
        import lightning.fabric.accelerators.cuda as fabric_cuda
        import lightning.pytorch.accelerators.cuda as pl_cuda
    except Exception:
        return

    def _noop(_device):
        return

    pl_cuda._check_cuda_matmul_precision = _noop
    fabric_cuda._check_cuda_matmul_precision = _noop


_disable_lightning_cuda_matmul_capability_probe()


def _pick_free_tcp_port(excluded_ports: Optional[set[int]] = None) -> int:
    excluded = excluded_ports or set()
    for _ in range(32):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("0.0.0.0", 0))
            port = int(sock.getsockname()[1])
            if port in excluded:
                continue
            return port
        except OSError:
            continue
        finally:
            sock.close()
    raise RuntimeError("Unable to allocate a free TCP port for DDP rendezvous.")


def _ensure_free_master_port_for_ddp(stage_tag: str) -> None:
    local_rank_raw = os.environ.get("LOCAL_RANK")
    local_rank = None
    if local_rank_raw is not None:
        try:
            local_rank = int(local_rank_raw)
        except ValueError:
            local_rank = None
    if local_rank is not None and local_rank != 0:
        return

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    raw_port = os.environ.get("MASTER_PORT")
    excluded_ports: set[int] = set()
    if raw_port is not None:
        try:
            excluded_ports.add(int(raw_port))
        except ValueError:
            pass

    selected_port = _pick_free_tcp_port(excluded_ports=excluded_ports)
    os.environ["MASTER_PORT"] = str(selected_port)
    print(f"[{stage_tag}] MASTER_PORT={selected_port}")


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FlatImageDataset(Dataset):
    """Recursively loads images from a directory tree."""

    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root}")

        self.image_paths = sorted(
            p
            for p in self.root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
        )
        if not self.image_paths:
            raise RuntimeError(f"No images found under: {self.root}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        with Image.open(self.image_paths[idx]) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, num_hiddens: int, num_residual_hiddens: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, num_residual_hiddens, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_residual_hiddens)
        self.conv2 = nn.Conv2d(num_residual_hiddens, num_hiddens, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_hiddens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class ResidualStack(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ResidualBlock(
                    in_channels=in_channels,
                    num_hiddens=num_hiddens,
                    num_residual_hiddens=num_residual_hiddens,
                )
                for _ in range(num_residual_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
        num_downsamples: int = 3,
    ):
        super().__init__()
        self.num_downsamples = int(num_downsamples)
        if self.num_downsamples <= 0:
            raise ValueError(f"num_downsamples must be positive, got {self.num_downsamples}")

        down = []
        ch = int(in_channels)
        for i in range(self.num_downsamples):
            out_ch = max(1, int(num_hiddens // 2)) if i == 0 else int(num_hiddens)
            down.append(nn.Conv2d(ch, out_ch, kernel_size=4, stride=2, padding=1))
            ch = out_ch
        self.down = nn.ModuleList(down)
        self.conv = nn.Conv2d(ch, int(num_hiddens), kernel_size=3, stride=1, padding=1)
        self.res = ResidualStack(
            in_channels=int(num_hiddens),
            num_hiddens=int(num_hiddens),
            num_residual_layers=int(num_residual_layers),
            num_residual_hiddens=int(num_residual_hiddens),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.down:
            x = F.relu(conv(x))
        return self.res(self.conv(x))


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
        out_channels: int = 3,
        num_upsamples: int = 3,
    ):
        super().__init__()
        self.num_upsamples = int(num_upsamples)
        if self.num_upsamples <= 0:
            raise ValueError(f"num_upsamples must be positive, got {self.num_upsamples}")

        self.conv = nn.Conv2d(int(in_channels), int(num_hiddens), kernel_size=3, stride=1, padding=1)
        self.res = ResidualStack(
            in_channels=int(num_hiddens),
            num_hiddens=int(num_hiddens),
            num_residual_layers=int(num_residual_layers),
            num_residual_hiddens=int(num_residual_hiddens),
        )
        up = []
        ch = int(num_hiddens)
        for i in range(self.num_upsamples):
            if i == self.num_upsamples - 1:
                out_ch = int(out_channels)
            elif i == self.num_upsamples - 2:
                out_ch = max(1, int(num_hiddens // 2))
            else:
                out_ch = int(num_hiddens)
            up.append(nn.ConvTranspose2d(ch, out_ch, kernel_size=4, stride=2, padding=1))
            ch = out_ch
        self.up = nn.ModuleList(up)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res(self.conv(x))
        for i, deconv in enumerate(self.up):
            x = deconv(x)
            if i != len(self.up) - 1:
                x = F.relu(x)
        return x


class VectorQuantizerEMAWithIndices(nn.Module):
    """
    EMA vector quantizer adapted from src/models/bottleneck.py::VectorQuantizerEMA,
    but additionally returns token indices in [B, H, W].
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        epsilon: float = 1e-10,
    ):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.commitment_cost = float(commitment_cost)
        self.ema_decay = float(ema_decay)
        self.epsilon = float(epsilon)

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.normal_()
        self.embedding.weight.requires_grad = False

        self.register_buffer("ema_cluster_size", torch.zeros(self.num_embeddings))
        self.register_buffer("_ema_w", torch.randn(self.num_embeddings, self.embedding_dim))

        # Transformer special tokens.
        self.pad_token_id = self.num_embeddings
        self.bos_token_id = self.num_embeddings + 1
        self.vocab_size = self.num_embeddings + 2

    def forward(self, z_e: torch.Tensor):
        if z_e.dim() != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(z_e.shape)}")
        if z_e.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected channel dim {self.embedding_dim} but got {z_e.shape[1]}"
            )

        b, _, h, w = z_e.shape
        z_e_bhwc = z_e.permute(0, 2, 3, 1).contiguous()
        z_flat = z_e_bhwc.view(-1, self.embedding_dim)

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

        quantized = self.embedding(indices).view_as(z_e_bhwc)

        if self.training:
            with torch.no_grad():
                counts = encodings.sum(dim=0).to(self.ema_cluster_size.dtype)
                self.ema_cluster_size.mul_(self.ema_decay).add_(counts, alpha=1 - self.ema_decay)

                dw = (encodings.t() @ z_flat).to(self._ema_w.dtype)
                self._ema_w.mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)

                n = self.ema_cluster_size.sum()
                cluster_size = (
                    (self.ema_cluster_size + self.epsilon)
                    / (n + self.num_embeddings * self.epsilon)
                    * n
                )
                self.embedding.weight.copy_(self._ema_w / cluster_size.unsqueeze(1))

        # EMA style commitment loss.
        loss = self.commitment_cost * F.mse_loss(quantized.detach(), z_e_bhwc)

        # Straight-through.
        quantized = z_e_bhwc + (quantized - z_e_bhwc).detach()

        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        token_map = indices.view(b, h, w)
        return quantized, loss, perplexity, token_map

    @torch.no_grad()
    def decode_indices(self, token_map: torch.Tensor) -> torch.Tensor:
        """
        Decode token map [B, H, W] back to quantized latent [B, C, H, W].
        """
        if token_map.dim() != 3:
            raise ValueError(f"Expected [B, H, W], got {tuple(token_map.shape)}")
        b, h, w = token_map.shape
        z = self.embedding(token_map.long().reshape(-1))  # [B*H*W, C]
        z = z.reshape(b, h, w, self.embedding_dim).permute(0, 3, 1, 2).contiguous()
        return z


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_hiddens: int = 128,
        num_downsamples: int = 3,
        num_residual_layers: int = 2,
        num_residual_hiddens: int = 32,
        embedding_dim: int = 64,
        num_embeddings: int = 512,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels=in_channels,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            num_downsamples=num_downsamples,
        )
        self.pre = nn.Conv2d(num_hiddens, embedding_dim, kernel_size=1)
        self.quantizer = VectorQuantizerEMAWithIndices(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            ema_decay=ema_decay,
        )
        self.post = nn.Conv2d(embedding_dim, num_hiddens, kernel_size=3, padding=1)
        self.decoder = Decoder(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            out_channels=in_channels,
            num_upsamples=num_downsamples,
        )

    def forward(self, x: torch.Tensor):
        z = self.pre(self.encoder(x))
        z_q, vq_loss, perplexity, token_map = self.quantizer(z)
        recon = torch.tanh(self.decoder(self.post(z_q)))
        return recon, vq_loss, perplexity, token_map

    @torch.no_grad()
    def encode_to_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        z = self.pre(self.encoder(x))
        _, _, _, token_map = self.quantizer(z)
        return token_map, token_map.shape[1], token_map.shape[2]

    @torch.no_grad()
    def decode_from_tokens(self, token_map: torch.Tensor) -> torch.Tensor:
        z_q = self.quantizer.decode_indices(token_map)
        recon = torch.tanh(self.decoder(self.post(z_q)))
        return recon


def _make_image_grid(x: torch.Tensor, nrow: int = 8) -> torch.Tensor:
    x = x.detach().cpu().clamp(-1, 1)
    x = (x + 1.0) / 2.0
    return utils.make_grid(x, nrow=nrow)


def save_image_grid(x: torch.Tensor, path: str, nrow: int = 8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    utils.save_image(_make_image_grid(x, nrow=nrow), path)


def _resolve_wandb_logger(logger_obj):
    if isinstance(logger_obj, WandbLogger):
        return logger_obj
    for lg in getattr(logger_obj, "loggers", []):
        if isinstance(lg, WandbLogger):
            return lg
    return None


def log_image_grid_wandb(
    logger_obj,
    key: str,
    x: torch.Tensor,
    step: int,
    nrow: int = 8,
    caption: Optional[str] = None,
) -> bool:
    wb_logger = _resolve_wandb_logger(logger_obj)
    if wb_logger is None or wandb is None:
        return False
    grid = _make_image_grid(x, nrow=nrow)
    grid_np = grid.permute(1, 2, 0).mul(255).clamp(0, 255).byte().numpy()
    wb_image = wandb.Image(grid_np, caption=caption) if caption else wandb.Image(grid_np)
    wb_logger.experiment.log({key: wb_image}, step=int(step))
    return True


def log_scalar_wandb(logger_obj, key: str, value: float, step: int) -> bool:
    wb_logger = _resolve_wandb_logger(logger_obj)
    if wb_logger is None or wandb is None:
        return False
    wb_logger.experiment.log({key: float(value)}, step=int(step))
    return True


class Stage1VQVAE(pl.LightningModule):
    def __init__(
        self,
        model: VQVAE,
        lr: float,
        bottleneck_weight: float,
        out_dir: str,
        val_vis_images: Optional[torch.Tensor] = None,
        lr_schedule: str = "cosine",
        warmup_epochs: int = 1,
        min_lr_ratio: float = 0.1,
    ):
        super().__init__()
        self.model = model
        self.lr = float(lr)
        self.bottleneck_weight = float(bottleneck_weight)
        self.out_dir = out_dir
        self.best_val = float("inf")
        self.val_vis_images = val_vis_images
        self.lr_schedule = str(lr_schedule)
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.min_lr_ratio = float(max(0.0, min(min_lr_ratio, 1.0)))

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if self.lr_schedule != "cosine":
            return opt

        max_epochs = max(1, int(getattr(self.trainer, "max_epochs", 1) or 1))
        warmup = min(self.warmup_epochs, max_epochs - 1)
        min_ratio = self.min_lr_ratio

        def lr_lambda(epoch: int) -> float:
            step_idx = int(epoch) + 1
            if warmup > 0 and step_idx <= warmup:
                return 0.1 + 0.9 * (step_idx / float(max(1, warmup)))
            decay_steps = max(1, max_epochs - warmup)
            decay_idx = min(max(step_idx - warmup, 0), decay_steps)
            t = decay_idx / float(decay_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * t))
            return min_ratio + (1.0 - min_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def training_step(self, batch, batch_idx):
        x, _ = batch
        recon, vq_loss, perplexity, _ = self.model(x)
        recon_loss = F.mse_loss(recon, x)
        loss = recon_loss + self.bottleneck_weight * vq_loss
        bs = x.size(0)

        self.log(
            "stage1/train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=bs,
        )
        self.log(
            "stage1/recon_loss",
            recon_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=bs,
        )
        self.log(
            "stage1/vq_loss",
            vq_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=bs,
        )
        self.log(
            "stage1/perplexity",
            perplexity,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=bs,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        recon, vq_loss, _, _ = self.model(x)
        recon_loss = F.mse_loss(recon, x)
        loss = recon_loss + self.bottleneck_weight * vq_loss
        psnr = 10.0 * torch.log10(4.0 / torch.clamp(recon_loss.detach(), min=1e-8))
        bs = x.size(0)

        self.log(
            "stage1/val_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=bs,
        )
        self.log(
            "stage1/val_psnr",
            psnr,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=bs,
        )
        return loss

    def on_validation_epoch_end(self):
        if not self.trainer.is_global_zero or self.trainer.sanity_checking:
            return

        os.makedirs(self.out_dir, exist_ok=True)
        cur = self.trainer.callback_metrics.get("stage1/val_loss")
        if cur is not None:
            cur_val = float(cur.detach().cpu().item())
            torch.save(self.model.state_dict(), os.path.join(self.out_dir, "ae_last.pt"))
            if cur_val < self.best_val:
                self.best_val = cur_val
                torch.save(self.model.state_dict(), os.path.join(self.out_dir, "ae_best.pt"))

        if self.val_vis_images is not None and self.val_vis_images.numel() > 0:
            x_vis = self.val_vis_images.to(self.device)
            with torch.no_grad():
                recon_vis, _, _, _ = self.model(x_vis)
            epoch = int(self.current_epoch + 1)
            step = int(self.global_step)
            logged_real = log_image_grid_wandb(
                self.logger,
                key="stage1/real",
                x=x_vis,
                step=step,
                caption=f"epoch={epoch} real",
            )
            logged_recon = log_image_grid_wandb(
                self.logger,
                key="stage1/recon",
                x=recon_vis,
                step=step,
                caption=f"epoch={epoch} recon",
            )
            if not (logged_real and logged_recon):
                save_image_grid(
                    x_vis,
                    os.path.join(self.out_dir, f"stage1_epoch{epoch:03d}_real.png"),
                )
                save_image_grid(
                    recon_vis,
                    os.path.join(self.out_dir, f"stage1_epoch{epoch:03d}_recon.png"),
                )


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = float(dropout)
        self.resid_dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        b, t, c = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_kv = (k, v)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=(kv_cache is None and t > 1),
            dropout_p=(self.dropout if self.training else 0.0),
        )
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out, new_kv


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attn_out, new_kv = self.attn(self.ln1(x), kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_kv


@dataclass
class TokenTransformerConfig:
    vocab_size: int
    h: int
    w: int
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 8
    d_ff: int = 1536
    dropout: float = 0.1


class TokenTransformerPrior(nn.Module):
    """
    Decoder-only transformer over flattened VQ tokens.
    """

    def __init__(self, cfg: TokenTransformerConfig, bos_token_id: int, pad_token_id: int):
        super().__init__()
        self.cfg = cfg
        self.bos_token_id = int(bos_token_id)
        self.pad_token_id = int(pad_token_id)

        self.tokens_per_image = int(cfg.h * cfg.w)
        self.max_len = 1 + self.tokens_per_image
        self.total_vocab = int(cfg.vocab_size + 2)  # content + BOS + PAD

        self.token_emb = nn.Embedding(self.total_vocab, cfg.d_model)
        self.pos_emb = nn.Embedding(self.max_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
                for _ in range(cfg.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, self.total_vocab, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[list] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, list]:
        b, l = x.shape
        pos = torch.arange(start_pos, start_pos + l, device=x.device, dtype=torch.long)
        h = self.token_emb(x) + self.pos_emb(pos).unsqueeze(0)
        h = self.drop(h)

        new_kv_cache = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            h, new_kv = block(h, kv_cache=layer_cache)
            new_kv_cache.append(new_kv)

        h = self.ln_f(h)
        logits = self.head(h)
        return logits, new_kv_cache

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        show_progress: bool = False,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        t = self.tokens_per_image

        seq = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=device)
        special = torch.tensor([self.bos_token_id, self.pad_token_id], device=device)
        kv_cache = None

        steps = tqdm(
            range(t),
            desc="[Stage2] sampling tokens",
            leave=False,
            dynamic_ncols=True,
            disable=(not show_progress),
        )
        for _ in steps:
            if kv_cache is None:
                inp = seq
                start_pos = 0
            else:
                inp = seq[:, -1:]
                start_pos = seq.size(1) - 1

            logits, kv_cache = self(inp, kv_cache=kv_cache, start_pos=start_pos)
            logits = logits[:, -1, :] / max(float(temperature), 1e-8)
            logits[:, special] = float("-inf")
            if top_k is not None and top_k > 0:
                k = min(int(top_k), int(logits.size(-1)))
                v, ix = torch.topk(logits, k, dim=-1)
                mask = torch.full_like(logits, float("-inf"))
                mask.scatter_(1, ix, v)
                logits = mask
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            seq = torch.cat([seq, nxt], dim=1)
        return seq[:, 1:]


class Stage2Prior(pl.LightningModule):
    def __init__(
        self,
        prior: TokenTransformerPrior,
        vqvae_for_decode: VQVAE,
        lr: float,
        out_dir: str,
        h: int,
        w: int,
        sample_every_steps: int = 2000,
        sample_batch_size: int = 64,
        sample_temperature: float = 0.8,
        sample_top_k: int = 64,
        sample_image_size: Optional[int] = 64,
        fid_real_images: Optional[torch.Tensor] = None,
        fid_num_samples: int = 64,
        fid_feature: int = 64,
        lr_schedule: str = "cosine",
        warmup_epochs: int = 1,
        min_lr_ratio: float = 0.1,
    ):
        super().__init__()
        self.prior = prior
        self.vqvae_for_decode = vqvae_for_decode
        self.lr = float(lr)
        self.out_dir = out_dir
        self.h = int(h)
        self.w = int(w)
        self.sample_every_steps = int(sample_every_steps)
        self.sample_batch_size = int(sample_batch_size)
        self.sample_temperature = max(float(sample_temperature), 1e-8)
        self.sample_top_k = None if int(sample_top_k) <= 0 else int(sample_top_k)
        self.sample_image_size = (
            None
            if sample_image_size is None or int(sample_image_size) <= 0
            else int(sample_image_size)
        )
        self.fid_real_images = fid_real_images
        self.fid_num_samples = max(0, int(fid_num_samples))
        self.fid_feature = int(fid_feature)
        self.lr_schedule = str(lr_schedule)
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.min_lr_ratio = float(max(0.0, min(min_lr_ratio, 1.0)))
        self._last_sample_step = -1
        self._fid_warned_unavailable = False
        self._fid_warned_compute_failed = False

        self.vqvae_for_decode.eval()
        for p in self.vqvae_for_decode.parameters():
            p.requires_grad_(False)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.prior.parameters(), lr=self.lr)
        if self.lr_schedule != "cosine":
            return opt

        max_epochs = 1
        if self.trainer is not None:
            max_epochs = int(getattr(self.trainer, "max_epochs", 1) or 1)
        max_epochs = max(1, max_epochs)
        warmup_epochs = min(self.warmup_epochs, max_epochs - 1)
        min_ratio = self.min_lr_ratio

        def lr_lambda(epoch: int) -> float:
            step_idx = int(epoch) + 1
            if warmup_epochs > 0 and step_idx <= warmup_epochs:
                return 0.1 + 0.9 * (step_idx / float(max(1, warmup_epochs)))
            decay_steps = max(1, max_epochs - warmup_epochs)
            decay_idx = min(max(step_idx - warmup_epochs, 0), decay_steps)
            t = decay_idx / float(decay_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * t))
            return min_ratio + (1.0 - min_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def training_step(self, batch, batch_idx):
        (tok_flat,) = batch if isinstance(batch, (tuple, list)) else (batch,)
        tok_flat = tok_flat.long()
        b = tok_flat.size(0)
        bos = int(self.prior.bos_token_id)
        pad = int(self.prior.pad_token_id)

        seq = torch.cat(
            [torch.full((b, 1), bos, dtype=torch.long, device=tok_flat.device), tok_flat],
            dim=1,
        )
        x_in = seq[:, :-1]
        y = seq[:, 1:]
        logits, _ = self.prior(x_in)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=pad,
        )
        self.log(
            "stage2/loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=b,
        )
        return loss

    def on_fit_start(self):
        self.vqvae_for_decode.to(self.device)
        self.vqvae_for_decode.eval()

    def _dist_initialized(self) -> bool:
        return bool(torch.distributed.is_available() and torch.distributed.is_initialized())

    def _dist_barrier(self) -> None:
        if self._dist_initialized():
            torch.distributed.barrier()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.sample_every_steps <= 0:
            return
        if self.global_step <= 0 or (self.global_step % self.sample_every_steps) != 0:
            return
        if int(self.global_step) == int(self._last_sample_step):
            return
        self._last_sample_step = int(self.global_step)
        if self._dist_initialized():
            # Keep all ranks synchronized while rank 0 performs sampling/FID.
            self._dist_barrier()
            sample_error_flag = torch.zeros(1, dtype=torch.int32, device=self.device)
            if self.trainer.is_global_zero:
                try:
                    self._sample_and_save(step=int(self.global_step))
                except Exception as exc:
                    sample_error_flag.fill_(1)
                    print(f"[Stage2] sampling failed at step {self.global_step}: {exc}")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            torch.distributed.all_reduce(sample_error_flag, op=torch.distributed.ReduceOp.MAX)
            self._dist_barrier()
            if int(sample_error_flag.item()) != 0:
                raise RuntimeError(
                    f"Stage-2 sampling failed at step {self.global_step} on at least one rank."
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return
        if self.trainer.is_global_zero:
            self._sample_and_save(step=int(self.global_step))

    def on_train_epoch_end(self):
        self._dist_barrier()
        if self.trainer.is_global_zero:
            os.makedirs(self.out_dir, exist_ok=True)
            torch.save(self.prior.state_dict(), os.path.join(self.out_dir, "transformer_last.pt"))
        self._dist_barrier()

    @torch.no_grad()
    def _sample_and_save(self, step: int):
        self.prior.eval()
        self.vqvae_for_decode.eval()

        flat_tokens = self.prior.generate(
            batch_size=int(self.sample_batch_size),
            temperature=self.sample_temperature,
            top_k=self.sample_top_k,
            show_progress=True,
        )
        token_map = flat_tokens.reshape(-1, self.h, self.w).to(self.device)
        raw_imgs = self.vqvae_for_decode.decode_from_tokens(token_map)
        if self.sample_image_size is not None:
            imgs = F.interpolate(
                raw_imgs,
                size=(self.sample_image_size, self.sample_image_size),
                mode="bilinear",
                align_corners=False,
            )
        else:
            imgs = raw_imgs

        logged = log_image_grid_wandb(
            self.logger,
            key="stage2/samples",
            x=imgs,
            step=step,
            caption=f"step={step}",
        )
        if not logged:
            save_image_grid(
                imgs, os.path.join(self.out_dir, f"stage2_step{step:06d}_samples.png")
            )
        self._compute_and_log_fid(step=step, fake_imgs=raw_imgs)
        self.prior.train()

    @torch.no_grad()
    def _compute_and_log_fid(self, step: int, fake_imgs: torch.Tensor):
        if self.fid_num_samples <= 0:
            return
        if FrechetInceptionDistance is None:
            if not self._fid_warned_unavailable:
                print("[Stage2] FID unavailable: torchmetrics.image.fid not installed.")
                self._fid_warned_unavailable = True
            return
        if self.fid_real_images is None or self.fid_real_images.numel() == 0:
            return

        fake_u8 = ((fake_imgs.detach().cpu().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
        n = min(self.fid_num_samples, int(self.fid_real_images.size(0)), int(fake_u8.size(0)))
        if n < 2:
            return

        feat = int(self.fid_feature)
        valid_features = [64, 192, 768, 2048]
        if feat not in valid_features:
            feat = 64
        if n < feat:
            feat = 64

        # Create a local metric instance on rank 0 only. Avoid attaching it as
        # a module attribute at runtime, which can desync DDP buffer lists.
        fid_metric = FrechetInceptionDistance(feature=feat, sync_on_compute=False).to(self.device)
        fid_metric.reset()

        real_u8 = self.fid_real_images[:n].to(self.device)
        fake_u8 = fake_u8[:n].to(self.device)
        bs = 64
        for start in range(0, n, bs):
            end = start + bs
            fid_metric.update(real_u8[start:end], real=True)
            fid_metric.update(fake_u8[start:end], real=False)

        try:
            fid_value = float(fid_metric.compute().detach().cpu().item())
            print(f"[Stage2] FID @ step {step}: {fid_value:.4f}")
            logged = log_scalar_wandb(
                self.logger,
                key="stage2/fid",
                value=fid_value,
                step=step,
            )
            if not logged:
                self.log(
                    "stage2/fid",
                    fid_value,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                    sync_dist=False,
                    rank_zero_only=True,
                )
        except Exception as exc:
            if not self._fid_warned_compute_failed:
                print(f"[Stage2] FID compute failed at step {step}: {exc}")
                self._fid_warned_compute_failed = True
        fid_metric.reset()
        fid_metric.cpu()


@torch.no_grad()
def precompute_tokens(
    model: VQVAE,
    loader: DataLoader,
    device: torch.device,
    max_items: Optional[int] = None,
) -> Tuple[torch.Tensor, int, int]:
    """
    Encode dataset to VQ tokens for stage-2 training.
    Returns:
      tokens_flat: [N, H*W] int32
      H, W
    """
    model.eval()
    all_tokens = []
    seen = 0
    h = w = None

    for x, _ in tqdm(loader, desc="[Stage2] precompute tokens"):
        x = x.to(device)
        tokens, cur_h, cur_w = model.encode_to_tokens(x)
        if h is None:
            h, w = cur_h, cur_w
        all_tokens.append(tokens.reshape(tokens.size(0), -1).to(torch.int32).cpu())
        seen += tokens.size(0)
        if max_items is not None and seen >= max_items:
            break

    tokens_flat = torch.cat(all_tokens, dim=0)
    if max_items is not None:
        tokens_flat = tokens_flat[:max_items]
    return tokens_flat, int(h), int(w)


@torch.no_grad()
def collect_real_images_uint8(loader: DataLoader, max_items: int) -> Optional[torch.Tensor]:
    """Collect real images as uint8 tensors for FID reference."""
    if max_items <= 0:
        return None
    images = []
    seen = 0
    for x, _ in tqdm(loader, desc="[Stage2] collect FID real images"):
        keep = min(x.size(0), max_items - seen)
        if keep <= 0:
            break
        real_u8 = ((x[:keep].detach().cpu().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
        images.append(real_u8)
        seen += keep
        if seen >= max_items:
            break
    if not images:
        return None
    return torch.cat(images, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../../data/celeba/img_align_celeba",
        help="Directory containing CelebA image files.",
    )
    parser.add_argument("--image_size", type=int, default=64, help="CelebA resize size.")
    parser.add_argument("--out_dir", type=str, default="./runs/vqvae_celeba_64")
    parser.add_argument("--seed", type=int, default=0)

    # Stage 1 (VQ-VAE)
    parser.add_argument("--stage1_epochs", type=int, default=20)
    parser.add_argument("--stage1_lr", type=float, default=2e-4)
    parser.add_argument("--stage1_lr_schedule", type=str, default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--stage1_warmup_epochs", type=int, default=1)
    parser.add_argument("--stage1_min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--stage1_devices", type=int, default=2)
    parser.add_argument("--stage1_precision", type=str, default="32-true")
    parser.add_argument("--stage1_strategy", type=str, default="ddp", choices=["ddp", "auto"])
    parser.add_argument(
        "--stage1_init_ckpt",
        type=str,
        default=None,
        help="Optional stage-1 VQ-VAE checkpoint (.pt) for warm start.",
    )
    parser.add_argument(
        "--stage1_val_vis_batch_size",
        type=int,
        default=64,
        help="Number of validation reconstructions logged each stage-1 epoch.",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--bottleneck_weight", type=float, default=1.0)

    # VQ-VAE architecture
    parser.add_argument("--num_hiddens", type=int, default=128)
    parser.add_argument("--ae_num_downsamples", type=int, default=3)
    parser.add_argument("--num_res_layers", type=int, default=2)
    parser.add_argument("--num_res_hiddens", type=int, default=32)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_embeddings", type=int, default=512)
    parser.add_argument("--commitment_cost", type=float, default=0.25)
    parser.add_argument("--ema_decay", type=float, default=0.99)

    # Stage 2 (decoder-only prior)
    parser.add_argument("--stage2_epochs", type=int, default=20)
    parser.add_argument("--stage2_lr", type=float, default=3e-4)
    parser.add_argument("--stage2_lr_schedule", type=str, default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--stage2_warmup_epochs", type=int, default=1)
    parser.add_argument("--stage2_min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--stage2_batch_size", type=int, default=64)
    parser.add_argument("--stage2_grad_accum", type=int, default=1)
    parser.add_argument("--stage2_devices", type=int, default=2)
    parser.add_argument("--stage2_precision", type=str, default="32-true")
    parser.add_argument("--stage2_strategy", type=str, default="ddp_fork", choices=["ddp", "ddp_fork", "auto"])
    parser.add_argument(
        "--stage2_resume_from_last",
        action="store_true",
        default=True,
        help="Load stage2/transformer_last.pt if present.",
    )
    parser.add_argument(
        "--no_stage2_resume_from_last",
        action="store_false",
        dest="stage2_resume_from_last",
        help="Disable stage-2 resume from stage2/transformer_last.pt.",
    )
    parser.add_argument("--stage2_sample_every_steps", type=int, default=2000)
    parser.add_argument("--stage2_sample_batch_size", type=int, default=64)
    parser.add_argument("--stage2_sample_temperature", type=float, default=0.8)
    parser.add_argument("--stage2_sample_top_k", type=int, default=64)
    parser.add_argument("--stage2_sample_image_size", type=int, default=64)
    parser.add_argument("--stage2_fid_num_samples", type=int, default=64)
    parser.add_argument("--stage2_fid_feature", type=int, default=64)

    # Transformer
    parser.add_argument("--tf_d_model", type=int, default=384)
    parser.add_argument("--tf_heads", type=int, default=8)
    parser.add_argument("--tf_layers", type=int, default=8)
    parser.add_argument("--tf_ff", type=int, default=1536)
    parser.add_argument("--tf_dropout", type=float, default=0.1)

    # Token cache
    parser.add_argument("--token_subset", type=int, default=50000, help="Number of images used for stage-2 tokens.")
    parser.add_argument("--token_num_workers", type=int, default=0)

    # W&B
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_project", type=str, default="laser-scratch")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None, help="Base run name; stage suffix is added.")
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_dir", type=str, default="./wandb")

    args = parser.parse_args()
    if args.image_size <= 0:
        raise ValueError("image_size must be positive.")
    if args.ae_num_downsamples <= 0:
        raise ValueError("ae_num_downsamples must be positive.")
    if args.stage2_batch_size <= 0:
        raise ValueError("stage2_batch_size must be > 0.")
    if args.stage2_sample_batch_size <= 0:
        raise ValueError("stage2_sample_batch_size must be > 0.")
    if args.stage2_sample_temperature <= 0.0:
        raise ValueError("stage2_sample_temperature must be > 0.")

    pl.seed_everything(args.seed, workers=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")

    os.makedirs(args.out_dir, exist_ok=True)
    stage1_dir = os.path.join(args.out_dir, "stage1")
    stage2_dir = os.path.join(args.out_dir, "stage2")
    os.makedirs(stage1_dir, exist_ok=True)
    os.makedirs(stage2_dir, exist_ok=True)
    token_cache_path = os.path.join(stage2_dir, "tokens_cache.pt")

    run_base_name = args.wandb_name or f"vqvae_celeba_{args.image_size}"
    if "LASER_WANDB_GROUP" in os.environ:
        wandb_group = os.environ["LASER_WANDB_GROUP"]
    else:
        wandb_group = args.wandb_group or f"{run_base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.environ["LASER_WANDB_GROUP"] = wandb_group

    def _build_wandb_logger(stage_tag: str):
        if args.wandb_mode == "disabled":
            return False
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank not in (None, "0"):
            return False
        try:
            return WandbLogger(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=f"{run_base_name}_{stage_tag}",
                group=wandb_group,
                save_dir=args.wandb_dir,
                offline=(args.wandb_mode == "offline"),
                log_model=False,
            )
        except Exception as exc:
            print(f"[WandB] logger init failed ({exc}); continuing without W&B.")
            return False

    def _build_vqvae() -> VQVAE:
        return VQVAE(
            in_channels=3,
            num_hiddens=args.num_hiddens,
            num_downsamples=args.ae_num_downsamples,
            num_residual_layers=args.num_res_layers,
            num_residual_hiddens=args.num_res_hiddens,
            embedding_dim=args.embedding_dim,
            num_embeddings=args.num_embeddings,
            commitment_cost=args.commitment_cost,
            ema_decay=args.ema_decay,
        )

    def _load_vqvae_weights(model: VQVAE, ckpt_path: str, tag: str = "Stage1"):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"{tag} checkpoint not found at {ckpt_path}")
        try:
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"[{tag}] loaded VQ-VAE weights from {ckpt_path}")

    def _load_best_vqvae_weights(model: VQVAE):
        _load_vqvae_weights(model, os.path.join(stage1_dir, "ae_best.pt"), tag="Stage1")

    def _run_stage2_lightning(
        tokens_flat: torch.Tensor,
        h: int,
        w: int,
        vqvae_model: VQVAE,
        fid_real_images: Optional[torch.Tensor],
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("Stage-2 Lightning training requires CUDA.")
        if torch.cuda.device_count() < args.stage2_devices:
            raise RuntimeError(
                f"Requested {args.stage2_devices} GPUs for stage-2, but only {torch.cuda.device_count()} detected."
            )

        stage2_ds = TensorDataset(tokens_flat)
        stage2_dl = DataLoader(
            stage2_ds,
            batch_size=args.stage2_batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )

        cfg = TokenTransformerConfig(
            vocab_size=int(args.num_embeddings),
            h=int(h),
            w=int(w),
            d_model=int(args.tf_d_model),
            n_heads=int(args.tf_heads),
            n_layers=int(args.tf_layers),
            d_ff=int(args.tf_ff),
            dropout=float(args.tf_dropout),
        )
        prior = TokenTransformerPrior(
            cfg=cfg,
            bos_token_id=int(vqvae_model.quantizer.bos_token_id),
            pad_token_id=int(vqvae_model.quantizer.pad_token_id),
        )

        if args.stage2_resume_from_last:
            stage2_last_path = os.path.join(stage2_dir, "transformer_last.pt")
            if os.path.exists(stage2_last_path):
                try:
                    stage2_state = torch.load(stage2_last_path, map_location="cpu", weights_only=True)
                except TypeError:
                    stage2_state = torch.load(stage2_last_path, map_location="cpu")
                model_state = prior.state_dict()
                filtered = {
                    k: v for k, v in stage2_state.items() if k in model_state and model_state[k].shape == v.shape
                }
                skipped = set(stage2_state.keys()) - set(filtered.keys())
                prior.load_state_dict(filtered, strict=False)
                if skipped:
                    print(f"[Stage2] skipped {len(skipped)} incompatible keys: {sorted(skipped)}")
                print(f"[Stage2] resumed transformer weights from {stage2_last_path}")

        stage2_module = Stage2Prior(
            prior=prior,
            vqvae_for_decode=vqvae_model,
            lr=args.stage2_lr,
            out_dir=stage2_dir,
            h=int(h),
            w=int(w),
            sample_every_steps=args.stage2_sample_every_steps,
            sample_batch_size=args.stage2_sample_batch_size,
            sample_temperature=args.stage2_sample_temperature,
            sample_top_k=args.stage2_sample_top_k,
            sample_image_size=(
                args.stage2_sample_image_size if int(args.stage2_sample_image_size) > 0 else None
            ),
            fid_real_images=fid_real_images,
            fid_num_samples=args.stage2_fid_num_samples,
            fid_feature=args.stage2_fid_feature,
            lr_schedule=args.stage2_lr_schedule,
            warmup_epochs=args.stage2_warmup_epochs,
            min_lr_ratio=args.stage2_min_lr_ratio,
        )

        strategy: object = (args.stage2_strategy if args.stage2_devices > 1 else "auto")
        if strategy == "ddp_fork" and torch.cuda.is_initialized():
            print("[Stage2] CUDA already initialized; falling back from ddp_fork to ddp.")
            strategy = "ddp"
        if strategy in ("ddp", "ddp_fork"):
            from lightning.pytorch.strategies import DDPStrategy

            strategy = DDPStrategy(
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        if args.stage2_devices > 1:
            os.environ["LASER_DDP_PHASE"] = "stage2"
            _ensure_free_master_port_for_ddp("Stage2")

        trainer = pl.Trainer(
            accelerator="gpu",
            devices=args.stage2_devices,
            strategy=strategy,
            max_epochs=args.stage2_epochs,
            logger=_build_wandb_logger("stage2"),
            enable_checkpointing=False,
            gradient_clip_val=1.0,
            precision=args.stage2_precision,
            log_every_n_steps=10,
            deterministic=False,
            accumulate_grad_batches=args.stage2_grad_accum,
        )
        trainer.fit(stage2_module, train_dataloaders=stage2_dl)

    # During stage-2 DDP re-entry, skip stage-1 and tokenization.
    if os.environ.get("LASER_DDP_PHASE") == "stage2":
        if not os.path.exists(token_cache_path):
            raise FileNotFoundError(f"Missing token cache: {token_cache_path}")
        try:
            cache = torch.load(token_cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            cache = torch.load(token_cache_path, map_location="cpu")
        tokens_flat = cache["tokens_flat"]
        h, w = cache["shape"]
        fid_real_images = cache.get("fid_real_images")
        vqvae = _build_vqvae()
        _load_best_vqvae_weights(vqvae)
        _run_stage2_lightning(
            tokens_flat=tokens_flat,
            h=int(h),
            w=int(w),
            vqvae_model=vqvae,
            fid_real_images=fid_real_images,
        )
        return

    # CelebA 64x64 direct resize.
    resize_hw = (int(args.image_size), int(args.image_size))
    tfm = transforms.Compose(
        [
            transforms.Resize(resize_hw),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    full = FlatImageDataset(root=args.data_dir, transform=tfm)
    if len(full) < 2:
        raise RuntimeError("CelebA dataset needs at least 2 images for train/val split.")
    val_size = max(1, int(0.05 * len(full)))
    train_size = len(full) - val_size
    all_indices = torch.randperm(len(full), generator=torch.Generator().manual_seed(args.seed))
    train_indices = all_indices[:train_size].tolist()
    val_indices = all_indices[train_size:].tolist()
    train_set = Subset(full, train_indices)
    val_set = Subset(full, val_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    stage1_val_vis_images = None
    if args.stage1_val_vis_batch_size > 0 and len(val_set) > 0:
        vis_bs = min(int(args.stage1_val_vis_batch_size), len(val_set))
        vis_loader = DataLoader(val_set, batch_size=vis_bs, shuffle=False, num_workers=0, pin_memory=False)
        vis_images, _ = next(iter(vis_loader))
        stage1_val_vis_images = vis_images
        print(
            f"[Stage1] validation visuals: batch={stage1_val_vis_images.size(0)} "
            f"size={stage1_val_vis_images.size(-2)}x{stage1_val_vis_images.size(-1)}"
        )

    vqvae = _build_vqvae()
    if args.stage1_init_ckpt:
        _load_vqvae_weights(vqvae, args.stage1_init_ckpt, tag="Stage1 init")

    if args.stage1_epochs > 0:
        if not torch.cuda.is_available():
            raise RuntimeError("Stage-1 Lightning training requires CUDA.")
        if torch.cuda.device_count() < args.stage1_devices:
            raise RuntimeError(
                f"Requested {args.stage1_devices} GPUs for stage-1, but only {torch.cuda.device_count()} detected."
            )

        stage1_module = Stage1VQVAE(
            model=vqvae,
            lr=args.stage1_lr,
            bottleneck_weight=args.bottleneck_weight,
            out_dir=stage1_dir,
            val_vis_images=stage1_val_vis_images,
            lr_schedule=args.stage1_lr_schedule,
            warmup_epochs=args.stage1_warmup_epochs,
            min_lr_ratio=args.stage1_min_lr_ratio,
        )

        if args.stage1_devices > 1:
            os.environ["LASER_DDP_PHASE"] = "stage1"
            _ensure_free_master_port_for_ddp("Stage1")

        stage1_strategy: object = (args.stage1_strategy if args.stage1_devices > 1 else "auto")
        if stage1_strategy in ("ddp", "ddp_fork"):
            from lightning.pytorch.strategies import DDPStrategy

            stage1_strategy = DDPStrategy(
                broadcast_buffers=False,
                find_unused_parameters=False,
            )

        stage1_trainer = pl.Trainer(
            accelerator="gpu",
            devices=args.stage1_devices,
            strategy=stage1_strategy,
            max_epochs=args.stage1_epochs,
            logger=_build_wandb_logger("stage1"),
            enable_checkpointing=False,
            gradient_clip_val=args.grad_clip,
            precision=args.stage1_precision,
            log_every_n_steps=10,
            deterministic=False,
        )
        stage1_trainer.fit(stage1_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank not in (None, "0"):
            # Only rank 0 proceeds to tokenization/stage-2 launch.
            return

    _load_best_vqvae_weights(vqvae)

    # Reuse cached tokens when stage-1 was not retrained.
    use_token_cache = args.stage1_epochs <= 0 and os.path.exists(token_cache_path)
    if use_token_cache:
        print(f"[Stage2] loading cached tokens from {token_cache_path}")
        try:
            cache = torch.load(token_cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            cache = torch.load(token_cache_path, map_location="cpu")
        tokens_flat = cache["tokens_flat"]
        h, w = cache["shape"]
        fid_real_images = cache.get("fid_real_images")
    else:
        encode_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        vqvae = vqvae.to(encode_device)
        token_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.token_num_workers,
            pin_memory=True,
            persistent_workers=(args.token_num_workers > 0),
        )
        tokens_flat, h, w = precompute_tokens(
            model=vqvae,
            loader=token_loader,
            device=encode_device,
            max_items=min(args.token_subset, len(train_set)),
        )
        fid_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        fid_real_images = collect_real_images_uint8(fid_loader, max_items=args.stage2_fid_num_samples)
        vqvae = vqvae.cpu()
        torch.save(
            {
                "tokens_flat": tokens_flat,
                "shape": (h, w),
                "fid_real_images": fid_real_images,
            },
            token_cache_path,
        )

    print(f"[Stage2] token dataset: {tokens_flat.shape}   (H={h}, W={w})")

    if args.stage1_epochs <= 0:
        _run_stage2_lightning(
            tokens_flat=tokens_flat,
            h=int(h),
            w=int(w),
            vqvae_model=vqvae,
            fid_real_images=fid_real_images,
        )
        return

    # After stage-1 DDP, restart process for clean stage-2 DDP launch.
    os.environ["LASER_DDP_PHASE"] = "stage2"
    for k in (
        "LOCAL_RANK",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "NODE_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        os.environ.pop(k, None)
    print("[Stage2] restarting process for clean DDP launch...")
    cmd = [sys.executable, __file__, *sys.argv[1:]]
    env = os.environ.copy()
    ret = subprocess.call(cmd, env=env, close_fds=True)
    if ret != 0:
        raise RuntimeError(f"Stage-2 restart process failed with exit code {ret}.")


if __name__ == "__main__":
    main()

