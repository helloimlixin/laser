#!/usr/bin/env python3
import argparse
import math
import os
from pathlib import Path
from typing import Optional

import lightning as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms

from proto import (
    DEFAULT_STAGE1_SOURCE_RUN,
    DEFAULT_STAGE2_SOURCE_RUN,
    LASER,
    FlatImageDataset,
    _compute_quantized_rq_losses,
    _compute_stage2_sample_reference_stats,
    _decode_stage2_candidates_in_chunks,
    _default_run_name,
    _expected_token_cache_meta,
    _find_latest_stage2_token_cache,
    _import_real_wandb,
    _load_module_checkpoint,
    _load_token_cache,
    _resolve_stage2_token_cache_from_wandb_run,
    _select_best_stage2_samples,
    _token_cache_is_compatible,
    precompute_tokens,
    save_image_grid,
)
try:
    from spatial_prior import SpatialDepthPrior, build_spatial_depth_prior_config
except ModuleNotFoundError:
    from laser_transformer import SpatialDepthPrior, build_spatial_depth_prior_config

wandb = _import_real_wandb()


def _disable_lightning_cuda_matmul_capability_probe() -> None:
    try:
        import lightning.fabric.accelerators.cuda as fabric_cuda
        import lightning.pytorch.accelerators.cuda as pl_cuda
    except Exception:
        return

    def _noop() -> None:
        return

    for mod in (fabric_cuda, pl_cuda):
        fn = getattr(mod, '_check_cuda_matmul_precision', None)
        if fn is not None:
            setattr(mod, '_check_cuda_matmul_precision', _noop)


_disable_lightning_cuda_matmul_capability_probe()


class Stage1LaserModule(pl.LightningModule):
    def __init__(
        self,
        model: LASER,
        lr: float,
        bottleneck_weight: float,
        out_dir: str,
        lr_schedule: str = 'cosine',
        warmup_epochs: int = 1,
        min_lr_ratio: float = 0.1,
    ):
        super().__init__()
        self.model = model
        self.lr = float(lr)
        self.bottleneck_weight = float(bottleneck_weight)
        self.out_dir = str(out_dir)
        self.lr_schedule = str(lr_schedule)
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.min_lr_ratio = float(max(0.0, min(min_lr_ratio, 1.0)))
        self.best_val = float('inf')

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if self.lr_schedule != 'cosine':
            return opt
        max_epochs = max(1, int(getattr(self.trainer, 'max_epochs', 1) or 1))
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
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}

    def training_step(self, batch, batch_idx):
        x, _ = batch
        recon, b_loss, _ = self.model(x)
        recon_loss = F.mse_loss(recon, x)
        loss = recon_loss + self.bottleneck_weight * b_loss
        bs = x.size(0)
        self.log('stage1/train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)
        self.log('stage1/recon_loss', recon_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)
        self.log('stage1/bottleneck_loss', b_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        recon, b_loss, _ = self.model(x)
        recon_loss = F.mse_loss(recon, x)
        loss = recon_loss + self.bottleneck_weight * b_loss
        psnr = 10.0 * torch.log10(4.0 / torch.clamp(recon_loss.detach(), min=1e-8))
        bs = x.size(0)
        self.log('stage1/val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)
        self.log('stage1/val_psnr', psnr, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)
        return loss

    def on_validation_epoch_end(self):
        if not self.trainer.is_global_zero or self.trainer.sanity_checking:
            return
        os.makedirs(self.out_dir, exist_ok=True)
        cur = self.trainer.callback_metrics.get('stage1/val_loss')
        if cur is None:
            return
        cur_val = float(cur.detach().cpu().item())
        torch.save(self.model.state_dict(), os.path.join(self.out_dir, 'ae_last.pt'))
        if cur_val < self.best_val:
            self.best_val = cur_val
            torch.save(self.model.state_dict(), os.path.join(self.out_dir, 'ae_best.pt'))


class Stage2LightningModule(pl.LightningModule):
    def __init__(
        self,
        prior: SpatialDepthPrior,
        ae_for_decode: LASER,
        h: int,
        w: int,
        d: int,
        lr: float,
        out_dir: str,
        rq_atom_loss_weight: float = 1.0,
        rq_coeff_loss_weight: float = 1.0,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.01,
        weight_decay: float = 0.0,
        sample_every_steps: int = 0,
        sample_batch_size: int = 8,
        sample_candidate_factor: int = 4,
        sample_temperature: float = 1.0,
        sample_top_k: Optional[int] = 256,
        sample_image_size: Optional[int] = None,
        sample_reference_stats: Optional[dict] = None,
    ):
        super().__init__()
        self.prior = prior
        self.ae_for_decode = ae_for_decode
        self.h = int(h)
        self.w = int(w)
        self.d = int(d)
        self.lr = float(lr)
        self.out_dir = str(out_dir)
        self.rq_atom_loss_weight = float(rq_atom_loss_weight)
        self.rq_coeff_loss_weight = float(rq_coeff_loss_weight)
        self.warmup_steps = max(0, int(warmup_steps))
        self.min_lr_ratio = float(max(0.0, min(float(min_lr_ratio), 1.0)))
        self.weight_decay = float(max(0.0, weight_decay))
        self.sample_every_steps = int(sample_every_steps)
        self.sample_batch_size = int(sample_batch_size)
        self.sample_candidate_factor = max(1, int(sample_candidate_factor))
        self.sample_temperature = float(sample_temperature)
        self.sample_top_k = None if sample_top_k is None or int(sample_top_k) <= 0 else int(sample_top_k)
        self.sample_image_size = None if sample_image_size is None or int(sample_image_size) <= 0 else int(sample_image_size)
        self.sample_reference_stats = sample_reference_stats
        self._last_sample_step = -1

        self.ae_for_decode.eval()
        for p in self.ae_for_decode.parameters():
            p.requires_grad_(False)

    def configure_optimizers(self):
        optimizer_cls = torch.optim.AdamW if self.weight_decay > 0 else torch.optim.Adam
        opt = optimizer_cls(self.prior.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.warmup_steps <= 0 and self.min_lr_ratio >= 1.0:
            return opt

        estimated_steps = getattr(self.trainer, 'estimated_stepping_batches', None)
        total_steps = max(1, int(estimated_steps or 1))
        warmup_steps = min(self.warmup_steps, max(0, total_steps - 1))
        min_lr_ratio = self.min_lr_ratio

        def lr_lambda(step: int) -> float:
            cur_step = int(step) + 1
            if warmup_steps > 0 and cur_step <= warmup_steps:
                return max(0.01, cur_step / float(max(1, warmup_steps)))
            progress = (cur_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}

    def on_fit_start(self):
        self.ae_for_decode.to(self.device)
        self.ae_for_decode.eval()
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)

    def training_step(self, batch, batch_idx):
        tok_flat = batch[0] if isinstance(batch, (tuple, list)) else batch
        tok_flat = tok_flat.to(self.device).long()
        bsz = tok_flat.size(0)
        tok_grid = tok_flat.view(bsz, self.h * self.w, self.d)
        logits = self.prior(tok_grid)
        vocab = self.prior.cfg.vocab_size
        per_token_ce = F.cross_entropy(
            logits.reshape(-1, vocab),
            tok_grid.reshape(-1),
            reduction='none',
        ).view(bsz, self.h * self.w, self.d)
        ce_loss, atom_ce_loss, coeff_ce_loss, loss = _compute_quantized_rq_losses(
            per_token_ce,
            atom_loss_weight=self.rq_atom_loss_weight,
            coeff_loss_weight=self.rq_coeff_loss_weight,
            coeff_depth_weighting='none',
            coeff_focal_gamma=0.0,
            coeff_logits=None,
        )
        self.log('stage2/train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bsz)
        self.log('stage2/ce_loss', ce_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=bsz)
        self.log('stage2/atom_ce_loss', atom_ce_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=bsz)
        self.log('stage2/coeff_ce_loss', coeff_ce_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=bsz)
        optimizer = self.optimizers()
        if optimizer is not None:
            self.log('stage2/lr', optimizer.param_groups[0]['lr'], on_step=True, on_epoch=False, sync_dist=False, batch_size=bsz)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.sample_every_steps <= 0:
            return
        if self.global_step <= 0 or (self.global_step % self.sample_every_steps) != 0:
            return
        if int(self.global_step) == int(self._last_sample_step):
            return
        self._last_sample_step = int(self.global_step)
        if not self.trainer.is_global_zero:
            return
        self.prior.eval()
        with torch.no_grad():
            candidate_batch_size = max(self.sample_batch_size, self.sample_batch_size * self.sample_candidate_factor)
            tokens_gen = self.prior.generate(
                batch_size=candidate_batch_size,
                temperature=self.sample_temperature,
                top_k=self.sample_top_k,
                show_progress=False,
            ).view(-1, self.h, self.w, self.d)
            imgs = _decode_stage2_candidates_in_chunks(
                self.ae_for_decode,
                tokens_gen,
                decode_batch_size=max(1, min(8, self.sample_batch_size)),
            )
            imgs_raw = imgs[:min(int(self.sample_batch_size), int(imgs.size(0)))].clone()
            imgs = _select_best_stage2_samples(
                imgs,
                keep=self.sample_batch_size,
                reference_stats=self.sample_reference_stats,
            )
            if self.sample_image_size is not None and (imgs.size(-2) != self.sample_image_size or imgs.size(-1) != self.sample_image_size):
                imgs = F.interpolate(imgs, size=(self.sample_image_size, self.sample_image_size), mode='bilinear', align_corners=False)
            raw_path = os.path.join(self.out_dir, f'stage2_step{int(self.global_step):06d}_raw_samples.png')
            sample_path = os.path.join(self.out_dir, f'stage2_step{int(self.global_step):06d}_samples.png')
            save_image_grid(imgs_raw, raw_path)
            save_image_grid(imgs, sample_path)
            logger = getattr(self, 'logger', None)
            experiment = getattr(logger, 'experiment', None)
            if experiment is not None and wandb is not None:
                experiment.log({
                    'stage2/raw_samples': [wandb.Image(raw_path, caption=f'step={int(self.global_step)} raw')],
                    'stage2/samples': [wandb.Image(sample_path, caption=f'step={int(self.global_step)}')],
                    'trainer/global_step': int(self.global_step),
                })
        self.prior.train()

    def on_train_epoch_end(self):
        if not self.trainer.is_global_zero:
            return
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.prior.state_dict(), out_dir / 'prior_last.pt')

    def on_fit_end(self):
        if not self.trainer.is_global_zero:
            return
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.prior.state_dict(), out_dir / 'prior_final.pt')


class CelebADataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, image_size: int, batch_size: int, num_workers: int, seed: int):
        super().__init__()
        self.data_dir = str(data_dir)
        self.image_size = int(image_size)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.seed = int(seed)
        self.train_set = None
        self.val_set = None
        self.full_eval_set = None

    def setup(self, stage: Optional[str] = None):
        tfm = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        full = FlatImageDataset(root=self.data_dir, transform=tfm)
        self.full_eval_set = full
        if len(full) < 2:
            raise RuntimeError('Dataset needs at least 2 images for train/val split.')
        val_size = max(1, int(0.05 * len(full)))
        train_size = len(full) - val_size
        all_indices = torch.randperm(len(full), generator=torch.Generator().manual_seed(self.seed))
        train_indices = all_indices[:train_size].tolist()
        val_indices = all_indices[train_size:].tolist()
        self.train_set = Subset(full, train_indices)
        self.val_set = Subset(full, val_indices)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=min(64, self.batch_size), shuffle=False, num_workers=max(1, self.num_workers // 2), pin_memory=True)


class TokenDataModule(pl.LightningDataModule):
    def __init__(self, token_dataset, batch_size: int):
        super().__init__()
        self.token_dataset = token_dataset
        self.batch_size = int(batch_size)

    def train_dataloader(self):
        return DataLoader(self.token_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=(len(self.token_dataset) >= self.batch_size))


def build_laser_from_args(args: argparse.Namespace) -> LASER:
    return LASER(
        in_channels=3,
        num_hiddens=args.num_hiddens,
        num_downsamples=args.ae_num_downsamples,
        num_residual_layers=args.num_res_layers,
        resolution=args.image_size,
        embedding_dim=args.embedding_dim,
        num_embeddings=args.num_atoms,
        sparsity_level=args.sparsity_level,
        commitment_cost=args.commitment_cost,
        n_bins=args.n_bins,
        coef_max=args.coef_max,
        coef_quantization=args.coef_quantization,
        coef_mu=args.coef_mu,
        quantize_sparse_coeffs=args.quantize_sparse_coeffs,
        patch_based=args.patch_based,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        patch_reconstruction=args.patch_reconstruction,
    )


def _build_wandb_logger(args: argparse.Namespace, suffix: str):
    if args.wandb_mode == 'disabled' or wandb is None:
        return None
    return WandbLogger(project=args.wandb_project, name=f'{args.wandb_name}_{suffix}', mode=args.wandb_mode)


def _resolve_stage1_checkpoint(args: argparse.Namespace, stage1_dir: Path) -> Path:
    if args.stage1_source_ckpt:
        ckpt = Path(args.stage1_source_ckpt).expanduser().resolve()
        if not ckpt.exists():
            raise FileNotFoundError(f'Requested stage-1 source checkpoint does not exist: {ckpt}')
        return ckpt
    if args.stage1_epochs > 0:
        ckpt = stage1_dir / 'ae_best.pt'
        if not ckpt.exists():
            raise FileNotFoundError(f'Missing stage-1 checkpoint for stage-2 Lightning run: {ckpt}')
        return ckpt
    if args.stage1_source_run:
        raise NotImplementedError('proto_lightning.py does not yet resolve stage-1 checkpoints from W&B runs; use --stage1_source_ckpt.')
    raise FileNotFoundError('Stage 2 needs a stage-1 checkpoint. Set --stage1_epochs > 0 or pass --stage1_source_ckpt.')


def _build_stage2_token_cache(args: argparse.Namespace, data: CelebADataModule, laser: LASER, token_cache_path: Path):
    token_subset = None if args.token_subset <= 0 else min(int(args.token_subset), len(data.full_eval_set))
    expected_meta = _expected_token_cache_meta(args, data.full_eval_set, token_subset, laser)

    candidate_paths = []
    if args.stage2_source_token_cache:
        candidate_paths.append(Path(args.stage2_source_token_cache).expanduser().resolve())
    elif args.stage2_source_run:
        candidate_paths.append(_resolve_stage2_token_cache_from_wandb_run(args.stage2_source_run, token_cache_path.parent))
    if token_cache_path.exists() and not args.rebuild_token_cache:
        candidate_paths.append(token_cache_path)
    latest_cache = _find_latest_stage2_token_cache(token_cache_path.parent.parent, token_cache_path.parent.parent)
    if latest_cache is not None:
        candidate_paths.append(latest_cache)

    seen = set()
    for candidate in candidate_paths:
        candidate = Path(candidate).expanduser().resolve()
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        cache = _load_token_cache(str(candidate))
        compatible, _ = _token_cache_is_compatible(cache, expected_meta)
        if compatible:
            if candidate != token_cache_path:
                token_cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(cache, str(token_cache_path))
            return cache

    token_loader = DataLoader(
        data.full_eval_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.num_workers // 2),
        pin_memory=True,
        persistent_workers=False,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokens_flat, coeffs_flat, h, w, d = precompute_tokens(laser.to(device), token_loader, device, max_items=token_subset)
    cache = {'tokens_flat': tokens_flat, 'shape': (h, w, d), 'meta': expected_meta}
    if coeffs_flat is not None:
        cache['coeffs_flat'] = coeffs_flat
    token_cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, str(token_cache_path))
    return cache


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Lightning LASER trainer for proto, including stage 1 and stage 2.')
    p.add_argument('--data_dir', type=str, default='/cache/home/xl598/Projects/data/celeba')
    p.add_argument('--image_size', type=int, default=128)
    p.add_argument('--out_dir', type=str, default='/scratch/xl598/runs/proto_lightning')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--stage1_epochs', type=int, default=10)
    p.add_argument('--stage2_epochs', type=int, default=10)
    p.add_argument('--stage1_lr', type=float, default=2e-4)
    p.add_argument('--stage2_lr', type=float, default=1e-3)
    p.add_argument('--bottleneck_weight', type=float, default=1.0)
    p.add_argument('--stage1_lr_schedule', type=str, default='cosine', choices=['constant', 'cosine'])
    p.add_argument('--stage1_warmup_epochs', type=int, default=1)
    p.add_argument('--stage1_min_lr_ratio', type=float, default=0.1)
    p.add_argument('--stage1_devices', type=int, default=1)
    p.add_argument('--stage2_devices', type=int, default=1)
    p.add_argument('--stage1_strategy', type=str, default='ddp', choices=['auto', 'ddp', 'ddp_fork'])
    p.add_argument('--stage2_strategy', type=str, default='ddp', choices=['auto', 'ddp', 'ddp_fork'])
    p.add_argument('--stage1_precision', type=str, default='16-mixed')
    p.add_argument('--stage2_precision', type=str, default='16-mixed')
    p.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'])
    p.add_argument('--wandb_project', type=str, default='laser-scratch')
    p.add_argument('--wandb_name', type=str, default='proto_lightning')
    p.add_argument('--stage1_source_ckpt', type=str, default='')
    p.add_argument('--stage1_source_run', type=str, default='')
    p.add_argument('--stage2_source_ckpt', type=str, default='')
    p.add_argument('--stage2_source_run', type=str, default='')
    p.add_argument('--stage2_source_token_cache', type=str, default='')
    p.add_argument('--token_subset', type=int, default=0)
    p.add_argument('--rebuild_token_cache', action='store_true')
    p.add_argument('--stage2_batch_size', type=int, default=16)
    p.add_argument('--stage2_rq_atom_loss_weight', type=float, default=1.0)
    p.add_argument('--stage2_rq_coeff_loss_weight', type=float, default=1.0)
    p.add_argument('--stage2_warmup_steps', type=int, default=500)
    p.add_argument('--stage2_min_lr_ratio', type=float, default=0.01)
    p.add_argument('--stage2_weight_decay', type=float, default=0.01)
    p.add_argument('--stage2_sample_every_steps', type=int, default=0)
    p.add_argument('--stage2_sample_batch_size', type=int, default=8)
    p.add_argument('--stage2_sample_candidate_factor', type=int, default=4)
    p.add_argument('--stage2_sample_temperature', type=float, default=0.5)
    p.add_argument('--stage2_sample_top_k', type=int, default=0)
    p.add_argument('--stage2_sample_image_size', type=int, default=128)
    p.add_argument('--tf_d_model', type=int, default=512)
    p.add_argument('--tf_heads', type=int, default=8)
    p.add_argument('--tf_layers', type=int, default=12)
    p.add_argument('--tf_global_tokens', type=int, default=0)
    p.add_argument('--tf_ff', type=int, default=1024)
    p.add_argument('--tf_dropout', type=float, default=0.1)
    p.add_argument('--num_hiddens', type=int, default=128)
    p.add_argument('--ae_num_downsamples', type=int, default=4)
    p.add_argument('--num_res_layers', type=int, default=2)
    p.add_argument('--embedding_dim', type=int, default=16)
    p.add_argument('--num_atoms', type=int, default=1024)
    p.add_argument('--sparsity_level', type=int, default=8)
    p.add_argument('--n_bins', type=int, default=256)
    p.add_argument('--coef_max', type=float, default=3.0)
    p.add_argument('--coef_quantization', type=str, default='uniform', choices=['uniform', 'mu_law'])
    p.add_argument('--coef_mu', type=float, default=0.0)
    p.add_argument('--commitment_cost', type=float, default=0.25)
    p.add_argument('--quantize_sparse_coeffs', action='store_true', default=True)
    p.add_argument('--patch_based', action='store_true', default=False)
    p.add_argument('--patch_size', type=int, default=4)
    p.add_argument('--patch_stride', type=int, default=2)
    p.add_argument('--patch_reconstruction', type=str, default='center_crop', choices=['center_crop', 'hann'])
    args = p.parse_args()
    if not args.wandb_name:
        args.wandb_name = _default_run_name('celeba', args.image_size, bool(args.quantize_sparse_coeffs))
    if args.stage1_epochs <= 0 and not args.stage1_source_ckpt and not args.stage1_source_run:
        args.stage1_source_run = DEFAULT_STAGE1_SOURCE_RUN
    if args.stage2_epochs > 0 and not args.stage2_source_ckpt and not args.stage2_source_run and not args.stage2_source_token_cache:
        args.stage2_source_run = DEFAULT_STAGE2_SOURCE_RUN
    return args


def _make_strategy(name: str, devices: int):
    strategy: object = name if devices > 1 else 'auto'
    if strategy in ('ddp', 'ddp_fork'):
        from lightning.pytorch.strategies import DDPStrategy
        strategy = DDPStrategy(broadcast_buffers=False, find_unused_parameters=False)
    return strategy


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    out_root = Path(args.out_dir).expanduser().resolve()
    stage1_dir = out_root / 'stage1'
    stage2_dir = out_root / 'stage2'

    data = CelebADataModule(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    data.setup()

    laser = build_laser_from_args(args)
    if args.stage1_epochs > 0:
        stage1_module = Stage1LaserModule(
            model=laser,
            lr=args.stage1_lr,
            bottleneck_weight=args.bottleneck_weight,
            out_dir=str(stage1_dir),
            lr_schedule=args.stage1_lr_schedule,
            warmup_epochs=args.stage1_warmup_epochs,
            min_lr_ratio=args.stage1_min_lr_ratio,
        )
        stage1_trainer = pl.Trainer(
            accelerator='gpu',
            devices=args.stage1_devices,
            strategy=_make_strategy(args.stage1_strategy, args.stage1_devices),
            max_epochs=args.stage1_epochs,
            logger=_build_wandb_logger(args, 'stage1'),
            enable_checkpointing=False,
            gradient_clip_val=1.0,
            precision=args.stage1_precision,
            log_every_n_steps=10,
            deterministic=False,
        )
        stage1_trainer.fit(stage1_module, datamodule=data)

    stage1_ckpt = _resolve_stage1_checkpoint(args, stage1_dir)
    _load_module_checkpoint(laser, stage1_ckpt)
    laser.eval()

    if args.stage2_epochs <= 0:
        return
    if not args.quantize_sparse_coeffs:
        raise NotImplementedError('proto_lightning.py stage2 currently supports quantized sparse coefficients only.')

    token_cache_path = stage2_dir / 'tokens_cache.pt'
    cache = _build_stage2_token_cache(args, data, laser, token_cache_path)
    tokens_flat = cache['tokens_flat']
    coeffs_flat = cache.get('coeffs_flat', None)
    h, w, d = cache['shape']
    sample_reference_stats = _compute_stage2_sample_reference_stats(
        laser,
        tokens_flat,
        coeffs_flat,
        h,
        w,
        d,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    )

    token_dataset = TensorDataset(tokens_flat)
    token_dm = TokenDataModule(token_dataset, batch_size=args.stage2_batch_size)
    prior = SpatialDepthPrior(
        build_spatial_depth_prior_config(
            laser.bottleneck,
            H=h,
            W=w,
            D=d,
            d_model=args.tf_d_model,
            n_heads=args.tf_heads,
            n_spatial_layers=args.tf_layers,
            n_depth_layers=max(1, args.tf_layers // 2),
            d_ff=args.tf_ff,
            dropout=args.tf_dropout,
            n_global_spatial_tokens=args.tf_global_tokens,
            real_valued_coeffs=False,
            coeff_max_fallback=args.coef_max,
        )
    )
    if args.stage2_source_ckpt:
        _load_module_checkpoint(prior, Path(args.stage2_source_ckpt).expanduser().resolve())

    stage2_module = Stage2LightningModule(
        prior=prior,
        ae_for_decode=laser,
        h=h,
        w=w,
        d=d,
        lr=args.stage2_lr,
        out_dir=str(stage2_dir),
        rq_atom_loss_weight=args.stage2_rq_atom_loss_weight,
        rq_coeff_loss_weight=args.stage2_rq_coeff_loss_weight,
        warmup_steps=args.stage2_warmup_steps,
        min_lr_ratio=args.stage2_min_lr_ratio,
        weight_decay=args.stage2_weight_decay,
        sample_every_steps=args.stage2_sample_every_steps,
        sample_batch_size=args.stage2_sample_batch_size,
        sample_candidate_factor=args.stage2_sample_candidate_factor,
        sample_temperature=args.stage2_sample_temperature,
        sample_top_k=args.stage2_sample_top_k,
        sample_image_size=args.stage2_sample_image_size,
        sample_reference_stats=sample_reference_stats,
    )
    stage2_trainer = pl.Trainer(
        accelerator='gpu',
        devices=args.stage2_devices,
        strategy=_make_strategy(args.stage2_strategy, args.stage2_devices),
        max_epochs=args.stage2_epochs,
        logger=_build_wandb_logger(args, 'stage2'),
        enable_checkpointing=False,
        gradient_clip_val=1.0,
        precision=args.stage2_precision,
        log_every_n_steps=10,
        deterministic=False,
    )
    stage2_trainer.fit(stage2_module, datamodule=token_dm)


if __name__ == '__main__':
    main()
