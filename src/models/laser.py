import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import math

from .encoder import Encoder
from .decoder import Decoder
from .bottleneck import DictionaryLearning
from .lpips import LPIPS
from .losses import multi_resolution_dct_loss, multi_resolution_gradient_loss

import torchmetrics
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
import wandb

class LASER(pl.LightningModule):
    def __init__(
            self,
            in_channels,
            num_hiddens,
            num_embeddings,
            embedding_dim,
            sparsity_level,
            num_residual_blocks,
            num_residual_hiddens,
            commitment_cost,
            learning_rate,
            beta,
            bottleneck_loss_weight=0.5,
            ksvd_iterations=1,
            dictionary_update_frequency=0,
            perceptual_weight=1.0,
            compute_fid=False,
            patch_size=1,
            multi_res_dct_weight=0.0,
            multi_res_dct_levels=3,
            multi_res_grad_weight=0.0,
            multi_res_grad_levels=3,
            use_online_learning=False,
            use_backprop_only=False,
            dict_learning_rate=1e-3,
            sparse_solver='omp',
            iht_iterations=10,
            iht_step_size=None,
            lista_layers=5,
            lista_tied_weights=False,
            lista_initial_threshold=0.1,
            fista_alpha=0.1,
            fista_tolerance=1e-3,
            fista_max_steps=50,
            sparsity_reg_weight=0.01,
            patch_stride=None,
            orthogonality_weight=0.01,
            per_pixel_sparse_coding=False,
            patch_flatten_order='channel_first',
            # Pattern quantization for autoregressive generation
            use_pattern_quantizer=False,
            num_patterns=2048,
            pattern_commitment_cost=0.25,
            pattern_ema_decay=0.99,
            pattern_temperature=1.0,
    ):
        """Initialize LASER model.

        Args:
            in_channels: Number of input channels (3 for RGB)
            num_hiddens: Number of hidden units
            num_embeddings: Number of dictionary atoms
            embedding_dim: Dimension of latent space
            sparsity_level: Number of non-zero coefficients in sparse coding
            num_residual_blocks: Number of residual blocks
            num_residual_hiddens: Number of hidden units in residual blocks
            commitment_cost: Commitment cost for bottleneck
            learning_rate: Learning rate for encoder/decoder
            beta: Beta parameter for Adam optimizer
            bottleneck_loss_weight: Weight for bottleneck loss term in total loss
            ksvd_iterations: Number of K-SVD dictionary update iterations per forward pass
            dictionary_update_frequency: Update dictionary every N training steps (0 = every step)
            perceptual_weight: Weight for perceptual loss
            compute_fid: Whether to compute FID
            patch_size: Spatial patch size (int or tuple) encoded per dictionary token
            multi_res_dct_weight: weight for DCT-based high-frequency loss
            multi_res_dct_levels: number of pyramid levels for DCT loss
            multi_res_grad_weight: weight for edge-preserving gradient loss
            multi_res_grad_levels: number of pyramid levels for gradient loss
            use_online_learning: whether to use online dictionary learning (vs K-SVD)
            use_backprop_only: whether to use only backprop for dictionary learning (no K-SVD/online)
            dict_learning_rate: learning rate for online dictionary updates
            sparse_solver: sparse coding algorithm ('omp', 'iht', 'topk', 'lista')
            iht_iterations: number of IHT iterations (if using IHT)
            iht_step_size: step size for IHT (None = auto-compute)
            lista_layers: number of LISTA layers (if using LISTA)
            lista_tied_weights: whether to share weights across LISTA layers
            lista_initial_threshold: initial soft threshold for LISTA
            fista_alpha: shrinkage parameter for FISTA sparse coding
            fista_tolerance: convergence tolerance for FISTA
            fista_max_steps: max iterations for FISTA
            sparsity_reg_weight: L1 regularization weight on sparse coefficients
            patch_stride: optional patch stride (defaults to patch_size; use patch_size/2 for 50% overlap)
            orthogonality_weight: weight for dictionary orthogonality loss (decorrelates atoms)
        """
        super(LASER, self).__init__()

        # Store parameters
        self.learning_rate = learning_rate
        self.beta = beta
        self.perceptual_weight = perceptual_weight
        self.log_images_every_n_steps = 100
        self.dictionary_update_frequency = dictionary_update_frequency
        self.compute_fid = compute_fid
        self.mr_dct_weight = multi_res_dct_weight
        self.mr_dct_levels = multi_res_dct_levels
        self.mr_grad_weight = multi_res_grad_weight
        self.mr_grad_levels = multi_res_grad_levels
        self.sparsity_reg_weight = sparsity_reg_weight
        self.bottleneck_loss_weight = bottleneck_loss_weight
        self.orthogonality_weight = orthogonality_weight

        # Initialize encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            num_hiddens=num_hiddens,
            num_residual_blocks=num_residual_blocks,
            num_residual_hiddens=num_residual_hiddens
        )

        self.pre_bottleneck = nn.Conv2d(in_channels=num_hiddens,
                                        out_channels=embedding_dim,
                                        kernel_size=1,
                                        stride=1)

        # Pattern quantization disabled inside bottleneck; handled externally if needed
        self.use_pattern_quantizer = False

        # Initialize Dictionary Learning bottleneck
        self.bottleneck = DictionaryLearning(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            sparsity_level=sparsity_level,
            commitment_cost=commitment_cost,
            ksvd_iterations=ksvd_iterations,
            patch_size=patch_size,
            use_online_learning=use_online_learning,
            use_backprop_only=use_backprop_only,
            dict_learning_rate=dict_learning_rate,
            sparse_solver=sparse_solver,
            iht_iterations=iht_iterations,
            iht_step_size=iht_step_size,
            lista_layers=lista_layers,
            lista_tied_weights=lista_tied_weights,
            lista_initial_threshold=lista_initial_threshold,
            fista_alpha=fista_alpha,
            fista_tolerance=fista_tolerance,
            fista_max_steps=fista_max_steps,
            patch_stride=patch_stride,
            per_pixel_sparse_coding=per_pixel_sparse_coding,
            patch_flatten_order=patch_flatten_order,
            # Pattern quantization disabled in bottleneck
            use_pattern_quantizer=False,
            num_patterns=num_patterns,
            pattern_commitment_cost=pattern_commitment_cost,
            pattern_ema_decay=pattern_ema_decay,
            pattern_temperature=pattern_temperature,
        )

        self.post_bottleneck = nn.Conv2d(in_channels=embedding_dim,
                                         out_channels=num_hiddens,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1)

        # Initialize decoder
        self.decoder = Decoder(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_blocks=num_residual_blocks,
            num_residual_hiddens=num_residual_hiddens
        )

        # Initialize LPIPS for perceptual loss only if used
        self.lpips = LPIPS() if self.perceptual_weight > 0 else None
        if self.lpips is not None:
            self.lpips.eval()
            for p in self.lpips.parameters():
                p.requires_grad = False

        # Separate metrics per split to avoid state leakage across train/val/test
        self.train_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.test_psnr = PeakSignalNoiseRatio(data_range=1.0)

        self.val_ssim = StructuralSimilarityIndexMeasure()
        self.test_ssim = StructuralSimilarityIndexMeasure()

        # Initialize metrics for Frechet Inception Distance (FID Score)
        if self.compute_fid:
            self.test_fid = FrechetInceptionDistance(feature=64, normalize=True)
        else:
            self.test_fid = None
        if self.test_fid is not None:
            self.test_fid.eval()
            for p in self.test_fid.parameters():
                p.requires_grad = False

        # Cache for validation visualization
        self._val_vis_batch = None
        # Interval for diagnostic logging to catch outlier batches
        self.diag_log_interval = 100

        # Save hyperparameters
        self.save_hyperparameters()

    def encode(self, x):
        """
        Encode input to latent representation.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            z_dl: latent representation from dictionary learning bottleneck
            bottleneck_loss: loss from the dictionary learning bottleneck
            coefficients: sparse coefficients
            pattern_indices: (optional) discrete pattern indices if pattern quantization enabled
            pattern_info: (optional) pattern quantization metrics if enabled
        """
        z_e = self.encoder(x)
        z_e = self.pre_bottleneck(z_e)

        z_dl, bottleneck_loss, coefficients = self.bottleneck(z_e)
        return z_dl, bottleneck_loss, coefficients

    def decode(self, z_dl):
        """
        Decode latent representation to reconstruction.

        Args:
            z_dl: latent representation from dictionary learning bottleneck

        Returns:
            x_recon: reconstruction of the input
        """
        z_dl = self.post_bottleneck(z_dl)
        x_recon = self.decoder(z_dl)
        return x_recon

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            tuple: (recon, bottleneck_loss, coefficients) or
                   (recon, bottleneck_loss, coefficients, pattern_indices, pattern_info)
                   if pattern quantization is enabled
        """
        z = self.encoder(x)
        z = self.pre_bottleneck(z)

        z_dl, bottleneck_loss, coefficients = self.bottleneck(z)
        z_dl = self.post_bottleneck(z_dl)
        recon = self.decoder(z_dl)
        return recon, bottleneck_loss, coefficients

    def compute_metrics(self, batch, prefix='train'):
        """Compute metrics for a batch."""
        # Get input
        x = batch[0] if isinstance(batch, (list, tuple)) else batch

        # Forward pass - handle pattern quantization
        recon_raw, bottleneck_loss, coefficients = self(x)
        pattern_indices = None
        pattern_info = None

        # Keep raw tensors for loss; create sanitized copies for metrics/visualization only
        recon_vis = torch.nan_to_num(recon_raw.detach(), nan=0.0, posinf=1.0, neginf=-1.0).clamp_(-1.0, 1.0)
        x_vis = torch.nan_to_num(x.detach(), nan=0.0, posinf=1.0, neginf=-1.0).clamp_(-1.0, 1.0)
        
        # Compute losses
        recon_loss = F.mse_loss(recon_raw, x)
        sparsity_loss = coefficients.abs().mean()
        # Basic stats for debug/monitoring
        input_mean = x.mean()
        input_std = x.std()
        recon_mean = recon_raw.mean()
        recon_std = recon_raw.std()
        if hasattr(self.bottleneck, "_last_diag"):
            diag = self.bottleneck._last_diag
        else:
            diag = {}
        
        # Perceptual loss - only compute during training for quality
        if self.lpips is not None and self.perceptual_weight > 0 and prefix == 'train':
            perceptual_loss = self.lpips(recon_raw, x).mean()
        else:
            perceptual_loss = torch.tensor(0.0, device=x.device)
        
        # Multi-resolution DCT loss - only compute during validation/testing for speed
        if self.mr_dct_weight > 0 and prefix != 'train':
            mr_dct_loss = multi_resolution_dct_loss(recon_raw, x, num_levels=self.mr_dct_levels)
        else:
            mr_dct_loss = torch.tensor(0.0, device=x.device)
        
        # Multi-resolution gradient loss - only compute during validation/testing for speed
        if self.mr_grad_weight > 0 and prefix != 'train':
            mr_grad_loss = multi_resolution_gradient_loss(recon_raw, x, num_levels=self.mr_grad_levels)
        else:
            mr_grad_loss = torch.tensor(0.0, device=x.device)

        # Orthogonality loss - encourages decorrelated dictionary atoms
        # This improves sparse coding by reducing redundancy between atoms
        if self.orthogonality_weight > 0:
            ortho_loss = self.bottleneck.orthogonality_loss()
        else:
            ortho_loss = torch.tensor(0.0, device=x.device)

        # Total loss
        total_loss = (
            recon_loss +
            self.bottleneck_loss_weight * bottleneck_loss +
            self.perceptual_weight * perceptual_loss +
            self.mr_dct_weight * mr_dct_loss +
            self.mr_grad_weight * mr_grad_loss +
            self.sparsity_reg_weight * sparsity_loss +
            self.orthogonality_weight * ortho_loss
        )
        
        # Compute metrics on de-normalized tensors if mean/std available; otherwise assume [-1,1]
        dm = getattr(getattr(self, "trainer", None), "datamodule", None)
        x_clean = torch.nan_to_num(x.detach(), nan=0.0, posinf=1.0, neginf=-1.0)
        recon_clean = torch.nan_to_num(recon_raw.detach(), nan=0.0, posinf=1.0, neginf=-1.0)
        if dm is not None and hasattr(dm, "config") and hasattr(dm.config, "mean") and hasattr(dm.config, "std"):
            mean = torch.tensor(dm.config.mean, device=x_clean.device, dtype=x_clean.dtype).view(1, -1, 1, 1)
            std = torch.tensor(dm.config.std, device=x_clean.device, dtype=x_clean.dtype).view(1, -1, 1, 1)
            x_dn = (x_clean * std + mean).clamp(0.0, 1.0)
            recon_dn = (recon_clean * std + mean).clamp(0.0, 1.0)
        else:
            x_dn = (x_clean + 1.0) / 2.0
            recon_dn = (recon_clean + 1.0) / 2.0

        if prefix == 'train':
            psnr_metric = self.train_psnr
            ssim = torch.tensor(0.0, device=x_dn.device)
        elif prefix == 'val':
            psnr_metric = self.val_psnr
            ssim = self.val_ssim(recon_dn, x_dn)
        else:
            psnr_metric = self.test_psnr
            ssim = self.test_ssim(recon_dn, x_dn)

        psnr = psnr_metric(recon_dn, x_dn)
        
        # Compute sparsity
        sparsity = (coefficients.abs() > 1e-6).float().mean()
        
        # Log metrics; explicitly set on_step/on_epoch to avoid Lightning warnings in DDP
        log_kwargs = dict(on_step=prefix == 'train', on_epoch=True, sync_dist=True)
        self.log(f'{prefix}/loss', total_loss, prog_bar=True, **log_kwargs)
        self.log(f'{prefix}/recon_loss', recon_loss, **log_kwargs)
        self.log(f'{prefix}/bottleneck_loss', bottleneck_loss, **log_kwargs)
        self.log(f'{prefix}/weighted_bottleneck_loss', self.bottleneck_loss_weight * bottleneck_loss, **log_kwargs)
        self.log(f'{prefix}/perceptual_loss', perceptual_loss, **log_kwargs)
        self.log(f'{prefix}/mr_dct_loss', mr_dct_loss, **log_kwargs)
        self.log(f'{prefix}/mr_grad_loss', mr_grad_loss, **log_kwargs)
        self.log(f'{prefix}/sparsity_loss', sparsity_loss, **log_kwargs)
        self.log(f'{prefix}/orthogonality_loss', ortho_loss, **log_kwargs)
        self.log(f'{prefix}/psnr', psnr, prog_bar=True, **log_kwargs)
        self.log(f'{prefix}/input_mean', input_mean, **log_kwargs)
        self.log(f'{prefix}/input_std', input_std, **log_kwargs)
        self.log(f'{prefix}/recon_mean', recon_mean, **log_kwargs)
        self.log(f'{prefix}/recon_std', recon_std, **log_kwargs)
        if diag:
            self.log(f'{prefix}/dict_norm_max', diag.get("dict_norm_max", torch.tensor(0.0, device=x.device)), **log_kwargs)
            self.log(f'{prefix}/dict_norm_min', diag.get("dict_norm_min", torch.tensor(0.0, device=x.device)), **log_kwargs)
            self.log(f'{prefix}/coeff_norm_max', diag.get("coeff_norm_max", torch.tensor(0.0, device=x.device)), **log_kwargs)
            self.log(f'{prefix}/coeff_norm_mean', diag.get("coeff_norm_mean", torch.tensor(0.0, device=x.device)), **log_kwargs)
        if prefix != 'train':
            self.log(f'{prefix}/ssim', ssim, prog_bar=True, **log_kwargs)
        self.log(f'{prefix}/sparsity', sparsity, **log_kwargs)

        # Log bottleneck subcomponents for diagnostics
        last_losses = getattr(self.bottleneck, '_last_bottleneck_losses', None)
        if last_losses:
            self.log(f'{prefix}/e_latent_loss', last_losses['e_latent_loss'], **log_kwargs)
            self.log(f'{prefix}/dl_latent_loss', last_losses['dl_latent_loss'], **log_kwargs)
            self.log(f'{prefix}/pattern_loss', last_losses['pattern_loss'], **log_kwargs)

        # Occasional diagnostic logging to catch outlier batches
        if prefix == 'train' and (self.global_step % self.diag_log_interval == 0):
            x_abs_max = torch.nan_to_num(x).abs().max()
            recon_abs_max = torch.nan_to_num(recon_raw).abs().max()
            coeff_abs_max = torch.nan_to_num(coefficients).abs().max()
            coeff_abs_mean = torch.nan_to_num(coefficients).abs().mean()
            nan_frac = (~torch.isfinite(recon_raw)).float().mean()
            diag_kwargs = dict(on_step=True, on_epoch=False, sync_dist=True, prog_bar=False)
            self.log('train/diag/input_abs_max', x_abs_max, **diag_kwargs)
            self.log('train/diag/recon_abs_max', recon_abs_max, **diag_kwargs)
            self.log('train/diag/coeff_abs_max', coeff_abs_max, **diag_kwargs)
            self.log('train/diag/coeff_abs_mean', coeff_abs_mean, **diag_kwargs)
            self.log('train/diag/recon_nan_frac', nan_frac, **diag_kwargs)
        
        return total_loss, recon_vis, x_vis

    def training_step(self, batch, batch_idx):
        """Training step."""
        # Prevent unused dictionary gradients from accumulating when not optimized via backprop
        if not self.bottleneck.use_backprop_only:
            self.bottleneck.dictionary.grad = None

        # Control dictionary updates based on update frequency
        # Enable K-SVD/online updates only if not using backprop-only mode
        # and if we're at the right step according to dictionary_update_frequency
        if not self.bottleneck.use_backprop_only:
            if self.dictionary_update_frequency == 0:
                # Update every step
                self.bottleneck.enable_ksvd_update = True
            else:
                # Update only every N steps
                current_step = self.global_step
                should_update = (current_step % self.dictionary_update_frequency) == 0
                self.bottleneck.enable_ksvd_update = should_update
        else:
            # In backprop-only mode, never use K-SVD updates
            self.bottleneck.enable_ksvd_update = False

        loss, recon, x = self.compute_metrics(batch, prefix='train')

        # Log images periodically
        if batch_idx % self.log_images_every_n_steps == 0:
            self.log_images(x, recon, prefix='train')

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        if batch_idx == 0:
            self._maybe_store_val_batch(batch)
        loss, recon, x = self.compute_metrics(batch, prefix='val')
        
        # Log images periodically
        if batch_idx % self.log_images_every_n_steps == 0:
            self.log_images(x, recon, prefix='val')
        
        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        loss, recon, x = self.compute_metrics(batch, prefix='test')
        
        # Update FID if enabled
        if self.test_fid is not None:
            # Ensure the TorchMetrics module lives on the same device as the current rank
            fid_device = x.device
            self.test_fid = self.test_fid.to(fid_device)
            x_fid = torch.nan_to_num(x.detach(), nan=0.0, posinf=1.0, neginf=-1.0)
            recon_fid = torch.nan_to_num(recon.detach(), nan=0.0, posinf=1.0, neginf=-1.0)
            x_fid = ((x_fid + 1.0) / 2.0).clamp_(0.0, 1.0).to(fid_device, dtype=torch.float32)
            recon_fid = ((recon_fid + 1.0) / 2.0).clamp_(0.0, 1.0).to(fid_device, dtype=torch.float32)
            self.test_fid.update(x_fid, real=True)
            self.test_fid.update(recon_fid, real=False)
        
        # Log images periodically
        if batch_idx % self.log_images_every_n_steps == 0:
            self.log_images(x, recon, prefix='test')
        
        return loss

    def on_test_epoch_end(self):
        """Compute FID at the end of test epoch."""
        if self.test_fid is not None:
            fid_score = self.test_fid.compute()
            self.log('test/fid', fid_score, sync_dist=True)
            self.test_fid.reset()

    def log_images(self, x, recon, prefix='val', max_images=8):
        """Log reconstruction images to wandb."""
        # Only log from rank zero in DDP to avoid multi-process logger contention
        if getattr(getattr(self, "trainer", None), "is_global_zero", False) is False:
            return
        # If logger is disabled or doesn't support experiment logging, skip
        if not getattr(self, "logger", None) or not hasattr(self.logger, "experiment") or not hasattr(self.logger.experiment, "log"):
            return
        
        # Take first 32 images
        x = x[:32]
        recon = recon[:32]
        
        # Create image grids
        # De-normalize using datamodule config if available; otherwise assume [-1,1] → [0,1]
        dm = getattr(getattr(self, "trainer", None), "datamodule", None)
        if dm is not None and hasattr(dm, "config") and hasattr(dm.config, "mean") and hasattr(dm.config, "std"):
            mean = torch.tensor(dm.config.mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
            std = torch.tensor(dm.config.std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
            x_disp = x * std + mean
            recon_disp = recon * std + mean
        else:
            x_disp = (x + 1.0) / 2.0
            recon_disp = (recon + 1.0) / 2.0
        x_disp = x_disp.clamp(0.0, 1.0)
        recon_disp = recon_disp.clamp(0.0, 1.0)
        x_grid = torchvision.utils.make_grid(x_disp, nrow=8, normalize=False)
        recon_grid = torchvision.utils.make_grid(recon_disp, nrow=8, normalize=False)
        
        # Sanitize NaN/Inf and clamp to [0,1] before converting to numpy
        x_grid = torch.nan_to_num(x_grid, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        recon_grid = torch.nan_to_num(recon_grid, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        
        # Convert to numpy and transpose to correct format (H,W,C)
        x_grid = x_grid.cpu().numpy().transpose(1, 2, 0)
        recon_grid = recon_grid.cpu().numpy().transpose(1, 2, 0)
        
        # Log to wandb
        self.logger.experiment.log({
            f"{prefix}/images": [
                wandb.Image(x_grid, caption="Original"),
                wandb.Image(recon_grid, caption="Reconstructed")
            ],
            f"{prefix}/reconstruction_error": F.mse_loss(recon, x).item(),
            "global_step": self.global_step
        })

    def configure_optimizers(self):
        """Configure optimizers."""
        main_params = list(self.encoder.parameters()) + \
                      list(self.pre_bottleneck.parameters()) + \
                      list(self.post_bottleneck.parameters()) + \
                      list(self.decoder.parameters())

        param_groups = [
            {"params": main_params, "lr": self.learning_rate, "weight_decay": 1e-4},
        ]

        # Give the dictionary its own (typically higher) LR and no weight decay
        if self.bottleneck.use_backprop_only:
            dict_params = list(self.bottleneck.parameters())
            dict_lr = getattr(self.bottleneck, "dict_learning_rate", self.learning_rate)
            param_groups.append({"params": dict_params, "lr": dict_lr, "weight_decay": 0.0})

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.learning_rate,
            betas=(self.beta, 0.999),
            weight_decay=0.0,
        )

        return optimizer
    
    def _maybe_store_val_batch(self, batch):
        """Cache a small val batch (CPU) for visualization."""
        if not getattr(getattr(self, "trainer", None), "is_global_zero", False):
            return
        if getattr(self, "_val_vis_batch", None) is not None:
            return
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            x, y = batch, None
        x_keep = x[:4].detach().cpu()
        y_keep = y[:4].detach().cpu() if y is not None and hasattr(y, "detach") else None
        self._val_vis_batch = (x_keep, y_keep)

    def _denormalize_for_display(self, x):
        """Convert normalized tensor to [0,1] range for visualization."""
        dm = getattr(getattr(self, "trainer", None), "datamodule", None)
        if dm is not None and hasattr(dm, "config") and hasattr(dm.config, "mean") and hasattr(dm.config, "std"):
            mean = torch.tensor(dm.config.mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
            std = torch.tensor(dm.config.std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
            x_disp = x * std + mean
        else:
            x_disp = (x + 1.0) / 2.0
        return x_disp.clamp(0.0, 1.0)

    def _latent_rgb_projection(self, z_latent):
        """Project latent feature maps to RGB via PCA (per-batch)."""
        b, c, h, w = z_latent.shape
        feats = z_latent.permute(0, 2, 3, 1).reshape(-1, c)
        feats_centered = feats - feats.mean(dim=0, keepdim=True)
        try:
            _, _, v = torch.pca_lowrank(feats_centered, q=min(c, 6))
            proj = feats_centered @ v[:, :3]
        except Exception:
            proj = feats_centered[:, :3]
        proj = proj.view(b, h, w, 3)
        proj_np = []
        for i in range(b):
            img = proj[i]
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)
            proj_np.append(img.detach().cpu().numpy())
        return proj_np

    def _sparse_heatmaps(self, coefficients, spatial_size):
        """Generate per-image sparse-activation heatmaps (mean |coeff| per latent location)."""
        b = spatial_size[0]
        h_in, w_in = spatial_size[1], spatial_size[2]
        num_patches = coefficients.shape[1] // b
        patch_h, patch_w = self.bottleneck.patch_size
        h_tiles = math.ceil(h_in / patch_h)
        w_tiles = math.ceil(w_in / patch_w)
        energy = coefficients.abs().view(self.bottleneck.num_embeddings, b, num_patches).mean(dim=0)  # [B, P]
        try:
            energy = energy.view(b, h_tiles, w_tiles)
        except Exception:
            # Fallback: infer square-ish grid
            side = int(math.sqrt(num_patches))
            energy = energy.view(b, side, num_patches // side)
        heat = energy.unsqueeze(1)
        heat = F.interpolate(heat, size=(h_in, w_in), mode='nearest')  # [B,1,H,W]
        heat_np = []
        for i in range(b):
            hmap = heat[i, 0]
            hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-6)
            heat_np.append(hmap.detach().cpu().numpy())
        return heat_np

    def _log_val_latent_visuals(self):
        """Log latent RGB projections and sparse-code heatmaps on cached val batch."""
        if not getattr(getattr(self, "trainer", None), "is_global_zero", False):
            return
        if not getattr(self, "logger", None) or not hasattr(self.logger, "experiment"):
            return
        if self._val_vis_batch is None:
            return
        x_cpu, _ = self._val_vis_batch
        x = x_cpu.to(self.device)
        self.eval()
        with torch.no_grad():
            z = self.encoder(x)
            z = self.pre_bottleneck(z)
            z_dl, _, coefficients = self.bottleneck(z)
            latent_rgb = self._latent_rgb_projection(z_dl)
            heatmaps = self._sparse_heatmaps(coefficients, (x.shape[0], x.shape[2], x.shape[3]))
        log_payload = {}
        cmap = plt.cm.inferno
        for idx in range(x.shape[0]):
            latent_img = latent_rgb[idx]
            heat = heatmaps[idx]
            heat_rgb = cmap(heat)[..., :3]
            log_payload.setdefault("val/latent_rgb", []).append(wandb.Image(latent_img, caption=f"latent_rgb_{idx}"))
            log_payload.setdefault("val/sparse_heatmap", []).append(wandb.Image(heat_rgb, caption=f"sparse_heatmap_{idx}"))
        if log_payload:
            log_payload["global_step"] = self.global_step
            self.logger.experiment.log(log_payload)
        self.train()
        self._val_vis_batch = None

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self._val_vis_batch = None

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        self._log_val_latent_visuals()

    def on_test_start(self):
        super().on_test_start()
        if self.test_fid is not None:
            # Align metric buffers with the rank's device and clear any stale states
            self.test_fid = self.test_fid.to(self.device)
            self.test_fid.reset()
