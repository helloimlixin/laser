import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import torchvision
from contextlib import nullcontext
from typing import Optional, Sequence, Tuple

from .encoder import Encoder
from .decoder import Decoder
from .bottleneck import DictionaryLearning, SparseCodes
from .lpips import LPIPS

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
            perceptual_weight=1.0,
            compute_fid=False,
            log_images_every_n_steps=100,
            diag_log_interval=0,
            enable_val_latent_visuals=False,
            dict_learning_rate=None,
            patch_based=True,
            patch_size=4,
            patch_stride=2,
            patch_reconstruction="hann",
            sparsity_reg_weight=0.01,
            coherence_weight=0.0,
            **kwargs,
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
            perceptual_weight: Weight for perceptual loss
            compute_fid: Whether to compute FID
            log_images_every_n_steps: image logging cadence; 0 disables image logging
            diag_log_interval: diagnostic logging cadence; 0 disables extra train diagnostics
            enable_val_latent_visuals: whether to run the extra validation PCA/heatmap pass
            dict_learning_rate: optional learning rate override for dictionary atoms
            patch_based: whether to use latent patch sparse coding instead of per-site coding
            patch_size: latent patch size used for sparse coding
            patch_stride: latent patch stride used for sparse coding
            patch_reconstruction: patch stitching rule, either 'center_crop' or 'hann'
            sparsity_reg_weight: L1 regularization weight on sparse coefficients
            coherence_weight: weight for dictionary coherence regularization
        """
        super(LASER, self).__init__()

        legacy_coherence = kwargs.pop("orthogonality_weight", None)
        kwargs.pop("perceptual_batch_size", None)
        kwargs.pop("use_online_learning", None)
        kwargs.pop("use_backprop_only", None)
        kwargs.pop("sparse_solver", None)
        kwargs.pop("fast_omp", None)
        kwargs.pop("omp_diag_eps", None)
        kwargs.pop("omp_cholesky_eps", None)
        kwargs.pop("sparse_coding_scheme", None)
        kwargs.pop("lista_steps", None)
        kwargs.pop("lista_step_size_init", None)
        kwargs.pop("lista_threshold_init", None)
        kwargs.pop("lista_layers", None)
        kwargs.pop("lista_tied_weights", None)
        kwargs.pop("lista_initial_threshold", None)
        kwargs.pop("dictionary_update_mode", None)
        kwargs.pop("dict_ema_decay", None)
        kwargs.pop("dict_ema_eps", None)
        kwargs.pop("patch_flatten_order", None)
        kwargs.pop("per_pixel_sparse_coding", None)
        if legacy_coherence is not None and float(coherence_weight) == 0.0:
            coherence_weight = float(legacy_coherence)
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported LASER arguments: {unknown}")

        # Store parameters
        self.learning_rate = learning_rate
        self.beta = beta
        self.perceptual_weight = perceptual_weight
        self.log_images_every_n_steps = max(int(log_images_every_n_steps), 0)
        self.compute_fid = compute_fid
        self.sparsity_reg_weight = sparsity_reg_weight
        self.bottleneck_loss_weight = bottleneck_loss_weight
        self.coherence_weight = coherence_weight
        self.diag_log_interval = max(int(diag_log_interval), 0)
        self.enable_val_latent_visuals = bool(enable_val_latent_visuals)
        if dict_learning_rate is not None and float(dict_learning_rate) <= 0.0:
            dict_learning_rate = None

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

        # Initialize Dictionary Learning bottleneck
        self.bottleneck = DictionaryLearning(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            sparsity_level=sparsity_level,
            commitment_cost=commitment_cost,
            dict_learning_rate=dict_learning_rate,
            patch_based=patch_based,
            patch_size=patch_size,
            patch_stride=patch_stride,
            patch_reconstruction=patch_reconstruction,
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

        # Save hyperparameters
        self.save_hyperparameters()

    def _should_log_images(self, batch_idx):
        return self.log_images_every_n_steps > 0 and batch_idx % self.log_images_every_n_steps == 0

    def _ddp_barrier_if_needed(self):
        """Keep ranks aligned after rank-0-only work inside a step (avoids NCCL timeouts)."""
        trainer = self._trainer_ref()
        if trainer is None or getattr(trainer, "world_size", 1) <= 1:
            return
        strategy = getattr(trainer, "strategy", None)
        barrier = getattr(strategy, "barrier", None)
        if callable(barrier):
            barrier()

    def _should_log_train_diagnostics(self):
        return self.diag_log_interval > 0 and self.global_step % self.diag_log_interval == 0

    def _bottleneck_autocast_context(self, z):
        if z.is_cuda and torch.is_autocast_enabled():
            return torch.autocast(device_type=z.device.type, enabled=False)
        return nullcontext()

    def _trainer_ref(self):
        return self.__dict__.get("_trainer", None)

    def on_fit_start(self):
        self.bottleneck.normalize_dictionary_()

    def on_before_optimizer_step(self, optimizer):
        del optimizer
        self.bottleneck.project_dictionary_gradient_()

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # Renormalize after the step: Lightning runs zero_grad (and on_before_zero_grad)
        # after forward but before backward inside the optimizer closure, so in-place
        # dictionary updates there break autograd for the current batch.
        optimizer.step(closure=optimizer_closure)
        self.bottleneck.normalize_dictionary_()

    def encode(self, x):
        """
        Encode input to latent representation.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            z_dl: latent representation from dictionary learning bottleneck
            bottleneck_loss: loss from the dictionary learning bottleneck
            sparse_codes: sparse support/value tensors
        """
        z_e = self.encoder(x)
        z_e = self.pre_bottleneck(z_e)
        with self._bottleneck_autocast_context(z_e):
            z_dl, bottleneck_loss, sparse_codes = self.bottleneck(z_e.float())
        return z_dl, bottleneck_loss, sparse_codes

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

    @torch.no_grad()
    def encode_to_tokens(
        self,
        x: torch.Tensor,
        *,
        coeff_vocab_size: int,
        coeff_max: float,
        coeff_quantization: str = "uniform",
        coeff_mu: float = 0.0,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Encode images to interleaved quantized sparse tokens."""
        z_dl, _, sparse_codes = self.encode(x)
        tokens, _ = self.bottleneck.sparse_codes_to_tokens(
            sparse_codes,
            coeff_vocab_size=coeff_vocab_size,
            coeff_max=coeff_max,
            coeff_quantization=coeff_quantization,
            coeff_mu=coeff_mu,
        )
        return tokens, (int(z_dl.shape[-2]), int(z_dl.shape[-1]))

    @torch.no_grad()
    def infer_latent_hw(self, image_hw: Tuple[int, int]) -> Tuple[int, int]:
        """Infer encoder latent spatial size for a given input image size."""
        image_h, image_w = int(image_hw[0]), int(image_hw[1])
        if image_h <= 0 or image_w <= 0:
            raise ValueError(f"image_hw must be positive, got {(image_h, image_w)}")
        device = next(self.parameters()).device
        dummy = torch.zeros(
            1,
            int(self.hparams.in_channels),
            image_h,
            image_w,
            device=device,
            dtype=torch.float32,
        )
        z = self.pre_bottleneck(self.encoder(dummy))
        return int(z.shape[-2]), int(z.shape[-1])

    @torch.no_grad()
    def decode_from_tokens(
        self,
        tokens: torch.Tensor,
        *,
        latent_hw: Optional[Tuple[int, int]] = None,
        atom_vocab_size: Optional[int] = None,
        coeff_vocab_size: Optional[int] = None,
        coeff_bin_values: Optional[Sequence[float] | torch.Tensor] = None,
        coeff_max: Optional[float] = None,
        coeff_quantization: str = "uniform",
        coeff_mu: float = 0.0,
    ) -> torch.Tensor:
        """Decode a quantized sparse-token grid back to image space."""
        z_q = self.bottleneck.tokens_to_latent(
            tokens,
            latent_hw=latent_hw,
            atom_vocab_size=atom_vocab_size,
            coeff_vocab_size=coeff_vocab_size,
            coeff_bin_values=coeff_bin_values,
            coeff_max=coeff_max,
            coeff_quantization=coeff_quantization,
            coeff_mu=coeff_mu,
        )
        return self.decode(z_q)

    decode_tokens = decode_from_tokens

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            tuple: (recon, bottleneck_loss, sparse_codes)
        """
        z = self.encoder(x)
        z = self.pre_bottleneck(z)
        with self._bottleneck_autocast_context(z):
            z_dl, bottleneck_loss, sparse_codes = self.bottleneck(z.float())
        z_dl = self.post_bottleneck(z_dl)
        recon = self.decoder(z_dl)
        return recon, bottleneck_loss, sparse_codes

    def _dense_coeff_abs_mean(self, sparse_codes: SparseCodes):
        num_sites = max(int(sparse_codes.values.shape[0] * sparse_codes.values.shape[1] * sparse_codes.values.shape[2]), 1)
        denom = float(sparse_codes.num_embeddings * num_sites)
        return sparse_codes.values.abs().sum() / denom

    def _support_fraction(self, sparse_codes: SparseCodes):
        num_sites = max(int(sparse_codes.values.shape[0] * sparse_codes.values.shape[1] * sparse_codes.values.shape[2]), 1)
        denom = float(sparse_codes.num_embeddings * num_sites)
        return sparse_codes.support.numel() / denom

    def _effective_coeff_nonzero_fraction(self, sparse_codes: SparseCodes, threshold=1e-6):
        num_sites = max(int(sparse_codes.values.shape[0] * sparse_codes.values.shape[1] * sparse_codes.values.shape[2]), 1)
        denom = float(sparse_codes.num_embeddings * num_sites)
        return (sparse_codes.values.abs() > threshold).float().sum() / denom

    def compute_metrics(self, batch, prefix='train'):
        """Compute metrics for a batch."""
        # Get input
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        needs_train_diag = prefix == 'train' and self._should_log_train_diagnostics()
        should_log_distribution_metrics = prefix != 'train' or needs_train_diag

        recon_raw, bottleneck_loss, sparse_codes = self(x)

        # Keep raw tensors for loss; create sanitized copies for metrics/visualization only
        recon_vis = torch.nan_to_num(recon_raw.detach(), nan=0.0, posinf=1.0, neginf=-1.0).clamp_(-1.0, 1.0)
        x_vis = torch.nan_to_num(x.detach(), nan=0.0, posinf=1.0, neginf=-1.0).clamp_(-1.0, 1.0)
        
        # Compute losses
        recon_loss = F.mse_loss(recon_raw, x)
        sparsity_loss = self._dense_coeff_abs_mean(sparse_codes)
        input_mean = input_std = recon_mean = recon_std = None
        if should_log_distribution_metrics:
            input_mean = x.mean()
            input_std = x.std()
            recon_mean = recon_raw.mean()
            recon_std = recon_raw.std()
        diag = getattr(self.bottleneck, "_last_diag", {})
        
        # Perceptual loss - only compute during training for quality
        if self.lpips is not None and self.perceptual_weight > 0 and prefix == 'train':
            perceptual_loss = self.lpips(recon_raw, x).mean()
        else:
            perceptual_loss = recon_raw.new_zeros(())

        # Proto-style dictionary learning uses a normalized dictionary plus an
        # optional coherence penalty rather than a dense orthogonality target.
        if self.coherence_weight > 0:
            coherence_loss = self.bottleneck.coherence_penalty()
        else:
            coherence_loss = recon_raw.new_zeros(())

        # Total loss
        total_loss = (
            recon_loss +
            self.bottleneck_loss_weight * bottleneck_loss +
            self.perceptual_weight * perceptual_loss +
            self.sparsity_reg_weight * sparsity_loss +
            self.coherence_weight * coherence_loss
        )
        
        # Compute PSNR on de-normalized tensors if mean/std available; otherwise assume [-1,1].
        dm = getattr(self._trainer_ref(), "datamodule", None)
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

        ssim = None
        if prefix == 'train':
            psnr = self.train_psnr(recon_dn, x_dn)
        elif prefix == 'val':
            psnr = self.val_psnr(recon_dn, x_dn)
            ssim = self.val_ssim(recon_dn, x_dn)
        else:
            psnr = self.test_psnr(recon_dn, x_dn)
            ssim = self.test_ssim(recon_dn, x_dn)
        
        # Compute sparsity
        sparsity = self._support_fraction(sparse_codes)
        effective_sparsity = self._effective_coeff_nonzero_fraction(sparse_codes)
        
        # Epoch-level metrics must use sync_dist in DDP so Lightning can reduce across ranks.
        trainer = self._trainer_ref()
        _ws = getattr(trainer, "world_size", 1) if trainer is not None else 1
        log_kwargs = dict(on_step=prefix == 'train', on_epoch=True, sync_dist=(_ws > 1))
        self.log(f'{prefix}/loss', total_loss, prog_bar=True, **log_kwargs)
        self.log(f'{prefix}/recon_loss', recon_loss, **log_kwargs)
        self.log(f'{prefix}/bottleneck_loss', bottleneck_loss, **log_kwargs)
        self.log(f'{prefix}/weighted_bottleneck_loss', self.bottleneck_loss_weight * bottleneck_loss, **log_kwargs)
        self.log(f'{prefix}/perceptual_loss', perceptual_loss, **log_kwargs)
        self.log(f'{prefix}/sparsity_loss', sparsity_loss, **log_kwargs)
        self.log(f'{prefix}/coherence_loss', coherence_loss, **log_kwargs)
        if psnr is not None:
            self.log(f'{prefix}/psnr', psnr, prog_bar=True, **log_kwargs)
        if should_log_distribution_metrics:
            self.log(f'{prefix}/input_mean', input_mean, **log_kwargs)
            self.log(f'{prefix}/input_std', input_std, **log_kwargs)
            self.log(f'{prefix}/recon_mean', recon_mean, **log_kwargs)
            self.log(f'{prefix}/recon_std', recon_std, **log_kwargs)
        if diag and should_log_distribution_metrics:
            self.log(f'{prefix}/dict_norm_max', diag.get("dict_norm_max", torch.tensor(0.0, device=x.device)), **log_kwargs)
            self.log(f'{prefix}/dict_norm_min', diag.get("dict_norm_min", torch.tensor(0.0, device=x.device)), **log_kwargs)
            self.log(f'{prefix}/dict_norm_mean', diag.get("dict_norm_mean", torch.tensor(0.0, device=x.device)), **log_kwargs)
            self.log(f'{prefix}/coeff_abs_max', diag.get("coeff_abs_max", torch.tensor(0.0, device=x.device)), **log_kwargs)
            self.log(f'{prefix}/coeff_abs_mean', diag.get("coeff_abs_mean", torch.tensor(0.0, device=x.device)), **log_kwargs)
            coherence_max, coherence_mean_abs, coherence_rms = self.bottleneck.coherence_stats()
            self.log(f'{prefix}/dict_coherence', coherence_max, **log_kwargs)
            self.log(f'{prefix}/dict_coherence_mean_abs', coherence_mean_abs, **log_kwargs)
            self.log(f'{prefix}/dict_coherence_rms', coherence_rms, **log_kwargs)
        if ssim is not None:
            self.log(f'{prefix}/ssim', ssim, prog_bar=True, **log_kwargs)
        self.log(f'{prefix}/sparsity', sparsity, **log_kwargs)
        self.log(f'{prefix}/effective_sparsity', effective_sparsity, **log_kwargs)

        # Log bottleneck subcomponents for diagnostics
        if self.bottleneck._last_e_latent_loss is not None:
            self.log(f'{prefix}/e_latent_loss', self.bottleneck._last_e_latent_loss, **log_kwargs)
        if self.bottleneck._last_dl_latent_loss is not None:
            self.log(f'{prefix}/dl_latent_loss', self.bottleneck._last_dl_latent_loss, **log_kwargs)

        # Occasional diagnostic logging to catch outlier batches
        if needs_train_diag:
            x_abs_max = torch.nan_to_num(x).abs().max()
            recon_abs_max = torch.nan_to_num(recon_raw).abs().max()
            coeff_abs_max = torch.nan_to_num(sparse_codes.values).abs().max()
            coeff_abs_mean = self._dense_coeff_abs_mean(sparse_codes)
            nan_frac = (~torch.isfinite(recon_raw)).float().mean()
            diag_kwargs = dict(on_step=True, on_epoch=False, sync_dist=False, prog_bar=False)
            self.log('train/diag/input_abs_max', x_abs_max, **diag_kwargs)
            self.log('train/diag/recon_abs_max', recon_abs_max, **diag_kwargs)
            self.log('train/diag/coeff_abs_max', coeff_abs_max, **diag_kwargs)
            self.log('train/diag/coeff_abs_mean', coeff_abs_mean, **diag_kwargs)
            self.log('train/diag/recon_nan_frac', nan_frac, **diag_kwargs)
        
        return total_loss, recon_vis, x_vis

    def training_step(self, batch, batch_idx):
        """Training step."""
        loss, recon, x = self.compute_metrics(batch, prefix='train')

        # Log images periodically
        if self._should_log_images(batch_idx):
            self.log_images(x, recon, prefix='train')
            self._ddp_barrier_if_needed()

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        if batch_idx == 0:
            self._maybe_store_val_batch(batch)
        loss, recon, x = self.compute_metrics(batch, prefix='val')
        
        # Log images periodically
        if self._should_log_images(batch_idx):
            self.log_images(x, recon, prefix='val')
            self._ddp_barrier_if_needed()
        
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
        if self._should_log_images(batch_idx):
            self.log_images(x, recon, prefix='test')
            self._ddp_barrier_if_needed()
        
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
        if getattr(self._trainer_ref(), "is_global_zero", False) is False:
            return
        # If logger is disabled or doesn't support experiment logging, skip
        if not getattr(self, "logger", None) or not hasattr(self.logger, "experiment") or not hasattr(self.logger.experiment, "log"):
            return
        
        # Take a small fixed subset to keep W&B logging cheap.
        x = x[:max_images]
        recon = recon[:max_images]
        
        # Create image grids
        # De-normalize using datamodule config if available; otherwise assume [-1,1] → [0,1]
        dm = getattr(self._trainer_ref(), "datamodule", None)
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
        x_grid = torchvision.utils.make_grid(x_disp, nrow=min(8, max_images), normalize=False)
        recon_grid = torchvision.utils.make_grid(recon_disp, nrow=min(8, max_images), normalize=False)
        
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
            {"params": main_params, "lr": self.learning_rate},
        ]

        # Match proto.py: shared Adam, with an optional dictionary-specific LR.
        dict_lr = getattr(self.bottleneck, "dict_learning_rate", None)
        if dict_lr is None:
            dict_lr = self.learning_rate
        param_groups.append({"params": [self.bottleneck.dictionary], "lr": dict_lr})

        optimizer = torch.optim.Adam(
            param_groups,
            betas=(self.beta, 0.999),
        )

        return optimizer
    
    def _maybe_store_val_batch(self, batch):
        """Cache a small val batch (CPU) for visualization."""
        if not self.enable_val_latent_visuals:
            return
        if not getattr(self._trainer_ref(), "is_global_zero", False):
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
        dm = getattr(self._trainer_ref(), "datamodule", None)
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

    def _sparse_heatmaps(self, sparse_codes, image_hw):
        """Generate per-image sparse-activation heatmaps (mean |coeff| per latent location)."""
        h_in, w_in = image_hw
        energy = sparse_codes.values.abs().sum(dim=-1) / float(sparse_codes.num_embeddings)
        heat = energy.unsqueeze(1)
        heat = F.interpolate(heat, size=(h_in, w_in), mode='nearest')  # [B,1,H,W]
        heat_np = []
        for i in range(energy.shape[0]):
            hmap = heat[i, 0]
            hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-6)
            heat_np.append(hmap.detach().cpu().numpy())
        return heat_np

    def _log_val_latent_visuals(self):
        """Log latent RGB projections and sparse-code heatmaps on cached val batch."""
        if not self.enable_val_latent_visuals:
            return
        if not getattr(self._trainer_ref(), "is_global_zero", False):
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
            with self._bottleneck_autocast_context(z):
                z_dl, _, sparse_codes = self.bottleneck(z.float())
            latent_rgb = self._latent_rgb_projection(z_dl)
            heatmaps = self._sparse_heatmaps(sparse_codes, (x.shape[2], x.shape[3]))
        log_payload = {}
        import matplotlib.pyplot as plt
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
        if self.enable_val_latent_visuals:
            self._log_val_latent_visuals()
            self._ddp_barrier_if_needed()

    def on_test_start(self):
        super().on_test_start()
        if self.test_fid is not None:
            # Align metric buffers with the rank's device and clear any stale states
            self.test_fid = self.test_fid.to(self.device)
            self.test_fid.reset()
