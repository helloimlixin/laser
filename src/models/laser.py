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
            coef_max=None,
            bounded_omp_refine_steps=8,
            sparsity_reg_weight=0.01,
            coherence_weight=0.0,
            warmup_steps=0,
            min_lr_ratio=0.01,
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
            coef_max: optional hard bound applied to sparse coefficients during support refinement
            bounded_omp_refine_steps: projected refinement steps for bounded OMP coefficient updates
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
        self.warmup_steps = max(int(warmup_steps), 0)
        self.min_lr_ratio = float(min_lr_ratio)
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
            coef_max=coef_max,
            bounded_omp_refine_steps=bounded_omp_refine_steps,
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
        # Dictionary atom snapshots for trajectory animation
        self._dict_snapshots = []
        self._dict_snapshot_steps = []

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
    def encode_to_atoms_and_coeffs(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """Encode images to sparse atom ids plus real-valued coefficients."""
        z_dl, _, sparse_codes = self.encode(x)
        return (
            sparse_codes.support,
            sparse_codes.values,
            (int(z_dl.shape[-2]), int(z_dl.shape[-1])),
        )

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

    def reconstruct_latent_from_atoms_and_coeffs(
        self,
        atom_ids: torch.Tensor,
        coeffs: torch.Tensor,
        *,
        latent_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Reconstruct a latent map directly from sparse atom ids and coefficients."""
        if atom_ids.dim() != 4 or coeffs.dim() != 4:
            raise ValueError(
                f"Expected atom_ids/coeffs with shape [B,H,W,D], got {tuple(atom_ids.shape)} and {tuple(coeffs.shape)}"
            )
        if tuple(atom_ids.shape) != tuple(coeffs.shape):
            raise ValueError(
                f"atom_ids and coeffs shape mismatch: {tuple(atom_ids.shape)} vs {tuple(coeffs.shape)}"
            )

        if self.bottleneck._is_patch_based():
            if latent_hw is None:
                raise ValueError("latent_hw is required for patch-based sparse latent reconstruction")
            height, width = int(latent_hw[0]), int(latent_hw[1])
        else:
            height, width = int(atom_ids.shape[1]), int(atom_ids.shape[2])

        coeffs = coeffs.to(device=atom_ids.device, dtype=self.bottleneck.dictionary.dtype)
        return self.bottleneck._reconstruct_sparse(
            atom_ids.to(torch.long),
            coeffs,
            height,
            width,
        )

    @torch.no_grad()
    def decode_from_atoms_and_coeffs(
        self,
        atom_ids: torch.Tensor,
        coeffs: torch.Tensor,
        *,
        latent_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Decode sparse atom ids plus real-valued coefficients back to image space."""
        z_q = self.reconstruct_latent_from_atoms_and_coeffs(
            atom_ids,
            coeffs,
            latent_hw=latent_hw,
        )
        return self.decode(z_q)

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
        
        # Keep epoch metrics synchronized across ranks whenever DDP is active.
        # Relying on a local world_size probe here has been brittle on cluster runs.
        log_kwargs = dict(on_step=prefix == 'train', on_epoch=True, sync_dist=True)
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

    def _snapshot_dictionary(self):
        """Store a copy of the current dictionary atoms for trajectory animation.

        Called once per validation epoch (not every N training steps) to keep
        the snapshot list short and the final GIF fast to render.
        """
        if not self.enable_val_latent_visuals:
            return
        if not getattr(self._trainer_ref(), "is_global_zero", False):
            return
        with torch.no_grad():
            atoms = self.bottleneck.dictionary.detach().cpu().clone()
        self._dict_snapshots.append(atoms)
        self._dict_snapshot_steps.append(self.global_step)

    def _pca_basis_and_project(self, snapshots_np):
        """Compute PCA basis from the final snapshot and project all snapshots."""
        import numpy as np
        final = snapshots_np[-1]
        mean = final.mean(axis=0, keepdims=True)
        centered = final - mean
        try:
            _, s_vals, vt = np.linalg.svd(centered, full_matrices=False)
            basis = vt[:2]
            total_var = (s_vals ** 2).sum()
            pc1_var = float(s_vals[0] ** 2 / total_var * 100) if total_var > 0 else 0.0
            pc2_var = float(s_vals[1] ** 2 / total_var * 100) if len(s_vals) > 1 and total_var > 0 else 0.0
        except np.linalg.LinAlgError:
            atom_dim = final.shape[1]
            basis = np.eye(min(2, atom_dim), atom_dim)
            pc1_var = pc2_var = 0.0
        projected = [(snap - mean) @ basis.T for snap in snapshots_np]
        return projected, mean, basis, pc1_var, pc2_var

    def _log_dict_scatter(self):
        """Log a static PCA scatter of current dictionary atoms (cheap, once per val epoch)."""
        if len(self._dict_snapshots) < 1:
            return
        if not getattr(self._trainer_ref(), "is_global_zero", False):
            return
        logger = getattr(self, "logger", None)
        experiment = getattr(logger, "experiment", None) if logger else None
        if experiment is None or not hasattr(experiment, "log"):
            return

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import tempfile, os

        snapshots_np = [s.t().numpy() for s in self._dict_snapshots]
        projected, mean, basis, pc1_var, pc2_var = self._pca_basis_and_project(snapshots_np)
        num_atoms = snapshots_np[-1].shape[0]
        atom_dim = snapshots_np[-1].shape[1]
        pts = projected[-1]

        # Color by displacement from initial position (if more than one snapshot)
        if len(projected) > 1:
            disp = np.sqrt(((pts - projected[0]) ** 2).sum(axis=1))
            disp_norm = disp / (disp.max() + 1e-8)
        else:
            disp_norm = np.zeros(num_atoms)

        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=disp_norm, cmap="plasma",
                        s=18, alpha=0.85, edgecolors="k", linewidths=0.3)
        ax.set_title(
            f"Dictionary Atoms (PCA)  |  Step {self.global_step}  |  "
            f"{num_atoms} atoms, {atom_dim}-dim",
            fontsize=10,
        )
        ax.set_xlabel(f"PC1 ({pc1_var:.1f}% var)")
        ax.set_ylabel(f"PC2 ({pc2_var:.1f}% var)")
        cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("Displacement from init", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        fig.tight_layout()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            png_path = tmp.name
        fig.savefig(png_path, dpi=120)
        plt.close(fig)

        experiment.log({
            "val/dictionary_scatter": wandb.Image(png_path),
            "global_step": self.global_step,
        })
        try:
            os.unlink(png_path)
        except OSError:
            pass

    def _generate_dict_animation(self):
        """Build the full trajectory GIF from all accumulated snapshots.

        Called once at the end of training (on_fit_end), not every val epoch.
        """
        if len(self._dict_snapshots) < 2:
            return
        if not getattr(self._trainer_ref(), "is_global_zero", False):
            return
        logger = getattr(self, "logger", None)
        experiment = getattr(logger, "experiment", None) if logger else None
        if experiment is None or not hasattr(experiment, "log"):
            return

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
        import numpy as np
        import tempfile, os

        snapshots_np = [s.t().numpy() for s in self._dict_snapshots]
        steps = list(self._dict_snapshot_steps)
        num_atoms = snapshots_np[0].shape[0]
        atom_dim = snapshots_np[0].shape[1]

        projected, mean, basis, pc1_var, pc2_var = self._pca_basis_and_project(snapshots_np)

        # Per-frame displacement from initial position
        displacements = []
        for proj in projected:
            disp = np.sqrt(((proj - projected[0]) ** 2).sum(axis=1))
            displacements.append(disp)

        # Compute global axis limits
        all_pts = np.concatenate(projected, axis=0)
        margin = 0.08
        x_range = max(all_pts[:, 0].max() - all_pts[:, 0].min(), 1e-6)
        y_range = max(all_pts[:, 1].max() - all_pts[:, 1].min(), 1e-6)
        x_lim = (all_pts[:, 0].min() - margin * x_range, all_pts[:, 0].max() + margin * x_range)
        y_lim = (all_pts[:, 1].min() - margin * y_range, all_pts[:, 1].max() + margin * y_range)

        total_disp = displacements[-1]
        disp_norm = total_disp / (total_disp.max() + 1e-8)
        cmap = plt.cm.plasma
        colors = cmap(disp_norm)

        fig, ax = plt.subplots(figsize=(9, 7))

        def update(frame_idx):
            ax.clear()
            pts = projected[frame_idx]
            disp = displacements[frame_idx]
            step = steps[frame_idx]

            for k in range(num_atoms):
                trail_x = [projected[t][k, 0] for t in range(frame_idx + 1)]
                trail_y = [projected[t][k, 1] for t in range(frame_idx + 1)]
                ax.plot(trail_x, trail_y, color=colors[k], alpha=0.25, lw=0.6)

            sc = ax.scatter(pts[:, 0], pts[:, 1], c=disp_norm, cmap="plasma",
                            s=18, alpha=0.85, edgecolors="k", linewidths=0.3)

            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_title(
                f"Dictionary Atom Trajectories (PCA projection)\n"
                f"Step {step}  |  {num_atoms} atoms, {atom_dim}-dim  |  "
                f"Frame {frame_idx + 1}/{len(projected)}",
                fontsize=10,
            )
            ax.set_xlabel(f"PC1 ({pc1_var:.1f}% var)")
            ax.set_ylabel(f"PC2 ({pc2_var:.1f}% var)")

            mean_disp = float(disp.mean())
            max_disp = float(disp.max())
            settled = int((disp < 0.01 * (total_disp.max() + 1e-8)).sum())
            stats_text = (
                f"Mean displacement: {mean_disp:.4f}\n"
                f"Max displacement:  {max_disp:.4f}\n"
                f"Settled atoms (<1% of max): {settled}/{num_atoms}"
            )
            ax.text(
                0.02, 0.02, stats_text,
                transform=ax.transAxes, fontsize=7.5,
                verticalalignment="bottom", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
            )

            if not hasattr(update, "_cbar_added"):
                cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
                cbar.set_label("Relative total displacement", fontsize=8)
                cbar.ax.tick_params(labelsize=7)
                update._cbar_added = True

        anim = FuncAnimation(fig, update, frames=len(projected), interval=500)
        fig.tight_layout()
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
            gif_path = tmp.name
        anim.save(gif_path, writer=PillowWriter(fps=2))
        plt.close(fig)

        experiment.log({
            "val/dictionary_atom_trajectories": wandb.Video(gif_path, format="gif"),
            "global_step": self.global_step,
        })
        try:
            os.unlink(gif_path)
        except OSError:
            pass

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
        """Configure optimizers with optional cosine LR schedule."""
        import math

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

        # Cosine annealing with optional warmup.
        total_steps = self.trainer.estimated_stepping_batches
        warmup = self.warmup_steps
        min_ratio = self.min_lr_ratio

        if total_steps <= 0:
            return optimizer

        def lr_lambda(step):
            if step < warmup:
                return max(min_ratio, step / max(warmup, 1))
            progress = (step - warmup) / max(total_steps - warmup, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    
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
        """Generate per-image sparse coefficient energy heatmaps.

        Uses L2 norm of the coefficient vector at each spatial location,
        upsampled with nearest-neighbor to preserve sharp patch boundaries.
        """
        h_in, w_in = image_hw
        # L2 energy per location — highlights where the sparse code is
        # working hardest to represent the signal.
        energy = sparse_codes.values.pow(2).sum(dim=-1).sqrt()  # [B, H, W]
        heat = energy.unsqueeze(1)
        heat = F.interpolate(heat, size=(h_in, w_in), mode='nearest')
        heat_np = []
        for i in range(heat.shape[0]):
            hmap = heat[i, 0]
            hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-6)
            heat_np.append(hmap.detach().cpu().numpy())
        return heat_np

    def _recon_error_heatmaps(self, x, recon, image_hw):
        """Generate per-pixel reconstruction error heatmaps."""
        h_in, w_in = image_hw
        err = (x - recon).pow(2).mean(dim=1, keepdim=True).sqrt()  # [B,1,H,W] RMS per pixel
        heat_np = []
        for i in range(err.shape[0]):
            hmap = err[i, 0]
            hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-6)
            heat_np.append(hmap.detach().cpu().numpy())
        return heat_np

    def _log_val_latent_visuals(self):
        """Log latent RGB projections, sparse heatmaps, and reconstruction diagnostics."""
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
            recon = self.decoder(self.post_bottleneck(z_dl))
            image_hw = (x.shape[2], x.shape[3])
            latent_rgb = self._latent_rgb_projection(z_dl)
            sparse_heat = self._sparse_heatmaps(sparse_codes, image_hw)
            error_heat = self._recon_error_heatmaps(x, recon, image_hw)
        log_payload = {}
        import matplotlib.pyplot as plt
        cmap = plt.cm.inferno
        x_disp = self._denormalize_for_display(x).detach().cpu()
        recon_disp = self._denormalize_for_display(recon).detach().cpu()
        for idx in range(x.shape[0]):
            orig_np = x_disp[idx].permute(1, 2, 0).clamp(0, 1).numpy()
            recon_np = recon_disp[idx].permute(1, 2, 0).clamp(0, 1).numpy()
            latent_img = latent_rgb[idx]
            sparse_rgb = cmap(sparse_heat[idx])[..., :3]
            error_rgb = cmap(error_heat[idx])[..., :3]
            cap = f"idx={idx}"
            log_payload.setdefault("val/original", []).append(wandb.Image(orig_np, caption=cap))
            log_payload.setdefault("val/reconstruction", []).append(wandb.Image(recon_np, caption=cap))
            log_payload.setdefault("val/latent_rgb", []).append(wandb.Image(latent_img, caption=cap))
            log_payload.setdefault("val/sparse_heatmap", []).append(wandb.Image(sparse_rgb, caption=cap))
            log_payload.setdefault("val/recon_error_map", []).append(wandb.Image(error_rgb, caption=cap))
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
            self._snapshot_dictionary()
            self._log_dict_scatter()
            self._ddp_barrier_if_needed()

    def on_fit_end(self):
        """Generate the full trajectory animation GIF once at the end of training."""
        if self.enable_val_latent_visuals:
            self._generate_dict_animation()
            self._ddp_barrier_if_needed()

    def on_test_start(self):
        super().on_test_start()
        if self.test_fid is not None:
            # Align metric buffers with the rank's device and clear any stale states
            self.test_fid = self.test_fid.to(self.device)
            self.test_fid.reset()
