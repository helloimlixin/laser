import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import torchvision

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
            dict_learning_rate=0.1,
            sparse_solver='omp',
            iht_iterations=10,
            iht_step_size=None,
            sparsity_reg_weight=0.01,
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
            sparse_solver: sparse coding algorithm ('omp', 'iht', 'topk')
            iht_iterations: number of IHT iterations (if using IHT)
            iht_step_size: step size for IHT (None = auto-compute)
            sparsity_reg_weight: L1 regularization weight on sparse coefficients
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
            ksvd_iterations=ksvd_iterations,
            patch_size=patch_size,
            use_online_learning=use_online_learning,
            use_backprop_only=use_backprop_only,
            dict_learning_rate=dict_learning_rate,
            sparse_solver=sparse_solver,
            iht_iterations=iht_iterations,
            iht_step_size=iht_step_size,
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

        # Initialize the PSNR metric (evaluate on de-normalized [0,1] images)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)

        # Initialize the SSIM metric
        self.ssim = StructuralSimilarityIndexMeasure()

        # Initialize metrics for Frechet Inception Distance (FID Score)
        if self.compute_fid:
            self.test_fid = FrechetInceptionDistance(feature=64, normalize=True)
        else:
            self.test_fid = None

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
            tuple: (recon, bottleneck_loss, coefficients)
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
        
        # Forward pass
        recon_raw, bottleneck_loss, coefficients = self(x)
        # Keep raw tensors for loss; create sanitized copies for metrics/visualization only
        recon_vis = torch.nan_to_num(recon_raw.detach(), nan=0.0, posinf=1.0, neginf=-1.0).clamp_(-1.0, 1.0)
        x_vis = torch.nan_to_num(x.detach(), nan=0.0, posinf=1.0, neginf=-1.0).clamp_(-1.0, 1.0)
        
        # Compute losses
        recon_loss = F.mse_loss(recon_raw, x)
        
        # L1 regularization on sparse coefficients to encourage true sparsity
        sparsity_loss = torch.abs(coefficients).mean()
        
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
        
        # Total loss
        total_loss = (
            recon_loss +
            bottleneck_loss +
            self.perceptual_weight * perceptual_loss +
            self.mr_dct_weight * mr_dct_loss +
            self.mr_grad_weight * mr_grad_loss +
            self.sparsity_reg_weight * sparsity_loss  # L1 penalty on coefficients
        )
        
        # Compute metrics on clamped/sanitized tensors
        # Denormalize from [-1, 1] to [0, 1]
        x_01 = (x_vis + 1.0) / 2.0
        recon_01 = (recon_vis + 1.0) / 2.0
        
        psnr = self.psnr(recon_01, x_01)
        # Only compute SSIM during validation/testing for speed
        if prefix != 'train':
            ssim = self.ssim(recon_01, x_01)
        else:
            ssim = torch.tensor(0.0, device=x_01.device)
        
        # Compute sparsity
        sparsity = (coefficients.abs() > 1e-6).float().mean()
        
        # Log metrics; explicitly set on_step/on_epoch to avoid Lightning warnings in DDP
        log_kwargs = dict(on_step=prefix == 'train', on_epoch=True, sync_dist=True)
        self.log(f'{prefix}/loss', total_loss, prog_bar=True, **log_kwargs)
        self.log(f'{prefix}/recon_loss', recon_loss, **log_kwargs)
        self.log(f'{prefix}/bottleneck_loss', bottleneck_loss, **log_kwargs)
        self.log(f'{prefix}/perceptual_loss', perceptual_loss, **log_kwargs)
        self.log(f'{prefix}/sparsity_loss', sparsity_loss, **log_kwargs)
        self.log(f'{prefix}/mr_dct_loss', mr_dct_loss, **log_kwargs)
        self.log(f'{prefix}/mr_grad_loss', mr_grad_loss, **log_kwargs)
        self.log(f'{prefix}/psnr', psnr, prog_bar=True, **log_kwargs)
        if prefix != 'train':
            self.log(f'{prefix}/ssim', ssim, prog_bar=True, **log_kwargs)
        self.log(f'{prefix}/sparsity', sparsity, **log_kwargs)
        
        return total_loss, recon_vis, x_vis

    def training_step(self, batch, batch_idx):
        """Training step."""
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
            # Denormalize to [0, 1]
            x_01 = (x + 1.0) / 2.0
            recon_01 = (recon + 1.0) / 2.0
            
            # Convert to uint8 [0, 255]
            x_uint8 = (x_01 * 255).clamp(0, 255).to(torch.uint8)
            recon_uint8 = (recon_01 * 255).clamp(0, 255).to(torch.uint8)
            
            self.test_fid.update(x_uint8, real=True)
            self.test_fid.update(recon_uint8, real=False)
        
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
        # Include encoder, decoder, and bottleneck parameters
        params = list(self.encoder.parameters()) + \
                 list(self.pre_bottleneck.parameters()) + \
                 list(self.post_bottleneck.parameters()) + \
                 list(self.decoder.parameters())
        
        # Include dictionary parameters if using backprop-only mode
        if self.bottleneck.use_backprop_only:
            params += list(self.bottleneck.parameters())

        # Add weight decay for regularization to prevent overfitting
        optimizer = torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            betas=(self.beta, 0.999),
            weight_decay=1e-4  # L2 regularization
        )

        return optimizer
