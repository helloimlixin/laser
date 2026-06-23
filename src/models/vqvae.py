import os
import tempfile
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import torchvision

from .decoder import Decoder
from .encoder import Encoder
from .audio_codec import AudioDecoder, AudioEncoder, canonical_int_tuple
from .bottleneck import VectorQuantizerEMA
from .lpips import LPIPS
from .utils import fid_has_enough_samples

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance

from src.audio_logging import (
    build_audio_log_payload,
    compute_audio_energy_matching_loss,
    compute_audio_reconstruction_metrics,
    compute_waveform_multires_stft_loss,
    extract_audio_metadata_from_batch,
    has_audio_metadata,
)
from src.codebook_visuals import (
    render_codebook_scatter,
    save_codebook_trajectory_gif,
    select_codebook_vectors,
)
from src.wandb_media import log_wandb_images, log_wandb_payload, log_wandb_video


class VQVAE(pl.LightningModule):
    def __init__(
            self,
            in_channels,
            num_hiddens,
            num_embeddings,
            embedding_dim,
            num_residual_blocks,
            num_residual_hiddens,
            commitment_cost,
            decay,
            perceptual_weight,
            learning_rate,
            beta,
            compute_fid=False,
            fid_feature=2048,
            audio_energy_loss_weight=0.0,
            audio_backbone="spectrogram",
            audio_downsample_rates=(4, 4, 4),
            audio_dilation_cycle=(1, 3, 9),
            audio_multires_stft_loss_weight=0.0,
            audio_multires_stft_fft_sizes=(512, 1024, 2048),
            audio_waveform_l1_weight=0.0,
            codebook_init=False,
            dead_code_threshold=0.0,
            num_downsamples=2,
            resolution=None,
            channel_multipliers=None,
            backbone_latent_channels=None,
            attn_resolutions=(),
            dropout=0.0,
            use_mid_attention=True,
            decoder_extra_residual_layers=1,
            enable_codebook_visuals=False,
            codebook_visual_max_vectors=1024,
            speaker_conditioning=False,
            speaker_conditioning_num_speakers=0,
            speaker_embedding_dim=64,
            speaker_conversion_log=True,
    ):
        """Initialize VQVAE model.

        Args:
            in_channels: Number of input channels, 3 for RGB images
            num_hiddens: number of hidden units (hidden dimensions)
            num_embeddings: Number of embeddings in codebook
            embedding_dim: Dimension of each embedding
            num_residual_blocks: Number of residual blocks in encoder and decoder
            commitment_cost: Commitment cost for VQ
            decay: Decay factor for EMA
            perceptual_weight: Weight for perceptual loss
            learning_rate: Learning rate for optimization
            beta: Beta parameter for optimizer
            compute_fid: Whether to compute FID metric
            fid_feature: Inception feature size for reconstruction FID
            enable_codebook_visuals: whether to log VQ codebook PCA scatter/GIF visualizations
            codebook_visual_max_vectors: max codebook entries to draw in PCA visualizations
        """
        super().__init__()

        # Store model parameters
        self.learning_rate = learning_rate
        self.beta = beta
        self.decay = decay
        self.perceptual_weight = perceptual_weight
        self.audio_energy_loss_weight = float(audio_energy_loss_weight)
        self.audio_backbone = str(audio_backbone or "spectrogram").strip().lower()
        if self.audio_backbone in {"2d", "image", "stft", "logmag"}:
            self.audio_backbone = "spectrogram"
        if self.audio_backbone in {"raw", "wav"}:
            self.audio_backbone = "waveform"
        if self.audio_backbone not in {"spectrogram", "waveform"}:
            raise ValueError(
                f"Unsupported VQVAE audio_backbone {audio_backbone!r}; expected 'spectrogram' or 'waveform'"
            )
        self.audio_downsample_rates = canonical_int_tuple(audio_downsample_rates, default=(4, 4, 4))
        self.audio_dilation_cycle = canonical_int_tuple(audio_dilation_cycle, default=(1, 3, 9))
        self.audio_multires_stft_loss_weight = float(audio_multires_stft_loss_weight)
        self.audio_multires_stft_fft_sizes = canonical_int_tuple(
            audio_multires_stft_fft_sizes,
            default=(512, 1024, 2048),
        )
        self.audio_waveform_l1_weight = float(audio_waveform_l1_weight)
        self.codebook_init = bool(codebook_init)
        self.dead_code_threshold = float(dead_code_threshold)
        self.num_downsamples = int(num_downsamples)
        # Attention U-Net (DDPM-style) backbone params, shared with LASER.
        self.resolution = None if resolution is None else int(resolution)
        self.channel_multipliers = (
            None if channel_multipliers is None else tuple(int(m) for m in channel_multipliers)
        )
        self.backbone_latent_channels = (
            None if backbone_latent_channels is None else int(backbone_latent_channels)
        )
        self.attn_resolutions = tuple(int(a) for a in attn_resolutions) if attn_resolutions else ()
        self.dropout = float(dropout)
        self.use_mid_attention = bool(use_mid_attention)
        self.decoder_extra_residual_layers = int(decoder_extra_residual_layers)
        self.enable_codebook_visuals = bool(enable_codebook_visuals)
        self.codebook_visual_max_vectors = max(1, int(codebook_visual_max_vectors))
        self.speaker_conditioning = bool(speaker_conditioning)
        self.speaker_conditioning_num_speakers = max(0, int(speaker_conditioning_num_speakers or 0))
        self.speaker_embedding_dim = max(1, int(speaker_embedding_dim or 1))
        self.speaker_conversion_log = bool(speaker_conversion_log)
        self.compute_fid = compute_fid
        self.fid_feature = int(fid_feature)
        self.in_channels = int(in_channels)
        self._viz_train = None
        self._viz_val = None
        self._viz_test = None
        self._codebook_snapshots = []
        self._codebook_snapshot_steps = []

        self.is_waveform_audio = self.audio_backbone == "waveform" and int(in_channels) == 1
        self.speaker_conditioning_enabled = bool(
            self.is_waveform_audio
            and self.speaker_conditioning
            and self.speaker_conditioning_num_speakers > 0
        )

        enc_dec_kwargs = None
        if not self.is_waveform_audio:
            # Attention U-Net backbone (same as LASER); waveform audio uses AudioEncoder.
            if self.resolution is None or self.resolution <= 0:
                raise ValueError("resolution must be a positive integer for the U-Net backbone")
            if self.channel_multipliers is None:
                raise ValueError("channel_multipliers must be set explicitly")
            self.num_downsamples = len(self.channel_multipliers) - 1
            if self.backbone_latent_channels is None:
                self.backbone_latent_channels = int(embedding_dim)
            enc_dec_kwargs = dict(
                ch=num_hiddens,
                out_ch=in_channels,
                ch_mult=self.channel_multipliers,
                num_res_blocks=num_residual_blocks,
                attn_resolutions=self.attn_resolutions,
                dropout=self.dropout,
                resamp_with_conv=True,
                in_channels=in_channels,
                resolution=self.resolution,
                z_channels=self.backbone_latent_channels,
                double_z=False,
                use_mid_attention=self.use_mid_attention,
            )

        # Initialize model components
        if self.is_waveform_audio:
            self.encoder = AudioEncoder(
                in_channels=in_channels,
                num_hiddens=num_hiddens,
                num_residual_layers=num_residual_blocks,
                num_residual_hiddens=num_residual_hiddens,
                downsample_rates=self.audio_downsample_rates,
                dilation_cycle=self.audio_dilation_cycle,
            )
            self.pre_bottleneck = nn.Conv1d(
                in_channels=num_hiddens,
                out_channels=embedding_dim,
                kernel_size=1,
                stride=1,
            )
        else:
            self.encoder = Encoder(**enc_dec_kwargs)
            if self.backbone_latent_channels == int(embedding_dim):
                self.pre_bottleneck = nn.Identity()
            else:
                self.pre_bottleneck = nn.Conv2d(
                    in_channels=self.backbone_latent_channels,
                    out_channels=embedding_dim,
                    kernel_size=1,
                    stride=1,
                )
        
        self.vector_quantizer = VectorQuantizerEMA(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            ema_decay=decay,
            codebook_init=self.codebook_init,
            dead_code_threshold=self.dead_code_threshold,
        )

        if self.is_waveform_audio:
            self.post_bottleneck = nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_hiddens,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            if self.speaker_conditioning_enabled:
                self.speaker_embedding = nn.Embedding(
                    self.speaker_conditioning_num_speakers,
                    self.speaker_embedding_dim,
                )
                self.speaker_to_decoder = nn.Linear(self.speaker_embedding_dim, num_hiddens)
            else:
                self.speaker_embedding = None
                self.speaker_to_decoder = None
            self.decoder = AudioDecoder(
                in_channels=num_hiddens,
                num_hiddens=num_hiddens,
                num_residual_layers=num_residual_blocks,
                num_residual_hiddens=num_residual_hiddens,
                out_channels=in_channels,
                upsample_rates=self.audio_downsample_rates,
                dilation_cycle=self.audio_dilation_cycle,
            )
        else:
            if self.backbone_latent_channels == int(embedding_dim):
                self.post_bottleneck = nn.Identity()
            else:
                self.post_bottleneck = nn.Conv2d(
                    in_channels=embedding_dim,
                    out_channels=self.backbone_latent_channels,
                    kernel_size=1,
                    stride=1,
                )
            self.speaker_embedding = None
            self.speaker_to_decoder = None
            self.decoder = Decoder(
                **dict(enc_dec_kwargs, extra_res_blocks=self.decoder_extra_residual_layers)
            )

        # Initialize LPIPS only if used
        self.lpips = LPIPS() if self.perceptual_weight > 0 else None

        if self.compute_fid:
            self.val_rfid = FrechetInceptionDistance(feature=self.fid_feature, normalize=True)
            self.test_fid = FrechetInceptionDistance(feature=self.fid_feature, normalize=True)
        else:
            self.val_rfid = None
            self.test_fid = None
        for metric in (self.val_rfid, self.test_fid):
            if metric is None:
                continue
            metric.eval()
            for p in metric.parameters():
                p.requires_grad = False

        # Separate metric instances per split avoid state leakage and let us
        # control what shows up in the progress bar.
        self.train_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.test_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.val_ssim = StructuralSimilarityIndexMeasure()
        self.test_ssim = StructuralSimilarityIndexMeasure()

        # Save hyperparameters for logging
        self.save_hyperparameters()

    def _to_metric_rgb(self, x: torch.Tensor) -> torch.Tensor:
        if int(x.size(1)) == 3:
            return x
        if int(x.size(1)) == 1:
            return x.repeat(1, 3, 1, 1)
        raise ValueError(
            f"VQVAE metrics/logging expect 1 or 3 channels, got tensor with shape {tuple(x.shape)}"
        )

    def _trainer_ref(self):
        return self.__dict__.get("_trainer", None)

    def _is_log_rank_zero(self) -> bool:
        trainer = self._trainer_ref()
        if trainer is not None:
            return bool(getattr(trainer, "is_global_zero", True))
        return int(getattr(self, "global_rank", 0)) == 0

    def _wandb_epoch_end_step(self) -> int:
        return int(getattr(self, "global_step", 0) or 0) + 1

    def _ddp_barrier_if_needed(self):
        trainer = self._trainer_ref()
        if trainer is None or getattr(trainer, "world_size", 1) <= 1:
            return
        strategy = getattr(trainer, "strategy", None)
        barrier = getattr(strategy, "barrier", None)
        if callable(barrier):
            barrier()

    def _quantize(self, z: torch.Tensor):
        if z.ndim == 3:
            z_q, loss, perplexity, encodings = self.vector_quantizer(z.unsqueeze(2))
            return z_q.squeeze(2), loss, perplexity, encodings
        return self.vector_quantizer(z)

    def _speaker_indices_from_audio_meta(
        self,
        audio_meta,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if not self.speaker_conditioning_enabled:
            return None
        raw = None
        if isinstance(audio_meta, Mapping):
            raw = audio_meta.get("speaker_index", None)
        if raw is None:
            return torch.zeros(int(batch_size), dtype=torch.long, device=device)
        if torch.is_tensor(raw):
            indices = raw.to(device=device, dtype=torch.long).view(-1)
        else:
            indices = torch.as_tensor(raw, dtype=torch.long, device=device).view(-1)
        if int(indices.numel()) < int(batch_size):
            pad = torch.zeros(int(batch_size) - int(indices.numel()), dtype=torch.long, device=device)
            indices = torch.cat([indices, pad], dim=0)
        indices = indices[: int(batch_size)]
        return indices.clamp_(0, self.speaker_conditioning_num_speakers - 1)

    def _condition_decoder_input(
        self,
        z: torch.Tensor,
        speaker_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.speaker_conditioning_enabled or z.ndim != 3:
            return z
        if speaker_indices is None:
            speaker_indices = torch.zeros(int(z.size(0)), dtype=torch.long, device=z.device)
        else:
            speaker_indices = speaker_indices.to(device=z.device, dtype=torch.long).view(-1)
            if int(speaker_indices.numel()) < int(z.size(0)):
                pad = torch.zeros(int(z.size(0)) - int(speaker_indices.numel()), dtype=torch.long, device=z.device)
                speaker_indices = torch.cat([speaker_indices, pad], dim=0)
            speaker_indices = speaker_indices[: int(z.size(0))]
        speaker_indices = speaker_indices.clamp(0, self.speaker_conditioning_num_speakers - 1)
        speaker_bias = self.speaker_to_decoder(self.speaker_embedding(speaker_indices))
        return z + speaker_bias.to(dtype=z.dtype).unsqueeze(-1)

    def _encode_quantized(self, x):
        z = self.pre_bottleneck(self.encoder(x))
        z_q, loss, perplexity, encodings = self._quantize(z)
        return z, z_q, loss, perplexity, encodings

    def encode_to_indices(self, x):
        """
        Encode images to codebook indices sequence. Returns:
          indices: LongTensor [B, H_z*W_z]
          H_z, W_z: latent spatial dims
        """
        with torch.no_grad():
            z, _, _, _, encodings = self._encode_quantized(x)
            if z.ndim == 3:
                B, _, W_z = z.shape
                H_z = 1
            else:
                B, _, H_z, W_z = z.shape
            indices = torch.argmax(encodings, dim=1).view(B, H_z * W_z)
        return indices, H_z, W_z

    def decode_from_indices(self, indices, H_z, W_z, speaker_indices=None):
        """
        Decode codebook indices back to images using the decoder.
        Args:
          indices: LongTensor [B, H_z*W_z]
          H_z, W_z: latent spatial dims
        Returns:
          recon: FloatTensor [B, C, H, W]
        """
        B = indices.size(0)
        # Gather codebook vectors and reshape to latent feature map
        codebook = self.vector_quantizer.embedding.weight  # [K, D]
        z_q_flat = codebook[indices.view(-1)]              # [B*H_z*W_z, D]
        if self.is_waveform_audio:
            z_q = z_q_flat.view(B, W_z, -1).permute(0, 2, 1).contiguous()
        else:
            z_q = z_q_flat.view(B, H_z, W_z, -1).permute(0, 3, 1, 2).contiguous()  # [B, D, H_z, W_z]
        z_q = self.post_bottleneck(z_q)
        z_q = self._condition_decoder_input(z_q, speaker_indices)
        recon = self.decoder(z_q)
        return recon

    def encode(self, x):
        """
        Encode input to latent representation
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            z_q: Quantized latent representation
            indices: Indices of the codebook entries
        """
        _, z_q, quantization_loss, _, _ = self._encode_quantized(x)
        return z_q, quantization_loss
    
    def decode(self, z_q, speaker_indices=None):
        """
        Decode latent representation to reconstruction
        
        Args:
            z_q: Quantized latent representation
        
        Returns:
            x_recon: Reconstructed input
        """
        z_q = self.post_bottleneck(z_q)
        z_q = self._condition_decoder_input(z_q, speaker_indices)
        return self.decoder(z_q)

    def forward(self, x, speaker_indices=None):
        _, z_q, vq_loss, perplexity, _ = self._encode_quantized(x)
        recon = self.decode(z_q, speaker_indices=speaker_indices)
        if recon.shape[-1] != x.shape[-1] and recon.ndim == 3 and x.ndim == 3:
            if recon.shape[-1] > x.shape[-1]:
                recon = recon[..., : x.shape[-1]]
            else:
                recon = F.pad(recon, (0, x.shape[-1] - recon.shape[-1]))

        # Return as tuple instead of dict
        return recon, vq_loss, perplexity

    def compute_metrics(self, batch, prefix='train'):
        """Compute all metrics for training, validation, and test steps.

        Args:
            batch: Input batch of data
            prefix: Metric prefix ('train', 'val', or 'test')

        Returns:
            dict: Dictionary containing all computed metrics, losses, and reconstructed images
        """
        # Unpack the batch - ensure we get a single tensor
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        batch_size = int(x.size(0))
        audio_meta = extract_audio_metadata_from_batch(batch)
        is_audio = has_audio_metadata(audio_meta)
        is_waveform_audio = is_audio and x.ndim == 3
        speaker_indices = self._speaker_indices_from_audio_meta(
            audio_meta,
            batch_size=batch_size,
            device=x.device,
        )
        
        # Forward pass
        recon_raw, vq_loss, perplexity = self(x, speaker_indices=speaker_indices)
        # Keep raw for loss; sanitized copies for metrics/visualization
        recon_vis = torch.nan_to_num(recon_raw.detach(), nan=0.0, posinf=1.0, neginf=-1.0).clamp_(-1.0, 1.0)
        x_vis = torch.nan_to_num(x.detach(), nan=0.0, posinf=1.0, neginf=-1.0).clamp_(-1.0, 1.0)

        # Compute reconstruction loss (ensure it's a scalar)
        recon_loss = F.mse_loss(recon_raw, x).mean()
        waveform_l1_loss = torch.zeros((), device=x.device, dtype=recon_raw.dtype)
        weighted_waveform_l1_loss = torch.zeros((), device=x.device, dtype=recon_raw.dtype)
        if is_waveform_audio and self.audio_waveform_l1_weight > 0:
            waveform_l1_loss = F.l1_loss(recon_raw, x)
            weighted_waveform_l1_loss = self.audio_waveform_l1_weight * waveform_l1_loss
            recon_loss = recon_loss + weighted_waveform_l1_loss

        # Compute perceptual loss only if enabled
        if (not is_waveform_audio) and self.perceptual_weight > 0 and self.lpips is not None:
            x_norm = self._to_metric_rgb(x * 2.0 - 1.0)
            x_recon_norm = self._to_metric_rgb(recon_raw * 2.0 - 1.0)
            perceptual_loss = self.lpips(x_recon_norm, x_norm).mean()
        else:
            perceptual_loss = torch.zeros((), device=x.device, dtype=recon_raw.dtype)

        # Compute total loss
        total_loss = (1 - self.perceptual_weight) * recon_loss + vq_loss + self.perceptual_weight * perceptual_loss

        # Add PSNR calculation on de-normalized [0,1] images for stable progression
        dm = getattr(self._trainer_ref(), "datamodule", None)
        if is_waveform_audio:
            x_dn = None
            recon_dn = None
        elif dm is not None and hasattr(dm, "config") and hasattr(dm.config, "mean") and hasattr(dm.config, "std"):
            mean = torch.tensor(dm.config.mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
            std = torch.tensor(dm.config.std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
            x_dn = (x * std + mean).clamp(0.0, 1.0)
            recon_dn = (recon_raw * std + mean).clamp(0.0, 1.0)
        else:
            x_dn = ((x + 1.0) / 2.0).clamp(0.0, 1.0)
            recon_dn = ((recon_raw + 1.0) / 2.0).clamp(0.0, 1.0)

        if not is_audio and prefix == 'val' and self.val_rfid is not None:
            self.val_rfid.update(self._to_metric_rgb(x_dn), real=True)
            self.val_rfid.update(self._to_metric_rgb(recon_dn), real=False)
        elif not is_audio and prefix == 'test' and self.test_fid is not None:
            self.test_fid.update(self._to_metric_rgb(x_dn), real=True)
            self.test_fid.update(self._to_metric_rgb(recon_dn), real=False)

        psnr = None
        ssim = None
        audio_metrics = {}
        audio_energy_metrics = {}
        audio_stft_metrics = {}
        audio_energy_loss = torch.zeros((), device=x.device, dtype=recon_raw.dtype)
        weighted_audio_energy_loss = torch.zeros((), device=x.device, dtype=recon_raw.dtype)
        audio_stft_loss = torch.zeros((), device=x.device, dtype=recon_raw.dtype)
        weighted_audio_stft_loss = torch.zeros((), device=x.device, dtype=recon_raw.dtype)
        if is_audio:
            audio_source = getattr(dm, "config", {"dataset": "vctk"})
            if is_waveform_audio and self.audio_multires_stft_loss_weight > 0:
                audio_stft_metrics = compute_waveform_multires_stft_loss(
                    x,
                    recon_raw,
                    fft_sizes=self.audio_multires_stft_fft_sizes,
                )
                audio_stft_loss = audio_stft_metrics.get(
                    "audio_multires_stft_loss",
                    torch.zeros((), device=x.device, dtype=recon_raw.dtype),
                )
                weighted_audio_stft_loss = self.audio_multires_stft_loss_weight * audio_stft_loss
                total_loss = total_loss + weighted_audio_stft_loss
            audio_metrics = compute_audio_reconstruction_metrics(
                x,
                recon_raw,
                audio_meta=audio_meta,
                audio_source=audio_source,
                compute_visqol=prefix != "train",
            )
            if (not is_waveform_audio) and self.audio_energy_loss_weight > 0:
                audio_energy_metrics = compute_audio_energy_matching_loss(
                    x,
                    recon_raw,
                    audio_meta=audio_meta,
                    audio_source=audio_source,
                )
                audio_energy_loss = audio_energy_metrics.get(
                    "audio_energy_loss",
                    torch.zeros((), device=x.device, dtype=recon_raw.dtype),
                )
        else:
            if prefix == 'train':
                psnr = self.train_psnr(recon_dn, x_dn)
            elif prefix == 'val':
                psnr = self.val_psnr(recon_dn, x_dn)
                ssim = self.val_ssim(recon_dn, x_dn)
            else:
                psnr = self.test_psnr(recon_dn, x_dn)
                ssim = self.test_ssim(recon_dn, x_dn)

        if is_audio and (not is_waveform_audio) and self.audio_energy_loss_weight > 0:
            weighted_audio_energy_loss = self.audio_energy_loss_weight * audio_energy_loss
            total_loss = total_loss + weighted_audio_energy_loss

        # Always synchronize epoch metrics so DDP runs do not emit reduction warnings.
        log_kwargs = dict(
            on_step=prefix == 'train',
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(f'{prefix}/loss', total_loss, prog_bar=True, **log_kwargs)
        self.log(f'{prefix}/recon_loss', recon_loss, **log_kwargs)
        if is_waveform_audio and self.audio_waveform_l1_weight > 0:
            self.log(f'{prefix}/audio_waveform_l1_loss', waveform_l1_loss, **log_kwargs)
            self.log(f'{prefix}/weighted_audio_waveform_l1_loss', weighted_waveform_l1_loss, **log_kwargs)
        self.log(f'{prefix}/vq_loss', vq_loss, **log_kwargs)
        self.log(f'{prefix}/perceptual_loss', perceptual_loss, **log_kwargs)
        self.log(f'{prefix}/perplexity', perplexity, prog_bar=True, **log_kwargs)
        if is_audio:
            for name, value in audio_metrics.items():
                self.log(f'{prefix}/{name}', value, prog_bar=(name == "audio_lsd"), **log_kwargs)
            for name, value in audio_stft_metrics.items():
                self.log(f'{prefix}/{name}', value, prog_bar=False, **log_kwargs)
            if is_waveform_audio and self.audio_multires_stft_loss_weight > 0:
                self.log(f'{prefix}/weighted_audio_multires_stft_loss', weighted_audio_stft_loss, **log_kwargs)
        if is_audio and (not is_waveform_audio) and self.audio_energy_loss_weight > 0:
            self.log(f'{prefix}/weighted_audio_energy_loss', weighted_audio_energy_loss, **log_kwargs)
            for name, value in audio_energy_metrics.items():
                self.log(f'{prefix}/{name}', value, prog_bar=False, **log_kwargs)
        elif not is_audio:
            self.log(f'{prefix}/psnr', psnr, prog_bar=True, **log_kwargs)
            if ssim is not None:
                self.log(f'{prefix}/ssim', ssim, prog_bar=True, **log_kwargs)

        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'vq_loss': vq_loss,
            'perceptual_loss': perceptual_loss,
            'audio_energy_loss': audio_energy_loss,
            'audio_multires_stft_loss': audio_stft_loss,
            'perplexity': perplexity,
            'psnr': psnr,
            'ssim': ssim,
            'audio_metrics': audio_metrics,
            'x': x_vis,
            'x_recon': recon_vis,
            'audio_meta': audio_meta,
        }

    def training_step(self, batch, batch_idx):
        """Perform the training step."""
        metrics = self.compute_metrics(batch, prefix='train')
        self._viz_train = {
            "x": metrics['x'],
            "x_recon": metrics['x_recon'],
            "audio_meta": metrics.get("audio_meta"),
        }
        return metrics

    def on_train_epoch_end(self):
        if self._viz_train is not None:
            self._log_images(
                self._viz_train["x"],
                self._viz_train["x_recon"],
                split='train',
                audio_meta=self._viz_train.get("audio_meta"),
            )
            self._ddp_barrier_if_needed()
            self._viz_train = None

    def validation_step(self, batch, batch_idx):
        """Perform the validation step."""
        metrics = self.compute_metrics(batch, prefix='val')
        self._viz_val = {
            "x": metrics['x'],
            "x_recon": metrics['x_recon'],
            "audio_meta": metrics.get("audio_meta"),
        }
        return metrics

    def on_validation_epoch_start(self):
        if self.val_rfid is not None:
            self.val_rfid = self.val_rfid.to(self.device)
            self.val_rfid.reset()

    def on_validation_epoch_end(self):
        if self.val_rfid is not None:
            # Skip compute() if no updates landed this epoch — audio runs with
            # compute_fid=true allocate the metric but never call update().
            if fid_has_enough_samples(self.val_rfid):
                rfid_score = self.val_rfid.compute()
                self.log('val/rfid', rfid_score, sync_dist=True)
            self.val_rfid.reset()
        if self._viz_val is not None:
            self._log_images(
                self._viz_val["x"],
                self._viz_val["x_recon"],
                split='val',
                audio_meta=self._viz_val.get("audio_meta"),
            )
            self._ddp_barrier_if_needed()
            self._viz_val = None
        if self.enable_codebook_visuals:
            self._snapshot_codebook()
            self._log_codebook_scatter()
            self._ddp_barrier_if_needed()

    def on_test_epoch_start(self):
        """Reset test-only metrics before aggregating over the full test set."""
        self._viz_test = None
        if self.val_rfid is not None:
            self.val_rfid.reset()
        if self.test_fid is not None:
            self.test_fid = self.test_fid.to(self.device)
            self.test_fid.reset()

    def test_step(self, batch, batch_idx):
        """Perform the test step."""
        metrics = self.compute_metrics(batch, prefix='test')
        self._viz_test = {
            "x": metrics['x'],
            "x_recon": metrics['x_recon'],
            "audio_meta": metrics.get("audio_meta"),
        }
        return metrics

    def on_test_epoch_end(self):
        """Log reconstructions and compute FID after the full test set."""
        if self._viz_test is not None:
            self._log_images(
                self._viz_test["x"],
                self._viz_test["x_recon"],
                split='test',
                audio_meta=self._viz_test.get("audio_meta"),
            )
            self._ddp_barrier_if_needed()
            self._viz_test = None
        if self.test_fid is not None:
            if fid_has_enough_samples(self.test_fid):
                fid_score = self.test_fid.compute()
                self.log('test/fid', fid_score, sync_dist=True)
            self.test_fid.reset()

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Store the scheduler as an attribute
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            }
        }

    def on_fit_start(self):
        super().on_fit_start()
        self._snapshot_codebook()

    def _snapshot_codebook(self):
        """Store a stable subset of VQ codebook vectors for trajectory animation."""
        if not self.enable_codebook_visuals or not self._is_log_rank_zero():
            return
        with torch.no_grad():
            vectors = self.vector_quantizer.embedding.weight.detach().cpu()
            vectors = select_codebook_vectors(vectors, self.codebook_visual_max_vectors)
        step = int(getattr(self, "global_step", 0) or 0)
        if self._codebook_snapshot_steps and self._codebook_snapshot_steps[-1] == step:
            return
        self._codebook_snapshots.append(vectors)
        self._codebook_snapshot_steps.append(step)

    def _log_codebook_scatter(self):
        if not self.enable_codebook_visuals or not self._is_log_rank_zero():
            return
        if not self._codebook_snapshots:
            return
        logger = getattr(self, "logger", None)
        if logger is None:
            return
        image = render_codebook_scatter(
            self._codebook_snapshots,
            self._codebook_snapshot_steps,
            title="VQ Codebook Vectors (PCA)",
        )
        if image is None:
            return
        step = self._wandb_epoch_end_step()
        log_wandb_images(
            logger,
            "val/vq_codebook_scatter",
            [image],
            step=step,
            captions=[f"vq codebook scatter step={step}"],
        )

    def _generate_codebook_animation(self):
        if not self.enable_codebook_visuals or not self._is_log_rank_zero():
            return
        if len(self._codebook_snapshots) < 2:
            return
        logger = getattr(self, "logger", None)
        if logger is None:
            return
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
            gif_path = tmp.name
        try:
            saved = save_codebook_trajectory_gif(
                self._codebook_snapshots,
                self._codebook_snapshot_steps,
                gif_path,
                title="VQ Codebook Vector Trajectories (PCA)",
                fps=2,
            )
            if saved is None:
                return
            step = self._wandb_epoch_end_step()
            log_wandb_video(
                logger,
                "val/vq_codebook_trajectories",
                [str(saved)],
                step=step,
                captions=[f"vq codebook trajectories step={step}"],
                formats=["gif"],
            )
        finally:
            try:
                os.unlink(gif_path)
            except OSError:
                pass

    def on_fit_end(self):
        if self.enable_codebook_visuals:
            self._snapshot_codebook()
            self._log_codebook_scatter()
            self._generate_codebook_animation()
            self._ddp_barrier_if_needed()

    def _log_images(self, x, x_recon, split='train', audio_meta=None):
        """
        Log images to Weights & Biases.

        Args:
            x (torch.Tensor): Original images
            x_recon (torch.Tensor): Reconstructed images
            split (str): Data split (train/val/test)
        """
        # Only log from rank zero in DDP to avoid multi-process logger contention
        if getattr(self._trainer_ref(), "is_global_zero", False) is False:
            return
        # Take first 16 images
        x = x[:32]
        x_recon = x_recon[:32]
        dm = getattr(self._trainer_ref(), "datamodule", None)
        logger = getattr(self, "logger", None)
        step = int(self.global_step)
        if audio_meta is not None and x.ndim == 3 and dm is not None and hasattr(dm, "config"):
            payload = {
                f"{split}/reconstruction_error": F.mse_loss(x_recon, x).item(),
            }
            artifact_dir = getattr(self.logger, "save_dir", None) or getattr(self._trainer_ref(), "default_root_dir", None)
            payload.update(
                build_audio_log_payload(
                    x,
                    x_recon,
                    audio_meta=audio_meta,
                    audio_source=dm.config,
                    split=split,
                    max_items=4,
                    artifact_dir=artifact_dir,
                )
            )
            if (
                self.speaker_conditioning_enabled
                and self.speaker_conversion_log
                and split != "train"
                and self.speaker_conditioning_num_speakers > 1
            ):
                with torch.no_grad():
                    source_speakers = self._speaker_indices_from_audio_meta(
                        audio_meta,
                        batch_size=int(x.size(0)),
                        device=x.device,
                    )
                    target_speakers = (source_speakers + 1) % self.speaker_conditioning_num_speakers
                    _, z_q, _, _, _ = self._encode_quantized(x)
                    converted = self.decode(z_q, speaker_indices=target_speakers)
                    if converted.shape[-1] != x.shape[-1]:
                        if converted.shape[-1] > x.shape[-1]:
                            converted = converted[..., : x.shape[-1]]
                        else:
                            converted = F.pad(converted, (0, x.shape[-1] - converted.shape[-1]))
                    converted = torch.nan_to_num(converted.detach(), nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
                payload.update(
                    build_audio_log_payload(
                        x,
                        converted,
                        audio_meta=audio_meta,
                        audio_source=dm.config,
                        split=f"{split}/speaker_conversion",
                        max_items=4,
                        artifact_dir=artifact_dir,
                    )
                )
            log_wandb_payload(logger, payload, step=step)
            return

        # Create grids with smaller size
        # De-normalize using datamodule config if available; otherwise assume [-1,1] → [0,1]
        if dm is not None and hasattr(dm, "config") and hasattr(dm.config, "mean") and hasattr(dm.config, "std"):
            mean = torch.tensor(dm.config.mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
            std = torch.tensor(dm.config.std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
            x_disp = x * std + mean
            x_recon_disp = x_recon * std + mean
        else:
            x_disp = (x + 1.0) / 2.0
            x_recon_disp = (x_recon + 1.0) / 2.0
        x_disp = x_disp.clamp(0.0, 1.0)
        x_recon_disp = x_recon_disp.clamp(0.0, 1.0)
        x_disp = self._to_metric_rgb(x_disp)
        x_recon_disp = self._to_metric_rgb(x_recon_disp)
        x_grid = torchvision.utils.make_grid(x_disp, nrow=8, normalize=False)
        x_recon_grid = torchvision.utils.make_grid(x_recon_disp, nrow=8, normalize=False)

        # Sanitize NaN/Inf and clamp to [0,1] before converting to numpy
        x_grid = torch.nan_to_num(x_grid, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        x_recon_grid = torch.nan_to_num(x_recon_grid, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)

        # Convert to numpy and transpose to correct format (H,W,C)
        x_grid = x_grid.cpu().numpy().transpose(1, 2, 0)
        x_recon_grid = x_recon_grid.cpu().numpy().transpose(1, 2, 0)

        # Log to wandb using the experiment attribute
        log_wandb_images(
            logger,
            f"{split}/images",
            [x_grid, x_recon_grid],
            step=step,
            captions=["Original", "Reconstructed"],
        )
        payload = {
            f"{split}/reconstruction_error": F.mse_loss(x_recon, x).item(),
        }
        if audio_meta is not None and dm is not None and hasattr(dm, "config"):
            payload.update(
                build_audio_log_payload(
                    x,
                    x_recon,
                    audio_meta=audio_meta,
                    audio_source=dm.config,
                    split=split,
                    max_items=4,
                    artifact_dir=getattr(self.logger, "save_dir", None) or getattr(self._trainer_ref(), "default_root_dir", None),
                )
            )
        log_wandb_payload(logger, payload, step=step)


# test the VQVAE model
if __name__ == "__main__":
    vqvae = VQVAE(in_channels=3, num_hiddens=128, num_residual_blocks=2, num_residual_hiddens=32, num_embeddings=1024, embedding_dim=32, commitment_cost=0.25, decay=0.99, perceptual_weight=0.1, learning_rate=1e-4, beta=1.0, compute_fid=True)
    x = torch.randn(4, 3, 256, 256)  # batch_size x 3 x 256 x 256
    print(vqvae(x)[0].shape)  # batch_size x 3 x 256 x 256
