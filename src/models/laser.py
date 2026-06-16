import ast
import os
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import torchvision
from contextlib import nullcontext
from typing import Optional, Sequence, Tuple

from .bottleneck import DictionaryLearning, SparseCodes
from .discriminator import (
    AudioMultiScalePeriodDiscriminator,
    NLayerDiscriminator,
    adopt_weight,
    feature_matching_loss,
    multi_hinge_d_loss,
    multi_hinge_g_loss,
    multi_lsgan_d_loss,
    multi_lsgan_g_loss,
    multi_vanilla_d_loss,
)
from .lpips import LPIPS
from .audio_codec import AudioDecoder, AudioEncoder, canonical_int_tuple
from .unet import Decoder as VQGANDecoder
from .unet import Encoder as VQGANEncoder
from .utils import fid_has_enough_samples
from ._laser_visuals import VisualsMixin

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance

from src.audio_logging import (
    audio_config_from_source,
    build_audio_log_payload,
    compute_audio_energy_matching_loss,
    compute_audio_reconstruction_metrics,
    compute_waveform_multires_stft_loss,
    compute_waveform_multiscale_mel_loss,
    extract_audio_metadata_from_batch,
    has_audio_metadata,
)
from src.codebook_visuals import (
    render_codebook_scatter,
    save_codebook_trajectory_gif,
    select_codebook_vectors,
)
from src.wandb_media import log_wandb_images, log_wandb_payload, log_wandb_video


def _canonical_backbone(raw) -> str:
    # The attention U-Net (VQGAN/DDPM-style) is the only stage-1 backbone.
    # Legacy aliases ("simple", "scratch_vqvae", ...) are accepted and routed to
    # the U-Net so older configs and checkpoints keep loading.
    return "vqgan"


def _canonical_attn_resolutions(values) -> Tuple[int, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        raw = values.strip()
        if not raw:
            return ()
        return tuple(sorted({int(part) for part in raw.split(",") if part.strip()}))
    return tuple(sorted({int(v) for v in values}))


def _canonical_channel_multipliers(values) -> Optional[Tuple[int, ...]]:
    if values is None:
        return None
    if isinstance(values, str):
        raw = values.strip()
        if not raw or raw.lower() in {"none", "null"}:
            return None
        if raw[0] in "[(":
            values = ast.literal_eval(raw)
        else:
            values = [part for part in raw.split(",") if part.strip()]
    mults = tuple(int(v) for v in values)
    if not mults:
        return None
    if any(mult <= 0 for mult in mults):
        raise ValueError(f"channel_multipliers must be positive, got {mults}")
    return mults

class LASER(VisualsMixin, pl.LightningModule):
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
            backbone="simple",
            resolution=None,
            num_downsamples=4,
            attn_resolutions=(),
            dropout=0.0,
            channel_multipliers=None,
            backbone_latent_channels=None,
            max_ch_mult=4,
            decoder_extra_residual_layers=1,
            use_mid_attention=True,
            bottleneck_loss_weight=0.5,
            recon_mse_weight=1.0,
            recon_l1_weight=0.0,
            recon_edge_weight=0.0,
            perceptual_weight=1.0,
            perceptual_start_step=0,
            perceptual_warmup_steps=0,
            adversarial_weight=0.0,
            adversarial_start_step=0,
            adversarial_warmup_steps=0,
            audio_adversarial_type="hifigan",
            audio_disc_periods=(2, 3, 5, 7, 11),
            audio_disc_num_scales=3,
            audio_disc_max_channels=512,
            audio_disc_stft_fft_sizes=(),
            audio_feature_matching_weight=0.0,
            audio_mel_loss_weight=0.0,
            audio_mel_fft_sizes=(512, 1024, 2048),
            audio_mel_n_mels=80,
            discriminator_learning_rate=None,
            discriminator_beta1=0.5,
            discriminator_beta2=0.9,
            discriminator_channels=64,
            discriminator_layers=3,
            compute_fid=False,
            fid_feature=2048,
            log_images_every_n_steps=100,
            diag_log_interval=0,
            enable_val_latent_visuals=False,
            codebook_visual_max_vectors=1024,
            dict_learning_rate=None,
            patch_based=False,
            patch_size=2,
            patch_stride=2,
            patch_reconstruction="tile",
            coef_max=None,
            omp_residual_tolerance=None,
            bounded_omp_refine_steps=8,
            variational_coeffs=False,
            variational_coeff_refine_weight=0.0,
            variational_coeff_target_std=0.25,
            variational_coeff_min_std=0.01,
            data_init_from_first_batch=False,
            separable_dictionary_rank=0,
            separable_dictionary_factor_dims=None,
            bottleneck_type="dictionary",
            rq_code_depth=4,
            rq_shared_codebook=True,
            rq_decay=0.99,
            rq_restart_unused_codes=True,
            sparsity_reg_weight=0.01,
            coherence_weight=0.0,
            audio_energy_loss_weight=0.0,
            audio_multires_loss_weight=0.0,
            audio_multires_scales=(1, 2, 4, 8),
            audio_backbone="spectrogram",
            audio_downsample_rates=(4, 4, 4),
            audio_dilation_cycle=(1, 3, 9),
            audio_multires_stft_loss_weight=0.0,
            audio_multires_stft_fft_sizes=(512, 1024, 2048),
            audio_waveform_l1_weight=0.0,
            out_tanh=True,
            bypass_bottleneck=False,
            warmup_steps=0,
            min_lr_ratio=0.01,
            disc_start_step=0,
            disc_num_layers=3,
            disc_channels=64,
            disc_norm="batch",
            disc_spectral=False,
            disc_loss="hinge",
            disc_learning_rate=None,
            use_adaptive_disc_weight=True,
            disc_factor=1.0,
            disc_weight_max=10000.0,
            grad_clip_val=0.0,
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
            backbone: encoder/decoder family: 'simple', 'scratch_vqvae', or VQGAN/DDPM-style aliases
            resolution: input image resolution for the VQGAN-style backbone
            num_downsamples: number of spatial downsampling stages for variable-depth backbones
            attn_resolutions: spatial resolutions that should include self-attention blocks
            dropout: residual-block dropout for the VQGAN-style backbone
            channel_multipliers: optional explicit per-level width schedule for the
                VQGAN-style backbone; overrides max_ch_mult and sets the number of
                downsampling stages to len(channel_multipliers) - 1
            backbone_latent_channels: width of the continuous VQGAN/DDPM latent
                before projecting into the sparse bottleneck embedding_dim
            max_ch_mult: cap on channel multipliers for the VQGAN-style backbone
            decoder_extra_residual_layers: extra decoder residual blocks per level for the VQGAN-style backbone
            use_mid_attention: whether to keep the bottleneck self-attention block enabled
            commitment_cost: Commitment cost for bottleneck
            learning_rate: Learning rate for encoder/decoder
            beta: Beta parameter for Adam optimizer
            bottleneck_loss_weight: Weight for bottleneck loss term in total loss
            recon_mse_weight: Weight for pixel MSE reconstruction loss
            recon_l1_weight: Weight for pixel L1 reconstruction loss
            recon_edge_weight: Weight for first-order image gradient reconstruction loss
            perceptual_weight: Weight for perceptual loss
            perceptual_start_step: Global step before which LPIPS is disabled
            perceptual_warmup_steps: Linear LPIPS ramp length after perceptual_start_step
            adversarial_weight: Weight for PatchGAN generator loss. Disabled at 0.
            adversarial_start_step: Global step before which adversarial loss is disabled
            adversarial_warmup_steps: Linear adversarial weight ramp length after start
            audio_adversarial_type: waveform discriminator family; "hifigan" enables
                a compact multi-period/multi-scale critic for raw waveform audio
            discriminator_learning_rate: Optional discriminator LR; defaults to learning_rate
            discriminator_beta1: Discriminator Adam beta1
            discriminator_beta2: Discriminator Adam beta2
            discriminator_channels: Base discriminator channel count
            discriminator_layers: Number of strided discriminator blocks
            compute_fid: Whether to compute FID
            fid_feature: Inception feature size for reconstruction FID
            log_images_every_n_steps: image logging cadence; 0 disables image logging
            diag_log_interval: diagnostic logging cadence; 0 disables extra train diagnostics
            enable_val_latent_visuals: whether to log sparse heatmaps and dictionary atom trajectories
            codebook_visual_max_vectors: max dictionary atoms to draw in PCA scatter/GIF visualizations
            dict_learning_rate: optional learning rate override for dictionary atoms
            patch_based: whether to use latent patch sparse coding instead of per-site coding
            patch_size: latent patch size used for sparse coding
            patch_stride: latent patch stride used for sparse coding
            patch_reconstruction: patch stitching rule, one of 'center_crop', 'hann', or 'tile'
            coef_max: optional hard bound applied to sparse coefficients during support refinement
            bounded_omp_refine_steps: projected refinement steps for bounded OMP coefficient updates
            variational_coeffs: whether to sample sparse coefficients from a learned posterior
            variational_coeff_refine_weight: weight for the refinement-around-OMP loss on
                the coefficient posterior (Gaussian-KL math against a data-dependent
                reference; renamed from variational_coeff_kl_weight in May 2026)
            variational_coeff_target_std: std of the reference Gaussian in the refinement
                term and upper bound on the learned posterior std (renamed from
                variational_coeff_prior_std in May 2026)
            variational_coeff_min_std: minimum std allowed in the learned posterior
            data_init_from_first_batch: initialize dictionary atoms from first-batch
                encoder latents before the first OMP solve.
            sparsity_reg_weight: L1 regularization weight for active sparse coefficient magnitude
            coherence_weight: weight for dictionary coherence regularization
            audio_energy_loss_weight: weight for audio magnitude-energy matching
            audio_multires_loss_weight: weight for audio multi-resolution spectrogram matching
            audio_multires_scales: pooling scales for the multi-resolution audio loss
            audio_backbone: 'spectrogram' for 2D VCTK features or 'waveform' for raw waveform batches
            audio_downsample_rates: waveform encoder/decoder stride schedule
            audio_dilation_cycle: residual dilation cycle for waveform audio blocks
            audio_multires_stft_loss_weight: weight for raw-waveform multi-resolution STFT loss
            audio_multires_stft_fft_sizes: FFT sizes for raw-waveform multi-resolution STFT loss
            audio_waveform_l1_weight: additional raw-waveform L1 loss weight
            out_tanh: whether to bound decoder output to the normalized target range [-1, 1]
            bypass_bottleneck: diagnostic mode that trains encoder/decoder without sparse coding
            adversarial_weight: base weight for the PatchGAN generator loss. 0 (default)
                disables the discriminator and keeps the legacy single-optimizer
                automatic-optimization path untouched. >0 adds a PatchGAN critic and
                switches stage-1 training to manual optimization. This is the main lever
                for sharp, high-frequency detail and costs zero extra latent tokens.
            disc_start_step: training batches before the discriminator starts (autoencoder
                warmup); below it, training is pure reconstruction.
            disc_num_layers: PatchGAN downsampling depth (receptive-field size)
            disc_channels: PatchGAN base channel width (ndf)
            disc_norm: PatchGAN normalization: 'batch' (default), 'group' (DDP/small-batch
                safe), or 'none' (pair with disc_spectral=True)
            disc_spectral: wrap discriminator convs in spectral norm for a stabler critic
            disc_loss: adversarial objective, 'hinge' (default) or 'vanilla'
            disc_learning_rate: optional LR for the discriminator (defaults to learning_rate)
            use_adaptive_disc_weight: scale the generator loss by the VQGAN adaptive weight
                (ratio of recon-grad to adv-grad norms at the decoder output layer)
            disc_factor: post-warmup multiplier applied to both adversarial losses
            disc_weight_max: clamp on the adaptive discriminator weight
        """
        super(LASER, self).__init__()

        # Retired (June 2026 dictionary simplification): accept-and-ignore so old
        # checkpoint hparams / configs still construct the model.
        for _retired in (
            "dictionary_update_mode",
            "dictionary_through_decoder",
            "dead_atom_revival_steps",
            "dictionary_usage_ema_decay",
            "dictionary_usage_grad_scale",
            "dictionary_usage_grad_min",
            "dictionary_usage_grad_max",
            "dictionary_ksvd_lr",
            "dictionary_ksvd_update_every",
            "dictionary_ksvd_min_usage",
            "dictionary_ksvd_max_atoms_per_step",
            "online_ksvd_enabled",
            "online_ksvd_start_step",
            "online_ksvd_interval_steps",
            "online_ksvd_stop_step",
            "online_ksvd_max_samples",
            "online_ksvd_max_atoms",
            "online_ksvd_blend",
            "online_ksvd_min_coeff",
        ):
            kwargs.pop(_retired, None)
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported LASER arguments: {unknown}")

        in_channels = int(in_channels)
        audio_backbone_requested = str(audio_backbone or "spectrogram").strip().lower()
        waveform_audio_requested = audio_backbone_requested in {"waveform", "raw", "wav"}
        if in_channels != 3 and float(perceptual_weight) > 0.0:
            perceptual_weight = 0.0
        if (
            in_channels != 3
            and not (waveform_audio_requested and in_channels == 1)
            and float(adversarial_weight) > 0.0
        ):
            adversarial_weight = 0.0
        if in_channels != 3 and bool(compute_fid):
            compute_fid = False
        if disc_start_step == 0 and int(adversarial_start_step or 0) > 0:
            disc_start_step = adversarial_start_step
        if disc_learning_rate is None and discriminator_learning_rate is not None:
            disc_learning_rate = discriminator_learning_rate
        if int(disc_channels) == 64 and int(discriminator_channels) != 64:
            disc_channels = discriminator_channels
        if int(disc_num_layers) == 3 and int(discriminator_layers) != 3:
            disc_num_layers = discriminator_layers

        # Store parameters
        self.learning_rate = learning_rate
        self.beta = beta
        self.perceptual_weight = perceptual_weight
        self.perceptual_start_step = max(int(perceptual_start_step), 0)
        self.perceptual_warmup_steps = max(int(perceptual_warmup_steps), 0)
        self.adversarial_weight = float(adversarial_weight)
        self.adversarial_start_step = max(int(adversarial_start_step), 0)
        self.adversarial_warmup_steps = max(int(adversarial_warmup_steps), 0)
        self.discriminator_learning_rate = (
            float(discriminator_learning_rate)
            if discriminator_learning_rate is not None
            else float(learning_rate)
        )
        self.discriminator_beta1 = float(discriminator_beta1)
        self.discriminator_beta2 = float(discriminator_beta2)
        self.discriminator_channels = int(discriminator_channels)
        self.discriminator_layers = int(discriminator_layers)
        self.automatic_optimization = True
        self.manual_gradient_clip_val = None
        self.log_images_every_n_steps = max(int(log_images_every_n_steps), 0)
        self.compute_fid = compute_fid
        self.fid_feature = int(fid_feature)
        self.bottleneck_loss_weight = bottleneck_loss_weight
        self.sparsity_reg_weight = float(sparsity_reg_weight)
        self.coherence_weight = float(coherence_weight)
        self.recon_mse_weight = float(recon_mse_weight)
        self.recon_l1_weight = float(recon_l1_weight)
        self.recon_edge_weight = float(recon_edge_weight)
        self.audio_energy_loss_weight = float(audio_energy_loss_weight)
        self.audio_multires_loss_weight = float(audio_multires_loss_weight)
        self.audio_multires_scales = tuple(
            int(scale) for scale in audio_multires_scales if int(scale) > 0
        )
        if not self.audio_multires_scales:
            self.audio_multires_scales = (1,)
        self.audio_backbone = str(audio_backbone or "spectrogram").strip().lower()
        if self.audio_backbone in {"2d", "image", "stft", "logmag"}:
            self.audio_backbone = "spectrogram"
        if self.audio_backbone in {"raw", "wav"}:
            self.audio_backbone = "waveform"
        if self.audio_backbone not in {"spectrogram", "waveform"}:
            raise ValueError(
                f"Unsupported LASER audio_backbone {audio_backbone!r}; expected 'spectrogram' or 'waveform'"
            )
        self.audio_adversarial_type = str(audio_adversarial_type or "none").strip().lower()
        if self.audio_adversarial_type in {"", "off", "false", "0"}:
            self.audio_adversarial_type = "none"
        if self.audio_adversarial_type not in {"none", "hifigan"}:
            raise ValueError(
                "audio_adversarial_type must be 'none' or 'hifigan', got "
                f"{audio_adversarial_type!r}"
            )
        self.audio_disc_periods = canonical_int_tuple(audio_disc_periods, default=(2, 3, 5, 7, 11))
        self.audio_disc_num_scales = max(1, int(audio_disc_num_scales))
        self.audio_disc_max_channels = max(1, int(audio_disc_max_channels))
        # DAC/Encodec-style complex-STFT critics (empty = MPD/MSD only). Empty is a
        # valid value here, so canonical_int_tuple (which forbids empty) is only used
        # for the non-empty case.
        _stft_sizes = audio_disc_stft_fft_sizes
        if isinstance(_stft_sizes, str):
            _stft_sizes = _stft_sizes.strip()
            if _stft_sizes in ("", "()", "[]"):
                _stft_sizes = ()
        if _stft_sizes is None or (hasattr(_stft_sizes, "__len__") and len(_stft_sizes) == 0):
            self.audio_disc_stft_fft_sizes = ()
        else:
            self.audio_disc_stft_fft_sizes = canonical_int_tuple(_stft_sizes, default=(512,))
        # HiFi-GAN feature-matching + multi-scale mel reconstruction losses.
        self.audio_feature_matching_weight = float(audio_feature_matching_weight)
        self.audio_mel_loss_weight = float(audio_mel_loss_weight)
        self.audio_mel_fft_sizes = canonical_int_tuple(audio_mel_fft_sizes, default=(512, 1024, 2048))
        self.audio_mel_n_mels = max(1, int(audio_mel_n_mels))
        self.audio_downsample_rates = canonical_int_tuple(audio_downsample_rates, default=(4, 4, 4))
        self.audio_dilation_cycle = canonical_int_tuple(audio_dilation_cycle, default=(1, 3, 9))
        self.audio_multires_stft_loss_weight = float(audio_multires_stft_loss_weight)
        self.audio_multires_stft_fft_sizes = canonical_int_tuple(
            audio_multires_stft_fft_sizes,
            default=(512, 1024, 2048),
        )
        self.audio_waveform_l1_weight = float(audio_waveform_l1_weight)
        self.out_tanh = bool(out_tanh)
        self.bypass_bottleneck = bool(bypass_bottleneck)
        self.diag_log_interval = max(int(diag_log_interval), 0)
        self.enable_val_latent_visuals = bool(enable_val_latent_visuals)
        self.codebook_visual_max_vectors = max(1, int(codebook_visual_max_vectors))
        self.warmup_steps = max(int(warmup_steps), 0)
        self.min_lr_ratio = float(min_lr_ratio)
        self.adversarial_weight = float(adversarial_weight)
        self.disc_start_step = max(int(disc_start_step), 0)
        self.disc_num_layers = max(int(disc_num_layers), 1)
        self.disc_channels = max(int(disc_channels), 1)
        self.disc_norm = str(disc_norm or "none").strip().lower()
        self.disc_spectral = bool(disc_spectral)
        self.disc_loss = str(disc_loss or "hinge").strip().lower()
        if self.disc_loss not in {"hinge", "vanilla", "lsgan"}:
            raise ValueError(f"disc_loss must be 'hinge', 'vanilla', or 'lsgan', got {disc_loss!r}")
        self.disc_learning_rate = (
            None if disc_learning_rate is None else float(disc_learning_rate)
        )
        self.use_adaptive_disc_weight = bool(use_adaptive_disc_weight)
        self.disc_factor = float(disc_factor)
        self.disc_weight_max = float(disc_weight_max)
        # In manual optimization Lightning refuses Trainer-level gradient
        # clipping, so the adversarial path applies it itself with this value.
        self.grad_clip_val = float(grad_clip_val or 0.0)
        self.manual_gradient_clip_val = self.grad_clip_val
        self.backbone = _canonical_backbone(backbone)
        self.resolution = None if resolution is None else int(resolution)
        self.num_downsamples = int(num_downsamples)
        self.attn_resolutions = _canonical_attn_resolutions(attn_resolutions)
        self.dropout = float(dropout)
        self.channel_multipliers = _canonical_channel_multipliers(channel_multipliers)
        self.backbone_latent_channels = (
            None if backbone_latent_channels is None else int(backbone_latent_channels)
        )
        self.max_ch_mult = int(max_ch_mult)
        self.decoder_extra_residual_layers = int(decoder_extra_residual_layers)
        self.use_mid_attention = bool(use_mid_attention)
        if dict_learning_rate is not None and float(dict_learning_rate) <= 0.0:
            dict_learning_rate = None
        if self.num_downsamples < 0:
            raise ValueError(f"num_downsamples must be non-negative, got {self.num_downsamples}")
        if self.max_ch_mult <= 0:
            raise ValueError(f"max_ch_mult must be positive, got {self.max_ch_mult}")
        if self.backbone_latent_channels is not None and self.backbone_latent_channels <= 0:
            raise ValueError(
                "backbone_latent_channels must be positive when provided, got "
                f"{self.backbone_latent_channels}"
            )
        if self.decoder_extra_residual_layers < 0:
            raise ValueError(
                "decoder_extra_residual_layers must be non-negative, "
                f"got {self.decoder_extra_residual_layers}"
            )
        self.is_waveform_audio = self.audio_backbone == "waveform" and int(in_channels) == 1
        if self.audio_backbone == "waveform" and int(in_channels) != 1:
            raise ValueError("LASER waveform audio backbone expects in_channels=1")

        if not self.is_waveform_audio:
            # The attention U-Net backbone needs a positive input resolution and a
            # channel-multiplier schedule (waveform audio uses the 1D AudioEncoder).
            if self.resolution is None or self.resolution <= 0:
                raise ValueError("resolution must be a positive integer for the U-Net backbone")
            if self.channel_multipliers is None:
                self.channel_multipliers = tuple(
                    min(2 ** i, self.max_ch_mult) for i in range(self.num_downsamples + 1)
                )
            else:
                self.num_downsamples = len(self.channel_multipliers) - 1
            if self.backbone_latent_channels is None:
                self.backbone_latent_channels = int(embedding_dim)
        else:
            self.channel_multipliers = ()
            self.backbone_latent_channels = int(embedding_dim)

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
            self.post_bottleneck = nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_hiddens,
                kernel_size=3,
                stride=1,
                padding=1,
            )
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
            self.encoder = VQGANEncoder(**enc_dec_kwargs)
            if self.backbone_latent_channels == int(embedding_dim):
                self.pre_bottleneck = nn.Identity()
                self.post_bottleneck = nn.Identity()
            else:
                # Match Taming's quant_conv/post_quant_conv pattern so the
                # backbone latent can stay wide even when the sparse
                # embedding_dim is small.
                self.pre_bottleneck = nn.Conv2d(
                    in_channels=self.backbone_latent_channels,
                    out_channels=embedding_dim,
                    kernel_size=1,
                    stride=1,
                )
                self.post_bottleneck = nn.Conv2d(
                    in_channels=embedding_dim,
                    out_channels=self.backbone_latent_channels,
                    kernel_size=1,
                    stride=1,
                )
            self.decoder = VQGANDecoder(
                **dict(enc_dec_kwargs, extra_res_blocks=self.decoder_extra_residual_layers)
            )

        # Bottleneck: OMP dictionary learning, or kakaobrain-style residual quantization.
        bottleneck_type = str(bottleneck_type).strip().lower()
        if bottleneck_type == "dictionary":
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
                variational_coeffs=variational_coeffs,
                variational_coeff_refine_weight=variational_coeff_refine_weight,
                variational_coeff_target_std=variational_coeff_target_std,
                variational_coeff_min_std=variational_coeff_min_std,
                data_init_from_first_batch=data_init_from_first_batch,
                separable_dictionary_rank=separable_dictionary_rank,
                separable_dictionary_factor_dims=separable_dictionary_factor_dims,
            )
        elif bottleneck_type == "rq":
            from .rq_bottleneck import RQBottleneck
            self.bottleneck = RQBottleneck(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                code_depth=rq_code_depth,
                shared_codebook=rq_shared_codebook,
                decay=rq_decay,
                restart_unused_codes=rq_restart_unused_codes,
                commitment_cost=commitment_cost,
            )
        else:
            raise ValueError(
                f"bottleneck_type must be 'dictionary' or 'rq', got {bottleneck_type!r}"
            )
        self.bottleneck_type = bottleneck_type
        if self.bypass_bottleneck:
            self.bottleneck.requires_grad_(False)

        # Initialize LPIPS for perceptual loss only if used
        self.lpips = LPIPS() if self.perceptual_weight > 0 else None
        if self.lpips is not None:
            self.lpips.eval()
            for p in self.lpips.parameters():
                p.requires_grad = False
        # Adversarial critic. Images use the PatchGAN path; raw waveform audio
        # uses a compact HiFi-GAN-style multi-period/multi-scale discriminator.
        # When active, stage-1 training switches to manual optimization so the
        # autoencoder and discriminator can be optimized separately.
        self.discriminator = None
        self._adversarial_enabled = self.adversarial_weight > 0.0 and (
            not self.is_waveform_audio or self.audio_adversarial_type != "none"
        )
        if self.adversarial_weight > 0.0 and self.is_waveform_audio and self.audio_adversarial_type == "none":
            warnings.warn(
                "adversarial_weight>0 is ignored for waveform audio because "
                "audio_adversarial_type='none'.",
                RuntimeWarning,
                stacklevel=2,
            )
        if self._adversarial_enabled:
            if self.is_waveform_audio:
                self.discriminator = AudioMultiScalePeriodDiscriminator(
                    in_channels=in_channels,
                    num_filters=self.disc_channels,
                    max_filters=self.audio_disc_max_channels,
                    num_layers=self.disc_num_layers,
                    periods=self.audio_disc_periods,
                    num_scales=self.audio_disc_num_scales,
                    stft_fft_sizes=self.audio_disc_stft_fft_sizes,
                    spectral=self.disc_spectral,
                )
            else:
                self.discriminator = NLayerDiscriminator(
                    in_channels=in_channels,
                    num_filters=self.disc_channels,
                    num_layers=self.disc_num_layers,
                    norm=self.disc_norm,
                    spectral=self.disc_spectral,
                )
            # Two optimizers -> Lightning manual optimization. The legacy
            # automatic path (and all its dictionary/LR hooks) is preserved
            # verbatim whenever the discriminator is absent.
            self.automatic_optimization = False
            # Per-batch step counter for LR scheduling / discriminator warmup. In
            # manual optimization Lightning increments ``global_step`` once per
            # ``optimizer.step()`` (twice per batch here), so we track batches
            # ourselves. Persistent so the schedule/warmup survive resume. Only
            # registered in adversarial mode so legacy (automatic-path)
            # checkpoints still load with strict=True.
            self.register_buffer(
                "_manual_train_step", torch.zeros((), dtype=torch.long), persistent=True
            )

        # Separate metrics per split to avoid state leakage across train/val/test
        self.train_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.test_psnr = PeakSignalNoiseRatio(data_range=1.0)

        self.val_ssim = StructuralSimilarityIndexMeasure()
        self.test_ssim = StructuralSimilarityIndexMeasure()

        # Lazily instantiate FID metrics only when validation/test actually
        # starts so stage-1 fit startup does not reserve two extra Inception
        # networks on every DDP rank.
        self.val_rfid = None
        self.test_fid = None

        # Cache for validation visualization
        self._val_vis_batch = None
        # Dictionary atom snapshots for trajectory animation
        self._dict_snapshots = []
        self._dict_snapshot_steps = []
        # Dedupe manual media logs per prefix and optimizer step.
        self._media_log_steps = set()
        self._lr_base_lrs = ()
        self._disc_lr_base_lrs = ()
        # Stashes the graph-carrying recon/input between the generator and
        # discriminator halves of a manual-optimization training step.
        self._adv_cache = None
        self._lr_total_steps = 1
        # Updated by configure_optimizers / on_train_start; "uninit" means the
        # schedule helper hasn't been reached yet — _lr_multiplier_for_step will
        # behave as if the schedule is disabled when min_lr_ratio>=1 and
        # warmup_steps<=0 (the no-op guard at the top of the function).
        self._lr_total_steps_source = "uninit"

        # Save hyperparameters
        backbone = self.backbone
        resolution = self.resolution
        num_downsamples = self.num_downsamples
        attn_resolutions = self.attn_resolutions
        dropout = self.dropout
        channel_multipliers = self.channel_multipliers
        backbone_latent_channels = self.backbone_latent_channels
        max_ch_mult = self.max_ch_mult
        decoder_extra_residual_layers = self.decoder_extra_residual_layers
        use_mid_attention = self.use_mid_attention
        self.save_hyperparameters()

    def _should_log_images(self, batch_idx, prefix='train'):
        if self.log_images_every_n_steps <= 0:
            return False
        if prefix == 'train':
            step = int(getattr(self, "global_step", 0))
            return step > 0 and step % self.log_images_every_n_steps == 0
        return int(batch_idx) == 0

    def _new_fid_metric(self):
        metric = FrechetInceptionDistance(feature=self.fid_feature, normalize=True)
        metric.eval()
        for p in metric.parameters():
            p.requires_grad = False
        return metric

    def _ensure_val_rfid(self):
        if not self.compute_fid:
            return None
        if self.val_rfid is None:
            self.val_rfid = self._new_fid_metric()
        self.val_rfid = self.val_rfid.to(self.device)
        return self.val_rfid

    def _ensure_test_fid(self):
        if not self.compute_fid:
            return None
        if self.test_fid is None:
            self.test_fid = self._new_fid_metric()
        self.test_fid = self.test_fid.to(self.device)
        return self.test_fid

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

    def _is_log_rank_zero(self):
        # Most reliable check first: ask torch.distributed directly. Both DDP
        # ranks have the same `is_global_zero` problem when Lightning's hook
        # hasn't attached the trainer yet, and ``self.global_rank`` can be 0 on
        # both ranks before that point.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        trainer = self._trainer_ref()
        if trainer is not None:
            flag = getattr(trainer, "is_global_zero", None)
            if flag is not None:
                return bool(flag)
        try:
            return int(getattr(self, "global_rank", 0)) == 0
        except Exception:
            pass
        for env_key in ("RANK", "SLURM_PROCID"):
            raw = os.environ.get(env_key)
            if raw is None:
                continue
            try:
                return int(raw) == 0
            except ValueError:
                continue
        return True

    def _claim_media_log(self, prefix, step):
        key = (str(prefix), int(step))
        if key in self._media_log_steps:
            return False
        self._media_log_steps.add(key)
        return True

    def _wandb_step(self, experiment=None, *, requested_step: Optional[int] = None) -> int:
        step = int(self.global_step if requested_step is None else requested_step)
        exp = experiment
        if exp is None:
            logger = getattr(self, "logger", None)
            exp = getattr(logger, "experiment", None) if logger is not None else None
        if exp is not None:
            for attr in ("step", "_step"):
                raw = getattr(exp, attr, None)
                if raw is None:
                    continue
                try:
                    step = max(step, int(raw))
                except (TypeError, ValueError):
                    continue
        return step

    def _wandb_epoch_end_step(self, experiment=None, *, requested_step: Optional[int] = None) -> int:
        base = int(self.global_step if requested_step is None else requested_step)
        return self._wandb_step(experiment, requested_step=base + 1)

    def _resolve_lr_total_steps(self, trainer) -> Tuple[int, str]:
        """Return ``(total_steps, source)`` for the cosine LR schedule.

        Tries explicit trainer limits first, then ``max_epochs * num_training_batches``.
        Lightning's ``estimated_stepping_batches`` property can force dataloader
        setup during DDP optimizer construction, so only a precomputed instance
        value is used as a final fallback. Returns ``(1, "fallback")`` if none of
        them are finite/positive — callers should treat that as "schedule disabled"
        and log a warning. The two-step resolve in ``configure_optimizers`` plus
        ``on_train_start`` covers trainer state that is not ready until fit starts.
        """
        if trainer is None:
            return 1, "fallback"

        def _coerce_positive_finite(value) -> Optional[int]:
            if value is None:
                return None
            try:
                fvalue = float(value)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(fvalue) or fvalue <= 0:
                return None
            return max(1, int(fvalue))

        candidates = [("max_steps", getattr(trainer, "max_steps", None))]
        max_epochs = getattr(trainer, "max_epochs", None)
        num_train_batches = getattr(trainer, "num_training_batches", None)
        if max_epochs is not None and num_train_batches is not None:
            max_epochs_resolved = _coerce_positive_finite(max_epochs)
            num_batches_resolved = _coerce_positive_finite(num_train_batches)
            if max_epochs_resolved is not None and num_batches_resolved is not None:
                product = max_epochs_resolved * num_batches_resolved
            else:
                product = None
            candidates.append(("max_epochs*num_training_batches", product))

        for source, value in candidates:
            resolved = _coerce_positive_finite(value)
            if resolved is not None:
                return resolved, source

        est = None
        descriptor = getattr(type(trainer), "estimated_stepping_batches", None)
        if isinstance(descriptor, property):
            trainer_state = getattr(trainer, "__dict__", {})
            est = trainer_state.get("estimated_stepping_batches") if isinstance(trainer_state, dict) else None
        else:
            try:
                est = getattr(trainer, "estimated_stepping_batches", None)
            except Exception:
                est = None
        resolved = _coerce_positive_finite(est)
        if resolved is not None:
            return resolved, "estimated_stepping_batches"
        return 1, "fallback"

    def _lr_multiplier_for_step(self, step: int) -> float:
        if self.warmup_steps <= 0 and self.min_lr_ratio >= 1.0:
            return 1.0
        total_steps = max(1, int(getattr(self, "_lr_total_steps", 1)))
        warmup = min(self.warmup_steps, max(0, total_steps - 1))
        step = max(0, int(step))
        if warmup > 0 and step < warmup:
            return max(self.min_lr_ratio, step / float(max(1, warmup)))
        progress = (step - warmup) / float(max(1, total_steps - warmup))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def _apply_scheduled_lrs(self, optimizer, step: int, base_lrs=None) -> None:
        if base_lrs is None:
            if not self._lr_base_lrs:
                self._lr_base_lrs = tuple(float(group["lr"]) for group in optimizer.param_groups)
            base_lrs = self._lr_base_lrs
        scale = self._lr_multiplier_for_step(step)
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = float(base_lr) * float(scale)

    def on_fit_start(self):
        super().on_fit_start()
        if not self.bypass_bottleneck:
            self.bottleneck.normalize_dictionary_()
        self._snapshot_dictionary()

    def on_train_start(self):
        """Re-resolve and report the LR schedule horizon.

        ``configure_optimizers`` runs before some trainer state is finalized
        (notably ``num_training_batches`` for iterable datamodules). Re-resolve
        once more before the first training step so the cosine schedule does
        not silently collapse to ``min_lr_ratio`` because ``_lr_total_steps``
        was captured as 1. See A4 in the May 2026 review.
        """
        super().on_train_start()
        trainer = self._trainer_ref()
        total_steps, source = self._resolve_lr_total_steps(trainer)
        prior_steps = int(getattr(self, "_lr_total_steps", 1))
        prior_source = str(getattr(self, "_lr_total_steps_source", "uninit"))
        # Trust whichever resolution actually produced a non-fallback estimate.
        # Prefer the on_train_start resolution if it improved; otherwise keep
        # the configure_optimizers value (it may have been the only one that
        # worked for some trainer configurations).
        if source != "fallback" and (prior_source == "fallback" or total_steps > prior_steps):
            self._lr_total_steps = total_steps
            self._lr_total_steps_source = source
        if self._lr_total_steps <= 1 and (self.warmup_steps > 0 or self.min_lr_ratio < 1.0):
            warnings.warn(
                "LASER cosine LR schedule has _lr_total_steps=1; the schedule will "
                "collapse to min_lr_ratio immediately. Check trainer.max_steps / "
                "trainer.max_epochs / dataloader length.",
                RuntimeWarning,
                stacklevel=2,
            )
        if self._is_log_rank_zero():
            print(
                f"[LASER] LR schedule horizon: total_steps={self._lr_total_steps} "
                f"(source={self._lr_total_steps_source}, warmup_steps={self.warmup_steps}, "
                f"min_lr_ratio={self.min_lr_ratio})"
            )

    def on_before_optimizer_step(self, optimizer):
        del optimizer
        # Manual optimization (adversarial mode) drives dictionary gradient
        # projection inline in training_step so it only touches the autoencoder
        # optimizer; skip the automatic hook there to avoid double-projecting.
        if not self.automatic_optimization:
            return
        if not self.bypass_bottleneck:
            self.bottleneck.project_dictionary_gradient_()

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # Renormalize after the step: Lightning runs zero_grad (and on_before_zero_grad)
        # after forward but before backward inside the optimizer closure, so in-place
        # dictionary updates there break autograd for the current batch.
        step = int(getattr(self, "global_step", 0))
        self._apply_scheduled_lrs(optimizer, step=step)
        optimizer.step(closure=optimizer_closure)
        if not self.bypass_bottleneck:
            self.bottleneck.normalize_dictionary_()

    def _empty_sparse_codes_like(self, z_e: torch.Tensor) -> SparseCodes:
        batch_size, _, height, width = z_e.shape
        shape = (batch_size, height, width, 0)
        return SparseCodes(
            support=torch.empty(shape, device=z_e.device, dtype=torch.long),
            values=z_e.new_empty(shape),
            num_embeddings=max(int(getattr(self.bottleneck, "num_embeddings", 1)), 1),
        )

    def _to_bottleneck_input(self, z_e: torch.Tensor) -> torch.Tensor:
        if self.is_waveform_audio:
            if z_e.ndim != 3:
                raise ValueError(f"Expected waveform latent [B, C, T], got {tuple(z_e.shape)}")
            return z_e.unsqueeze(2)
        return z_e

    def _from_bottleneck_output(self, z_dl: torch.Tensor) -> torch.Tensor:
        if self.is_waveform_audio:
            if z_dl.ndim == 4 and int(z_dl.size(2)) == 1:
                return z_dl.squeeze(2)
            if z_dl.ndim == 3:
                return z_dl
            raise ValueError(f"Expected waveform bottleneck latent [B, C, 1, T], got {tuple(z_dl.shape)}")
        return z_dl

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
        z_e = self._to_bottleneck_input(z_e)
        if self.bypass_bottleneck:
            return z_e, z_e.new_zeros(()), self._empty_sparse_codes_like(z_e)
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
        z_dl = self._from_bottleneck_output(z_dl)
        z_dl = self.post_bottleneck(z_dl)
        x_recon = self.decoder(z_dl)
        return self._apply_output_activation(x_recon)

    def _apply_output_activation(self, x_recon: torch.Tensor) -> torch.Tensor:
        if self.out_tanh:
            return torch.tanh(x_recon)
        return x_recon

    def _image_to_unit_range(self, x: torch.Tensor, *, clamp: bool = True) -> torch.Tensor:
        """Map normalized image tensors to [0, 1] using the active datamodule stats."""
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        dm = getattr(self._trainer_ref(), "datamodule", None)
        if dm is not None and hasattr(dm, "config") and hasattr(dm.config, "mean") and hasattr(dm.config, "std"):
            stat_shape = (1, -1, 1) if x.ndim == 3 else (1, -1, 1, 1)
            mean = torch.tensor(dm.config.mean, device=x.device, dtype=x.dtype).view(*stat_shape)
            std = torch.tensor(dm.config.std, device=x.device, dtype=x.dtype).view(*stat_shape)
            x = x * std + mean
        else:
            x = (x + 1.0) / 2.0
        if clamp:
            x = x.clamp(0.0, 1.0)
        return x

    def _image_to_lpips_range(self, x: torch.Tensor) -> torch.Tensor:
        return self._image_to_unit_range(x, clamp=True).mul(2.0).sub(1.0)

    def _effective_perceptual_weight(self, prefix: str) -> float:
        if prefix != "train" or self.perceptual_weight <= 0:
            return 0.0
        step = int(getattr(self, "global_step", 0))
        if step < self.perceptual_start_step:
            return 0.0
        if self.perceptual_warmup_steps <= 0:
            return float(self.perceptual_weight)
        progress = (step - self.perceptual_start_step) / float(max(1, self.perceptual_warmup_steps))
        return float(self.perceptual_weight) * max(0.0, min(1.0, progress))

    def _effective_adversarial_weight(self, prefix: str) -> float:
        if prefix != "train" or self.adversarial_weight <= 0 or self.discriminator is None:
            return 0.0
        step = int(getattr(self, "global_step", 0))
        if step < self.adversarial_start_step:
            return 0.0
        if self.adversarial_warmup_steps <= 0:
            return float(self.adversarial_weight)
        progress = (step - self.adversarial_start_step) / float(max(1, self.adversarial_warmup_steps))
        return float(self.adversarial_weight) * max(0.0, min(1.0, progress))

    def _adversarial_generator_loss(self, recon: torch.Tensor) -> torch.Tensor:
        if self.discriminator is None:
            return recon.new_zeros(())
        return self._generator_adv_loss(self.discriminator(recon))

    def _discriminator_hinge_loss(self, real: torch.Tensor, fake: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.discriminator is None:
            zero = real.new_zeros(())
            return zero, zero, zero
        real_logits = self.discriminator(real.detach())
        fake_logits = self.discriminator(fake.detach())
        real_loss = F.relu(1.0 - real_logits).mean()
        fake_loss = F.relu(1.0 + fake_logits).mean()
        return 0.5 * (real_loss + fake_loss), real_logits.mean(), fake_logits.mean()

    def _set_discriminator_requires_grad(self, enabled: bool) -> None:
        if self.discriminator is None:
            return
        for param in self.discriminator.parameters():
            param.requires_grad_(enabled)

    @staticmethod
    def _raw_optimizer(optimizer):
        return getattr(optimizer, "optimizer", optimizer)

    def _clip_manual_optimizer(self, optimizer) -> None:
        trainer = self._trainer_ref()
        clip_val = getattr(self, "manual_gradient_clip_val", None)
        if clip_val is None:
            clip_val = getattr(trainer, "gradient_clip_val", 0.0)
        clip_val = float(clip_val or 0.0)
        if clip_val > 0.0:
            params = [
                param
                for group in optimizer.param_groups
                for param in group["params"]
                if param.grad is not None
            ]
            if params:
                torch.nn.utils.clip_grad_norm_(params, max_norm=clip_val, norm_type=2.0)

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
        if self.bypass_bottleneck:
            raise RuntimeError("encode_to_tokens is unavailable when bypass_bottleneck=true")
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
        if self.bypass_bottleneck:
            raise RuntimeError("encode_to_atoms_and_coeffs is unavailable when bypass_bottleneck=true")
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
        if self.is_waveform_audio:
            dummy = torch.zeros(
                1,
                int(self.hparams.in_channels),
                image_w,
                device=device,
                dtype=torch.float32,
            )
        else:
            dummy = torch.zeros(
                1,
                int(self.hparams.in_channels),
                image_h,
                image_w,
                device=device,
                dtype=torch.float32,
            )
        z = self.pre_bottleneck(self.encoder(dummy))
        z = self._to_bottleneck_input(z)
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
        z_dl, bottleneck_loss, sparse_codes = self.encode(x)
        return self.decode(z_dl), bottleneck_loss, sparse_codes

    def _sparse_coeff_denominator(self, sparse_codes: SparseCodes) -> float:
        num_sites = max(
            int(
                sparse_codes.values.shape[0]
                * sparse_codes.values.shape[1]
                * sparse_codes.values.shape[2]
            ),
            1,
        )
        return float(sparse_codes.num_embeddings * num_sites)

    def _dense_coeff_abs_mean(self, sparse_codes: SparseCodes):
        return sparse_codes.values.abs().sum() / self._sparse_coeff_denominator(sparse_codes)

    def _support_fraction(self, sparse_codes: SparseCodes):
        return sparse_codes.support.numel() / self._sparse_coeff_denominator(sparse_codes)

    def _effective_coeff_nonzero_fraction(self, sparse_codes: SparseCodes, threshold=1e-6):
        return (
            (sparse_codes.values.abs() > threshold).float().sum()
            / self._sparse_coeff_denominator(sparse_codes)
        )

    def _reconstruction_edge_loss(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.recon_edge_weight <= 0.0:
            return recon.new_zeros(())
        losses = []
        if recon.shape[-1] > 1:
            losses.append(
                F.l1_loss(
                    recon[..., :, 1:] - recon[..., :, :-1],
                    target[..., :, 1:] - target[..., :, :-1],
                )
            )
        if recon.shape[-2] > 1:
            losses.append(
                F.l1_loss(
                    recon[..., 1:, :] - recon[..., :-1, :],
                    target[..., 1:, :] - target[..., :-1, :],
                )
            )
        if not losses:
            return recon.new_zeros(())
        return torch.stack(losses).mean()

    def _normalized_audio_unit(self, spec: torch.Tensor, audio_source) -> torch.Tensor:
        config = audio_config_from_source(audio_source)
        mean = torch.tensor(config["mean"], dtype=spec.dtype, device=spec.device).view(1, -1, 1, 1)
        std = torch.tensor(config["std"], dtype=spec.dtype, device=spec.device).view(1, -1, 1, 1)
        return (spec * std + mean).clamp(0.0, 1.0)

    def _audio_multiresolution_spectrogram_loss(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        audio_source,
    ) -> dict[str, torch.Tensor]:
        if self.audio_multires_loss_weight <= 0.0:
            return {}
        if recon.ndim != 4 or target.ndim != 4 or int(recon.size(1)) != 1 or int(target.size(1)) != 1:
            return {}

        recon_unit = self._normalized_audio_unit(recon, audio_source)
        target_unit = self._normalized_audio_unit(target, audio_source)
        eps = 1.0e-6
        l1_terms = []
        mse_terms = []
        convergence_terms = []
        for scale in self.audio_multires_scales:
            if scale > 1:
                recon_scaled = F.avg_pool2d(recon_unit, kernel_size=scale, stride=scale, ceil_mode=True)
                target_scaled = F.avg_pool2d(target_unit, kernel_size=scale, stride=scale, ceil_mode=True)
            else:
                recon_scaled = recon_unit
                target_scaled = target_unit
            diff = recon_scaled - target_scaled
            l1_terms.append(diff.abs().mean())
            mse_terms.append(diff.pow(2).mean())
            convergence_terms.append(
                torch.linalg.vector_norm(diff.float())
                / torch.linalg.vector_norm(target_scaled.float()).clamp_min(eps)
            )

        l1_loss = torch.stack(l1_terms).mean()
        mse_loss = torch.stack(mse_terms).mean()
        convergence_loss = torch.stack(convergence_terms).mean().to(dtype=recon.dtype)
        freq_profile_loss = F.l1_loss(recon_unit.mean(dim=-1), target_unit.mean(dim=-1))
        time_profile_loss = F.l1_loss(recon_unit.mean(dim=-2), target_unit.mean(dim=-2))
        profile_loss = 0.5 * (freq_profile_loss + time_profile_loss)
        total = l1_loss + 0.5 * mse_loss + 0.25 * convergence_loss + 0.5 * profile_loss
        return {
            "audio_multires_loss": total,
            "audio_multires_l1_loss": l1_loss,
            "audio_multires_mse_loss": mse_loss,
            "audio_multires_convergence_loss": convergence_loss,
            "audio_multires_profile_loss": profile_loss,
        }

    def _get_last_layer_weight(self) -> torch.Tensor:
        """Decoder output-conv weight, used for the adaptive adversarial weight."""
        return self.decoder.conv_out.weight

    def _adaptive_disc_weight(self, reference_loss, g_loss) -> torch.Tensor:
        """VQGAN adaptive weight: balance recon vs adversarial gradients at the
        decoder output layer so neither term dominates as training proceeds."""
        last = self._get_last_layer_weight()
        ref_grad = torch.autograd.grad(reference_loss, last, retain_graph=True, allow_unused=True)[0]
        g_grad = torch.autograd.grad(g_loss, last, retain_graph=True, allow_unused=True)[0]
        if ref_grad is None or g_grad is None:
            return last.new_tensor(1.0)
        weight = ref_grad.norm() / (g_grad.norm() + 1e-4)
        return torch.clamp(weight, 0.0, self.disc_weight_max).detach()

    def _discriminator_loss(self, real: torch.Tensor, fake: torch.Tensor):
        """Critic loss on real images vs detached reconstructions."""
        logits_real = self.discriminator(real.contiguous())
        logits_fake = self.discriminator(fake.contiguous())
        if self.disc_loss == "hinge":
            d_fn = multi_hinge_d_loss
        elif self.disc_loss == "lsgan":
            d_fn = multi_lsgan_d_loss
        else:
            d_fn = multi_vanilla_d_loss
        return d_fn(logits_real, logits_fake), logits_real, logits_fake

    def _generator_adv_loss(self, logits_fake) -> torch.Tensor:
        """Generator adversarial loss matching the configured critic loss
        (least-squares for ``lsgan``; otherwise the non-saturating hinge G-loss)."""
        if self.disc_loss == "lsgan":
            return multi_lsgan_g_loss(logits_fake)
        return multi_hinge_g_loss(logits_fake)

    @staticmethod
    def _logits_mean(logits) -> torch.Tensor:
        """Return a scalar mean for tensor or multi-discriminator logits."""
        if isinstance(logits, torch.Tensor):
            return logits.mean()
        means = [item.mean() for item in logits]
        if not means:
            raise ValueError("expected at least one discriminator logit tensor")
        return torch.stack(means).mean()

    def compute_metrics(self, batch, prefix='train', *, include_adversarial: bool = True, return_raw: bool = False):
        """Compute metrics for a batch."""
        # Get input
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        audio_meta = extract_audio_metadata_from_batch(batch)
        is_audio = has_audio_metadata(audio_meta)
        is_waveform_audio = is_audio and torch.is_tensor(x) and x.ndim == 3
        needs_train_diag = prefix == 'train' and self._should_log_train_diagnostics()
        should_log_distribution_metrics = prefix != 'train' or needs_train_diag

        recon_raw, bottleneck_loss, sparse_codes = self(x)

        # Keep raw tensors for loss; create sanitized copies for metrics/visualization only
        recon_vis = torch.nan_to_num(recon_raw.detach(), nan=0.0, posinf=1.0, neginf=-1.0).clamp_(-1.0, 1.0)
        x_vis = torch.nan_to_num(x.detach(), nan=0.0, posinf=1.0, neginf=-1.0).clamp_(-1.0, 1.0)
        
        # Compute losses
        recon_mse_loss = F.mse_loss(recon_raw, x)
        recon_l1_loss = F.l1_loss(recon_raw, x)
        recon_edge_loss = self._reconstruction_edge_loss(recon_raw, x)
        recon_loss = (
            self.recon_mse_weight * recon_mse_loss
            + self.recon_l1_weight * recon_l1_loss
            + self.recon_edge_weight * recon_edge_loss
        )
        input_mean = input_std = recon_mean = recon_std = None
        if should_log_distribution_metrics:
            input_mean = x.mean()
            input_std = x.std()
            recon_mean = recon_raw.mean()
            recon_std = recon_raw.std()
        diag = {} if self.bypass_bottleneck else getattr(self.bottleneck, "_last_diag", {})
        
        # Perceptual loss - only compute during training for quality
        perceptual_weight = self._effective_perceptual_weight(prefix)
        if self.lpips is not None and perceptual_weight > 0:
            perceptual_loss = self.lpips(
                self._image_to_lpips_range(recon_raw),
                self._image_to_lpips_range(x),
            ).mean()
        else:
            perceptual_loss = recon_raw.new_zeros(())
        adversarial_weight = 0.0
        adversarial_generator_loss = recon_raw.new_zeros(())

        sparsity_reg_loss = torch.nan_to_num(sparse_codes.values).abs().mean()
        weighted_sparsity_reg_loss = self.sparsity_reg_weight * sparsity_reg_loss

        total_loss = (
            recon_loss
            + self.bottleneck_loss_weight * bottleneck_loss
            + perceptual_weight * perceptual_loss
            + adversarial_weight * adversarial_generator_loss
            + weighted_sparsity_reg_loss
        )
        
        # Compute PSNR/SSIM/rFID on de-normalized image tensors.
        x_dn = self._image_to_unit_range(x.detach(), clamp=True)
        recon_dn = self._image_to_unit_range(recon_raw.detach(), clamp=True)

        if not is_audio and prefix == 'val' and self.val_rfid is not None:
            self.val_rfid.update(x_dn, real=True)
            self.val_rfid.update(recon_dn, real=False)
        elif not is_audio and prefix == 'test' and self.test_fid is not None:
            self.test_fid.update(x_dn, real=True)
            self.test_fid.update(recon_dn, real=False)

        psnr = None
        ssim = None
        audio_metrics = {}
        audio_energy_metrics = {}
        audio_multires_metrics = {}
        audio_energy_loss = recon_raw.new_zeros(())
        weighted_audio_energy_loss = recon_raw.new_zeros(())
        audio_multires_loss = recon_raw.new_zeros(())
        weighted_audio_multires_loss = recon_raw.new_zeros(())
        waveform_l1_loss = recon_raw.new_zeros(())
        weighted_waveform_l1_loss = recon_raw.new_zeros(())
        weighted_audio_multires_stft_loss = recon_raw.new_zeros(())
        audio_mel_loss = recon_raw.new_zeros(())
        weighted_audio_mel_loss = recon_raw.new_zeros(())
        if not is_audio:
            if prefix == 'train':
                psnr = self.train_psnr(recon_dn, x_dn)
            elif prefix == 'val':
                psnr = self.val_psnr(recon_dn, x_dn)
                ssim = self.val_ssim(recon_dn, x_dn)
            else:
                psnr = self.test_psnr(recon_dn, x_dn)
                ssim = self.test_ssim(recon_dn, x_dn)
        else:
            dm = getattr(self._trainer_ref(), "datamodule", None)
            audio_source = getattr(dm, "config", {"dataset": "vctk"})
            if is_waveform_audio:
                if self.audio_waveform_l1_weight > 0:
                    waveform_l1_loss = F.l1_loss(recon_raw, x)
                    weighted_waveform_l1_loss = self.audio_waveform_l1_weight * waveform_l1_loss
                    total_loss = total_loss + weighted_waveform_l1_loss
                if self.audio_multires_stft_loss_weight > 0:
                    audio_multires_metrics = compute_waveform_multires_stft_loss(
                        x,
                        recon_raw,
                        fft_sizes=self.audio_multires_stft_fft_sizes,
                    )
                    if audio_multires_metrics:
                        audio_multires_loss = audio_multires_metrics["audio_multires_stft_loss"]
                        weighted_audio_multires_stft_loss = (
                            self.audio_multires_stft_loss_weight * audio_multires_loss
                        )
                        total_loss = total_loss + weighted_audio_multires_stft_loss
                if self.audio_mel_loss_weight > 0:
                    mel_sr = 16000
                    _sr_getter = getattr(audio_source, "get", None)
                    if callable(_sr_getter):
                        try:
                            mel_sr = int(_sr_getter("sample_rate", mel_sr) or mel_sr)
                        except (TypeError, ValueError):
                            mel_sr = 16000
                    mel_metrics = compute_waveform_multiscale_mel_loss(
                        x,
                        recon_raw,
                        sample_rate=mel_sr,
                        fft_sizes=self.audio_mel_fft_sizes,
                        n_mels=self.audio_mel_n_mels,
                    )
                    if mel_metrics:
                        audio_mel_loss = mel_metrics["audio_mel_loss"]
                        weighted_audio_mel_loss = self.audio_mel_loss_weight * audio_mel_loss
                        total_loss = total_loss + weighted_audio_mel_loss
            else:
                audio_multires_metrics = self._audio_multiresolution_spectrogram_loss(
                    recon_raw,
                    x,
                    audio_source,
                )
                if audio_multires_metrics:
                    audio_multires_loss = audio_multires_metrics["audio_multires_loss"]
                    weighted_audio_multires_loss = self.audio_multires_loss_weight * audio_multires_loss
                    total_loss = total_loss + weighted_audio_multires_loss
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
                    recon_raw.new_zeros(()),
                )
                weighted_audio_energy_loss = self.audio_energy_loss_weight * audio_energy_loss
                total_loss = total_loss + weighted_audio_energy_loss
        
        # Logging kwargs are defined here (before the adversarial term) so the
        # reported {prefix}/loss reflects every contribution and the adversarial
        # metrics can share the same on_step/on_epoch/sync_dist policy.
        log_kwargs = dict(
            on_step=(prefix == 'train'),
            on_epoch=(prefix != 'train'),
            sync_dist=True,
            batch_size=int(x.size(0)),
        )

        # Adversarial generator term. During training, once the
        # discriminator warmup completes, push the decoder toward synthesizing
        # high-frequency detail; the adaptive weight keeps it balanced against
        # the reconstruction objective. In val/test we only log the value for
        # monitoring without touching the loss.
        adv_g_loss = recon_raw.new_zeros(())
        disc_weight = recon_raw.new_zeros(())
        if include_adversarial and self._adversarial_enabled and self.discriminator is not None:
            step = int(self._manual_train_step)
            active = step >= self.disc_start_step
            factor = adopt_weight(self.disc_factor, step, self.disc_start_step)
            if active:
                adversarial_weight = float(self.adversarial_weight) * float(factor)
            # Feature matching is the core HiFi-GAN signal: only available for the
            # waveform critics (which expose intermediate features) and when its
            # weight is on. The image PatchGAN path keeps the logits-only behavior.
            want_fm = (
                is_waveform_audio
                and self.audio_feature_matching_weight > 0.0
                and isinstance(self.discriminator, AudioMultiScalePeriodDiscriminator)
            )
            if prefix == 'train' and active and recon_raw.requires_grad:
                if want_fm:
                    logits_fake, feats_fake = self.discriminator(recon_raw, return_features=True)
                else:
                    logits_fake = self.discriminator(recon_raw)
                adv_g_loss = self._generator_adv_loss(logits_fake)
                adversarial_generator_loss = adv_g_loss
                reference_loss = recon_loss + perceptual_weight * perceptual_loss
                if is_waveform_audio:
                    reference_loss = (
                        reference_loss
                        + weighted_waveform_l1_loss
                        + weighted_audio_multires_stft_loss
                        + weighted_audio_mel_loss
                    )
                if self.use_adaptive_disc_weight:
                    disc_weight = self._adaptive_disc_weight(reference_loss, adv_g_loss)
                else:
                    disc_weight = recon_raw.new_tensor(1.0)
                weighted_adv = self.adversarial_weight * disc_weight * factor * adv_g_loss
                total_loss = total_loss + weighted_adv
                self.log(f'{prefix}/weighted_adv_g_loss', weighted_adv, **log_kwargs)
                if want_fm:
                    # Real features are a detached target (no grad to the critic
                    # here; its own update runs separately and zeroes these grads).
                    with torch.no_grad():
                        _, feats_real = self.discriminator(x, return_features=True)
                    fm_loss = feature_matching_loss(feats_real, feats_fake)
                    weighted_fm = self.audio_feature_matching_weight * factor * fm_loss
                    total_loss = total_loss + weighted_fm
                    self.log(f'{prefix}/audio_feature_matching_loss', fm_loss, **log_kwargs)
                    self.log(f'{prefix}/weighted_audio_feature_matching_loss', weighted_fm, **log_kwargs)
                # Hand the graph-carrying tensors to the manual training step's
                # discriminator update so it reuses this forward pass.
                self._adv_cache = {"recon": recon_raw, "real": x}
            elif prefix != 'train':
                with torch.no_grad():
                    adv_g_loss = self._generator_adv_loss(self.discriminator(recon_raw))
                adversarial_generator_loss = adv_g_loss
            self.log(f'{prefix}/adv_g_loss', adv_g_loss, **log_kwargs)
            self.log(f'{prefix}/disc_weight', disc_weight, **log_kwargs)

        # Compute sparsity
        sparsity = self._support_fraction(sparse_codes)
        effective_sparsity = self._effective_coeff_nonzero_fraction(sparse_codes)

        self.log(f'{prefix}/loss', total_loss, prog_bar=True, **log_kwargs)
        self.log(f'{prefix}/recon_loss', recon_loss, **log_kwargs)
        self.log(f'{prefix}/recon_mse_loss', recon_mse_loss, **log_kwargs)
        self.log(f'{prefix}/recon_l1_loss', recon_l1_loss, **log_kwargs)
        self.log(f'{prefix}/recon_edge_loss', recon_edge_loss, **log_kwargs)
        self.log(f'{prefix}/bottleneck_loss', bottleneck_loss, **log_kwargs)
        self.log(f'{prefix}/weighted_bottleneck_loss', self.bottleneck_loss_weight * bottleneck_loss, **log_kwargs)
        self.log(f'{prefix}/sparsity_reg_loss', sparsity_reg_loss, **log_kwargs)
        self.log(f'{prefix}/weighted_sparsity_reg_loss', weighted_sparsity_reg_loss, **log_kwargs)
        self.log(f'{prefix}/perceptual_loss', perceptual_loss, **log_kwargs)
        self.log(f'{prefix}/weighted_perceptual_loss', perceptual_weight * perceptual_loss, **log_kwargs)
        self.log(
            f'{prefix}/perceptual_weight_effective',
            recon_raw.new_tensor(float(perceptual_weight)),
            **log_kwargs,
        )
        self.log(f'{prefix}/adversarial_generator_loss', adversarial_generator_loss, **log_kwargs)
        self.log(
            f'{prefix}/weighted_adversarial_generator_loss',
            adversarial_weight * adversarial_generator_loss,
            **log_kwargs,
        )
        self.log(
            f'{prefix}/adversarial_weight_effective',
            recon_raw.new_tensor(float(adversarial_weight)),
            **log_kwargs,
        )
        if psnr is not None:
            self.log(f'{prefix}/psnr', psnr, prog_bar=True, **log_kwargs)
        if is_audio:
            for name, value in audio_metrics.items():
                self.log(f'{prefix}/{name}', value, prog_bar=(name == "audio_lsd"), **log_kwargs)
            for name, value in audio_energy_metrics.items():
                self.log(f'{prefix}/{name}', value, prog_bar=False, **log_kwargs)
            if self.audio_energy_loss_weight > 0:
                self.log(f'{prefix}/weighted_audio_energy_loss', weighted_audio_energy_loss, **log_kwargs)
            for name, value in audio_multires_metrics.items():
                self.log(f'{prefix}/{name}', value, prog_bar=False, **log_kwargs)
            if self.audio_multires_loss_weight > 0:
                self.log(f'{prefix}/weighted_audio_multires_loss', weighted_audio_multires_loss, **log_kwargs)
            if is_waveform_audio and self.audio_waveform_l1_weight > 0:
                self.log(f'{prefix}/audio_waveform_l1_loss', waveform_l1_loss, **log_kwargs)
                self.log(f'{prefix}/weighted_audio_waveform_l1_loss', weighted_waveform_l1_loss, **log_kwargs)
            if is_waveform_audio and self.audio_multires_stft_loss_weight > 0:
                self.log(
                    f'{prefix}/weighted_audio_multires_stft_loss',
                    weighted_audio_multires_stft_loss,
                    **log_kwargs,
                )
            if is_waveform_audio and self.audio_mel_loss_weight > 0:
                self.log(f'{prefix}/audio_mel_loss', audio_mel_loss, **log_kwargs)
                self.log(f'{prefix}/weighted_audio_mel_loss', weighted_audio_mel_loss, **log_kwargs)
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
            self.log(f'{prefix}/coeff_active_abs_mean', diag.get("coeff_active_abs_mean", torch.tensor(0.0, device=x.device)), **log_kwargs)
            self.log(f'{prefix}/coeff_clip_frac', diag.get("coeff_clip_frac", torch.tensor(0.0, device=x.device)), **log_kwargs)
            coherence_max, coherence_mean_abs, coherence_rms = self.bottleneck.coherence_stats()
            self.log(f'{prefix}/dict_coherence', coherence_max, **log_kwargs)
            self.log(f'{prefix}/dict_coherence_mean_abs', coherence_mean_abs, **log_kwargs)
            self.log(f'{prefix}/dict_coherence_rms', coherence_rms, **log_kwargs)
        if ssim is not None:
            self.log(f'{prefix}/ssim', ssim, prog_bar=True, **log_kwargs)
        self.log(f'{prefix}/sparsity', sparsity, **log_kwargs)
        self.log(f'{prefix}/effective_sparsity', effective_sparsity, **log_kwargs)

        # Log bottleneck subcomponents for diagnostics
        if not self.bypass_bottleneck and self.bottleneck._last_e_latent_loss is not None:
            self.log(f'{prefix}/e_latent_loss', self.bottleneck._last_e_latent_loss, **log_kwargs)
        if not self.bypass_bottleneck and self.bottleneck._last_dl_latent_loss is not None:
            self.log(f'{prefix}/dl_latent_loss', self.bottleneck._last_dl_latent_loss, **log_kwargs)
        if not self.bypass_bottleneck and getattr(self.bottleneck, "_last_coeff_refine_loss", None) is not None:
            self.log(f'{prefix}/coeff_refine_loss', self.bottleneck._last_coeff_refine_loss, **log_kwargs)
        if not self.bypass_bottleneck and getattr(self.bottleneck, "_last_weighted_coeff_refine_loss", None) is not None:
            self.log(
                f'{prefix}/weighted_coeff_refine_loss',
                self.bottleneck._last_weighted_coeff_refine_loss,
                **log_kwargs,
            )
        if not self.bypass_bottleneck and getattr(self.bottleneck, "_last_coeff_posterior_std", None) is not None:
            self.log(
                f'{prefix}/coeff_posterior_std',
                self.bottleneck._last_coeff_posterior_std,
                **log_kwargs,
            )
        if not self.bypass_bottleneck and getattr(self.bottleneck, "_last_coeff_target_std", None) is not None:
            self.log(
                f'{prefix}/coeff_target_std',
                self.bottleneck._last_coeff_target_std,
                **log_kwargs,
            )

        # Occasional diagnostic logging to catch outlier batches
        if needs_train_diag:
            x_abs_max = torch.nan_to_num(x).abs().max()
            recon_abs_max = torch.nan_to_num(recon_raw).abs().max()
            coeff_abs_max = torch.nan_to_num(sparse_codes.values).abs().max()
            coeff_active_abs_mean = torch.nan_to_num(sparse_codes.values).abs().mean()
            coeff_abs_mean = self._dense_coeff_abs_mean(sparse_codes)
            coeff_clip_frac = diag.get("coeff_clip_frac", torch.tensor(0.0, device=x.device))
            nan_frac = (~torch.isfinite(recon_raw)).float().mean()
            diag_kwargs = dict(on_step=True, on_epoch=False, sync_dist=False, prog_bar=False)
            self.log('train/diag/input_abs_max', x_abs_max, **diag_kwargs)
            self.log('train/diag/recon_abs_max', recon_abs_max, **diag_kwargs)
            self.log('train/diag/coeff_abs_max', coeff_abs_max, **diag_kwargs)
            self.log('train/diag/coeff_abs_mean', coeff_abs_mean, **diag_kwargs)
            self.log('train/diag/coeff_active_abs_mean', coeff_active_abs_mean, **diag_kwargs)
            self.log('train/diag/coeff_clip_frac', coeff_clip_frac, **diag_kwargs)
            self.log('train/diag/recon_nan_frac', nan_frac, **diag_kwargs)
        
        if return_raw:
            return total_loss, recon_vis, x_vis, recon_raw, x
        return total_loss, recon_vis, x_vis

    def training_step(self, batch, batch_idx):
        """Training step.

        Two paths share the same loss/logging via ``compute_metrics``:
          * legacy (no discriminator): Lightning automatic optimization, where
            LR scheduling and dictionary gradient/K-SVD hooks run via
            ``optimizer_step``/``on_before_optimizer_step``;
          * adversarial (discriminator present): manual optimization, which
            re-implements those hooks inline around two optimizer steps.
        """
        if not self._adversarial_enabled:
            loss, recon, x = self.compute_metrics(batch, prefix='train')
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite train/loss at global_step={int(getattr(self, 'global_step', 0))} "
                    f"batch_idx={int(batch_idx)}"
                )
            audio_meta = extract_audio_metadata_from_batch(batch)
            if self._should_log_images(batch_idx, prefix='train'):
                self.log_images(x, recon, prefix='train', audio_meta=audio_meta)
                self._ddp_barrier_if_needed()
            return loss

        return self._adversarial_training_step(batch, batch_idx)

    def _adversarial_training_step(self, batch, batch_idx):
        """Manual-optimization step: update autoencoder, then discriminator."""
        step = int(self._manual_train_step.item())
        opt_ae, opt_disc = self.optimizers()

        # --- Autoencoder / generator update ---
        self._apply_scheduled_lrs(opt_ae, step=step, base_lrs=self._lr_base_lrs)
        self._adv_cache = None
        loss, recon, x = self.compute_metrics(batch, prefix='train')
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite train/loss at global_step={int(getattr(self, 'global_step', 0))} "
                f"batch_idx={int(batch_idx)}"
            )
        opt_ae.zero_grad(set_to_none=True)
        self.manual_backward(loss)
        # Mirror on_before_optimizer_step (skipped in manual mode): project the
        # dictionary gradient before stepping only the autoencoder optimizer.
        if not self.bypass_bottleneck:
            self.bottleneck.project_dictionary_gradient_()
        self._clip_manual_optimizer(self._raw_optimizer(opt_ae))
        opt_ae.step()
        # Mirror the automatic optimizer_step post-update dictionary maintenance.
        if not self.bypass_bottleneck:
            self.bottleneck.normalize_dictionary_()

        # --- Discriminator update (reuses the cached forward) ---
        if self._adv_cache is not None and step >= self.disc_start_step:
            self._apply_scheduled_lrs(opt_disc, step=step, base_lrs=self._disc_lr_base_lrs)
            d_loss, logits_real, logits_fake = self._discriminator_loss(
                self._adv_cache["real"].detach(),
                self._adv_cache["recon"].detach(),
            )
            factor = adopt_weight(self.disc_factor, step, self.disc_start_step)
            d_loss = factor * d_loss
            opt_disc.zero_grad(set_to_none=True)
            self.manual_backward(d_loss)
            self._clip_manual_optimizer(self._raw_optimizer(opt_disc))
            opt_disc.step()
            d_log = dict(on_step=True, on_epoch=False, sync_dist=True, batch_size=int(x.size(0)))
            self.log('train/disc_loss', d_loss, prog_bar=True, **d_log)
            self.log('train/logits_real', self._logits_mean(logits_real), **d_log)
            self.log('train/logits_fake', self._logits_mean(logits_fake), **d_log)
        self._adv_cache = None

        self._manual_train_step += 1

        if self._should_log_images(batch_idx, prefix='train'):
            self.log_images(
                x, recon, prefix='train', audio_meta=extract_audio_metadata_from_batch(batch)
            )
            self._ddp_barrier_if_needed()
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        if batch_idx == 0:
            self._maybe_store_val_batch(batch)
        loss, recon, x = self.compute_metrics(batch, prefix='val')
        audio_meta = extract_audio_metadata_from_batch(batch)
        
        # When latent visuals are enabled, batch 0 is logged in richer form at
        # validation epoch end, so skip the duplicate simple recon grid here.
        skip_simple_val_log = self._supports_val_latent_heatmaps() and batch_idx == 0
        if self._should_log_images(batch_idx, prefix='val') and not skip_simple_val_log:
            self.log_images(x, recon, prefix='val', audio_meta=audio_meta)
            self._ddp_barrier_if_needed()
        
        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        loss, recon, x = self.compute_metrics(batch, prefix='test')
        audio_meta = extract_audio_metadata_from_batch(batch)

        # Log images periodically
        if self._should_log_images(batch_idx, prefix='test'):
            self.log_images(x, recon, prefix='test', audio_meta=audio_meta)
            self._ddp_barrier_if_needed()
        
        return loss

    def on_test_epoch_end(self):
        """Compute FID at the end of test epoch."""
        if self.test_fid is not None:
            if fid_has_enough_samples(self.test_fid):
                fid_score = self.test_fid.compute()
                self.log('test/fid', fid_score, sync_dist=True)
            self.test_fid.reset()

    def configure_optimizers(self):
        """Configure optimizers with optional cosine LR schedule."""
        main_params = list(self.encoder.parameters()) + \
                      list(self.pre_bottleneck.parameters()) + \
                      list(self.post_bottleneck.parameters()) + \
                      list(self.decoder.parameters())
        bottleneck_aux_params = []
        dictionary_params = []
        if not self.bypass_bottleneck:
            for name, param in self.bottleneck.named_parameters():
                if not param.requires_grad:
                    continue
                if name == "dictionary" or name.startswith("separable_dictionary_factors."):
                    dictionary_params.append(param)
                else:
                    bottleneck_aux_params.append(param)

        param_groups = [
            {"params": main_params, "lr": self.learning_rate},
        ]
        if bottleneck_aux_params:
            param_groups.append({"params": bottleneck_aux_params, "lr": self.learning_rate})

        # Match proto.py: shared Adam, with an optional dictionary-specific LR.
        dict_lr = getattr(self.bottleneck, "dict_learning_rate", None)
        if dict_lr is None:
            dict_lr = self.learning_rate
        if dictionary_params:
            param_groups.append({"params": dictionary_params, "lr": dict_lr})

        optimizer = torch.optim.Adam(
            param_groups,
            betas=(self.beta, 0.999),
        )
        trainer = self._trainer_ref()
        total_steps, source = self._resolve_lr_total_steps(trainer)
        self._lr_total_steps = total_steps
        self._lr_total_steps_source = source
        self._lr_base_lrs = tuple(float(group["lr"]) for group in optimizer.param_groups)
        if not self._adversarial_enabled:
            return optimizer

        # Adversarial mode: a second optimizer drives the PatchGAN critic. The
        # cosine LR schedule (same horizon) is shared via _disc_lr_base_lrs.
        disc_lr = self.disc_learning_rate if self.disc_learning_rate is not None else self.learning_rate
        disc_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=disc_lr,
            betas=(self.beta, 0.999),
        )
        self._disc_lr_base_lrs = tuple(float(group["lr"]) for group in disc_optimizer.param_groups)
        return [optimizer, disc_optimizer]
    
    def _maybe_store_val_batch(self, batch):
        """Cache a small val batch (CPU) for visualization."""
        if not self._supports_val_latent_heatmaps():
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

    def _supports_val_latent_heatmaps(self):
        """Whether image-space validation latent heatmaps apply to this run."""
        return bool(self.enable_val_latent_visuals and not self.is_waveform_audio)

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self._val_vis_batch = None
        metric = self._ensure_val_rfid()
        if metric is not None:
            metric.reset()

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        if self.val_rfid is not None:
            # Skip compute() if no updates landed this epoch (e.g. audio runs
            # with compute_fid=true never call update — see fid_has_enough_samples).
            if fid_has_enough_samples(self.val_rfid):
                rfid_score = self.val_rfid.compute()
                self.log('val/rfid', rfid_score, sync_dist=True)
            self.val_rfid.reset()
        if self.enable_val_latent_visuals:
            self._log_val_latent_visuals()
            self._snapshot_dictionary()
            self._log_dict_scatter()
            self._ddp_barrier_if_needed()

    def on_fit_end(self):
        """Generate the full trajectory animation GIF once at the end of training."""
        if self.enable_val_latent_visuals:
            self._snapshot_dictionary()
            self._log_dict_scatter()
            self._generate_dict_animation()
            self._ddp_barrier_if_needed()

    def on_test_start(self):
        super().on_test_start()
        metric = self._ensure_test_fid()
        if metric is not None:
            metric.reset()
