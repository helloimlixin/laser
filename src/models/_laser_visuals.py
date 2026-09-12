"""Visualization concern for :class:`~src.models.laser.LASER`.

Extracted from ``laser.py`` as a mixin: these methods produce reconstruction
grids, dictionary diagnostics, and latent/error heatmaps for W&B logging.
They operate on the LASER instance's state via ``self`` and carry no state of
their own.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from src.audio_logging import _cached_erb_filterbank, build_audio_log_payload
from src.codebook_visuals import (
    _figure_to_rgb_array,
    render_dictionary_diagnostics,
    select_codebook_vectors,
)
from src.wandb_media import log_wandb_images, log_wandb_payload


class VisualsMixin:
    """Image/codebook/heatmap visualization methods for LASER."""

    _LEGACY_WANDB_VISUAL_KEYS = (
        "val/latent_activation_map",
        "val/sparse_support_tracks",
        "val/sparse_coefficient_tracks",
        "val/sparse_codec_dashboard",
        "val/dictionary_atom_scatter",
        "val/dictionary_movement_scatter",
        "val/dictionary_scatter",
        "val/dictionary_atom_trajectories",
        "val/visual_support_jaccard_mean",
        "val/visual_support_retained_fraction_mean",
        "val/visual_support_retained_mass_mean",
        "val/visual_support_entered_atoms_per_frame",
        "val/visual_support_run_length_frames_p50",
        "val/visual_support_run_length_frames_p90",
    )

    def _remove_legacy_wandb_visual_summaries(self):
        """Prevent removed media from reappearing when W&B resumes a run."""
        if not self._is_log_rank_zero():
            return
        experiment = getattr(getattr(self, "logger", None), "experiment", None)
        summary = getattr(experiment, "summary", None)
        if summary is None:
            return
        for key in self._LEGACY_WANDB_VISUAL_KEYS:
            try:
                del summary[key]
            except (AttributeError, KeyError, TypeError):
                continue

    def _visual_split(self, key, split=None):
        if split not in (None, ""):
            return str(split)
        text = str(key)
        return text.split("/", 1)[0] if "/" in text else "misc"

    def _visual_name(self, key):
        text = str(key).replace("/", "_")
        cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
        return cleaned.strip("_") or "visual"

    def _visual_step(self, step):
        try:
            value = int(step)
        except (TypeError, ValueError):
            value = int(getattr(self, "global_step", 0) or 0)
        return max(value, 0)

    def _visual_root(self, key, split=None):
        logger = getattr(self, "logger", None)
        trainer = self._trainer_ref()
        candidates = (
            getattr(logger, "save_dir", None),
            getattr(getattr(logger, "experiment", None), "dir", None),
            getattr(trainer, "default_root_dir", None),
        )
        base = next((item for item in candidates if item not in (None, "")), ".")
        root = Path(base).expanduser().resolve() / "visual_media" / self._visual_split(key, split)
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            print(f"[LASER] Could not create local visual artifact dir {root}: {exc}", flush=True)
            return None
        return root

    def _visual_path(self, key, *, step=None, suffix=".png", index=None, split=None):
        root = self._visual_root(key, split=split)
        if root is None:
            return None
        stem = f"step_{self._visual_step(step):09d}_{self._visual_name(key)}"
        if index is not None:
            stem = f"{stem}_{int(index):02d}"
        return root / f"{stem}{suffix}"

    def _save_local_visual_image(self, key, image, *, step=None, index=None, split=None):
        path = self._visual_path(key, step=step, suffix=".png", index=index, split=split)
        if path is None:
            return None
        try:
            import numpy as np
            from PIL import Image

            if torch.is_tensor(image):
                array = image.detach().cpu().numpy()
            else:
                array = np.asarray(image)
            if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
                array = np.moveaxis(array, 0, -1)
            if array.ndim == 3 and array.shape[-1] == 4:
                array = array[..., :3]
            if array.ndim == 3 and array.shape[-1] == 1:
                array = array[..., 0]
            if array.ndim not in (2, 3):
                return None
            array = np.nan_to_num(array)
            if np.issubdtype(array.dtype, np.floating):
                if float(array.max(initial=0.0)) <= 1.0 and float(array.min(initial=0.0)) >= 0.0:
                    array = array * 255.0
                array = np.clip(array, 0.0, 255.0).astype("uint8")
            else:
                array = np.clip(array, 0, 255).astype("uint8")
            Image.fromarray(array).save(path)
            return path
        except Exception as exc:
            print(f"[LASER] Could not save local visual {path}: {exc}", flush=True)
            return None

    def _save_local_visual_payload(self, payload, *, step=None):
        for key, value in dict(payload).items():
            if not isinstance(value, dict) or str(value.get("kind", "")).lower() != "image":
                continue
            for idx, item in enumerate(list(value.get("items", []) or [])):
                self._save_local_visual_image(key, item, step=step, index=idx)

    def _snapshot_dictionary(self):
        """Keep aligned initial/latest atom samples for movement statistics.

        Only two snapshots are retained, so validation count cannot grow the
        visualization memory footprint.
        """
        if not self.enable_val_latent_visuals or not self._is_log_rank_zero():
            return
        with torch.no_grad():
            atoms = self.bottleneck.dictionary.detach().t().cpu()
            atoms = select_codebook_vectors(atoms, self.codebook_visual_max_vectors)
        step = int(getattr(self, "global_step", 0) or 0)
        if self._dict_snapshot_steps and self._dict_snapshot_steps[-1] == step:
            return
        if not self._dict_snapshots:
            self._dict_snapshots.append(atoms)
            self._dict_snapshot_steps.append(step)
        elif len(self._dict_snapshots) == 1:
            self._dict_snapshots.append(atoms)
            self._dict_snapshot_steps.append(step)
        else:
            self._dict_snapshots[-1] = atoms
            self._dict_snapshot_steps[-1] = step

    def _dictionary_visual_subset(self):
        """Select atoms while retaining every validation-active atom when possible."""
        atoms = self.bottleneck.dictionary.detach().t().cpu().to(torch.float32)
        count = int(atoms.size(0))
        limit = min(int(self.codebook_visual_max_vectors), count)
        usage = getattr(self, "_val_visual_atom_usage", None)
        contribution = getattr(self, "_val_visual_atom_contribution", None)
        if not torch.is_tensor(usage) or int(usage.numel()) != count:
            usage = torch.zeros(count, dtype=torch.float32)
        else:
            usage = usage.detach().cpu().to(torch.float32).reshape(-1)
        if not torch.is_tensor(contribution) or int(contribution.numel()) != count:
            contribution = torch.zeros(count, dtype=torch.float32)
        else:
            contribution = contribution.detach().cpu().to(torch.float32).reshape(-1)

        active = torch.nonzero(usage > 0, as_tuple=False).flatten()
        if int(active.numel()) >= limit:
            active_score = contribution.index_select(0, active)
            order = torch.argsort(active_score, descending=True, stable=True)
            atom_ids = active.index_select(0, order[:limit])
        else:
            chosen = torch.zeros(count, dtype=torch.bool)
            chosen[active] = True
            remaining = limit - int(active.numel())
            grid = torch.linspace(0, count - 1, steps=max(limit * 2, 1)).round().to(torch.long)
            filler = grid[~chosen.index_select(0, grid)]
            if int(filler.numel()) < remaining:
                filler = torch.nonzero(~chosen, as_tuple=False).flatten()
            atom_ids = torch.cat([active, filler[:remaining]], dim=0)
        atom_ids = torch.unique(atom_ids, sorted=True)
        return (
            atoms.index_select(0, atom_ids),
            usage.index_select(0, atom_ids),
            contribution.index_select(0, atom_ids),
            atom_ids,
        )

    def _log_dict_diagnostics(self):
        """Log actual-space similarity, load, and aligned movement summaries."""
        if not self.enable_val_latent_visuals or not self._is_log_rank_zero():
            return
        if not self._dict_snapshots:
            return
        logger = getattr(self, "logger", None)
        if logger is None:
            return
        atoms, usage, contribution, atom_ids = self._dictionary_visual_subset()
        step = self._wandb_epoch_end_step()
        movement_snapshots = None
        movement_steps = None
        if len(self._dict_snapshots) > 1:
            movement_snapshots = (self._dict_snapshots[0], self._dict_snapshots[-1])
            movement_steps = (self._dict_snapshot_steps[0], self._dict_snapshot_steps[-1])
        image = render_dictionary_diagnostics(
            atoms,
            usage,
            contribution,
            atom_ids=atom_ids,
            step=step,
            movement_snapshots=movement_snapshots,
            movement_steps=movement_steps,
            title="LASER dictionary",
        )
        if image is None:
            return
        self._save_local_visual_image("val/dictionary_diagnostics", image, step=step)
        log_wandb_images(
            logger,
            "val/dictionary_diagnostics",
            [image],
            step=step,
            captions=[
                "Actual-space cosine similarity, nearest-neighbor redundancy, "
                "load concentration, and aligned angular drift"
            ],
        )

    def log_images(self, x, recon, prefix='val', max_images=8, audio_meta=None, step=None):
        """Log reconstruction images to wandb."""
        # Only log from rank zero in DDP to avoid multi-process logger contention
        if not self._is_log_rank_zero():
            return
        logger = getattr(self, "logger", None)
        if logger is None:
            return
        # Deduplicate against the model/trainer step, not W&B's mutable internal
        # step. W&B advances its own counter after a media log, while gradient
        # accumulation can call this method multiple times before the trainer's
        # optimizer step changes.
        requested_step = int(self.global_step if step is None else step)
        if not self._claim_media_log(prefix, requested_step):
            return
        step = self._wandb_step(requested_step=requested_step)
        
        # Take a small fixed subset to keep W&B logging cheap.
        x = x[:max_images]
        recon = recon[:max_images]

        dm = getattr(self._trainer_ref(), "datamodule", None)
        if torch.is_tensor(x) and x.ndim == 3:
            payload = {
                f"{prefix}/reconstruction_error": F.mse_loss(recon, x).item(),
            }
            if audio_meta is not None and dm is not None and hasattr(dm, "config"):
                payload.update(
                    build_audio_log_payload(
                        x,
                        recon,
                        audio_meta=audio_meta,
                        audio_source=dm.config,
                        split=prefix,
                        max_items=min(4, max_images),
                        artifact_dir=(
                            getattr(self.logger, "save_dir", None)
                            or getattr(self._trainer_ref(), "default_root_dir", None)
                        ),
                    )
                )
            log_wandb_payload(logger, payload, step=step)
            return
        
        # De-normalize using datamodule config if available; otherwise assume [-1,1] → [0,1]
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
        # Stack originals on top of reconstructions per-item (channels last), then a single grid.
        # Logging one image (not a list of two) avoids Lightning's WandbLogger creating a panel
        # per list element under the same key.
        stacked = torch.cat([x_disp, recon_disp], dim=2)  # [B, C, 2*H, W]
        combined = torchvision.utils.make_grid(
            stacked, nrow=min(8, max_images), normalize=False
        )
        combined = torch.nan_to_num(combined, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        combined = self._wandb_display_array(combined)

        self._save_local_visual_image(
            f"{prefix}/reconstruction_grid",
            combined,
            step=step,
            split=prefix,
        )
        log_wandb_images(
            logger,
            f"{prefix}/reconstruction_grid",
            [combined],
            step=step,
            captions=["Originals (top) / Reconstructions (bottom)"],
        )
        payload = {
            f"{prefix}/reconstruction_error": F.mse_loss(recon, x).item(),
        }
        if audio_meta is not None and dm is not None and hasattr(dm, "config"):
            payload.update(
                build_audio_log_payload(
                    x,
                    recon,
                    audio_meta=audio_meta,
                    audio_source=dm.config,
                    split=prefix,
                    max_items=min(4, max_images),
                    artifact_dir=getattr(self.logger, "save_dir", None) or getattr(self._trainer_ref(), "default_root_dir", None),
                )
            )
        log_wandb_payload(logger, payload, step=step)

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

    def _wandb_display_array(self, tensor: torch.Tensor):
        tensor = torch.nan_to_num(tensor.detach().cpu(), nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        if tensor.dim() != 3:
            raise ValueError(f"Expected CHW tensor for display, got shape {tuple(tensor.shape)}")
        if tensor.shape[0] == 1:
            return tensor[0].numpy()
        return tensor.numpy().transpose(1, 2, 0)

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

    def _waveform_sparse_visual_payload(
        self,
        z_latent,
        sparse_codes,
        *,
        encoder_latent=None,
        waveform=None,
        reconstruction=None,
        sample_rate=None,
        duration_seconds=None,
        split="val",
    ):
        """Render perceptually grounded audio reconstruction diagnostics.

        Sparse support tracks, coefficient tracks, and latent activation maps
        are intentionally absent: their coordinate systems are not perceptual
        and made adjacent validation examples difficult to compare. Sparse-code
        statistics are still accumulated for the separate dictionary summary,
        while this dashboard shows only audible-domain evidence.
        """
        if z_latent.ndim != 4 or int(z_latent.size(2)) != 1:
            raise ValueError(
                "Expected waveform bottleneck latent [B,C,1,T], got "
                f"{tuple(z_latent.shape)}"
            )
        if encoder_latent is None:
            encoder_latent = z_latent
        if encoder_latent.shape != z_latent.shape:
            raise ValueError(
                "encoder_latent must match the sparse latent shape; got "
                f"{tuple(encoder_latent.shape)} vs {tuple(z_latent.shape)}"
            )

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if waveform is None or reconstruction is None:
            raise ValueError("waveform and reconstruction are required for audio diagnostics")
        if waveform.ndim != 3 or reconstruction.ndim != 3:
            raise ValueError(
                "Expected waveform and reconstruction [B,1,T], got "
                f"{tuple(waveform.shape)} and {tuple(reconstruction.shape)}"
            )
        sample_rate = max(
            1,
            int(sample_rate or getattr(self, "audio_sample_rate", 0) or 24_000),
        )
        limit = min(
            4,
            int(z_latent.size(0)),
            int(waveform.size(0)),
            int(reconstruction.size(0)),
        )
        frames = int(z_latent.size(-1))
        num_embeddings = max(int(getattr(sparse_codes, "num_embeddings", 1)), 1)
        global_usage = torch.zeros(num_embeddings, dtype=torch.float32)
        global_contribution = torch.zeros(num_embeddings, dtype=torch.float32)
        dashboard_items = []
        captions = []
        all_abs_coefficients = []

        for idx in range(limit):
            support = (
                sparse_codes.support[idx, 0]
                .detach()
                .to(torch.long)
                .cpu()
                .transpose(0, 1)
            )
            coefficients = (
                sparse_codes.values[idx, 0]
                .detach()
                .to(torch.float32)
                .cpu()
                .transpose(0, 1)
            )

            flat_support = support.reshape(-1).clamp(0, num_embeddings - 1)
            flat_abs = coefficients.abs().reshape(-1)
            usage = torch.bincount(flat_support, minlength=num_embeddings).to(torch.float32)
            contribution = torch.zeros(num_embeddings, dtype=torch.float32)
            contribution.index_add_(0, flat_support, flat_abs)
            global_usage.add_(usage)
            global_contribution.add_(contribution)
            all_abs_coefficients.append(coefficients.abs().reshape(-1))

            target = waveform[idx, 0].detach().to(torch.float32).cpu()
            estimate = reconstruction[idx, 0].detach().to(torch.float32).cpu()
            sample_count = min(int(target.numel()), int(estimate.numel()))
            target = target[:sample_count]
            estimate = estimate[:sample_count]
            audio_duration = sample_count / float(sample_rate)
            audio_time = np.arange(sample_count, dtype=np.float32) / float(sample_rate)
            audio_residual = estimate - target

            max_fft = min(1024, max(16, sample_count))
            n_fft = 1 << int(np.floor(np.log2(max_fft)))
            hop = max(1, n_fft // 4)
            window = torch.hann_window(n_fft, dtype=torch.float32)
            target_mag = torch.stft(
                target,
                n_fft=n_fft,
                hop_length=hop,
                win_length=n_fft,
                window=window,
                center=True,
                pad_mode="constant",
                return_complex=True,
            ).abs()
            estimate_mag = torch.stft(
                estimate,
                n_fft=n_fft,
                hop_length=hop,
                win_length=n_fft,
                window=window,
                center=True,
                pad_mode="constant",
                return_complex=True,
            ).abs()
            reference = target_mag.max().clamp_min(1.0e-7)
            target_db = (20.0 * torch.log10(target_mag.clamp_min(reference * 1.0e-4) / reference)).clamp(-80.0, 0.0)
            estimate_db = (20.0 * torch.log10(estimate_mag.clamp_min(reference * 1.0e-4) / reference)).clamp(-80.0, 12.0)
            spectral_delta_db = (estimate_db - target_db).clamp(-24.0, 24.0)

            num_bands = min(32, max(8, n_fft // 8))
            erb_filterbank = _cached_erb_filterbank(
                sample_rate=sample_rate,
                n_fft=n_fft,
                num_bands=num_bands,
                min_frequency=30.0,
                max_frequency=0.5 * sample_rate,
                device=target_mag.device,
                dtype=target_mag.dtype,
            )
            target_band_energy = (erb_filterbank @ target_mag.square()).clamp_min(1.0e-8)
            estimate_band_energy = (erb_filterbank @ estimate_mag.square()).clamp_min(1.0e-8)
            band_delta_db_raw = (
                10.0 * torch.log10(estimate_band_energy / target_band_energy)
            )
            band_delta_db = band_delta_db_raw.clamp(-18.0, 18.0)
            band_centers_hz = (
                erb_filterbank * torch.linspace(0.0, 0.5 * sample_rate, n_fft // 2 + 1)
            ).sum(dim=1) / erb_filterbank.sum(dim=1).clamp_min(1.0e-8)

            eps = 1.0e-8
            snr_db = 10.0 * torch.log10(
                target.square().mean().clamp_min(eps)
                / audio_residual.square().mean().clamp_min(eps)
            )
            critical_ratio = estimate_band_energy.sum() / target_band_energy.sum().clamp_min(eps)
            critical_log_mae_db = band_delta_db_raw.abs().mean()

            fig = plt.figure(figsize=(15, 13), constrained_layout=True)
            grid = fig.add_gridspec(4, 2, height_ratios=(0.8, 1.0, 1.0, 0.9))
            ax_waveform = fig.add_subplot(grid[0, :])
            ax_target_spec = fig.add_subplot(grid[1, 0])
            ax_estimate_spec = fig.add_subplot(grid[1, 1])
            ax_spectral_error = fig.add_subplot(grid[2, 0])
            ax_band_error = fig.add_subplot(grid[2, 1])
            ax_envelope = fig.add_subplot(grid[3, 0])
            ax_band_profile = fig.add_subplot(grid[3, 1])

            ax_waveform.plot(audio_time, target.numpy(), label="reference", lw=1.0, alpha=0.85)
            ax_waveform.plot(audio_time, estimate.numpy(), label="reconstruction", lw=0.9, alpha=0.78)
            ax_waveform.plot(audio_time, audio_residual.numpy(), label="error", lw=0.75, alpha=0.65)
            ax_waveform.set_xlim(0.0, max(audio_duration, 1.0e-6))
            ax_waveform.set_ylabel("amplitude")
            ax_waveform.set_xlabel("time (s)")
            ax_waveform.set_title(f"Waveform alignment — reconstruction SNR {float(snr_db):.2f} dB")
            ax_waveform.grid(True, alpha=0.2)
            ax_waveform.legend(ncol=3, loc="upper right", fontsize=8)

            target_image = ax_target_spec.imshow(
                target_db.numpy(),
                aspect="auto",
                origin="lower",
                interpolation="bilinear",
                cmap="magma",
                vmin=-80.0,
                vmax=0.0,
                extent=(0.0, audio_duration, 0.0, 0.5 * sample_rate / 1000.0),
            )
            ax_target_spec.set_ylabel("frequency (kHz)")
            ax_target_spec.set_xlabel("time (s)")
            ax_target_spec.set_title("Reference log spectrum")
            target_colorbar = fig.colorbar(
                target_image, ax=ax_target_spec, fraction=0.045, pad=0.02
            )
            target_colorbar.set_label("dB relative to reference peak")

            estimate_image = ax_estimate_spec.imshow(
                estimate_db.numpy(),
                aspect="auto",
                origin="lower",
                interpolation="bilinear",
                cmap="magma",
                vmin=-80.0,
                vmax=0.0,
                extent=(0.0, audio_duration, 0.0, 0.5 * sample_rate / 1000.0),
            )
            ax_estimate_spec.set_ylabel("frequency (kHz)")
            ax_estimate_spec.set_xlabel("time (s)")
            ax_estimate_spec.set_title("Reconstruction log spectrum (same scale)")
            estimate_colorbar = fig.colorbar(
                estimate_image, ax=ax_estimate_spec, fraction=0.045, pad=0.02
            )
            estimate_colorbar.set_label("dB relative to reference peak")

            spectral_error_image = ax_spectral_error.imshow(
                spectral_delta_db.numpy(),
                aspect="auto",
                origin="lower",
                interpolation="bilinear",
                cmap="RdBu_r",
                vmin=-24.0,
                vmax=24.0,
                extent=(0.0, audio_duration, 0.0, 0.5 * sample_rate / 1000.0),
            )
            ax_spectral_error.set_ylabel("frequency (kHz)")
            ax_spectral_error.set_xlabel("time (s)")
            ax_spectral_error.set_title("Signed spectral error (blue=missing, red=excess)")
            spectral_error_colorbar = fig.colorbar(
                spectral_error_image,
                ax=ax_spectral_error,
                fraction=0.045,
                pad=0.02,
            )
            spectral_error_colorbar.set_label("reconstruction − reference (dB)")

            band_error_image = ax_band_error.imshow(
                band_delta_db.numpy(),
                aspect="auto",
                origin="lower",
                interpolation="bilinear",
                cmap="RdBu_r",
                vmin=-18.0,
                vmax=18.0,
                extent=(0.0, audio_duration, 0.0, float(num_bands)),
            )
            band_tick_ids = np.linspace(0, num_bands - 1, num=min(6, num_bands)).round().astype(int)
            ax_band_error.set_yticks(band_tick_ids + 0.5)
            ax_band_error.set_yticklabels(
                [f"{float(band_centers_hz[i]) / 1000.0:.2g}" for i in band_tick_ids]
            )
            ax_band_error.set_ylabel("ERB-band center (kHz)")
            ax_band_error.set_xlabel("time (s)")
            ax_band_error.set_title(
                f"Critical-band energy error — mean |Δ| {float(critical_log_mae_db):.2f} dB"
            )
            band_error_colorbar = fig.colorbar(
                band_error_image, ax=ax_band_error, fraction=0.045, pad=0.02
            )
            band_error_colorbar.set_label("reconstruction − reference (dB)")

            envelope_window = min(sample_count, max(1, int(round(0.025 * sample_rate))))
            envelope_hop = min(envelope_window, max(1, int(round(0.010 * sample_rate))))

            def rms_envelope(signal):
                return F.avg_pool1d(
                    signal.square().view(1, 1, -1),
                    kernel_size=envelope_window,
                    stride=envelope_hop,
                ).sqrt().flatten()

            target_envelope = rms_envelope(target)
            estimate_envelope = rms_envelope(estimate)
            residual_envelope = rms_envelope(audio_residual)
            envelope_time = (
                torch.arange(target_envelope.numel()) * envelope_hop
                + 0.5 * envelope_window
            ) / float(sample_rate)
            ax_envelope.plot(
                envelope_time.numpy(), target_envelope.numpy(), label="reference", lw=1.5
            )
            ax_envelope.plot(
                envelope_time.numpy(), estimate_envelope.numpy(), label="reconstruction", lw=1.35
            )
            ax_envelope.plot(
                envelope_time.numpy(), residual_envelope.numpy(), label="error", lw=1.1
            )
            ax_envelope.set_xlim(0.0, max(audio_duration, 1.0e-6))
            ax_envelope.set_xlabel("time (s)")
            ax_envelope.set_ylabel("25 ms RMS")
            ax_envelope.set_title("Short-time energy envelope")
            ax_envelope.legend(fontsize=8)
            ax_envelope.grid(True, alpha=0.2)

            target_band_mean = target_band_energy.mean(dim=1)
            estimate_band_mean = estimate_band_energy.mean(dim=1)
            band_reference = target_band_mean.max().clamp_min(eps)
            target_band_db = 10.0 * torch.log10(
                target_band_mean.clamp_min(band_reference * 1.0e-8) / band_reference
            )
            estimate_band_db = 10.0 * torch.log10(
                estimate_band_mean.clamp_min(band_reference * 1.0e-8) / band_reference
            )
            ax_band_profile.semilogx(
                band_centers_hz.numpy(), target_band_db.numpy(), label="reference", lw=1.6
            )
            ax_band_profile.semilogx(
                band_centers_hz.numpy(), estimate_band_db.numpy(), label="reconstruction", lw=1.45
            )
            ax_band_profile.axhline(-50.0, color="0.5", ls=":", lw=1.0, label="loss activity floor")
            ax_band_profile.set_ylim(-80.0, 5.0)
            ax_band_profile.set_xlabel("ERB-band center frequency (Hz)")
            ax_band_profile.set_ylabel("mean energy (dB rel. reference peak)")
            ax_band_profile.set_title(
                f"Critical-band profile — total energy ratio {float(critical_ratio):.3f}"
            )
            ax_band_profile.legend(fontsize=8)
            ax_band_profile.grid(True, which="both", alpha=0.2)

            fig.suptitle(
                f"Audio reconstruction diagnostic — item {idx} | {sample_rate / 1000.0:.0f} kHz | "
                f"{audio_duration:.2f} s",
                fontsize=13,
            )
            dashboard_items.append(_figure_to_rgb_array(fig))
            plt.close(fig)
            captions.append(
                f"item={idx}; snr_db={float(snr_db):.4f}; "
                f"critical_band_energy_ratio={float(critical_ratio):.4f}; "
                f"critical_band_mean_abs_delta_db={float(critical_log_mae_db):.4f}"
            )

        self._val_visual_atom_usage = global_usage
        self._val_visual_atom_contribution = global_contribution
        payload = {}
        if dashboard_items:
            payload[f"{split}/audio_reconstruction_diagnostics"] = {
                "kind": "image",
                "items": dashboard_items,
                "caption": captions,
            }
            payload[f"{split}/visual_active_atom_count"] = float(
                (global_usage > 0).sum().item()
            )
            payload[f"{split}/visual_latent_frames"] = float(frames)
        if all_abs_coefficients:
            flattened = torch.cat(all_abs_coefficients)
            payload[f"{split}/visual_coefficient_abs_p95"] = float(
                torch.quantile(flattened, 0.95).item()
            )
        return payload

    def _log_val_latent_visuals(self):
        """Log reconstruction diagnostics and image-model latent summaries."""
        self._remove_legacy_wandb_visual_summaries()
        if not self._supports_val_latent_heatmaps():
            return
        if not getattr(self._trainer_ref(), "is_global_zero", False):
            return
        if not getattr(self, "logger", None) or not hasattr(self.logger, "experiment"):
            return
        if self._val_vis_batch is None:
            return
        x_cpu, _ = self._val_vis_batch
        x = x_cpu.to(self.device)
        was_training = bool(self.training)
        self.eval()
        try:
            with torch.no_grad():
                if self.is_waveform_audio:
                    z_e = self.encoder(x)
                    z_e = self.pre_bottleneck(z_e)
                    z_e = self._to_bottleneck_input(z_e)
                    with self._bottleneck_autocast_context(z_e):
                        z_dl, _, sparse_codes = self.bottleneck(z_e.float())
                else:
                    z_e = None
                    z_dl, _, sparse_codes = self.encode(x)
                recon = self.decode(z_dl)
                if self.is_waveform_audio:
                    sample_rate = max(int(getattr(self, "audio_sample_rate", 0) or 0), 1)
                    log_payload = self._waveform_sparse_visual_payload(
                        z_dl,
                        sparse_codes,
                        encoder_latent=z_e,
                        waveform=x,
                        reconstruction=recon,
                        sample_rate=sample_rate,
                        duration_seconds=float(x.size(-1)) / float(sample_rate),
                        split="val",
                    )
                    if log_payload:
                        step = self._wandb_epoch_end_step()
                        self._save_local_visual_payload(log_payload, step=step)
                        log_wandb_payload(self.logger, log_payload, step=step)
                    return
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
                orig_np = self._wandb_display_array(x_disp[idx])
                recon_np = self._wandb_display_array(recon_disp[idx])
                latent_img = latent_rgb[idx]
                sparse_rgb = cmap(sparse_heat[idx])[..., :3]
                error_rgb = cmap(error_heat[idx])[..., :3]
                cap = f"idx={idx}"
                log_payload.setdefault(
                    "val/original",
                    {"kind": "image", "items": [], "caption": []},
                )
                log_payload["val/original"]["items"].append(orig_np)
                log_payload["val/original"]["caption"].append(cap)
                log_payload.setdefault(
                    "val/reconstruction",
                    {"kind": "image", "items": [], "caption": []},
                )
                log_payload["val/reconstruction"]["items"].append(recon_np)
                log_payload["val/reconstruction"]["caption"].append(cap)
                log_payload.setdefault(
                    "val/latent_rgb",
                    {"kind": "image", "items": [], "caption": []},
                )
                log_payload["val/latent_rgb"]["items"].append(latent_img)
                log_payload["val/latent_rgb"]["caption"].append(cap)
                log_payload.setdefault(
                    "val/sparse_heatmap",
                    {"kind": "image", "items": [], "caption": []},
                )
                log_payload["val/sparse_heatmap"]["items"].append(sparse_rgb)
                log_payload["val/sparse_heatmap"]["caption"].append(cap)
                log_payload.setdefault(
                    "val/recon_error_map",
                    {"kind": "image", "items": [], "caption": []},
                )
                log_payload["val/recon_error_map"]["items"].append(error_rgb)
                log_payload["val/recon_error_map"]["caption"].append(cap)
            if log_payload:
                step = self._wandb_epoch_end_step()
                self._save_local_visual_payload(log_payload, step=step)
                log_wandb_payload(self.logger, log_payload, step=step)
        finally:
            if was_training:
                self.train()
            self._val_vis_batch = None
