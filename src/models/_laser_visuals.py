"""Visualization concern for :class:`~src.models.laser.LASER`.

Extracted from ``laser.py`` as a mixin: these methods produce reconstruction
grids, codebook scatter/animation, and latent/error heatmaps for W&B logging.
They operate on the LASER instance's state via ``self`` and carry no state of
their own.
"""

import os

import torch
import torch.nn.functional as F
import torchvision

from src.audio_logging import build_audio_log_payload
from src.codebook_visuals import (
    render_codebook_scatter,
    save_codebook_trajectory_gif,
    select_codebook_vectors,
)
from src.wandb_media import log_wandb_images, log_wandb_payload, log_wandb_video


class VisualsMixin:
    """Image/codebook/heatmap visualization methods for LASER."""

    def _snapshot_dictionary(self):
        """Store a copy of the current dictionary atoms for trajectory animation.

        Called once per validation epoch (not every N training steps) to keep
        the snapshot list short and the final GIF fast to render.
        """
        if not self.enable_val_latent_visuals or not self._is_log_rank_zero():
            return
        with torch.no_grad():
            atoms = self.bottleneck.dictionary.detach().t().cpu()
            atoms = select_codebook_vectors(atoms, self.codebook_visual_max_vectors)
        step = int(getattr(self, "global_step", 0) or 0)
        if self._dict_snapshot_steps and self._dict_snapshot_steps[-1] == step:
            return
        self._dict_snapshots.append(atoms)
        self._dict_snapshot_steps.append(step)

    def _log_dict_scatter(self):
        """Log a static PCA scatter of current dictionary atoms (cheap, once per val epoch)."""
        if not self.enable_val_latent_visuals or not self._is_log_rank_zero():
            return
        if not self._dict_snapshots:
            return
        logger = getattr(self, "logger", None)
        if logger is None:
            return
        image = render_codebook_scatter(
            self._dict_snapshots,
            self._dict_snapshot_steps,
            title="Dictionary Atoms (PCA)",
        )
        if image is None:
            return

        step = self._wandb_epoch_end_step()
        log_wandb_images(
            logger,
            "val/dictionary_scatter",
            [image],
            step=step,
            captions=[f"dictionary scatter step={step}"],
        )

    def _generate_dict_animation(self):
        """Build the full trajectory GIF from all accumulated snapshots.

        Called once at the end of training (on_fit_end), not every val epoch.
        """
        if not self.enable_val_latent_visuals or not self._is_log_rank_zero():
            return
        if len(self._dict_snapshots) < 2:
            return
        import tempfile, os

        logger = getattr(self, "logger", None)
        if logger is None:
            return
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
            gif_path = tmp.name
        try:
            saved = save_codebook_trajectory_gif(
                self._dict_snapshots,
                self._dict_snapshot_steps,
                gif_path,
                title="Dictionary Atom Trajectories (PCA)",
                fps=2,
            )
            if saved is None:
                return
            step = self._wandb_epoch_end_step()
            log_wandb_video(
                logger,
                "val/dictionary_atom_trajectories",
                [str(saved)],
                step=step,
                captions=[f"dictionary trajectories step={step}"],
                formats=["gif"],
            )
        finally:
            try:
                os.unlink(gif_path)
            except OSError:
                pass

    def log_images(self, x, recon, prefix='val', max_images=8, audio_meta=None):
        """Log reconstruction images to wandb."""
        # Only log from rank zero in DDP to avoid multi-process logger contention
        if not self._is_log_rank_zero():
            return
        logger = getattr(self, "logger", None)
        if logger is None:
            return
        step = self._wandb_step()
        if not self._claim_media_log(prefix, step):
            return
        
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
        
        # Create image grids
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
        x_grid = torchvision.utils.make_grid(x_disp, nrow=min(8, max_images), normalize=False)
        recon_grid = torchvision.utils.make_grid(recon_disp, nrow=min(8, max_images), normalize=False)
        
        # Sanitize NaN/Inf and clamp to [0,1] before converting to numpy
        x_grid = torch.nan_to_num(x_grid, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        recon_grid = torch.nan_to_num(recon_grid, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        
        # Convert to display-friendly arrays. W&B accepts 2D grayscale arrays,
        # which keeps single-channel spectrogram logging simple and explicit.
        x_grid = self._wandb_display_array(x_grid)
        recon_grid = self._wandb_display_array(recon_grid)
        
        log_wandb_images(
            logger,
            f"{prefix}/images",
            [x_grid, recon_grid],
            step=step,
            captions=["Original", "Reconstructed"],
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

    def _log_val_latent_visuals(self):
        """Log latent RGB projections, sparse heatmaps, and reconstruction diagnostics."""
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
                z = self.encoder(x)
                z = self.pre_bottleneck(z)
                with self._bottleneck_autocast_context(z):
                    z_dl, _, sparse_codes = self.bottleneck(z.float())
                recon = self._apply_output_activation(self.decoder(self.post_bottleneck(z_dl)))
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
                log_wandb_payload(self.logger, log_payload, step=step)
        finally:
            if was_training:
                self.train()
            self._val_vis_batch = None
