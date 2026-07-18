"""Visualization concern for :class:`~src.models.laser.LASER`.

Extracted from ``laser.py`` as a mixin: these methods produce reconstruction
grids, codebook scatter/animation, and latent/error heatmaps for W&B logging.
They operate on the LASER instance's state via ``self`` and carry no state of
their own.
"""

import os
from pathlib import Path

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
        self._save_local_visual_image("val/dictionary_scatter", image, step=step)
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

        logger = getattr(self, "logger", None)
        if logger is None:
            return
        step = self._wandb_epoch_end_step()
        gif_path = self._visual_path(
            "val/dictionary_atom_trajectories",
            step=step,
            suffix=".gif",
        )
        delete_after = False
        if gif_path is None:
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
                gif_path = Path(tmp.name)
            delete_after = True
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
            log_wandb_video(
                logger,
                "val/dictionary_atom_trajectories",
                [str(saved)],
                step=step,
                captions=[f"dictionary trajectories step={step}"],
                formats=["gif"],
            )
        finally:
            if delete_after:
                try:
                    os.unlink(gif_path)
                except OSError:
                    pass

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
