"""Stage-2 sample previews: decode generated tokens and save/log media."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import lightning as pl
import numpy as np
import torch
from PIL import Image

from src.audio_logging import (
    audio_config_from_source,
    build_generated_audio_log_payload,
)
from src.data.token_cache import load_token_cache
from src.s2 import sample as sample_s2, sample_slide, save_grid
from src.stage2_compat import (
    ensure_stage2_cache_metadata as add_cache_meta,
    load_stage1_decoder_bundle as load_s1,
)
from src.wandb_media import log_wandb_images, log_wandb_payload


def _prior_token_shape(mod: pl.LightningModule, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    prior_cfg = getattr(getattr(mod, "prior", None), "cfg", None)
    if prior_cfg is not None and all(hasattr(prior_cfg, key) for key in ("H", "W", "D")):
        return tuple(int(getattr(prior_cfg, key)) for key in ("H", "W", "D"))
    return tuple(int(v) for v in fallback)


def _sample_for_preview(
    mod: pl.LightningModule,
    s1,
    *,
    prior_shape: tuple[int, int, int],
    full_shape: tuple[int, int, int],
    n: int,
    temp: float,
    top_k: int,
    ctemp,
    cmode,
    dev: torch.device,
):
    """Sample full-size previews when a GPT prior was trained on latent crops."""
    prior = getattr(mod, "prior", None)
    prior_name = type(prior).__name__.lower() if prior is not None else ""
    full_shape = tuple(int(v) for v in full_shape)
    prior_shape = tuple(int(v) for v in prior_shape)
    if (
        prior is not None
        and "gpt" in prior_name
        and prior_shape[:2] != full_shape[:2]
        and prior_shape[2] == full_shape[2]
    ):
        return sample_slide(
            mod,
            s1,
            prior_shape,
            out_h=full_shape[0],
            out_w=full_shape[1],
            n=n,
            temp=temp,
            top_k=top_k,
            dev=dev,
        )
    return sample_s2(
        mod,
        s1,
        prior_shape,
        n=n,
        temp=temp,
        top_k=top_k,
        ctemp=ctemp,
        cmode=cmode,
        dev=dev,
    )


def _is_waveform_samples(samples: torch.Tensor) -> bool:
    return torch.is_tensor(samples) and samples.ndim == 3 and int(samples.size(1)) == 1


def _grid_uint8(images: list[np.ndarray], *, nrow: int) -> np.ndarray:
    if not images:
        raise ValueError("images must be non-empty")
    nrow = max(1, int(nrow))
    height, width, channels = images[0].shape
    rows = int(math.ceil(len(images) / float(nrow)))
    grid = np.full((rows * height, nrow * width, channels), 255, dtype=np.uint8)
    for idx, image in enumerate(images):
        row = idx // nrow
        col = idx % nrow
        grid[row * height: (row + 1) * height, col * width: (col + 1) * width] = image
    return grid


def _colorize_scalar_maps(
    maps: torch.Tensor,
    *,
    cmap_name: str,
    per_map: bool,
    value_min: Optional[float] = None,
    value_max: Optional[float] = None,
) -> list[np.ndarray]:
    maps = torch.nan_to_num(maps.detach().cpu().to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
    try:
        import matplotlib

        cmap = matplotlib.colormaps.get_cmap(cmap_name)
    except Exception:
        cmap = None

    out: list[np.ndarray] = []
    if maps.numel() == 0:
        return out
    if not per_map:
        global_min = float(maps.min().item()) if value_min is None else float(value_min)
        global_max = float(maps.max().item()) if value_max is None else float(value_max)
    for scalar_map in maps:
        if per_map:
            lo = float(scalar_map.min().item()) if value_min is None else float(value_min)
            hi = float(scalar_map.max().item()) if value_max is None else float(value_max)
        else:
            lo, hi = global_min, global_max
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            norm = torch.zeros_like(scalar_map)
        else:
            norm = ((scalar_map - lo) / (hi - lo)).clamp(0.0, 1.0)
        arr = norm.numpy()
        if cmap is not None:
            rgb = (cmap(arr)[..., :3] * 255.0).round().astype(np.uint8)
        else:
            gray = (arr * 255.0).round().astype(np.uint8)
            rgb = np.stack([gray, gray, gray], axis=-1)
        out.append(rgb)
    return out


def _save_scalar_map_grid(
    values: torch.Tensor,
    out_dir: Path,
    *,
    stem: str,
    max_items: int,
    max_depths: int,
    cmap_name: str,
    per_map: bool,
    value_min: Optional[float] = None,
    value_max: Optional[float] = None,
) -> Optional[Path]:
    if not torch.is_tensor(values) or values.ndim != 4:
        return None
    B, H, W, D = values.shape
    if B <= 0 or H <= 0 or W <= 0 or D <= 0:
        return None
    keep_b = min(int(max_items), int(B))
    keep_d = min(int(max_depths), int(D))
    maps = (
        values[:keep_b, :, :, :keep_d]
        .permute(0, 3, 1, 2)
        .reshape(keep_b * keep_d, H, W)
    )
    colored = _colorize_scalar_maps(
        maps,
        cmap_name=cmap_name,
        per_map=per_map,
        value_min=value_min,
        value_max=value_max,
    )
    if not colored:
        return None
    max_side = max(int(colored[0].shape[0]), int(colored[0].shape[1]))
    scale = max(1, int(math.ceil(96.0 / float(max(1, max_side)))))
    if scale > 1:
        resampling = getattr(getattr(Image, "Resampling", Image), "NEAREST", Image.NEAREST)
        colored = [
            np.asarray(
                Image.fromarray(image).resize(
                    (int(image.shape[1]) * scale, int(image.shape[0]) * scale),
                    resampling,
                )
            )
            for image in colored
        ]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.png"
    Image.fromarray(_grid_uint8(colored, nrow=keep_d)).save(path)
    return path


def _sparse_visual_tensors(
    batch,
    cache: Optional[dict],
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    atoms = getattr(batch, "atoms", None)
    coeffs = getattr(batch, "coeffs", None)
    coeff_bins = None
    if atoms is not None:
        return atoms, coeffs, coeff_bins

    tokens = getattr(batch, "toks", None)
    if tokens is None:
        return None, None, None
    tokens = tokens.detach()
    if tokens.ndim != 4:
        return None, None, None
    if int(tokens.size(-1)) >= 2 and int(tokens.size(-1)) % 2 == 0:
        atoms = tokens[..., 0::2]
        meta = cache.get("meta", {}) if isinstance(cache, dict) else {}
        atom_vocab = int(meta.get("num_atoms") or meta.get("atom_vocab_size") or 0)
        coeff_bins = tokens[..., 1::2]
        if atom_vocab > 0:
            coeff_bins = coeff_bins - atom_vocab
        return atoms, None, coeff_bins
    return tokens, None, None


def _build_sparse_visuals(batch, cache: Optional[dict], out_dir: Path, *, stem: str) -> dict[str, Path]:
    atoms, coeffs, coeff_bins = _sparse_visual_tensors(batch, cache)
    paths: dict[str, Path] = {}
    meta = cache.get("meta", {}) if isinstance(cache, dict) else {}
    atom_vocab = int(meta.get("num_atoms") or meta.get("atom_vocab_size") or 0)
    if atoms is not None:
        path = _save_scalar_map_grid(
            atoms,
            out_dir,
            stem=f"{stem}_atom_ids",
            max_items=4,
            max_depths=8,
            cmap_name="turbo",
            per_map=False,
            value_min=0.0,
            value_max=float(max(atom_vocab - 1, 1)) if atom_vocab > 0 else None,
        )
        if path is not None:
            paths["atom_id_maps"] = path
    if coeffs is not None:
        path = _save_scalar_map_grid(
            coeffs,
            out_dir,
            stem=f"{stem}_coeff_values",
            max_items=4,
            max_depths=8,
            cmap_name="coolwarm",
            per_map=True,
        )
        if path is not None:
            paths["coeff_value_maps"] = path
        path = _save_scalar_map_grid(
            coeffs.abs(),
            out_dir,
            stem=f"{stem}_coeff_abs",
            max_items=4,
            max_depths=8,
            cmap_name="magma",
            per_map=True,
        )
        if path is not None:
            paths["coeff_abs_maps"] = path
    if coeff_bins is not None:
        path = _save_scalar_map_grid(
            coeff_bins,
            out_dir,
            stem=f"{stem}_coeff_bins",
            max_items=4,
            max_depths=8,
            cmap_name="viridis",
            per_map=False,
        )
        if path is not None:
            paths["coeff_bin_maps"] = path
    return paths


def _sparse_generation_stats(batch, cache: Optional[dict]) -> dict[str, object]:
    atoms, coeffs, coeff_bins = _sparse_visual_tensors(batch, cache)
    stats: dict[str, object] = {}
    if atoms is not None:
        atoms_flat = atoms.detach().cpu().to(torch.float32).reshape(-1)
        if atoms_flat.numel() == 0:
            return stats
        stats["generation/atom_id_min"] = float(atoms_flat.min().item())
        stats["generation/atom_id_max"] = float(atoms_flat.max().item())
        unique_atoms = int(torch.unique(atoms_flat.to(torch.long)).numel())
        stats["generation/unique_atoms"] = unique_atoms
        meta = cache.get("meta", {}) if isinstance(cache, dict) else {}
        atom_vocab = int(meta.get("num_atoms") or meta.get("atom_vocab_size") or 0)
        if atom_vocab > 0:
            stats["generation/unique_atom_frac"] = float(unique_atoms) / float(atom_vocab)
        try:
            import wandb

            stats["generation/atom_id_hist"] = wandb.Histogram(atoms_flat.numpy())
        except Exception:
            pass
    if coeffs is not None:
        coeffs_flat = torch.nan_to_num(coeffs.detach().cpu().to(torch.float32).reshape(-1))
        if coeffs_flat.numel() > 0:
            stats["generation/coeff_mean"] = float(coeffs_flat.mean().item())
            stats["generation/coeff_std"] = float(coeffs_flat.std(unbiased=False).item())
            stats["generation/coeff_abs_mean"] = float(coeffs_flat.abs().mean().item())
            stats["generation/coeff_abs_max"] = float(coeffs_flat.abs().max().item())
            try:
                import wandb

                stats["generation/coeff_hist"] = wandb.Histogram(coeffs_flat.numpy())
            except Exception:
                pass
    if coeff_bins is not None:
        bins_flat = coeff_bins.detach().cpu().to(torch.float32).reshape(-1)
        if bins_flat.numel() > 0:
            stats["generation/coeff_bin_min"] = float(bins_flat.min().item())
            stats["generation/coeff_bin_max"] = float(bins_flat.max().item())
            try:
                import wandb

                stats["generation/coeff_bin_hist"] = wandb.Histogram(bins_flat.numpy())
            except Exception:
                pass
    return stats


def _load_image_for_wandb(path: Path):
    try:
        with Image.open(path) as img:
            return np.asarray(img.convert("RGB"))
    except Exception:
        return str(path)


def _log_sparse_visuals(logger, visual_paths: dict[str, Path], *, step: int, caption: str) -> None:
    for name, path in visual_paths.items():
        image = _load_image_for_wandb(path)
        log_wandb_images(
            logger,
            f"generation/{name}",
            [image],
            step=step,
            captions=[caption],
        )
        log_wandb_images(
            logger,
            f"s2/{name}",
            [image],
            step=step,
            captions=[caption],
        )


def _save_waveform_samples(samples: torch.Tensor, out_dir, *, stem: str, sample_rate: int) -> Path:
    out_root = Path(out_dir).expanduser().resolve() / "audio_samples" / stem
    out_root.mkdir(parents=True, exist_ok=True)
    waveforms = samples.detach().cpu().to(torch.float32)[:, 0].clamp(-1.0, 1.0)
    try:
        import numpy as np
        from scipy.io import wavfile

        for idx, waveform in enumerate(waveforms):
            pcm = waveform.numpy()
            pcm = np.clip(pcm, -1.0, 1.0)
            pcm = np.round(pcm * 32767.0).astype(np.int16)
            wavfile.write(out_root / f"{stem}_{idx:02d}.wav", int(sample_rate), pcm)
    except Exception:
        torch.save(waveforms, out_root / f"{stem}.pt")
    return out_root


class Stage2SamplePreviewCallback(pl.Callback):
    """Save decoded samples during stage-2 training without bloating the train script."""

    def __init__(
        self,
        *,
        cache_pt: str,
        out_dir: str,
        step_every: int = 0,
        epoch_every: int = 0,
        n: int = 4,
        temp: float = 1.0,
        top_k: int = 0,
        ctemp=None,
        cmode: Optional[str] = None,
        s1_root: str = "outputs",
        use_wandb: bool = False,
    ):
        super().__init__()
        self.cache_pt = str(cache_pt)
        self.out_dir = Path(out_dir).expanduser().resolve()
        self.step_every = max(0, int(step_every))
        self.epoch_every = max(0, int(epoch_every))
        self.n = max(1, int(n))
        self.temp = float(temp)
        self.top_k = int(top_k)
        self.ctemp = None if ctemp is None else float(ctemp)
        self.cmode = None if cmode is None else str(cmode).strip().lower()
        self.s1_root = str(s1_root)
        self.use_wandb = bool(use_wandb)
        self._last_step = -1
        self._last_epoch = -1
        self._cache = None
        self._s1 = None
        self._shape = None
        self._disabled_reason = None

    @staticmethod
    def _barrier_if_needed(trainer: pl.Trainer):
        if getattr(trainer, "world_size", 1) <= 1:
            return
        strategy = getattr(trainer, "strategy", None)
        barrier = getattr(strategy, "barrier", None)
        if callable(barrier):
            barrier()

    def _ready(self, dev: torch.device):
        if self._cache is None:
            # Cache metadata tells us both the token shape and how to recover the
            # matching stage-1 decoder for visual/audio previews.
            raw = load_token_cache(self.cache_pt)
            self._cache = add_cache_meta(
                raw,
                token_cache_path=self.cache_pt,
                output_root=self.s1_root,
            )
            self._shape = tuple(int(v) for v in self._cache["shape"])
        if self._s1 is None:
            self._s1 = load_s1(
                self._cache,
                token_cache_path=self.cache_pt,
                device=dev,
                output_root=self.s1_root,
            )
        else:
            self._s1.model = self._s1.model.to(dev)

    def _log(self, trainer: pl.Trainer, *, step: int, epoch: Optional[int], direct: Path, batch=None):
        if not self.use_wandb:
            return
        logger = getattr(trainer, "logger", None)
        if logger is None:
            return
        if epoch is not None:
            step = int(step) + 1
        step = int(step)

        bits = [f"step={step}"]
        if epoch is not None:
            bits.append(f"epoch={epoch}")
        cap = " ".join(bits)
        log_images = direct.is_file() and direct.suffix.lower() in {".png", ".jpg", ".jpeg"}
        if log_images:
            log_wandb_images(
                logger,
                "generation/samples",
                [_load_image_for_wandb(direct)],
                step=step,
                captions=[cap],
            )
            log_wandb_images(
                logger,
                "s2/samples",
                [_load_image_for_wandb(direct)],
                step=step,
                captions=[cap],
            )
        payload = {
            "generation/epoch": epoch,
            "s2/epoch": epoch,
        }
        if batch is not None and self._cache is not None:
            cache_meta = self._cache.get("meta", {}) if isinstance(self._cache, dict) else {}
            audio_meta = self._cache.get("audio_meta") if isinstance(self._cache, dict) else None
            payload.update(
                build_generated_audio_log_payload(
                    batch.imgs,
                    audio_source=cache_meta,
                    audio_meta=audio_meta,
                    split="generation",
                    max_items=min(4, int(batch.imgs.size(0))),
                    artifact_dir=self.out_dir,
                )
            )
            payload.update(_sparse_generation_stats(batch, self._cache))
        log_wandb_payload(logger, payload, step=step)

    @torch.no_grad()
    def _save(self, trainer: pl.Trainer, mod: pl.LightningModule, *, step: int, epoch: Optional[int] = None):
        if self._disabled_reason is not None:
            return
        dev = mod.device
        try:
            self._ready(dev)
        except FileNotFoundError as err:
            self._disabled_reason = str(err)
            print(f"Warning: disabling s2 sample previews; stage-1 decoder could not be loaded ({err})")
            return
        shape = _prior_token_shape(mod, self._shape)
        full_shape = tuple(int(v) for v in self._shape)
        was_training = bool(mod.training)
        mod.eval()
        try:
            batch = _sample_for_preview(
                mod,
                self._s1,
                prior_shape=shape,
                full_shape=full_shape,
                n=self.n,
                temp=self.temp,
                top_k=self.top_k,
                ctemp=self.ctemp,
                cmode=self.cmode,
                dev=dev,
            )
        finally:
            if was_training:
                mod.train()

        stem = f"s{step:07d}" if epoch is None else f"e{epoch:03d}_s{step:07d}"
        if _is_waveform_samples(batch.imgs):
            cache_meta = self._cache.get("meta", {}) if isinstance(self._cache, dict) else {}
            sample_rate = int(audio_config_from_source(cache_meta)["sample_rate"])
            direct = _save_waveform_samples(batch.imgs, self.out_dir, stem=stem, sample_rate=sample_rate)
        else:
            direct = save_grid(batch.imgs, self.out_dir, stem=stem)
            visual_paths = _build_sparse_visuals(batch, self._cache, self.out_dir, stem=stem)
            if self.use_wandb and trainer.is_global_zero:
                logger = getattr(trainer, "logger", None)
                log_step = int(step) + 1 if epoch is not None else int(step)
                cap = (
                    f"step={int(step)}"
                    if epoch is None
                    else f"step={log_step} epoch={int(epoch)}"
                )
                _log_sparse_visuals(logger, visual_paths, step=log_step, caption=cap)
        self._log(trainer, step=step, epoch=epoch, direct=direct, batch=batch)
        if epoch is None:
            print(f"Saved s2 samples at step {step}: {direct}")
        else:
            print(f"Saved s2 samples at epoch {epoch}, step {step}: {direct}")

    def on_train_batch_end(self, trainer, mod, outputs, batch, batch_idx):
        if self.step_every <= 0:
            return
        step = int(trainer.global_step)
        if step <= 0 or step == self._last_step or (step % self.step_every) != 0:
            return
        if trainer.is_global_zero:
            self._save(trainer, mod, step=step)
        self._barrier_if_needed(trainer)
        self._last_step = step

    def on_train_epoch_end(self, trainer, mod):
        if self.epoch_every <= 0:
            return
        epoch = int(trainer.current_epoch) + 1
        step = int(trainer.global_step)
        if epoch <= 0 or epoch == self._last_epoch or (epoch % self.epoch_every) != 0:
            return
        if step <= 0 or step == self._last_step:
            return
        if trainer.is_global_zero:
            self._save(trainer, mod, step=step, epoch=epoch)
        self._barrier_if_needed(trainer)
        self._last_epoch = epoch
        self._last_step = step

@torch.no_grad()
def save_final_generation_preview(
    *,
    trainer: pl.Trainer,
    mod: pl.LightningModule,
    cache_pt: str,
    out_dir: str,
    n: int,
    temp: float,
    top_k: int,
    ctemp,
    cmode,
    s1_root: str,
    use_wandb: bool,
    return_batch: bool = False,
):
    saver = Stage2SamplePreviewCallback(
        cache_pt=cache_pt,
        out_dir=out_dir,
        n=n,
        temp=temp,
        top_k=top_k,
        ctemp=ctemp,
        cmode=cmode,
        s1_root=s1_root,
        use_wandb=use_wandb,
    )
    saver._ready(mod.device)
    shape = _prior_token_shape(mod, saver._shape)
    full_shape = tuple(int(v) for v in saver._shape)
    was_training = bool(mod.training)
    mod.eval()
    try:
        batch = _sample_for_preview(
            mod,
            saver._s1,
            prior_shape=shape,
            full_shape=full_shape,
            n=saver.n,
            temp=saver.temp,
            top_k=saver.top_k,
            ctemp=saver.ctemp,
            cmode=saver.cmode,
            dev=mod.device,
        )
    finally:
        if was_training:
            mod.train()
    step = max(1, int(getattr(trainer, "global_step", 0) or 0))
    stem = f"final_s{step:07d}"
    if _is_waveform_samples(batch.imgs):
        cache_meta = saver._cache.get("meta", {}) if isinstance(saver._cache, dict) else {}
        sample_rate = int(audio_config_from_source(cache_meta)["sample_rate"])
        direct = _save_waveform_samples(batch.imgs, saver.out_dir, stem=stem, sample_rate=sample_rate)
    else:
        direct = save_grid(batch.imgs, saver.out_dir, stem=stem)
        visual_paths = _build_sparse_visuals(batch, saver._cache, saver.out_dir, stem=stem)
        if use_wandb:
            _log_sparse_visuals(
                getattr(trainer, "logger", None),
                visual_paths,
                step=step,
                caption=f"step={step}",
            )
    saver._log(trainer, step=step, epoch=None, direct=direct, batch=batch)
    if return_batch:
        return direct, batch, saver._cache
    return direct
