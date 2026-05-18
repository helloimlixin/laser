"""Stage-2 sample previews: decode generated tokens and save/log media."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import lightning as pl
import torch

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
        image_item = str(direct)
        log_images = direct.is_file() and direct.suffix.lower() in {".png", ".jpg", ".jpeg"}
        if log_images:
            try:
                from PIL import Image
                import numpy as np

                with Image.open(direct) as img:
                    image_item = np.asarray(img.convert("RGB"))
            except Exception:
                image_item = str(direct)
            log_wandb_images(
                logger,
                "generation/samples",
                [image_item],
                step=step,
                captions=[cap],
            )
            log_wandb_images(
                logger,
                "s2/samples",
                [image_item],
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
    saver._log(trainer, step=step, epoch=None, direct=direct, batch=batch)
    if return_batch:
        return direct, batch, saver._cache
    return direct
