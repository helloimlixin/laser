from __future__ import annotations

import errno
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


_WARNED_STORAGE_ERRORS: set[str] = set()
_MEDIA_DISABLED = str(os.environ.get("LASER_DISABLE_WANDB_MEDIA", "")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _warn_once(message: str) -> None:
    if message in _WARNED_STORAGE_ERRORS:
        return
    _WARNED_STORAGE_ERRORS.add(message)
    print(f"[LASER] {message}", file=sys.stderr, flush=True)


def _is_storage_error(exc: BaseException) -> bool:
    if isinstance(exc, OSError):
        return exc.errno in {
            errno.ENOSPC,
            errno.EDQUOT,
            errno.EXDEV,
        }
    text = str(exc).lower()
    return "disk quota exceeded" in text or "no space left on device" in text


def _media_disabled() -> bool:
    return _MEDIA_DISABLED


def _disable_media_after_storage_error(context: str) -> None:
    global _MEDIA_DISABLED
    _MEDIA_DISABLED = True
    _warn_once(f"Disabling W&B media logging after storage error in {context}.")


def _handle_storage_error(context: str, exc: BaseException, *, disable_media: bool = False) -> bool:
    if not _is_storage_error(exc):
        return False
    _warn_once(f"Skipping W&B {context} after storage error: {exc}")
    if disable_media:
        _disable_media_after_storage_error(context)
    return True


def _monotonic_wandb_step(logger: Any, step: Optional[int]) -> Optional[int]:
    if step is None:
        return None
    experiment = getattr(logger, "experiment", None)
    current = getattr(experiment, "step", None)
    try:
        if current is not None and int(step) < int(current):
            return None
    except (TypeError, ValueError):
        pass
    return int(step)


def _wandb_media_tmp_parent(logger: Any) -> Optional[Path]:
    for candidate in (
        getattr(logger, "save_dir", None),
        getattr(getattr(logger, "experiment", None), "dir", None),
    ):
        if candidate in (None, ""):
            continue
        try:
            path = Path(candidate).expanduser().resolve()
        except (OSError, RuntimeError):
            continue
        parent = path if path.name != "files" else path.parent
        return parent / ".wandb_media_tmp"
    return None


def _configure_wandb_media_tmp(wandb: Any, logger: Any) -> None:
    if _media_disabled():
        return
    parent = _wandb_media_tmp_parent(logger)
    if parent is None:
        return
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        if _handle_storage_error("media temp setup", exc, disable_media=True):
            return
        raise

    try:
        from wandb.sdk.data_types import _private as wandb_private
        from wandb.sdk.data_types import audio as wandb_audio

        current = getattr(wandb_private, "MEDIA_TMP", None)
        current_name = Path(getattr(current, "name", "") or "")
        if current_name.parent == parent:
            return
        media_tmp = tempfile.TemporaryDirectory(suffix="wandb-media", dir=str(parent))
        wandb_private.MEDIA_TMP = media_tmp
        wandb_audio.MEDIA_TMP = media_tmp
    except OSError as exc:
        if _handle_storage_error("media temp setup", exc, disable_media=True):
            return
        raise
    except Exception as exc:
        _warn_once(f"Could not relocate W&B media temp dir: {exc}")


def log_wandb_metrics(
    logger: Any,
    metrics: Mapping[str, Any],
    *,
    step: Optional[int] = None,
) -> None:
    filtered = {str(key): value for key, value in dict(metrics).items() if value is not None}
    if not filtered or logger is None:
        return
    log_step = _monotonic_wandb_step(logger, step)
    if hasattr(logger, "log_metrics"):
        try:
            logger.log_metrics(filtered, step=log_step)
        except Exception as exc:
            if _handle_storage_error("metrics log", exc):
                return
            raise
        return
    experiment = getattr(logger, "experiment", None)
    if experiment is None or not hasattr(experiment, "log"):
        return
    payload = dict(filtered)
    if step is not None:
        payload.setdefault("trainer/global_step", int(step))
    try:
        if log_step is None:
            experiment.log(payload)
        else:
            experiment.log(payload, step=log_step)
    except Exception as exc:
        if _handle_storage_error("metrics log", exc):
            return
        raise


def log_wandb_images(
    logger: Any,
    key: str,
    images: Sequence[Any],
    *,
    step: Optional[int] = None,
    captions: Optional[Sequence[Optional[str]]] = None,
) -> None:
    image_list = list(images)
    if not image_list or logger is None:
        return
    if _media_disabled():
        _warn_once(f"Skipping W&B image log for {key}; media logging is disabled.")
        return
    caption_list = None if captions is None else list(captions)
    if caption_list is not None and len(caption_list) != len(image_list):
        raise ValueError(f"Expected {len(image_list)} captions for {key}, found {len(caption_list)}")

    # Go straight to wandb.experiment.log to avoid Lightning's WandbLogger creating
    # a panel-per-list-element under the same key. A single log call with one payload
    # produces exactly one W&B card per key, regardless of list length.
    import wandb

    _configure_wandb_media_tmp(wandb, logger)

    wandb_images = [
        wandb.Image(image, caption=None if caption_list is None else caption_list[idx])
        for idx, image in enumerate(image_list)
    ]
    payload = {str(key): wandb_images[0] if len(wandb_images) == 1 else wandb_images}
    if step is not None:
        payload["trainer/global_step"] = int(step)
    experiment = getattr(logger, "experiment", None)
    if experiment is not None and hasattr(experiment, "log"):
        log_step = _monotonic_wandb_step(logger, step)
        if step is not None and log_step is None:
            return
        try:
            if log_step is None:
                experiment.log(payload)
            else:
                experiment.log(payload, step=log_step)
        except Exception as exc:
            if _handle_storage_error(f"image log for {key}", exc, disable_media=True):
                return
            raise


def log_wandb_audio(
    logger: Any,
    key: str,
    audios: Sequence[Any],
    *,
    sample_rates: Sequence[int],
    step: Optional[int] = None,
    captions: Optional[Sequence[Optional[str]]] = None,
) -> None:
    audio_list = list(audios)
    if not audio_list or logger is None:
        return
    if _media_disabled():
        _warn_once(f"Skipping W&B audio log for {key}; media logging is disabled.")
        return
    sample_rate_list = [int(rate) for rate in sample_rates]
    if len(sample_rate_list) != len(audio_list):
        raise ValueError(f"Expected {len(audio_list)} sample rates for {key}, found {len(sample_rate_list)}")
    caption_list = None if captions is None else list(captions)
    if caption_list is not None and len(caption_list) != len(audio_list):
        raise ValueError(f"Expected {len(audio_list)} captions for {key}, found {len(caption_list)}")

    import wandb

    _configure_wandb_media_tmp(wandb, logger)

    payload = {
        str(key): [
            wandb.Audio(
                audio,
                sample_rate=sample_rate_list[idx],
                caption=None if caption_list is None else caption_list[idx],
            )
            for idx, audio in enumerate(audio_list)
        ]
    }
    if step is not None:
        payload["trainer/global_step"] = int(step)
    experiment = getattr(logger, "experiment", None)
    if experiment is not None and hasattr(experiment, "log"):
        log_step = _monotonic_wandb_step(logger, step)
        if step is not None and log_step is None:
            return
        try:
            if log_step is None:
                experiment.log(payload)
            else:
                experiment.log(payload, step=log_step)
        except Exception as exc:
            if _handle_storage_error(f"audio log for {key}", exc, disable_media=True):
                return
            raise


def log_wandb_video(
    logger: Any,
    key: str,
    videos: Sequence[Any],
    *,
    step: Optional[int] = None,
    captions: Optional[Sequence[Optional[str]]] = None,
    formats: Optional[Sequence[Optional[str]]] = None,
    fps: Optional[Sequence[Optional[int]]] = None,
) -> None:
    video_list = list(videos)
    if not video_list or logger is None:
        return
    if _media_disabled():
        _warn_once(f"Skipping W&B video log for {key}; media logging is disabled.")
        return
    caption_list = None if captions is None else list(captions)
    if caption_list is not None and len(caption_list) != len(video_list):
        raise ValueError(f"Expected {len(video_list)} captions for {key}, found {len(caption_list)}")
    format_list = None if formats is None else list(formats)
    if format_list is not None and len(format_list) != len(video_list):
        raise ValueError(f"Expected {len(video_list)} formats for {key}, found {len(format_list)}")
    fps_list = None if fps is None else [None if item is None else int(item) for item in fps]
    if fps_list is not None and len(fps_list) != len(video_list):
        raise ValueError(f"Expected {len(video_list)} fps values for {key}, found {len(fps_list)}")

    import wandb

    _configure_wandb_media_tmp(wandb, logger)

    payload = {
        str(key): [
            wandb.Video(
                video,
                caption=None if caption_list is None else caption_list[idx],
                format=None if format_list is None else format_list[idx],
                fps=None if fps_list is None else fps_list[idx],
            )
            for idx, video in enumerate(video_list)
        ]
    }
    if step is not None:
        payload["trainer/global_step"] = int(step)
    experiment = getattr(logger, "experiment", None)
    if experiment is not None and hasattr(experiment, "log"):
        log_step = _monotonic_wandb_step(logger, step)
        if step is not None and log_step is None:
            return
        try:
            if log_step is None:
                experiment.log(payload)
            else:
                experiment.log(payload, step=log_step)
        except Exception as exc:
            if _handle_storage_error(f"video log for {key}", exc, disable_media=True):
                return
            raise


def log_wandb_payload(
    logger: Any,
    payload: Mapping[str, Any],
    *,
    step: Optional[int] = None,
) -> None:
    metrics = {}
    for key, value in dict(payload).items():
        if isinstance(value, Mapping) and "kind" in value and "items" in value:
            kind = str(value["kind"]).strip().lower()
            items = list(value.get("items", []) or [])
            if not items:
                continue
            if kind == "image":
                log_wandb_images(
                    logger,
                    str(key),
                    items,
                    step=step,
                    captions=value.get("caption"),
                )
                continue
            if kind == "audio":
                log_wandb_audio(
                    logger,
                    str(key),
                    items,
                    step=step,
                    captions=value.get("caption"),
                    sample_rates=value.get("sample_rate", []),
                )
                continue
            if kind == "video":
                log_wandb_video(
                    logger,
                    str(key),
                    items,
                    step=step,
                    captions=value.get("caption"),
                    formats=value.get("format"),
                    fps=value.get("fps"),
                )
                continue
            raise ValueError(f"Unsupported W&B media kind: {kind!r}")
        metrics[str(key)] = value

    log_wandb_metrics(logger, metrics, step=step)
