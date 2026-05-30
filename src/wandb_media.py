from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence


def log_wandb_metrics(
    logger: Any,
    metrics: Mapping[str, Any],
    *,
    step: Optional[int] = None,
) -> None:
    filtered = {str(key): value for key, value in dict(metrics).items() if value is not None}
    if not filtered or logger is None:
        return
    if hasattr(logger, "log_metrics"):
        logger.log_metrics(filtered, step=step)
        return
    experiment = getattr(logger, "experiment", None)
    if experiment is None or not hasattr(experiment, "log"):
        return
    payload = dict(filtered)
    if step is not None:
        payload.setdefault("trainer/global_step", int(step))
    experiment.log(payload)


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
    caption_list = None if captions is None else list(captions)
    if caption_list is not None and len(caption_list) != len(image_list):
        raise ValueError(f"Expected {len(image_list)} captions for {key}, found {len(caption_list)}")

    # Go straight to wandb.experiment.log to avoid Lightning's WandbLogger creating
    # a panel-per-list-element under the same key. A single log call with one payload
    # produces exactly one W&B card per key, regardless of list length.
    import wandb

    payload = {
        str(key): [
            wandb.Image(image, caption=None if caption_list is None else caption_list[idx])
            for idx, image in enumerate(image_list)
        ]
    }
    if step is not None:
        payload["trainer/global_step"] = int(step)
    experiment = getattr(logger, "experiment", None)
    if experiment is not None and hasattr(experiment, "log"):
        experiment.log(payload)


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
    sample_rate_list = [int(rate) for rate in sample_rates]
    if len(sample_rate_list) != len(audio_list):
        raise ValueError(f"Expected {len(audio_list)} sample rates for {key}, found {len(sample_rate_list)}")
    caption_list = None if captions is None else list(captions)
    if caption_list is not None and len(caption_list) != len(audio_list):
        raise ValueError(f"Expected {len(audio_list)} captions for {key}, found {len(caption_list)}")

    if hasattr(logger, "log_audio"):
        kwargs = {"sample_rate": sample_rate_list}
        if caption_list is not None:
            kwargs["caption"] = caption_list
        logger.log_audio(str(key), audio_list, step=step, **kwargs)
        return

    import wandb

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
        experiment.log(payload)


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
    caption_list = None if captions is None else list(captions)
    if caption_list is not None and len(caption_list) != len(video_list):
        raise ValueError(f"Expected {len(video_list)} captions for {key}, found {len(caption_list)}")
    format_list = None if formats is None else list(formats)
    if format_list is not None and len(format_list) != len(video_list):
        raise ValueError(f"Expected {len(video_list)} formats for {key}, found {len(format_list)}")
    fps_list = None if fps is None else [None if item is None else int(item) for item in fps]
    if fps_list is not None and len(fps_list) != len(video_list):
        raise ValueError(f"Expected {len(video_list)} fps values for {key}, found {len(fps_list)}")

    if hasattr(logger, "log_video"):
        kwargs = {}
        if caption_list is not None:
            kwargs["caption"] = caption_list
        if format_list is not None:
            kwargs["format"] = format_list
        if fps_list is not None:
            kwargs["fps"] = fps_list
        logger.log_video(str(key), video_list, step=step, **kwargs)
        return

    import wandb

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
        experiment.log(payload)


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
