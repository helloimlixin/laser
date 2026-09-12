"""Checkpoint, logging, and runtime helpers shared by both stages."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Iterable

def _stage1_fit_checkpoint_kwargs(ckpt_path):
    """Build Lightning fit kwargs for a fresh fit or trusted full-state resume."""
    kwargs = {"ckpt_path": ckpt_path}
    if ckpt_path:
        kwargs["weights_only"] = False
    return kwargs


def _dataset_key(name: object) -> str:
    return str(name or "").strip().lower().replace("-", "_")


def _configure_training_tempdir(output_dir: object) -> Path:
    """Choose a short writable temp path for checkpoints and worker IPC.

    Python multiprocessing appends its own socket directory and listener name
    below ``TMPDIR``.  Long NFS project paths can therefore exceed Linux's
    108-byte AF_UNIX limit before the first dataloader batch is delivered.
    """
    configured_tmp = str(os.environ.get("TMPDIR") or "").strip()
    if configured_tmp:
        tmpdir = Path(configured_tmp).expanduser().resolve()
    elif Path("/workspace").is_dir():
        digest = hashlib.sha1(str(Path(str(output_dir)).expanduser().resolve()).encode("utf-8")).hexdigest()[:12]
        tmpdir = Path("/workspace/tmp/laser") / digest
    else:
        digest = hashlib.sha1(str(Path(str(output_dir)).expanduser().resolve()).encode("utf-8")).hexdigest()[:12]
        tmpdir = Path("/tmp/laser") / digest
    os.environ["TMPDIR"] = str(tmpdir)
    os.environ["TEMP"] = str(tmpdir)
    os.environ["TMP"] = str(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(tmpdir)
    return tmpdir


def _optional_container_for_cli_tests(value):
    if value is None:
        return None
    try:
        from omegaconf import OmegaConf
    except Exception:
        return value
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _cfg_attr(obj, name: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _sample_text_prompts(cfg) -> list[str]:
    train_ar = _cfg_attr(cfg, "train_ar")
    prompts = _optional_container_for_cli_tests(_cfg_attr(train_ar, "sample_text_prompts"))
    if prompts:
        return list(prompts)
    ar = _cfg_attr(cfg, "ar")
    prompts = _optional_container_for_cli_tests(_cfg_attr(ar, "sample_text_prompts"))
    return list(prompts or [])


def _dedupe_tags(tags: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for tag in tags:
        text = str(tag).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _selected_checkpoint_paths(checkpoint_callback) -> list[Path]:
    """Return monitored top-k checkpoints plus last.ckpt, preserving order."""
    paths: list[Path] = []
    best_k = getattr(checkpoint_callback, "best_k_models", None) or {}
    if best_k:
        mode = str(getattr(checkpoint_callback, "mode", "min") or "min").lower()

        def _score(item):
            score = item[1]
            detach = getattr(score, "detach", None)
            if callable(detach):
                score = detach()
            cpu = getattr(score, "cpu", None)
            if callable(cpu):
                score = cpu()
            item_method = getattr(score, "item", None)
            if callable(item_method):
                return float(item_method())
            return float(score)

        paths.extend(
            Path(path)
            for path, _ in sorted(
                best_k.items(),
                key=_score,
                reverse=(mode == "max"),
            )
        )
    best_model_path = str(getattr(checkpoint_callback, "best_model_path", "") or "").strip()
    if best_model_path:
        paths.append(Path(best_model_path))
    last_model_path = str(getattr(checkpoint_callback, "last_model_path", "") or "").strip()
    if last_model_path:
        paths.append(Path(last_model_path))

    selected: list[Path] = []
    seen = set()
    for path in paths:
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        selected.append(resolved)
    return selected


def _refresh_last_checkpoint(trainer, checkpoint_callback) -> None:
    """Persist the trainer's current state before publishing selected checkpoints.

    Lightning's ``every_n_epochs`` checkpoint schedule does not refresh
    ``last.ckpt`` for a mid-epoch validation. Our runs validate several times
    per epoch, so an upload made there could otherwise label a stale checkpoint
    as ``last``. ``Trainer.save_checkpoint`` is deliberately called on every
    rank, as required by distributed checkpoint strategies; only rank zero
    writes for standard DDP.
    """
    if not bool(getattr(checkpoint_callback, "save_last", False)):
        return
    save_checkpoint = getattr(trainer, "save_checkpoint", None)
    if not callable(save_checkpoint):
        return
    directory_raw = str(getattr(checkpoint_callback, "dirpath", "") or "").strip()
    if not directory_raw:
        return
    directory = Path(directory_raw).expanduser()
    last_path = directory / "last.ckpt"
    save_checkpoint(str(last_path))
    checkpoint_callback.last_model_path = str(last_path)


def _make_selected_checkpoint_artifact_callback(callback_base):
    class SelectedCheckpointArtifactCallback(callback_base):
        """Upload top-k model checkpoints plus last.ckpt as one W&B artifact."""

        def __init__(
            self,
            checkpoint_callback,
            *,
            artifact_prefix: str = "model",
            every_n_epochs: int = 1,
        ):
            super().__init__()
            self.checkpoint_callback = checkpoint_callback
            self.artifact_prefix = str(artifact_prefix or "model")
            self.every_n_epochs = max(1, int(every_n_epochs or 1))
            self._last_signature = None

        def _signature(self, paths: list[Path]) -> tuple:
            # W&B computes its own content digest in ``artifact.add_file``.
            # Re-hashing up to four ~350 MB checkpoints here doubled synchronous
            # NFS reads at every validation. Nanosecond mtime + inode + size is
            # sufficient for detecting Lightning's atomic checkpoint rewrites;
            # W&B remains the content-addressed source of truth for the upload.
            signature = []
            for path in paths:
                stat = path.stat()
                signature.append((path.name, stat.st_size, stat.st_mtime_ns, stat.st_ino))
            return tuple(signature)

        def _upload(self, trainer, *, reason: str) -> None:
            if not bool(getattr(trainer, "is_global_zero", True)):
                return
            logger = getattr(trainer, "logger", None)
            experiment = getattr(logger, "experiment", None)
            if experiment is None or not hasattr(experiment, "log_artifact"):
                return
            if bool(getattr(self.checkpoint_callback, "save_last", False)):
                last_path = str(getattr(self.checkpoint_callback, "last_model_path", "") or "")
                if not last_path or not Path(last_path).is_file():
                    # A fork resumed at an upload boundary can still point at
                    # the parent run's best checkpoint before its first local
                    # validation. Never publish that incomplete set as latest.
                    return
            paths = _selected_checkpoint_paths(self.checkpoint_callback)
            if not paths:
                return
            signature = self._signature(paths)
            if signature == self._last_signature:
                return

            import wandb

            run_id = str(getattr(experiment, "id", "") or getattr(logger, "version", "") or "run")
            artifact = wandb.Artifact(
                name=f"{self.artifact_prefix}-{run_id}-selected-checkpoints",
                type="model",
                description="Automatically uploaded selected checkpoints: monitored top-k plus last.ckpt.",
                metadata={
                    "reason": reason,
                    "epoch": int(getattr(trainer, "current_epoch", -1)),
                    "global_step": int(getattr(trainer, "global_step", -1)),
                    "monitor": str(getattr(self.checkpoint_callback, "monitor", "") or ""),
                    "mode": str(getattr(self.checkpoint_callback, "mode", "") or ""),
                    "save_top_k": int(getattr(self.checkpoint_callback, "save_top_k", -1)),
                    "save_last": bool(getattr(self.checkpoint_callback, "save_last", False)),
                    "checkpoint_paths": [str(path) for path in paths],
                },
            )
            for path in paths:
                artifact.add_file(str(path), name=path.name)
            epoch = int(getattr(trainer, "current_epoch", 0))
            aliases = ["latest", "best-plus-last", f"epoch-{epoch:03d}"]
            experiment.log_artifact(artifact, aliases=aliases)
            self._last_signature = signature

        def on_validation_end(self, trainer, pl_module) -> None:
            if bool(getattr(trainer, "sanity_checking", False)):
                return
            # ModelCheckpoint callbacks are deliberately run after ordinary
            # callbacks by Lightning, regardless of list order. Refresh the
            # rolling checkpoint here, but defer publishing until the next
            # epoch starts so ``best_k_models`` includes this validation.
            _refresh_last_checkpoint(trainer, self.checkpoint_callback)

        def on_train_epoch_start(self, trainer, pl_module) -> None:
            epoch = int(getattr(trainer, "current_epoch", 0))
            if epoch <= 0:
                return
            completed_epoch = epoch - 1
            if (completed_epoch + 1) % self.every_n_epochs != 0:
                return
            self._upload(trainer, reason="post_checkpoint_validation")

        def on_train_end(self, trainer, pl_module) -> None:
            _refresh_last_checkpoint(trainer, self.checkpoint_callback)
            self._upload(trainer, reason="train_end")

    return SelectedCheckpointArtifactCallback


def _make_selected_checkpoint_file_callback(callback_base):
    class SelectedCheckpointFileCallback(callback_base):
        """Upload top-k checkpoints and last.ckpt to fixed W&B run-file slots.

        W&B artifacts are immutable, so uploading the selected set as an artifact
        after every validation creates an ever-growing version history. Run files
        can be replaced in place. The local slots are hard links, which also keeps
        the selected upload view from duplicating checkpoint bytes on disk.
        """

        def __init__(
            self,
            checkpoint_callback,
            *,
            upload_dir: str | os.PathLike[str],
            every_n_epochs: int = 1,
        ):
            super().__init__()
            self.checkpoint_callback = checkpoint_callback
            self.upload_dir = Path(upload_dir).expanduser().resolve()
            self.every_n_epochs = max(1, int(every_n_epochs or 1))
            self._last_signature = None

        @staticmethod
        def _source_signature(path: Path) -> tuple[str, int, int]:
            stat = path.stat()
            return path.name, int(stat.st_size), int(stat.st_mtime_ns)

        @staticmethod
        def _replace_hard_link(source: Path, destination: Path) -> None:
            temporary = destination.with_name(
                f".{destination.name}.{os.getpid()}.tmp"
            )
            temporary.unlink(missing_ok=True)
            try:
                os.link(source, temporary)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)

        def _slot_sources(self) -> list[tuple[str, Path]]:
            selected = _selected_checkpoint_paths(self.checkpoint_callback)
            last_raw = str(
                getattr(self.checkpoint_callback, "last_model_path", "") or ""
            ).strip()
            last_path = Path(last_raw).resolve() if last_raw else None

            best_paths = [
                path
                for path in selected
                if last_path is None or path.resolve() != last_path
            ]
            save_top_k = max(
                0, int(getattr(self.checkpoint_callback, "save_top_k", 0) or 0)
            )
            slots = [
                (f"best-{rank:02d}.ckpt", path)
                for rank, path in enumerate(best_paths[:save_top_k], start=1)
            ]
            if last_path is not None and last_path.is_file():
                slots.append(("last.ckpt", last_path))
            return slots

        def _upload(self, trainer) -> None:
            if not bool(getattr(trainer, "is_global_zero", True)):
                return
            logger = getattr(trainer, "logger", None)
            experiment = getattr(logger, "experiment", None)
            save = getattr(experiment, "save", None)
            if not callable(save):
                return

            slots = self._slot_sources()
            if not slots:
                return
            signature = tuple(
                (slot_name, *self._source_signature(source))
                for slot_name, source in slots
            )
            if signature == self._last_signature:
                return

            self.upload_dir.mkdir(parents=True, exist_ok=True)
            active_names = {slot_name for slot_name, _ in slots}
            for stale in self.upload_dir.glob("best-*.ckpt"):
                if stale.name not in active_names:
                    stale.unlink(missing_ok=True)
            if "last.ckpt" not in active_names:
                (self.upload_dir / "last.ckpt").unlink(missing_ok=True)

            for slot_name, source in slots:
                destination = self.upload_dir / slot_name
                self._replace_hard_link(source, destination)
                save(
                    str(destination),
                    base_path=str(self.upload_dir),
                    policy="now",
                )
            self._last_signature = signature

        def on_validation_end(self, trainer, pl_module) -> None:
            del pl_module
            if bool(getattr(trainer, "sanity_checking", False)):
                return
            _refresh_last_checkpoint(trainer, self.checkpoint_callback)
            epoch = int(getattr(trainer, "current_epoch", 0))
            if (epoch + 1) % self.every_n_epochs != 0:
                return
            self._upload(trainer)

        def on_train_end(self, trainer, pl_module) -> None:
            del pl_module
            _refresh_last_checkpoint(trainer, self.checkpoint_callback)
            self._upload(trainer)

    return SelectedCheckpointFileCallback


def _stage1_wandb_tags(cfg) -> list[str]:
    model_cfg = cfg.model
    backbone = str(getattr(model_cfg, "backbone", "") or "unknown").strip().lower()
    audio_backbone = str(getattr(model_cfg, "audio_backbone", "") or "").strip().lower()
    channel_multipliers = getattr(model_cfg, "channel_multipliers", None)
    if backbone == "ddpm" and channel_multipliers:
        num_downsamples = max(0, len(channel_multipliers) - 1)
    else:
        num_downsamples = int(getattr(model_cfg, "num_downsamples", 0) or 0)

    patch_based = bool(getattr(model_cfg, "patch_based", False))
    tags = [
        f"backbone={backbone}",
        f"downsamples={num_downsamples}",
        f"sparsity={int(getattr(model_cfg, 'sparsity_level', 0) or 0)}",
        f"patch_based={str(patch_based).lower()}",
    ]
    if audio_backbone:
        tags.append(f"audio_backbone={audio_backbone}")
    if patch_based:
        tags.append(f"patch_size={int(getattr(model_cfg, 'patch_size', 0) or 0)}")
    return tags

