import errno
import sys
import types

import numpy as np

from src.wandb_media import log_wandb_audio, log_wandb_metrics


def test_audio_storage_quota_error_is_best_effort(monkeypatch, tmp_path, capsys):
    class _Audio:
        def __init__(self, audio, *, sample_rate, caption=None):
            self.audio = audio
            self.sample_rate = sample_rate
            self.caption = caption

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(Audio=_Audio))

    class _Experiment:
        def log(self, payload):
            raise OSError(errno.EDQUOT, "Disk quota exceeded")

    class _Logger:
        save_dir = tmp_path
        experiment = _Experiment()

        def log_audio(self, *args, **kwargs):
            raise AssertionError("log_wandb_audio should bypass Lightning log_audio")

    log_wandb_audio(
        _Logger(),
        "train/audio_clips",
        [np.zeros(16, dtype=np.float32)],
        sample_rates=[16000],
        captions=["clip"],
        step=7,
    )

    captured = capsys.readouterr()
    assert "Skipping W&B audio log for train/audio_clips" in captured.err


def test_metric_storage_quota_error_is_best_effort(capsys):
    class _Logger:
        def log_metrics(self, metrics, step=None):
            raise OSError(errno.ENOSPC, "No space left on device")

    log_wandb_metrics(_Logger(), {"train/loss": 1.0}, step=3)

    captured = capsys.readouterr()
    assert "Skipping W&B metrics log" in captured.err
