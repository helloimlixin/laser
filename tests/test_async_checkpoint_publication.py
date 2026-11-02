import threading

import pytest
import torch

from src.training.async_checkpoint_publication import AsyncCheckpointPublisher
from src.training.background_checkpoint import BackgroundCheckpointWriter


def test_snapshot_alias_and_publication_do_not_block_training(tmp_path):
    started, release = threading.Event(), threading.Event()
    writer = BackgroundCheckpointWriter()
    publisher = AsyncCheckpointPublisher()
    source = tmp_path / "local.pt"
    persistent = tmp_path / "last.pt"
    best = tmp_path / "best.pt"
    weights = torch.arange(5)
    torch.save({"weights": weights, "step": 11}, source)
    uploaded = []

    def commit():
        started.set()
        assert release.wait(10)
        persistent.write_bytes(source.read_bytes())

    try:
        writer.submit(commit)
        assert started.wait(10)
        writer.append(lambda: best.write_bytes(persistent.read_bytes()))
        publisher.submit(writer.completion, lambda: uploaded.append(torch.load(best, weights_only=True)))
        weights.add_(100)
        assert not persistent.exists()
        assert uploaded == []
        release.set()
        writer.wait()
        publisher.close()
        assert uploaded[0]["step"] == 11
        assert torch.equal(uploaded[0]["weights"], torch.arange(5))
    finally:
        release.set()
        writer.close()
        publisher.close()


def test_failed_commit_prevents_alias_and_upload():
    writer = BackgroundCheckpointWriter()
    publisher = AsyncCheckpointPublisher()
    aliases, uploads = [], []

    def fail():
        raise OSError("shared storage failed")

    writer.submit(fail)
    writer.append(lambda: aliases.append(True))
    publisher.submit(writer.completion, lambda: uploads.append(True))
    with pytest.raises(OSError, match="shared storage failed"):
        writer.wait()
    with pytest.raises(OSError, match="shared storage failed"):
        publisher.close()
    assert aliases == [] and uploads == []
    writer.close()


def test_publication_close_reports_upload_failure():
    publisher = AsyncCheckpointPublisher()

    def fail():
        raise RuntimeError("upload failed")

    publisher.submit(None, fail)
    with pytest.raises(RuntimeError, match="upload failed"):
        publisher.close()
    publisher.close()
