import threading

import pytest
import torch

from src.training.background_checkpoint import BackgroundCheckpointWriter
from src.training import rqtransformer as training


def test_background_save_is_immutable_and_upload_waits_for_commit(tmp_path, monkeypatch):
    monkeypatch.setenv("LASER_CHECKPOINT_STAGING_DIR", str(tmp_path / "staging"))
    monkeypatch.setenv("LASER_CHECKPOINT_UPLOAD_CACHE_DIR", str(tmp_path / "local"))
    target = tmp_path / "persistent" / "last.pt"
    training.atomic_torch_save({"step": 0}, target)
    copying, release = threading.Event(), threading.Event()
    original_copy = training.shutil.copyfile

    def delayed_copy(source, destination):
        copying.set()
        assert release.wait(10)
        return original_copy(source, destination)

    monkeypatch.setattr(training.shutil, "copyfile", delayed_copy)
    uploaded = []
    weights = torch.arange(5)
    writer = BackgroundCheckpointWriter()
    try:
        training.atomic_torch_save(
            {"step": 1, "weights": weights}, target, background=writer,
            on_commit=lambda: uploaded.append(torch.load(target, weights_only=True)),
        )
        assert copying.wait(10)
        weights.add_(100)  # Model changes after the local snapshot has completed.
        assert torch.load(target, weights_only=True) == {"step": 0}
        assert uploaded == []
        release.set()
        writer.wait()
        assert len(uploaded) == 1
        assert uploaded[0]["step"] == 1
        assert torch.equal(uploaded[0]["weights"], torch.arange(5))
        assert training._checkpoint_upload_source(target).read_bytes() == target.read_bytes()
        assert not list((tmp_path / "staging").iterdir())
    finally:
        release.set()
        writer.close()


def test_background_failure_preserves_previous_checkpoint_and_blocks_upload(tmp_path, monkeypatch):
    monkeypatch.setenv("LASER_CHECKPOINT_STAGING_DIR", str(tmp_path / "staging"))
    target = tmp_path / "persistent" / "last.pt"
    training.atomic_torch_save({"step": 0}, target)

    def fail_copy(*_args):
        raise OSError("simulated storage failure")

    monkeypatch.setattr(training.shutil, "copyfile", fail_copy)
    monkeypatch.setattr(training.time, "sleep", lambda _seconds: None)
    uploaded = []
    writer = BackgroundCheckpointWriter()
    try:
        training.atomic_torch_save(
            {"step": 1}, target, background=writer, on_commit=lambda: uploaded.append(True),
        )
        with pytest.raises(OSError, match="simulated storage failure"):
            writer.wait()
        assert torch.load(target, weights_only=True) == {"step": 0}
        assert uploaded == []
        assert not target.with_suffix(".pt.tmp").exists()
        assert not list((tmp_path / "staging").iterdir())
    finally:
        writer.close()


def test_writer_bounds_pending_work_and_close_drains_it():
    first_started, release_first = threading.Event(), threading.Event()
    second_submitting, second_submitted = threading.Event(), threading.Event()
    completed = []
    writer = BackgroundCheckpointWriter()

    def first():
        first_started.set()
        assert release_first.wait(10)
        completed.append(1)

    def submit_second():
        second_submitting.set()
        writer.submit(lambda: completed.append(2))
        second_submitted.set()

    writer.submit(first)
    assert first_started.wait(10)
    submitter = threading.Thread(target=submit_second)
    submitter.start()
    try:
        assert second_submitting.wait(10)
        assert not second_submitted.wait(0.1)
        release_first.set()
        submitter.join(10)
        assert second_submitted.is_set()
        writer.close()
        assert completed == [1, 2]
    finally:
        release_first.set()
        submitter.join(10)


def test_background_save_requires_staging(tmp_path, monkeypatch):
    monkeypatch.delenv("LASER_CHECKPOINT_STAGING_DIR", raising=False)
    writer = BackgroundCheckpointWriter()
    try:
        with pytest.raises(ValueError, match="require local staging"):
            training.atomic_torch_save({"step": 1}, tmp_path / "last.pt", background=writer)
    finally:
        writer.close()
