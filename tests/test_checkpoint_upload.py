from pathlib import Path
import threading

import pytest

from src.training.checkpoint_upload import CheckpointUploader


def replace(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(value)
    temporary.replace(path)


def test_upload_is_nonblocking_immutable_and_keeps_latest_pending(tmp_path):
    started, release = threading.Event(), threading.Event()
    received = []
    checkpoint = tmp_path / 'last.pt'

    def upload(paths, epoch):
        if epoch == 1:
            started.set()
            assert release.wait(5)
        received.append((epoch, paths[0].read_text()))

    uploader = CheckpointUploader(tmp_path / 'snapshots', upload)
    replace(checkpoint, 'one')
    uploader.submit([checkpoint], 1)
    assert started.wait(5)
    replace(checkpoint, 'two')
    uploader.submit([checkpoint], 2)
    replace(checkpoint, 'three')
    uploader.submit([checkpoint], 3)
    # Snapshot copying also progresses while the older network upload blocks.
    with uploader.condition:
        assert uploader.condition.wait_for(lambda: uploader.pending is not None and
                                            uploader.pending[2] == 3, timeout=5)
    assert len(list((tmp_path / 'snapshots').iterdir())) == 2
    release.set()
    uploader.close()
    assert received == [(1, 'one'), (3, 'three')]
    assert checkpoint.read_text() == 'three'
    assert list((tmp_path / 'snapshots').iterdir()) == []


def test_background_failure_propagates_on_close_and_preserves_source(tmp_path):
    checkpoint = tmp_path / 'last.pt'
    checkpoint.write_text('durable')

    def upload(paths, epoch):
        raise ValueError('transfer failed')

    uploader = CheckpointUploader(tmp_path / 'snapshots', upload)
    uploader.submit([checkpoint], 1)
    with pytest.raises(RuntimeError, match='upload failed'):
        uploader.close()
    assert checkpoint.read_text() == 'durable'
    assert list((tmp_path / 'snapshots').iterdir()) == []
