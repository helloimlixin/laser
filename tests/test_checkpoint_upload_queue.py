from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
import torch

from src.training.checkpoint_upload_queue import CheckpointFile, CheckpointUploadQueue
from src.training.cc3m_text import commit_local_checkpoint, upload_checkpoint_file


def checkpoint(tmp_path, step, slots=('last.pt',)):
    path = tmp_path / f'{step}.pt'
    path.write_text(str(step))
    return CheckpointFile(path, step, 0, path.stat().st_size, str(step), slots)


def test_slow_upload_keeps_all_best_winners_and_coalesces_pending_last(tmp_path):
    started, release = Event(), Event()
    calls = []
    def upload(item, slots):
        if item.step == 1:
            started.set()
            assert release.wait(5)
        assert item.path.read_text() == str(item.step)
        calls.append((item.step, slots))
    queue = CheckpointUploadQueue(upload)
    first = checkpoint(tmp_path, 1)
    queue.submit(first)
    assert started.wait(2)
    items = [checkpoint(tmp_path, 2, ('last.pt', 'best-fid.pt')),
             checkpoint(tmp_path, 3),
             checkpoint(tmp_path, 4, ('last.pt', 'best-clip.pt')),
             checkpoint(tmp_path, 5), checkpoint(tmp_path, 6)]
    for item in items:queue.submit(item)
    assert first.path.exists()
    assert items[0].path.exists() and items[2].path.exists()
    assert not items[1].path.exists() and not items[3].path.exists()
    release.set();queue.close()
    assert calls == [(1, ('last.pt',)), (2, ('best-fid.pt',)),
                     (4, ('best-clip.pt',)), (6, ('last.pt',))]


def test_upload_failure_preserves_checkpoint_and_surfaces_on_training_thread(tmp_path):
    failed = Event()
    def upload(item, slots):
        failed.set()
        raise OSError('network unavailable')
    queue = CheckpointUploadQueue(upload)
    item = checkpoint(tmp_path, 1, ('last.pt', 'best-fid.pt'))
    queue.submit(item)
    assert failed.wait(2)
    with pytest.raises(RuntimeError, match='local state is preserved'):
        queue.close()
    assert item.path.exists()
    with pytest.raises(RuntimeError, match='upload failed'):
        queue.check()


def test_recovered_future_best_does_not_replace_active_trajectory_last(tmp_path):
    started, release = Event(), Event()
    calls=[]
    def upload(item, slots):
        if item.step == 1:
            started.set(); assert release.wait(5)
        assert item.path.read_text() == str(item.step)
        calls.append((item.step, slots))
    queue=CheckpointUploadQueue(upload)
    queue.submit(checkpoint(tmp_path, 1))
    assert started.wait(2)
    current=checkpoint(tmp_path, 2)
    future_best=checkpoint(tmp_path, 19, ('best-fid.pt','best-clip.pt'))
    queue.submit(current); queue.submit(future_best)
    assert current.path.exists() and future_best.path.exists()
    release.set(); queue.close()
    assert calls == [(1, ('last.pt',)), (19, ('best-fid.pt','best-clip.pt')), (2, ('last.pt',))]


def test_best_only_recovery_never_creates_a_last_alias(tmp_path):
    calls=[]
    queue=CheckpointUploadQueue(lambda item,slots:calls.append((item.step,slots)))
    queue.submit(checkpoint(tmp_path, 19, ('best-fid.pt','best-clip.pt')))
    queue.close()
    assert calls == [(19, ('best-fid.pt','best-clip.pt'))]


def test_transient_upload_failure_retries_without_losing_best_or_stopping_training(tmp_path):
    failed, permit = Event(), Event()
    calls, notices = [], []
    def upload(item, slots):
        if not permit.is_set():
            failed.set()
            raise ConnectionError('temporary connection failure')
        assert item.path.exists()
        calls.append((item.step, slots))
    queue = CheckpointUploadQueue(upload, retry_delay=.01,
        retryable=lambda error:isinstance(error, ConnectionError),
        on_retry=lambda item, slots, error, attempt:notices.append(attempt))
    best = checkpoint(tmp_path, 1, ('last.pt', 'best-fid.pt', 'best-clip.pt'))
    queue.submit(best)
    assert failed.wait(2)
    queue.check()  # A temporary upload failure must not abort optimizer updates.
    queue.submit(checkpoint(tmp_path, 2))
    assert best.path.exists()
    permit.set();queue.close()
    assert calls == [(1, ('last.pt', 'best-fid.pt', 'best-clip.pt')), (2, ('last.pt',))]
    assert notices


def test_local_saves_continue_while_online_upload_uses_stable_alias(tmp_path, monkeypatch):
    import base64
    import hashlib
    import wandb
    local = tmp_path / 'local'
    options = dict(local_checkpoints=str(local), output=str(tmp_path / 'output'),
        checkpoint_async_upload=True, wandb_entity='test', wandb_project='test')
    started, release = Event(), Event()
    saves = []
    wb = SimpleNamespace(id='test', summary={}, save=lambda *args, **kwargs:saves.append(args[0]))
    class Run:
        def file(self, slot):
            staging = local / 'online-checkpoints' / slot
            if not staging.exists():
                raise ValueError('No uploaded file yet')
            state = torch.load(staging, weights_only=True)
            if state['global_step'] == 1:
                started.set()
                assert release.wait(5)
                assert torch.load(staging, weights_only=True)['global_step'] == 1
            with staging.open('rb') as reader:
                md5 = base64.b64encode(hashlib.file_digest(reader, 'md5').digest()).decode()
            return SimpleNamespace(size=staging.stat().st_size, md5=md5)
    monkeypatch.setattr(wandb, 'Api', lambda **kwargs:SimpleNamespace(run=lambda _:Run()))
    queue = CheckpointUploadQueue(lambda item, slots:upload_checkpoint_file(item, slots, options, wb))
    commit_local_checkpoint(dict(global_step=1, epoch=0, model=torch.ones(3)),
        ['last.pt'], options, wb, queue)
    assert started.wait(2)
    commit_local_checkpoint(dict(global_step=2, epoch=0, model=torch.zeros(3)),
        ['last.pt'], options, wb, queue)
    assert torch.load(local / 'last.pt', weights_only=True)['global_step'] == 2
    assert torch.load(local / 'online-checkpoints/last.pt', weights_only=True)['global_step'] == 1
    release.set();queue.close()
    assert wb.summary['checkpoints/last_verified_step'] == 2
    assert all(Path(path).parent.name == 'online-checkpoints' for path in saves)


def test_recovery_acknowledges_identical_online_checkpoint_without_reupload(tmp_path, monkeypatch):
    import wandb
    item=checkpoint(tmp_path, 19, ('best-fid.pt','best-clip.pt'))
    class Run:
        def file(self, slot):
            return SimpleNamespace(size=item.size,md5=item.md5)
    monkeypatch.setattr(wandb, 'Api', lambda **kwargs:SimpleNamespace(run=lambda _:Run()))
    def unexpected_upload(*args, **kwargs):
        raise AssertionError('A checksum-verified file must not be uploaded again')
    wb=SimpleNamespace(id='test',summary={},save=unexpected_upload)
    options=dict(local_checkpoints=str(tmp_path/'local'),output=str(tmp_path/'output'),
        checkpoint_async_upload=True,wandb_entity='test',wandb_project='test')
    upload_checkpoint_file(item,item.slots,options,wb)
    assert wb.summary['checkpoints/online_verified'] is True
    assert 'checkpoints/last_verified_step' not in wb.summary
    assert not list((tmp_path/'local/online-checkpoints').iterdir())
