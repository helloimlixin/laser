from pathlib import Path
import json
import os

import pytest
import torch

from src.training import rqtransformer as training
from src.training import k4_checkpoint_io


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setenv('LASER_CHECKPOINT_IMMUTABLE_FILES', '1')
    monkeypatch.setenv('LASER_CHECKPOINT_STAGING_DIR', str(tmp_path/'staging'))
    monkeypatch.setenv('LASER_CHECKPOINT_UPLOAD_CACHE_DIR', str(tmp_path/'uploads'))
    return tmp_path


def test_replacement_preserves_best_and_bounds_retained_payloads(storage):
    latest, best = storage/'last.pt', storage/'best.pt'
    training.atomic_torch_save({'step': 1}, latest)
    first = latest.resolve()
    training.snapshot_checkpoint(latest, best)
    training.atomic_torch_save({'step': 2}, latest)
    assert latest.is_symlink() and best.is_symlink()
    assert torch.load(latest, weights_only=True)['step'] == 2
    assert torch.load(best, weights_only=True)['step'] == 1
    assert not first.exists()
    assert len(list((storage/'.checkpoint-data').glob('*.pt'))) == 2
    assert torch.load(training._checkpoint_upload_source(latest), weights_only=True)['step'] == 2
    training.remove_checkpoint(best)
    assert not best.exists()
    assert len(list((storage/'.checkpoint-data').glob('*.pt'))) == 1
    assert len(list((storage/'uploads/objects').glob('*.pt'))) == 1


def test_interrupted_copy_preserves_readable_previous_checkpoint(storage, monkeypatch):
    latest = storage/'last.pt'
    training.atomic_torch_save({'step': 1}, latest)
    previous = latest.resolve()

    def partial_copy(source, target):
        Path(target).write_bytes(b'incomplete')
        raise OSError('simulated shared-storage disconnect')

    monkeypatch.setattr(training.shutil, 'copyfile', partial_copy)
    with pytest.raises(OSError, match='disconnect'):
        training.atomic_torch_save({'step': 2}, latest)
    assert latest.resolve() == previous
    assert torch.load(latest, weights_only=True)['step'] == 1
    assert list((storage/'.checkpoint-data').glob('*.pt')) == [previous]


def test_large_payload_never_passes_through_rename(storage, monkeypatch):
    replace = training.os.replace

    def object_store_replace(source, target):
        if Path(target).suffix == '.pt' and not Path(source).is_symlink():
            raise OSError('object store cannot replace large payloads')
        return replace(source, target)

    monkeypatch.setattr(training.os, 'replace', object_store_replace)
    # Keep upload-cache hardlinks out of the simulated remote filesystem.
    monkeypatch.delenv('LASER_CHECKPOINT_UPLOAD_CACHE_DIR')
    latest = storage/'last.pt'
    for step in (1, 2):
        training.atomic_torch_save({'step': step}, latest)
        assert torch.load(latest, weights_only=True)['step'] == step


def test_immutable_ctime_finalization_keeps_local_upload(storage, monkeypatch):
    latest = storage/'last.pt'
    training.atomic_torch_save({'step': 1}, latest)
    local, receipt = training._local_checkpoint_paths(latest)
    metadata = json.loads(receipt.read_text())
    metadata['identity']['st_ctime_ns'] -= 1000000000
    receipt.write_text(json.dumps(metadata))

    def forbidden_copy(*args, **kwargs):
        raise AssertionError('An intact immutable checkpoint must upload locally')

    monkeypatch.setattr(training.shutil, 'copyfile', forbidden_copy)
    assert training._checkpoint_upload_source(latest) == local

    class Run:
        def save(self, path, **kwargs):
            assert torch.load(path, weights_only=True)['step'] == 1

    paths = training.upload_selected_checkpoint_files(
        Run(), last_checkpoint=latest, best_fid=[], upload_dir=storage/'slots',
    )
    assert len(paths) == 1


@pytest.mark.parametrize('field', ['st_dev', 'st_ino', 'st_size', 'st_mtime_ns'])
def test_changed_immutable_identity_rejects_cache(storage, field):
    latest = storage/'last.pt'
    training.atomic_torch_save({'step': 1}, latest)
    _, receipt = training._local_checkpoint_paths(latest)
    metadata = json.loads(receipt.read_text())
    metadata['identity'][field] += 1
    receipt.write_text(json.dumps(metadata))
    assert training._checkpoint_upload_source(latest) == latest


def test_ctime_relaxation_requires_immutable_storage(storage, monkeypatch):
    latest = storage/'last.pt'
    training.atomic_torch_save({'step': 1}, latest)
    _, receipt = training._local_checkpoint_paths(latest)
    metadata = json.loads(receipt.read_text())
    metadata['identity']['st_ctime_ns'] -= 1
    receipt.write_text(json.dumps(metadata))
    monkeypatch.delenv('LASER_CHECKPOINT_IMMUTABLE_FILES')
    assert training._checkpoint_upload_source(latest) == latest


def test_prune_cold_buffers_preserves_archives_active_models_and_upload_pins(storage):
    archived = storage / 'epochs' / 'epoch_005.pt'
    latest, best = storage / 'last.pt', storage / 'best.pt'
    for step, path in enumerate((archived, latest, best)):
        k4_checkpoint_io.atomic_torch_save({'step': step}, path)
    archived_local, _ = k4_checkpoint_io._local_checkpoint_paths(archived)
    pin = storage / 'upload-snapshot.pt'
    os.link(archived_local, pin)

    assert k4_checkpoint_io.prune_local_checkpoint_cache([latest, best]) == 1
    assert not archived_local.exists()
    assert torch.load(archived, weights_only=True)['step'] == 0
    assert torch.load(pin, weights_only=True)['step'] == 0
    for path in (latest, best):
        local, _ = k4_checkpoint_io._local_checkpoint_paths(path)
        assert local.exists()
    assert len(list((storage / 'uploads/objects').glob('*.pt'))) == 2
