import errno
import json
from pathlib import Path

import torch

from src.training.cc3m_joint import Checkpoints
from src.training.cc3m_storage import WorkspaceMirror, local_run_root


def test_workspace_disconnect_cannot_prevent_progress_or_checkpoint_save(tmp_path, monkeypatch):
    cfg = dict(local_checkpoints=str(tmp_path/'local/checkpoints'),
               output=str(tmp_path/'workspace/run'))
    output = local_run_root(cfg)/'train'
    output.mkdir(parents=True)
    real_mkdir = Path.mkdir

    def disconnected(path, *args, **kwargs):
        if path.is_relative_to(tmp_path/'workspace'):
            raise OSError(errno.ENOTCONN, 'Transport endpoint is not connected')
        return real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'mkdir', disconnected)
    from src.data.cc3m_joint import write_json
    write_json(output/'progress-rank0.json', dict(global_step=501))
    store = Checkpoints(cfg, None)
    store.submit(dict(global_step=501, model={'weight': torch.tensor([3.])}))
    store.close()
    checkpoint = Path(cfg['local_checkpoints'])/'last.pt'
    assert torch.load(checkpoint, weights_only=False)['global_step'] == 501
    mirror = WorkspaceMirror([(local_run_root(cfg), Path(cfg['output'])),
        (checkpoint.parent, Path(cfg['output'])/'checkpoints')])
    assert len(mirror.sync_once()) == 2
    assert (output/'progress-rank0.json').is_file() and checkpoint.is_file()
    monkeypatch.setattr(Path, 'mkdir', real_mkdir)
    assert mirror.sync_once() == []
    assert json.loads((Path(cfg['output'])/'train/progress-rank0.json').read_text())['global_step'] == 501
    copied = Path(cfg['output'])/'checkpoints/last.pt'
    assert copied.read_bytes() == checkpoint.read_bytes()
    assert json.loads(copied.with_suffix('.json').read_text())['global_step'] == 501


def test_mirror_replaces_old_directory_alias_without_touching_local_files(tmp_path):
    local, remote = tmp_path/'local', tmp_path/'workspace'
    (local/'train').mkdir(parents=True)
    (local/'train/progress.json').write_text('{"global_step": 502}')
    (local/'partial.tmp').write_text('unfinished')
    remote.mkdir()
    (remote/'train').symlink_to(local/'train', target_is_directory=True)
    mirror = WorkspaceMirror([(local, remote)])
    assert mirror.sync_once() == []
    assert not (remote/'train').is_symlink()
    assert (remote/'train/progress.json').read_text() == (local/'train/progress.json').read_text()
    assert not (remote/'partial.tmp').exists()
    (local/'train/progress.json').write_text('{"global_step": 503}')
    assert mirror.sync_once() == []
    assert json.loads((remote/'train/progress.json').read_text())['global_step'] == 503


def test_checkpoint_receipt_is_not_published_when_payload_copy_fails(tmp_path, monkeypatch):
    import shutil
    local, remote = tmp_path/'local', tmp_path/'workspace'
    local.mkdir()
    (local/'last.pt').write_bytes(b'checkpoint')
    (local/'last.json').write_text('{"global_step": 500}')
    real_copy = shutil.copyfile

    def fail_payload(source, target):
        if Path(source).suffix == '.pt':
            raise OSError(errno.ENOTCONN, 'Transport endpoint is not connected')
        return real_copy(source, target)

    monkeypatch.setattr(shutil, 'copyfile', fail_payload)
    mirror = WorkspaceMirror([(local, remote)])
    assert len(mirror.sync_once()) == 1
    assert not (remote/'last.json').exists()
    monkeypatch.setattr(shutil, 'copyfile', real_copy)
    assert mirror.sync_once() == []
    assert (remote/'last.pt').read_bytes() == b'checkpoint'
    assert (remote/'last.json').exists()
