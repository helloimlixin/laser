from pathlib import Path
from types import SimpleNamespace
import json
import threading

import pytest
import torch

from src.training.recoverable_checkpoints import LocalFirstCheckpoints


def payload(step, value=1.):
    return dict(epoch=1, global_step=step, fid=42.,
                state_dict={'weight': torch.tensor([value])},
                optimizer={'step': torch.tensor(step)},
                rng_state_by_rank=[{'cpu': torch.get_rng_state()}])


def wb():
    return SimpleNamespace(entity='test', project='test', id='test')


def test_complete_local_recovery_and_immutable_best(tmp_path):
    checkpoints = LocalFirstCheckpoints(tmp_path)
    last, best = tmp_path/'last.pt', tmp_path/'best.pt'
    state = payload(1)
    checkpoints.save(state, last)
    checkpoints.snapshot(last, best)
    state['state_dict']['weight'].fill_(99.)
    checkpoints.save(payload(2, 2.), last)
    checkpoints.save(payload(3, 3.), last)
    checkpoints.close()
    assert torch.load(best, weights_only=False)['state_dict']['weight'].item() == 1.
    restored = torch.load(last, weights_only=False)
    journal = json.loads((tmp_path/'latest.json').read_text())
    assert restored['global_step'] == journal['global_step'] == 3
    assert restored['optimizer']['step'].item() == 3
    assert torch.equal(restored['rng_state_by_rank'][0]['cpu'], state['rng_state_by_rank'][0]['cpu'])
    assert Path(journal['path']).stat().st_size == journal['bytes']
    assert len(list((tmp_path/'.checkpoint-data').glob('*.pt'))) == 2


def test_blocked_and_failed_cloud_cannot_block_or_corrupt_next_save(tmp_path):
    entered, release = threading.Event(), threading.Event()
    observed = []

    class Cloud:
        def __init__(self, *args):
            pass
        def __call__(self, paths, epoch):
            entered.set()
            assert release.wait(10)
            observed.append(torch.load(paths[0], weights_only=False)['global_step'])
            if len(observed) == 1:
                raise RuntimeError('simulated unavailable cloud')

    checkpoints = LocalFirstCheckpoints(tmp_path, cloud_factory=Cloud)
    last = tmp_path/'last.pt'
    try:
        checkpoints.save(payload(1), last)
        checkpoints.upload(wb(), last_checkpoint=last, best_fid=[], upload_dir=tmp_path/'unused')
        assert entered.wait(5)
        # This call must finish while the previous uploader is still blocked.
        done = threading.Event()
        errors = []
        def save_next():
            try:
                checkpoints.save(payload(2), last)
            except BaseException as error:
                errors.append(error)
            finally:
                done.set()
        worker = threading.Thread(target=save_next)
        worker.start()
        assert done.wait(5), 'local save waited for remote upload'
        worker.join()
        assert not errors
        assert torch.load(last, weights_only=False)['global_step'] == 2
        checkpoints.upload(wb(), last_checkpoint=last, best_fid=[], upload_dir=tmp_path/'unused')
    finally:
        release.set()
        checkpoints.close()
    assert observed == [1, 2]
    jobs = [json.loads(p.read_text()) for p in sorted((tmp_path/'cloud-jobs').glob('*/job.json'))]
    assert [job['state'] for job in jobs] == ['failed', 'verified']
    assert Path(jobs[0]['paths'][0]).is_file(), 'failed upload inputs must survive for retry'
    assert not Path(jobs[1]['paths'][0]).exists()


def test_incomplete_local_write_keeps_previous_checkpoint(tmp_path, monkeypatch):
    checkpoints = LocalFirstCheckpoints(tmp_path)
    last = tmp_path/'last.pt'
    checkpoints.save(payload(1), last)
    def interrupted(state, path):
        Path(path).write_bytes(b'incomplete checkpoint')
        raise OSError('simulated disk failure')
    monkeypatch.setattr(torch, 'save', interrupted)
    with pytest.raises(OSError, match='disk failure'):
        checkpoints.save(payload(2), last)
    checkpoints.close()
    assert torch.load(last, weights_only=False)['global_step'] == 1
    assert json.loads((tmp_path/'latest.json').read_text())['global_step'] == 1
    assert not list((tmp_path/'.checkpoint-data').glob('*.tmp'))


def test_periodic_recovery_does_not_publish_every_save(tmp_path):
    checkpoints = LocalFirstCheckpoints(tmp_path)
    state = payload(750)
    state['fid'] = None
    checkpoints.save(state, tmp_path/'last.pt')
    checkpoints.upload(wb(), last_checkpoint=tmp_path/'last.pt', best_fid=[], upload_dir=tmp_path/'unused')
    checkpoints.close()
    assert not (tmp_path/'cloud-jobs').exists()


def test_reject_shared_storage_target(tmp_path):
    checkpoints = LocalFirstCheckpoints(tmp_path/'local')
    try:
        with pytest.raises(ValueError, match='local recovery directory'):
            checkpoints.save(payload(1), tmp_path/'shared.pt')
    finally:
        checkpoints.close()
