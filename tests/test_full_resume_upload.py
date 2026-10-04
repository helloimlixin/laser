import json

import pytest
import torch

from src.training.full_resume_upload import FullResumeWinners, recovery_metadata
from scripts.tools.download_wandb_full_resume import download_verified


def checkpoint(path, *, fid=20., inception=60., step=10):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-5)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    payload = dict(
        state_dict=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=None,
        epoch=1, batch_idx=0, global_step=step, checkpoint_world_size=1,
        rng_state_by_rank=[dict(torch_cpu=torch.get_rng_state(),
                                torch_cuda=torch.get_rng_state())],
        config=dict(lr_schedule='constant', lr=2.5e-5, min_lr=2.5e-5,
                    accumulation_steps=2, total_batch_size=2048, seed=261001,
                    epochs=100, training_images=1281167),
        fid=fid + .01, inception_score=inception - 1,
        original_rqtransformer_metrics=dict(fid=fid, inception_score=inception,
                                            global_step=step))
    torch.save(payload, path)
    return payload


def test_independent_full_winners_survive_source_pruning_and_restore_adam(tmp_path):
    source = tmp_path / 'candidate.pt'
    first = checkpoint(source, fid=20, inception=60)
    winners = FullResumeWinners(tmp_path / 'winners')
    assert winners.consider(source) == ['fid', 'is']
    source.unlink()
    # IS improves while FID worsens: retain both distinct full snapshots.
    second = checkpoint(source, fid=21, inception=65, step=20)
    assert winners.consider(source) == ['is']
    source.unlink()
    assert winners.state['fid']['global_step'] == 10
    assert winners.state['is']['global_step'] == 20
    for kind, expected in [('fid', first), ('is', second)]:
        payload = torch.load(tmp_path / 'winners' / f'best-{kind}-resume.pt', weights_only=False)
        model = torch.nn.Linear(2, 2)
        model.load_state_dict(payload['state_dict'], strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.1)
        optimizer.load_state_dict(payload['optimizer'])
        assert optimizer.param_groups[0]['lr'] == 2.5e-5
        for got, wanted in zip(optimizer.state.values(), expected['optimizer']['state'].values()):
            assert got['step'].item() == 1
            assert torch.equal(got['exp_avg'], wanted['exp_avg'])
            assert torch.equal(got['exp_avg_sq'], wanted['exp_avg_sq'])
    assert {p.name for p in winners.upload_paths()} == {
        'best-fid-resume.pt', 'best-fid-resume.json', 'best-is-resume.pt',
        'best-is-resume.json', 'resume-winners.json'}
    reopened = FullResumeWinners(tmp_path / 'winners')
    assert reopened.state == json.loads(winners.state_path.read_text())


@pytest.mark.parametrize('mutation', [
    lambda p: p.pop('optimizer'),
    lambda p: p['optimizer']['state'].pop(next(iter(p['optimizer']['state']))),
    lambda p: p['rng_state_by_rank'][0].pop('torch_cuda'),
    lambda p: p.update(checkpoint_world_size=2),
    lambda p: p['config'].update(lr_schedule='cosine'),
    lambda p: p.update(batch_idx=1),
    lambda p: p['original_rqtransformer_metrics'].update(global_step=99),
])
def test_rejects_incomplete_or_wrong_position_recovery(tmp_path, mutation):
    payload = checkpoint(tmp_path / 'candidate.pt')
    mutation(payload)
    with pytest.raises(ValueError):
        recovery_metadata(payload)


def test_unscored_regular_save_cannot_replace_a_metric_winner(tmp_path):
    source = tmp_path / 'candidate.pt'
    checkpoint(source)
    winners = FullResumeWinners(tmp_path / 'winners')
    winners.consider(source)
    payload = checkpoint(source)
    payload.pop('original_rqtransformer_metrics')
    payload.update(fid=None, inception_score=None, global_step=12, batch_idx=4)
    torch.save(payload, source)
    assert winners.consider(source) == []
    assert winners.state['fid']['global_step'] == 10


def test_durable_winners_preserve_full_state_with_immutable_storage(tmp_path, monkeypatch):
    monkeypatch.setenv('LASER_CHECKPOINT_IMMUTABLE_FILES', '1')
    monkeypatch.setenv('LASER_CHECKPOINT_UPLOAD_CACHE_DIR', str(tmp_path / 'cache'))
    source = tmp_path / 'candidate.pt'
    checkpoint(source)
    winners = FullResumeWinners(tmp_path / 'local')
    winners.consider(source)
    archive = tmp_path / 'durable'
    winners.persist(archive)
    previous = (archive / 'best-fid-resume.pt').resolve()
    source.unlink()
    checkpoint(source, fid=19, inception=65, step=12)
    winners.consider(source)
    winners.persist(archive)
    for kind in ('fid', 'is'):
        target = archive / f'best-{kind}-resume.pt'
        assert target.is_symlink()
        payload = torch.load(target, map_location='cpu', weights_only=False)
        assert recovery_metadata(payload)['global_step'] == 12
        assert json.loads(target.with_suffix('.json').read_text())['global_step'] == 12
    assert not previous.exists()


def test_cloud_download_corruption_preserves_previous_local_recovery(tmp_path):
    class Remote:
        size = 3
        md5 = 'AAAAAAAAAAAAAAAAAAAAAA=='

        def download(self, *, root, replace):
            from pathlib import Path
            (Path(root) / 'last.pt').write_bytes(b'bad')

    class Run:
        def file(self, name):
            assert name == 'last.pt'
            return Remote()

    target = tmp_path / 'last.pt'
    target.write_bytes(b'previous full recovery')
    with pytest.raises(ValueError, match='checksum mismatch'):
        download_verified(Run(), 'last.pt', tmp_path)
    assert target.read_bytes() == b'previous full recovery'
    assert list(tmp_path.iterdir()) == [target]
