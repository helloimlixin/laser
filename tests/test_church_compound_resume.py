"""Relocating a continuation must preserve FID ranking and reject recipe drift."""
import argparse
import importlib.util
import json
from pathlib import Path
import pytest

SPEC = importlib.util.spec_from_file_location('support',
    Path(__file__).resolve().parents[1] / 'scripts/tools/church_compound_support.py')
support = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(support)


def make_resume(tmp_path, monkeypatch):
    tokenizer, reference, best = [tmp_path / name for name in ('tokenizer.pt', 'ref.npz', 'best.pt')]
    tokenizer.write_bytes(b'frozen tokenizer')
    reference.write_bytes(b'identical reference')
    best.write_bytes(b'retained best checkpoint')
    saved = dict(fid_reference_stats='/old/ref.npz', coeff_scales=[7., 3., 2., 1.],
                 lr=.0005, batch_size=16, total_batch_size=128, upload_checkpoints=True)
    (tmp_path / 'preflight.json').write_text(json.dumps(dict(
        tokenizer=dict(export_sha256=support.sha(tokenizer)),
        reference_sha256=support.sha(reference), coeff_scales=saved['coeff_scales'])))
    monkeypatch.setenv('CHURCH_BASE', str(tmp_path))
    monkeypatch.setenv('WORLD_SIZE', '8')
    args = argparse.Namespace(**dict(saved, checkpoint=tokenizer,
        fid_reference_stats=reference, checkpoint_dir=tmp_path, output=tmp_path))
    payload = dict(config=saved, checkpoint_world_size=8, rng_state_by_rank=list(range(8)),
                   best_fid=[(11.57, '/old/checkpoints/best.pt')])
    return payload, args


def test_reference_relocation_preserves_best_fid(tmp_path, monkeypatch):
    payload, args = make_resume(tmp_path, monkeypatch)
    actual = support.adapt_resume_payload(payload, args)
    assert actual['config']['fid_reference_stats'] == str(args.fid_reference_stats)
    assert actual['best_fid'] == [(11.57, str(tmp_path / 'best.pt'))]


@pytest.mark.parametrize('change', ['reference', 'tokenizer', 'batch', 'world', 'missing_best', 'uploads'])
def test_resume_rejects_incompatible_state(tmp_path, monkeypatch, change):
    payload, args = make_resume(tmp_path, monkeypatch)
    if change == 'reference':
        args.fid_reference_stats.write_bytes(b'different reference')
    elif change == 'tokenizer':
        args.checkpoint.write_bytes(b'different tokenizer')
    elif change == 'batch':
        args.total_batch_size = 256
    elif change == 'world':
        monkeypatch.setenv('WORLD_SIZE', '2')
    elif change == 'uploads':
        args.upload_checkpoints = False
    else:
        (tmp_path / 'best.pt').unlink()
    with pytest.raises((ValueError, AssertionError, FileNotFoundError)):
        support.adapt_resume_payload(payload, args)


def test_four_gpu_resume_remaps_cursor_without_replaying_images(tmp_path, monkeypatch):
    from src.training.rqtransformer import remap_resume_batch_index
    payload, args = make_resume(tmp_path, monkeypatch)
    payload['config']['world_size'] = 8
    monkeypatch.setenv('WORLD_SIZE', '4')
    result = support.adapt_resume_payload(payload, args)
    assert result['rng_state_by_rank'] == [0, 1, 2, 3]
    assert result['amarel_world_size_change']['accumulation_steps'] == 2
    batch, old_world = remap_resume_batch_index(904, saved_config=result['config'],
        batch_size=16, world_size=4, total_batch_size=128, global_step=36400,
        start_epoch=36, optimizer_steps_per_epoch=986)
    assert old_world == 8 and batch == 1808
    assert batch * 16 * 4 == 904 * 16 * 8


def test_next_four_gpu_allocation_preserves_all_saved_streams(tmp_path, monkeypatch):
    payload, args = make_resume(tmp_path, monkeypatch)
    payload.update(checkpoint_world_size=4, rng_state_by_rank=[0,1,2,3])
    payload['config']['world_size'] = 4
    monkeypatch.setenv('WORLD_SIZE', '4')
    result = support.adapt_resume_payload(payload, args)
    assert result['rng_state_by_rank'] == [0,1,2,3]
    assert 'amarel_world_size_change' not in result


def test_latest_checkpoint_requires_frozen_continuation_identity(tmp_path, monkeypatch):
    monkeypatch.setenv('CHURCH_BASE', str(tmp_path))
    (tmp_path / 'prepared').mkdir()
    (tmp_path / 'runtime-sha256.txt').write_text('frozen source manifest')
    (tmp_path / 'prepared/cache-ready.json').write_text(json.dumps(dict(cache_sha256='cache')))
    original, latest = tmp_path / 'original.pt', tmp_path / 'latest.pt'
    original.write_bytes(b'original validated checkpoint')
    latest.write_bytes(b'new complete checkpoint')
    preflight = dict(stage2_sha256=support.sha(original), tokenizer=dict(export_sha256='tokenizer'),
                     reference_sha256='reference')
    (tmp_path / 'preflight.json').write_text(json.dumps(preflight))
    payload = dict(global_step=36500,epoch=37,batch_idx=36,checkpoint_world_size=4,
        config=dict(batch_size=16,amarel_continuation=support.continuation_identity()),
        rng_state_by_rank=[0,1,2,3],scheduler=dict(last_epoch=36500,T_max=88740),
        optimizer=dict(state={i:dict(step=36500) for i in range(517)}))
    result = support.validate_recovered_checkpoint(payload, latest)
    assert result['step'] == 36500
    # A reviewed operational-only update can resume its exact predecessor.
    predecessor = dict(payload['config']['amarel_continuation'])
    (tmp_path / 'runtime-sha256.txt').write_text('updated allocation memory request')
    with pytest.raises(ValueError, match='validated continuation'):
        support.validate_recovered_checkpoint(payload, latest)
    preflight['accepted_continuation_identities'] = [predecessor]
    (tmp_path / 'preflight.json').write_text(json.dumps(preflight))
    assert support.validate_recovered_checkpoint(payload, latest)['step'] == 36500
    payload['optimizer']['state'][0]['step'] = 36499
    with pytest.raises(AssertionError):
        support.validate_recovered_checkpoint(payload, latest)
    payload['optimizer']['state'][0]['step'] = 36500
    payload['config']['amarel_continuation']['cache_sha256'] = 'changed cache'
    with pytest.raises(ValueError, match='validated continuation'):
        support.validate_recovered_checkpoint(payload, latest)


def test_fixed_online_slots_include_latest_and_retained_best(tmp_path):
    from src.training.rqtransformer import upload_selected_checkpoint_files
    latest, best = tmp_path / 'checkpoint.pt', tmp_path / 'best.pt'
    latest.write_bytes(b'latest complete state')
    best.write_bytes(b'best FID complete state')
    calls = []

    class OnlineRun:
        def save(self, path, *, base_path, policy):
            calls.append((Path(path).name, Path(path).read_bytes(), policy))

    upload_selected_checkpoint_files(OnlineRun(), last_checkpoint=latest,
        best_fid=[(11.57, str(best))], upload_dir=tmp_path / 'online')
    assert calls == [('last.pt', b'latest complete state', 'now'),
                     ('best-fid-01.pt', b'best FID complete state', 'now')]
