import copy
import hashlib

import pytest
import torch

from src.training.cc3m_recovery import load_best_weights, schedule_offset, verify_baseline


def case(tmp_path):
    model = torch.nn.Linear(3, 2)
    cfg = dict(arch={'width': 3}, stage1_sha256='a', codebook_sha256='b',
        reconstruction_codebook_sha256='c', tokenizer_backend='native_beam_pair',
        temperatures=.5, global_batch=8, sampling={'top_p': .9}, evaluation={'images': 16}, epochs=10)
    path = tmp_path/'best-fid.pt'
    torch.save(dict(model=model.state_dict(), config=copy.deepcopy(cfg), epoch=4,
        global_step=40, metrics=dict(fid=23.38, clip_score=.23)), path)
    cfg['warm_start'] = dict(checkpoint=str(path), checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        epoch=4, global_step=40, optimizer_policy='reset', expected_fid=23.38)
    return model, cfg


def test_best_model_restart_preserves_weights_and_validates_lineage(tmp_path):
    source, cfg = case(tmp_path)
    target = torch.nn.Linear(3, 2)
    result = load_best_weights(target, cfg, updates=10)
    assert result == dict(epoch=4, global_step=40, best=dict(fid=23.38, clip_score=.23))
    for key, expected in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], expected, rtol=0, atol=0)
    cfg['temperatures'] = .25
    with pytest.raises(ValueError, match='temperatures'):
        load_best_weights(target, cfg, updates=10)


def test_restart_rejects_corrupt_checkpoint_and_wrong_cursor(tmp_path):
    source, cfg = case(tmp_path)
    cfg['warm_start']['global_step'] = 41
    with pytest.raises(ValueError, match='completed epoch'):
        schedule_offset(cfg, 10)
    cfg['warm_start']['global_step'] = 40
    cfg['warm_start']['checkpoint_sha256'] = '0'*64
    with pytest.raises(ValueError, match='SHA256'):
        load_best_weights(source, cfg, 10)


def test_recovery_schedule_restarts_at_lower_lr_and_roundtrips(tmp_path):
    model, cfg = case(tmp_path)
    offset = schedule_offset(cfg, 10)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10*cfg['epochs']-offset)
    assert not opt.state and scheduler.last_epoch == 0 and opt.param_groups[0]['lr'] == 1e-4
    for _ in range(3):
        opt.zero_grad(); model(torch.ones(1, 3)).sum().backward(); opt.step(); scheduler.step()
    assert scheduler.last_epoch == (offset+3)-offset
    other = torch.optim.AdamW(model.parameters(), lr=1e-4)
    restored = torch.optim.lr_scheduler.CosineAnnealingLR(other, T_max=60)
    other.load_state_dict(opt.state_dict()); restored.load_state_dict(scheduler.state_dict())
    opt.step(); scheduler.step(); other.step(); restored.step()
    assert restored.last_epoch == scheduler.last_epoch
    assert other.param_groups[0]['lr'] == opt.param_groups[0]['lr']


def test_recovery_requires_original_fid_to_reproduce(tmp_path):
    _, cfg = case(tmp_path)
    verify_baseline(dict(fid=23.381), cfg)
    with pytest.raises(ValueError, match='refusing to train'):
        verify_baseline(dict(fid=27.14), cfg)
