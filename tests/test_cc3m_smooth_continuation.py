import copy

import pytest
import torch

from src.training.cc3m_text import (config_digest, create_lr_scheduler, cpu_snapshot,
                                  serialize_checkpoint)


def test_same_step_checkpoints_keep_separate_immutable_files(tmp_path):
    options = dict(local_checkpoints=str(tmp_path))
    payload = dict(global_step=21270, epoch=15, value=torch.tensor([1.]))
    first = serialize_checkpoint(payload, ['last.pt'], options, local_slots=True)
    other = dict(payload, value=torch.tensor([2.]))
    second = serialize_checkpoint(other, ['last.pt'], options, local_slots=True)
    assert first.path != second.path
    assert first.path.exists() and second.path.exists()
    assert torch.load(first.path, weights_only=True)['value'].item() == 1.
    assert torch.load(second.path, weights_only=True)['value'].item() == 2.
    assert torch.load(tmp_path/'last.pt', weights_only=True)['value'].item() == 2.


def test_original_best_continuation_keeps_adam_and_lr_smooth_despite_noisy_fid():
    torch.manual_seed(7)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    source = dict(epochs=100, lr=5e-5, min_lr=1e-5,
        lr_schedule='fid_adaptive_cosine', fid_lr_policy=dict(
            baseline_fid=27.53, patience=2, min_delta=.1, factor=.5, cooldown=1,
            decay_start_step=140, decay_steps=100))
    scheduler = create_lr_scheduler(optimizer, source, 10)
    for _ in range(150):
        model(torch.ones(2, 3)).square().mean().backward()
        optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
    scheduler.observe(25.09)
    before_lr = optimizer.param_groups[0]['lr']
    adam = cpu_snapshot(optimizer.state_dict()['state'])
    original_state = copy.deepcopy(scheduler.state_dict())
    target = dict(source, lr=before_lr, fid_lr_policy=dict(source['fid_lr_policy'],
        adaptive_reductions=False, decay_start_step=150, decay_steps=850),
        lr_schedule_migration=dict(source_config_sha256=config_digest(source),
            source_step=150, source_scheduler_sha256=config_digest(original_state)))
    smooth = create_lr_scheduler(optimizer, target, 10, 150, original_state, source)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(before_lr, abs=1e-16)
    for key, fields in adam.items():
        for name, value in fields.items():
            torch.testing.assert_close(value, optimizer.state_dict()['state'][key][name], rtol=0, atol=0)
    clone = copy.deepcopy(model)
    clone_optimizer = torch.optim.AdamW(clone.parameters(), lr=target['lr'])
    clone_optimizer.load_state_dict(cpu_snapshot(optimizer.state_dict()))
    resumed = create_lr_scheduler(clone_optimizer, target, 10, 150,
        smooth.state_dict(), target)
    previous = before_lr
    for fid in [26., 27., 28., 24., 28., 29., 30.]:
        for m, opt, sch in [(model, optimizer, smooth), (clone, clone_optimizer, resumed)]:
            m(torch.ones(2, 3)).square().mean().backward()
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        lr_before_observation = optimizer.param_groups[0]['lr']
        assert smooth.observe(fid) == resumed.observe(fid)
        assert optimizer.param_groups[0]['lr'] == lr_before_observation <= previous
        assert smooth.reductions == 0 and smooth.multiplier == 1.
        assert smooth.state_dict() == resumed.state_dict()
        for left, right in zip(model.parameters(), clone.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        previous = lr_before_observation
    for _ in range(1000-smooth.last_epoch):
        smooth.step()
        assert target['min_lr'] <= optimizer.param_groups[0]['lr'] <= previous
        previous = optimizer.param_groups[0]['lr']
    assert previous == target['min_lr']


def test_stalled_continuation_lr_repair_preserves_adam_and_resumes_plateau_cuts():
    from scripts.tools.prepare_cc3m_lr_repair import lower_lr_options, convert_lr_checkpoint
    torch.manual_seed(17)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4.9e-5)
    options = dict(epochs=100, train_items=100, total_batch_size=10, batch_size=5,
        accumulation=1, seed=17, coeff_scales=[1.]*4, coeff_target_temperature=.5,
        cache_sha256={'train':'a', 'validation':'b'}, runtime_sha256={'trainer':'c'},
        lr=4.9e-5, min_lr=1e-5, lr_schedule='fid_adaptive_cosine', warmup_epochs=0,
        fid_lr_policy=dict(baseline_fid=25.09, patience=2, min_delta=.1, factor=.5,
            cooldown=1, adaptive_reductions=False, decay_start_step=150, decay_steps=850))
    schedule = create_lr_scheduler(optimizer, options, 10)
    for _ in range(180):
        model(torch.ones(2,3)).square().mean().backward()
        optimizer.step(); schedule.step(); optimizer.zero_grad(set_to_none=True)
    schedule.observe(25.49)
    source = dict(config=options, global_step=180, model=cpu_snapshot(model.state_dict()),
        optimizer=cpu_snapshot(optimizer.state_dict()), scheduler=schedule.state_dict())
    original = copy.deepcopy(source)
    target = lower_lr_options(source)
    repaired = convert_lr_checkpoint(source, target)
    assert repaired['optimizer']['param_groups'][0]['lr'] == pytest.approx(
        original['optimizer']['param_groups'][0]['lr']*.5, abs=1e-16)
    for key, fields in original['optimizer']['state'].items():
        for name, value in fields.items():
            torch.testing.assert_close(value, repaired['optimizer']['state'][key][name], rtol=0, atol=0)
    assert source['scheduler'] == original['scheduler']
    assert repaired['scheduler']['best'] == original['scheduler']['best']
    assert repaired['scheduler']['last_observation_step'] == original['scheduler']['last_observation_step']
    assert repaired['scheduler']['bad_epochs'] == 0
    clone = copy.deepcopy(model)
    cloned_optimizer = torch.optim.AdamW(clone.parameters(), lr=target['lr'])
    cloned_optimizer.load_state_dict(repaired['optimizer'])
    resumed = create_lr_scheduler(cloned_optimizer, target, 10, 180, repaired['scheduler'], target)
    before = cloned_optimizer.param_groups[0]['lr']
    decisions=[]
    for fid in [25.4, 25.5, 25.6, 25.7, 24.8]:
        cloned_optimizer.zero_grad(set_to_none=True)
        clone(torch.ones(2,3)).square().mean().backward(); cloned_optimizer.step(); resumed.step()
        decisions.append(resumed.observe(fid)['decision'])
        assert target['min_lr'] <= cloned_optimizer.param_groups[0]['lr'] <= before
        before = cloned_optimizer.param_groups[0]['lr']
    assert decisions[:4] == ['cooldown','watch','watch','reduced']
    final = create_lr_scheduler(cloned_optimizer, target, 10, resumed.last_epoch,
        resumed.state_dict(), target)
    assert final.state_dict() == resumed.state_dict()


def test_lr_repair_rejects_an_increase_or_a_checkpoint_at_the_floor():
    from scripts.tools.prepare_cc3m_lr_repair import lower_lr_options
    with pytest.raises(ValueError, match='reduction'):
        lower_lr_options({}, factor=1.1)
    with pytest.raises(ValueError, match='floor'):
        lower_lr_options(dict(config={'min_lr':1e-5},
            optimizer={'param_groups':[{'lr':1e-5}]}, scheduler={'policy':{}}))
