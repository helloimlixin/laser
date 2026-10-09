import copy

import pytest
import torch

from scripts.tools.imagenet_loss_decay_lr import (
    ContinuationWarmupSchedule, create_scheduler, migrate_continuation, scale_saved_continuation)
from src.training.fid_adaptive_schedule import FidAdaptiveSchedule


def source():
    p = torch.nn.Parameter(torch.tensor([.2, -.4], dtype=torch.float64))
    optimizer = torch.optim.AdamW([p], lr=1e-6, betas=(.9, .95), weight_decay=1e-4)
    scheduler = FidAdaptiveSchedule(optimizer, initial_lr=1e-4, min_lr=0.,
        total_steps=100, baseline_fid=15.5)
    for _ in range(17):
        p.square().sum().backward(); optimizer.step(); optimizer.zero_grad(); scheduler.step()
    scheduler.observe(15.4)
    return p, optimizer.state_dict(), scheduler.state_dict()


def restore(p, optimizer_state, state):
    optimizer = torch.optim.AdamW([p])
    optimizer.load_state_dict(optimizer_state)
    return optimizer, create_scheduler(optimizer, initial_lr=1e-5, min_lr=0.,
        total_steps=100, completed_steps=state['last_epoch'], state_dict=state)


def test_trained_adam_and_absolute_clock_survive_schedule_migration():
    _, optimizer, state = source()
    old = copy.deepcopy(state)
    revised, new = migrate_continuation(optimizer, state, warmup_steps=10)
    assert revised['state'] is optimizer['state']
    assert state == old
    assert new['last_epoch'] == state['last_epoch'] == 17
    assert new['last_observation_step'] == 17 and new['best'] == 15.4
    assert new['source_controller'] == state
    assert revised['param_groups'][0]['lr'] == 3e-7
    for key in ('betas', 'weight_decay', 'eps', 'params'):
        assert revised['param_groups'][0][key] == optimizer['param_groups'][0][key]
    curve = lambda step: ContinuationWarmupSchedule.lr_at_step(new['policy'], step)
    assert curve(17) == 3e-7
    assert curve(22) == pytest.approx((3e-7 + 1e-5) / 2)
    assert curve(27) == 1e-5
    assert curve(100) == 0.


def test_resume_in_warmup_keeps_exact_next_adam_updates(tmp_path):
    p, optimizer_state, old = source()
    revised, state = migrate_continuation(optimizer_state, old, warmup_steps=10)
    optimizer, scheduler = restore(p, revised, state)
    for _ in range(4):
        p.square().sum().backward(); optimizer.step(); optimizer.zero_grad(); scheduler.step()
    torch.save(dict(parameter=p.detach(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict()), tmp_path/'resume.pt')
    raw = torch.load(tmp_path/'resume.pt', weights_only=False)
    q = torch.nn.Parameter(raw['parameter'].clone())
    resumed_optimizer, resumed_scheduler = restore(q, raw['optimizer'], raw['scheduler'])
    for _ in range(12):
        for parameter, opt, sched in ((p, optimizer, scheduler), (q, resumed_optimizer, resumed_scheduler)):
            parameter.square().sum().backward(); opt.step(); opt.zero_grad(); sched.step()
    assert torch.equal(p, q)
    assert scheduler.state_dict() == resumed_scheduler.state_dict()
    for key in ('step', 'exp_avg', 'exp_avg_sq'):
        assert torch.equal(optimizer.state[p][key], resumed_optimizer.state[q][key])


def test_official_fid_regression_is_monitored_without_halving_lr():
    p, optimizer_state, old = source()
    revised, state = migrate_continuation(optimizer_state, old, warmup_steps=10)
    optimizer, scheduler = restore(p, revised, state)
    for score in (15.6, 16., 16.2):
        scheduler.step()
        before = optimizer.param_groups[0]['lr']
        decision = scheduler.observe(score)
        assert decision['decision'] == 'monitor'
        assert decision['lr_after'] == decision['lr_before'] == before
        assert scheduler.multiplier == 1.
    assert scheduler.observe(16.2) is None


def test_mismatched_saved_lr_and_invalid_warmup_are_rejected():
    p, optimizer, state = source()
    bad = copy.deepcopy(optimizer); bad['param_groups'][0]['lr'] *= 2
    with pytest.raises(ValueError):migrate_continuation(bad, state)
    for options in ({'start_lr':0.}, {'peak_lr':float('nan')}, {'warmup_steps':100}):
        with pytest.raises(ValueError):migrate_continuation(optimizer, state, **options)
    revised, new = migrate_continuation(optimizer, state, warmup_steps=10)
    revised['param_groups'][0]['lr'] *= 2
    with pytest.raises(ValueError):restore(p, revised, new)


def test_post_warmup_continuation_preserves_decay_and_trained_adam():
    p, optimizer_state, old = source()
    revised, state = migrate_continuation(optimizer_state, old, warmup_steps=10)
    optimizer, scheduler = restore(p, revised, state)
    for _ in range(20):
        p.square().sum().backward();optimizer.step();optimizer.zero_grad();scheduler.step()
    saved_state = copy.deepcopy(scheduler.state_dict())
    q = torch.nn.Parameter(p.detach().clone())
    resumed_optimizer, resumed_scheduler = restore(q,copy.deepcopy(optimizer.state_dict()),saved_state)
    assert resumed_optimizer.param_groups[0]['lr']==optimizer.param_groups[0]['lr']<1e-5
    assert resumed_scheduler.last_epoch==37 and resumed_scheduler.state_dict()==saved_state
    for _ in range(10):
        for parameter,opt,sched in ((p,optimizer,scheduler),(q,resumed_optimizer,resumed_scheduler)):
            parameter.square().sum().backward();opt.step();opt.zero_grad();sched.step()
    assert torch.equal(p,q) and scheduler.state_dict()==resumed_scheduler.state_dict()
    for key in ('step','exp_avg','exp_avg_sq'):
        assert torch.equal(optimizer.state[p][key],resumed_optimizer.state[q][key])
    assert ContinuationWarmupSchedule.lr_at_step(saved_state['policy'],100)==0.


def test_tenfold_continuation_keeps_adam_and_clocks_and_resumes_exactly():
    p, optimizer_state, old = source()
    revised, state = migrate_continuation(optimizer_state, old, warmup_steps=10)
    optimizer, scheduler = restore(p, revised, state)
    for _ in range(20):
        p.square().sum().backward();optimizer.step();optimizer.zero_grad();scheduler.step()
    old_optimizer, old_scheduler = optimizer.state_dict(), copy.deepcopy(scheduler.state_dict())
    scaled, schedule = scale_saved_continuation(old_optimizer, old_scheduler, 10.)
    assert scaled['state'] is old_optimizer['state']
    for field in ('last_epoch', 'best', 'bad_epochs', 'reductions', 'last_observation_step', 'source_controller'):
        assert schedule[field] == old_scheduler[field]
    assert scaled['param_groups'][0]['lr'] == pytest.approx(old_optimizer['param_groups'][0]['lr'] * 10)
    assert schedule['policy']['initial_lr'] == 1e-4
    for step in (37, 38, 50, 99):
        assert ContinuationWarmupSchedule.lr_at_step(schedule['policy'], step) == pytest.approx(
            10 * ContinuationWarmupSchedule.lr_at_step(old_scheduler['policy'], step))
    assert ContinuationWarmupSchedule.lr_at_step(schedule['policy'], 100) == 0.
    for factor in (0., -1., float('nan'), float('inf')):
        with pytest.raises(ValueError):scale_saved_continuation(old_optimizer, old_scheduler, factor)
    q = torch.nn.Parameter(p.detach().clone())
    opt = torch.optim.AdamW([q]);opt.load_state_dict(scaled)
    sched = create_scheduler(opt, initial_lr=1e-4, min_lr=0., total_steps=100,
                             completed_steps=schedule['last_epoch'], state_dict=schedule)
    for _ in range(3):
        q.square().sum().backward();opt.step();opt.zero_grad();sched.step()
    r = torch.nn.Parameter(q.detach().clone())
    resumed = torch.optim.AdamW([r]);resumed.load_state_dict(copy.deepcopy(opt.state_dict()))
    reload = create_scheduler(resumed, initial_lr=1e-4, min_lr=0., total_steps=100,
                              completed_steps=sched.last_epoch, state_dict=sched.state_dict())
    for _ in range(5):
        for parameter, optim, sch in ((q, opt, sched), (r, resumed, reload)):
            parameter.square().sum().backward();optim.step();optim.zero_grad();sch.step()
    assert torch.equal(q, r) and sched.state_dict() == reload.state_dict()
    for field in ('step', 'exp_avg', 'exp_avg_sq'):
        assert torch.equal(opt.state[q][field], resumed.state[r][field])
