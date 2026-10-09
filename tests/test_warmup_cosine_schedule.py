import copy

import pytest
import torch

from src.training.warmup_cosine_schedule import WarmupCosineSchedule


def test_fresh_warmup_and_cosine_endpoints():
    opt = torch.optim.AdamW(torch.nn.Linear(2, 1).parameters(), lr=.0001)
    schedule = WarmupCosineSchedule(opt, initial_lr=.0001, min_lr=.00001,
                                    total_steps=20, warmup_steps=4)
    assert opt.param_groups[0]['lr'] == .000025
    for _ in range(4):
        schedule.step()
    assert opt.param_groups[0]['lr'] == .0001
    for _ in range(16):
        schedule.step()
    assert opt.param_groups[0]['lr'] == .00001


@pytest.mark.parametrize('boundary', [2, 4, 7])
def test_adam_updates_resume_identically_across_warmup_boundary(boundary):
    torch.manual_seed(91)
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=.0001)
    policy = dict(initial_lr=.0001, min_lr=.00001, total_steps=20, warmup_steps=4)
    schedule = WarmupCosineSchedule(opt, **policy)
    x = torch.ones(3, 2)

    def update(m, o, s):
        m(x).square().mean().backward()
        o.step(); s.step(); o.zero_grad(set_to_none=True)

    for _ in range(boundary):
        update(model, opt, schedule)
    clone = copy.deepcopy(model)
    resumed = torch.optim.AdamW(clone.parameters(), lr=.0001)
    resumed.load_state_dict(copy.deepcopy(opt.state_dict()))
    restored = WarmupCosineSchedule(resumed, **policy, completed_steps=boundary,
                                    state_dict=schedule.state_dict())
    for _ in range(10):
        update(model, opt, schedule)
        update(clone, resumed, restored)
        assert schedule.state_dict() == restored.state_dict()
        for a, b in zip(model.parameters(), clone.parameters()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    resumed.param_groups[0]['lr'] *= 2
    with pytest.raises(ValueError, match='optimizer disagree'):
        WarmupCosineSchedule(resumed, **policy, completed_steps=schedule.last_epoch,
                             state_dict=schedule.state_dict())
