import copy
import math

import pytest
import torch

from src.training.rqtransformer import (
    create_warmup_cosine_lr_scheduler,
    create_warmup_linear_lr_scheduler,
)


def optimizer():
    return torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=0.01)


def test_warmup_cosine_matches_closed_form_and_endpoints():
    opt = optimizer()
    schedule = create_warmup_cosine_lr_scheduler(
        opt, initial_lr=0.01, min_lr=0.001, total_steps=20,
        warmup_steps=4, warmup_start_ratio=0.01,
    )
    for step in range(22):
        expected = (0.01 * (0.01 + 0.99 * step / 4) if step < 4 else
                    0.001 + 0.0045 * (1 + math.cos(math.pi * min(step - 4, 16) / 16)))
        assert opt.param_groups[0]['lr'] == pytest.approx(expected)
        opt.step()
        schedule.step()


@pytest.mark.parametrize('resume_step', [2, 4, 11])
def test_warmup_cosine_resume_preserves_remaining_trajectory(resume_step):
    kwargs = dict(initial_lr=0.01, min_lr=0.001, total_steps=20,
                  warmup_steps=4, warmup_start_ratio=0.01)
    opt = optimizer()
    schedule = create_warmup_cosine_lr_scheduler(opt, **kwargs)
    for _ in range(resume_step):
        opt.step()
        schedule.step()
    state = copy.deepcopy(schedule.state_dict())
    resumed_opt = optimizer()
    resumed = create_warmup_cosine_lr_scheduler(
        resumed_opt, completed_steps=resume_step, state_dict=state, **kwargs,
    )
    for _ in range(20 - resume_step):
        assert resumed_opt.param_groups[0]['lr'] == opt.param_groups[0]['lr']
        opt.step()
        schedule.step()
        resumed_opt.step()
        resumed.step()
    with pytest.raises(ValueError, match='settings differ'):
        create_warmup_linear_lr_scheduler(
            optimizer(), completed_steps=resume_step, state_dict=state, **kwargs,
        )
