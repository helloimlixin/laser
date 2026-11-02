import copy
from types import SimpleNamespace

import pytest

from scripts.tools.imagenet_epoch68_runtime import (ANCHOR_STEP, TOTAL_STEPS, SOURCE_LR,
    continuation_state, create_scheduler)


def saved_cosine():
    return dict(last_epoch=ANCHOR_STEP,T_max=TOTAL_STEPS,eta_min=0.,base_lrs=[.0005],_last_lr=[SOURCE_LR])


def test_saved_lr_is_continuous_and_final_lr_is_exactly_zero():
    optimizer=SimpleNamespace(param_groups=[dict(lr=SOURCE_LR)])
    state=saved_cosine()
    original=copy.deepcopy(state)
    schedule=create_scheduler(optimizer,initial_lr=.0005,min_lr=0.,total_steps=TOTAL_STEPS,
        completed_steps=ANCHOR_STEP,state_dict=state)
    assert state==original
    assert optimizer.param_groups[0]['lr']==SOURCE_LR
    previous=SOURCE_LR
    for _ in range(TOTAL_STEPS-ANCHOR_STEP):
        schedule.step()
        current=optimizer.param_groups[0]['lr']
        assert 0.<=current<=previous
        previous=current
    assert current==0.


def test_plateau_and_resume_have_identical_decisions_and_lr():
    optimizer=SimpleNamespace(param_groups=[dict(lr=SOURCE_LR)])
    schedule=create_scheduler(optimizer,initial_lr=.0005,min_lr=0.,total_steps=TOTAL_STEPS,
        completed_steps=ANCHOR_STEP,state_dict=saved_cosine())
    for _ in range(3):
        schedule.step()
        schedule.observe(17.)
    assert schedule.reductions==1 and schedule.multiplier==.5
    restored_optimizer=SimpleNamespace(param_groups=copy.deepcopy(optimizer.param_groups))
    restored=create_scheduler(restored_optimizer,initial_lr=.0005,min_lr=0.,total_steps=TOTAL_STEPS,
        completed_steps=schedule.last_epoch,state_dict=schedule.state_dict())
    for fid in [16.7,16.8,17.,17.,17.,16.5]:
        schedule.step();restored.step()
        assert schedule.observe(fid)==restored.observe(fid)
        assert schedule.state_dict()==restored.state_dict()
        assert optimizer.param_groups==restored_optimizer.param_groups


def test_wrong_source_or_nonzero_floor_is_rejected():
    with pytest.raises(ValueError):continuation_state(saved_cosine(),ANCHOR_STEP+1)
    with pytest.raises(ValueError):
        create_scheduler(SimpleNamespace(param_groups=[dict(lr=SOURCE_LR)]),initial_lr=.0005,
            min_lr=1e-7,total_steps=TOTAL_STEPS,completed_steps=ANCHOR_STEP,state_dict=saved_cosine())
    changed=continuation_state(saved_cosine(),ANCHOR_STEP)
    changed['policy']['min_lr']=1e-7
    with pytest.raises(ValueError,match='policy'):
        continuation_state(changed,ANCHOR_STEP)
