import copy
import math

import pytest
import torch

from src.training.compound_history_schedule import (
    CompoundHistorySchedule, migrate_schedule, parameter_segments, repartition_optimizer_state)


def setup():
    names = ['gate','body0','body1','history0','history1']
    new = ['gate','history0','history1']
    parameters = [torch.nn.Parameter(torch.tensor(1.)) for _ in names]
    optimizer = torch.optim.AdamW(parameters, lr=1e-5)
    sum(parameters).backward()
    optimizer.step()
    old = optimizer.state_dict()
    old['param_groups'][0]['lr'] = 8e-6 * .5 * (1+math.cos(math.pi*500/1000))
    migrated = repartition_optimizer_state(old,names,new)
    groups = [dict(params=[parameters[i] for i in ids],compound_group=kind)
              for kind,ids in parameter_segments(names,new)]
    resumed = torch.optim.AdamW(groups,lr=8e-6)
    resumed.load_state_dict(migrated)
    source = dict(kind='original-rqtransformer-global-cosine-v1',last_epoch=500,
        policy=dict(initial_lr=8e-6,min_lr=0.,total_steps=1000,adaptive_reductions=False),
        last_observation_step=490,last_fid=15.,best=14.9)
    return names,new,old,migrated,resumed,source


def test_group_migration_preserves_parameter_ids_and_all_adam_tensor_objects():
    names,new,old,migrated,optimizer,source = setup()
    assert [i for g in migrated['param_groups'] for i in g['params']] == list(range(5))
    assert migrated['state'] is old['state']
    assert repartition_optimizer_state(migrated,names,new) is migrated
    for i in old['state']:
        for key in ('step','exp_avg','exp_avg_sq'):
            assert migrated['state'][i][key] is old['state'][i][key]


def test_new_path_warms_without_rate_jump_and_both_rates_end_at_zero():
    _,_,_,_,optimizer,source = setup()
    schedule = migrate_schedule(optimizer,source,completed_steps=500,initial_lr=8e-6,
                                total_steps=1000,history_peak_lr=1e-5,warmup_steps=20)
    assert schedule.rate_at('history',500) == schedule.rate_at('pretrained',500)
    assert schedule.rate_at('history',520) == 1e-5
    assert schedule.rate_at('pretrained',520) < schedule.rate_at('pretrained',500)
    assert schedule.rate_at('pretrained',1000) == schedule.rate_at('history',1000) == 0.


def test_interrupted_warmup_resume_has_identical_next_rates_and_adam_state():
    _,_,_,_,optimizer,source = setup()
    schedule = migrate_schedule(optimizer,source,completed_steps=500,initial_lr=8e-6,
                                total_steps=1000,history_peak_lr=1e-5,warmup_steps=20)
    for _ in range(7):schedule.step()
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    saved_schedule = schedule.state_dict()
    # Loading must use the saved warmup origin, rather than restart 200 updates.
    clone = torch.optim.AdamW([dict(params=[torch.nn.Parameter(torch.tensor(1.)) for _ in g['params']],
        compound_group=g['compound_group']) for g in saved_optimizer['param_groups']],lr=1e-5)
    clone.load_state_dict(saved_optimizer)
    restored = CompoundHistorySchedule(clone,saved_schedule)
    schedule.step(); restored.step()
    assert schedule.get_last_lr() == restored.get_last_lr()
    assert restored.last_epoch == 508 and restored.policy['start_step'] == 500
    assert all(int(s['step']) == 1 for s in clone.state.values())


def test_resume_rejects_lr_or_clock_mismatch_and_fid_does_not_change_rates():
    _,_,_,_,optimizer,source = setup()
    schedule = migrate_schedule(optimizer,source,completed_steps=500,initial_lr=8e-6,total_steps=1000)
    rates = schedule.get_last_lr()
    schedule.observe(99.)
    assert schedule.get_last_lr() == rates
    state = schedule.state_dict()
    optimizer.param_groups[0]['lr'] *= 2
    with pytest.raises(ValueError,match='does not match'):
        CompoundHistorySchedule(optimizer,state)


def test_runtime_adapter_restores_adam_before_applying_group_schedule(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from src.training.compound_history_schedule_hook import install
    names,new,old,_,prior,source = setup()
    model = torch.nn.Module()
    parameters = [p for g in prior.param_groups for p in g['params']]
    for name,p in zip(names,parameters):model.register_parameter(name,p)
    training = SimpleNamespace(optimizer_state_for_unwrapped_load=lambda state,model:state)
    reports = {}
    original_init = torch.optim.AdamW.__init__
    monkeypatch.setattr(torch.optim.AdamW,'__init__',original_init)
    monkeypatch.setenv('RANK','0')
    install(training,lambda:model,dict(parameter_names=names,new_parameter_names=new),
            lambda:dict(global_step=500),tmp_path,lambda path,value:reports.update({path.name:value}))
    resumed = torch.optim.AdamW(model.parameters(),lr=8e-6)
    resumed.load_state_dict(training.optimizer_state_for_unwrapped_load(old,model))
    assert len(resumed.param_groups) == 3
    assert all(int(state['step']) == 1 for state in resumed.state.values())
    schedule = training.create_cosine_lr_scheduler(resumed,initial_lr=8e-6,min_lr=0.,
        total_steps=1000,completed_steps=500,state_dict=source)
    assert schedule.last_epoch == 500
    assert reports['schedule-resume-rank0.json']['adam_states_preserved']
    # The custom checkpoint can resume directly without repeating migration.
    again = torch.optim.AdamW(model.parameters(),lr=8e-6)
    again.load_state_dict(training.optimizer_state_for_unwrapped_load(resumed.state_dict(),model))
    restored = training.create_cosine_lr_scheduler(again,initial_lr=8e-6,min_lr=0.,
        total_steps=1000,completed_steps=500,state_dict=schedule.state_dict())
    assert restored.state_dict() == schedule.state_dict()
