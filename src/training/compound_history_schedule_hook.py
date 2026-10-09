"""Runtime adapter for full-state compound training with discriminative rates."""
import math

import torch

from src.training.compound_history_schedule import (
    KIND, CompoundHistorySchedule, migrate_schedule, parameter_segments,
    repartition_optimizer_state)

ACTIVE_SCHEDULE = None


def current_schedule():
    return ACTIVE_SCHEDULE


def observe_official_fid(fid):
    if ACTIVE_SCHEDULE is None:
        raise RuntimeError('Official FID arrived before compound scheduler initialization')
    return ACTIVE_SCHEDULE.observe(fid)


def install(training, model_getter, transfer, recovery_getter, verify, record):
    original_init = torch.optim.AdamW.__init__

    def optimizer_init(self, params, *args, **kwargs):
        parameters = list(params)
        named = list(model_getter().named_parameters())
        names = [name for name, _ in named]
        if names != transfer['parameter_names'] or [id(p) for p in parameters] != [id(p) for _, p in named]:
            raise ValueError('Compound optimizer construction parameter order changed')
        groups = [dict(params=[parameters[i] for i in ids], compound_group=kind)
                  for kind, ids in parameter_segments(names, transfer['new_parameter_names'])]
        return original_init(self, groups, *args, **kwargs)
    torch.optim.AdamW.__init__ = optimizer_init

    original_convert = training.optimizer_state_for_unwrapped_load
    def convert(state, model):
        state = original_convert(state, model)
        return repartition_optimizer_state(state, transfer['parameter_names'], transfer['new_parameter_names'])
    training.optimizer_state_for_unwrapped_load = convert

    def create_scheduler(optimizer, *, initial_lr, min_lr, total_steps, completed_steps=0, state_dict=None):
        global ACTIVE_SCHEDULE
        recovery = recovery_getter()
        if state_dict is None or min_lr != 0 or completed_steps != recovery['global_step']:
            raise ValueError('Full optimizer/scheduler progress must be restored')
        if state_dict['kind'] == KIND:
            policy = state_dict['policy']
            if (policy['pretrained_peak_lr'],policy['min_lr'],policy['total_steps'],state_dict['last_epoch']) != (
                    initial_lr,min_lr,total_steps,completed_steps):
                raise ValueError('Compound scheduler settings changed on resume')
            ACTIVE_SCHEDULE = CompoundHistorySchedule(optimizer,state_dict)
            migrated = False
        else:
            ACTIVE_SCHEDULE = migrate_schedule(optimizer,state_dict,completed_steps=completed_steps,
                initial_lr=initial_lr,total_steps=total_steps,history_peak_lr=1e-5,warmup_steps=200)
            migrated = True
        names = transfer['parameter_names']
        live = optimizer.state_dict()
        for group in live['param_groups']:
            ids = group['params']
            expected_kind = 'history' if names[ids[0]] in set(transfer['new_parameter_names']) else 'pretrained'
            if group['compound_group'] != expected_kind:
                raise ValueError('Compound learning rate attached to the wrong parameters')
        report = dict(global_step=completed_steps,schedule=ACTIVE_SCHEDULE.state_dict(),
            migration_applied=migrated,adam_states_preserved=True,optimizer_parameters=len(optimizer.state),
            groups=[dict(kind=g['compound_group'],parameter_tensors=len(g['params']),lr=g['lr'])
                    for g in optimizer.param_groups],zero_floor=True,epoch100_endpoint=total_steps)
        import os
        record(verify / ('schedule-resume-rank'+os.environ['RANK']+'.json'),report)
        return ACTIVE_SCHEDULE
    training.create_cosine_lr_scheduler = create_scheduler
