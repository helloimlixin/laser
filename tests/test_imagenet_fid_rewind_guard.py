import copy

import pytest
import torch

from scripts.tools.imagenet_fid_rewind_guard import regression_request
from scripts.tools.imagenet_epoch64_lower_lr import lower_rewind_lr
from scripts.tools.imagenet_zero_floor_lr import create_scheduler
from src.training.fid_adaptive_schedule import FidAdaptiveSchedule


def metrics(fid):
    return dict(global_step=40690, fid=fid, metric_backend='original_rqtransformer',
                real_images=50000, generated_images=50000)


def test_any_official_increase_rewinds_to_best_and_equal_or_better_continues():
    best = [(15.554077729782705, '/best-full.pt')]
    decision = dict(lr_after=8.8e-8)
    assert regression_request(metrics(best[0][0]), best, decision) is None
    assert regression_request(metrics(15.5), best, decision) is None
    request = regression_request(metrics(best[0][0] + 1e-6), best, decision)
    assert request['source_checkpoint'] == '/best-full.pt'
    assert request['requested_lr'] == 8.8e-8
    assert request['best_fid'] == best[0][0]
    assert request['strict_comparison']
    request = regression_request(metrics(15.6), [(15.58,'/other.pt'),*best], decision)
    assert request['source_checkpoint'] == '/best-full.pt'


def test_rejects_nonofficial_or_incomplete_evaluations_and_missing_best():
    best = [(15.55, '/best-full.pt')]
    for patch in (dict(metric_backend='torchmetrics'),dict(real_images=256),
                  dict(generated_images=256),dict(fid=float('nan'))):
        with pytest.raises(ValueError):
            regression_request(dict(metrics(15.7),**patch),best,dict(lr_after=1e-7))
    with pytest.raises(ValueError):
        regression_request(metrics(15.7),[],dict(lr_after=1e-7))


def future_winner():
    parameter = torch.nn.Parameter(torch.tensor([.3, -.7]))
    optimizer = torch.optim.AdamW([parameter],lr=1e-6,betas=(.9,.95))
    for _ in range(7):
        parameter.grad = torch.tensor([.2,-.3]);optimizer.step()
    state = dict(kind='fid-adaptive-cosine-v1',
        policy=dict(initial_lr=1e-4,min_lr=0.,total_steps=27544,
            baseline_fid=15.764706963349738,patience=1,min_delta=.02,factor=.5,
            cooldown=0,decay_start_step=5008,decay_steps=22536),
        last_epoch=5634,multiplier=.001,best=15.5,bad_epochs=0,cooldown_remaining=0,
        reductions=4,last_observation_step=5634,last_fid=15.5)
    optimizer.param_groups[0]['lr'] = FidAdaptiveSchedule.lr_at_step(
        state['policy'],state['last_epoch'],state['multiplier'])
    return parameter,optimizer,state


def test_rewind_can_restore_a_newer_winner_without_resetting_adam_or_clock():
    parameter,optimizer,state = future_winner()
    before = copy.deepcopy(optimizer.state_dict())
    saved,state_after = lower_rewind_lr(optimizer.state_dict(),state,
                                       initial_lr=4e-8,scored_epoch64=False)
    optimizer.load_state_dict(saved)
    assert state_after['last_epoch'] == state_after['last_observation_step'] == 5634
    assert state_after['best'] == state_after['last_fid'] == 15.5
    assert state_after['reductions'] == 4
    for key,value in before['state'][0].items():
        assert torch.equal(value,optimizer.state[parameter][key])
    controller = create_scheduler(optimizer,initial_lr=1e-4,min_lr=0.,total_steps=27544,
                                  completed_steps=5634,state_dict=state_after)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(4e-8)
    parameter.grad = torch.tensor([.1,.2]);optimizer.step();controller.step()
    assert int(optimizer.state[parameter]['step']) == 8
    assert controller.last_epoch == 5635


def test_zero_endpoint_can_rewind_and_stop_with_saved_adam_intact():
    parameter,optimizer,state = future_winner()
    before = copy.deepcopy(optimizer.state_dict())
    saved,restored_state = lower_rewind_lr(optimizer.state_dict(),state,
        initial_lr=0.,scored_epoch64=False,allow_zero=True)
    optimizer.load_state_dict(saved)
    controller = create_scheduler(optimizer,initial_lr=1e-4,min_lr=0.,total_steps=27544,
        completed_steps=5634,state_dict=restored_state)
    assert optimizer.param_groups[0]['lr'] == 0.
    assert controller.last_epoch == 5634
    for key,value in before['state'][0].items():
        assert torch.equal(value,optimizer.state[parameter][key])


def test_repeated_halving_has_no_hidden_positive_floor():
    _,optimizer,state = future_winner()
    state['multiplier'] = 1e-20
    previous = FidAdaptiveSchedule.lr_at_step(state['policy'],5634,state['multiplier'])
    optimizer.param_groups[0]['lr'] = previous
    revised,after = lower_rewind_lr(optimizer.state_dict(),state,
        initial_lr=previous*.5,scored_epoch64=False)
    assert revised['param_groups'][0]['lr'] == previous*.5
    zero,_ = lower_rewind_lr(optimizer.state_dict(),state,
        initial_lr=0.,scored_epoch64=False,allow_zero=True)
    assert zero['param_groups'][0]['lr'] == 0.
