import copy

import pytest
import torch

from src.training.fid_plateau_lr import FIDPlateauLR
from src.training.rqtransformer import create_cosine_lr_scheduler


def source():
    parameter=torch.nn.Parameter(torch.ones(1))
    opt=torch.optim.AdamW([parameter],lr=.00066)
    old=create_cosine_lr_scheduler(opt,initial_lr=.00066,min_lr=3e-7,total_steps=1000,completed_steps=215)
    parameter.grad=torch.ones_like(parameter);opt.step()
    policy=dict(anchor_step=215,anchor_lr=opt.param_groups[0]['lr'],initial_lr=.00066,min_lr=3e-7,hold_lr=.0006,ramp_steps=10,
                total_steps=1000,factor=.8,patience=2,relative_threshold=.005,cooldown_evaluations=1,
                initial_best_fid=29.6,initial_last_evaluation_step=200)
    return parameter,opt,old,policy


def advance(schedule,step):
    while schedule.last_epoch<step:schedule.step()


def test_holds_while_improving_and_reduces_only_after_repeated_plateaus():
    _,opt,old,p=source();moments=copy.deepcopy(opt.state_dict()['state'])
    s=FIDPlateauLR(opt,policy=p,completed_steps=215,state_dict=old.state_dict())
    assert s.get_last_lr()==[p['anchor_lr']]
    for k,v in moments[0].items():torch.testing.assert_close(opt.state_dict()['state'][0][k],v,rtol=0,atol=0)
    for step,fid in ((250,27.),(300,25.),(350,23.),(400,22.)):
        advance(s,step);assert s.observe_fid(fid)['action']=='hold'
        assert s.get_last_lr()==[p['hold_lr']]
    advance(s,450);assert s.observe_fid(22.1)['action']=='hold'
    assert s.observe_fid(22.1)['action']=='duplicate-evaluation-ignored'
    advance(s,500);assert s.observe_fid(22.05)['action']=='reduce'
    assert s.get_last_lr()==[p['hold_lr']*.8]
    advance(s,550);assert s.observe_fid(22.2)['bad_evaluations']==0
    advance(s,600);assert s.observe_fid(21.)['bad_evaluations']==0


@pytest.mark.parametrize('resume_step',[215,219,225,250,350,450,500,550,600,1000])
def test_resume_preserves_moments_lr_and_fid_patience(resume_step):
    parameter,opt,old,p=source();s=FIDPlateauLR(opt,policy=p,completed_steps=215,state_dict=old.state_dict())
    observations={250:28.,300:27.,350:27.1,400:27.2,450:27.3,500:26.,550:26.01,600:26.02}
    for step in range(216,resume_step+1):
        parameter.grad=parameter.square().detach();opt.step();s.step()
        if step in observations:s.observe_fid(observations[step])
    rp=torch.nn.Parameter(parameter.detach().clone());ro=torch.optim.AdamW([rp],lr=.00066)
    ro.load_state_dict(copy.deepcopy(opt.state_dict()))
    rs=FIDPlateauLR(ro,policy=p,completed_steps=resume_step,state_dict=s.state_dict())
    for step in range(resume_step+1,1001):
        parameter.grad=parameter.square().detach();rp.grad=rp.square().detach()
        opt.step();s.step();ro.step();rs.step()
        if step in observations:assert s.observe_fid(observations[step])==rs.observe_fid(observations[step])
        assert s.state_dict()==rs.state_dict()
        torch.testing.assert_close(parameter,rp,rtol=0,atol=0)


def test_many_plateaus_respect_floor_and_invalid_observation_does_not_mutate_state():
    _,opt,old,p=source();p['factor']=.01
    s=FIDPlateauLR(opt,policy=p,completed_steps=215,state_dict=old.state_dict())
    before=copy.deepcopy(s.state_dict())
    with pytest.raises(ValueError):s.observe_fid(float('nan'))
    assert s.state_dict()==before
    for step in range(216,1001):s.step();s.observe_fid(30.)
    assert s.get_last_lr()==[3e-7]
    with pytest.raises(ValueError,match='exhausted'):s.step()


@pytest.mark.parametrize('failure',['wrong_step','wrong_source','wrong_lr','changed_policy','missing_state'])
def test_unverified_migration_is_rejected(failure):
    _,opt,old,p=source();state=old.state_dict();step=215
    if failure=='wrong_step':step=216
    if failure=='wrong_source':state['T_max']=1100
    if failure=='wrong_lr':opt.param_groups[0]['lr']*=1.01
    if failure=='changed_policy':
        state=FIDPlateauLR(opt,policy=p,completed_steps=step,state_dict=state).state_dict();p['factor']=.9
    if failure=='missing_state':state=None
    with pytest.raises(ValueError):FIDPlateauLR(opt,policy=p,completed_steps=step,state_dict=state)


@pytest.mark.parametrize('previous_reduction', [False, True])
def test_explicit_downward_revision_retains_adam_and_fid_history(previous_reduction):
    _,opt,old,p=source()
    s=FIDPlateauLR(opt,policy=p,completed_steps=215,state_dict=old.state_dict())
    advance(s,250);s.observe_fid(27.)
    advance(s,300);s.observe_fid(27.1)
    if previous_reduction:
        advance(s,350);s.observe_fid(27.2)
    saved=s.state_dict();moments=copy.deepcopy(opt.state_dict()['state'])
    revised=dict(p,anchor_step=s.last_epoch,anchor_lr=s.current_lr,hold_lr=s.current_lr*.92,
                 initial_best_fid=s.best_fid,initial_last_evaluation_step=s.last_evaluation_step)
    updated=FIDPlateauLR(opt,policy=revised,completed_steps=s.last_epoch,
                         state_dict=saved,revision_from=p)
    for name in ('best_fid','bad_evaluations','cooldown_remaining','reductions','last_evaluation_step'):
        assert getattr(updated,name)==getattr(s,name)
    for key,value in moments[0].items():
        torch.testing.assert_close(opt.state_dict()['state'][0][key],value,rtol=0,atol=0)
    advance(updated,revised['anchor_step']+5)
    assert updated.current_lr==pytest.approx((revised['anchor_lr']+revised['hold_lr'])/2)
    recovered=FIDPlateauLR(opt,policy=revised,completed_steps=updated.last_epoch,
                           state_dict=updated.state_dict(),revision_from=p)
    advance(updated,revised['anchor_step']+20);advance(recovered,revised['anchor_step']+20)
    assert updated.state_dict()==recovered.state_dict()
    assert recovered.current_lr==pytest.approx(revised['hold_lr'])
    # A plateau during a new ramp takes priority and cannot be undone by it.
    if not previous_reduction:
        assert recovered.observe_fid(27.2)['action']=='reduce'
        lowered=recovered.current_lr
        recovered.step();assert recovered.current_lr==lowered


@pytest.mark.parametrize('failure',['missing_revision','wrong_revision','wrong_anchor','wrong_step','reset_best','reset_evaluation','changed_decay'])
def test_downward_revision_rejects_unverified_source_or_reset_history(failure):
    _,opt,old,p=source();s=FIDPlateauLR(opt,policy=p,completed_steps=215,state_dict=old.state_dict())
    advance(s,300);s.observe_fid(27.)
    revised=dict(p,anchor_step=300,anchor_lr=s.current_lr,hold_lr=.00055,
                 initial_best_fid=s.best_fid,initial_last_evaluation_step=s.last_evaluation_step)
    previous=p
    if failure=='missing_revision':previous=None
    if failure=='wrong_revision':previous=dict(p,hold_lr=.00059)
    if failure=='wrong_anchor':revised['anchor_lr']=.00059
    if failure=='wrong_step':revised['anchor_step']=299
    if failure=='reset_best':revised['initial_best_fid']=29.6
    if failure=='reset_evaluation':revised['initial_last_evaluation_step']=200
    if failure=='changed_decay':revised['factor']=.9
    with pytest.raises(ValueError):
        FIDPlateauLR(opt,policy=revised,completed_steps=300,state_dict=s.state_dict(),revision_from=previous)
