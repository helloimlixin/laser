"""Verify pre-rewind replay, LR boundaries, and checkpoint continuation on CPU."""
import copy
import json
import math
from pathlib import Path
import torch
from lr_policy import LearningRateBranch


class NativeScheduler:
    def __init__(self, optimizer):
        self.after_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=62600,eta_min=0.)
    def step(self, epoch=None):self.after_scheduler.step(epoch)
    def get_last_lr(self):return self.after_scheduler.get_last_lr()
    def state_dict(self):return dict(warmup=None,after=self.after_scheduler.state_dict())
    def load_state_dict(self,state):self.after_scheduler.load_state_dict(state['after'])


def setup(step, policy, lr=None):
    parameter=torch.nn.Parameter(torch.ones(()))
    optimizer=torch.optim.AdamW([parameter],lr=.0005,betas=(.9,.95),weight_decay=.0001)
    native=NativeScheduler(optimizer)
    current=.00025*(1+math.cos(math.pi*step/62600)) if lr is None else lr
    optimizer.param_groups[0]['lr']=current
    native.after_scheduler.last_epoch=step
    native.after_scheduler._step_count=step+1
    native.after_scheduler._last_lr=[current]
    branch=LearningRateBranch(native,optimizer,policy,step)
    return parameter,optimizer,native,branch


def main():
    plan=json.loads((Path(__file__).parent/'plan.json').read_text());policy=plan['policy']
    _,opt,native,trial=setup(38812,policy)
    _,control,control_native,_=setup(38812,policy)
    for _ in range(50):
        opt.step();trial.step();control.step();control_native.step()
        assert opt.param_groups[0]['lr']==control.param_groups[0]['lr']
    _,opt,native,trial=setup(42567,policy)
    opt.step();trial.step()
    assert trial.after_scheduler.last_epoch==42568
    assert math.isclose(opt.param_groups[0]['lr'],.00025*(1+math.cos(math.pi*.68)),abs_tol=1e-14)
    before=opt.param_groups[0]['lr'];opt.step();trial.step()
    assert before<opt.param_groups[0]['lr']<.00015
    assert trial.value_at_step(69*626)==.00015
    assert trial.value_at_step(84*626)==.00015
    assert trial.value_at_step(100*626)==.00001
    assert trial.value_at_step(101*626)==.00001
    values=[trial.value_at_step(i*626) for i in range(84,101)]
    assert all(a>=b for a,b in zip(values,values[1:]))
    step=90*626
    _,opt,native,trial=setup(step,policy,lr=trial.value_at_step(step))
    state=copy.deepcopy(trial.state_dict());optimizer_state=copy.deepcopy(opt.state_dict())
    _,resumed,resume_native,resume_trial=setup(step,policy,lr=trial.value_at_step(step))
    resumed.load_state_dict(optimizer_state);resume_native.load_state_dict(state)
    resume_trial=LearningRateBranch(resume_native,resumed,policy,step,state['learning_rate_branch'])
    for _ in range(50):
        opt.step();trial.step();resumed.step();resume_trial.step()
        assert opt.param_groups[0]['lr']==resumed.param_groups[0]['lr']
        assert trial.after_scheduler.last_epoch==resume_trial.after_scheduler.last_epoch
    result=dict(passed=True,original_cosine_replay_identical_for_tested_steps=True,
        ramp_continuous_at_epoch68=True,peak_and_floor_verified=True,monotone_final_cosine=True,
        checkpoint_resume_identical_for_tested_steps=True,
        schedule=[dict(epoch=e,original_lr=.00025*(1+math.cos(math.pi*e/100)),new_lr=trial.value_at_step(e*626)) for e in [62,68,69,70,76,80,84,90,94,98,100]])
    Path(plan['output'],'lr-policy-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
