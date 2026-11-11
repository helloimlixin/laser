"""Original cosine clock, exact replay teacher boundary, and FP32 EMA."""
import copy
import math
import torch


def teacher_temperature(step,policy):
    return policy['initial'] if step<policy['change_step'] else policy['final']


class ModelEMA:
    def __init__(self,model,decay):
        self.model=copy.deepcopy(model).requires_grad_(False).eval()
        self.decay=float(decay);self.updates=0

    @torch.no_grad()
    def update(self,model):
        torch._foreach_lerp_(list(self.model.parameters()),list(model.parameters()),1.-self.decay)
        for dst,src in zip(self.model.buffers(),model.buffers()):dst.copy_(src)
        self.updates+=1


class LearningRateRefinement:
    def __init__(self,original,optimizer,policy,step,saved_policy=None):
        self.original=original;self.optimizer=optimizer;self.policy=dict(policy)
        self.after_scheduler=original.after_scheduler
        assert policy['kind']=='original_cosine_100'
        assert self.after_scheduler.last_epoch==step and self.after_scheduler.T_max==62600
        if saved_policy is not None:assert saved_policy==self.policy
        assert all(math.isclose(group['lr'],self.value_at_step(step),rel_tol=1e-8,abs_tol=1e-12) for group in optimizer.param_groups)

    def value_at_step(self,step):
        return self.policy['base_lr']*.5*(1.+math.cos(math.pi*step/self.policy['original_horizon_steps']))

    def step(self):self.original.step()

    def get_last_lr(self):return self.original.get_last_lr()

    def state_dict(self):
        state=self.original.state_dict();state['late_refinement']=self.policy;return state

    def load_state_dict(self,state):
        assert state['late_refinement']==self.policy
        self.original.load_state_dict(state)
