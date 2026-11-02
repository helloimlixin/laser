"""Checkpointed cosine continuation that reaches exactly zero at epoch100."""
import copy
import json
import math
import os
from pathlib import Path

SELECTION=json.loads((Path(os.environ['LASER_RUN_BASE'])/'selection.json').read_text())
ANCHOR_STEP=int(SELECTION['source_global_step'])
END=62600
PEAK=1.5e-4


def coefficient_sigma(step):
    return float(SELECTION['coefficient_sigma'])


class RefinementSchedule:
    kind='imagenet-selected-checkpoint-cosine-to-zero-at100-v1'
    def __init__(self,optimizer,*,completed_steps,state_dict):
        self.optimizer=optimizer
        self.last_epoch=int(completed_steps)
        self.policy=dict(anchor_step=ANCHOR_STEP,total_steps=END,initial_lr=PEAK,min_lr=0.)
        if not ANCHOR_STEP<=self.last_epoch<=END:raise ValueError('Continuation step outside schedule')
        if not state_dict or state_dict['last_epoch']!=self.last_epoch:raise ValueError('Missing synchronized source scheduler')
        if state_dict.get('kind')==self.kind:
            if state_dict['policy']!=self.policy:raise ValueError('Changed continuation schedule')
            self.best=state_dict['best'];self.last_observation_step=state_dict['last_observation_step']
        else:
            if self.last_epoch!=ANCHOR_STEP:raise ValueError('Initial scheduler reset must use the selected source')
            self.best=float(SELECTION['source_fid']);self.last_observation_step=ANCHOR_STEP
        self._apply()
    def rate(self,step):
        if step>=END:return 0.
        return PEAK*.5*(1+math.cos(math.pi*(step-ANCHOR_STEP)/(END-ANCHOR_STEP)))
    def _apply(self):
        value=self.rate(self.last_epoch)
        for group in self.optimizer.param_groups:group['lr']=value;group['initial_lr']=PEAK
    def step(self):
        if self.last_epoch>=END:raise ValueError('Schedule exhausted')
        self.last_epoch+=1;self._apply()
    def get_last_lr(self):return [g['lr'] for g in self.optimizer.param_groups]
    def observe(self,fid):
        if not math.isfinite(fid):raise ValueError('Nonfinite FID')
        self.best=min(self.best,float(fid));self.last_observation_step=self.last_epoch
        return dict(fid=fid,decision='cosine schedule retained',step=self.last_epoch,
            lr_before=self.get_last_lr()[0],lr_after=self.get_last_lr()[0],best_fid=self.best,reductions=0)
    def state_dict(self):
        return dict(kind=self.kind,policy=copy.deepcopy(self.policy),last_epoch=self.last_epoch,
            best=self.best,last_observation_step=self.last_observation_step,_last_lr=self.get_last_lr())
