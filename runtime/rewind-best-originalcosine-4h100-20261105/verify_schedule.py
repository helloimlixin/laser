"""Verify that unscaling changes LR amplitude, preserving the cosine and Adam clocks."""
import math,os,sys,json
from pathlib import Path
b=Path('/tmp/laser-best-originalcosine-4h100-20261105/frozen')
os.environ['RQ_UPSTREAM_ROOT']=str(b/'upstream')
sys.path[:0]=[str(b/'source'),str(b)]
import torch
from omegaconf import OmegaConf
from src import rq_upstream
from rqvae.optimizer.scheduler import create_scheduler
from runtime.lr_recovery import apply_lr_amplitude
p=torch.load(b/'inputs/resume-checkpoint.pt',map_location='cpu',mmap=True,weights_only=False)
param=torch.nn.Parameter(torch.zeros(1))
opt=torch.optim.AdamW([param],lr=.0005,betas=(.9,.95),weight_decay=.0001)
cfg=OmegaConf.load(b/'inputs/stage2-training-config.yaml')
sched=create_scheduler(opt,cfg.optimizer.warmup,626,100)
sched.load_state_dict(p['scheduler'])
opt.param_groups[0]['lr']=p['optimizer']['param_groups'][0]['lr']
opt.param_groups[0]['initial_lr']=p['optimizer']['param_groups'][0]['initial_lr']
opt.state[param]=dict(step=torch.tensor(45698.),exp_avg=torch.ones_like(param)*.7,exp_avg_sq=torch.ones_like(param)*.9)
state_before={k:v.clone() for k,v in opt.state[param].items()}
proof=apply_lr_amplitude(opt,sched,previous_factor=.15,desired_factor=1.0)
assert sched.after_scheduler.last_epoch==45698 and sched.after_scheduler.T_max==62600
assert math.isclose(opt.param_groups[0]['lr'],8.467203366908699e-5,rel_tol=1e-10)
assert math.isclose(sched.after_scheduler.base_lrs[0],.0005,rel_tol=1e-12)
for k,v in state_before.items():assert torch.equal(v,opt.state[param][k])
for i in range(1,11):
 opt.step();sched.step()
 expected=.0005*.5*(1+math.cos(math.pi*(45698+i)/62600))
 assert math.isclose(opt.param_groups[0]['lr'],expected,rel_tol=1e-10)
assert sched.after_scheduler.last_epoch==45708
proof.update(passed=True,original_base_lr=.0005,adam_moments_and_age_unchanged_by_unscaling=True,ten_following_cosine_steps_match_original_formula=True)
out=Path('/workspace/Projects/laser/outputs/imnet-j16384-best14574-originalcosine-continue100-4h100-20261105')
(out/'schedule-verification.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof,indent=2))
