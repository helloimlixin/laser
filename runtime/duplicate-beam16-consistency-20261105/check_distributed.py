import os,sys,json,importlib.util
from pathlib import Path
import torch
base=Path('/tmp/laser-beam16-duplicate-20261105')
torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
torch.distributed.init_process_group('nccl')
spec=importlib.util.spec_from_file_location('trial',base/'runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
p=torch.nn.Parameter(torch.zeros(8,device='cuda'));p.grad=torch.full_like(p,float(m.rank()+1))
m.synchronize_gradients([p]);assert torch.equal(p.grad,torch.full_like(p,3.))
n,means=m.global_metrics(m.rank()+1,{'loss':float(m.rank()+1)},p.device)
assert n==3 and abs(means['loss']-5/3)<1e-12
if m.rank()==0:print(json.dumps(dict(passed=True,reduction='SUM',weighted_global_metric=True)))
torch.distributed.destroy_process_group()
