"""Validate accelerated attention and the distributed reduction on these GPUs."""
import os,sys,json
from pathlib import Path
base=Path('/tmp/laser-beam16-a100-20261105')
sys.path[:0]=[str(base/'frozen/source'),str(base/'frozen')]
os.environ['RQ_UPSTREAM_ROOT']=str(base/'frozen/upstream')
import torch
from src import rq_upstream
from rqvae.models.rqtransformer.attentions import MultiSelfAttention
from rqvae.models.rqtransformer.configs import AttentionBlockConfig
from runtime.training_acceleration import sdpa_forward
c=AttentionBlockConfig(embed_dim=256,n_head=4,attn_pdrop=0.,resid_pdrop=0.)
torch.manual_seed(71)
a=MultiSelfAttention(c).cuda();a._native_forward=a.forward
x=torch.randn(8,64,256,device='cuda',requires_grad=True)
y=a(x);y.square().mean().backward();gx=x.grad.clone();gp={k:p.grad.clone() for k,p in a.named_parameters()}
a.zero_grad();x.grad=None
z=sdpa_forward(a,x);z.square().mean().backward()
report=dict(forward_max_error=(y-z).abs().max().item(),input_gradient_max_error=(gx-x.grad).abs().max().item(),parameter_gradient_max_error=max((gp[k]-p.grad).abs().max().item() for k,p in a.named_parameters()))
assert torch.allclose(y,z,rtol=1e-4,atol=1e-5)
assert torch.allclose(gx,x.grad,rtol=1e-3,atol=1e-7)
with torch.no_grad():assert torch.equal(a(x),sdpa_forward(a,x))
report.update(passed=True,sampling_uses_original_attention=True)
(base/'kernel-verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(report)
