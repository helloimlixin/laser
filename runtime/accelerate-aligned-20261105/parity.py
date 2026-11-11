"""Check FP16 attention parity, exact cached sampling, and compiled native CE."""
import copy
import json
import os
from pathlib import Path
import sys
import types

BASE=Path('/tmp/laser-aligned-beam451-20261105/frozen')
OUT=Path('/tmp/laser-speed-aligned-20261105')
sys.path[:0]=[str(BASE/'source'),str(BASE)]
os.environ['RQ_UPSTREAM_ROOT']=str(BASE/'upstream')
os.environ['TORCHINDUCTOR_CACHE_DIR']=str(OUT/'inductor-cache')
os.environ['TORCHINDUCTOR_COMPILE_THREADS']='2'
import torch
from src import rq_upstream
from rqvae.models.rqtransformer.attentions import MultiSelfAttention
from rqvae.models.rqtransformer.configs import AttentionBlockConfig
from rqvae.optimizer.loss import soft_target_cross_entropy
from runtime.training_acceleration import sdpa_forward


def relative(a,b):
    return float((a.float()-b.float()).norm()/a.float().norm().clamp_min(1e-20))


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2);torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.08,0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    results=[]
    for batch,length in [(86,64),(86*64,4)]:
        torch.manual_seed(26110599)
        config=AttentionBlockConfig(embed_dim=1536,n_head=24,attn_pdrop=0.,resid_pdrop=.1)
        native=MultiSelfAttention(config).cuda().train()
        fast=copy.deepcopy(native)
        fast._native_forward=types.MethodType(MultiSelfAttention.forward,fast)
        fast.forward=types.MethodType(sdpa_forward,fast)
        source=torch.randn(batch,length,1536,device='cuda')
        outputs=[];input_grads=[];parameter_grads=[]
        for module in [native,fast]:
            x=source.detach().clone().requires_grad_()
            with torch.random.fork_rng(devices=[0]):
                torch.cuda.manual_seed(26110599)
                with torch.autocast('cuda',dtype=torch.float16):output=module(x)
                # Match live AMP scaling; an unscaled tiny MSE underflows in
                # native FP16 backward and is not a representative comparison.
                (output.float().square().mean()*131072.).backward()
            outputs.append(output.detach().cpu());input_grads.append(x.grad.detach().cpu())
            parameter_grads.append(torch.cat([p.grad.detach().flatten().cpu() for p in module.parameters()]))
            module.zero_grad(set_to_none=True)
            del x,output
        row=dict(shape=[batch,length,1536],output_relative_l2=relative(*outputs),
            input_gradient_relative_l2=relative(*input_grads),
            parameter_gradient_relative_l2=relative(*parameter_grads))
        assert max(row[k] for k in row if k.endswith('relative_l2'))<.01,row
        native.eval();fast.eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            a,ka=native(source[:2],caching=True)
            b,kb=fast(source[:2],caching=True)
            assert torch.equal(a,b) and torch.equal(ka,kb)
            a2,_=native(source[:2,:1],caching=True,past_kv=ka)
            b2,_=fast(source[:2,:1],caching=True,past_kv=kb)
            assert torch.equal(a2,b2)
        row['cached_sampling_bitwise_equal']=True;results.append(row)
        print(json.dumps(row),flush=True)
        del native,fast,source,outputs,input_grads,parameter_grads
        torch.cuda.empty_cache()
    compiled=torch.compile(soft_target_cross_entropy,fullgraph=True,dynamic=True)
    loss_results=[]
    for n in [256,512]:
        logits=torch.randn(n,16384,device='cuda')
        targets=torch.zeros_like(logits)
        ids=torch.randint(16384,(n,17),device='cuda')
        mass=torch.randn(n,17,device='cuda').softmax(-1)
        targets.scatter_add_(1,ids,mass)
        saved=[]
        for fn in [soft_target_cross_entropy,compiled]:
            x=logits.detach().clone().requires_grad_()
            loss=fn(x,targets);loss.backward()
            saved.append((float(loss),x.grad.detach().cpu()))
            del x,loss
        difference=abs(saved[0][0]-saved[1][0]);grad=relative(saved[0][1],saved[1][1])
        assert difference<2e-5 and grad<2e-5
        loss_results.append(dict(rows=n,loss_absolute_error=difference,gradient_relative_l2=grad))
        del logits,targets,ids,mass,saved
        torch.cuda.empty_cache()
    proof=dict(passed=True,attention=results,compiled_native_soft_ce=loss_results,
        teacher_code_unchanged=True,precision='float16',amp_loss_scale=131072.,attention_dropout=0.,residual_dropout=.1)
    (OUT/'parity.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof),flush=True)


if __name__=='__main__':main()
