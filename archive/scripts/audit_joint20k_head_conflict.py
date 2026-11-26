#!/usr/bin/env python3
"""CPU-only probe of shared-classifier gradients across physical codebooks."""
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path('/workspace/Projects/laser')
BASE=Path('/tmp/laser-rq-joint20k-depth5-rqnoise-k2048-rngfix-8h100-imagenet-20261017')
OUT=ROOT/'outputs/imagenet-laser-joint20k-prior-diagnosis-20261017/head-conflict-epoch4'


def write(name,value):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def main():
    os.environ['RQ_UPSTREAM_ROOT']=str(BASE/'upstream')
    sys.path[:0]=[str(BASE/'source'),str(BASE/'upstream'),str(BASE),str(BASE/'runtime')]
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from src.training.laser_joint_recipe import JointLaserRecipe
    from src.training.rq_imagenet_runtime import TarImageNet
    from rqvae.utils.config import load_config
    from rqvae.img_datasets.transforms import create_transforms
    torch.set_num_threads(32);torch.manual_seed(261018401)
    # Check the manual derivative against autograd before using the probe.
    x=torch.randn(11,7);w=torch.randn(9,7,requires_grad=True);y=torch.rand(11,9)
    z=x@w.T;loss=-(y*z.log_softmax(-1)).sum()/len(x);loss.backward()
    expected=(z.detach().softmax(-1)*y.sum(-1,keepdim=True)-y).T@x/len(x)
    torch.testing.assert_close(w.grad,expected)
    profile=JointLaserRecipe(BASE)
    saved=torch.load('/tmp/laser-joint20k-noise-ablation-20261017/control-epoch004.pt',
        map_location='cpu',mmap=True,weights_only=False)
    profile.verify_resume(saved);assert saved['global_step']==2504
    with torch.device('meta'):model=profile.model(OmegaConf.create(saved['config']['arch']))
    model.load_state_dict(saved['state_dict'],strict=True,assign=True)
    model=model.requires_grad_(False).eval()
    weight=model.classifier.linear.weight
    wstates=[s for s in saved['optimizer']['state'].values() if s['exp_avg_sq'].shape==weight.shape]
    bstates=[s for s in saved['optimizer']['state'].values() if s['exp_avg_sq'].shape==model.classifier.linear.bias.shape]
    assert len(wstates)==len(bstates)==1
    eps=saved['optimizer']['param_groups'][0]['eps']
    wp=wstates[0]['exp_avg_sq'].sqrt().add(eps).rsqrt()
    bp=bstates[0]['exp_avg_sq'].sqrt().add(eps).rsqrt()
    del saved,wstates,bstates
    aux,_=profile.tokenizer(torch.device('cpu'))
    config=load_config(BASE/'inputs/stage2-training-config.yaml')
    data=TarImageNet(BASE/'data','val',create_transforms(config.dataset,split='val'))
    indices=np.random.default_rng(261018401).permutation(len(data))[:128].tolist()
    depth=5
    gw=torch.zeros(depth,*weight.shape);gb=torch.zeros(depth,len(model.classifier.linear.bias))
    capture=[]
    hook=model.classifier.layer_norm.register_forward_hook(lambda _,__,output:capture.append(output.detach()))
    count=0
    with torch.inference_mode():
        for first in range(0,len(indices),8):
            batch=[data[i] for i in indices[first:first+8]]
            images=torch.stack([v[0] for v in batch]);labels=torch.tensor([v[1] for v in batch])
            targets,codes=profile.targets(aux,images)
            logits=model(codes,aux,labels,amp=False)
            hidden=capture.pop().reshape(-1,depth,weight.shape[1])
            p=logits.softmax(-1)
            delta=(p*targets.sum(-1,keepdim=True)-targets).reshape(-1,depth,weight.shape[0])
            for d in range(depth):
                gw[d].addmm_(delta[:,d].T,hidden[:,d]);gb[d]+=delta[:,d].sum(0)
            count+=len(images)*64
            write('status.json',dict(phase='probing',images=first+len(batch),target=len(indices),pid=os.getpid(),time=time.time()))
    hook.remove();gw/=count;gb/=count

    def summarize(weights,bias):
        g=torch.cat((weights.flatten(1),bias),1)
        norms=g.norm(dim=-1)
        cosine=(g@g.T)/(norms[:,None]*norms[None,:]).clamp_min(1.e-30)
        shared=g.mean(0)
        return dict(per_depth_gradient_norm=norms.tolist(),cosine_matrix=cosine.clamp(-1,1).tolist(),
            per_depth_alignment_with_shared_update=((g@shared)/(norms*shared.norm()).clamp_min(1.e-30)).tolist(),
            negative_cross_depth_pairs=int((cosine.triu(1)<0).sum()))

    result=dict(completed=True,checkpoint_global_step=2504,images=len(indices),val_indices=indices,
        raw=summarize(gw,gb),adam_metric=summarize(gw*wp[None],gb*bp[None]),
        cpu_float32=True,soft_targets_and_stochastic_prefixes=True,
        derivative_checked_against_autograd=True,model_updated=False,
        codebooks_shared_exact=bool((aux.quantizer.coefficients==aux.quantizer.coefficients[:1]).all()),
        per_depth_coefficient_rms=aux.quantizer.coefficients.square().mean(-1).sqrt().tolist(),
        limits='One validation probe, no dropout. Adam metric uses the saved diagonal second moment, without momentum history. Gradient conflict does not prove a generation bottleneck.',time=time.time())
    write('report.json',result);write('status.json',dict(phase='completed',time=time.time()))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
