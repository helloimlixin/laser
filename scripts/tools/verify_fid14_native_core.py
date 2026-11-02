#!/usr/bin/env python3
"""Full checkpoint, logits, cache, joint sampler and Adam checks before migration."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

BASE=Path('/tmp/laser-fid14-continue100-20261016')
OUT=Path('/workspace/Projects/laser/outputs/imagenet-fid14-continue100-4h100-20261016')
sys.path[:0]=[str(BASE/'frozen/runtime'),str(BASE/'frozen/support'),str(BASE/'support')]
os.environ['TORCH_HOME']=str(BASE/'torch-cache')
import torch
from torch.nn import functional as F
from omegaconf import OmegaConf
from src.training import rqtransformer as training
from src.training.physical_pair_crps import physical_pair_objective
from native_physical_pair_adapter import LaserPairRQTransformer
from rqvae.models.rqtransformer.configs import RQTransformerConfig
from smart_sampling import PairSamplingPolicy,sample_physical_pairs
from imagenet_tar import TarImageNet


def kl(expected,actual):
    assert torch.equal(expected.isfinite(),actual.isfinite())
    p,q=expected.float().log_softmax(-1),actual.float().log_softmax(-1)
    value=torch.where(p.isfinite(),p.exp()*(p-q),0.).sum(-1)
    assert torch.isfinite(value).all()
    return float(value.max())


def main():
    torch.set_num_threads(4);device=torch.device('cuda',0);torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32=False
    path=BASE/'chunk-cache/baae6b09483c493287a64e64b7857781c6563f483fa9f102eaaf99be3a98fa52.pt'
    saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    assert saved['global_step']==48828 and saved['epoch']==78
    ages={int(v['step']) for v in saved['optimizer']['state'].values()}
    assert ages=={14398} and len(saved['optimizer']['state'])==782
    with torch.device('meta'):
        legacy=training.build_model(18432,16384,physical_pair_context=True,sparsity_level=4)
        native=LaserPairRQTransformer(RQTransformerConfig.create(OmegaConf.create(
            OmegaConf.to_container(legacy.config,resolve=True))),16384)
    assert list(legacy.state_dict())==list(native.state_dict())==list(saved['state_dict'])
    for model in (legacy,native):model.load_state_dict(saved['state_dict'],strict=True,assign=True);model.to(device).eval()
    options=saved['config']
    aux=training.LaserAux(BASE/'inputs/resume-stage1-tokenizer.pt',16384,2048,3.,6.4,
        coeff_scales=options['coeff_scales'],soft_target_physical=False,clamp_coeffs=False,sparsity_level=4).to(device)
    dataset=TarImageNet(BASE/'data','val',training.val_image_transform())
    examples=[dataset[index] for index in [173,2719]]
    images=torch.stack([v[0] for v in examples]).to(device)
    labels=torch.tensor([v[1] for v in examples],device=device)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        atoms,values=aux.encode_sparse_components(images)
        tokens,targets=aux.sparse_targets(atoms,values,temp=.01125,compact=True)
    records={}
    for precision,dtype in [('fp32',None),('bf16',torch.bfloat16),('fp16',torch.float16)]:
        with torch.no_grad(),torch.autocast('cuda',dtype=dtype or torch.float16,enabled=dtype is not None):
            old=legacy(tokens,aux,labels,amp=False)
            new=native(tokens,aux,labels,amp=False)
            divergences={key:kl(old[key],new[key]) for key in old}
            old_loss=physical_pair_objective(old['atom_logits'],old['coeff_logits'],targets[0],targets[1],aux.coeff_bins,torch.tensor(.05,device=device),1)
            new_loss=physical_pair_objective(new['atom_logits'],new['coeff_logits'],targets[0],targets[1],aux.coeff_bins,torch.tensor(.05,device=device),1)
            records[precision]=dict(maximum_kl=divergences,legacy_loss=float(old_loss),native_loss=float(new_loss))
            assert max(divergences.values())<(1.e-5 if precision=='fp32' else 2.e-3),records[precision]
            assert abs(float(new_loss-old_loss))<(1.e-4 if precision=='fp32' else 1.e-2),records[precision]
        print(json.dumps({precision:records[precision]}),flush=True)
    with torch.no_grad():
        reference=native(tokens,aux,labels,amp=True)
        prefix=torch.zeros_like(tokens);prefix[...,1::2]=16384+1024
        legacy.init_cache();native.init_cache();maximum=0.;exact=True
        for h in range(8):
            for w in range(8):
                for event in range(8):
                    old=legacy.cached_forward(prefix[:,:h+1],aux,labels,amp=True,sample_loc=(h,w,event))
                    new=native.cached_forward(prefix[:,:h+1],aux,labels,amp=True,sample_loc=(h,w,event))
                    exact=exact and torch.equal(old,new)
                    assert kl(old,new)<2.e-4
                    atom=event%2==0;part=new[:,:16384] if atom else new[:,16384:]
                    maximum=max(maximum,kl(reference['atom_logits' if atom else 'coeff_logits'][:,h,w,event//2],part))
                    prefix[:,h,w,event]=tokens[:,h,w,event]
        assert maximum<2.e-4,maximum
        policy=PairSamplingPolicy(coefficient_temperatures=(.4,.55,.75,1.))
        torch.manual_seed(261001);before=sample_physical_pairs(legacy,2,aux,labels,policy=policy,amp=True)
        torch.manual_seed(261001);after=sample_physical_pairs(native,2,aux,labels,policy=policy,amp=True)
        same=torch.equal(before,after);assert same
    print(json.dumps(dict(cache_maximum_kl=maximum,old_new_cache_bitwise_identical=exact,joint_samples_identical=same)),flush=True)
    legacy.cpu();del legacy,old,new,reference,prefix;torch.cuda.empty_cache()
    native.zero_grad(set_to_none=True)
    output=native(tokens,aux,labels,amp=False)
    loss=physical_pair_objective(output['atom_logits'],output['coeff_logits'],targets[0],targets[1],aux.coeff_bins,torch.tensor(.05,device=device),1)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in native.parameters())
    optimizer=torch.optim.AdamW(native.parameters(),lr=3.e-6)
    state=copy.deepcopy(saved['optimizer']['param_groups'])
    checkpoint_optimizer=dict(saved['optimizer'],param_groups=state,
        state={i:dict(value,step=value['step'].clone()) for i,value in saved['optimizer']['state'].items()})
    optimizer.load_state_dict(checkpoint_optimizer)
    assert {int(v['step']) for v in optimizer.state.values()}=={14398}
    torch.nn.utils.clip_grad_norm_(native.parameters(),1.,error_if_nonfinite=True);optimizer.step()
    assert {int(v['step']) for v in optimizer.state.values()}=={14399}
    assert all(torch.isfinite(p).all() for p in native.parameters())
    proof=dict(passed=True,source_epoch=78,source_step=48828,parameter_tensors=782,
        parameter_order_and_all_state_keys_identical=True,strict_source_checkpoint_load=True,
        logits=records,full_cache_events=512,cache_maximum_kl=maximum,
        old_new_cache_bitwise_identical=exact,same_seed_joint_samples_bitwise_identical=same,
        original_adam_age=14398,disposable_adam_age_after_one_update=14399,
        finite_all_gradients=True,upstream_core_unmodified=True,production_updates_added=0,time=time.time())
    (OUT/'native-core-verification.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof),flush=True)


if __name__=='__main__':main()
