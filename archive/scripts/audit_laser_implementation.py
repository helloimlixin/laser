"""Read-only prediction diagnostics with a bounded allocation on a live GPU."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path('/workspace/Projects/laser')
BASE=Path('/tmp/laser-implementation-audit-20261015')
LASER=Path('/tmp/laser-rq-validated-ar-imagenet-20261015')
RQ=Path('/tmp/laser-original-rq-imagenet-20261014')
OUT=ROOT/'outputs/imagenet-implementation-audit-20261015'
sys.path[:0]=[str(BASE/'source'),str(BASE),str(ROOT),str(LASER/'upstream')]
os.environ['RQ_UPSTREAM_ROOT']=str(LASER/'upstream')

import numpy as np
import torch
from src.training.rq_imagenet_recipes import LaserRecipe,RQRecipe
from src.training.rq_imagenet_runtime import TarImageNet,write_json
from src.models.laser_historical_rq_adapter import build_historical_pair_model
from rqvae.utils.config import load_config,augment_arch_defaults
from rqvae.img_datasets.transforms import create_transforms


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=['laser','rq','historical'],required=True)
    parser.add_argument('--gpu',type=int,default=3)
    parser.add_argument('--images',type=int,default=64)
    parser.add_argument('--cache-sites',type=int,default=0,
        help='Also verify live-checkpoint scalar teacher forcing against cached sampling (up to 64 sites)')
    args=parser.parse_args()
    device=torch.device('cuda',args.gpu);torch.cuda.set_device(device);torch.set_num_threads(4)
    assert torch.cuda.mem_get_info(device)[0]>20*1024**3,'Insufficient spare memory for isolated diagnostics'
    torch.cuda.set_per_process_memory_fraction(.18,device)
    checkpoint={
        'laser':LASER/'checkpoint-cache/objects/8d5586f67424fffc8e70ac467357ecb0cdd41e0de0cde70d620ec278b1b1777a.pt',
        'rq':RQ/'checkpoint-cache/objects/438bef664e530d8983eee58d813280be0327e8350f372fbddd1f72527c51e432.pt',
        'historical':Path('/tmp/laser-historical-recovery-20261015/best-epoch5.pt')}[args.kind]
    saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    profile=RQRecipe(RQ) if args.kind=='rq' else LaserRecipe(LASER)
    if args.kind=='historical':profile.recipe.tokenizer.coefficient_scales=list(saved['config']['coeff_scales'])
    config=load_config((RQ if args.kind=='rq' else LASER)/'inputs/stage2-training-config.yaml')
    arch=augment_arch_defaults(config.arch);arch.vocab_size=16384
    with torch.device('meta'):
        model=(build_historical_pair_model(arch,num_atoms=16384,coefficient_bins=2048)
               if args.kind=='historical' else profile.model(arch))
    model.load_state_dict(saved['state_dict'],strict=True,assign=True)
    step=saved['global_step']
    assert step==(3175 if args.kind=='historical' else 8764)
    adam_ages=sorted({int(s['step']) for s in saved.get('optimizer',{}).get('state',{}).values()})
    scheduler_step=saved.get('scheduler',{}).get('last_epoch')
    del saved
    model.eval().requires_grad_(False).to(device)
    aux,_=profile.tokenizer(device);profile.adapt(model,aux)
    before=profile.digest(aux)
    sums={};rows={};hook_sites=0
    cache_report=None

    def add(key,value):
        value=value.detach().double().cpu();sums[key]=sums.get(key,torch.zeros_like(value))+value

    @torch.inference_mode()
    def evaluate(split):
        nonlocal cache_report
        data=TarImageNet(LASER/'data',split,create_transforms(config.dataset,split='val'),seed=261001)
        indices=np.linspace(0,len(data)-1,args.images,dtype=int).tolist()
        rows[split]=[]
        torch.manual_seed(261015);torch.cuda.manual_seed_all(261015)
        count=0
        for first in range(0,len(indices),2):
            pairs=[data[i] for i in indices[first:first+2]]
            images=torch.stack([p[0] for p in pairs]).to(device)
            labels=torch.tensor([p[1] for p in pairs],device=device)
            target,codes=profile.targets(aux,images)
            captured={}
            def capture(module,inputs,output):
                captured['head_input']=inputs[0].float().square().mean((0,2)).sqrt().cpu()
                captured['head_output']=output.float().square().mean((0,2)).sqrt().cpu()
            handle=model.head_transformer.register_forward_hook(capture)
            output=model(codes,aux,labels,amp=True);handle.remove()
            if split=='val' and first==0 and args.kind=='laser' and args.cache_sites:
                generated=torch.zeros_like(codes);generated[...,1::2]=16384+1024
                model.init_cache();kl=[];absolute=[]
                for site in range(min(args.cache_sites,64)):
                    h,w=divmod(site,8)
                    for event in range(8):
                        cached=model.cached_forward(generated[:,:h+1],aux,labels,
                            amp=True,sample_loc=(h,w,event))
                        key='atom_logits' if event%2==0 else 'coeff_logits'
                        actual=cached[:,:16384] if event%2==0 else cached[:,16384:]
                        expected=output[key][:,h,w,event//2]
                        assert torch.equal(torch.isfinite(actual),torch.isfinite(expected))
                        p=expected.float().log_softmax(-1);q=actual.float().log_softmax(-1)
                        kl.append(float(torch.where(p.isfinite(),p.exp()*(p-q),0).sum(-1).max()))
                        absolute.append(float((actual-expected)[torch.isfinite(expected)].abs().max()))
                        generated[:,h,w,event]=codes[:,h,w,event]
                model.init_cache()
                cache_report=dict(sites=min(args.cache_sites,64),events=len(kl),
                    max_kl=max(kl),max_absolute_logit_error=max(absolute),
                    finite_support_masks_identical=True,fp16=True)
                assert max(kl)<1.e-4,cache_report
            add(split+'/head_input_rms',captured['head_input']*len(images))
            add(split+'/head_output_rms',captured['head_output']*len(images))
            groups=([('code',output,codes)] if args.kind=='rq' else
                [('support',output['atom_logits'],codes[...,0::2]),('coefficient',output['coeff_logits'],codes[...,1::2]-16384)])
            for name,logits,truth in groups:
                logp=logits.float().log_softmax(-1);probs=logp.exp()
                nll=-logp.gather(-1,truth[...,None]).squeeze(-1).mean((1,2))
                add(split+'/'+name+'/nll',nll.sum(0))
                add(split+'/'+name+'/entropy',-torch.where(logp.isfinite(),probs*logp,0).sum(-1).mean((1,2)).sum(0))
                add(split+'/'+name+'/accuracy',(logits.argmax(-1)==truth).float().mean((1,2)).sum(0))
                if name=='code':q=target
                elif name=='coefficient':
                    centers=aux.coeff_bins[truth]*aux.coeff_scales
                    values=aux.coeff_bins[None,:]*aux.coeff_scales[:,None]
                    q=(-(centers[...,None]-values).square()/(2*.125**2)).softmax(-1)
                else:q=None
                if q is not None:
                    add(split+'/'+name+'/soft_ce',-(q*logp).sum(-1).mean((1,2)).sum(0))
                    add(split+'/'+name+'/target_entropy',-(q*q.clamp_min(1.e-30).log()).sum(-1).mean((1,2)).sum(0))
                for i in range(len(images)):
                    if name==groups[0][0]:rows[split].append(dict(index=indices[first+i]))
                    rows[split][count+i][name+'_nll']=nll[i].tolist()
            if args.kind!='rq':
                atoms=codes[...,0::2];raw=aux.dictionary.T[atoms]
                _,basis,coefficients,_=model._physical_pairs(codes,aux)
                add(split+'/raw_basis_cosine',(raw*basis).sum(-1).mean((1,2)).sum(0))
                # Dictionary retrieval from Q measures how much the representation shifts atom identity.
                recovered=(basis.reshape(-1,256)@aux.dictionary).argmax(-1).reshape_as(atoms)
                add(split+'/basis_nearest_raw_atom_accuracy',(recovered==atoms).float().mean((1,2)).sum(0))
                reconstructed=(basis*coefficients[...,None]).sum(-2)
                add(split+'/physical_latent_energy',reconstructed.square().sum(-1).mean((1,2)).sum())
            if split=='val':
                hook=model.body_transformer.register_forward_hook(lambda module,inputs,output:output.roll(1,0))
                shuffled=model(codes,aux,labels,amp=True);hook.remove()
                for name,_,truth in groups:
                    logits=shuffled if args.kind=='rq' else shuffled['atom_logits' if name=='support' else 'coeff_logits']
                    add('val/'+name+'/shuffled_nll',-logits.float().log_softmax(-1).gather(-1,truth[...,None]).squeeze(-1).mean((1,2)).sum(0))
                if args.kind!='rq':
                    changed=codes.clone();changed[...,1::2]=16384+1024
                    prediction=model(changed,aux,labels,amp=True)['atom_logits']
                    add('val/support/zero_coefficient_context_nll',-prediction.float().log_softmax(-1).gather(-1,codes[...,0::2,None]).squeeze(-1).mean((1,2)).sum(0))
            count+=len(images)
            del output,groups,target,codes,images,labels
            if count%16==0:print(json.dumps(dict(kind=args.kind,split=split,images=count)),flush=True)
        return count

    started=time.time()
    for split in ['train','val']:evaluate(split)
    report=dict(kind=args.kind,checkpoint=str(checkpoint),checkpoint_step=step,
        images_per_split=args.images,preprocessing='identical deterministic center crops for both splits',
        training_center_crops_are_not_a_replay_of_seen_augmentation_batches=True,
        adam_ages=adam_ages,scheduler_step=scheduler_step,metrics={k:(v/args.images).tolist() for k,v in sums.items()},
        paired_image_rows=rows,elapsed_seconds=time.time()-started,
        peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3,auxiliary_losses=0,
        live_training_modified=False,teacher_forcing_cache_parity=cache_report)
    assert profile.digest(aux)==before
    if args.kind!='rq':
        report['coefficient_center_quantization_kl_upper_bound_per_depth']=(
            (aux.coeff_scales*6/2047/2).square()/(2*.125**2)).cpu().tolist()
    write_json(OUT/(args.kind+'-diagnostics.json'),report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['paired_image_rows']}),flush=True)


if __name__=='__main__':main()
