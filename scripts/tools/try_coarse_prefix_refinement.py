"""Conditional learnability probe for whole finer pairs with full coarse maps."""
import argparse
from dataclasses import replace,fields
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'runtime')]

import torch

from src.models.coarse_prefix_pair_refiner import CoarsePrefixPairRefiner
from src.models.typed_pair_rq import TypedPairCodebook
from src.models.structured_pair_energy import structured_pair_cross_entropy
from src.training.typed_pair_targets import typed_pair_targets,typed_pair_cross_entropy
from scripts.tools.try_two_vocabulary_pairs import record


def finer(target):
    return replace(target,candidate_atoms=target.candidate_atoms[...,1:,:],
        candidate_bins=target.candidate_bins[...,1:,:],pair_weights=target.pair_weights[...,1:,:],
        coefficient_weights_given_atom=target.coefficient_weights_given_atom[...,1:,:],
        residual_energies=target.residual_energies[...,1:])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--codebook',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--eval-images',type=int,default=256)
    p.add_argument('--key-file',type=Path)
    p.add_argument('--head',choices=['physical','conditional'],default='physical')
    p.add_argument('--initial-checkpoint',type=Path)
    p.add_argument('--eval-every',type=int,default=100)
    p.add_argument('--run-suffix',default='')
    p.add_argument('--local-only',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(8);torch.manual_seed(261030)
    torch.backends.cuda.matmul.allow_tf32=False
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'result.json').exists(): raise FileExistsError('trial exists')
    data=torch.load(args.cache,map_location='cpu',mmap=True,weights_only=False)
    saved=torch.load(args.codebook,map_location='cuda',weights_only=True)
    book=TypedPairCodebook(saved['dictionary'],saved['coeff_bins'],depth=4).cuda()
    model=CoarsePrefixPairRefiner(256,book.dictionary,book.coeff_bins,
        conditional_coefficients=args.head=='conditional').cuda()
    if args.initial_checkpoint:
        initial=torch.load(args.initial_checkpoint,map_location='cpu',mmap=True,weights_only=False)
        missing,extra=model.load_state_dict(initial['state_dict'],strict=False)
        if extra or any(not key.startswith(('coefficient_atom_projection.','coefficient_refiner.')) for key in missing):
            raise ValueError((missing,extra))
        del initial
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=1e-4,fused=True)
    config=dict(conditional_finer_levels_only=True,full_coarse_spatial_context=not args.local_only,
        within_level_conditional_independence=True,coarse_level_must_be_supplied=True,
        joint_physical_head=args.head=='physical',decoder_head=args.head,
        initial_checkpoint=str(args.initial_checkpoint) if args.initial_checkpoint else None,
        teacher='stochastic whole pairs',relative_temperature=.003,
        teacher_top_k=1024,steps=args.steps,batch_size=args.batch_size,train_images=len(data['train']['latent']),
        eval_images=args.eval_images,lr=3e-4,warmup_steps=50,seed=261030,
        parameters=sum(v.numel() for v in model.parameters()),
        evaluation_split='image-disjoint ImageNet train images; no official validation or FID')
    record(args.output/'config.json',config)
    wb=None
    if args.key_file:
        import wandb
        os.environ['WANDB_API_KEY']=args.key_file.read_text().strip()
        wb=wandb.init(entity='helloimlixin-rutgers',project='laser',
            id=f'imagenet-full-coarse-pair-refiner-{args.steps}steps-20261029'+args.run_suffix,
            group='two-vocab-structured-pair-investigation-20261029',config=config,dir=str(args.output))
    train=data['train'];heldout=data['heldout']
    assert not set(train['filenames']) & set(heldout['filenames'])
    latent_train=train['latent'].cuda();labels_train=train['labels'].cuda()
    latent_eval=heldout['latent'][:args.eval_images].cuda();labels_eval=heldout['labels'][:args.eval_images].cuda()
    fixed_hard=typed_pair_targets(book,latent_eval,top_k=1,stochastic=False)

    @torch.no_grad()
    def evaluate(step,local_only=None):
        local_only=args.local_only if local_only is None else local_only
        model.eval();totals=torch.zeros(3,3,device='cuda');per_image=[]
        for first in range(0,len(latent_eval),4):
            end=min(first+4,len(latent_eval))
            target=finer(type(fixed_hard)(**{field.name:getattr(fixed_hard,field.name)[first:end] for field in fields(fixed_hard)}))
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=model(target.tokens,book,labels_eval[first:end],local_only=local_only)
                _,a,c=typed_pair_cross_entropy(output,target)
            totals+=torch.stack((a,c,a+c))*(end-first)
            if step==args.steps:
                atoms,coeffs=book.fields(target.tokens)
                al=-output['atom_logits'].float().log_softmax(-1).gather(-1,atoms[...,1:,None]).squeeze(-1)
                cl=-output['coeff_logits'].float().log_softmax(-1).gather(-1,coeffs[...,1:,None]).squeeze(-1)
                per_image.append(torch.stack((al,cl),-1).mean((1,2)).cpu())
        totals/=len(latent_eval);model.train()
        if per_image:torch.save(torch.cat(per_image),args.output/('per-image-nll.pt' if local_only==args.local_only else 'per-image-alternate.pt'))
        return dict(step=step,local_only=local_only,atom_nll=totals[0].tolist(),
            coeff_nll=totals[1].tolist(),joint_nll=totals[2].tolist(),mean_joint_nll=float(totals[2].mean()))

    history=[evaluate(0)];print(json.dumps(history[-1]),flush=True)
    order=torch.Generator(device='cuda').manual_seed(300)
    teacher=torch.Generator(device='cuda').manual_seed(301)
    permutation=torch.randperm(len(latent_train),device='cuda',generator=order);cursor=0
    torch.manual_seed(261030);started=time.monotonic()
    for step in range(1,args.steps+1):
        if cursor+args.batch_size>len(permutation):
            permutation=torch.randperm(len(latent_train),device='cuda',generator=order);cursor=0
        ids=permutation[cursor:cursor+args.batch_size];cursor+=args.batch_size
        target=typed_pair_targets(book,latent_train[ids],top_k=1024,relative_temperature=.003,generator=teacher)
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups: group['lr']=3e-4*min(1.,step/50)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output=model(target.tokens,book,labels_train[ids],local_only=args.local_only)
            if args.head=='physical':loss,a,c=structured_pair_cross_entropy(output,finer(target),book.coeff_bins)
            else:loss,a,c=typed_pair_cross_entropy(output,finer(target))
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite refiner loss')
        optimizer.step()
        if step%20==0 or step==1:
            metrics=dict(step=step,joint_ce=float(loss.detach()),atom_ce=a.detach().tolist(),
                coeff_ce=c.detach().tolist(),grad_norm=float(grad),seconds=time.monotonic()-started)
            print(json.dumps(metrics),flush=True)
            if wb: wb.log({'train/joint_ce':metrics['joint_ce']},step=step)
        if step%args.eval_every==0 or step==args.steps:
            history.append(evaluate(step));print(json.dumps(history[-1]),flush=True)
            record(args.output/'metrics.json',history)
            if wb: wb.log({f'heldout/atom_nll/depth{d+2}':x for d,x in enumerate(history[-1]['atom_nll'])},step=step)
    alternate=evaluate(args.steps,local_only=not args.local_only)
    result=dict(config=config,evaluation=history,alternate_context_evaluation=alternate,
                seconds=time.monotonic()-started,all_steps_finite=True)
    torch.save(dict(state_dict=model.state_dict(),codebook=book.state_dict(),optimizer=optimizer.state_dict(),
        config=config,step=args.steps,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
        teacher_rng=teacher.get_state(),order_rng=order.get_state(),permutation=permutation,cursor=cursor),
        args.output/'checkpoint.pt')
    record(args.output/'result.json',result)
    if wb:
        wb.summary['mean_finer_joint_nll']=history[-1]['mean_joint_nll'];wb.finish()
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
