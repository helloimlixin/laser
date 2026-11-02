#!/usr/bin/env python3
"""Matched short continuations using native RQ training and LASER targets."""
import argparse
from contextlib import nullcontext
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path('/workspace/Projects/laser')
BASE = Path('/tmp/laser-compound512-atom-identity-pilot-20261016')
OUT = ROOT/'outputs/imagenet-compound512-atom-identity-pilot-20261016'
GROUP = OUT.name
os.environ['RQ_UPSTREAM_ROOT'] = str(BASE/'upstream')
os.environ['TORCH_HOME'] = str(BASE/'torch-cache')
sys.path[:0] = [str(BASE/'source'),str(BASE/'runtime'),str(BASE/'upstream'),str(ROOT)]
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Subset
from omegaconf import OmegaConf
from src.models.laser_atom_identity_adapter import build_atom_identity_model
from src.training.laser_compound_replay_recipe import CompoundReplayRecipe
from src.training.laser_pair_ablation import restore_named_adam
from src.training.rq_imagenet_runtime import TarImageNet,cpu_snapshot,write_json
from src.training.exact_global_batch import ExactGlobalBatchSampler
from src.training.rq_imagenet_shared import training_backward
from src.training import rq_imagenet_shared as shared
from runtime.amp_replay import ReplayWindow,recoverable_amp_step
from rqvae.utils.config import load_config,augment_optimizer_defaults
from rqvae.img_datasets.transforms import create_transforms
from rqvae.optimizer.optimizer import create_resnet_optimizer
from rqvae.optimizer.scheduler import create_scheduler

RUNS = {'control':'imagenet-compound512-identity-control128-4h100-20261016',
        'atom_identity':'imagenet-compound512-atom-identity128-4h100-20261016'}


def restore_rng(state, device):
    random.setstate(state['python']);np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu']);torch.cuda.set_rng_state(state['cuda'],device)


def identity_model(profile, arch, branch):
    if branch == 'control':return profile.model(arch)
    return build_atom_identity_model(arch,num_atoms=16384,coefficient_bins=512)


def init():
    rank,world,local = (int(os.environ[k]) for k in ['RANK','WORLD_SIZE','LOCAL_RANK'])
    assert world == 4
    torch.set_num_threads(4);device=torch.device('cuda',local);torch.cuda.set_device(device)
    from datetime import timedelta
    dist.init_process_group('nccl',device_id=device,timeout=timedelta(minutes=60))
    return rank,world,device


@torch.inference_mode()
def prepare_probe(rank,world,device):
    profile=CompoundReplayRecipe(BASE);aux,_=profile.tokenizer(device)
    config=load_config(BASE/'inputs/stage2-training-config.yaml')
    excluded=json.loads((OUT/'prior-validation-indices.json').read_text())
    pool=np.setdiff1d(np.arange(50000),np.asarray(excluded))
    indices=np.random.default_rng(261016781).choice(pool,1024,replace=False).tolist()
    assert not set(indices)&set(excluded) and len(set(indices))==1024
    data=TarImageNet(BASE/'data','val',create_transforms(config.dataset,split='val'))
    local_indices=indices[rank::world]
    loader=DataLoader(Subset(data,local_indices),batch_size=4,num_workers=4,pin_memory=True)
    torch.manual_seed(261016781+rank);torch.cuda.manual_seed(261016781+rank)
    collected={key:[] for key in ['codes','coefficients','labels']}
    image_hash=hashlib.sha256();before=profile.digest(aux)
    for cursor,(images,labels) in enumerate(loader):
        image_hash.update(images.numpy().tobytes())
        targets,codes=profile.targets(aux,images.to(device,non_blocking=True))
        collected['codes'].append(codes.cpu())
        collected['coefficients'].append(targets['coefficients'].cpu())
        collected['labels'].append(labels)
        if cursor%16==0:
            write_json(OUT/f'probe-progress-rank{rank}.json',dict(images=(cursor+1)*4,time=time.time()))
    assert profile.digest(aux)==before
    cache={key:torch.cat(values) for key,values in collected.items()}
    cache['indices']=local_indices
    cache['metadata']=dict(images=256,image_sha256=image_hash.hexdigest(),seed=261016781+rank,
        coefficient_bins=512,target_temperature=.03125,source_stage1_unchanged=True)
    (BASE/'probe').mkdir(exist_ok=True)
    path=BASE/f'probe/rank{rank}.pt';torch.save(cache,path)
    record=dict(rank=rank,path=str(path),bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**cache['metadata'])
    all_records=[None]*world;dist.all_gather_object(all_records,record)
    if rank==0:
        write_json(OUT/'probe-verification.json',dict(passed=True,images=1024,indices=indices,
            excluded_previous_validation_images=len(excluded),ranks=all_records,
            stochastic_coefficient_prefixes_precomputed_once=True,training_updates_added=0))
    dist.barrier()


@torch.no_grad()
def evaluate(model,aux,probe,branch,step,device,rank,world,run=None):
    was_training=model.training;model.eval();rows=[]
    with torch.random.fork_rng(devices=[device.index]):
        for first in range(0,len(probe['codes']),4):
            codes=probe['codes'][first:first+4].to(device)
            values=probe['coefficients'][first:first+4].to(device)
            labels=probe['labels'][first:first+4].to(device)
            p=(-(values[...,None]-aux.coeff_bins).square()/.03125).softmax(-1)
            output=model(codes,aux,labels,amp=True)
            a=-output['atom_logits'].float().log_softmax(-1).gather(-1,codes[...,0::2,None]).squeeze(-1)
            c=-(p*output['coeff_logits'].float().log_softmax(-1)).sum(-1)
            entropy=-(p*p.clamp_min(1.e-30).log()).sum(-1)
            positive=(output['coeff_logits'].float().softmax(-1)*(aux.coeff_bins>0)).sum(-1)
            sign=((positive>.5)==(values>0)).float()
            top1=(output['atom_logits'].argmax(-1)==codes[...,0::2]).float()
            quantities={name:tensor.mean((1,2)).cpu().tolist() for name,tensor in
                [('atom_ce',a),('coefficient_ce',c),('coefficient_entropy',entropy),('sign_accuracy',sign),('atom_top1',top1)]}
            for i in range(len(codes)):
                rows.append(dict(index=probe['indices'][first+i],**{key:values[i] for key,values in quantities.items()}))
    gathered=[None]*world;dist.all_gather_object(gathered,rows)
    result=None
    if rank==0:
        rows=sorted(sum(gathered,[]),key=lambda row:row['index'])
        assert len(rows)==1024 and len({r['index'] for r in rows})==1024
        means={key:np.asarray([row[key] for row in rows]).mean(0).tolist()
               for key in ['atom_ce','coefficient_ce','coefficient_entropy','sign_accuracy','atom_top1']}
        result=dict(branch=branch,global_step=step,images=1024,per_depth=means,rows=rows,
            mean_atom_ce=float(np.mean(means['atom_ce'])),
            mean_joint_pair_ce=float(np.mean(means['atom_ce'])+np.mean(means['coefficient_ce'])),
            mean_coefficient_excess_ce=float(np.mean(means['coefficient_ce'])-np.mean(means['coefficient_entropy'])),
            probe_sha256_by_rank=[x['sha256'] for x in json.loads((OUT/'probe-verification.json').read_text())['ranks']])
        write_json(OUT/branch/f'heldout-step{step}.json',result)
        if run:
            metrics={'train/global_step':step,'eval/atom_ce':result['mean_atom_ce'],
                'eval/joint_pair_ce':result['mean_joint_pair_ce'],
                'eval/coefficient_excess_ce':result['mean_coefficient_excess_ce']}
            metrics.update({f'eval/sign_accuracy_depth{d+1}':value for d,value in enumerate(means['sign_accuracy'])})
            run.log(metrics)
        print(json.dumps(dict(branch=branch,phase='heldout',global_step=step,
            atom_ce=result['mean_atom_ce'],joint_pair_ce=result['mean_joint_pair_ce'])),flush=True)
    model.train(was_training)
    return result


@torch.no_grad()
def verify_cache(model,aux,probe,device,rank,world):
    model.eval()
    truth=probe['codes'][:2].to(device);labels=probe['labels'][:2].to(device)
    expected=model(truth,aux,labels,amp=True)
    prefix=torch.zeros_like(truth);prefix[...,1::2]=16384+256
    model.init_cache();maximum=0.
    for h in range(8):
        for w in range(8):
            for event in range(8):
                actual=model.cached_forward(prefix[:,:h+1],aux,labels,amp=True,sample_loc=(h,w,event))
                actual=actual[:,:16384] if event%2==0 else actual[:,16384:]
                target=expected['atom_logits' if event%2==0 else 'coeff_logits'][:,h,w,event//2]
                assert torch.equal(actual.isfinite(),target.isfinite())
                p=target.float().log_softmax(-1);q=actual.float().log_softmax(-1)
                kl=torch.where(p.isfinite(),p.exp()*(p-q),0.).sum(-1)
                assert torch.isfinite(kl).all()
                maximum=max(maximum,float(kl.max()))
                prefix[:,h,w,event]=truth[:,h,w,event]
    value=torch.tensor(maximum,device=device);dist.all_reduce(value,op=dist.ReduceOp.MAX)
    assert float(value)<2.e-4,float(value)
    model.init_cache()
    return dict(passed=True,cache_events_per_rank=512,ranks=world,maximum_fp16_kl=float(value))


def open_run(branch,step):
    import wandb
    os.environ['WANDB_API_KEY']=Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
    plan=json.loads((OUT/'plan.json').read_text())
    run=wandb.init(entity='helloimlixin-rutgers',project='laser',id=RUNS[branch],
        name=f'Compound512 atom identity pilot | {branch} | matched128',group=GROUP,
        dir=str(BASE/'wandb'),resume='allow',config=dict(wandb_id=RUNS[branch],experiment_group=GROUP,
            branch=branch,source_step=1252,additional_updates=128,global_batch=2048,world_size=4,
            microbatch=64,accumulation_steps=8,coefficient_bins=512,auxiliary_loss_terms=0,
            sampled_trajectory_banks=0,fid_num_samples=50000,fid_real_split='train',fid_reference_images=1281167,
            source_checkpoint_sha256=plan['source_checkpoint_sha256'],
            added_parameters=4194304 if branch=='atom_identity' else 0))
    assert int(run.summary.get('train/global_step',0))<=step
    run.define_metric('train/global_step')
    run.define_metric('train/*',step_metric='train/global_step')
    run.define_metric('eval/*',step_metric='train/global_step')
    run.define_metric('eval/fid',step_metric='train/global_step',overwrite=True)
    run.summary['trial/status']='training' if step==1252 else 'fid_evaluation'
    write_json(OUT/branch/'wandb.json',dict(url=run.url,id=RUNS[branch]))
    for path in (OUT/'reproducibility').iterdir():
        if path.is_file():run.save(str(path),base_path=str(OUT),policy='now')
    return run


def main(args,rank,world,device):
    branch=args.branch;folder=OUT/branch;local=BASE/branch
    folder.mkdir(exist_ok=True);local.mkdir(exist_ok=True)
    profile=CompoundReplayRecipe(BASE);plan=json.loads((OUT/'plan.json').read_text())
    config=load_config(BASE/'inputs/stage2-training-config.yaml')
    if args.action=='fid':
        checkpoint=json.loads((folder/'checkpoint.json').read_text())['local_path']
    else:checkpoint=plan['local_source_checkpoint']
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    profile.verify_resume(saved)
    step=int(saved['global_step']);assert step==(1380 if args.action=='fid' else 1252)
    config.arch=OmegaConf.create(saved['config']['arch'])
    with torch.device('meta'):model=identity_model(profile,OmegaConf.create(saved['config']['arch']),branch)
    if branch=='atom_identity' and args.action!='fid':model.load_compound_state(saved['state_dict'],assign=True)
    else:model.load_state_dict(saved['state_dict'],strict=True,assign=True)
    assert len(list(model.parameters()))==(838 if branch=='atom_identity' else 837)
    model.to(device);aux,_=profile.tokenizer(device);before_digest=profile.digest(aux)
    probe=torch.load(BASE/f'probe/rank{rank}.pt',weights_only=False,map_location='cpu')
    run=open_run(branch,step) if rank==0 else None
    dist.barrier()
    if args.action=='fid':
        del saved
        proof=verify_cache(model,aux,probe,device,rank,world)
        if rank==0:write_json(folder/'sampling-verification.json',proof)
        shared.BASE,shared.OUT,shared.PROFILE,shared.RUN=BASE,folder,profile,RUNS[branch]
        shared.original_metrics.INCEPTION_PATH=BASE/'upstream/rqvae/metrics/inception.py'
        result=shared.generation(model,aux,device,rank,world,3,run,batch=128,seed=261001)
        assert profile.digest(aux)==before_digest
        if rank==0:
            result.update(branch=branch,global_step=1380,source_step=1252,additional_updates=128,
                generated_images=50000,real_images=1281167,real_split='train',generation_seed=261001,
                rejection_sampling=False,sampling=OmegaConf.to_container(profile.recipe.sampling,resolve=True))
            write_json(folder/'fid50k.json',result)
            run.log({'train/global_step':1380,'eval/fid':result['fid'],'eval/inception_score':result['inception_score']})
            run.summary['trial/status']='complete';run.finish()
        dist.barrier();return
    optimizer=create_resnet_optimizer(model,augment_optimizer_defaults(config.optimizer))
    scheduler=create_scheduler(optimizer,config.optimizer.warmup,626,100)
    source_names=list(saved['state_dict']);source_indices=[i for g in saved['optimizer']['param_groups'] for i in g['params']]
    assert len(source_names)==len(source_indices)==837
    if branch=='atom_identity':
        states={name:saved['optimizer']['state'][index] for name,index in zip(source_names,source_indices)}
        restore_named_adam(optimizer,model,states)
        for own,old in zip(optimizer.param_groups,saved['optimizer']['param_groups']):
            own.update({k:v for k,v in old.items() if k!='params'})
    else:optimizer.load_state_dict(saved['optimizer'])
    scheduler.load_state_dict(saved['scheduler'])
    scaler=torch.amp.GradScaler('cuda');scaler.load_state_dict(saved['scaler'])
    assert scheduler.after_scheduler.last_epoch==1252 and scheduler.after_scheduler.T_max==62600
    assert saved['next_epoch']==2 and saved['next_microbatch']==0 and len(saved['rng_state_by_rank'])==4
    rng=saved['rng_state_by_rank'][rank]
    for name,param in model.named_parameters():
        if name=='atom_identity_embedding.weight':
            assert param not in optimizer.state and torch.count_nonzero(param)==0
        else:assert int(optimizer.state[param]['step'])==1252
    if branch=='atom_identity':
        with torch.device('meta'):reference=profile.model(OmegaConf.create(saved['config']['arch']))
        reference.load_state_dict(saved['state_dict'],strict=True,assign=True);reference.eval().to(device)
        codes=probe['codes'][:2].to(device);labels=probe['labels'][:2].to(device)
        model.eval()
        with torch.no_grad():
            expected=reference(codes,aux,labels,amp=True);actual=model(codes,aux,labels,amp=True)
            for key in actual:torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
        del reference,expected,actual,codes,labels
    if rank==0:write_json(folder/'initialization-verification.json',dict(passed=True,source_step=1252,
        existing_adam_tensors=837,existing_adam_age=1252,new_parameter_tensors=int(branch=='atom_identity'),
        initial_logits_preserved=True,scheduler_restored=True,amp_scaler_restored=True,rng_streams=4))
    initial=evaluate(model,aux,probe,branch,1252,device,rank,world,run)
    del saved
    gc.collect();torch.cuda.empty_cache()
    ddp=DDP(model,device_ids=[device.index],broadcast_buffers=False,gradient_as_bucket_view=True,bucket_cap_mb=100)
    train=TarImageNet(BASE/'data','train',create_transforms(config.dataset,split='train'),seed=261001)
    sampler=ExactGlobalBatchSampler(train,2048,world,rank,8,261001);sampler.set_epoch(2)
    order=np.asarray(list(itertools.chain.from_iterable(itertools.islice(iter(sampler),1024))),dtype=np.int32)
    assert len(order)==65536
    order_hash=hashlib.sha256(order.tobytes()).hexdigest()
    loader=DataLoader(train,batch_sampler=sampler,num_workers=8,pin_memory=True,persistent_workers=True,prefetch_factor=2)
    restore_rng(rng,device)
    optimizer.zero_grad(set_to_none=True);model.train();replay=ReplayWindow(device)
    target_hash=hashlib.sha256();image_hash=hashlib.sha256();label_hash=hashlib.sha256()
    window={};total_retries=0;start=time.monotonic()
    def backward(images,labels,absolute):
        gpu_images=images.to(device,non_blocking=True);gpu_labels=labels.to(device,non_blocking=True)
        sync=(absolute+1)%8==0
        captured={}
        def targets_audit(targets,codes):
            captured['targets_sha_bytes']=codes.cpu().numpy().tobytes()+targets['coefficients'].cpu().numpy().tobytes()
        with nullcontext() if sync else ddp.no_sync():
            loss=training_backward(profile,ddp,aux,gpu_images,gpu_labels,scaler,
                world*len(images)/sampler.global_count(absolute),amp=True,on_targets=targets_audit)
        window[absolute]=(len(images),loss,dict(profile.last_loss_metrics),captured['targets_sha_bytes'])
    for absolute,(images,labels) in enumerate(loader):
        if absolute%8==0:replay.begin();window.clear();start=time.monotonic()
        image_hash.update(images[:1].numpy().tobytes());label_hash.update(labels.numpy().tobytes())
        replay.append(images,labels,absolute);backward(images,labels,absolute)
        if (absolute+1)%8:continue
        norm,retries=recoverable_amp_step(model.parameters(),optimizer,scaler,max_norm=1.,replay_backward=lambda:replay.replay(backward))
        total_retries+=retries;scheduler.step();optimizer.zero_grad(set_to_none=True);step+=1
        stats=torch.stack([sum(n*loss for n,loss,_,_ in window.values()),
            torch.tensor(sum(n for n,_,_,_ in window.values()),device=device),
            sum(n*d['atom_ce'] for n,_,d,_ in window.values()),
            sum(n*d['coefficient_ce'] for n,_,d,_ in window.values())])
        dist.all_reduce(stats)
        for key in sorted(window):target_hash.update(window[key][3])
        row={'train/global_step':step,'train/additional_updates':step-1252,'train/epoch':step/626,
            'train/loss':float(stats[0]/stats[1]),'train/atom_ce':float(stats[2]/stats[1]),
            'train/coefficient_ce':float(stats[3]/stats[1]),'train/lr':optimizer.param_groups[0]['lr'],
            'train/grad_norm':float(norm),'train/amp_overflow_retries':retries,'train/amp_scale':scaler.get_scale(),
            'train/images_per_second':float(stats[1])/(time.monotonic()-start),'time':time.time()}
        write_json(folder/f'progress-rank{rank}.json',row)
        if rank==0 and (step-1252<=3 or (step-1252)%10==0 or step==1380):
            run.log(row);print(json.dumps(row),flush=True)
            with (local/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if step==1380:break
    assert step==1380
    cursor=128*8
    state=dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device))
    rngs=[None]*world;dist.all_gather_object(rngs,state)
    digest=dict(order_sha256=order_hash,labels_sha256=label_hash.hexdigest(),
        first_image_per_microbatch_sha256=image_hash.hexdigest(),targets_and_prefix_sha256=target_hash.hexdigest())
    hashes=[None]*world;dist.all_gather_object(hashes,digest)
    after=evaluate(model,aux,probe,branch,step,device,rank,world,run)
    proof=verify_cache(model,aux,probe,device,rank,world)
    assert profile.digest(aux)==before_digest
    ages={}
    for name,param in model.named_parameters():
        age=int(optimizer.state[param]['step']);expected_age=128 if name=='atom_identity_embedding.weight' else 1380
        assert age==expected_age,(name,age);ages[str(age)]=ages.get(str(age),0)+1
    if branch=='atom_identity':assert torch.isfinite(model.atom_identity_embedding.weight).all() and model.atom_identity_embedding.weight.abs().sum()>0
    if rank==0:
        payload=cpu_snapshot(dict(run_id=RUNS[branch],branch=branch,state_dict=model.state_dict(),
            optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
            global_step=1380,epoch=2,next_epoch=2,next_microbatch=cursor,rng_state_by_rank=rngs,
            parameter_names=list(dict(model.named_parameters())),config=OmegaConf.to_container(config,resolve=True),
            **profile.checkpoint_extra(),**{profile.checkpoint_key:profile.recipe.tokenizer.sha256},
            source_checkpoint_sha256=plan['source_checkpoint_sha256'],source_step=1252,
            runtime=dict(world_size=4,microbatch=64,accumulation_steps=8,global_batch=2048,
                parameter_count=sum(p.numel() for p in model.parameters()),parameter_tensors=len(list(model.parameters())),
                fresh_stage2=False,fresh_optimizer=False)))
        path=local/'checkpoint.pt';temporary=path.with_suffix('.tmp');torch.save(payload,temporary);temporary.replace(path)
        del payload
        receipt=dict(local_path=str(path),bytes=path.stat().st_size,global_step=1380,optimizer_age_groups=ages,
            rng_streams=4,next_epoch=2,next_microbatch=cursor,full_optimizer_and_rng=True)
        write_json(folder/'checkpoint.json',receipt)
        write_json(folder/'sampling-verification.json',proof)
        result=dict(passed=True,branch=branch,source_step=1252,global_step=1380,additional_updates=128,
            optimizer_age_groups=ages,hashes_by_rank=hashes,amp_overflow_retries=total_retries,
            before_atom_ce=initial['mean_atom_ce'],after_atom_ce=after['mean_atom_ce'],
            after_joint_pair_ce=after['mean_joint_pair_ce'],tokenizer_unchanged=True,
            identity_embedding_rms=float(model.atom_identity_embedding.weight.square().mean().sqrt()) if branch=='atom_identity' else None)
        write_json(folder/'result.json',result)
        run.summary['trial/status']='trained_awaiting_gate';run.save(str(folder/'result.json'),base_path=str(OUT),policy='now')
        run.finish();print(json.dumps(result),flush=True)
    dist.barrier()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['probe','train','fid'])
    parser.add_argument('--branch',choices=list(RUNS),default='control')
    args=parser.parse_args();rank,world,device=init()
    if args.action=='probe':prepare_probe(rank,world,device)
    else:main(args,rank,world,device)
    dist.barrier();dist.destroy_process_group()
