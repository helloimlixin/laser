"""Recoverable full-image training of the fresh, structured pair hierarchy."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import fields
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'runtime')]

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from src.models.coarse_fine_pair_prior import build_scratch_pair_prior,joint_hierarchy_loss
from src.models.typed_pair_rq import TypedPairCodebook
from src.training.exact_global_batch import ExactGlobalBatchSampler
from src.training.pair_scratch_data import IndexedPairImages
from src.training.rqtransformer import LaserAux,image_transform
from src.training.typed_pair_targets import typed_pair_targets


def record(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def rate_at_step(step,total,warmup):
    if step<=warmup:return step/max(1,warmup)
    return .5*(1+math.cos(math.pi*min(1.,(step-warmup)/max(1,total-warmup))))


def snapshot(value):
    if isinstance(value,torch.Tensor):return value.detach().to(device='cpu',copy=True)
    if isinstance(value,dict):return {key:snapshot(item) for key,item in value.items()}
    if isinstance(value,list):return [snapshot(item) for item in value]
    if isinstance(value,tuple):return tuple(snapshot(item) for item in value)
    return value


class CheckpointWriter:
    def __init__(self,local,persistent,run_id,key_file,cloud=True):
        self.local,self.persistent=Path(local),Path(persistent)
        self.local.mkdir(parents=True,exist_ok=True);self.persistent.mkdir(parents=True,exist_ok=True)
        self.run_id,self.key_file,self.cloud=run_id,key_file,cloud
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='pair-recovery')
        self.pending=None

    def check(self):
        if self.pending is not None and self.pending.done():self.pending.result()

    def submit(self,payload):
        if self.pending is not None:self.pending.result()
        def write():
            step=payload['global_step'];target=self.local/f'step-{step:08d}.pt'
            temporary=target.with_suffix('.pending');torch.save(payload,temporary);temporary.replace(target)
            alias=self.local/'latest.pending'
            if alias.exists():alias.unlink()
            os.link(target,alias);alias.replace(self.local/'latest.pt')
            # The writer owns immutable CPU tensors, never live CUDA weights.
            recovery=dict(step=step,epoch=payload['epoch'],micro_batch_cursor=payload['micro_batch_cursor'],
                          local=str(target),local_bytes=target.stat().st_size,cloud_verified=False)
            record(self.local/'latest.json',recovery)
            durable=self.persistent/'latest.pending'
            shutil.copyfile(target,durable);durable.replace(self.persistent/'latest.pt')
            if self.cloud:
                import wandb
                api=wandb.Api(timeout=300);run=api.run('helloimlixin-rutgers/laser/'+self.run_id)
                uploaded=run.upload_file(str(self.local/'latest.pt'),root=str(self.local))
                if uploaded.size!=target.stat().st_size:raise RuntimeError('checkpoint upload size mismatch')
                recovery['cloud_verified']=True
                run.summary['recoverable_step']=step;run.summary['recoverable_epoch']=payload['epoch'];run.summary.update()
            record(self.local/'latest.json',recovery);record(self.persistent/'latest.json',recovery)
            older=sorted(self.local.glob('step-*.pt'))[:-2]
            for path in older:path.unlink()
            return recovery
        self.pending=self.pool.submit(write)

    def close(self):
        try:
            if self.pending is not None:self.pending.result()
        finally:self.pool.shutdown(wait=True)


def rng_state(teacher):
    return dict(torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state(),
                teacher=teacher.get_state(),numpy=np.random.get_state(),python=random.getstate())


def restore_rng(state,teacher):
    torch.set_rng_state(state['torch_cpu']);torch.cuda.set_rng_state(state['torch_cuda'])
    teacher.set_state(state['teacher']);np.random.set_state(state['numpy']);random.setstate(state['python'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--persistent-dir',type=Path,required=True)
    p.add_argument('--key-file',type=Path,required=True)
    p.add_argument('--resume',type=Path)
    p.add_argument('--canary',action='store_true')
    args=p.parse_args();cfg=json.loads(args.config.read_text())
    rank,world=int(os.environ['RANK']),int(os.environ['WORLD_SIZE'])
    local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local);device=torch.device('cuda',local)
    torch.set_num_threads(cfg.get('cpu_threads',8));torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    dist.init_process_group('nccl',device_id=device)
    if world!=cfg['world_size']:raise ValueError('this exact batch recipe requires its saved world size')
    run_dir=args.run_dir;run_dir.mkdir(parents=True,exist_ok=True)
    persistent=args.persistent_dir;persistent.mkdir(parents=True,exist_ok=True)
    recipe_digest=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    global_batch=cfg['microbatch']*world*(1 if args.canary else cfg['accumulation'])
    accumulation=1 if args.canary else cfg['accumulation']
    if not args.canary and global_batch!=cfg['global_batch']:raise ValueError('global batch mismatch')
    dataset=IndexedPairImages(cfg['archive'],cfg['index'],image_transform(),seed=cfg['seed'])
    sampler=ExactGlobalBatchSampler(dataset,global_batch,world,rank,accumulation,seed=cfg['data_seed'])
    loader=DataLoader(dataset,batch_sampler=sampler,num_workers=cfg['workers'],pin_memory=True,
        persistent_workers=True,prefetch_factor=2,multiprocessing_context='spawn')
    frozen=torch.load(cfg['codebook'],map_location='cpu',weights_only=True)
    book=TypedPairCodebook(frozen['dictionary'],frozen['coeff_bins'],depth=4).to(device)
    coarse_book=TypedPairCodebook(frozen['dictionary'],frozen['coeff_bins'],depth=1).to(device)
    tokenizer=LaserAux(Path(cfg['tokenizer']),book.num_atoms,book.coeff_vocab_size,16.,
        coeff_bin_centers=book.coeff_bins.cpu(),coeff_scales=[1.,1.,1.,1.],sparsity_level=4).to(device).eval()
    tokenizer.requires_grad_(False)
    torch.testing.assert_close(tokenizer.dictionary,book.dictionary,atol=0,rtol=0)
    cached=torch.load(cfg['heldout_cache'],map_location='cpu',mmap=True,weights_only=False)
    val=cached['heldout'];selection=torch.arange(rank,len(val['latent']),world)
    latent_val=val['latent'][selection].to(device);labels_val=val['labels'][selection].to(device)
    hard_val=typed_pair_targets(book,latent_val,stochastic=False,top_k=1)
    del cached,val
    # Deliberately reset AFTER loading the frozen tokenizer: no prior weights
    # or optimizer moments can enter a fresh stage-two model.
    torch.manual_seed(cfg['seed'])
    model=build_scratch_pair_prior(book).to(device).train()
    with torch.random.fork_rng(devices=[device.index]):
        probe=type(hard_val)(**{field.name:getattr(hard_val,field.name)[:1] for field in fields(hard_val)})
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss,_,_=joint_hierarchy_loss(model(probe.tokens,book,labels_val[:1],coarse_book,amp=True),probe,book.coeff_bins)
        loss.backward()
    inactive=[name for name,param in model.named_parameters() if param.grad is None]
    if any(not name.startswith('coarse.backbone.head_mlp.') for name in inactive):
        raise RuntimeError('unexpected unused trainable parameters: '+str(inactive))
    for name,param in model.named_parameters():
        if name in inactive:param.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    fingerprint=hashlib.sha256()
    for name,param in model.named_parameters():
        fingerprint.update(name.encode());fingerprint.update(param.detach().flatten()[:32].float().cpu().numpy().tobytes())
    audit=dict(fresh_stage2=args.resume is None,fresh_optimizer=args.resume is None,
        initialization_seed=cfg['seed'],model_sampled_sha256=fingerprint.hexdigest(),inactive_parameters=inactive,
        coarse_parameters=sum(p.numel() for p in model.coarse.parameters() if p.requires_grad),
        refiner_parameters=sum(p.numel() for p in model.refiner.parameters() if p.requires_grad),
        frozen_tokenizer=True,initial_stage2_checkpoint=None if args.resume is None else str(args.resume))
    model=DDP(model,device_ids=[local],broadcast_buffers=False,gradient_as_bucket_view=True)
    optimizer=torch.optim.AdamW([
        dict(params=[p for p in model.module.coarse.parameters() if p.requires_grad],lr=cfg['coarse_lr'],base_lr=cfg['coarse_lr']),
        dict(params=[p for p in model.module.refiner.parameters() if p.requires_grad],lr=cfg['fine_lr'],base_lr=cfg['fine_lr'])],
        betas=(.9,.95),weight_decay=cfg['weight_decay'],fused=True)
    audit['initial_optimizer_state_entries']=len(optimizer.state)
    teacher=torch.Generator(device=device).manual_seed(cfg['teacher_seed']+rank)
    step,start_epoch,start_cursor=0,0,0
    total_steps=cfg['epochs']*sampler.steps_per_epoch
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',mmap=True,weights_only=False)
        if saved['recipe_digest']!=recipe_digest or saved['world_size']!=world:raise ValueError('resume recipe/layout mismatch')
        model.module.load_state_dict(saved['state_dict'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        step,start_epoch,start_cursor=saved['global_step'],saved['epoch'],saved['micro_batch_cursor']
        resume_rng=saved['rng_by_rank'][rank];del saved
    else:resume_rng=None
    torch.manual_seed(cfg['seed']+1000+rank);random.seed(cfg['seed']+rank);np.random.seed(cfg['seed']+rank)
    if resume_rng is not None:restore_rng(resume_rng,teacher)
    record(run_dir/f'initialization-rank{rank}.json',audit)
    wb=None;writer=None
    if rank==0:
        record(persistent/'initialization.json',audit);record(persistent/'recipe.json',cfg)
        if not args.canary:
            import wandb
            os.environ['WANDB_API_KEY']=args.key_file.read_text().strip()
            wb=wandb.init(entity='helloimlixin-rutgers',project='laser',id=cfg['run_id'],
                name='ImageNet | fresh coarse-to-fine whole-pair prior | 4 H100',
                group='imagenet-structured-pair-scratch-20261029',config={**cfg,**audit},
                dir=str(run_dir),resume='must' if args.resume else 'never')
        writer=CheckpointWriter(run_dir/'checkpoints',persistent/'checkpoints',cfg['run_id'],
            args.key_file,cloud=not args.canary)
    stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)

    @torch.no_grad()
    def evaluate():
        model.eval();totals=torch.zeros(3,4,device=device);count=0
        for first in range(0,len(latent_val),4):
            target=type(hard_val)(**{field.name:getattr(hard_val,field.name)[first:first+4] for field in fields(hard_val)})
            with torch.autocast('cuda',dtype=torch.bfloat16):
                _,a,c=joint_hierarchy_loss(model.module(target.tokens,book,labels_val[first:first+4],coarse_book,amp=True),
                                            target,book.coeff_bins)
            n=len(target.tokens);totals+=torch.stack((a,c,a+c))*n;count+=n
        dist.all_reduce(totals);number=torch.tensor(count,device=device);dist.all_reduce(number)
        totals/=number;model.train()
        metrics={f'heldout/{name}/depth{depth+1}':float(totals[i,depth])
            for i,name in enumerate(['atom_nll','coefficient_nll','joint_nll']) for depth in range(4)}
        metrics['heldout/joint_nll']=float(totals[2].mean());return metrics

    def save(epoch,cursor):
        states=[None]*world;dist.all_gather_object(states,rng_state(teacher))
        if rank==0:
            writer.check()
            payload=dict(state_dict=snapshot(model.module.state_dict()),optimizer=snapshot(optimizer.state_dict()),
                codebook=snapshot(book.state_dict()),recipe=cfg,recipe_digest=recipe_digest,world_size=world,
                global_step=step,epoch=epoch,micro_batch_cursor=cursor,rng_by_rank=states,
                initialization=audit,steps_per_epoch=sampler.steps_per_epoch)
            writer.submit(payload)

    @torch.no_grad()
    def images(epoch):
        from src.rqvae_metrics import DistributedOriginalRQVAEMetrics
        from torchvision.utils import save_image
        model.eval()
        metric=DistributedOriginalRQVAEMetrics(device,compute_inception_score=True,reference_stats_path=cfg['fid_reference'])
        generator=torch.Generator(device=device).manual_seed(cfg['evaluation_seed']+rank)
        local_count=cfg['fid_samples']//world
        for first in range(0,local_count,cfg['sample_batch']):
            n=min(cfg['sample_batch'],local_count-first)
            labels=(torch.arange(first,first+n,device=device)*world+rank)%1000
            tokens=model.module.sample(n,book,coarse_book,labels,generator=generator,amp=True)
            z=book.reconstruct(tokens)
            pixels=tokenizer.decoder(tokenizer.post_quant_conv(z.permute(0,3,1,2).contiguous())).clamp(-1,1)
            pixels=((pixels+1)*.5).clamp(0,1);metric.update(pixels,real=False)
            if rank==0 and first==0:
                path=run_dir/f'preview-epoch{epoch:03d}.png';save_image(pixels[:32],path,nrow=8)
                if wb:
                    import wandb
                    wb.log({'samples':wandb.Image(str(path))},step=step)
            if first%1024==0:print(json.dumps(dict(phase='fid',rank=rank,generated=first+n,epoch=epoch)),flush=True)
        torch.manual_seed(cfg['evaluation_seed']+100)
        fid,score,std=metric.compute();model.train()
        return dict(fid50k=fid,inception_score50k=score,inception_score50k_std=std,epoch=epoch)

    initial=evaluate()
    if rank==0:
        record(run_dir/'initial-validation.json',initial)
        if wb:wb.log(initial,step=step)
        print(json.dumps(dict(phase='initialized',audit=audit,heldout=initial)),flush=True)
    started=time.monotonic();last_time=started
    try:
        for epoch in range(start_epoch,cfg['epochs']):
            sampler.set_epoch(epoch);cursor=start_cursor if epoch==start_epoch else 0;sampler.set_start_batch(cursor)
            optimizer.zero_grad(set_to_none=True);totals=torch.zeros(3,4,device=device);local_images=0
            for offset,(pixels,labels) in enumerate(loader):
                micro=cursor+offset;n=len(pixels)
                if micro%accumulation==0:
                    scale=rate_at_step(step+1,total_steps,cfg['warmup_steps'])
                    for group in optimizer.param_groups:group['lr']=group['base_lr']*scale
                with torch.no_grad(),torch.autocast('cuda',enabled=False):
                    latent=tokenizer.quant_conv(tokenizer.encoder(pixels.to(device,non_blocking=True))).permute(0,2,3,1).float()
                    target=typed_pair_targets(book,latent,top_k=cfg['teacher_top_k'],
                        relative_temperature=cfg['teacher_temperature'],generator=teacher)
                sync=(micro+1)%accumulation==0
                with nullcontext() if sync else model.no_sync():
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        output=model(target.tokens,book,labels.to(device,non_blocking=True),coarse_book,amp=True)
                        loss,a,c=joint_hierarchy_loss(output,target,book.coeff_bins)
                        scaled=loss*(world*n/sampler.global_count(micro))
                    if not torch.isfinite(loss):raise FloatingPointError('nonfinite pair loss')
                    scaled.backward()
                totals+=torch.stack((a,c,a+c)).detach()*n;local_images+=n
                if not sync:continue
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();optimizer.zero_grad(set_to_none=True);step+=1
                dist.all_reduce(totals);number=torch.tensor(local_images,device=device);dist.all_reduce(number)
                totals/=number
                if rank==0:
                    if writer:writer.check()
                    now=time.monotonic()
                    metrics={f'train/{name}/depth{d+1}':float(totals[i,d])
                        for i,name in enumerate(['atom_ce','coefficient_ce','joint_ce']) for d in range(4)}
                    metrics.update({'train/joint_ce':float(totals[2].mean()),'train/grad_norm':float(norm),
                        'train/coarse_lr':optimizer.param_groups[0]['lr'],'train/fine_lr':optimizer.param_groups[1]['lr'],
                        'train/images_per_second':int(number)/(now-last_time),'epoch':epoch+(micro+1)/(len(sampler)+cursor)})
                    status=dict(step=step,epoch=epoch,micro_batch_cursor=micro+1,metrics=metrics,
                        peak_cuda_bytes=torch.cuda.max_memory_allocated(),seconds=now-started)
                    record(run_dir/'status.json',status);record(persistent/'status.json',status)
                    print(json.dumps(status),flush=True)
                    if wb:wb.log(metrics,step=step)
                    last_time=now
                totals.zero_();local_images=0
                stop_flag=torch.tensor(int(stopping),device=device);dist.all_reduce(stop_flag,op=dist.ReduceOp.MAX)
                if step==1 or step%cfg['checkpoint_every']==0 or bool(stop_flag):save(epoch,micro+1)
                if step%cfg['validate_every']==0:
                    validation=evaluate()
                    if rank==0 and wb:wb.log(validation,step=step)
                if args.canary and step>=2:
                    save(epoch,micro+1)
                    if rank==0:record(run_dir/'canary-passed.json',dict(steps=step,all_steps_finite=True,audit=audit))
                    return
                if bool(stop_flag):return
            start_cursor=0
            with torch.random.fork_rng(devices=[device.index]):evaluation=images(epoch+1)
            validation=evaluate()
            if rank==0:
                record(persistent/f'evaluation-epoch{epoch+1:03d}.json',{**evaluation,**validation})
                if wb:wb.log({**evaluation,**validation},step=step)
            save(epoch+1,0)
    finally:
        if rank==0 and writer:writer.close()
        if wb:wb.finish()
        dist.destroy_process_group()


if __name__=='__main__':main()
