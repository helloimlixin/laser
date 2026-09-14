#!/usr/bin/env python3
"""Fresh LASER prior with the paper's Church RQ-Transformer optimization recipe.

The frozen LASER tokenizer, vocabulary and calibrated target temperature remain
LASER-specific. Upstream did not release its stage-2 training driver.
"""
import argparse
from datetime import timedelta
from itertools import islice
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
SOURCE_RUN=ROOT/'outputs/church-compact-rq-stage2-20260913'
SNAPSHOT=SOURCE_RUN/'source-snapshot'
UPSTREAM=SNAPSHOT/'outputs/church-rq-baseline-scratch-20260912/upstream-source'
sys.path[:0]=[str(UPSTREAM),str(SNAPSHOT),str(ROOT)]
# The old snapshot predates src/__init__.py. Bind this namespace explicitly so
# Python cannot select the mutable workspace package over the recorded sources.
import src
src.__path__=[str(SNAPSHOT/'src')]
os.environ.setdefault('TORCH_HOME','/workspace/tmp/official-rqvae-eval-cache')

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader,DistributedSampler
from omegaconf import OmegaConf
from rqvae.optimizer import create_scheduler
from rqvae.optimizer.optimizer import create_resnet_optimizer
from rqvae.metrics.fid import get_inception_model
from torchvision.utils import save_image
from src.original_rq_training import (atomic_json,file_sha256,state_sha256,seed_all,
    load_stage2_config,fresh_transformer,CachedLatents,FeatureMoments,fid_from_moments)
from src.compact_rq_training import FrozenCompactTokenizer
from src.scaled_atom_training import accumulated_update,evaluate_validation


@torch.no_grad()
def evaluate_samples(model,tokenizer,inception,n,epoch,output,reference,device,rank,world):
    model.eval()
    moments=FeatureMoments(device)
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(71000+rank)
        local_total=len(range(rank,n,world))
        for offset in range(0,local_total,100):
            size=min(100,local_total-offset)
            codes=model.sample(torch.zeros(size,8,8,4,dtype=torch.long,device=device),
                model_aux=tokenizer,temperature=1.,top_k=1400,top_p=1.,amp=True,
                cached=True,is_tqdm=False)
            images=tokenizer.decode_code(codes).mul(.5).add(.5).clamp(0,1)
            assert torch.isfinite(images).all()
            moments.update(inception(images))
            if rank==0:
                if offset==0:
                    save_image(images[:64],output/f'samples-{n}-epoch{epoch:03d}.png',nrow=8)
                atomic_json(output/'status.json',dict(phase='evaluating_generation',epoch=epoch,
                    samples_per_rank_done=offset+size,samples_total=n,updated_unix=time.time(),pid=os.getpid()))
    statistics=moments.finish()
    assert statistics[0]==n
    score=fid_from_moments(statistics,reference) if rank==0 else 0.
    value=torch.tensor(score,device=device,dtype=torch.float64)
    dist.broadcast(value,0)
    if rank==0:
        np.savez(output/f'statistics-{n}-epoch{epoch:03d}.npz',mu=statistics[1],sigma=statistics[2],samples=n)
        atomic_json(output/f'fid-{n}-epoch{epoch:03d}.json',dict(epoch=epoch,n=n,fid=score,
            temperature=1.,top_k=1400,top_p=1.,reference=str(reference)))
    model.train()
    return value.item()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--run-id',required=True)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--max-updates',type=int)
    p.add_argument('--offline',action='store_true')
    args=p.parse_args()
    rank,world=int(os.environ['RANK']),int(os.environ['WORLD_SIZE'])
    assert world==2
    device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=True
    dist.init_process_group('nccl',timeout=timedelta(hours=3))
    out=args.output.resolve()
    if rank==0:
        out.mkdir(parents=True,exist_ok=False)
    dist.barrier()
    if rank==0:
        manifest=json.loads((SOURCE_RUN/'source-manifest.json').read_text())
        for name,expected in manifest.items():
            assert file_sha256(SNAPSHOT/name)==expected,name
        atomic_json(out/'dependency-manifest.json',manifest)
        (out/'training-script.py').write_text(Path(__file__).read_text())
        atomic_json(out/'source-provenance.json',dict(dependency_snapshot=str(SNAPSHOT),
            verified_files=len(manifest),driver_sha256=file_sha256(__file__)))
    dist.barrier()
    cache=json.loads((args.cache/'complete.json').read_text())
    calibration=json.loads(args.calibration.read_text())
    assert cache['images']==126227 and cache['shape']==[126227,8,8,256]
    assert cache['world_size']==2 and cache['frozen_state_unchanged']
    assert cache['checkpoint_sha256']==calibration['source_checkpoint_sha256']
    assert cache['codebook_sha256']==calibration['codebook_sha256']
    if rank==0:
        assert file_sha256(cache['latent_cache'])==cache['cache_sha256']
        assert file_sha256(cache['checkpoint'])==cache['checkpoint_sha256']
        assert file_sha256(cache['codebook'])==cache['codebook_sha256']
    dist.barrier()
    temperature=calibration['selected_temperature']
    config=load_stage2_config(UPSTREAM,cache['checkpoint'])
    config.arch.vocab_size=config.dataset.vocab_size=32769
    config.experiment.batch_size=args.batch_size
    config.experiment.total_batch_size=2048
    config.experiment.accumulation_steps=2048//(world*args.batch_size)
    config.experiment.sample.top_k=config.sampling.top_k=1400
    config.loss.temp=temperature
    config.vqvae.codebook=cache['codebook']
    config.vqvae.codebook_sha256=cache['codebook_sha256']
    config.lr_control=dict(type='published_cosine_only',fid_triggered_reductions=False)
    config.recipe=dict(source='https://arxiv.org/html/2203.01941#A3',
        sampling_source='https://arxiv.org/html/2203.01941#A4',
        training_driver='Local implementation; upstream training loop was not released',
        adaptations=['Frozen LASER compact 32769-word tokenizer instead of RQVAE',
            'Calibrated LASER target temperature 0.125 instead of raw RQVAE temperature 0.5',
            'Two GPUs with gradient accumulation instead of four GPUs'],
        checkpoint_policy='Retain latest best full states for FID4096 and validation soft CE')
    config.training_precision='FP16 model autocast; FP32 codebook geometry and cross entropy'
    run=None
    if rank==0:
        OmegaConf.save(config,out/'config.yaml')
        atomic_json(out/'cache-provenance.json',cache)
        atomic_json(out/'temperature-calibration.json',calibration)
        if not args.offline:
            for name in ('WANDB_SERVICE','_WANDB_SERVICE'):
                os.environ.pop(name,None)
            import wandb
            run=wandb.init(project='laser',entity='helloimlixin-rutgers',id=args.run_id,
                name=args.run_id,resume='never',dir=str(out),config={
                    **OmegaConf.to_container(config,resolve=True),
                    'pipeline':'frozen-laser-compact-rq2-paper-batch2048-cosine',
                    'stage2_from_scratch':True,'stage1_frozen':True,
                    'upstream_commit':'341395e562ac347f5eb62db9f5f08b9f2cc42a60',
                    'stage2_trainer':'new driver; upstream did not release training loop'})
            atomic_json(out/'wandb.json',dict(run_id=run.id,url=run.url))
    tokenizer=FrozenCompactTokenizer(cache['checkpoint'],cache['codebook']).to(device).eval()
    assert state_sha256(tokenizer)==cache['frozen_state_sha256']
    seed_all(0)
    model,optimizer=fresh_transformer(config)
    initial_hash=state_sha256(model)
    hashes=[None]*world
    dist.all_gather_object(hashes,initial_hash)
    assert len(set(hashes))==1 and not optimizer.state
    assert sum(p.numel() for p in model.parameters())==386882561
    if rank==0:
        initialization=dict(stage2_from_scratch=True,seed=0,pretrained_stage2_checkpoint=None,
            initial_optimizer_entries=0,initial_weights_sha256=initial_hash,
            initial_weights_identical_across_ranks=True,parameters=386882561,
            tokenizer_checkpoint=cache['checkpoint'],tokenizer_sha256=cache['checkpoint_sha256'],
            codebook_sha256=cache['codebook_sha256'],temperature=temperature,
            upstream_commit='341395e562ac347f5eb62db9f5f08b9f2cc42a60')
        atomic_json(out/'initialization.json',initialization)
        print(json.dumps(initialization),flush=True)
        if run:
            run.summary.update(initialization)
            run.summary['training_status']='training_from_scratch'
    model.to(device)
    optimizer=create_resnet_optimizer(model,config.optimizer)
    ddp=DistributedDataParallel(model,device_ids=[device.index],broadcast_buffers=False)
    dataset=CachedLatents(cache['latent_cache'])
    sampler=DistributedSampler(dataset,num_replicas=world,rank=rank,shuffle=True,seed=0)
    loader=DataLoader(dataset,sampler=sampler,batch_size=args.batch_size,num_workers=8,
        pin_memory=True,persistent_workers=True,drop_last=False)
    accumulation=2048//(world*args.batch_size)
    assert accumulation*world*args.batch_size==2048
    steps_per_epoch=math.ceil(len(loader)/accumulation)
    assert steps_per_epoch==62
    scheduler=create_scheduler(optimizer,config.optimizer.warmup,steps_per_epoch,300)
    scaler=torch.amp.GradScaler('cuda',init_scale=65536.)
    best_fid=best_validation=float('inf')
    heldout=torch.load(args.cache/f'validation-rank{rank}.pt',map_location='cpu',weights_only=True)
    inception=None
    if not args.max_updates:
        inception=get_inception_model().eval().requires_grad_(False).to(device)
    reference=ROOT/'outputs/lsun-church-bar-20260910/assets/lsun_256_church.npz'
    seed_all(rank)
    stopping={'requested':False}
    def on_signal(signum,frame):
        stopping['requested']=True
    signal.signal(signal.SIGTERM,on_signal)
    signal.signal(signal.SIGINT,on_signal)
    step=attempts=skipped=consecutive_skips=0
    started=time.time()
    torch.cuda.reset_peak_memory_stats(device)

    def status(record):
        if rank==0:
            record.update(updated_unix=time.time(),elapsed_seconds=time.time()-started,pid=os.getpid())
            atomic_json(out/'status.json',record)
            with (out/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
            if run:
                run.log({f'stage2/{k}':v for k,v in record.items() if isinstance(v,(int,float))})

    def save(epoch,batch,filename='last.pt'):
        states=[None]*world
        dist.all_gather_object(states,dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device),
            numpy=np.random.get_state(),python=random.getstate()))
        if rank==0:
            dest=out/filename
            torch.save(dict(epoch=epoch,batch_in_epoch=batch,step=step,attempts=attempts,
                skipped_amp_updates=skipped,state_dict=model.state_dict(),optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng_states=states,
                best_fid=best_fid,best_validation=best_validation,
                tokenizer=cache,temperature_calibration=calibration,
                initial_weights_sha256=initial_hash,config=OmegaConf.to_container(config,resolve=True)),
                dest.with_suffix('.tmp'))
            dest.with_suffix('.tmp').replace(dest)
        dist.barrier()

    for epoch_index in range(300):
        sampler.set_epoch(epoch_index)
        model.train()
        iterator=iter(loader)
        consumed=0
        for update_index in range(steps_per_epoch):
            batches=list(islice(iterator,accumulation))
            assert batches
            consumed+=len(batches)
            metrics=accumulated_update(ddp,tokenizer,optimizer,scaler,batches,temperature,
                                       max_gn=config.optimizer.max_gn)
            attempts+=1
            if metrics['optimizer_updated']:
                scheduler.step()
                step+=1
                consecutive_skips=0
            else:
                skipped+=1
                consecutive_skips+=1
            epoch_fraction=epoch_index+(update_index+1)/steps_per_epoch
            if attempts<=10 or attempts%10==0:
                status(dict(phase='training',optimizer_step=step,attempted_updates=attempts,
                    epoch=epoch_fraction,lr=optimizer.param_groups[0]['lr'],lr_multiplier=1.,
                    effective_batch_size=2048,accumulation_steps=accumulation,
                    skipped_amp_updates=skipped,consecutive_amp_skips=consecutive_skips,
                    peak_gpu_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3,
                    peak_gpu_reserved_gib=torch.cuda.max_memory_reserved(device)/1024**3,**metrics))
            if consecutive_skips>=8:
                save(epoch_index,consumed)
                raise FloatingPointError('Eight consecutive AMP skips; checkpoint saved')
            stop=torch.tensor(int(stopping['requested']),device=device)
            dist.all_reduce(stop,op=dist.ReduceOp.MAX)
            smoke_done=args.max_updates and step>=args.max_updates
            if stop.item() or smoke_done:
                assert state_sha256(tokenizer)==cache['frozen_state_sha256']
                if smoke_done:
                    # Exercise the actual released cached sampler and frozen decoder.
                    model.eval()
                    with torch.no_grad():
                        codes=model.sample(torch.zeros(2,8,8,4,dtype=torch.long,device=device),
                            model_aux=tokenizer,temperature=1.,top_k=1400,top_p=1.,amp=True,
                            cached=True,is_tqdm=False)
                        decoded=tokenizer.decode_code(codes)
                        assert decoded.shape==(2,3,256,256) and torch.isfinite(decoded).all()
                    atomic_json(out/f'sampler-smoke-rank{rank}.json',dict(passed=True,
                        code_shape=list(codes.shape),top_k=1400,tokenizer_unchanged=True))
                save(epoch_index,consumed)
                status(dict(phase='preflight_complete' if smoke_done else 'paused',optimizer_step=step,
                            epoch=epoch_fraction,peak_gpu_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3))
                if run:
                    run.summary['training_status']='paused_with_checkpoint'
                    run.finish()
                dist.destroy_process_group()
                return
            if step==25 and metrics['optimizer_updated']:
                save(epoch_index,consumed)
        assert consumed==len(loader)
        epoch=epoch_index+1
        save(epoch,0)
        if epoch%10==0 and rank==0:
            torch.save(dict(epoch=epoch,step=step,state_dict=model.state_dict(),tokenizer=cache),
                       out/f'epoch{epoch}_model.pt')
        dist.barrier()
        if epoch==1 or epoch%5==0:
            validation=evaluate_validation(model,tokenizer,heldout,device,rank,temperature)
            if validation['soft_ce']<best_validation:
                best_validation=validation['soft_ce']
                save(epoch,0,'best-validation.pt')
                if rank==0:
                    atomic_json(out/'best-validation.json',dict(epoch=epoch,step=step,**validation))
            if rank==0:
                atomic_json(out/f'validation-epoch{epoch:03d}.json',dict(epoch=epoch,**validation))
                if run:
                    run.log({'validation/epoch':epoch,**{f'validation/{k}':v for k,v in validation.items()}})
            score=evaluate_samples(model,tokenizer,inception,4096,epoch,out,reference,device,rank,world)
            assert state_sha256(tokenizer)==cache['frozen_state_sha256']
            if score<best_fid:
                best_fid=score
                save(epoch,0,'best-fid.pt')
                if rank==0:
                    atomic_json(out/'best-fid.json',dict(epoch=epoch,step=step,fid=score,n=4096,
                        top_k=1400,top_p=1.,temperature=1.))
            if rank==0:
                if run:
                    import wandb
                    run.log({'generation/fid_4096':score,'generation/epoch':epoch,
                        'generation/samples':wandb.Image(str(out/f'samples-4096-epoch{epoch:03d}.png')),
                        'stage2/lr_multiplier':1.})
                    run.summary['last_fid_4096']=score
                    run.summary['best_fid_4096']=best_fid
                    run.summary['best_validation_soft_ce']=best_validation
            if epoch%50==0:
                fid50k=evaluate_samples(model,tokenizer,inception,50000,epoch,out,reference,device,rank,world)
                if rank==0 and run:
                    run.log({'generation/fid_50000':fid50k,'generation/epoch':epoch})
            save(epoch,0)
    status(dict(phase='complete',optimizer_step=step,epochs=300))
    if run:
        run.summary['training_status']='complete'
        run.finish()
    dist.destroy_process_group()


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        # Make failures visible even if the last progress entry said 'training'.
        if '--output' in sys.argv:
            output=Path(sys.argv[sys.argv.index('--output')+1])
            if output.exists():
                atomic_json(output/f'failure-rank{os.environ.get("RANK","unknown")}.json',
                    dict(error_type=type(error).__name__,error=str(error),updated_unix=time.time()))
        raise
