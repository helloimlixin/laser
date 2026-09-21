#!/usr/bin/env python3
"""Train or resume the consistent-RQVAE pipeline without editing its snapshot."""
import argparse
import fcntl
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
sys.path.insert(0,str(ROOT))
from src.training.stage2_resume import (capture_rng_state, validate_resume_payload,
    restore_training_state, resumed_iterator)
from src.training.stochastic_targets import (TARGET_POLICY_VERSION, depth_temperatures,
    install_compact_target_policy)
import src.training.stochastic_targets as target_implementation
bootstrap=argparse.ArgumentParser(add_help=False)
bootstrap.add_argument('--pipeline-dir',type=Path,default=ROOT/'outputs/church-consistent-rqvae-20260914')
BASE=bootstrap.parse_known_args()[0].pipeline_dir.resolve()
SNAPSHOT=BASE/'stage2-source'
UPSTREAM=SNAPSHOT/'outputs/church-rq-baseline-scratch-20260912/upstream-source'
sys.path[:0]=[str(UPSTREAM),str(SNAPSHOT),str(ROOT)]
import src
src.__path__=[str(SNAPSHOT/'src')]
os.environ.setdefault('TORCH_HOME','/workspace/tmp/official-rqvae-eval-cache')
from src.original_rq_training import load_stage2_config

SOURCE_RUN=BASE
_OWNED_OUTPUT=False

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
def validation_target_entropy(tokenizer,latents,device,rank,temperature):
    """Replay the validation targets to separate their entropy from model KL."""
    depth=int(tokenizer.code_shape[-1])
    total=torch.zeros(depth+1,device=device,dtype=torch.float64)
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(61000+rank)
        for z in latents.split(4):
            probability,_=tokenizer.quantizer.get_soft_codes(z.to(device),temp=temperature,stochastic=True)
            entropy=-(probability*probability.clamp_min(1e-30).log()).sum(-1).reshape(-1,depth)
            total[:depth]+=entropy.double().sum(0)
            total[-1]+=len(entropy)
    dist.all_reduce(total)
    return (total[:depth]/total[-1]).cpu().tolist()


@torch.no_grad()
def evaluate_samples(model,tokenizer,inception,n,epoch,output,reference,device,rank,world,sampling,decode_batch_size=8):
    if n!=50000:
        raise ValueError('Reported FID requires exactly 50000 generated samples')
    model.eval()
    moments=FeatureMoments(device)
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(71000+rank)
        local_total=len(range(rank,n,world))
        for offset in range(0,local_total,100):
            size=min(100,local_total-offset)
            codes=model.sample(torch.zeros(size,8,8,4,dtype=torch.long,device=device),
                model_aux=tokenizer,**sampling,amp=True,
                cached=True,is_tqdm=False)
            grid=[]
            for chunk in codes.split(decode_batch_size):
                images=tokenizer.decode_code(chunk).mul(.5).add(.5).clamp(0,1)
                assert torch.isfinite(images).all()
                moments.update(inception(images))
                if rank==0 and offset==0 and sum(len(x) for x in grid)<64:
                    grid.append(images.detach().cpu())
            if rank==0:
                if offset==0:
                    save_image(torch.cat(grid)[:64],output/f'samples-{n}-epoch{epoch:03d}.png',nrow=8)
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
            **sampling,reference=str(reference),real_samples=126227))
    model.train()
    return value.item()


def main():
    global _OWNED_OUTPUT
    p=argparse.ArgumentParser()
    p.add_argument('--pipeline-dir',type=Path,default=BASE)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--run-id',required=True)
    p.add_argument('--batch-size',type=int)
    p.add_argument('--resume',type=Path,help='Restore a full last.pt at an optimizer boundary')
    p.add_argument('--validate-resume',action='store_true',help='Validate resume metadata on CPU and exit')
    p.add_argument('--fid-every',type=int,default=50)
    p.add_argument('--fid-start-epoch',type=int,default=1)
    p.add_argument('--validation-every',type=int,default=5)
    p.add_argument('--stop-after-epoch',type=int,help='Pause at this epoch without changing the 300-epoch LR schedule')
    p.add_argument('--keep-best',action='store_true',help='Retain best validation and FID model weights in addition to last.pt')
    p.add_argument('--memory-fraction',type=float,help='Optional per-process CUDA allocation limit for a concurrent trial')
    p.add_argument('--decode-batch-size',type=int,default=8)
    p.add_argument('--sampling-temperature',type=float,default=1.)
    p.add_argument('--top-k',type=int,default=1400,help='0 disables the top-k cap')
    p.add_argument('--top-p',type=float,default=1.)
    p.add_argument('--max-updates',type=int)
    p.add_argument('--offline',action='store_true')
    p.add_argument('--smoke-cache',action='store_true',help='Repeat a diagnostic cache for a full-batch preflight only')
    args=p.parse_args()
    if args.validate_resume and not args.resume:
        p.error('--validate-resume requires --resume')
    if args.fid_every<1 or not math.isfinite(args.sampling_temperature) or args.sampling_temperature<=0 or args.top_k<0 or not 0<args.top_p<=1:
        p.error('Invalid FID cadence or sampling setting')
    if args.fid_start_epoch<1 or args.validation_every<1 or args.decode_batch_size<1 or (args.stop_after_epoch is not None and not 1<=args.stop_after_epoch<=300):
        p.error('Invalid evaluation or stopping epoch')
    if args.memory_fraction is not None and not 0<args.memory_fraction<=1:
        p.error('Memory fraction must be in (0,1]')
    resume=torch.load(args.resume,map_location='cpu',weights_only=False,mmap=True) if args.resume else None
    if args.batch_size is None:
        args.batch_size=resume['config']['experiment']['batch_size'] if resume else 512
    if args.batch_size<1 or 2048%(2*args.batch_size):
        p.error('The per-rank batch size must divide the global batch of 2048 over two ranks')
    cache=json.loads((args.cache/'complete.json').read_text())
    calibration=json.loads(args.calibration.read_text())
    temperature=calibration['selected_temperature']
    depth_temperatures(temperature,4)
    target_policy=calibration.get('target_policy')
    if target_policy is not None:
        if target_policy!=TARGET_POLICY_VERSION:
            raise ValueError('Unsupported stochastic target policy')
        if calibration.get('target_implementation_sha256')!=file_sha256(target_implementation.__file__):
            raise ValueError('Stochastic target implementation differs from the calibrated source')
    elif not isinstance(temperature,(int,float)):
        raise ValueError('Depth temperatures require an explicit versioned target policy')
    if resume:
        validate_resume_payload(resume,world=2,batch_size=args.batch_size,cache=cache,calibration=calibration)
        if args.max_updates is not None and args.max_updates<=resume['step']:
            p.error('--max-updates must exceed the restored optimizer step')
        if args.stop_after_epoch is not None and args.stop_after_epoch<=resume['epoch']:
            p.error('--stop-after-epoch must exceed the restored epoch')
        if args.validate_resume:
            print(json.dumps(dict(valid=True,checkpoint=str(args.resume),epoch=resume['epoch'],
                batch_in_epoch=resume['batch_in_epoch'],optimizer_step=resume['step'],
                batch_size_per_rank=args.batch_size,world_size=2,training_started=False)),flush=True)
            return
    sampling=dict(temperature=args.sampling_temperature,top_k=args.top_k or None,top_p=args.top_p)
    rank,world=int(os.environ['RANK']),int(os.environ['WORLD_SIZE'])
    assert world==2
    device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    if args.memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction,device)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=True
    dist.init_process_group('nccl',timeout=timedelta(hours=3))
    out=args.output.resolve()
    if rank==0:
        # The old driver has no file lock: also check its live process before
        # allowing continuation, including a continuation into a new folder.
        for folder in {out, args.resume.resolve().parent if args.resume else out}:
            status_path=folder/'status.json'
            if status_path.exists():
                old_status=json.loads(status_path.read_text())
                pid=old_status.get('pid')
                process_path=Path(f'/proc/{pid}/cmdline')
                if pid and process_path.exists() and str(folder).encode() in process_path.read_bytes():
                    raise RuntimeError(f'A training process still owns {folder} (PID {pid})')
        if resume and not args.offline:
            wandb_path=args.resume.resolve().parent/'wandb.json'
            if wandb_path.exists() and json.loads(wandb_path.read_text())['run_id']!=args.run_id:
                raise ValueError('Resume must use the original W&B run ID')
        out.mkdir(parents=True,exist_ok=bool(resume))
        writer_lock=(out/'.training.lock').open('a')
        fcntl.flock(writer_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        _OWNED_OUTPUT=True
    dist.barrier()
    if rank==0:
        provenance=out/'resumes'/str(time.time_ns()) if resume else out
        provenance.mkdir(parents=True,exist_ok=True)
        manifest=json.loads((SOURCE_RUN/'stage2-source-manifest.json').read_text())
        for name,expected in manifest.items():
            assert file_sha256(SNAPSHOT/name)==expected,name
        atomic_json(provenance/'dependency-manifest.json',manifest)
        (provenance/'training-script.py').write_text(Path(__file__).read_text())
        atomic_json(provenance/'source-provenance.json',dict(dependency_snapshot=str(SNAPSHOT),
            verified_files=len(manifest),driver_sha256=file_sha256(__file__),
            resume_helper_sha256=file_sha256(ROOT/'src/training/stage2_resume.py'),
            target_implementation_sha256=file_sha256(target_implementation.__file__) if target_policy else None,
            resumed_checkpoint=str(args.resume) if resume else None))
        if target_policy:
            (provenance/'stochastic_targets.py').write_text(Path(target_implementation.__file__).read_text())
    dist.barrier()
    assert not args.smoke_cache or (args.max_updates and args.offline)
    expected_images=128 if args.smoke_cache else 126227
    assert cache['images']==expected_images and cache['shape']==[expected_images,8,8,256]
    assert cache['world_size']==2 and cache['frozen_state_unchanged']
    assert cache['stage1_epochs']==(1 if args.smoke_cache else 3) and cache['smoke_only']==args.smoke_cache
    assert cache['data_protocol_sha256']==file_sha256(BASE/'reference/data-protocol.json')
    assert cache['checkpoint_sha256']==calibration['source_checkpoint_sha256']
    assert cache['codebook_sha256']==calibration['codebook_sha256']
    if rank==0:
        assert file_sha256(cache['latent_cache'])==cache['cache_sha256']
        assert file_sha256(cache['checkpoint'])==cache['checkpoint_sha256']
        assert file_sha256(cache['codebook'])==cache['codebook_sha256']
    dist.barrier()
    config=load_stage2_config(UPSTREAM,cache['checkpoint'])
    config.arch.vocab_size=config.dataset.vocab_size=32769
    config.experiment.batch_size=args.batch_size
    config.experiment.total_batch_size=2048
    config.experiment.accumulation_steps=2048//(world*args.batch_size)
    config.experiment.sample.top_k=config.sampling.top_k=sampling['top_k']
    config.experiment.sample.top_p=config.sampling.top_p=sampling['top_p']
    config.sampling.temp=sampling['temperature']
    config.loss.temp=temperature
    config.vqvae.codebook=cache['codebook']
    config.vqvae.codebook_sha256=cache['codebook_sha256']
    config.vqvae.stage1_finetune_epochs=cache['stage1_epochs']
    config.lr_control=dict(type='published_cosine_only',fid_triggered_reductions=False)
    config.recipe=dict(source='https://arxiv.org/html/2203.01941#A3',
        sampling_source='https://arxiv.org/html/2203.01941#A4',
        training_driver='Local implementation; upstream training loop was not released',
        adaptations=['Frozen LASER compact 32769-word tokenizer instead of RQVAE',
            f'Calibrated LASER target temperature {temperature} instead of raw RQVAE temperature 0.5',
            'Two GPUs with gradient accumulation instead of four GPUs'],
        checkpoint_policy=('Latest full state and best validation/FID model weights' if args.keep_best else
                           'Latest full checkpoint only; best metric values logged to W&B'))
    if target_policy:
        config.target_policy=dict(kind=target_policy,calibration_sha256=file_sha256(args.calibration),
            implementation_sha256=file_sha256(target_implementation.__file__),
            target_entropy_nats_per_depth=calibration['target_entropy_nats_per_depth'])
    config.data_protocol=json.loads((BASE/'reference/data-protocol.json').read_text())
    config.training_precision='FP16 model autocast; FP32 codebook geometry and cross entropy'
    if resume:
        for field in ('arch','optimizer','loss','experiment'):
            expected=OmegaConf.to_container(config[field],resolve=True)
            saved=resume['config'][field]
            if field=='experiment':
                expected=dict(expected);saved=dict(saved)
                expected.pop('sample',None);saved.pop('sample',None)
            if expected!=saved:
                raise ValueError(f'Resume training configuration mismatch: {field}')
    run=None
    if rank==0:
        OmegaConf.save(config,provenance/'config.yaml')
        atomic_json(provenance/'cache-provenance.json',cache)
        atomic_json(provenance/'temperature-calibration.json',calibration)
        if not args.offline:
            for name in ('WANDB_SERVICE','_WANDB_SERVICE'):
                os.environ.pop(name,None)
            import wandb
            run=wandb.init(project='laser',entity='helloimlixin-rutgers',id=args.run_id,
                name=args.run_id,resume='must' if resume else 'never',dir=str(out),config=None if resume else {
                    **OmegaConf.to_container(config,resolve=True),
                    'pipeline':'laser-three-epoch-finetune-compact-rq2-paper-batch2048-cosine',
                    'stage2_from_scratch':True,'stage1_frozen':True,
                    'upstream_commit':'341395e562ac347f5eb62db9f5f08b9f2cc42a60',
                    'stage2_trainer':'new driver; upstream did not release training loop'})
            atomic_json(out/'wandb.json',dict(run_id=run.id,url=run.url))
            if resume:
                run.summary['resumed_from_checkpoint']=str(args.resume)
    tokenizer=FrozenCompactTokenizer(cache['checkpoint'],cache['codebook']).to(device).eval()
    assert state_sha256(tokenizer)==cache['frozen_state_sha256']
    if target_policy:
        install_compact_target_policy(tokenizer.quantizer)
    seed_all(0)
    model,optimizer=fresh_transformer(config)
    initial_hash=state_sha256(model)
    hashes=[None]*world
    dist.all_gather_object(hashes,initial_hash)
    assert len(set(hashes))==1 and not optimizer.state
    assert sum(p.numel() for p in model.parameters())==386882561
    if rank==0 and not resume:
        initialization=dict(stage2_from_scratch=True,seed=0,pretrained_stage2_checkpoint=None,
            initial_optimizer_entries=0,initial_weights_sha256=initial_hash,
            initial_weights_identical_across_ranks=True,parameters=386882561,
            tokenizer_checkpoint=cache['checkpoint'],tokenizer_sha256=cache['checkpoint_sha256'],
            codebook_sha256=cache['codebook_sha256'],temperature=temperature,
            stage1_finetune_epochs=cache['stage1_epochs'],
            upstream_commit='341395e562ac347f5eb62db9f5f08b9f2cc42a60')
        atomic_json(provenance/'initialization.json',initialization)
        print(json.dumps(initialization),flush=True)
        if run:
            run.summary.update(initialization)
            run.summary['training_status']='training_from_scratch'
    model.to(device)
    optimizer=create_resnet_optimizer(model,config.optimizer)
    ddp=DistributedDataParallel(model,device_ids=[device.index],broadcast_buffers=False)
    dataset=CachedLatents(cache['latent_cache'])
    if args.smoke_cache:
        dataset=torch.utils.data.ConcatDataset([dataset]*32)  # two full 2048-image diagnostic updates
    sampler=DistributedSampler(dataset,num_replicas=world,rank=rank,shuffle=True,seed=0)
    loader=DataLoader(dataset,sampler=sampler,batch_size=args.batch_size,num_workers=8,
        pin_memory=True,persistent_workers=True,drop_last=False)
    accumulation=2048//(world*args.batch_size)
    assert accumulation*world*args.batch_size==2048
    steps_per_epoch=math.ceil(len(loader)/accumulation)
    assert steps_per_epoch==(2 if args.smoke_cache else 62)
    scheduler=create_scheduler(optimizer,config.optimizer.warmup,steps_per_epoch,300)
    scaler=torch.amp.GradScaler('cuda',init_scale=65536.)
    start_epoch=start_batch=0
    pending_rng=None
    if resume:
        start_epoch,start_batch=validate_resume_payload(resume,world=world,batch_size=args.batch_size,
            cache=cache,calibration=calibration,loader_batches=len(loader),accumulation=accumulation)
        restore_training_state(resume,model,optimizer,scheduler,scaler)
        initial_hash=resume['initial_weights_sha256']
        pending_rng=resume['rng_states'][rank]
    best_fid=best_validation=float('inf')
    heldout=torch.load(args.cache/f'validation-rank{rank}.pt',map_location='cpu',weights_only=True)
    inception=None
    if not args.max_updates:
        inception=get_inception_model().eval().requires_grad_(False).to(device)
    reference=BASE/'reference/real-statistics.npz'
    assert reference.is_file()
    reference_receipt=json.loads((BASE/'reference/complete.json').read_text())
    assert file_sha256(reference)==reference_receipt['sha256']
    assert reference_receipt['data_protocol_sha256']==file_sha256(BASE/'reference/data-protocol.json')
    assert reference_receipt['images']==126227 and reference_receipt['padded_or_dropped_images']==0
    fid_protocol=dict(generated_samples=50000,real_samples=reference_receipt['images'],
        reference_sha256=reference_receipt['sha256'],sampling=sampling,
        decode_and_inception_batch_size=args.decode_batch_size)
    if resume:
        best_validation=resume.get('best_validation',float('inf'))
        if resume.get('fid_protocol')==fid_protocol:
            best_fid=resume.get('best_fid',float('inf'))
    seed_all(rank)
    stopping={'requested':False}
    def on_signal(signum,frame):
        stopping['requested']=True
    signal.signal(signal.SIGTERM,on_signal)
    signal.signal(signal.SIGINT,on_signal)
    step=attempts=skipped=consecutive_skips=0
    if resume:
        step=resume['step'];attempts=resume['attempts'];skipped=resume['skipped_amp_updates']
        consecutive_skips=resume.get('consecutive_amp_skips',0)
        # End-of-epoch states from the original driver do not distinguish
        # pre-evaluation from post-evaluation. Re-evaluate that boundary; its
        # stochastic evaluators fork RNG and cannot alter the training stream.
        pending_evaluation=resume.get('pending_evaluation_epoch')
        if pending_evaluation is None and start_epoch>0 and start_batch==0 and 'fid_protocol' not in resume:
            pending_evaluation=start_epoch
        if pending_evaluation is not None:
            start_epoch=pending_evaluation-1
            start_batch=len(loader)
        if rank==0:
            atomic_json(provenance/'resume.json',dict(checkpoint=str(args.resume),epoch=start_epoch,
                batch_in_epoch=start_batch,optimizer_step=step,attempted_updates=attempts,
                skipped_amp_updates=skipped,fid_protocol=fid_protocol))
        del resume
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

    def save(epoch,batch,*,pending_evaluation_epoch=None):
        states=[None]*world
        dist.all_gather_object(states,capture_rng_state(device))
        if rank==0:
            dest=out/'last.pt'
            torch.save(dict(epoch=epoch,batch_in_epoch=batch,step=step,attempts=attempts,
                skipped_amp_updates=skipped,state_dict=model.state_dict(),optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng_states=states,
                best_fid=best_fid,best_validation=best_validation,
                consecutive_amp_skips=consecutive_skips,fid_protocol=fid_protocol,
                pending_evaluation_epoch=pending_evaluation_epoch,
                tokenizer=cache,temperature_calibration=calibration,
                initial_weights_sha256=initial_hash,config=OmegaConf.to_container(config,resolve=True)),
                dest.with_suffix('.tmp'))
            dest.with_suffix('.tmp').replace(dest)
        dist.barrier()

    def save_best(filename,epoch,metric):
        if args.keep_best and rank==0:
            dest=out/filename
            torch.save(dict(epoch=epoch,step=step,state_dict=model.state_dict(),
                tokenizer=cache,temperature_calibration=calibration,selection_metric=metric,
                config=OmegaConf.to_container(config,resolve=True),fid_protocol=fid_protocol,
                checkpoint_kind='evaluation_model_weights'),dest.with_suffix('.tmp'))
            dest.with_suffix('.tmp').replace(dest)
        dist.barrier()

    for epoch_index in range(start_epoch,300):
        model.train()
        consumed=start_batch if epoch_index==start_epoch else 0
        iterator=resumed_iterator(loader,sampler,epoch_index,consumed,rng_state=pending_rng,device=device)
        pending_rng=None
        for update_index in range(math.ceil(consumed/accumulation),steps_per_epoch):
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
                    with torch.no_grad(), torch.random.fork_rng(devices=[device.index]):
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
        save(epoch,0,pending_evaluation_epoch=epoch)
        # Latest full checkpoint already saved above.
        dist.barrier()
        if not args.max_updates and (epoch==1 or epoch%args.validation_every==0):
            validation=evaluate_validation(model,tokenizer,heldout,device,rank,temperature)
            if target_policy:
                entropy=validation_target_entropy(tokenizer,heldout,device,rank,temperature)
                validation['target_entropy']=sum(entropy)/len(entropy)
                validation['target_kl']=validation['soft_ce']-validation['target_entropy']
                validation.update({f'depth_{d}_target_entropy':h for d,h in enumerate(entropy)})
            if validation['soft_ce']<best_validation:
                best_validation=validation['soft_ce']
                save_best('best-validation.pt',epoch,dict(name='validation/soft_ce',value=best_validation))
                if rank==0:
                    atomic_json(out/'best-validation.json',dict(epoch=epoch,step=step,**validation))
            if rank==0:
                atomic_json(out/f'validation-epoch{epoch:03d}.json',dict(epoch=epoch,**validation))
                if run:
                    run.log({'validation/epoch':epoch,**{f'validation/{k}':v for k,v in validation.items()}})
        if not args.max_updates and epoch>=args.fid_start_epoch and (epoch==1 or epoch%args.fid_every==0):
            score=evaluate_samples(model,tokenizer,inception,50000,epoch,out,reference,device,rank,world,sampling,args.decode_batch_size)
            assert state_sha256(tokenizer)==cache['frozen_state_sha256']
            if score<best_fid:
                best_fid=score
                save_best('best-fid-50000.pt',epoch,dict(name='generation/fid_50000',value=best_fid))
                if rank==0:
                    atomic_json(out/'best-fid-50000.json',dict(epoch=epoch,step=step,fid=score,**fid_protocol))
            if rank==0:
                if run:
                    import wandb
                    run.log({'generation/fid_50000':score,'generation/epoch':epoch,
                        'generation/samples':wandb.Image(str(out/f'samples-50000-epoch{epoch:03d}.png')),
                        'stage2/lr_multiplier':1.})
                    run.summary['last_fid_50000']=score
                    run.summary['best_fid_50000']=best_fid
                    run.summary['best_validation_soft_ce']=best_validation
        save(epoch,0)
        if args.stop_after_epoch is not None and epoch>=args.stop_after_epoch and epoch<300:
            status(dict(phase='paused',optimizer_step=step,epoch=epoch,reason='requested_trial_epoch_limit'))
            if run:
                run.summary['training_status']='paused_with_checkpoint'
                run.finish()
            dist.destroy_process_group()
            return
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
        if _OWNED_OUTPUT and '--output' in sys.argv:
            output=Path(sys.argv[sys.argv.index('--output')+1])
            if output.exists():
                atomic_json(output/f'failure-rank{os.environ.get("RANK","unknown")}.json',
                    dict(error_type=type(error).__name__,error=str(error),updated_unix=time.time()))
        raise
