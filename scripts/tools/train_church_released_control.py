#!/usr/bin/env python3
"""Train only the original Church prior against the frozen released RQ-VAE.

The stage-2 loop was not released; architecture, stochastic targets, loss,
initialization helper, optimizer helper, cosine scheduler and sampler are upstream.
"""
import argparse
from datetime import timedelta
import fcntl
from itertools import islice
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

bootstrap = argparse.ArgumentParser(add_help=False)
bootstrap.add_argument('--run-root', type=Path, required=True)
ROOT = bootstrap.parse_known_args()[0].run_root.resolve()
sys.path[:0] = [str(ROOT/'runtime/upstream'), str(ROOT/'runtime')]
os.environ.setdefault('TORCH_HOME', '/workspace/tmp/official-rqvae-eval-cache')

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from omegaconf import OmegaConf
from rqvae.optimizer import create_scheduler
from rqvae.optimizer.optimizer import create_resnet_optimizer
from rqvae.metrics.fid import get_inception_model
from rqvae.utils.config import augment_arch_defaults
from torchvision.utils import save_image
from src.original_rq_training import (atomic_json, file_sha256, state_sha256, seed_all,
    load_stage2_config, load_tokenizer, fresh_transformer, CachedLatents,
    accumulated_update, evaluate_validation, FeatureMoments, fid_from_moments)
from src.training.stage2_resume import (capture_rng_state, restore_rng_state,
    restore_training_state, resumed_iterator)
from src.training.church_continuation import ranked_candidates, upload_checkpoints
from src.training.church_released_control import validate_control_resume

OWNED_OUTPUT = None


@torch.inference_mode()
def preview(model, tokenizer, out, device, rank, run, *, step, epoch, sampling):
    if rank == 0:
        rng = capture_rng_state(device)
        mode = model.training
        try:
            model.eval()
            seed_all(71000)
            codes = model.sample(torch.zeros(100,8,8,4,dtype=torch.long,device=device),
                model_aux=tokenizer, **sampling, amp=True, cached=True, is_tqdm=False)
            images = torch.cat([tokenizer.decode_code(part).mul(.5).add(.5).clamp(0,1).cpu()
                                for part in codes.split(8)])
            assert images.shape == (100,3,256,256) and torch.isfinite(images).all()
            path = out/f'preview-step{step:07d}.png'
            save_image(images, path, nrow=10)
            atomic_json(out/'preview-latest.json', dict(optimizer_step=step, epoch=epoch,
                samples=100, rows=10, columns=10, seed=71000, path=str(path), sampling=sampling,
                training_rng_preserved=True))
            if run:
                import wandb
                run.log({'preview/optimizer_step':step, 'preview/epoch':epoch,
                         'preview/samples':wandb.Image(str(path))})
        finally:
            model.train(mode)
            restore_rng_state(rng, device)
        assert torch.equal(torch.get_rng_state(),rng['torch'])
        assert torch.equal(torch.cuda.get_rng_state(device),rng['cuda'])
    dist.barrier()


@torch.inference_mode()
def evaluate_fid(model, tokenizer, inception, out, device, rank, sampling, epoch):
    model.eval()
    rng = capture_rng_state(device)
    moments = FeatureMoments(device)
    try:
        seed_all(71000+rank)
        for offset in range(0,25000,100):
            codes = model.sample(torch.zeros(100,8,8,4,dtype=torch.long,device=device),
                model_aux=tokenizer, **sampling, amp=True, cached=True, is_tqdm=False)
            images = []
            for part in codes.split(8):
                decoded = tokenizer.decode_code(part).mul(.5).add(.5).clamp(0,1)
                assert torch.isfinite(decoded).all()
                moments.update(inception(decoded))
                if rank == 0 and offset == 0:
                    images.append(decoded.cpu())
            if rank == 0:
                if offset == 0:
                    save_image(torch.cat(images), out/f'samples-epoch{epoch:03d}.png', nrow=10)
                atomic_json(out/'status.json', dict(phase='evaluating_generation',epoch=epoch,
                    samples_per_rank_done=offset+100,samples_total=50000,updated_unix=time.time(),pid=os.getpid()))
        stats = moments.finish()
        assert stats[0] == 50000
        score = fid_from_moments(stats,ROOT/'reference/real-statistics.npz') if rank == 0 else 0.
        value = torch.tensor(score,device=device,dtype=torch.float64)
        dist.broadcast(value,0)
        if rank == 0:
            np.savez(out/f'statistics-epoch{epoch:03d}.npz',mu=stats[1],sigma=stats[2],samples=50000)
            atomic_json(out/f'fid-epoch{epoch:03d}.json',dict(epoch=epoch,fid=score,samples=50000,sampling=sampling))
        return value.item()
    finally:
        restore_rng_state(rng,device)
        model.train()


def main():
    global OWNED_OUTPUT
    p = argparse.ArgumentParser(description=__doc__, parents=[bootstrap])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--run-id',required=True)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--resume',type=Path)
    p.add_argument('--max-updates',type=int,help='Offline preflight only; production always starts fresh')
    p.add_argument('--offline',action='store_true')
    p.add_argument('--preflight-preview',action='store_true')
    p.add_argument('--memory-fraction',type=float,default=.40)
    args = p.parse_args()
    if args.batch_size < 1 or 2048 % (2*args.batch_size):
        p.error('Per-rank microbatch must divide global batch 2048 over two GPUs')
    if args.max_updates and (not args.offline or args.max_updates < 1):
        p.error('Bounded preflights must be offline and have positive update counts')
    rank,world = int(os.environ['RANK']),int(os.environ['WORLD_SIZE'])
    assert world == 2
    device = torch.device('cuda',int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction,device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    dist.init_process_group('nccl',timeout=timedelta(hours=3))
    out = args.output.resolve()
    if rank == 0:
        out.mkdir(parents=True,exist_ok=bool(args.resume))
        lock = (out/'.training.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    dist.barrier()
    OWNED_OUTPUT = out
    for name,digest in json.loads((ROOT/'runtime-manifest.json').read_text()).items():
        assert file_sha256(ROOT/'runtime'/name) == digest,name
    cache = json.loads((ROOT/'cache/complete.json').read_text())
    assert cache['images'] == 126227 and cache['shape'] == [126227,8,8,256]
    assert cache['frozen_state_unchanged'] and not cache['hard_codes_cached']
    assert file_sha256(cache['checkpoint']) == cache['checkpoint_sha256']
    assert cache['checkpoint_sha256'] == 'ba008ec2e192a6d4084a8fd511927a789c68a8459d6e8bfc22122ae02887b800'
    assert file_sha256(cache['config']) == cache['config_sha256']
    if rank == 0:
        assert file_sha256(cache['latent_cache']) == cache['cache_sha256']
    dist.barrier()
    assert file_sha256(ROOT/'cache'/f'validation-rank{rank}.pt') == cache['validation_sha256'][rank]
    reference = json.loads((ROOT/'reference/complete.json').read_text())
    assert file_sha256(ROOT/'reference/real-statistics.npz') == reference['sha256']
    assert file_sha256(ROOT/'reference/data-protocol.json') == reference['data_protocol_sha256'] == cache['data_protocol_sha256']
    tokenizer,_ = load_tokenizer(cache['checkpoint'],cache['config'],device)
    assert state_sha256(tokenizer) == cache['frozen_state_sha256']
    config = load_stage2_config(ROOT/'runtime/upstream',cache['checkpoint'])
    published = OmegaConf.load(ROOT/'tokenizer/transformer-config.yaml')
    assert OmegaConf.to_container(config.arch,resolve=True) == OmegaConf.to_container(augment_arch_defaults(published.arch),resolve=True)
    config.experiment.batch_size = args.batch_size
    config.experiment.total_batch_size = 2048
    config.experiment.accumulation_steps = 2048//(2*args.batch_size)
    config.experiment.sample.top_k = config.sampling.top_k = 1400
    config.experiment.sample.top_p = config.sampling.top_p = 1.
    assert config.optimizer.init_lr == .0005 and config.optimizer.weight_decay == .0001
    assert list(config.optimizer.betas) == [.9,.95] and config.loss.temp == .5
    assert config.experiment.epochs == 300 and config.arch.vocab_size == 16384
    sampling = dict(temperature=1.,top_k=1400,top_p=1.)
    protocol = dict(generated_samples=50000,real_samples=126227,reference_sha256=reference['sha256'],
        sampling=sampling,seed=71000,world_size=2,generation_batch_per_gpu=100,decode_and_inception_batch_size=8)
    config_dict = OmegaConf.to_container(config,resolve=True)
    if rank == 0:
        OmegaConf.save(config,out/'config.yaml')
    seed_all(0)
    model,unused_optimizer = fresh_transformer(config)
    assert len(unused_optimizer.state) == 0
    del unused_optimizer
    initial_hash = state_sha256(model)
    hashes = [None]*world
    dist.all_gather_object(hashes,initial_hash)
    assert len(set(hashes)) == 1
    assert sum(p.numel() for p in model.parameters()) == 370087936
    model.to(device)
    optimizer = create_resnet_optimizer(model,config.optimizer)
    assert not optimizer.state
    ddp = DistributedDataParallel(model,device_ids=[device.index],broadcast_buffers=False)
    dataset = CachedLatents(cache['latent_cache'])
    sampler = DistributedSampler(dataset,num_replicas=world,rank=rank,shuffle=True,seed=0)
    loader = DataLoader(dataset,sampler=sampler,batch_size=args.batch_size,num_workers=4,
        pin_memory=True,persistent_workers=True,drop_last=False)
    accumulation = config.experiment.accumulation_steps
    steps_per_epoch = math.ceil(len(loader)/accumulation)
    assert steps_per_epoch == 62
    scheduler = create_scheduler(optimizer,config.optimizer.warmup,steps_per_epoch,300)
    scaler = torch.amp.GradScaler('cuda',init_scale=65536.)
    heldout = torch.load(ROOT/'cache'/f'validation-rank{rank}.pt',map_location='cpu',weights_only=True)
    inception = get_inception_model().eval().requires_grad_(False).to(device)
    # Exercise the actual metric backend before committing to a long training job.
    with torch.inference_mode():
        features = inception(tokenizer.decode(tokenizer.quantizer(heldout[:2].to(device))[0]).mul(.5).add(.5).clamp(0,1))
        assert features.shape == (2,2048) and torch.isfinite(features).all()
    start_epoch = start_batch = step = attempts = skipped = consecutive = 0
    pending_rng = None
    ranked = []
    best_validation = float('inf')
    if args.resume:
        payload = torch.load(args.resume,map_location='cpu',weights_only=False,mmap=True)
        start_epoch,start_batch = validate_control_resume(payload,config=config_dict,cache=cache,
            protocol=protocol,run_id=args.run_id,loader_batches=len(loader))
        restore_training_state(payload,model,optimizer,scheduler,scaler)
        step,attempts,skipped = payload['step'],payload['attempts'],payload['skipped_amp_updates']
        consecutive = payload.get('consecutive_amp_skips',0)
        initial_hash = payload['initial_weights_sha256']
        pending_rng = payload['rng_states'][rank]
        ranked = payload['ranked_fid_checkpoints']
        best_validation = payload.get('best_validation',float('inf'))
        if payload.get('pending_evaluation_epoch') is not None:
            start_epoch = payload['pending_evaluation_epoch']-1
            start_batch = len(loader)
        del payload
        if args.max_updates and args.max_updates <= step:
            raise ValueError('Preflight update limit must exceed restored step')
        for row in ranked:
            assert (out/row['path']).is_file(),'Restore retained FID checkpoints beside last.pt'
    run = None
    if rank == 0 and not args.offline:
        import wandb
        for name in ['WANDB_SERVICE','_WANDB_SERVICE']:
            os.environ.pop(name,None)
        run = wandb.init(project='laser',entity='helloimlixin-rutgers',id=args.run_id,name=args.run_id,
            resume='must' if args.resume else 'never',dir=str(out),config={**config_dict,
                'pipeline':'released-Church-RQVAE-frozen-original-RQTransformer-control',
                'stage1_finetune':False,'stage2_from_scratch':True,'initialization_seed':0,
                'tokenizer_checkpoint_sha256':cache['checkpoint_sha256'],
                'lr_schedule':'paper cosine 5e-4 to zero; 18600 updates; no FID-dependent changes',
                'preview_every_steps':200,'preview_rows':10,'preview_columns':10,'preview_seed':71000,
                'fid_every_epochs':10,'validation_every_epochs':5,
                'checkpoint_policy':'full last plus best three FID50k; upload at steps 1 and 25, every epoch, graceful stop',
                'training_loop_provenance':'local driver; upstream did not release stage-2 training loop'})
        atomic_json(out/'wandb.json',dict(run_id=run.id,url=run.url))
    if rank == 0:
        atomic_json(out/('resume.json' if args.resume else 'initialization.json'),dict(
            stage2_from_scratch=not bool(args.resume),pretrained_stage2_checkpoint=None,
            initial_optimizer_entries=0,initial_weights_sha256=initial_hash,parameters=370087936,
            initial_weights_identical_across_ranks=True,resumed_checkpoint=str(args.resume) if args.resume else None,
            initial_lr=.0005,steps_per_epoch=62,planned_optimizer_steps=18600,
            runtime_manifest_sha256=file_sha256(ROOT/'runtime-manifest.json'),fid_protocol=protocol))
    seed_all(rank)
    stopping = {'requested':False}
    def on_signal(signum,frame):
        stopping['requested'] = True
    signal.signal(signal.SIGTERM,on_signal)
    signal.signal(signal.SIGINT,on_signal)
    started = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    def status(**record):
        if rank == 0:
            record.update(updated_unix=time.time(),elapsed_seconds=time.time()-started,pid=os.getpid())
            atomic_json(out/'status.json',record)
            with (out/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
            if run:
                rng = capture_rng_state(device)
                try:
                    run.log({f'stage2/{k}':v for k,v in record.items() if isinstance(v,(int,float))})
                finally:
                    restore_rng_state(rng,device)
    def save(epoch,batch,filename='last.pt',pending=None):
        states = [None]*world
        dist.all_gather_object(states,pending_rng if pending_rng is not None else capture_rng_state(device))
        if rank == 0:
            destination = out/filename
            torch.save(dict(epoch=epoch,batch_in_epoch=batch,step=step,attempts=attempts,
                skipped_amp_updates=skipped,consecutive_amp_skips=consecutive,
                state_dict=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict(),rng_states=states,config=config_dict,tokenizer=cache,
                run_id=args.run_id,initial_weights_sha256=initial_hash,fid_protocol=protocol,
                ranked_fid_checkpoints=ranked,best_validation=best_validation,pending_evaluation_epoch=pending),
                destination.with_suffix('.tmp'))
            destination.with_suffix('.tmp').replace(destination)
        dist.barrier()
    def upload(epoch):
        if rank == 0 and run:
            rng = capture_rng_state(device)
            try:
                upload_checkpoints(run,out,ranked,epoch=epoch,step=step,protocol=protocol)
                retained = {row['path'] for row in ranked}
                for path in out.glob('fid-epoch*-step*.pt'):
                    if path.name not in retained:
                        path.unlink()
            finally:
                restore_rng_state(rng,device)
        dist.barrier()
    status(phase='resumed' if args.resume else 'ready',optimizer_step=step,epoch=start_epoch,
           lr=optimizer.param_groups[0]['lr'],global_batch=2048,tokenizer_frozen=True)
    for epoch_index in range(start_epoch,300):
        model.train()
        consumed = start_batch if epoch_index == start_epoch else 0
        iterator = resumed_iterator(loader,sampler,epoch_index,consumed,rng_state=pending_rng,device=device)
        pending_rng = None
        loss_sum = 0.
        image_count = 0
        for update_index in range(math.ceil(consumed/accumulation),steps_per_epoch):
            batches = list(islice(iterator,accumulation))
            assert batches
            consumed += len(batches)
            metrics = accumulated_update(ddp,tokenizer,optimizer,scaler,batches,max_gn=config.optimizer.max_gn)
            decisions = [None]*world
            dist.all_gather_object(decisions,metrics['optimizer_updated'])
            assert len(set(decisions)) == 1,'Optimizer success differs between ranks'
            attempts += 1
            if metrics['optimizer_updated']:
                scheduler.step()
                step += 1
                consecutive = 0
            else:
                skipped += 1
                consecutive += 1
            loss_sum += metrics['loss']*metrics['global_images']
            image_count += metrics['global_images']
            epoch_fraction = epoch_index+(update_index+1)/steps_per_epoch
            if attempts <= 10 or attempts % 10 == 0:
                status(phase='training',optimizer_step=step,attempted_updates=attempts,epoch=epoch_fraction,
                    lr=optimizer.param_groups[0]['lr'],effective_batch_size=2048,accumulation_steps=accumulation,
                    skipped_amp_updates=skipped,peak_gpu_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3,**metrics)
            if consecutive >= 8:
                save(epoch_index,consumed)
                upload(epoch_fraction)
                raise FloatingPointError('Eight consecutive AMP skips; checkpoint saved')
            if metrics['optimizer_updated'] and step % 200 == 0:
                preview(model,tokenizer,out,device,rank,run,step=step,epoch=epoch_fraction,sampling=sampling)
            if args.max_updates and step == 2 and metrics['optimizer_updated']:
                save(epoch_index,consumed,filename='preflight-step2.pt')
            stop = torch.tensor(int(stopping['requested']),device=device)
            dist.all_reduce(stop,op=dist.ReduceOp.MAX)
            smoke_done = args.max_updates and step >= args.max_updates
            if stop.item() or smoke_done:
                assert state_sha256(tokenizer) == cache['frozen_state_sha256']
                if smoke_done:
                    validation = evaluate_validation(model,tokenizer,heldout,device,rank)
                    if rank == 0:
                        atomic_json(out/'validation-preflight.json',validation)
                    if args.preflight_preview:
                        preview(model,tokenizer,out,device,rank,run,step=step,epoch=epoch_fraction,sampling=sampling)
                save(epoch_index,consumed)
                upload(epoch_fraction)
                status(phase='preflight_complete' if smoke_done else 'paused',optimizer_step=step,
                    epoch=epoch_fraction,final_weights_sha256=state_sha256(model),tokenizer_unchanged=True,
                    peak_gpu_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3)
                if run:
                    run.summary['training_status']='paused_with_checkpoint'
                    run.finish()
                dist.destroy_process_group()
                return
            if metrics['optimizer_updated'] and step in [1,25]:
                save(epoch_index,consumed)
                upload(epoch_fraction)
        assert consumed == len(loader)
        epoch = epoch_index+1
        save(epoch,0,pending=epoch)
        if epoch == 1 or epoch % 5 == 0:
            validation = evaluate_validation(model,tokenizer,heldout,device,rank)
            best_validation = min(best_validation,validation['soft_ce'])
            if rank == 0:
                atomic_json(out/f'validation-epoch{epoch:03d}.json',dict(epoch=epoch,**validation))
                if run:
                    run.log({'validation/epoch':epoch,**{f'validation/{k}':v for k,v in validation.items()}})
        if epoch == 1 or epoch % 10 == 0:
            score = evaluate_fid(model,tokenizer,inception,out,device,rank,sampling,epoch)
            assert state_sha256(tokenizer) == cache['frozen_state_sha256']
            ranked = ranked_candidates(ranked,fid=score,epoch=epoch,step=step)
            for row in ranked:
                if row['step'] == step:
                    save(epoch,0,filename=row['path'])
            if rank == 0:
                atomic_json(out/'ranked-fid.json',dict(checkpoints=ranked,protocol=protocol))
                if run:
                    import wandb
                    run.log({'generation/epoch':epoch,'generation/fid_50000':score,
                        'generation/samples':wandb.Image(str(out/f'samples-epoch{epoch:03d}.png'))})
                    run.summary['best_fid_50000']=ranked[0]['fid']
        save(epoch,0)
        upload(epoch)
        if image_count:
            status(phase='epoch_complete',epoch=epoch,optimizer_step=step,epoch_soft_ce=loss_sum/image_count)
    status(phase='complete',epochs=300,optimizer_step=step)
    if run:
        run.summary['training_status']='complete'
        run.finish()
    dist.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if OWNED_OUTPUT is not None:
            atomic_json(OWNED_OUTPUT/f'failure-rank{os.environ.get("RANK","unknown")}.json',
                dict(error_type=type(error).__name__,error=str(error),updated_unix=time.time()))
        raise
