#!/usr/bin/env python3
"""Continue the corrected scratch Church run on one or more GPUs."""
import argparse
import codecs
from datetime import timedelta
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image
from scripts.tools.build_sign_probe_cache import sha256_file
from archive.scripts.train_church_ffhq_recipe import write_json
from src.church_ffhq_archived import full_training_cache, FullBatchEpochStream, ARCHIVE_SHA256, UPSTREAM_CONFIG
from src.church_joint_geometry import make_prior, objective, early_decay_lr
from src.church_joint_distributed import JointObjective, backward_batch, generation_range
from src.church_relative_noise import RelativeChurchAux
from src.ffhq_v4_archived import atomic_torch_save, scheduled_geometry_weight
from src.rqvae_metrics import DistributedOriginalRQVAEMetrics


@torch.no_grad()
def evaluate(model, aux, data, rank, world):
    model.eval()
    totals = {}
    for first in range(rank*16, len(data['atoms']), world*16):
        atoms = data['atoms'][first:first+16].cuda().long()
        _, metrics = objective(model, aux, atoms, data['coefficients'][first:first+16].cuda(), .05, stochastic=False)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.) + value*len(atoms)
    keys=sorted(totals)
    values=torch.tensor([totals[k] for k in keys], device='cuda', dtype=torch.float64)
    if world > 1: dist.all_reduce(values)
    return {k:v/len(data['atoms']) for k,v in zip(keys,values.cpu().tolist())}


@torch.no_grad()
def generate(model, aux, config, args, directory, count, seed, step, rank, world, log):
    directory.mkdir(parents=True, exist_ok=True)
    model.eval()
    metric=DistributedOriginalRQVAEMetrics('cuda', reference_stats_path=config['fid_stats'])
    first, end=generation_range(count, rank, world)
    identity={'step':step, 'count':count, 'seed':seed, 'batch':args.generation_batch,
        'world_size':world, 'rank':rank, 'first':first, 'end':end,
        'sampler':'archived k250 atom p1 coefficient p0.85 temperatures1 AMP',
        'seed_rule':'base seed + rank * 1000003; RNG state checkpointed per shard'}
    shard_path=directory/f'shard-{rank:02d}.pt'
    atoms_all, ids_all=[], []
    done=0
    started=time.monotonic()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed+rank*1000003)
        if shard_path.exists():
            saved=torch.load(shard_path,map_location='cpu',weights_only=False)
            if saved['identity'] != identity: raise ValueError('Evaluation shard protocol changed')
            atoms_all=[saved['atoms']]; ids_all=[saved['coefficient_ids']]
            done=len(saved['atoms'])
            for key in ('fake_sum','fake_cross','fake_count'):
                getattr(metric,key).copy_(saved[key])
            torch.set_rng_state(saved['cpu_rng'])
            torch.cuda.set_rng_state(saved['cuda_rng'])
            del saved
        while first+done < end:
            batch=min(args.generation_batch,end-first-done)
            atoms,ids=model.sample_compound(batch,aux,atom_top_k=250,atom_top_p=1.,coeff_top_p=.85,
                atom_temperature=1.,coeff_temperature=1.,amp=True)
            atoms_all.append(atoms.cpu().short()); ids_all.append(ids.cpu().short())
            for offset in range(0,batch,32):
                images=((aux.decode_compound(atoms[offset:offset+32],ids[offset:offset+32])+1)/2).clamp(0,1)
                metric.update(images,real=False)
                if rank==0 and done==0 and offset==0: save_image(images,directory/'samples.png',nrow=8)
            done+=batch
            if done % (4*args.generation_batch)==0 or first+done==end:
                atoms_all=[torch.cat(atoms_all)];ids_all=[torch.cat(ids_all)]
                atomic_torch_save({'identity':identity,'atoms':atoms_all[0],'coefficient_ids':ids_all[0],
                    **{k:getattr(metric,k).cpu() for k in ('fake_sum','fake_cross','fake_count')},
                    'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},shard_path)
            row={'phase':'generation','rank':rank,'generated_on_rank':done,'target_on_rank':end-first,
                'target_samples':count,'generation_seconds':time.monotonic()-started,'generation_batch':args.generation_batch}
            write_json(directory/f'progress-{rank:02d}.json',{'pid':os.getpid(),**row})
            if rank==0: log(row)
        del atoms_all,ids_all
    fid,_,_=metric.compute()
    assert int(metric.fake_count)==count
    result={'fid':float(fid),'samples':count,'seed':seed,'atom_top_k':250,
        'coefficient_sampling':'archived FFHQ nucleus p=0.85','temperature':1.,
        'precision':'archived sampler amp=True; FP32 decoder; Church original RQ-VAE Inception',
        'seconds':time.monotonic()-started,'world_size':world,'generation_batch_per_gpu':args.generation_batch,
        'seed_rule':identity['seed_rule'],'optimizer_step':step}
    if rank==0:
        shards=[torch.load(directory/f'shard-{r:02d}.pt',map_location='cpu',weights_only=False) for r in range(world)]
        codes={k:torch.cat([s[k] for s in shards]) for k in ('atoms','coefficient_ids')}
        assert len(codes['atoms'])==count
        atomic_torch_save(codes,directory/'generated-codes.pt')
        write_json(directory/'metrics.json',result)
    if world>1: dist.barrier()
    model.init_cache()
    del metric
    gc.collect();torch.cuda.empty_cache()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-checkpoint',type=Path)
    parser.add_argument('--microbatch',type=int,default=32)
    parser.add_argument('--generation-batch',type=int,default=256)
    parser.add_argument('--stop-after-step',type=int,default=0)
    parser.add_argument('--verification-skip-evaluation',action='store_true')
    parser.add_argument('--wandb-id')
    args=parser.parse_args()
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    local_rank=int(os.environ.get('LOCAL_RANK',0));torch.cuda.set_device(local_rank)
    if world>1: dist.init_process_group('nccl',timeout=timedelta(minutes=30))
    if args.verification_skip_evaluation and (args.wandb_id or args.output.resolve()==ROOT/'outputs/church-joint-geometry-20260912/joint'):
        parser.error('Cannot skip production evaluations')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8)
    torch.serialization.add_safe_globals([codecs.encode])
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=True
    source=args.source_checkpoint or args.output/'last.pt'
    saved=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
    config=saved['config']
    if config['batch_size'] % (world*args.microbatch): parser.error('Unequal rank microbatches')
    assert config['parameters']==404738048 and config['batch_size']==256
    assert config['initialization']['kind']=='random' and config['stage2_checkpoint_loaded'] is None
    assert config['archived_code_sha256']==sha256_file(ROOT/'src/ffhq_v4_archived.py')==ARCHIVE_SHA256
    assert config['upstream_config_sha256']==sha256_file(UPSTREAM_CONFIG)
    assert config['source_hashes']['src/church_joint_geometry.py']==sha256_file(ROOT/'src/church_joint_geometry.py')
    assert config['calibration_sha256']==sha256_file(Path(config['calibration']))
    assert config['cache_sha256']==sha256_file(Path(config['cache']))
    assert config['stage1_sha256']==sha256_file(Path(config['stage1']))
    assert config['source_hashes']['src/church_relative_noise.py']==sha256_file(ROOT/'src/church_relative_noise.py')
    torch.manual_seed(config['seed'])
    model=make_prior().cuda()
    model.load_state_dict(saved['state_dict'],strict=True)
    aux=RelativeChurchAux(Path(config['stage1']),16384,2048,3.,coeff_scales=config['coeff_scales'],
        sparsity_level=4,soft_target_physical=False,sigma_cap=config['coefficient_noise_sigma_cap'],
        relative_sigma=config['relative_sigma'],truncate=config['truncate']).cuda().eval().requires_grad_(False)
    raw=torch.load(config['cache'],map_location='cpu',weights_only=False,mmap=True)
    splits,scales=full_training_cache(raw);del raw
    assert scales==config['coeff_scales']
    stream=FullBatchEpochStream(len(splits['train']['atoms']),config['seed']+1)
    stream.load_state_dict(saved['stream'])
    optimizer=torch.optim.AdamW([{'params':model.parameters(),'weight_decay':1e-4,'lr_multiplier':1.}],
        lr=config['lr'],betas=(.9,.95),fused=True)
    optimizer.load_state_dict(saved['optimizer'])
    step,elapsed,best_fid,best_step,bad_checks,pending_eval,initialized=[saved[k] for k in
        ('step','elapsed_seconds','best_fid','best_step','bad_checks','pending_eval','initialized')]
    assert len(optimizer.state)>0
    del saved
    wrapped=JointObjective(model,aux)
    if world>1: wrapped=DDP(wrapped,device_ids=[local_rank],broadcast_buffers=False,
        find_unused_parameters=True,gradient_as_bucket_view=True)
    # DDP's initial replica synchronization writes parameters once. Freeze
    # version tracking starts after that initialization, before any updates.
    frozen=[(v,v._version) for v in (*aux.parameters(),*aux.buffers())]
    execution={'world_size':world,'microbatch_per_gpu':args.microbatch,'effective_batch':config['batch_size'],
        'generation_batch_per_gpu':args.generation_batch,'resumed_step':step,'source_checkpoint':str(source),
        'optimizer_restored':True,'stream_restored':True,'precision':'unchanged FP32/TF32 training',
        'source_hashes':{p:sha256_file(ROOT/p) for p in ['src/church_joint_distributed.py',
            'scripts/tools/train_church_joint_distributed.py','src/church_ffhq_archived.py','src/training/rqtransformer.py']}}
    if rank==0: write_json(args.output/'execution.json',execution)
    wb=None
    if rank==0 and args.wandb_id:
        import wandb
        wb=wandb.init(entity='helloimlixin-rutgers',project='laser',id=args.wandb_id,
            dir=str(args.output),resume='must')
        wb.config.update({'distributed_execution':execution},allow_val_change=True)
        wb.summary['training_status']='running'
    started=time.monotonic()
    stopping={'signal':None}
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,lambda number,frame:stopping.update(signal=number))

    def log(row):
        if rank!=0:return
        full={'optimizer_step':step,'epoch_progress':stream.epoch+stream.position/stream.size,
            'elapsed_seconds':elapsed+time.monotonic()-started,**row}
        print(json.dumps(full,allow_nan=False),flush=True)
        with (args.output/'history.jsonl').open('a') as handle:handle.write(json.dumps(full,allow_nan=False)+'\n')
        write_json(args.output/'status.json',{'pid':os.getpid(),'best_fid':best_fid,'best_step':best_step,
            'bad_checks':bad_checks,'world_size':world,**full})
        if wb:wb.log(full)

    def save_last():
        model.init_cache()
        assert all(v._version==version and v.grad is None for v,version in frozen)
        if rank==0:
            atomic_torch_save({'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),
                'stream':stream.state_dict(),'config':config,'step':step,'best_fid':best_fid,'best_step':best_step,
                'bad_checks':bad_checks,'pending_eval':pending_eval,'initialized':initialized,
                'elapsed_seconds':elapsed+time.monotonic()-started,'execution':execution},args.output/'last.pt')
            log({'phase':'checkpoint_saved'})
        if world>1:dist.barrier()

    def save_model(name):
        model.init_cache()
        if rank==0:atomic_torch_save({'state_dict':model.state_dict(),'config':config,'step':step,
            'screen_fid':best_fid},args.output/name)
        if world>1:dist.barrier()

    def validate():
        nonlocal best_fid,best_step,bad_checks,pending_eval,initialized
        optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        directory=args.output/f'evaluations/step-{step:06d}';directory.mkdir(parents=True,exist_ok=True)
        if not (directory/'teacher-forcing.json').exists():
            values={s:evaluate(model,aux,splits[s],rank,world) for s in ('validation','train_probe')}
            if rank==0:write_json(directory/'teacher-forcing.json',values)
            log({'phase':'validation',**{f'{s}/{k}':v for s,ms in values.items() for k,v in ms.items()}})
        if world>1:dist.barrier()
        if config['fid_samples']:
            path=directory/'screen/metrics.json'
            if path.exists():
                result=json.loads(path.read_text());assert result['samples']==config['fid_samples'] and result['seed']==12701
            else:result=generate(model,aux,config,args,directory/'screen',config['fid_samples'],12701,step,rank,world,log)
            if best_fid is None or result['fid']<best_fid-.25:
                best_fid,best_step,bad_checks=result['fid'],step,0
                save_model('best-screen.pt')
            elif best_step != step:bad_checks+=1
            log({'phase':'screen_complete','screen/fid':result['fid'],'screen/samples':result['samples']})
            if wb:wb.log({'samples':wandb.Image(str(directory/'screen/samples.png')),'optimizer_step':step})
        epoch=step//config['steps_per_epoch']
        if config['confirmation_samples'] and (epoch==10 or epoch%50==0):
            path=directory/'full/metrics.json'
            if path.exists():
                full=json.loads(path.read_text());assert full['samples']==config['confirmation_samples'] and full['seed']==17701
            else:full=generate(model,aux,config,args,directory/'full',config['confirmation_samples'],17701,step,rank,world,log)
            log({'phase':'full_fid_complete','full/fid':full['fid'],'full/samples':full['samples']})
        initialized,pending_eval=True,None
        save_last()

    def pause():
        flag=torch.tensor(int(bool(stopping['signal']) or (args.stop_after_step and step>=args.stop_after_step)),device='cuda')
        if world>1:dist.all_reduce(flag,op=dist.ReduceOp.MAX)
        if not flag.item():return False
        save_last();log({'phase':'paused','signal':stopping['signal']})
        if wb:wb.summary['training_status']='paused';wb.finish()
        return True

    log({'phase':'ready','resumed':True,'parameters':config['parameters'],'execution':execution})
    try:
        if args.verification_skip_evaluation:pending_eval=None
        elif pending_eval is not None:validate()
        if pause():return
        while stream.epoch+stream.position/stream.size<config['epochs']:
            indices,_,epoch_end=stream.next(config['batch_size']);step+=1
            lr=early_decay_lr(step-1,config['maximum_steps'],config['steps_per_epoch'],peak=config['lr'])
            weight=scheduled_geometry_weight(.05,step/config['steps_per_epoch'],2.,3.)
            for group in optimizer.param_groups:group['lr']=lr*group['lr_multiplier']
            optimizer.zero_grad(set_to_none=True);model.train()
            torch.cuda.synchronize();iteration=time.monotonic()
            totals=backward_batch(wrapped,indices,splits['train'],weight,step,config['seed'],args.microbatch,rank,world)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
            optimizer.step();torch.cuda.synchronize()
            if step%4==0 or args.verification_skip_evaluation:
                log({'phase':'train',**{f'train/{k}':v for k,v in totals.items()},'train/lr':lr,
                    'train/geometry_weight':weight,'train/gradient_norm':float(norm),
                    'train/step_seconds':time.monotonic()-iteration,
                    'gpu/peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30})
            if epoch_end and ((stream.epoch+1)==1 or (stream.epoch+1)%config['eval_every']==0 or stream.epoch+1==config['epochs']):pending_eval=step
            if pause():return
            if pending_eval is not None and not args.verification_skip_evaluation:
                save_last();validate()
                if pause():return
            elif step%200==0 or epoch_end:save_last()
        save_last();save_model('final.pt')
        selected=torch.load(args.output/'best-screen.pt',map_location='cpu',weights_only=False,mmap=True)
        model.load_state_dict(selected['state_dict'],strict=True);selected_step=selected['step']
        del selected,optimizer,wrapped;gc.collect();torch.cuda.empty_cache()
        result={'reason':'maximum_epochs','selected_step':selected_step,'best_screen_fid':best_fid,
            'trained_from_scratch':True,'initialization':config['initialization'],'frozen_stage1_verified':True}
        if config['confirmation_samples']:result['generation']=generate(model,aux,config,args,args.output/'selected-fid',
            config['confirmation_samples'],27701,selected_step,rank,world,log)
        if rank==0:write_json(args.output/'results.json',result)
        log({'phase':'complete',**result})
        if wb:wb.summary['results']=result;wb.summary['training_status']='complete';wb.finish()
    except Exception as error:
        log({'phase':'failed','error':repr(error)})
        if wb:wb.finish(exit_code=1)
        raise
    finally:
        if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
