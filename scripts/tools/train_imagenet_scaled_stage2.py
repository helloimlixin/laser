#!/usr/bin/env python3
"""Precache ImageNet, then train the smallest released class-conditional RQ prior."""
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

ROOT = Path(os.environ.get('LASER_PROJECT_ROOT',Path(__file__).resolve().parents[2]))
UPSTREAM = ROOT/'outputs/church-rq-baseline-scratch-20260912/upstream-source'
sys.path[:0] = [str(Path(__file__).resolve().parents[2]),str(UPSTREAM)]
os.environ.setdefault('TORCH_HOME','/workspace/tmp/official-rqvae-eval-cache')

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset
from torchvision.utils import save_image
from rqvae.img_datasets.transforms import create_transforms
from rqvae.optimizer.scheduler import create_scheduler
from src.original_rq_training import (atomic_json,file_sha256,state_sha256,seed_all,
    fresh_transformer,FeatureMoments,fid_from_moments)
from src.scaled_atom_training import FrozenScaledTokenizer,soft_cross_entropy
from src.compact_rq_training import FrozenCompactTokenizer
from src.imagenet_scaled_stage2 import (ManifestImages,CachedClassLatents,
    load_imagenet_config,enable_sdpa,conditional_update)
from src.tokenizer_fidelity import require_tokenizer_fidelity
from src.imagenet_sample_grid import sample_requested_grid,REQUESTED_CLASSES,GRID_SEED
from src.scaled_atom_sampling import SAMPLER_SETTINGS,sample_codes


class RepeatedCache(Dataset):
    def __init__(self,dataset,length):
        self.dataset,self.length=dataset,length
    def __len__(self):
        return self.length
    def __getitem__(self,index):
        return self.dataset[index%len(self.dataset)]


@torch.inference_mode()
def build_cache(args,tokenizer,config,rank,world,device,status):
    cache=args.cache
    cache.mkdir(exist_ok=True,parents=True)
    source=args.preparation
    spec=dict(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=file_sha256(args.checkpoint),
        codebook=str(args.codebook.resolve()),codebook_sha256=file_sha256(args.codebook),
        train_views=args.views,world_size=world,dtype='float32',code_dtype='uint32',seed=421,
        train_manifest_sha256=file_sha256(source/'train-manifest.json'),
        val_manifest_sha256=file_sha256(source/'val-manifest.json'),
        data=str(args.data.resolve()),train_transform='released Resize256/RandomCrop256/RandomHorizontalFlip',
        augmentation_policy='finite cached image views; epoch selects (epoch+image_index)%views',
        soft_targets='full-vocabulary stochastic RQ regenerated every training visit',
        token_formula=f'0=zero; 1+atom_id*{args.levels}+coefficient_bin',
        vocab_size=tokenizer.quantizer.vocab_size,coefficient_levels=args.levels,
        smoke_only=bool(args.cache_smoke_items),cache_smoke_items=args.cache_smoke_items)
    marker=cache/'in-progress.json'
    if rank==0:
        if marker.exists():
            assert json.loads(marker.read_text())==spec,'Incompatible partial cache'
        else:
            assert not list(cache.glob('*.npy')),'Unverified cache arrays already exist'
            atomic_json(marker,spec)
    dist.barrier()
    if (cache/'complete.json').exists():
        completed=json.loads((cache/'complete.json').read_text())
        assert all(completed[k]==v for k,v in spec.items())
        status('verified_existing_cache',images=completed['train_images'])
        return completed
    before=state_sha256(tokenizer)
    reuse_spec=None
    if args.reuse_cache:
        reuse_spec=json.loads((args.reuse_cache/'in-progress.json').read_text())
        for name in ('checkpoint_sha256','train_manifest_sha256','val_manifest_sha256','seed','dtype','world_size','train_views'):
            assert reuse_spec[name]==spec[name],f'Incompatible encoder cache reuse: {name}'
    for split in ('val','train'):
        manifest=json.loads((source/f'{split}-manifest.json').read_text())
        if args.cache_smoke_items:
            manifest['samples']=manifest['samples'][:args.cache_smoke_items]
            manifest['images']=len(manifest['samples'])
        count=manifest['images']
        begin=count*rank//world
        end=count*(rank+1)//world
        if rank==0:
            labels=np.asarray([row[1] for row in manifest['samples']],dtype=np.int16)
            np.save(cache/f'{split}-labels.npy',labels)
        for view in range(args.views if split=='train' else 1):
            paths=[cache/f'{split}-view{view}-latents.npy',cache/f'{split}-view{view}-codes.npy']
            shapes=[(count,8,8,256),(count,8,8,4)]
            dtypes=[np.float32,np.uint32]
            if rank==0:
                for path,shape,dtype in zip(paths,shapes,dtypes):
                    if not path.exists():
                        array=np.lib.format.open_memmap(path,mode='w+',dtype=dtype,shape=shape)
                        del array
                    else:
                        array=np.load(path,mmap_mode='r')
                        assert array.shape==shape and array.dtype==dtype
            dist.barrier()
            progress=cache/f'{split}-view{view}-rank{rank}.json'
            done=json.loads(progress.read_text())['done'] if progress.exists() else 0
            assert 0<=done<=end-begin
            arrays=[np.load(path,mmap_mode='r+') for path in paths]
            if reuse_spec is not None:
                old_progress=args.reuse_cache/f'{split}-view{view}-rank{rank}.json'
                old_latents=args.reuse_cache/f'{split}-view{view}-latents.npy'
                reusable=json.loads(old_progress.read_text())['done'] if old_progress.exists() else 0
                if reusable>done:
                    old_values=np.load(old_latents,mmap_mode='r')
                    assert old_values.shape==shapes[0] and old_values.dtype==np.float32
                    assert reusable<=end-begin
                    while done<reusable:
                        stop=min(done+args.cache_batch_size,reusable)
                        z=torch.from_numpy(old_values[begin+done:begin+stop].copy()).to(device)
                        assert torch.isfinite(z).all()
                        hard=tokenizer.quantizer.quantize(z)
                        arrays[0][begin+done:begin+stop]=z.cpu().numpy()
                        arrays[1][begin+done:begin+stop]=hard['codes'].cpu().numpy().astype(np.uint32)
                        done=stop
                        if done%(args.cache_batch_size*50)==0 or done==reusable:
                            for array in arrays:array.flush()
                            atomic_json(progress,dict(done=done,total=end-begin,encoder_latents_reused=True,updated_unix=time.time()))
                            status('reusing_encoder_cache',split=split,view=view+1,
                                images_completed_estimate=min(count,done*world),images_total=count)
                    del old_values
            transform=create_transforms(config.dataset,split=split)
            images=ManifestImages(args.data/split,manifest,transform,view=view,seed=421,
                                  indices=range(begin+done,end))
            loader=DataLoader(images,batch_size=args.cache_batch_size,num_workers=args.workers,
                pin_memory=True,persistent_workers=args.workers>0,shuffle=False)
            for batch,(xs,labels,indices) in enumerate(loader):
                z=tokenizer.encode(xs.to(device,non_blocking=True))
                hard=tokenizer.quantizer.quantize(z)
                assert z.dtype==torch.float32 and torch.isfinite(z).all()
                assert hard['codes'].min()>=0 and hard['codes'].max()<tokenizer.quantizer.vocab_size
                arrays[0][indices.numpy()]=z.cpu().numpy()
                arrays[1][indices.numpy()]=hard['codes'].cpu().numpy().astype(np.uint32)
                done+=len(xs)
                if batch==0:
                    # Direct codec parity and class-label alignment before accepting a cache.
                    torch.testing.assert_close(tokenizer.quantizer.embed(hard['codes']).sum(-2),hard['quantized'])
                    assert all(manifest['samples'][int(i)][1]==int(y) for i,y in zip(indices,labels))
                    np.testing.assert_array_equal(arrays[0][indices.numpy()],z.cpu().numpy())
                    np.testing.assert_array_equal(arrays[1][indices.numpy()],hard['codes'].cpu().numpy())
                if batch%50==0 or done==end-begin:
                    for array in arrays:
                        array.flush()
                    atomic_json(progress,dict(done=done,total=end-begin,updated_unix=time.time()))
                    status('caching',split=split,view=view+1,views=args.views if split=='train' else 1,
                        images_completed_estimate=min(count,done*world),images_total=count)
            assert done==end-begin
            for array in arrays:
                array.flush()
            del arrays,loader,images
            dist.barrier()
        if split=='val' and not args.cache_smoke_items and not (cache/'tokenizer-rfid.json').exists():
            from rqvae.metrics.fid import get_inception_model
            inception=get_inception_model().eval().requires_grad_(False).to(device)
            moments=FeatureMoments(device)
            codes=np.load(cache/'val-view0-codes.npy',mmap_mode='r')
            for start in range(begin,end,32):
                ids=torch.from_numpy(codes[start:min(start+32,end)].astype(np.int64)).to(device)
                decoded=torch.cat([tokenizer.decode_code(chunk) for chunk in ids.split(8)])
                moments.update(inception(decoded.mul(.5).add(.5).clamp(0,1)))
                if (start-begin)%1024==0:
                    status('tokenizer_reconstruction_audit',images_completed_estimate=min(count,(start-begin+len(ids))*world),images_total=count)
            result=moments.finish()
            assert result[0]==50000
            if rank==0:
                quality=json.loads(args.fidelity_report.read_text())
                reference=Path(quality['reference_statistics'])
                rfid=fid_from_moments(result,reference)
                audit=dict(images=50000,original_rfid=quality['original_rfid'],converted_rfid=rfid,
                    reference_statistics=str(reference),same_validation_images=True,same_reference_statistics=True,
                    checkpoint_sha256=spec['checkpoint_sha256'],codebook_sha256=spec['codebook_sha256'])
                atomic_json(cache/'tokenizer-rfid.json',audit)
                require_tokenizer_fidelity(cache/'tokenizer-rfid.json',spec['checkpoint_sha256'],
                    spec['codebook_sha256'],args.max_rfid_drift)
                status('tokenizer_reconstruction_audit_complete',expanded_tokenizer_rfid=rfid,images=50000)
            del inception,moments,codes
            torch.cuda.empty_cache()
            dist.barrier()
    assert before==state_sha256(tokenizer),'Frozen tokenizer changed during caching'
    if rank==0:
        completed=dict(**spec,train_images=min(args.cache_smoke_items or 1281167,1281167),
                       val_images=min(args.cache_smoke_items or 50000,50000),
                       frozen_state_sha256=before,complete=True,updated_unix=time.time())
        atomic_json(cache/'complete.json',completed)
        status('cache_complete',images=completed['train_images'],train_views=args.views)
    dist.barrier()
    return json.loads((cache/'complete.json').read_text())


@torch.inference_mode()
def validation(model,tokenizer,dataset,temperature,device,rank,world,batch_size):
    model.eval()
    totals=torch.zeros(7,device=device,dtype=torch.float64)
    loader=DataLoader(Subset(dataset,range(rank,len(dataset),world)),batch_size=batch_size,
                      num_workers=2,pin_memory=True)
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(61000+rank)
        for z,labels,hard in loader:
            z,labels,hard=z.to(device),labels.to(device),hard.to(device)
            targets,codes=tokenizer.quantizer.get_soft_codes(z,temp=temperature,stochastic=True)
            logits=model(codes,model_aux=tokenizer,cond=labels,amp=True)
            totals[0]+=soft_cross_entropy(logits,targets)*len(z)
            del targets,logits
            logits=model(hard,model_aux=tokenizer,cond=labels,amp=True)
            # Hard cross entropy also chunks rows to avoid a full FP32 logit copy.
            for d in range(4):
                xs=logits[...,d,:].reshape(-1,tokenizer.quantizer.vocab_size)
                ys=hard[...,d].reshape(-1)
                value=torch.zeros((),device=device)
                for start in range(0,len(xs),128):
                    value+=torch.nn.functional.cross_entropy(xs[start:start+128].float(),ys[start:start+128],reduction='sum')
                totals[2+d]+=value/64
            totals[6]+=len(z)
    dist.all_reduce(totals)
    assert totals[6]==len(dataset) and torch.isfinite(totals).all()
    totals[:6]/=totals[6]
    model.train()
    return dict(soft_ce=float(totals[0]),hard_code_nll=float(totals[2:6].mean()),
        **{f'depth_{d}_nll':float(totals[2+d]) for d in range(4)},images=len(dataset))


@torch.inference_mode()
def generation(model,tokenizer,inception,count,epoch,out,reference,device,rank,world,status,
               sampler_name='original'):
    model.eval()
    settings=SAMPLER_SETTINGS[sampler_name]
    moments=FeatureMoments(device)
    indices=list(range(rank,count,world))
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(73000+rank)
        for start in range(0,len(indices),32):
            selected=indices[start:start+32]
            labels=torch.tensor([i%1000 for i in selected],device=device)
            if settings['mode']=='joint':
                codes=model.sample(torch.zeros(len(labels),8,8,4,device=device,dtype=torch.long),
                    model_aux=tokenizer,cond=labels,temperature=settings['temperature'],
                    top_k=settings['top_k'],top_p=settings['top_p'],amp=True)
            else:
                codes=sample_codes(model,tokenizer,labels,settings)
            images=[]
            for chunk in codes.split(8):
                images.append(tokenizer.decode_code(chunk).mul(.5).add(.5).clamp(0,1))
            images=torch.cat(images)
            assert torch.isfinite(images).all()
            moments.update(inception(images))
            if start%256==0:
                status('generating',epoch=epoch,samples=count,sampler_name=sampler_name,
                    completed=min(count,(start+len(selected))*world))
    result=moments.finish()
    assert result[0]==count
    score=None
    if rank==0:
        score=fid_from_moments(result,reference)
        suffix='' if sampler_name=='original' else f'-{sampler_name}'
        atomic_json(out/f'fid-epoch{epoch:03d}-{count}{suffix}.json',dict(epoch=epoch,samples=count,fid=score,
            reference=str(reference),sampler_name=sampler_name,**settings,
            class_labels='sample_index modulo 1000',seed=73000))
    dist.barrier()
    model.train()
    return score


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--preparation',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--codebook',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path('/workspace/Projects/data/imagenet2012'))
    p.add_argument('--run-id',default='imagenet-rfid421-scaled-rq8-480m-20260913')
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--cache-batch-size',type=int,default=64)
    p.add_argument('--views',type=int,default=2)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--max-updates',type=int,default=0)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--offline',action='store_true')
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--cache-only',action='store_true')
    p.add_argument('--cache-smoke-items',type=int,default=0)
    p.add_argument('--levels',type=int,default=8)
    p.add_argument('--compact',action='store_true',help='Use frozen atom-specific coefficient levels')
    p.add_argument('--fidelity-report',type=Path)
    p.add_argument('--max-rfid-drift',type=float,default=.1)
    p.add_argument('--reuse-cache',type=Path)
    p.add_argument('--sample-grid-on-start',action='store_true')
    p.add_argument('--sampler',choices=list(SAMPLER_SETTINGS),default='original')
    p.add_argument('--online-images',action='store_true',
        help='Draw fresh released ImageNet augmentations each epoch and encode them during training')
    args=p.parse_args()
    assert args.views>=1 and args.batch_size>0
    assert not args.compact or args.levels in (2,4)
    assert args.cache_smoke_items>=0
    assert not args.cache_smoke_items or (args.cache_only and args.offline)
    rank,world=int(os.environ['RANK']),int(os.environ['WORLD_SIZE'])
    device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=True
    dist.init_process_group('nccl',timeout=timedelta(hours=3))
    args.output=args.output.resolve()
    out=args.output
    if rank==0:
        out.mkdir(parents=True,exist_ok=args.resume or args.preflight)
    dist.barrier()
    calibration=json.loads((args.preparation/'temperature-calibration.json').read_text())
    assert file_sha256(args.checkpoint)==calibration['source_checkpoint_sha256']
    assert file_sha256(args.codebook)==calibration['codebook_sha256']
    quality=None
    if not args.preflight and not args.cache_smoke_items:
        if args.fidelity_report is None:
            raise ValueError('Production requires a matched 50k tokenizer fidelity report')
        quality=require_tokenizer_fidelity(args.fidelity_report,calibration['source_checkpoint_sha256'],
            calibration['codebook_sha256'],args.max_rfid_drift)
    temperature=calibration['selected_temperature']
    vocab_size=1+16384*args.levels
    config=load_imagenet_config(UPSTREAM,vocab_size=vocab_size)
    config.experiment.batch_size=args.batch_size
    config.loss.temp=temperature
    config.vqvae=dict(ckpt=str(args.checkpoint.resolve()),codebook=str(args.codebook.resolve()))
    config.cache=dict(path=str(args.cache.resolve()),views=args.views,stochastic_targets_each_visit=True)
    config.cache.training_latents_reused=not args.online_images
    config.training_data=dict(mode='online-images' if args.online_images else 'cached-latents',
        augmentation=('released Resize256/RandomCrop256/RandomHorizontalFlip; fresh per image and epoch'
            if args.online_images else 'two finite cached views'),
        reproducibility='image index and epoch determine augmentation seed',encoder_batch_size=16)
    config.target_temperature_policy=dict(original_rq_temperature=.5,
        sparse_tokenizer_temperature=temperature,calibration=str(args.preparation/'temperature-calibration.json'))
    config.training_precision='FP16 model autocast; FP32 tokenizer geometry and full soft CE; SDPA attention'
    config.pipeline=f'imagenet-class-conditional-frozen-scaled-atom-rq{args.levels}'
    compact_spec=torch.load(args.codebook,map_location='cpu',weights_only=True) if args.compact else None
    depth_specific=bool(compact_spec and compact_spec['kind']=='depth_adaptive_scaled_atom_rq')
    config.coefficient_construction=('depth-and-atom-specific' if depth_specific else
        'atom-specific' if args.compact else 'shared')
    config.coefficient_tables=4 if depth_specific else 1
    del compact_spec
    if args.compact:
        config.pipeline=f'imagenet-class-conditional-frozen-compact-adaptive-rq{args.levels}'
    config.tokenizer_fidelity=quality
    config.stage2_from_scratch=True
    config.stage1_frozen=True
    config.source_rfid=calibration['source_rfid']
    config.source_run='helloimlixin-rutgers/laser/imga16384k4altbn64-b128-b300-20260830000755'
    config.reference_run='helloimlixin-rutgers/laser/church-scaled-rq8-scratch-20260913'
    if args.compact:
        config.reference_run='helloimlixin-rutgers/laser/church-compact-rq32k-scratch-20260913'
    config.upstream_commit='341395e562ac347f5eb62db9f5f08b9f2cc42a60'
    config.recipe=('released ImageNet 480M architecture/optimizer/schedule; compact tokenizer and calibrated target temperature; '
        + ('fresh image augmentations' if args.online_images else 'finite cached augmentations'))
    config.sample_grid=dict(classes=[dict(id=c,name=n) for c,n in REQUESTED_CLASSES],
        samples_per_class=8,seed=GRID_SEED,layout='10 labeled rows x 8 samples')
    config.generation_sampler=dict(name=args.sampler,**SAMPLER_SETTINGS[args.sampler])
    config.generation_fid_policy='original sampler retained; selected sampler logged separately'
    if args.sampler=='published_imagenet_480m':
        config.sampling=dict(temp=1.,top_k=256,top_p=.95)
        config.generation_fid_policy='published 480M checkpoint sampler; generation/published_fid_*; earlier sampler histories retained'
    run=None
    if rank==0:
        OmegaConf.save(config,out/'config.yaml')
        atomic_json(out/'temperature-calibration.json',calibration)
        if not args.offline:
            for name in ('WANDB_SERVICE','_WANDB_SERVICE'):
                os.environ.pop(name,None)
            import wandb
            run=wandb.init(entity='helloimlixin-rutgers',project='laser',id=args.run_id,name=args.run_id,
                resume='allow' if args.resume else 'never',dir=str(out),config=OmegaConf.to_container(config,resolve=True))
            if args.resume:
                run.config.update(OmegaConf.to_container(config,resolve=True),allow_val_change=True)
            atomic_json(out/'wandb.json',dict(run_id=run.id,url=run.url))
            run.summary['training_status']='prebuilding_cache'
    started=time.time()
    def status(phase,**values):
        if rank==0:
            record=dict(phase=phase,updated_unix=time.time(),elapsed_seconds=time.time()-started,pid=os.getpid(),**values)
            atomic_json(out/'status.json',record)
            with (out/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
            if run:
                run.summary['training_status']=phase
                run.log({f'{"stage2" if phase=="training" else phase}/{k}':v for k,v in values.items() if isinstance(v,(int,float))})
    tokenizer=(FrozenCompactTokenizer(args.checkpoint,args.codebook) if args.compact else
        FrozenScaledTokenizer(args.checkpoint,args.codebook,levels=args.levels)).to(device).eval()
    assert tokenizer.quantizer.vocab_size==vocab_size
    frozen_hash=state_sha256(tokenizer)
    if args.preflight:
        cache=json.loads((args.cache/'complete.json').read_text())
        assert cache['smoke_only']
    else:
        cache=build_cache(args,tokenizer,config,rank,world,device,status)
        assert cache['frozen_state_sha256']==frozen_hash
    atomic_json(out/f'cache-provenance-rank{rank}.json',cache)
    if args.cache_only:
        status('cache_only_complete',images=cache['train_images'])
        if run:run.finish()
        dist.destroy_process_group()
        return
    seed_all(0)
    model,empty_optimizer=fresh_transformer(config)
    assert not empty_optimizer.state
    del empty_optimizer
    initial_hash=state_sha256(model)
    initial_hashes=[None]*world
    dist.all_gather_object(initial_hashes,initial_hash)
    assert len(set(initial_hashes))==1
    parameters=sum(x.numel() for x in model.parameters())
    assert parameters>480000000 and model.vocab_size_cond==1000
    count=enable_sdpa(model)
    assert count==16
    model.to(device)
    # Same released all-parameter AdamW grouping, using the fused CUDA kernel.
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.optimizer.init_lr,
        betas=tuple(config.optimizer.betas),weight_decay=config.optimizer.weight_decay,fused=True)
    ddp=DistributedDataParallel(model,device_ids=[device.index],broadcast_buffers=False,
                                gradient_as_bucket_view=True,bucket_cap_mb=100)
    accumulation=2048//(world*args.batch_size)
    assert accumulation*world*args.batch_size==2048
    train_images=2048*max(args.max_updates,3) if args.preflight else len(CachedClassLatents(args.cache,include_hard_codes=False))
    per_rank=math.ceil(train_images/world)
    batches_per_epoch=math.ceil(per_rank/args.batch_size)
    steps_per_epoch=math.ceil(batches_per_epoch/accumulation)
    scheduler=create_scheduler(optimizer,config.optimizer.warmup,steps_per_epoch,100)
    scaler=torch.amp.GradScaler('cuda',init_scale=65536.)
    step=attempts=skipped=consecutive_skips=start_epoch=start_batch=0
    resume_rng=None
    resumed_from_step=None
    if args.resume and (out/'last.pt').exists():
        saved=torch.load(out/'last.pt',map_location='cpu',weights_only=False,mmap=True)
        assert saved['cache']['checkpoint_sha256']==cache['checkpoint_sha256']
        assert saved['cache']['codebook_sha256']==cache['codebook_sha256']
        assert saved['world_size']==world and saved['batch_size']==args.batch_size
        model.load_state_dict(saved['state_dict'],strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler'])
        start_epoch,start_batch=saved['epoch'],saved['batch_in_epoch']
        step,attempts,skipped=saved['step'],saved['attempts'],saved['skipped_amp_updates']
        initial_hash=saved['initial_weights_sha256']
        resume_rng=saved['rng_states'][rank]
        resumed_from_step=step
        del saved
    initialization=dict(parameters=parameters,stage2_from_scratch=True,pretrained_stage2_checkpoint=None,
        initial_weights_sha256=initial_hash,initial_weights_identical_across_ranks=True,
        optimizer_initially_empty=resumed_from_step is None,resumed_from_step=resumed_from_step,
        tokenizer_frozen=True,tokenizer_sha256=cache['checkpoint_sha256'],
        codebook_sha256=cache['codebook_sha256'],microbatch_per_gpu=args.batch_size,world_size=world,
        gradient_accumulation=accumulation,effective_batch_size=2048,steps_per_epoch=steps_per_epoch,
        architecture='ImageNet 480M: width1536 spatial12 depth4 heads24',vocabulary=vocab_size,
        coefficient_levels=args.levels,
        coefficient_construction=config.coefficient_construction,
        coefficient_tables=config.coefficient_tables,
        class_vocabulary=1000,target_temperature=temperature,sdpa_attention_layers=count,
        training_data_mode=config.training_data.mode,generation_sampler=config.generation_sampler.name)
    if rank==0:
        atomic_json(out/'initialization.json',initialization)
        if run:
            run.summary.update(initialization)
    status('initialization_complete',**initialization)
    stop_requested=False
    def on_signal(signum,frame):
        nonlocal stop_requested
        stop_requested=True
    signal.signal(signal.SIGTERM,on_signal)
    signal.signal(signal.SIGINT,on_signal)
    def save(epoch,batch):
        states=[None]*world
        dist.all_gather_object(states,dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device),
            numpy=np.random.get_state(),python=random.getstate()))
        if rank==0:
            path=out/'last.pt'
            torch.save(dict(epoch=epoch,batch_in_epoch=batch,step=step,attempts=attempts,
                skipped_amp_updates=skipped,state_dict=model.state_dict(),optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng_states=states,
                cache=cache,config=OmegaConf.to_container(config,resolve=True),world_size=world,
                batch_size=args.batch_size,initial_weights_sha256=initial_hash),path.with_suffix('.tmp'))
            path.with_suffix('.tmp').replace(path)
        dist.barrier()
    heldout=CachedClassLatents(args.cache,'val')
    inception=None
    if not args.preflight:
        from rqvae.metrics.fid import get_inception_model
        inception=get_inception_model().eval().requires_grad_(False).to(device)
    reference=ROOT/'third_party/rq-vae-transformer/assets/fid_stats/imagenet_256_train.npz'
    assert reference.is_file()
    def preview(epoch):
        status('sampling_class_grid',epoch=epoch,optimizer_step=step,classes=10,samples_per_class=8,
            sampler_name=args.sampler)
        target=sample_requested_grid(model,tokenizer,out,step,epoch,device=device,rank=rank,world=world,
            sampler_name=args.sampler)
        if rank==0 and run:
            import wandb
            picture=wandb.Image(str(target),caption=f'10 fixed ImageNet classes, 8 samples each; step {step}; sampler {args.sampler}')
            run.log({'generation/class_conditional_samples':picture,'samples/requested_classes_10x8':picture,
                'generation/epoch':epoch,'generation/optimizer_step':step})
            run.summary['last_class_grid']=str(target)
            run.summary['last_class_grid_optimizer_step']=step
            run.summary['generation_sampler']=dict(name=args.sampler,**SAMPLER_SETTINGS[args.sampler])
        dist.barrier()
    if args.sample_grid_on_start and not args.preflight:
        preview(start_epoch+start_batch/(steps_per_epoch*accumulation))
    seed_all(rank)
    torch.cuda.reset_peak_memory_stats(device)
    train_manifest=json.loads((args.preparation/'train-manifest.json').read_text()) if args.online_images else None
    for epoch_index in range(start_epoch,100):
        if args.online_images:
            dataset=ManifestImages(args.data/'train',train_manifest,
                create_transforms(config.dataset,split='train'),view=epoch_index,seed=421,
                indices=calibration['fit_indices'][:32] if args.preflight else None)
        else:
            dataset=CachedClassLatents(args.cache,epoch=epoch_index,include_hard_codes=False)
        if args.preflight:
            dataset=RepeatedCache(dataset,train_images)
        sampler=DistributedSampler(dataset,num_replicas=world,rank=rank,shuffle=True,seed=0)
        sampler.set_epoch(epoch_index)
        loader=DataLoader(dataset,sampler=sampler,batch_size=args.batch_size,num_workers=args.workers,
                          pin_memory=True,persistent_workers=args.workers>0,drop_last=False)
        iterator=iter(loader)
        consumed=start_batch if epoch_index==start_epoch else 0
        for _ in range(consumed):
            next(iterator)
        if resume_rng is not None:
            torch.set_rng_state(resume_rng['torch']);torch.cuda.set_rng_state(resume_rng['cuda'],device)
            np.random.set_state(resume_rng['numpy']);random.setstate(resume_rng['python'])
            resume_rng=None
        model.train()
        for update_index in range(consumed//accumulation,steps_per_epoch):
            batches=list(islice(iterator,accumulation))
            assert batches
            consumed+=len(batches)
            update_start=time.time()
            metrics=conditional_update(ddp,tokenizer,optimizer,scaler,batches,temperature,max_gn=1.,
                inputs_are_images=args.online_images)
            attempts+=1
            if metrics['optimizer_updated']:
                scheduler.step();step+=1;consecutive_skips=0
            else:
                skipped+=1;consecutive_skips+=1
            epoch_fraction=epoch_index+(update_index+1)/steps_per_epoch
            status('training',optimizer_step=step,attempted_updates=attempts,epoch=epoch_fraction,
                lr=optimizer.param_groups[0]['lr'],skipped_amp_updates=skipped,consecutive_amp_skips=consecutive_skips,
                update_seconds=time.time()-update_start,images_per_second=metrics['global_images']/(time.time()-update_start),
                peak_gpu_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3,**metrics)
            if consecutive_skips>=8:
                save(epoch_index,consumed)
                raise FloatingPointError('Eight consecutive AMP skips; checkpoint saved')
            stop=torch.tensor(int(stop_requested),device=device)
            dist.all_reduce(stop,op=dist.ReduceOp.MAX)
            finished=args.max_updates and step>=args.max_updates
            if stop.item() or finished:
                save(epoch_index,consumed)
                if args.preflight and finished:
                    assert before_or_equal(tokenizer,frozen_hash)
                    saved=torch.load(out/'last.pt',map_location='cpu',weights_only=False,mmap=True)
                    model.load_state_dict(saved['state_dict'],strict=True)
                    finite=all(torch.isfinite(x).all() for x in saved['state_dict'].values() if x.is_floating_point())
                    finite=finite and all(torch.isfinite(v).all() for x in saved['optimizer']['state'].values()
                        for v in x.values() if torch.is_tensor(v) and v.is_floating_point())
                    assert finite and len(saved['rng_states'])==world
                    del saved
                    model.eval()
                    with torch.inference_mode():
                        z,label,_=heldout[0]
                        targets,codes=tokenizer.quantizer.get_soft_codes(z[None].to(device),temp=temperature,stochastic=False)
                        logits=model(codes,model_aux=tokenizer,cond=torch.tensor([label],device=device),amp=True)
                        dense=model.compute_loss(logits.float(),targets,use_soft_target=True)
                        chunked=soft_cross_entropy(logits,targets)
                        torch.testing.assert_close(dense,chunked,atol=2e-5,rtol=2e-5)
                        other=model(codes,model_aux=tokenizer,cond=torch.tensor([(label+1)%1000],device=device),amp=True)
                        assert not torch.allclose(logits[:,0,0,0],other[:,0,0,0])
                        del targets,logits,other
                        labels=torch.tensor([0,207,281,979],device=device)
                        settings=SAMPLER_SETTINGS[args.sampler]
                        if settings['mode']=='joint':
                            sampled=model.sample(torch.zeros(4,8,8,4,device=device,dtype=torch.long),model_aux=tokenizer,
                                cond=labels,temperature=settings['temperature'],top_k=settings['top_k'],
                                top_p=settings['top_p'],amp=True)
                        else:
                            sampled=sample_codes(model,tokenizer,labels,settings)
                        decoded=tokenizer.decode_code(sampled)
                        assert sampled.min()>=0 and sampled.max()<vocab_size and torch.isfinite(decoded).all()
                        if rank==0:
                            save_image(decoded.mul(.5).add(.5).clamp(0,1),out/'decode-smoke.png',nrow=4)
                    if rank==0:
                        sources=[Path(__file__).resolve(),
                            *[ROOT/'src'/name for name in ('imagenet_scaled_stage2.py','scaled_atom_training.py',
                                'scaled_atom_rq.py','original_rq_training.py','tokenizer_fidelity.py',
                                'imagenet_sample_grid.py','scaled_atom_sampling.py',
                                'compact_rq_training.py','adaptive_scaled_atom_rq.py')],
                            *sorted(UPSTREAM.rglob('*.py')),
                            UPSTREAM/'configs/imagenet256/stage2/in256-rqtransformer-8x8x4-480M.yaml']
                        atomic_json(out/'verification.json',dict(preflight_passed=True,strict_checkpoint_reload=True,
                            saved_tensors_finite=bool(finite),class_conditioning_verified=True,
                            released_sampler_decode_passed=True,dense_ce=float(dense),chunked_ce=float(chunked),
                            source_hashes={str(path.relative_to(ROOT)):file_sha256(path) for path in sources},
                            optimizer_updates=step,**initialization))
                status('preflight_complete' if args.preflight else 'paused',optimizer_step=step)
                if run:run.finish()
                dist.destroy_process_group()
                return
            if step==25 or (step and step%100==0 and metrics['optimizer_updated']):
                save(epoch_index,consumed)
        assert consumed==len(loader)
        del iterator,loader,dataset
        epoch=epoch_index+1
        save(epoch,0)
        if epoch%2==0 and rank==0:
            torch.save(dict(epoch=epoch,step=step,state_dict=model.state_dict(),cache=cache,
                config=OmegaConf.to_container(config,resolve=True)),out/f'epoch{epoch:03d}_model.pt')
        dist.barrier()
        if epoch==1 or epoch%2==0:
            status('validation',epoch=epoch)
            metrics=validation(model,tokenizer,heldout,temperature,device,rank,world,16)
            if rank==0:
                atomic_json(out/f'validation-epoch{epoch:03d}.json',dict(epoch=epoch,**metrics))
                if run:run.log({'validation/epoch':epoch,**{f'validation/{k}':v for k,v in metrics.items()}})
            preview(epoch)
            for samples in ([4096,50000] if epoch%10==0 else [4096]):
                if args.sampler=='published_imagenet_480m':
                    score=generation(model,tokenizer,inception,samples,epoch,out,reference,device,rank,world,status,
                        sampler_name=args.sampler)
                    if rank==0 and run:
                        run.log({f'generation/published_fid_{samples}':score,'generation/epoch':epoch,
                            'generation/sampler_name':args.sampler})
                        run.summary[f'last_published_fid_{samples}']=score
                    continue
                score=generation(model,tokenizer,inception,samples,epoch,out,reference,device,rank,world,status)
                if rank==0 and run:
                    run.log({f'generation/fid_{samples}':score,'generation/epoch':epoch})
                    run.summary[f'last_fid_{samples}']=score
                if args.sampler!='original':
                    score=generation(model,tokenizer,inception,samples,epoch,out,reference,device,rank,world,status,
                        sampler_name=args.sampler)
                    if rank==0 and run:
                        run.log({f'generation/sampler_fid_{samples}':score,'generation/epoch':epoch,
                            'generation/sampler_name':args.sampler})
                        run.summary[f'last_sampler_fid_{samples}']=score
            assert before_or_equal(tokenizer,frozen_hash)
            save(epoch,0)
    status('complete',optimizer_step=step,epochs=100)
    if run:run.finish()
    dist.destroy_process_group()


def before_or_equal(tokenizer,expected):
    return state_sha256(tokenizer)==expected


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        if '--output' in sys.argv:
            output=Path(sys.argv[sys.argv.index('--output')+1])
            if output.exists():
                atomic_json(output/f'failure-rank{os.environ.get("RANK","unknown")}.json',
                    dict(error_type=type(error).__name__,error=str(error),updated_unix=time.time()))
        raise
