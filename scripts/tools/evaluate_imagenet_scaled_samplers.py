#!/usr/bin/env python3
"""Compare expanded-vocabulary samplers on one immutable ImageNet checkpoint."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
UPSTREAM=ROOT/'outputs/church-rq-baseline-scratch-20260912/upstream-source'
sys.path[:0]=[str(ROOT),str(UPSTREAM)]
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from rqvae.models import create_model
from rqvae.metrics.fid import get_inception_model
from src.original_rq_training import atomic_json,file_sha256,FeatureMoments,fid_from_moments
from src.scaled_atom_training import FrozenScaledTokenizer
from src.imagenet_scaled_stage2 import enable_sdpa
from src.imagenet_sample_grid import REQUESTED_CLASSES,GRID_SEED,save_class_labeled_grid
from src.scaled_atom_sampling import SAMPLER_SETTINGS,sample_codes


@torch.inference_mode()
def diagnose(model,tokenizer,base,device,rank,world):
    indices=np.linspace(0,49999,32,dtype=np.int64)[rank::world]
    cache=base/'cache'
    codes=torch.from_numpy(np.load(cache/'val-view0-codes.npy',mmap_mode='r')[indices].astype(np.int64)).to(device)
    labels=torch.from_numpy(np.load(cache/'val-labels.npy',mmap_mode='r')[indices].astype(np.int64)).to(device)
    logits=model(codes,model_aux=tokenizer,cond=labels,amp=True)
    statistics=torch.zeros(4,6,dtype=torch.float64,device=device)
    for depth in range(4):
        rows=logits[...,depth,:].reshape(-1,tokenizer.quantizer.vocab_size)
        for chunk in rows.split(32):
            probabilities=chunk.float().softmax(-1)
            sorted_p=probabilities.sort(-1,descending=True).values
            cumulative=sorted_p.cumsum(-1)
            mass=cumulative[:,16383]
            nucleus=(cumulative<.92).sum(-1)+1
            baseline_count=(cumulative[:,:16384]<mass[:,None]*.92).sum(-1)+1
            kept=cumulative.gather(1,(baseline_count-1)[:,None]).squeeze(1)
            statistics[depth]+=torch.stack([mass.sum(),nucleus.double().sum(),
                (nucleus>16384).double().sum(),kept.sum(),baseline_count.double().sum(),
                mass.new_tensor(len(chunk))]).double()
    dist.all_reduce(statistics)
    return [dict(depth=d,positions=int(x[5]),mass_in_top16384=float(x[0]/x[5]),
        full_p92_mean_candidate_count=float(x[1]/x[5]),fraction_p92_exceeds16384=float(x[2]/x[5]),
        baseline_retained_joint_mass=float(x[3]/x[5]),baseline_mean_candidates=float(x[4]/x[5]))
        for d,x in enumerate(statistics)]


@torch.inference_mode()
def grid(model,tokenizer,setting,path,step,epoch,device,rank,world):
    labels=torch.tensor([c for c,_ in REQUESTED_CLASSES],device=device).repeat_interleave(8)
    torch.manual_seed(GRID_SEED+rank)
    local_codes=sample_codes(model,tokenizer,labels[rank::world],setting)
    images=torch.cat([tokenizer.decode_code(x).mul(.5).add(.5).clamp(0,1) for x in local_codes.split(8)])
    images_all=[torch.empty_like(images) for _ in range(world)]
    codes_all=[torch.empty_like(local_codes) for _ in range(world)]
    dist.all_gather(images_all,images);dist.all_gather(codes_all,local_codes)
    if rank==0:
        output=torch.empty(80,3,256,256)
        codes=torch.empty(80,8,8,4,dtype=torch.long)
        for r in range(world):output[r::world]=images_all[r].cpu();codes[r::world]=codes_all[r].cpu()
        names=[f'class {i}' for i in range(1000)]
        for c,n in REQUESTED_CLASSES:names[c]=n
        save_class_labeled_grid(output,torch.tensor([c for c,_ in REQUESTED_CLASSES]),names,path,samples_per_class=8)
        torch.save(codes,path.with_suffix('.codes.pt'))
        atomic_json(path.with_suffix('.json'),dict(optimizer_step=step,epoch=epoch,settings=setting,
            classes=[dict(id=c,name=n) for c,n in REQUESTED_CLASSES],labels=labels.cpu().tolist(),
            seed=GRID_SEED,rank_seed_rule='seed + rank',world_size=world,samples_per_class=8))
    dist.barrier()


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=1024)
    p.add_argument('--settings',nargs='+',default=list(SAMPLER_SETTINGS))
    p.add_argument('--run-id')
    p.add_argument('--skip-diagnostics',action='store_true')
    args=p.parse_args()
    rank,world=int(os.environ['RANK']),int(os.environ['WORLD_SIZE'])
    device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=True
    dist.init_process_group('nccl',timeout=timedelta(hours=2))
    out=args.output.resolve();base=args.base.resolve()
    if rank==0:out.mkdir(parents=True,exist_ok=False)
    dist.barrier();started=time.time()
    run=None
    def status(phase,**values):
        if rank==0:
            record=dict(phase=phase,elapsed_seconds=time.time()-started,updated_unix=time.time(),**values)
            atomic_json(out/'status.json',record);print(json.dumps(record),flush=True)
            if run:run.summary.update(record)
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    config=OmegaConf.create(payload['config'])
    assert payload['cache']['codebook_sha256']==file_sha256(base/'scaled-atom-codebook.pt')
    model,_=create_model(config.arch,ema=False)
    model.load_state_dict(payload['state_dict'],strict=True)
    step,epoch=payload['step'],payload['epoch']
    del payload
    enable_sdpa(model);model.requires_grad_(False).to(device).eval()
    tokenizer=FrozenScaledTokenizer(base/'assets/best_rfid_slot1_model.pt',base/'scaled-atom-codebook.pt').to(device).eval()
    inception=get_inception_model().requires_grad_(False).to(device).eval()
    reference=ROOT/'third_party/rq-vae-transformer/assets/fid_stats/imagenet_256_train.npz'
    spec=None
    if rank==0:
        spec=dict(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=file_sha256(args.checkpoint),
            codebook_sha256=file_sha256(base/'scaled-atom-codebook.pt'),optimizer_step=step,epoch=epoch,
            settings={name:SAMPLER_SETTINGS[name] for name in args.settings},samples=args.samples,
            reference=str(reference),seed=73000,world_size=world,sampling_batch_per_gpu=32,
            class_labels='sample_index modulo 1000',training_run='imagenet-rfid421-rq8-refit-480m-20260913',
            source_hashes={name:file_sha256(ROOT/name) for name in ['src/scaled_atom_sampling.py',
                'scripts/tools/evaluate_imagenet_scaled_samplers.py']})
        atomic_json(out/'specification.json',spec)
        if args.run_id:
            import wandb
            run=wandb.init(entity='helloimlixin-rutgers',project='laser',id=args.run_id,
                name=args.run_id,dir=str(out),config=spec,resume='never')
    dist.barrier()
    if not args.skip_diagnostics:
        status('diagnosing_filter_on_heldout_prefixes')
        diagnostics=diagnose(model,tokenizer,base,device,rank,world)
        if rank==0:
            atomic_json(out/'filter-diagnostics.json',dict(images=32,prefixes='heldout hard validation tokens',depths=diagnostics))
            print(json.dumps(dict(filter_diagnostics=diagnostics)),flush=True)
    results={}
    for name in args.settings:
        settings=SAMPLER_SETTINGS[name]
        moment=FeatureMoments(device)
        usage=torch.zeros(4,10,dtype=torch.float64,device=device)
        indices=list(range(rank,args.samples,world))
        torch.manual_seed(73000+rank)
        status('sampling',setting=name,samples=args.samples)
        for start in range(0,len(indices),32):
            selected=indices[start:start+32]
            labels=torch.tensor([i%1000 for i in selected],device=device)
            codes=sample_codes(model,tokenizer,labels,settings)
            decoded=torch.cat([tokenizer.decode_code(x).mul(.5).add(.5).clamp(0,1) for x in codes.split(8)])
            assert torch.isfinite(decoded).all()
            moment.update(inception(decoded))
            for d in range(4):
                values=codes[...,d].flatten()
                usage[d,0]+=(values==0).sum();usage[d,1]+=len(values)
                bins=(values[values!=0]-1)%8
                usage[d,2:]+=torch.bincount(bins,minlength=8)
            if start%128==0:status('sampling',setting=name,completed=min(args.samples,(start+len(selected))*world),samples=args.samples)
        measured=moment.finish();dist.all_reduce(usage)
        assert measured[0]==args.samples
        if rank==0:
            status('computing_fid',setting=name)
            score=fid_from_moments(measured,reference)
            np.savez(out/f'{name}-statistics.npz',mu=measured[1],sigma=measured[2],images=args.samples)
            results[name]=dict(fid=score,samples=args.samples,settings=settings,
                zero_fraction_per_depth=(usage[:,0]/usage[:,1]).cpu().tolist(),
                coefficient_frequencies_per_depth=(usage[:,2:]/usage[:,1,None]).cpu().tolist())
            atomic_json(out/'results.json',dict(**spec,results=results))
            print(json.dumps(dict(setting=name,**results[name])),flush=True)
        dist.barrier()
        target=out/f'{name}-10x8.png'
        grid(model,tokenizer,settings,target,step,epoch,device,rank,world)
        if rank==0 and run:
            import wandb
            run.log({'setting':name,'fid':results[name]['fid'],f'fid/{name}':results[name]['fid'],
                f'grids/{name}':wandb.Image(str(target),caption=json.dumps(settings))})
    status('complete',results=results)
    if run:run.finish()
    dist.destroy_process_group()


if __name__=='__main__':main()
