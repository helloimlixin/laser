#!/usr/bin/env python3
"""Rebuild Church latents, compact book and calibration from the new tokenizer."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time

# Use the same immutable dependencies as the preceding stage-2 comparison.
import train_church_laser_original_recipe as recipe
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader,Subset
from torchvision.utils import save_image
from rqvae.img_datasets.lsun import LSUNClass
from rqvae.img_datasets.transforms import create_transforms
from rqvae.metrics.fid import get_inception_model,frechet_distance,mean_covar_numpy
from src.original_rq_training import (atomic_json,file_sha256,state_sha256,IndexedImages,
    validation_latents,FeatureMoments,fid_from_moments)
from src.scaled_atom_rq import FrozenSparseBackbone,continuous_matching_pursuit,fit_signed_levels
from src.adaptive_scaled_atom_rq import AdaptiveScaledAtomRQ,fit_atom_levels
from src.compact_rq_training import FrozenCompactTokenizer
from calibrate_scaled_atom_temperature import measure

ROOT=recipe.ROOT


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage1-directory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    rank,world=int(os.environ['RANK']),int(os.environ['WORLD_SIZE'])
    assert world==2
    device=torch.device('cuda',int(os.environ['LOCAL_RANK']));torch.cuda.set_device(device)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=True
    dist.init_process_group('nccl',timeout=timedelta(hours=3))
    out=args.output.resolve();cache=out/'cache'
    if rank==0:cache.mkdir(parents=True,exist_ok=False)
    dist.barrier();started=time.time()
    def status(phase,**values):
        if rank==0:
            row=dict(phase=phase,elapsed_seconds=time.time()-started,updated_unix=time.time(),**values)
            atomic_json(out/'status.json',row);print(json.dumps(row),flush=True)
    completed=json.loads((args.stage1_directory/'complete.json').read_text())
    assert completed['smoke_only']==args.smoke
    assert completed['epoch']==(1 if args.smoke else 3)
    checkpoint=Path(completed['checkpoint'])
    ckpt_hash=file_sha256(checkpoint);assert ckpt_hash==completed['checkpoint_sha256']
    backbone=FrozenSparseBackbone(checkpoint).to(device).eval()
    before=state_sha256(backbone)
    config=recipe.load_stage2_config(recipe.UPSTREAM)
    dataset=LSUNClass('/tmp/laser-sign-data','church',create_transforms(config.dataset,split='train'))
    assert len(dataset)==126227
    n=128 if args.smoke else len(dataset)
    path=cache/'latents-fp32.npy'
    if rank==0:
        array=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(n,8,8,256));del array
    dist.barrier();array=np.load(path,mmap_mode='r+')
    indices=list(range(rank,n,world))
    loader=DataLoader(Subset(IndexedImages(dataset),indices),batch_size=64,num_workers=8,pin_memory=True)
    seen=0
    for images,ids in loader:
        z=backbone.encode(images.to(device,non_blocking=True))
        assert z.dtype==torch.float32 and torch.isfinite(z).all()
        array[ids.numpy()]=z.cpu().numpy();seen+=len(images)
        if seen%1280==0 or seen==len(indices):status('caching_new_encoder',images_per_rank=seen,images=n)
    array.flush();del array
    heldout=validation_latents(backbone,config,device,rank,world)
    torch.save(heldout,cache/f'validation-rank{rank}.pt')
    assert before==state_sha256(backbone)
    dist.barrier()
    book=out/'compact-codebook.pt'
    array=np.load(path,mmap_mode='r')
    if rank==0:
        fit_start,fit_count=(64,64) if args.smoke else (64000,4096)
        fit_values=np.array(array[fit_start:fit_start+fit_count],copy=True)
        coefficients=[]
        for start in range(0,len(fit_values),16):
            z=torch.as_tensor(fit_values[start:start+16],device=device)
            coefficients.append(continuous_matching_pursuit(z,backbone.dictionary)['coefficients'].cpu())
        initial=fit_signed_levels(torch.cat(coefficients).to(device),2)
        quantizer=AdaptiveScaledAtomRQ(backbone.dictionary,initial.expand(16384,-1))
        passes=1 if args.smoke else 8
        trace=fit_atom_levels(quantizer,fit_values,passes=passes,batch_size=16,prior_weight=4.,
            callback=lambda row:status('fitting_new_compact_book',**row))
        torch.save(dict(format_version=1,kind='adaptive_scaled_atom_rq',dictionary=quantizer.dictionary.cpu(),
            levels=quantizer.levels.cpu(),depth=4,zero_token=0,code_shape=[8,8,4],
            source_stage1_checkpoint=str(checkpoint),source_stage1_checkpoint_sha256=ckpt_hash),book)
        atomic_json(out/'fit-report.json',dict(fit_indices=[fit_start,fit_start+fit_count],
            passes=passes,levels=2,prior_weight=4.,trace=trace,initial_shared_levels=initial.cpu().tolist(),
            source_checkpoint_sha256=ckpt_hash,smoke_only=args.smoke))
        del quantizer,coefficients,fit_values
    dist.barrier()
    tokenizer=FrozenCompactTokenizer(checkpoint,book).to(device).eval()
    token_hash=state_sha256(tokenizer)
    torch.testing.assert_close(tokenizer.backbone.dictionary,backbone.dictionary,rtol=0,atol=0)
    del backbone
    if rank==0:
        reference_path=ROOT/'outputs/church-scaled-atom-stage2-20260913/temperature-calibration.json'
        reference=json.loads(reference_path.read_text())
        ids=list(range(32,48)) if args.smoke else list(range(62048,62176))
        z=torch.from_numpy(array[ids].copy()).to(device)
        control=reference['original_rq_control'];limit=control['sampled_to_hard_residual_mse_ratio']+.02
        sweep=[]
        for temperature in [.5,.25,.125,.0625,.03125,.015625,.0078125,.00390625]:
            result=measure(tokenizer.quantizer,z,temperature,99200);sweep.append(result)
            status('calibrating_new_token_noise',**result)
            if result['sampled_to_hard_residual_mse_ratio']<=limit:break
        assert sweep[-1]['sampled_to_hard_residual_mse_ratio']<=limit
        atomic_json(out/'temperature-calibration.json',dict(selected_temperature=sweep[-1]['temperature'],
            calibration_indices=ids,images=len(ids),original_rq_control=control,allowed_mse_ratio=limit,
            compact_rq_sweep=sweep,original_rq_control_reused=True,control_report=str(reference_path),
            control_report_sha256=file_sha256(reference_path),source_checkpoint_sha256=ckpt_hash,
            codebook_sha256=file_sha256(book),smoke_only=args.smoke,
            selection_rule='Largest tested temperature with sampled/hard latent MSE ratio <= original RQ control + .02'))
    dist.barrier()
    # Matched screen uses fresh originals and reconstructions from the same images.
    screen_n=32 if args.smoke else 4096
    loader=DataLoader(Subset(IndexedImages(dataset),list(range(rank,screen_n,world))),
        batch_size=16,num_workers=4,pin_memory=True)
    inception=get_inception_model().eval().requires_grad_(False).to(device)
    originals,reconstructions=FeatureMoments(device),FeatureMoments(device)
    pixel=torch.zeros(2,device=device,dtype=torch.float64)
    for images,ids in loader:
        real=images.to(device).mul(.5).add(.5)
        z=torch.from_numpy(array[ids.numpy()].copy()).to(device)
        codes=tokenizer.quantizer.quantize(z)['codes']
        decoded=tokenizer.decode_code(codes).mul(.5).add(.5).clamp(0,1)
        assert torch.isfinite(decoded).all()
        originals.update(inception(real));reconstructions.update(inception(decoded))
        pixel[0]+=(real-decoded).square().sum();pixel[1]+=real.numel()
        if rank==0 and ids[0]==0:save_image(torch.cat([real,decoded]),out/'reconstruction-screen.png',nrow=8)
    a,b=originals.finish(),reconstructions.finish();dist.all_reduce(pixel)
    assert a[0]==b[0]==screen_n
    assert token_hash==state_sha256(tokenizer)
    if rank==0:
        matched=float(frechet_distance(a[1],a[2],b[1],b[2]))
        official=ROOT/'outputs/lsun-church-bar-20260910/assets/lsun_256_church.npz'
        refscore=fid_from_moments(b,official)
        assert np.isfinite(matched) and np.isfinite(refscore)
        mse=float(pixel[0]/pixel[1])
        atomic_json(out/'reconstruction-screen.json',dict(images=screen_n,matched_rfid=matched,
            reference_rfid=refscore,pixel_mse=mse,psnr=-10*np.log10(mse),
            old_compact_matched_rfid4096=8.392327092054956,smoke_only=args.smoke,
            checkpoint_sha256=ckpt_hash,codebook_sha256=file_sha256(book),tokenizer_unchanged=True))
        np.savez(out/'reconstruction-statistics.npz',mu=b[1],sigma=b[2],original_mu=a[1],original_sigma=a[2])
        receipt=dict(phase='complete',images=n,dtype='float32',shape=[n,8,8,256],world_size=world,
            checkpoint=str(checkpoint),checkpoint_sha256=ckpt_hash,codebook=str(book),codebook_sha256=file_sha256(book),
            levels=2,levels_per_atom=2,vocab_size=32769,latent_cache=str(path),cache_sha256=file_sha256(path),
            frozen_state_sha256=token_hash,frozen_state_unchanged=True,hard_codes_cached=False,
            stochastic_targets_recomputed_each_visit=True,cache_reused=False,stage1_epochs=completed['epoch'],
            transform='official Church resize256/center crop/normalize[-1,1]',precision='FP32; TF32 disabled',
            validation_shard_sha256={f'validation-rank{r}.pt':file_sha256(cache/f'validation-rank{r}.pt') for r in range(world)},
            smoke_only=args.smoke,updated_unix=time.time())
        atomic_json(cache/'complete.json',receipt)
        status('complete',matched_rfid=matched,images=n,smoke_only=args.smoke)
    dataset.env.close();dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    try:main()
    except BaseException as error:
        import sys
        if '--output' in sys.argv:
            out=Path(sys.argv[sys.argv.index('--output')+1]);out.mkdir(parents=True,exist_ok=True)
            atomic_json(out/f'failure-rank{os.environ.get("RANK","unknown")}.json',
                dict(type=type(error).__name__,error=str(error),updated_unix=time.time()))
        raise
