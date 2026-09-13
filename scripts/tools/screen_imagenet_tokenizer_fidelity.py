#!/usr/bin/env python3
"""Separate coefficient precision from pursuit geometry using a fixed ImageNet cache."""
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
os.environ.setdefault('TORCH_HOME','/workspace/tmp/official-rqvae-eval-cache')
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from rqvae.img_datasets.transforms import create_transforms
from rqvae.metrics.fid import get_inception_model,frechet_distance
from src.original_rq_training import atomic_json,file_sha256,state_sha256,FeatureMoments,fid_from_moments
from src.scaled_atom_rq import (FrozenSparseBackbone,ScaledAtomRQ,continuous_matching_pursuit,
    orthogonal_matching_pursuit,fit_signed_levels)
from src.imagenet_scaled_stage2 import ManifestImages,load_imagenet_config


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=ROOT/'outputs/imagenet-scaled-rq-stage2-20260913')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--images',type=int,default=4096)
    p.add_argument('--variants',nargs='+',default=['omp4','mp4','rq8','rq16','rq32','omp8','omp16','omp32','omp64'])
    p.add_argument('--levels-file',type=Path)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--matched-reference',action='store_true')
    p.add_argument('--reference-statistics',type=Path)
    p.add_argument('--baseline-results',type=Path)
    p.add_argument('--candidate-codebooks',type=Path)
    args=p.parse_args()
    rank,world=int(os.environ['RANK']),int(os.environ['WORLD_SIZE'])
    device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=True
    dist.init_process_group('nccl',timeout=timedelta(hours=1))
    out=args.output.resolve()
    if rank==0:out.mkdir(exist_ok=False,parents=True)
    dist.barrier()
    started=time.time()
    def status(phase,**values):
        if rank==0:
            row=dict(phase=phase,elapsed_seconds=time.time()-started,updated_unix=time.time(),**values)
            atomic_json(out/'status.json',row);print(json.dumps(row),flush=True)
    checkpoint=args.source/'assets/best_rfid_slot1_model.pt'
    backbone=FrozenSparseBackbone(checkpoint).to(device).eval()
    frozen_hash=state_sha256(backbone)
    gram=backbone.dictionary.T@backbone.dictionary
    checkpoint_hash=file_sha256(checkpoint)
    if args.levels_file:
        spec=torch.load(args.levels_file,weights_only=True,map_location='cpu')
        assert spec['checkpoint_sha256']==checkpoint_hash
    else:
        source_spec=json.loads((args.source/'temperature-calibration.json').read_text())
        fit_ids=source_spec['fit_indices']
        manifest=json.loads((args.source/'train-manifest.json').read_text())
        transform=create_transforms(load_imagenet_config(UPSTREAM).dataset,split='train')
        dataset=ManifestImages('/workspace/Projects/data/imagenet2012/train',manifest,transform,
            seed=421,indices=fit_ids[rank::world])
        loader=DataLoader(dataset,batch_size=16,num_workers=4,pin_memory=True)
        coeffs={'mp':[],'omp':[]}
        for images,_,_ in loader:
            z=backbone.encode(images.to(device))
            coeffs['mp'].append(continuous_matching_pursuit(z,backbone.dictionary)['coefficients'].cpu())
            coeffs['omp'].append(orthogonal_matching_pursuit(z,backbone.dictionary,gram)['coefficients'].cpu())
        parts=[None]*world
        dist.all_gather_object(parts,{k:torch.cat(v) for k,v in coeffs.items()})
        levels={}
        for kind in ['mp','omp']:
            values=torch.cat([x[kind] for x in parts]).to(device)
            levels[kind]={str(n):fit_signed_levels(values,n).cpu() for n in [8,16,32,64,128]}
        spec=dict(dictionary=backbone.dictionary.cpu(),levels=levels,fit_indices=fit_ids,
                  checkpoint_sha256=checkpoint_hash)
    if rank==0:torch.save(spec,out/'coefficient-levels.pt')
    levels={k:{int(n):v.to(device) for n,v in group.items()} for k,group in spec['levels'].items()}
    quantizers={n:ScaledAtomRQ(backbone.dictionary,levels['mp'][n]).to(device) for n in [8,16,32,64,128]}
    # Reuse the rejected run's exact eight-level codebook as a control.
    old=torch.load(args.source/'scaled-atom-codebook.pt',weights_only=True,map_location='cpu')
    quantizers[8]=ScaledAtomRQ(backbone.dictionary,old['levels']['8'].to(device)).to(device)
    custom={}
    if args.candidate_codebooks:
        candidate_spec=torch.load(args.candidate_codebooks,weights_only=True,map_location='cpu')
        assert candidate_spec['checkpoint_sha256']==checkpoint_hash
        torch.testing.assert_close(candidate_spec['dictionary'],backbone.dictionary.cpu(),rtol=0,atol=0)
        custom={name:ScaledAtomRQ(backbone.dictionary,book.to(device)).to(device)
                for name,book in candidate_spec['candidates'].items()}
        if rank==0:torch.save(candidate_spec,out/'candidate-codebooks.pt')
    count=min(args.images,50000)
    indices=sorted(np.random.default_rng(73421).choice(50000,count,replace=False).tolist())
    selected=indices[rank::world]
    cached=np.load(args.source/'cache/val-view0-latents.npy',mmap_mode='r')
    assert cached.dtype==np.float32 and cached.shape==(50000,8,8,256)
    inception=get_inception_model().eval().requires_grad_(False).to(device)
    moments={name:FeatureMoments(device) for name in args.variants}
    sums={name:torch.zeros(2,device=device,dtype=torch.float64) for name in args.variants}
    reference=ROOT/'third_party/rq-vae-transformer/assets/fid_stats/imagenet_256_train.npz'
    if args.matched_reference:
        if args.reference_statistics:
            reference=args.reference_statistics.resolve()
        else:
            original_moments=FeatureMoments(device)
            manifest=json.loads((args.source/'val-manifest.json').read_text())
            transform=create_transforms(load_imagenet_config(UPSTREAM).dataset,split='val')
            dataset=ManifestImages('/workspace/Projects/data/imagenet2012/val',manifest,transform,
                                   seed=421,indices=selected)
            loader=DataLoader(dataset,batch_size=64,num_workers=4,pin_memory=True)
            for batch,(images,_,ids) in enumerate(loader):
                images=images.to(device)
                if batch==0:
                    direct=backbone.encode(images[:4])
                    stored=torch.from_numpy(cached[ids[:4].numpy()].copy()).to(device)
                    torch.testing.assert_close(direct,stored,atol=2e-5,rtol=2e-5)
                original_moments.update(inception(images.mul(.5).add(.5).clamp(0,1)))
                if batch%20==0:status('original_validation_statistics',images=min(count,(batch+1)*64*world),total=count)
            original_stats=original_moments.finish()
            reference=out/'original-validation-statistics.npz'
            if rank==0:
                np.savez(reference,mu=original_stats[1],sigma=original_stats[2],images=count,
                         validation_indices=np.asarray(indices))
            del original_moments,loader,dataset
            dist.barrier()
        check=np.load(reference)
        assert int(check['images'])==count and np.array_equal(check['validation_indices'],np.asarray(indices))
    for start in range(0,len(selected),args.batch_size):
        batch_ids=selected[start:start+args.batch_size]
        z=torch.from_numpy(cached[batch_ids].copy()).to(device)
        omp=(orthogonal_matching_pursuit(z,backbone.dictionary,gram)
             if any(name.startswith('omp') for name in args.variants) else None)
        mp=continuous_matching_pursuit(z,backbone.dictionary) if 'mp4' in args.variants else None
        for name in args.variants:
            if name in custom:reconstruction=custom[name].quantize(z)['quantized']
            elif name=='omp4':reconstruction=omp['quantized']
            elif name=='mp4':reconstruction=mp['quantized']
            elif name.startswith('rq'):
                reconstruction=quantizers[int(name[2:])].quantize(z)['quantized']
            elif name.startswith('omp'):
                n=int(name[3:]);book=levels['omp'][n]
                # Keep OMP support and final least-squares values, quantizing only coefficients.
                book=torch.cat([book[book<0],book.new_zeros(1),book[book>0]])
                bins=torch.bucketize(omp['coefficients'].contiguous(),(book[:-1]+book[1:])/2)
                reconstruction=(backbone.dictionary.T[omp['atoms']]*book[bins][...,None]).sum(-2)
            else:raise ValueError(name)
            sums[name][0]+=(z-reconstruction).square().sum()
            sums[name][1]+=len(z)
            decoded=torch.cat([backbone.decode(chunk) for chunk in reconstruction.split(8)]).mul(.5).add(.5).clamp(0,1)
            moments[name].update(inception(decoded))
        if start%(args.batch_size*8)==0:
            status('screening',images=min(count,(start+len(z))*world),total=count,variants=args.variants)
    results={}
    if args.baseline_results:
        baseline=json.loads(args.baseline_results.read_text())
        assert baseline['checkpoint_sha256']==checkpoint_hash
        assert baseline['validation_indices']==indices and Path(baseline['reference']).resolve()==reference.resolve()
        results['omp4']=baseline['results']['omp4']
    for name in args.variants:
        measured=moments[name].finish()
        dist.all_reduce(sums[name])
        if rank==0:
            status('computing_fid',variant=name)
            fid=fid_from_moments(measured,reference)
            np.savez(out/f'{name}-statistics.npz',mu=measured[1],sigma=measured[2],images=count)
            results[name]=dict(rfid=fid,images=measured[0],latent_mse=float(sums[name][0]/(sums[name][1]*64*256)))
            if 'omp4' in results:results[name]['rfid_delta_vs_same_subset_omp']=fid-results['omp4']['rfid']
            atomic_json(out/'results.json',dict(checkpoint_sha256=checkpoint_hash,images=count,
                validation_indices=indices,results=results,reference=str(reference),
                full_validation=count==50000,fit_on_training_images_only=True))
            print(json.dumps(dict(variant=name,**results[name])),flush=True)
        dist.barrier()
    assert state_sha256(backbone)==frozen_hash
    status('complete',results=results)
    dist.destroy_process_group()


if __name__=='__main__':main()
