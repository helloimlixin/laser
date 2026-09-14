#!/usr/bin/env python3
"""Fit stage-specific atom coefficients while retaining a 32k token alphabet."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.compact_rq_training import DepthAdaptiveScaledAtomRQ
from src.scaled_atom_rq import continuous_matching_pursuit,fit_signed_levels
from src.original_rq_training import atomic_json,file_sha256


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=ROOT/'outputs/imagenet-rfid421-rq8-refit-20260913')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--images',type=int,default=8192)
    p.add_argument('--passes',type=int,default=16)
    p.add_argument('--levels',type=int,choices=[2,4],default=2)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8)
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    calibration=json.loads((args.source/'temperature-calibration.json').read_text())
    spec=torch.load(args.source/'scaled-atom-codebook.pt',map_location='cpu',weights_only=True)
    dictionary=spec['dictionary'].to(args.device)
    values=np.load(args.source/'cache/train-view0-latents.npy',mmap_mode='r')
    excluded=set(calibration['calibration_indices'])
    indices=[int(i) for i in np.random.default_rng(91332769).permutation(len(values)) if int(i) not in excluded][:args.images]
    latents=values[indices].copy()
    coefficients=[]
    for start in range(0,len(latents),16):
        z=torch.from_numpy(latents[start:start+16]).to(args.device)
        coefficients.append(continuous_matching_pursuit(z,dictionary)['coefficients'])
    coefficients=torch.cat(coefficients)
    initial=torch.stack([fit_signed_levels(coefficients[...,d],args.levels).expand(16384,-1) for d in range(4)])
    q=DepthAdaptiveScaledAtomRQ(dictionary,initial)
    prior=q.levels.clone()
    trace=[]
    started=time.time()
    for iteration in range(args.passes):
        counts=torch.zeros(4,q.vocab_size,device=args.device,dtype=torch.float64)
        sums=torch.zeros_like(counts)
        error=0.
        for start in range(0,len(latents),16):
            z=torch.from_numpy(latents[start:start+16]).to(args.device)
            result=q.quantize(z,return_projections=True)
            for d in range(4):
                ids=result['codes'][...,d].flatten()
                counts[d]+=torch.bincount(ids,minlength=q.vocab_size)
                sums[d]+=torch.bincount(ids,weights=result['projections'][...,d].flatten().double(),minlength=q.vocab_size)
            error+=(z-result['quantized']).square().sum().item()
        updated=(sums[:,1:].reshape_as(prior)+4.*prior)/(counts[:,1:].reshape_as(prior)+4.)
        updated=updated.sort(-1).values
        assert (updated[...,:args.levels//2]<0).all() and (updated[...,args.levels//2:]>0).all()
        assert (updated[...,1:]>updated[...,:-1]).all()
        for d,child in enumerate(q.codebooks):
            child.levels.copy_(updated[d])
        row=dict(iteration=iteration+1,latent_mse_before_update=error/latents.size,
            elapsed_seconds=time.time()-started)
        trace.append(row)
        print(json.dumps(row),flush=True)
        atomic_json(args.output/'status.json',dict(phase='fitting_depth_coefficients',**row))
        if iteration+1 in (4,8,16,args.passes):
            out=args.output/f'pass{iteration+1:03d}'
            out.mkdir(exist_ok=True)
            torch.save(dict(format_version=1,kind='depth_adaptive_scaled_atom_rq',
                dictionary=dictionary.cpu(),levels=q.levels.cpu(),depth=4,zero_token=0,code_shape=[8,8,4],
                source_stage1_checkpoint_sha256=spec['checkpoint_sha256']),out/f'compact{args.levels}-codebook.pt')
            atomic_json(out/f'fit{args.levels}.json',dict(fit_indices=indices,train_view=0,fit_on_training_images_only=True,
                method='Depth-specific atom Lloyd calibration; fixed tables during encoding',
                passes=iteration+1,prior_weight=4.,trace=trace,depth_specific=True,
                source_hashes={str(path.relative_to(ROOT)):file_sha256(path) for path in
                    [Path(__file__),ROOT/'src/compact_rq_training.py']}))
    atomic_json(args.output/'status.json',dict(phase='complete',elapsed_seconds=time.time()-started))


if __name__=='__main__':main()
