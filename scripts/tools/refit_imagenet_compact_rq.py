#!/usr/bin/env python3
"""Alternate greedy RQ assignment and joint final-reconstruction coefficient fits."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.adaptive_scaled_atom_rq import AdaptiveScaledAtomRQ
from src.compact_rq_fitting import coefficient_normal_equations,solve_regularized_levels
from src.original_rq_training import atomic_json,file_sha256


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--study',type=Path,default=ROOT/'outputs/imagenet-compact-rq-study-20260913')
    p.add_argument('--source',type=Path,default=ROOT/'outputs/imagenet-rfid421-rq8-refit-20260913')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--levels',type=int,nargs='+',default=[2,4])
    p.add_argument('--passes',type=int,default=32)
    p.add_argument('--device',default='cuda:3')
    p.add_argument('--damping',type=float,default=.5)
    p.add_argument('--prior-weight',type=float,default=4.)
    args=p.parse_args()
    assert 0<args.damping<=1 and args.prior_weight>0
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8)
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    values=np.load(args.source/'cache/train-view0-latents.npy',mmap_mode='r')
    started=time.time()
    for levels in args.levels:
        spec=torch.load(args.study/f'compact{levels}-codebook.pt',map_location='cpu',weights_only=True)
        fit=json.loads((args.study/f'fit{levels}.json').read_text())
        indices=fit['fit_indices']
        latents=values[indices].copy()
        q=AdaptiveScaledAtomRQ(spec['dictionary'].to(args.device),spec['levels'].to(args.device))
        prior=q.levels.clone()
        ridge=args.prior_weight*q.norms[:,None].expand_as(prior).flatten().double()
        trace=[]
        for iteration in range(args.passes):
            ii,vv=[],[]
            rhs=torch.zeros(q.vocab_size-1,device=args.device,dtype=torch.float64)
            error=0.
            for start in range(0,len(latents),16):
                z=torch.from_numpy(latents[start:start+16]).to(args.device)
                result=q.quantize(z)
                normal,b=coefficient_normal_equations(q.dictionary,result['codes'],z,levels)
                ii.append(normal._indices());vv.append(normal._values());rhs+=b
                error+=(z-result['quantized']).square().sum().item()
            normal=torch.sparse_coo_tensor(torch.cat(ii,1),torch.cat(vv),
                (q.vocab_size-1,q.vocab_size-1),device=args.device).coalesce()
            proposed=solve_regularized_levels(normal,rhs,q.levels,prior,ridge)
            signs=prior.sign()
            proposed=signs*(signs*proposed).clamp_min(1e-4)
            proposed=proposed.sort(-1).values
            assert (proposed[:,1:]>proposed[:,:-1]).all()
            q.levels.lerp_(proposed,args.damping)
            row=dict(iteration=iteration+1,levels=levels,latent_mse_before_update=error/latents.size,
                elapsed_seconds=time.time()-started,normal_nonzeros=normal._nnz())
            trace.append(row)
            atomic_json(args.output/'status.json',dict(phase='joint_final_fit',**row))
            print(json.dumps(row),flush=True)
            if iteration+1 in (4,8,16,32,64,args.passes):
                folder=args.output/f'pass{iteration+1:03d}'
                folder.mkdir(exist_ok=True)
                torch.save({**spec,'levels':q.levels.cpu()},folder/f'compact{levels}-codebook.pt')
                atomic_json(folder/f'fit{levels}.json',{**fit,'initial_codebook_sha256':file_sha256(args.study/f'compact{levels}-codebook.pt'),
                    'method':'joint least squares on final greedy RQ reconstruction',
                    'prior_weight':args.prior_weight,'damping':args.damping,'joint_trace':trace,
                    'source_hashes':{str(path.relative_to(ROOT)):file_sha256(path) for path in
                        [Path(__file__),ROOT/'src/compact_rq_fitting.py']}})
            del normal,ii,vv
    atomic_json(args.output/'status.json',dict(phase='complete',elapsed_seconds=time.time()-started))


if __name__=='__main__':main()
