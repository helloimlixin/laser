"""Fit relative noise on training images, preserving the lower greedy NMSE."""
import argparse
import json
import os
from pathlib import Path
import sys

BASE=Path('/tmp/laser-rq24576-native-8h100-20261108')
sys.path[:0]=[str(BASE/'source'),str(BASE)]
os.environ['RQ_UPSTREAM_ROOT']=str(BASE/'upstream')
import numpy as np
import torch
from src import rq_upstream
from runtime.tokenizer import read_vectors
from rqvae.models.rqvae.quantizations import RQBottleneck


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--rank',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();rank=args.rank;out=args.output
    torch.set_num_threads(2);torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(2.5*2**30/torch.cuda.get_device_properties(0).total_memory)
    vectors,_=read_vectors(BASE/'inputs/joint-codebook.pt')
    q=RQBottleneck(latent_shape=[8,8,256],code_shape=[8,8,4],n_embed=24576,decay=.99,
        shared_codebook=False,restart_unused_codes=False).cuda().eval().requires_grad_(False)
    for d,t in enumerate(q.codebooks):
        t.weight[:-1].copy_(vectors[d]);t.weight[-1].zero_();t.embed_ema.copy_(vectors[d]);t.cluster_size_ema.fill_(1.)
    data=np.load(BASE/f'cache/held-latents-rank{rank}.npy',mmap_mode='r')
    profiles=[('greedy',0.,771001)]+[(f't{temp:.3f}_s{seed}',temp,seed) for temp in [.075,.08,.085,.09,.1] for seed in [771001,781001]]
    for name,temp,seed in profiles:
        torch.manual_seed(seed+rank);torch.cuda.manual_seed(seed+rank);values=[];entropy=torch.zeros(4,device='cuda',dtype=torch.float64)
        for first in range(0,len(data),8):
            x=torch.tensor(np.array(data[first:first+8]),device='cuda')
            if temp:
                soft,codes=q.get_soft_codes(x,temp=temp,stochastic=True)
                for depth in range(4):
                    a=soft[...,depth,:];entropy[depth]+=-(a*a.clamp_min(1e-30).log()).sum().double()
                del soft,a
            else:_,codes=q.quantize(x)
            reconstructed=q.embed_code(codes)
            values.append(torch.stack([(x-reconstructed).square().flatten(1).sum(1),x.square().flatten(1).sum(1)],1).double().cpu())
        np.savez(out/f'{name}-rank-{rank}.npz',per_image=torch.cat(values).numpy(),entropy=entropy.cpu().numpy(),temperature=temp,seed=seed)
        print(json.dumps(dict(rank=rank,profile=name,images=len(data))),flush=True)
    (out/f'passed-rank-{rank}.json').write_text(json.dumps(dict(passed=True,rank=rank,images=len(data),
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)))


if __name__=='__main__':main()
