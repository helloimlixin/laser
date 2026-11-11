"""Compare frozen stochastic RQ targets on the cached 10k training holdout."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

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
    parser=argparse.ArgumentParser();parser.add_argument('--rank',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(2);torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(4*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False
    vectors,spec=read_vectors(BASE/'inputs/joint-codebook.pt')
    q=RQBottleneck(latent_shape=[8,8,256],code_shape=[8,8,4],n_embed=24576,
        decay=.99,shared_codebook=False,restart_unused_codes=False).cuda()
    for depth,table in enumerate(q.codebooks):
        table.weight[:-1].copy_(vectors[depth]);table.weight[-1].zero_()
        table.embed_ema.copy_(vectors[depth]);table.cluster_size_ema.fill_(1.)
    q.requires_grad_(False).eval()
    latents=np.load(BASE/f'cache/held-latents-rank{args.rank}.npy',mmap_mode='r')
    result={};started=time.time()
    for temperature in [0.,.5,.375,.25]:
        torch.manual_seed(771001+args.rank);torch.cuda.manual_seed(771001+args.rank)
        sums=torch.zeros(10,device='cuda',dtype=torch.float64)
        for first in range(0,len(latents),8):
            x=torch.tensor(np.array(latents[first:first+8]),device='cuda')
            _,greedy=q.quantize(x)
            if temperature:
                soft,codes=q.get_soft_codes(x,temp=temperature,stochastic=True)
                entropy=-(soft*soft.clamp_min(1e-30).log()).sum(-1)
                sums[2:6]+=entropy.sum((0,1,2));del soft,entropy
            else:codes=greedy
            recon=q.embed_code(codes)
            sums[0]+=(x-recon).square().sum();sums[1]+=x.numel()
            sums[6:10]+=(codes!=greedy).sum((0,1,2))
            del x,greedy,codes,recon
        vals=sums.cpu().tolist();sites=len(latents)*64
        result[str(temperature)]=dict(latent_mse=vals[0]/vals[1],
            entropy_by_depth=[v/sites for v in vals[2:6]],
            code_flip_fraction_vs_greedy_by_depth=[v/sites for v in vals[6:10]])
        print(json.dumps(dict(rank=args.rank,temperature=temperature,**result[str(temperature)])),flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/f'rank-{args.rank}.json').write_text(json.dumps(dict(rank=args.rank,
        images=len(latents),training_holdout=True,validation_used=False,codewords=24576,
        measures_are_latent_not_rfid=True,results=result,seconds=time.time()-started),indent=2)+'\n')


if __name__=='__main__':main()
