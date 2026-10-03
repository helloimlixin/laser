"""Verify a full 2,040-image update over six real GPU ranks before production."""
import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf
from src.training.cc3m_compound import build_model,make_aux,objective
from src.training.rqtransformer import gather_rank_rng_states
from scripts.tools.build_cc3m_compound_cache import write_json,text_tokenizer


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--cache-shard',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();o=OmegaConf.to_container(OmegaConf.load(a.config).options,resolve=True)
    rank,world,local=[int(os.environ[k]) for k in ('RANK','WORLD_SIZE','LOCAL_RANK')]
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.manual_seed(71)
    dist.init_process_group('nccl')
    data=torch.load(a.cache_shard,map_location='cpu',weights_only=True)
    scales=torch.tensor(data['meta']['coeff_abs_max'])/3
    aux=make_aux(o,scales.tolist(),device)
    prior=build_model(o).to(device);model=DDP(prior,device_ids=[local],broadcast_buffers=False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=o['lr'],betas=(.9,.95),weight_decay=1e-4,fused=True)
    tokenizer=text_tokenizer(.1)
    start=time.monotonic()
    for micro in range(o['accumulation']):
        begin=(micro*world+rank)*o['batch_size'];end=begin+o['batch_size']
        text=torch.tensor([r.ids for r in tokenizer.encode_batch(data['captions'][begin:end])],device=device)
        atoms=data['atoms'][begin:end].long().to(device)
        coeffs=(data['coeffs'][begin:end]/scales).to(device)
        with (nullcontext() if micro+1==o['accumulation'] else model.no_sync()),torch.autocast('cuda',dtype=torch.bfloat16):
            loss,_=objective(model,aux,atoms,coeffs,text,5.,o['accumulation'])
            loss.backward()
    grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step();optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize()
    probe=prior.cond_classifier.linear.weight.detach().float().sum()
    probes=[torch.zeros_like(probe) for _ in range(world)];dist.all_gather(probes,probe)
    assert all(torch.equal(probes[0],value) for value in probes)
    rng=gather_rank_rng_states(device)
    row=torch.tensor([float(loss.detach())*o['accumulation'],float(grad),torch.cuda.max_memory_allocated()/2**30,time.monotonic()-start],device=device)
    rows=[torch.zeros_like(row) for _ in range(world)];dist.all_gather(rows,row)
    assert all(torch.isfinite(value).all() for value in rows)
    if rank==0:
        write_json(a.output,dict(passed=True,world_size=world,total_batch_size=o['batch_size']*o['accumulation']*world,
            synchronized_parameters=True,rng_states=len(rng),ranks=[x.tolist() for x in rows],
            columns=['loss','grad_norm','peak_allocated_gib','seconds'],purpose='Integration only; production starts fresh'))
        print('Full six-GPU CC3M compound update passed',flush=True)
    dist.destroy_process_group()


if __name__=='__main__':main()
