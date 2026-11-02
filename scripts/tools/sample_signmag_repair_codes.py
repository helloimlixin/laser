"""Freeze class-balanced codes from the existing projected sign/magnitude prior."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=2000)
    p.add_argument('--batch-size',type=int,default=128)
    args=p.parse_args()
    sys.path[:0]=[str(args.source),str(args.source/'runtime')]
    import torch
    import torch.distributed as dist
    from types import SimpleNamespace
    from src.models.sign_magnitude_prior import SignMagnitudeRQTransformer
    from src.training.orthogonal_gmm_runtime import model_config
    rank,world=int(os.environ.get('RANK',0)),int(os.environ.get('WORLD_SIZE',1))
    if args.samples%1000 or args.samples%world: p.error('samples must divide by 1000 and worker count')
    if world>1:
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group('nccl',device_id=torch.device('cuda',int(os.environ['LOCAL_RANK'])))
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    args.output.mkdir(parents=True,exist_ok=True)
    target=args.output/f'rank-{rank:03d}.pt'
    if target.exists():raise FileExistsError(target)
    cache=torch.load(args.cache,map_location='cpu',mmap=True,weights_only=False)
    aux=SimpleNamespace(dictionary=cache['dictionary'].cuda(),
        coeff_scales=torch.as_tensor(cache['coeff_scales'],device='cuda'))
    saved=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    model=SignMagnitudeRQTransformer(model_config(),16384).cuda().eval()
    if saved.get('config',{}).get('projected_atom_head'):
        helper=Path(__file__).resolve().parents[2]/'src/models/dictionary_atom_readout.py'
        spec=importlib.util.spec_from_file_location('dictionary_atom_policy',helper)
        policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
        policy.install_projected_dictionary_readout(model,aux.dictionary)
    model.load_state_dict(saved['state_dict'],strict=True);del saved
    torch.manual_seed(261031+rank)
    aa,gg,yy=[],[],[];started=time.monotonic()
    with torch.no_grad():
        for first in range(0,args.samples//world,args.batch_size):
            count=min(args.batch_size,args.samples//world-first)
            labels=(torch.arange(first,first+count,device='cuda')*world+rank)%1000
            a,g=model.sample_compound(count,aux,cond=labels,atom_temperature=.85,
                atom_top_k=16384,atom_top_p=.9,coeff_temperature=1.,coeff_top_k=0,coeff_top_p=1.,amp=True)
            aa.append(a.cpu());gg.append(g.cpu());yy.append(labels.cpu())
            print(json.dumps(dict(rank=rank,generated=first+count,seconds=time.monotonic()-started)),flush=True)
    torch.save(dict(atoms=torch.cat(aa),gamma=torch.cat(gg),labels=torch.cat(yy),
        coeff_scales=aux.coeff_scales.cpu(),checkpoint=str(args.checkpoint),rank=rank,world=world,
        seed=261031,samples=args.samples,batch_size=args.batch_size,
        sampling=dict(atom_temperature=.85,atom_top_p=.9,coeff_temperature=1.,coeff_top_p=1.)),target)
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
