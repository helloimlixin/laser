#!/usr/bin/env python3
"""Exercise distributed original-Inception FID and shard resume on real weights."""
import codecs
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
import torch.distributed as dist
from scripts.tools.train_church_joint_distributed import generate
from src.church_joint_geometry import make_prior
from src.church_relative_noise import RelativeChurchAux


def main():
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl',timeout=timedelta(minutes=10))
    torch.set_num_threads(8);torch.serialization.add_safe_globals([codecs.encode])
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    base=ROOT/'outputs/church-joint-geometry-20260912/multi-gpu'
    saved=torch.load(base/'pre-migration-best-screen.pt',map_location='cpu',weights_only=False,mmap=True)
    config=saved['config'];model=make_prior().cuda();model.load_state_dict(saved['state_dict'],strict=True)
    aux=RelativeChurchAux(Path(config['stage1']),16384,2048,3.,coeff_scales=config['coeff_scales'],sparsity_level=4,
        soft_target_physical=False,sigma_cap=config['coefficient_noise_sigma_cap'],relative_sigma=config['relative_sigma'],
        truncate=config['truncate']).cuda().eval().requires_grad_(False)
    directory=base/'generation-verification'
    log=lambda row:print(json.dumps(row),flush=True) if rank==0 else None
    args=SimpleNamespace(generation_batch=512)
    first=generate(model,aux,config,args,directory,1025,99177,4930,rank,world,log)
    cached=generate(model,aux,config,args,directory,1025,99177,4930,rank,world,log)
    assert first['fid']==cached['fid']
    if rank==0:
        codes=torch.load(directory/'generated-codes.pt',map_location='cpu',weights_only=False)
        assert codes['atoms'].shape==(1025,8,8,4) and codes['coefficient_ids'].shape==(1025,8,8,4)
        assert codes['atoms'].min()>=0 and codes['atoms'].max()<16384
        assert codes['coefficient_ids'].min()>=0 and codes['coefficient_ids'].max()<2048
        assert (codes['atoms'].sort(-1).values.diff(dim=-1)>0).all()
        shards=[torch.load(directory/f'shard-{r:02d}.pt',map_location='cpu',weights_only=False) for r in range(world)]
        assert [int(s['fake_count']) for s in shards]==[512,513]
        result={'passed':True,'total_samples':1025,'shard_samples':[512,513],
            'fid':first['fid'],'cached_resume_identical_fid':True,'code_shapes_and_ranges_verified':True,
            'distinct_support_verified':True}
        (base/'generation-verification.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    dist.destroy_process_group()


if __name__=='__main__':main()
