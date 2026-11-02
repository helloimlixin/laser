#!/usr/bin/env python3
"""Compare frozen FP32 stage-1 batch sizes with exact target equivalence."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT=Path('/workspace/Projects/laser');SOURCE=Path('/tmp/laser-compound512-native-imagenet-20261016')
OUT=ROOT/'outputs/imagenet-native-support-order-speed-20261016'
sys.path[:0]=[str(ROOT),str(ROOT/'runtime')]
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader,Subset
from src.training.laser_compound_replay_recipe import CompoundReplayRecipe
from src.training.rq_imagenet_runtime import TarImageNet,write_json
from rqvae.utils.config import load_config
from rqvae.img_datasets.transforms import create_transforms


def main():
    rank=int(os.environ['RANK']);assert int(os.environ['WORLD_SIZE'])==4
    device=torch.device('cuda',int(os.environ['LOCAL_RANK']));torch.cuda.set_device(device);torch.set_num_threads(4)
    dist.init_process_group('nccl',device_id=device)
    profile=CompoundReplayRecipe(SOURCE);aux,_=profile.tokenizer(device);before=profile.digest(aux)
    config=load_config(SOURCE/'inputs/stage2-training-config.yaml')
    data=TarImageNet(SOURCE/'data','train',create_transforms(config.dataset,split='train'),seed=261001)
    indices=torch.randperm(len(data),generator=torch.Generator().manual_seed(271016))[:64].tolist()
    images,_=next(iter(DataLoader(Subset(data,indices),batch_size=64,num_workers=4)))
    image_sha=hashlib.sha256(images.numpy().tobytes()).hexdigest();images=images.to(device)
    chunk=[8,16,32,64][rank];profile.recipe.targets.encoder_batch_size=chunk
    torch.manual_seed(271017);profile.targets(aux,images)
    times=[]
    for _ in range(3):
        torch.manual_seed(271018);torch.cuda.synchronize(device);start=time.monotonic()
        target,codes=profile.targets(aux,images)
        torch.cuda.synchronize(device);times.append(time.monotonic()-start)
    values=target['coefficients'];atoms=target['atoms']
    baseline_values=values.clone();baseline_atoms=atoms.clone();baseline_codes=codes.clone()
    dist.broadcast(baseline_values,src=0);dist.broadcast(baseline_atoms,src=0);dist.broadcast(baseline_codes,src=0)
    result=dict(rank=rank,encoder_batch_size=chunk,images=64,image_sha256=image_sha,seconds=times,
        median_seconds=statistics.median(times),images_per_second_per_gpu=64/statistics.median(times),
        atom_ids_identical=bool(torch.equal(atoms,baseline_atoms)),
        continuous_coefficients_identical=bool(torch.equal(values,baseline_values)),
        stochastic_prefix_codes_identical=bool(torch.equal(codes,baseline_codes)),
        atom_disagreements=int((atoms!=baseline_atoms).sum()),
        sampled_coefficient_disagreements=int((codes[...,1::2]!=baseline_codes[...,1::2]).sum()),
        maximum_coefficient_difference=float((values-baseline_values).abs().max()),
        stage1_precision='FP32',frozen_tokenizer_unchanged=profile.digest(aux)==before,time=time.time())
    records=[None]*4;dist.all_gather_object(records,result)
    if rank==0:
        assert len({r['image_sha256'] for r in records})==1 and all(r['frozen_tokenizer_unchanged'] for r in records)
        write_json(OUT/'encoder-batches.json',dict(passed=True,records=records,training_updates_added=0,
            reference_encoder_batch_size=8,time=time.time()))
        print(json.dumps(records),flush=True)
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
