#!/usr/bin/env python3
"""Full-validation rFID of the actual stochastic compound training codes."""
import json
import os
from pathlib import Path
import sys
import time

BASE = Path('/tmp/laser-compound512-native-imagenet-20261016')
ROOT = Path('/workspace/Projects/laser')
OUT = ROOT/'outputs/imagenet-compound512-gap-20261016'
os.environ['RQ_UPSTREAM_ROOT'] = str(BASE/'upstream')
os.environ['TORCH_HOME'] = str(BASE/'torch-cache')
sys.path[:0] = [str(BASE/'source'),str(BASE/'runtime'),str(BASE/'upstream'),str(ROOT)]
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from src.training.laser_compound_replay_recipe import CompoundReplayRecipe
from src.training.rq_imagenet_runtime import TarImageNet, write_json
from src import rqvae_metrics as metrics
from rqvae.img_datasets.transforms import create_transforms
from rqvae.utils.config import load_config


@torch.inference_mode()
def main():
    rank,world,local = (int(os.environ[k]) for k in ['RANK','WORLD_SIZE','LOCAL_RANK'])
    assert world == 4
    torch.set_num_threads(4); device = torch.device('cuda',local);torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(4.5*1024**3/torch.cuda.get_device_properties(device).total_memory, device)
    torch.manual_seed(261016601+rank);torch.cuda.manual_seed(261016601+rank)
    dist.init_process_group('nccl',device_id=device)
    profile = CompoundReplayRecipe(BASE); aux,_ = profile.tokenizer(device)
    config = load_config(BASE/'inputs/stage2-training-config.yaml')
    data = TarImageNet(BASE/'data','val',create_transforms(config.dataset,split='val'))
    loader = DataLoader(Subset(data,range(rank,len(data),world)),batch_size=4,num_workers=4,pin_memory=True)
    metrics.INCEPTION_PATH = BASE/'upstream/rqvae/metrics/inception.py'
    metric = metrics.DistributedOriginalRQVAEMetrics(device)
    before = profile.digest(aux);started=time.time()
    for index,(images,_) in enumerate(loader):
        images=images.to(device,non_blocking=True)
        targets,codes=profile.targets(aux,images)
        del targets
        decoded=profile.decode(aux,codes)
        assert torch.isfinite(decoded).all()
        metric.update(images.mul(.5).add(.5).clamp(0,1),real=True)
        metric.update(decoded.mul(.5).add(.5).clamp(0,1),real=False)
        if rank == 0 and index%100==0:
            write_json(OUT/'training-rfid-progress.json',dict(reconstructed_on_rank=min((index+1)*4,12500),
                target_per_rank=12500,time=time.time()))
    assert int(metric.real_count) == int(metric.fake_count) == 12500
    value,_,_ = metric.compute()
    assert profile.digest(aux) == before
    if rank == 0:
        clean=json.loads((BASE/'tokenizer-verification.json').read_text())['measured_validation_rfid']
        proof=dict(passed=True,training_distribution_rfid=value,clean_nearest_bin_rfid=clean,
            rfid_delta=value-clean,real_images=50000,reconstructed_images=50000,reference_split='val',
            coefficient_bins=512,coefficient_temperature=.03125,normalized_sigma=.125,
            actual_training_tokenizer=True,source_stage1_unchanged=True,training_updates_added=0,
            stage2_model_used=False,generation_fid=False,seed=261016601,
            original_rq_inception=True,peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3,
            elapsed_seconds=time.time()-started)
        write_json(OUT/'training-distribution-rfid.json',proof);print(json.dumps(proof),flush=True)
    dist.barrier();dist.destroy_process_group()


if __name__ == '__main__':main()
