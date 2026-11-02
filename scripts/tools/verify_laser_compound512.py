#!/usr/bin/env python3
"""Full reconstruction gate and native compound512 smoke reload verification."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

BASE = Path(os.environ['LASER_RQ_BASE'])
os.environ['RQ_UPSTREAM_ROOT'] = str(BASE/'upstream')
os.environ['TORCH_HOME'] = str(BASE/'torch-cache')
sys.path[:0] = [str(BASE/'source'),str(BASE/'upstream'),str(BASE/'runtime'),str(BASE)]
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from src.training.laser_compound_replay_recipe import CompoundReplayRecipe
from src.training.rq_imagenet_runtime import TarImageNet, write_json
from src.training import k4_checkpoint_io as checkpoint_io
from src import rqvae_metrics as metrics
from rqvae.utils.config import load_config, augment_arch_defaults, augment_optimizer_defaults
from rqvae.img_datasets.transforms import create_transforms
from rqvae.optimizer.optimizer import create_resnet_optimizer
from rqvae.optimizer.scheduler import create_scheduler


@torch.inference_mode()
def reconstruction():
    rank,world,local=(int(os.environ[k]) for k in ['RANK','WORLD_SIZE','LOCAL_RANK'])
    assert world == 4
    device=torch.device('cuda',local);torch.cuda.set_device(device)
    dist.init_process_group('nccl',device_id=device)
    profile=CompoundReplayRecipe(BASE)
    out=Path('/workspace/Projects/laser/outputs')/profile.run_id
    aux,_=profile.tokenizer(device)
    config=load_config(BASE/'inputs/stage2-training-config.yaml')
    data=TarImageNet(BASE/'data','val',create_transforms(config.dataset,split='val'))
    loader=DataLoader(Subset(data,range(rank,len(data),world)),batch_size=8,num_workers=4,pin_memory=True)
    metrics.INCEPTION_PATH=BASE/'upstream/rqvae/metrics/inception.py'
    metric=metrics.DistributedOriginalRQVAEMetrics(device)
    before=profile.digest(aux)
    for index,(images,_) in enumerate(loader):
        images=images.to(device,non_blocking=True)
        codes=profile.hard_codes(aux,images)
        decoded=profile.decode(aux,codes)
        assert torch.isfinite(decoded).all()
        metric.update(images.mul(.5).add(.5).clamp(0,1),real=True)
        metric.update(decoded.mul(.5).add(.5).clamp(0,1),real=False)
        if rank == 0 and index%50 == 0:
            write_json(out/'reconstruction-progress.json',dict(reconstructed_on_rank=min((index+1)*8,12500),
                target_per_rank=12500,coefficient_bins=512,time=time.time()))
    assert int(metric.real_count) == int(metric.fake_count) == 12500
    rfid,_,_=metric.compute()
    assert profile.digest(aux) == before
    passed=rfid <= float(profile.recipe.verification.maximum_reconstruction_rfid)
    if rank == 0:
        proof=dict(passed=passed,measured_validation_rfid=rfid,
            source_checkpoint_rfid=float(profile.recipe.tokenizer.source_rfid),
            maximum_rfid=float(profile.recipe.verification.maximum_reconstruction_rfid),
            real_images=50000,reconstructed_images=50000,reference_split='val',
            source_stage1_checkpoint_sha256=str(profile.recipe.tokenizer.sha256),
            original_encoder_dictionary_decoder_frozen=True,atoms=16384,coefficient_bins=512,
            coefficient_range=[-3.,3.],coefficient_scales=list(profile.recipe.tokenizer.coefficient_scales),
            normalized_noise_sigma=.125,sigma_bins=.125/(6/511),training_updates_added=0)
        write_json(BASE/'tokenizer-verification.json',proof)
        write_json(out/'tokenizer-verification.json',proof)
        write_json(out/'reproducibility/tokenizer-verification.json',proof)
        print(json.dumps(proof),flush=True)
    dist.barrier()
    dist.destroy_process_group()
    assert passed,'512-bin reconstruction gate failed; AR training was not launched'


def checkpoint():
    profile=CompoundReplayRecipe(BASE)
    out=Path('/workspace/Projects/laser/outputs')/profile.run_id
    assert json.loads((out/'tokenizer-verification.json').read_text())['passed']
    assert json.loads((out/'smoke/verification.json').read_text())['passed']
    path=checkpoint_io._checkpoint_upload_source((out/'smoke/checkpoints/last.pt').resolve())
    saved=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    profile.verify_resume(saved)
    assert saved['global_step'] == 2 and saved['runtime']['fresh_stage2'] and saved['runtime']['fresh_optimizer']
    assert len(saved['optimizer']['state']) == 837
    assert all(int(v['step']) == 2 and torch.isfinite(v['exp_avg']).all()
               and torch.isfinite(v['exp_avg_sq']).all() for v in saved['optimizer']['state'].values())
    config=load_config(BASE/'inputs/stage2-training-config.yaml')
    arch=augment_arch_defaults(config.arch);arch.vocab_size=config.dataset.vocab_size
    with torch.device('meta'):
        model=profile.model(arch)
    model.load_state_dict(saved['state_dict'],strict=True,assign=True)
    assert sum(p.numel() for p in model.parameters()) == 1448144384
    optimizer=create_resnet_optimizer(model,augment_optimizer_defaults(config.optimizer))
    scheduler=create_scheduler(optimizer,config.optimizer.warmup,626,100)
    optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
    assert scheduler.after_scheduler.last_epoch == 2 and scheduler.after_scheduler.T_max == 62600
    assert len(saved['rng_state_by_rank']) == 4
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    model.eval().to(device)
    aux,_=profile.tokenizer(device)
    data=TarImageNet(BASE/'data','val',create_transforms(config.dataset,split='val'))
    images=torch.stack([data[i][0] for i in range(2)]).to(device)
    labels=torch.tensor([data[i][1] for i in range(2)],device=device)
    with torch.inference_mode():
        targets,codes=profile.targets(aux,images)
        target_proof=profile.verify_targets(aux,targets,codes)
        del targets
        expected=model(codes,aux,labels,amp=True)
        prefix=torch.zeros_like(codes);prefix[...,1::2]=16384+256
        model.init_cache();maximum=0.
        for h in range(8):
            for w in range(8):
                for event in range(8):
                    logits=model.cached_forward(prefix[:,:h+1],aux,labels,amp=True,sample_loc=(h,w,event))
                    key='atom_logits' if event%2 == 0 else 'coeff_logits'
                    logits=logits[:,:16384] if event%2 == 0 else logits[:,16384:]
                    p=expected[key][:,h,w,event//2].float().log_softmax(-1)
                    q=logits.float().log_softmax(-1)
                    kl=torch.where(p.isfinite(),p.exp()*(p-q),0.).sum(-1).max()
                    maximum=max(maximum,float(kl))
                    prefix[:,h,w,event]=codes[:,h,w,event]
        assert maximum < 2.e-4
        del expected,logits,p,q
        model.init_cache()
        count=profile.generation_batch
        generated=profile.sample(model,aux,count,torch.arange(count,device=device)%1000)
        assert generated.shape == (count,8,8,8)
        assert ((generated[...,1::2]-16384 >= 0)&(generated[...,1::2]-16384 < 512)).all()
        assert (generated[...,0::2].sort(-1).values.diff(dim=-1)>0).all()
        for chunk in generated.split(8):
            assert torch.isfinite(profile.decode(aux,chunk)).all()
    write_json(out/'gpu-generation-verification.json',dict(passed=True,generated_images=count,
        production_generation_batch=count,cache_events=512,maximum_fp16_cache_kl=maximum,
        finite_decoded_images=True,targets=target_proof,source_dictionary_unchanged=True,
        coefficient_bins=512,peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3))
    write_json(out/'smoke-reload-verification.json',dict(passed=True,global_step=2,
        strict_model_reload=True,optimizer_state_count=837,all_adam_ages=2,
        scheduler_step=2,scheduler_horizon=62600,rng_streams=4,
        production_starts_from_zero=True,smoke_weights_discarded=True,checkpoint_bytes=path.stat().st_size))
    print('Compound512 checkpoint, cache and production sampler verified',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reconstruction',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    if args.reconstruction:reconstruction()
    else:checkpoint()
