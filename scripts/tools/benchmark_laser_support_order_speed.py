#!/usr/bin/env python3
"""Measure compute and actual four-GPU training separately, using disposable weights."""
import argparse
from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

ROOT=Path('/workspace/Projects/laser')
SOURCE=Path('/tmp/laser-compound512-native-imagenet-20261016')
PROBE=Path('/tmp/laser-compound512-atom-identity-pilot-20261016/probe')
OUT=ROOT/'outputs/imagenet-native-support-order-speed-20261016'
sys.path[:0]=[str(ROOT),str(ROOT/'runtime')]
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from src.models.laser_native_support_order_adapter import build_native_support_order_model
import src.models.laser_native_support_order_adapter as adapter
from src.training.laser_compound_replay_recipe import CompoundReplayRecipe
from src.training.rq_sdpa_backend import sdpa_attention
from src.training.rq_imagenet_runtime import TarImageNet,write_json
from src.training.rq_imagenet_shared import training_backward,seed_all
from src.training.exact_global_batch import ExactGlobalBatchSampler
from runtime.amp_replay import ReplayWindow,recoverable_amp_step
from rqvae.utils.config import load_config,augment_arch_defaults,augment_optimizer_defaults
from rqvae.optimizer.optimizer import create_resnet_optimizer
from rqvae.optimizer.scheduler import create_scheduler
from rqvae.img_datasets.transforms import create_transforms


def validate_cache(model,aux,probe,device):
    model.eval();truth=probe['codes'][:2].to(device);labels=probe['labels'][:2].to(device)
    with torch.no_grad():
        reference=model(truth,aux,labels,amp=True)
        prefix=torch.zeros_like(truth);prefix[...,1::2]=16384+256;model.init_cache();maximum=0.
        for h in range(8):
            for w in range(8):
                for event,external in enumerate(model.event_positions):
                    actual=model.cached_forward(prefix[:,:h+1],aux,labels,amp=True,sample_loc=(h,w,event))
                    atom=external%2==0;actual=actual[:,:16384] if atom else actual[:,16384:]
                    target=reference['atom_logits' if atom else 'coeff_logits'][:,h,w,external//2]
                    assert torch.equal(actual.isfinite(),target.isfinite())
                    p=target.float().log_softmax(-1);q=actual.float().log_softmax(-1)
                    kl=torch.where(p.isfinite(),p.exp()*(p-q),0.).sum(-1)
                    assert torch.isfinite(kl).all();maximum=max(maximum,float(kl.max()))
                    prefix[:,h,w,external]=truth[:,h,w,external]
        assert maximum<2.e-4,maximum
        altered=truth.clone();altered[:,0,0,1]=16384+((altered[:,0,0,1]-16384+100)%512)
        changed=model(altered,aux,labels,amp=True)
        torch.testing.assert_close(changed['atom_logits'][:,0,0],reference['atom_logits'][:,0,0],rtol=0,atol=0)
        torch.testing.assert_close(changed['coeff_logits'][:,0,0,0],reference['coeff_logits'][:,0,0,0],rtol=0,atol=0)
    model.init_cache()
    return dict(passed=True,events=512,maximum_fp16_kl=maximum,future_coefficient_causality=True)


def model_benchmark(model,aux,profile,probe,device,rank):
    codes=probe['codes'][:128].to(device);labels=probe['labels'][:128].to(device)
    values=probe['coefficients'][:64].to(device)
    probabilities=(-(values[...,None]-aux.coeff_bins).square()/.03125).softmax(-1)
    target=dict(atoms=codes[:64,...,0::2],coefficient_probabilities=probabilities)
    def backward():
        output=model(codes[:64],aux,labels[:64],amp=True)
        loss=profile.loss(model,output,target);(loss*1024.).backward()
        return loss
    records=[]
    for backend in ['native','sdpa']:
        with sdpa_attention(model) if backend=='sdpa' else nullcontext():
            model.train()
            for _ in range(2):
                model.zero_grad(set_to_none=True);backward()
            measurements=[];gpu_seconds=[]
            for _ in range(5):
                model.zero_grad(set_to_none=True)
                start_event=torch.cuda.Event(enable_timing=True);end_event=torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize(device);start=time.monotonic();start_event.record()
                loss=backward();end_event.record();torch.cuda.synchronize(device)
                measurements.append(time.monotonic()-start);gpu_seconds.append(start_event.elapsed_time(end_event)/1000)
            # Validation deliberately stays outside every throughput timer.
            start=time.monotonic()
            assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            validation_seconds=time.monotonic()-start
            model.zero_grad(set_to_none=True);model.eval()
            record=dict(backend=backend,microbatch=64,forward_backward_seconds=measurements,
                median_forward_backward_seconds=statistics.median(measurements),cuda_seconds=gpu_seconds,
                images_per_second_per_gpu=64/statistics.median(measurements),
                excluded_gradient_validation_seconds=validation_seconds)
            if backend=='sdpa':record['cache_verification']=validate_cache(model,aux,probe,device)
            sampling_modes=['legacy_full_topk','skip_redundant_topk'] if backend=='native' else ['skip_redundant_topk']
            sampling=[];legacy_codes=None
            helper=adapter.sample_from_logits
            def legacy(logits,**kwargs):
                if kwargs['top_k'] is None:kwargs['top_k']=logits.shape[-1]
                return helper(logits,**kwargs)
            for sampling_mode in sampling_modes:
                with patch.object(adapter,'sample_from_logits',legacy) if sampling_mode=='legacy_full_topk' else nullcontext():
                    torch.manual_seed(262001);model.sample_sparse(16,aux,labels[:16],amp=True,atom_temperature=.9,atom_top_p=.9,coeff_top_p=.85)
                    times=[]
                    for repeat in range(3):
                        torch.manual_seed(262002+repeat);torch.cuda.synchronize(device);start=time.monotonic()
                        generated=model.sample_sparse(128,aux,labels,amp=True,atom_temperature=.9,atom_top_p=.9,coeff_top_p=.85)
                        torch.cuda.synchronize(device);times.append(time.monotonic()-start)
                        if repeat==0:
                            if sampling_mode=='legacy_full_topk':legacy_codes=generated.clone()
                            elif backend=='native':assert torch.equal(generated,legacy_codes)
                    sampling.append(dict(mode=sampling_mode,batch=128,seconds=times,
                        median_seconds=statistics.median(times),images_per_second_per_gpu=128/statistics.median(times),
                        excludes_decoder_and_inception=True))
            record['sampling']=sampling;records.append(record)
            write_json(OUT/f'model-progress-rank{rank}.json',dict(records=records,time=time.time()))
    return dict(passed=True,rank=rank,records=records,fully_autoregressive=True,
        skipped_topk_bitwise_equivalence=True,production_training_updates=0)


def distributed_benchmark(model,aux,profile,probe,config,device,rank,world,backend):
    optimizer=create_resnet_optimizer(model,augment_optimizer_defaults(config.optimizer))
    scheduler=create_scheduler(optimizer,config.optimizer.warmup,626,100)
    scaler=torch.amp.GradScaler('cuda')
    ddp=DDP(model,device_ids=[device.index],broadcast_buffers=False,gradient_as_bucket_view=True,bucket_cap_mb=100)
    dataset=TarImageNet(SOURCE/'data','train',create_transforms(config.dataset,split='train'),seed=261001)
    sampler=ExactGlobalBatchSampler(dataset,2048,world,rank,8,261001);sampler.set_epoch(0)
    loader=DataLoader(dataset,batch_sampler=sampler,num_workers=8,pin_memory=True,persistent_workers=True,prefetch_factor=2)
    iterator=iter(loader);replay=ReplayWindow(device);model.train();optimizer.zero_grad(set_to_none=True)
    maximum_retries=0;timed_start=None;times=[]
    def backward(images,labels,absolute):
        with nullcontext() if (absolute+1)%8==0 else ddp.no_sync():
            return training_backward(profile,ddp,aux,images.to(device,non_blocking=True),labels.to(device,non_blocking=True),
                scaler,world*len(images)/sampler.global_count(absolute),amp=True)
    for update in range(12):
        if update==2:
            torch.cuda.synchronize(device);dist.barrier();timed_start=time.monotonic()
        start=time.monotonic();replay.begin();loss_sum=torch.zeros((),device=device)
        for micro in range(8):
            images,labels=next(iterator);absolute=update*8+micro
            replay.append(images,labels,absolute);loss_sum+=backward(images,labels,absolute)
        norm,retries=recoverable_amp_step(model.parameters(),optimizer,scaler,max_norm=1.,replay_backward=lambda:replay.replay(backward))
        maximum_retries=max(maximum_retries,retries)
        scheduler.step();optimizer.zero_grad(set_to_none=True);dist.all_reduce(loss_sum)
        torch.cuda.synchronize(device)
        if update>=2:times.append(time.monotonic()-start)
    elapsed=time.monotonic()-timed_start
    duration=torch.tensor(elapsed,device=device);dist.all_reduce(duration,op=dist.ReduceOp.MAX)
    assert all(int(state['step'])==12 for state in optimizer.state.values())
    proof=validate_cache(model,aux,probe,device)
    return dict(passed=True,rank=rank,backend=backend,world_size=world,microbatch=64,accumulation=8,
        global_batch=2048,warmup_updates=2,timed_updates=10,timed_images=20480,
        max_rank_elapsed_seconds=float(duration),end_to_end_images_per_second=20480/float(duration),
        seconds_per_optimizer_update=float(duration)/10,local_step_seconds=times,
        includes=['DataLoader','FP32 frozen stage1 encoder','sparse target construction','forward and backward',
            'DDP communication','gradient clipping','AdamW step','cosine scheduler','scalar metric reduction'],
        excludes=['checkpoint persistence','W&B upload','FID generation'],
        maximum_amp_retries=maximum_retries,adam_age=12,optimizer_state_count=len(optimizer.state),
        cache_verification=proof,production_training_updates=0,weights_discarded=True,
        fully_autoregressive=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['model','distributed'])
    parser.add_argument('--backend',choices=['native','sdpa'],default='native');args=parser.parse_args()
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);assert world==4
    device=torch.device('cuda',int(os.environ['LOCAL_RANK']));torch.cuda.set_device(device);torch.set_num_threads(4)
    from datetime import timedelta
    dist.init_process_group('nccl',device_id=device,timeout=timedelta(minutes=30))
    seed_all(26101697);OUT.mkdir(exist_ok=True)
    config=load_config(SOURCE/'inputs/stage2-training-config.yaml')
    with torch.device(device):model=build_native_support_order_model(augment_arch_defaults(config.arch),num_atoms=16384,coefficient_bins=512,order='support_first')
    profile=CompoundReplayRecipe(SOURCE);profile.recipe.atom_ce_weight=1.
    aux,_=profile.tokenizer(device);before=profile.digest(aux)
    probe=torch.load(PROBE/'rank0.pt',map_location='cpu',weights_only=False)
    if args.action=='model':result=model_benchmark(model,aux,profile,probe,device,rank)
    else:
        with sdpa_attention(model) if args.backend=='sdpa' else nullcontext():
            result=distributed_benchmark(model,aux,profile,probe,config,device,rank,world,args.backend)
    assert profile.digest(aux)==before
    result.update(tokenizer_unchanged=True,torch_version=torch.__version__,time=time.time())
    tag=args.action+('-'+args.backend if args.action=='distributed' else '')
    write_json(OUT/f'{tag}-rank{rank}.json',result)
    records=[None]*world;dist.all_gather_object(records,result)
    if rank==0:
        write_json(OUT/f'{tag}.json',dict(passed=True,ranks=records,fully_autoregressive=True,time=time.time()))
        print(json.dumps(dict(passed=True,mode=tag,ranks=records)),flush=True)
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
