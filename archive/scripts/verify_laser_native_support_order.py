#!/usr/bin/env python3
"""Four-GPU correctness/memory preflight; disposable fresh weights, no FID run."""
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path('/workspace/Projects/laser')
BASE=Path('/tmp/laser-compound512-native-imagenet-20261016')
PROBE=Path('/tmp/laser-compound512-atom-identity-pilot-20261016/probe')
OUT=ROOT/'outputs/imagenet-native-support-order-preflight-20261016'
sys.path[:0]=[str(ROOT),str(ROOT/'runtime')]
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from src import rq_upstream
from src.models.laser_native_support_order_adapter import build_native_support_order_model
from src.training.laser_compound_replay_recipe import CompoundReplayRecipe
from src.training.laser_hard_ar import verify_tokens
from src.training.rq_imagenet_runtime import write_json
from rqvae.utils.config import load_config,augment_arch_defaults


def digest(model):
    result=hashlib.sha256()
    for name,value in model.state_dict().items():
        result.update(name.encode());result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def main():
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    assert world==4
    device=torch.device('cuda',int(os.environ['LOCAL_RANK']));torch.cuda.set_device(device)
    torch.set_num_threads(4)
    from datetime import timedelta
    dist.init_process_group('nccl',device_id=device,timeout=timedelta(minutes=30))
    order='support_first' if rank<2 else 'interleaved'
    torch.manual_seed(26101694);torch.cuda.manual_seed_all(26101694)
    arch=augment_arch_defaults(load_config(BASE/'inputs/stage2-training-config.yaml').arch)
    with torch.device(device):model=build_native_support_order_model(arch,num_atoms=16384,coefficient_bins=512,order=order)
    initialization_sha=digest(model)
    profile=CompoundReplayRecipe(BASE);aux,_=profile.tokenizer(device);before=profile.digest(aux)
    probe=torch.load(PROBE/f'rank{rank}.pt',map_location='cpu',weights_only=False)
    codes=probe['codes'].to(device);labels=probe['labels'].to(device)
    values=probe['coefficients'].to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,betas=(.9,.95),weight_decay=.0001)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024.)
    model.train()
    def backward(batch):
        output=model(codes[:batch],aux,labels[:batch],amp=True)
        p=(-(values[:batch,...,None]-aux.coeff_bins).square()/.03125).softmax(-1)
        target=dict(atoms=codes[:batch,...,0::2],coefficient_probabilities=p)
        loss=profile.loss(model,output,target)
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        return float(loss.detach())
    loss=backward(8)
    scaler.unscale_(optimizer)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
    assert all(int(state['step'])==1 for state in optimizer.state.values())
    benchmarks=[]
    for batch in [8,16,32,64]:
        optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize(device);start=time.monotonic()
        try:
            measured_loss=backward(batch);torch.cuda.synchronize(device)
            elapsed=time.monotonic()-start
            validation_start=time.monotonic()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            result=dict(microbatch=batch,passed=True,forward_backward_seconds=elapsed,
                gradient_validation_seconds=time.monotonic()-validation_start,
                loss=measured_loss,peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
        except torch.cuda.OutOfMemoryError:
            result=dict(microbatch=batch,passed=False,out_of_memory=True)
        finally:optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        benchmarks.append(result)
        write_json(OUT/f'progress-rank{rank}.json',dict(order=order,phase='batch_memory',benchmarks=benchmarks,time=time.time()))
        if not result['passed']:break
    model.eval()
    with torch.no_grad():
        truth=codes[:2];condition=labels[:2]
        expected=model(truth,aux,condition,amp=True)
        altered=truth.clone();altered[:,0,0,1]=16384+((altered[:,0,0,1]-16384+100)%512)
        changed=model(altered,aux,condition,amp=True)
        if order=='support_first':
            torch.testing.assert_close(changed['atom_logits'][:,0,0],expected['atom_logits'][:,0,0],rtol=0,atol=0)
            torch.testing.assert_close(changed['coeff_logits'][:,0,0,0],expected['coeff_logits'][:,0,0,0],rtol=0,atol=0)
        del changed,altered
        prefix=torch.zeros_like(truth);prefix[...,1::2]=16384+256
        model.init_cache();maximum=0.
        for h in range(8):
            for w in range(8):
                for event,external in enumerate(model.event_positions):
                    actual=model.cached_forward(prefix[:,:h+1],aux,condition,amp=True,sample_loc=(h,w,event))
                    atom=external%2==0;actual=actual[:,:16384] if atom else actual[:,16384:]
                    target=expected['atom_logits' if atom else 'coeff_logits'][:,h,w,external//2]
                    assert torch.equal(actual.isfinite(),target.isfinite())
                    p=target.float().log_softmax(-1);q=actual.float().log_softmax(-1)
                    kl=torch.where(p.isfinite(),p.exp()*(p-q),0.).sum(-1)
                    assert torch.isfinite(kl).all();maximum=max(maximum,float(kl.max()))
                    prefix[:,h,w,external]=truth[:,h,w,external]
        assert maximum<2.e-4,maximum
        model.init_cache();del expected,prefix
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize(device);start=time.monotonic()
        generated=model.sample_sparse(128,aux,labels[:128],amp=True,atom_temperature=.9,atom_top_p=.9,coeff_top_p=.85)
        torch.cuda.synchronize(device);sampling_seconds=time.monotonic()-start
        verify_tokens(aux,generated)
        decoded=profile.decode(aux,generated[:2]);assert decoded.shape==(2,3,256,256) and torch.isfinite(decoded).all()
    assert profile.digest(aux)==before
    result=dict(passed=True,rank=rank,order=order,initialization_sha256=initialization_sha,
        parameter_count=sum(p.numel() for p in model.parameters()),parameter_tensors=len(list(model.parameters())),
        native_spatial_layers=42,native_depth_layers=6,native_depth_events=8,separate_coefficient_micro_transformer=False,
        original_codec_and_tokenizer_unchanged=True,source_stage1_rfid=4.21400802130438,
        coefficient_bins=512,atoms=16384,auxiliary_loss_terms=0,sampled_trajectory_banks=0,
        disposable_optimizer_updates=1,production_training_updates=0,initial_loss=loss,gradient_norm=float(norm),
        memory_benchmarks=benchmarks,cache_events=512,maximum_fp16_kl=maximum,
        current_coefficient_causality_verified=order=='support_first',generated_images=128,
        sampling_seconds=sampling_seconds,sampling_images_per_second=128/sampling_seconds,
        sampling_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),fid_computed=False,time=time.time())
    write_json(OUT/f'rank{rank}.json',result)
    records=[None]*world;dist.all_gather_object(records,result)
    if rank==0:
        assert len({r['initialization_sha256'] for r in records})==1
        write_json(OUT/'verification.json',dict(passed=True,production_training_launched=False,
            exact_paired_initial_weights=True,tokenizer_unchanged=True,
            interleaved_and_support_first_ready_for_controlled_scratch_comparison=True,
            training_quality_not_yet_tested=True,ranks=records,upstream=rq_upstream.provenance(),time=time.time()))
        print(json.dumps(dict(passed=True,production_training_launched=False,ranks=records)),flush=True)
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
