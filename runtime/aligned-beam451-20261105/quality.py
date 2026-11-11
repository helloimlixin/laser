"""Calibrate on training images, then validate the actual aligned teacher on 50k."""
import argparse
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

BASE=Path('/tmp/laser-aligned-beam451-20261105/frozen')
OUT=BASE.parent/'quality'
PUBLIC=Path('/workspace/Projects/laser/outputs/imnet-j16384-a4096-beam16-aligned-soft-scratch2-4a100-20261105')
sys.path[:0]=[str(BASE/'source'),str(BASE)]
os.environ['RQ_UPSTREAM_ROOT']=str(BASE/'upstream')
os.environ['TORCH_HOME']=str(BASE.parent/'torch-cache')
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader,Subset
from omegaconf import OmegaConf
from src import rq_upstream,rqvae_metrics as metrics
from src.models.rfid_preserved_joint_tokenizer import RFIDPreservedJointTokenizer
from src.training.rq_imagenet_runtime import TarImageNet
from rqvae.utils.config import load_config
from rqvae.img_datasets.transforms import create_transforms
from runtime.beam_teacher import beam_pool,posterior,conditional_targets,aligned_targets,verify_conditionals


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.pending');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def module_sha(module):
    h=hashlib.sha256()
    for name,value in module.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def temperature_calibration(pool):
    def stats(t):
        weights=posterior(pool,t)
        entropy=-(weights*weights.clamp_min(1e-30).log()).sum(-1).mean()
        cost=pool['costs'].gather(1,pool['canonical_index'][:,None]).squeeze(-1)
        excess=((weights*pool['costs']).sum(-1)-cost).mean()/cost.mean()
        return float(entropy),float(excess)
    def limit(column,target):
        lo,hi=1e-8,2.
        if stats(hi)[column]<=target:return hi
        for _ in range(48):
            middle=math.sqrt(lo*hi)
            if stats(middle)[column]<=target:lo=middle
            else:hi=middle
        return lo
    t_entropy=limit(0,.8)  # Expected entropy .2 nat per residual depth.
    t_error=limit(1,.001) # Mean latent error at most .1% above canonical beam.
    temperature=min(t_entropy,t_error)
    entropy,excess=stats(temperature)
    return dict(sequence_temperature=temperature,sequence_entropy=entropy,
        expected_conditional_entropy_per_depth=entropy/4,
        mean_relative_latent_error_above_canonical=excess,
        max_candidate_relative_latent_error=.01,
        calibration_images=2048,calibration_split='train',validation_used=False,
        entropy_target_per_depth=.2,max_mean_relative_latent_error=.001,
        temperature_limit_entropy=t_entropy,temperature_limit_error=t_error)


@torch.inference_mode()
def main(args):
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    device=torch.device('cuda',int(os.environ['LOCAL_RANK']));torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    torch.set_float32_matmul_precision('highest')
    dist.init_process_group('nccl',device_id=device)
    torch.manual_seed(261105451+rank);torch.cuda.manual_seed(261105451+rank)
    verify_conditionals()
    aux=RFIDPreservedJointTokenizer(BASE/'inputs/stage1-model.pt',BASE/'inputs/joint-codebook.pt',
                                   BASE/'inputs/reconstruction-codebook.pt').to(device).eval().requires_grad_(False)
    before=module_sha(aux)
    config=load_config(BASE/'inputs/stage2-training-config.yaml')
    split='train' if args.phase=='calibrate' else 'val'
    dataset=TarImageNet(BASE/'data',split,create_transforms(config.dataset,split='val'))
    if args.phase=='calibrate':
        indices=np.random.default_rng(261105451).choice(len(dataset),2048,replace=False).tolist()
        if rank==0:write(OUT/'calibration-indices.json',dict(indices=indices,split='train',seed=261105451))
    else:indices=list(range(50000));assert len(dataset)==50000
    loader=DataLoader(Subset(dataset,indices[rank::world]),batch_size=args.batch_size,
        num_workers=4,pin_memory=True,persistent_workers=True,prefetch_factor=1,
        multiprocessing_context='spawn')
    collected=[];seen=0;started=time.monotonic();alignment_proof=None
    if args.phase=='full':
        calibration=json.loads((OUT/'calibration.json').read_text())
        temperature=calibration['sequence_temperature']
        metrics.INCEPTION_PATH=BASE/'upstream/rqvae/metrics/inception.py'
        inception=metrics.OriginalRQVAEInception().to(device).eval().requires_grad_(False)
        accum={name:metrics.DistributedOriginalRQVAEMetrics(device,inception=inception) for name in ['canonical','teacher']}
        sums=torch.zeros(7,dtype=torch.float64,device=device)
    for batch,(images,_) in enumerate(loader):
        images=images.to(device,non_blocking=True)
        latent=aux.encode(images)
        pool=beam_pool(aux.physical_book,latent)
        row=torch.arange(len(pool['costs']),device=device)
        canonical=pool['candidates'][row,pool['canonical_index']].reshape(*latent.shape[:-1],4)
        if batch==0:
            original=aux.quantize_reconstruction(latent[:2])['codes']
            assert torch.equal(canonical[:2],original), 'Canonical teacher differs from measured beam16 encoder'
            assert not any(p.requires_grad for p in aux.parameters())
            if args.phase=='calibrate':
                small=beam_pool(aux.physical_book,latent[:2])
                weights=posterior(small,.005)
                chosen=torch.multinomial(weights,1).squeeze(-1)
                codes=small['candidates'][torch.arange(len(chosen),device=device),chosen]
                targets=conditional_targets(small['candidates'],weights,codes,16384)
                torch.testing.assert_close(targets.sum(-1),torch.ones_like(targets[...,0]),rtol=2e-5,atol=2e-5)
                conditionals=targets.gather(-1,codes[...,None]).squeeze(-1).prod(-1)
                torch.testing.assert_close(conditionals,weights.gather(1,chosen[:,None]).squeeze(-1),rtol=2e-4,atol=2e-7)
                zero=posterior(small,0)
                assert torch.equal(zero.argmax(-1),small['canonical_index'])
                native=aux.quantizer.embed_code(canonical[:2])
                physical=aux.physical_book.embed(canonical[:2]).sum(-2)
                torch.testing.assert_close(native,physical,rtol=1e-6,atol=1e-6)
                alignment_proof=dict(passed=True,canonical_codes_equal_saved_beam16=True,
                    zero_temperature_reproduces_measured_encoder=True,
                    exact_conditionals_of_same_sequence_distribution=True,
                    sampled_prefix_joint_probability_verified=True,
                    native_physical_decoding_equal=True,frozen_tokenizer=True,
                    conditional_probability_unit_test=verify_conditionals())
        if args.phase=='calibrate':
            collected.append({k:v.cpu() for k,v in pool.items() if k in ['costs','canonical_index','valid']})
        else:
            weights=posterior(pool,temperature)
            sampled=torch.multinomial(weights,1).squeeze(-1)
            codes=pool['candidates'][row,sampled].reshape_as(canonical)
            real=images.mul(.5).add(.5).clamp(0,1)
            features,_=inception(real);features=features.double()
            total=features.sum(0);cross=features.T@features
            for name,encoding in [('canonical',canonical),('teacher',codes)]:
                fake=aux.decode_code(encoding).float().mul(.5).add(.5).clamp(0,1)
                metric=accum[name];metric.real_sum.add_(total);metric.real_cross.add_(cross);metric.real_count.add_(len(images))
                metric.update(fake,real=False)
                offset=1 if name=='canonical' else 3
                sums[offset]+=(fake-real).square().mean((1,2,3)).sum()
                sums[offset+1]+=(latent-aux.quantizer.embed_code(encoding)).square().mean((1,2,3)).sum()
            sums[0]+=len(images)
            sums[5]+=-(weights*weights.clamp_min(1e-30).log()).sum()
            sums[6]+=len(weights)
        seen+=len(images)
        if rank==0 and (batch==0 or batch%16==0):
            progress=dict(phase=args.phase,images_on_rank=seen,target_on_rank=len(indices[rank::world]),
                elapsed_seconds=time.monotonic()-started,time=time.time())
            write(OUT/'status.json',progress);print(json.dumps(progress),flush=True)
    assert module_sha(aux)==before
    if args.phase=='calibrate':
        local={k:torch.cat([v[k] for v in collected]) for k in collected[0]}
        gathered=[None]*world;dist.all_gather_object(gathered,local)
        if rank==0:
            combined={k:torch.cat([v[k] for v in gathered]) for k in local}
            calibration=temperature_calibration(combined)
            assert calibration['sequence_entropy']>0 and calibration['mean_relative_latent_error_above_canonical']<=.00101
            write(OUT/'calibration.json',calibration)
            write(OUT/'alignment-verification.json',alignment_proof)
            recipe=OmegaConf.load(BASE/'inputs/recipe.yaml')
            recipe.targets.sequence_temperature=calibration['sequence_temperature']
            recipe.targets.calibration='2048 training images; sequence entropy and latent error budget; no validation'
            OmegaConf.save(recipe,BASE/'inputs/recipe.yaml')
            print(json.dumps(calibration),flush=True)
    else:
        results={}
        for name,metric in accum.items():
            fid,_,_=metric.compute();results[name]=dict(rfid=float(fid),images=50000)
        dist.all_reduce(sums)
        if rank==0:
            for name,offset in [('canonical',1),('teacher',3)]:
                results[name].update(pixel_mse=float(sums[offset]/sums[0]),latent_mse=float(sums[offset+1]/sums[0]))
            canonical_fid=results['canonical']['rfid'];teacher_fid=results['teacher']['rfid']
            passed=canonical_fid<4.60 and teacher_fid<4.60 and teacher_fid-canonical_fid<=.05
            report=dict(passed=passed,results=results,calibration=calibration,
                reference_canonical_rfid=4.5122860099749005,validation_images=50000,
                teacher_minus_canonical_rfid=teacher_fid-canonical_fid,
                actual_mean_sequence_entropy=float(sums[5]/sums[6]),
                teacher='beam16 sequence posterior with exact prefix conditionals',
                physical_dictionary_coefficients_and_CNN_unchanged=True,
                validation_used_for_calibration=False,stochastic_seed=261105451,
                gate=dict(canonical_rfid_below=4.60,teacher_rfid_below=4.60,max_teacher_delta=.05))
            write(OUT/'full-results.json',report)
            print(json.dumps(report),flush=True)
            if passed:
                recipe=OmegaConf.load(BASE/'inputs/recipe.yaml')
                recipe.tokenizer.measured_rfid=canonical_fid
                recipe.tokenizer.teacher_sampled_rfid=teacher_fid
                OmegaConf.save(recipe,BASE/'inputs/recipe.yaml')
                for n in ['calibration.json','alignment-verification.json','full-results.json']:
                    (BASE/'inputs'/n).write_bytes((OUT/n).read_bytes())
                (PUBLIC/'recipe.yaml').write_bytes((BASE/'inputs/recipe.yaml').read_bytes())
            assert passed,'Aligned teacher failed reconstruction gate; training must not start'
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['calibrate','full'],required=True)
    parser.add_argument('--batch-size',type=int,default=16)
    main(parser.parse_args())
