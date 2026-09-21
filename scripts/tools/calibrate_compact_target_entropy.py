#!/usr/bin/env python3
"""Calibrate residual-depth entropy against a reference tokenizer on matching pixels."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.training.stochastic_targets import (TARGET_POLICY_VERSION, calibrate_depth_temperatures,
    install_compact_target_policy)
import src.training.stochastic_targets as target_implementation
bootstrap = argparse.ArgumentParser(add_help=False)
bootstrap.add_argument('--pipeline-dir', type=Path, required=True)
BASE = bootstrap.parse_known_args()[0].pipeline_dir.resolve()
SNAPSHOT = BASE / 'stage2-source'
UPSTREAM = SNAPSHOT / 'outputs/church-rq-baseline-scratch-20260912/upstream-source'
sys.path[:0] = [str(UPSTREAM), str(SNAPSHOT), str(ROOT)]
import src
src.__path__ = [str(SNAPSHOT / 'src')]
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from rqvae.img_datasets.lsun import LSUNClass
from rqvae.img_datasets.transforms import create_transforms
from src.compact_rq_training import FrozenCompactTokenizer
from src.original_rq_training import atomic_json, file_sha256, load_stage2_config, load_tokenizer, state_sha256


@torch.inference_mode()
def measure(tokenizer, latents, temperature, seed):
    depth = int(tokenizer.code_shape[-1])
    sums = torch.zeros(3 + 2*depth, device=latents.device, dtype=torch.float64)
    with torch.random.fork_rng(devices=[latents.device.index]):
        torch.manual_seed(seed)
        for z in latents.split(2):
            hard, _, _ = tokenizer.quantizer(z)
            p, codes = tokenizer.quantizer.get_soft_codes(z, temp=temperature, stochastic=True)
            sampled = tokenizer.get_code_emb_with_depth(codes)[0].sum(-2)
            sums[0] += (z-hard).double().square().sum()
            sums[1] += (z-sampled).double().square().sum()
            sums[2] += (sampled-hard).double().square().sum()
            entropy = -(p*p.clamp_min(1e-30).log()).sum(-1).reshape(-1,depth)
            maximum = p.amax(-1).reshape(-1,depth)
            sums[3:3+depth] += entropy.double().sum(0)
            sums[3+depth:] += (1-maximum).double().sum(0)
    nsites = latents.numel() // latents.shape[-1]
    values = sums.cpu().tolist()
    return dict(temperature=temperature, seed=seed, images=len(latents),
                hard_latent_mse=values[0]/latents.numel(), sampled_latent_mse=values[1]/latents.numel(),
                sampled_to_hard_residual_mse_ratio=values[1]/values[0],
                sampled_minus_hard_latent_mse=values[2]/latents.numel(),
                entropy_nats_per_depth=[v/nsites for v in values[3:3+depth]],
                expected_non_argmax_fraction_per_depth=[v/nsites for v in values[3+depth:]])


@torch.inference_mode()
def pixel_check(tokenizer, latents, originals, temperature, seed):
    mse = perturbation = 0.
    with torch.random.fork_rng(devices=[latents.device.index]):
        torch.manual_seed(seed)
        for start in range(0,len(latents),4):
            z=latents[start:start+4]
            _,codes=tokenizer.quantizer.get_soft_codes(z,temp=temperature,stochastic=True)
            decoded=tokenizer.decode_code(codes).mul(.5).add(.5).clamp(0,1)
            hard,_,_=tokenizer.quantizer(z)
            hard_decoded=tokenizer.decode(hard).mul(.5).add(.5).clamp(0,1)
            real=originals[start:start+4].to(z.device).mul(.5).add(.5)
            mse+=(decoded-real).double().square().sum().item()
            perturbation+=(decoded-hard_decoded).double().square().sum().item()
    mse/=originals.numel();perturbation/=originals.numel()
    return dict(images=len(latents),pixel_mse=mse,psnr_db=-10*math.log10(mse),
                pixel_mse_relative_to_hard_reconstruction=perturbation,
                metric='paired reconstruction diagnostic; not FID')


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pipeline-dir',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--reference-checkpoint',type=Path,required=True)
    p.add_argument('--reference-config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=Path('/tmp/laser-sign-data'))
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--images',type=int,default=128)
    p.add_argument('--check-images',type=int,default=256)
    p.add_argument('--decode-images',type=int,default=32)
    p.add_argument('--replicates',type=int,default=3)
    args=p.parse_args()
    if min(args.images,args.check_images,args.replicates)<1 or not 0<=args.decode_images<=args.check_images:
        p.error('Positive calibration/check counts and valid decode count required')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.time()
    device=torch.device(args.device);torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.08,device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    def status(phase,**values):
        row=dict(phase=phase,elapsed_seconds=time.time()-started,**values)
        atomic_json(out/'status.json',row);print(json.dumps(row),flush=True)
    manifest=json.loads((BASE/'stage2-source-manifest.json').read_text())
    for path,digest in manifest.items():assert file_sha256(SNAPSHOT/path)==digest,path
    cache=json.loads((args.cache/'complete.json').read_text())
    for key in ['checkpoint','codebook']:assert file_sha256(cache[key])==cache[key+'_sha256']
    assert cache['data_protocol_sha256']==file_sha256(BASE/'reference/data-protocol.json')
    tokenizer=FrozenCompactTokenizer(cache['checkpoint'],cache['codebook']).to(device).eval()
    before=state_sha256(tokenizer);assert before==cache['frozen_state_sha256']
    install_compact_target_policy(tokenizer.quantizer)
    reference,reference_config=load_tokenizer(args.reference_checkpoint,args.reference_config,device)
    reference_before=state_sha256(reference)
    config=load_stage2_config(UPSTREAM)
    dataset=LSUNClass(str(args.data_root),'church',create_transforms(config.dataset,split='train'))
    assert len(dataset)==cache['images']
    indices=np.random.default_rng(20260915).choice(len(dataset),args.images+args.check_images,replace=False)
    data=DataLoader(Subset(dataset,indices.tolist()),batch_size=8,num_workers=4)
    originals=[];reference_z=[]
    for xs,_ in data:
        originals.append(xs)
        reference_z.append(reference.encode(xs.to(device)).cpu())
    originals=torch.cat(originals);reference_z=torch.cat(reference_z).to(device)
    array=np.load(cache['latent_cache'],mmap_mode='r')
    latents=torch.from_numpy(np.array(array[indices],copy=True)).to(device)
    fresh=tokenizer.encode(originals[:4].to(device))
    torch.testing.assert_close(fresh,latents[:4],atol=2e-5,rtol=2e-5)
    status('reference_encoded',images=len(originals))
    reference_measures=[measure(reference,reference_z[:args.images],.5,99200+i) for i in range(args.replicates)]
    goals=np.mean([r['entropy_nats_per_depth'] for r in reference_measures],axis=0).tolist()
    status('calibrating_depth_temperatures',target_entropy=goals)
    fitted=calibrate_depth_temperatures(tokenizer.quantizer,latents[:args.images],goals,seed=99200)
    temperature=fitted['selected_temperature']
    status('depth_temperatures_fitted',temperature=temperature)
    checks={}
    for name,temp in [('baseline',.125),('entropy_matched',temperature)]:
        checks[name]=[measure(tokenizer,latents[args.images:],temp,100200+i) for i in range(args.replicates)]
        status('checking_distortion',setting=name,entropy=checks[name][0]['entropy_nats_per_depth'])
    checks['reference']=[measure(reference,reference_z[args.images:],.5,100200+i) for i in range(args.replicates)]
    pixels={}
    if args.decode_images:
        for name,temp in [('baseline',.125),('entropy_matched',temperature)]:
            pixels[name]=pixel_check(tokenizer,latents[args.images:args.images+args.decode_images],
                originals[args.images:args.images+args.decode_images],temp,110200)
    assert state_sha256(tokenizer)==before and state_sha256(reference)==reference_before
    report=dict(**fitted,source_checkpoint_sha256=cache['checkpoint_sha256'],
        codebook_sha256=cache['codebook_sha256'],data_protocol_sha256=cache['data_protocol_sha256'],
        target_implementation_sha256=file_sha256(target_implementation.__file__),
        reference_checkpoint=str(args.reference_checkpoint.resolve()),reference_checkpoint_sha256=file_sha256(args.reference_checkpoint),
        reference_config=str(args.reference_config.resolve()),reference_config_sha256=file_sha256(args.reference_config),
        reference_temperature=.5,reference_calibration=reference_measures,
        calibration_indices=indices[:args.images].tolist(),check_indices=indices[args.images:].tolist(),
        checks=checks,pixel_checks=pixels,images=args.images,check_images=args.check_images,replicates=args.replicates,
        selection_rule='Match reference mean target entropy per depth on sampled residual prefixes; distortion measured independently',
        smoke_only=False,frozen_state_unchanged=True,source_manifest=manifest,
        script_sha256=file_sha256(__file__),elapsed_seconds=time.time()-started,
        precision='FP32 geometry/decoder; TF32 disabled',
        data_transform='Original RQ-VAE loader and deterministic resize/center crop; same pixels for both tokenizers')
    atomic_json(out/'calibration.json',report)
    (out/'calibration-script.py').write_text(Path(__file__).read_text())
    status('complete',selected_temperature=temperature,output=str(out/'calibration.json'))


if __name__=='__main__':
    main()
