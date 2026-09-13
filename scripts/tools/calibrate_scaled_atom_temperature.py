#!/usr/bin/env python3
"""Match stochastic-target distortion to the released RQ temp=.5 control."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
UPSTREAM=ROOT/'outputs/church-rq-baseline-scratch-20260912/upstream-source'
sys.path[:0]=[str(UPSTREAM),str(ROOT)]
import torch
from torch.utils.data import DataLoader,Subset
from rqvae.img_datasets.lsun import LSUNClass
from rqvae.img_datasets.transforms import create_transforms
from src.original_rq_training import atomic_json,load_tokenizer,load_stage2_config,file_sha256
from src.scaled_atom_training import FrozenScaledTokenizer


@torch.inference_mode()
def measure(quantizer,latents,temperature,seed):
    totals=torch.zeros(7,device=latents.device,dtype=torch.float64)
    for index,z in enumerate(latents.split(2)):
        hard,_,_=quantizer(z)
        torch.manual_seed(seed+index)
        probability,codes=quantizer.get_soft_codes(z,temp=temperature,stochastic=True)
        if hasattr(quantizer,'dictionary'):
            sampled=quantizer.embed(codes).sum(-2)
        else:
            sampled=quantizer.embed_code(codes)
        totals[0]+=(z-hard).square().sum()
        totals[1]+=(z-sampled).square().sum()
        totals[2]+=(sampled-hard).square().sum()
        for d in range(4):
            p=probability[...,d,:]
            totals[3+d]+=(-(p*p.clamp_min(1e-30).log()).sum(-1)).sum()
    total=totals.cpu().tolist()
    return dict(temperature=temperature,hard_latent_mse=total[0]/latents.numel(),
        sampled_latent_mse=total[1]/latents.numel(),
        sampled_to_hard_residual_mse_ratio=total[1]/total[0],
        sampled_minus_hard_latent_mse=total[2]/latents.numel(),
        entropy_nats_per_depth=[x/(len(latents)*64) for x in total[3:]])


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--images',type=int,default=128)
    args=p.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    started=time.time()
    checkpoint=ROOT/'outputs/lsun-church-sign-pattern-20260910/assets/best_rfid_slot1_model.pt'
    codebook=ROOT/'outputs/church-scaled-atom-rq-20260912/sweep/scaled-atom-codebooks.pt'
    scaled=FrozenScaledTokenizer(checkpoint,codebook).to(args.device).eval()
    completed=json.loads((ROOT/'outputs/church-rq-baseline-scratch-20260912/stage1/complete.json').read_text())
    reference,_=load_tokenizer(completed['checkpoint'],completed['config'],args.device)
    config=load_stage2_config(UPSTREAM)
    dataset=LSUNClass('/tmp/laser-sign-data','church',create_transforms(config.dataset,split='train'))
    indices=list(range(62048,62048+args.images))
    loader=DataLoader(Subset(dataset,indices),batch_size=16,num_workers=4)
    original_latents,scaled_latents=[],[]
    for images,_ in loader:
        images=images.to(args.device)
        original_latents.append(reference.encode(images))
        scaled_latents.append(scaled.encode(images))
    original_latents=torch.cat(original_latents)
    scaled_latents=torch.cat(scaled_latents)
    control=measure(reference.quantizer,original_latents,.5,99200)
    print(json.dumps(dict(kind='original_rq_reference',**control)),flush=True)
    results=[]
    limit=control['sampled_to_hard_residual_mse_ratio']+.02
    for temperature in [.5,.25,.125,.0625,.03125,.015625]:
        result=measure(scaled.quantizer,scaled_latents,temperature,99200)
        results.append(result)
        print(json.dumps(dict(kind='scaled_rq8',**result)),flush=True)
        if result['sampled_to_hard_residual_mse_ratio']<=limit:
            break
    chosen=next((row['temperature'] for row in results if row['sampled_to_hard_residual_mse_ratio']<=limit),None)
    if chosen is None:
        raise RuntimeError('No tested temperature meets the original-RQ distortion control')
    report=dict(selected_temperature=chosen,calibration_indices=indices,images=args.images,
        selection_rule='Largest tested temperature <=0.5 whose sampled/hard final latent MSE ratio is no greater than the original RQ temp=.5 ratio plus .02',
        original_rq_control=control,scaled_rq_sweep=results,allowed_mse_ratio=limit,
        source_checkpoint_sha256=file_sha256(checkpoint),codebook_sha256=file_sha256(codebook),
        control_checkpoint=completed['checkpoint'],control_checkpoint_sha256=completed['checkpoint_sha256'],
        precision='FP32; TF32 disabled',noise='Exact RQ stochastic codeword sampling, no additive physical coefficient noise',
        elapsed_seconds=time.time()-started)
    atomic_json(args.output,report)
    print(json.dumps({'selected_temperature':chosen,'output':str(args.output)}),flush=True)


if __name__=='__main__':
    main()
