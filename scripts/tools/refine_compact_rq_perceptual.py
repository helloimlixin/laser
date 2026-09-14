#!/usr/bin/env python3
"""Fit only compact coefficient levels using training-image L1 and LPIPS loss."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
UPSTREAM=ROOT/'outputs/church-rq-baseline-scratch-20260912/upstream-source'
sys.path[:0]=[str(ROOT),str(UPSTREAM)]
os.environ.setdefault('TORCH_HOME','/workspace/tmp/official-rqvae-eval-cache')
import numpy as np
import torch
from torch.utils.data import DataLoader
from rqvae.img_datasets.transforms import create_transforms
from rqvae.losses.vqgan.lpips import LPIPS
from src.adaptive_scaled_atom_rq import AdaptiveScaledAtomRQ
from src.scaled_atom_rq import FrozenSparseBackbone
from src.imagenet_scaled_stage2 import ManifestImages,load_imagenet_config
from src.original_rq_training import atomic_json,state_sha256,file_sha256,seed_all


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=ROOT/'outputs/imagenet-rfid421-rq8-refit-20260913')
    p.add_argument('--study',type=Path,default=ROOT/'outputs/imagenet-compact-rq-study-20260913')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--levels',type=int,default=2)
    p.add_argument('--steps',type=int,default=1024)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--lr',type=float,default=.003)
    p.add_argument('--device',default='cuda:2')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    seed_all(913)
    torch.set_num_threads(8)
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=True
    spec=torch.load(args.study/f'compact{args.levels}-codebook.pt',map_location='cpu',weights_only=True)
    fit=json.loads((args.study/f'fit{args.levels}.json').read_text())
    cache=json.loads((args.source/'cache/complete.json').read_text())
    backbone=FrozenSparseBackbone(Path(cache['checkpoint'])).to(args.device).eval()
    before=state_sha256(backbone)
    perceptual=LPIPS().to(args.device).eval().requires_grad_(False)
    q=AdaptiveScaledAtomRQ(backbone.dictionary,spec['levels'].to(args.device))
    prior=q.levels.abs().log()
    log_coefficients=torch.nn.Parameter(prior.clone())
    signs=q.levels.sign()
    optimizer=torch.optim.Adam([log_coefficients],lr=args.lr)
    manifest=json.loads((args.source/'train-manifest.json').read_text())
    dataset=ManifestImages(Path(cache['data'])/'train',manifest,
        create_transforms(load_imagenet_config(UPSTREAM).dataset,split='train'),
        seed=cache['seed'],indices=fit['fit_indices'])
    loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True,num_workers=4,
        persistent_workers=True,pin_memory=True)
    values=np.load(args.source/'cache/train-view0-latents.npy',mmap_mode='r')
    started=time.time()
    iterator=iter(loader)
    history=[]
    for step in range(1,args.steps+1):
        try:images,labels,indices=next(iterator)
        except StopIteration:
            iterator=iter(loader);images,labels,indices=next(iterator)
        images=images.to(args.device)
        z=torch.from_numpy(values[indices.numpy()].copy()).to(args.device)
        if step==1:
            with torch.no_grad():
                torch.testing.assert_close(backbone.encode(images[:2]),z[:2],atol=2e-5,rtol=2e-5)
        levels=(signs*log_coefficients.exp()).sort(-1).values
        with torch.no_grad():
            q.levels.copy_(levels)
            codes=q.quantize(z)['codes']
            packed=(codes-1).clamp_min(0)
            atoms=packed//args.levels
            bins=packed%args.levels
            vectors=q.dictionary.T[atoms]*(codes!=0)[...,None]
        quantized=(vectors*levels[atoms,bins][...,None]).sum(-2)
        decoded=backbone.decode(quantized)
        l1=(decoded-images).abs().mean()
        lpips=perceptual(decoded,images)
        regularization=(log_coefficients-prior).square().mean()
        loss=l1+lpips+.01*regularization
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite coefficient image loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert torch.isfinite(log_coefficients.grad).all()
        optimizer.step()
        with torch.no_grad():
            log_coefficients.clamp_(prior-.5,prior+.5)
        if step==1 or step%32==0:
            row=dict(step=step,loss=float(loss.detach()),l1=float(l1.detach()),lpips=float(lpips.detach()),
                regularization=float(regularization.detach()),elapsed_seconds=time.time()-started)
            history.append(row)
            atomic_json(args.output/'status.json',dict(phase='perceptual_coefficient_fit',**row))
            print(json.dumps(row),flush=True)
        if step in (128,256,512,1024,args.steps):
            folder=args.output/f'step{step:04d}'
            folder.mkdir(exist_ok=True)
            levels=(signs*log_coefficients.detach().exp()).sort(-1).values
            assert (levels[:,1:]>levels[:,:-1]).all()
            torch.save({**spec,'levels':levels.cpu()},folder/f'compact{args.levels}-codebook.pt')
            atomic_json(folder/f'fit{args.levels}.json',{**fit,'method':'training-image L1 plus frozen VGG LPIPS; coefficient parameters only',
                'learning_rate':args.lr,'optimizer_steps':step,'history':history,
                'initial_codebook_sha256':file_sha256(args.study/f'compact{args.levels}-codebook.pt'),
                'source_hashes':{str(Path(__file__).relative_to(ROOT)):file_sha256(__file__)}})
    assert state_sha256(backbone)==before
    atomic_json(args.output/'status.json',dict(phase='complete',steps=args.steps,backbone_unchanged=True,
        elapsed_seconds=time.time()-started))


if __name__=='__main__':main()
