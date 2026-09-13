#!/usr/bin/env python3
"""Compare compact scaled books on identical cached latents and image pixels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
UPSTREAM=ROOT/'outputs/church-rq-baseline-scratch-20260912/upstream-source'
sys.path[:0]=[str(UPSTREAM),str(ROOT),str(ROOT/'scripts/tools')]
os.environ.setdefault('TORCH_HOME','/workspace/tmp/official-rqvae-eval-cache')
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from torchvision import transforms
from rqvae.img_datasets.lsun import LSUNClass
from rqvae.metrics.fid import get_inception_model,frechet_distance
from src.scaled_atom_rq import FrozenSparseBackbone,ScaledAtomRQ
from src.adaptive_scaled_atom_rq import AdaptiveScaledAtomRQ
from evaluate_scaled_atom_rq import make_contact_sheet,write_json,fingerprint,sha256


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--candidates',type=Path,nargs='+',required=True)
    p.add_argument('--images',type=int,default=4096)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--batch-size',type=int,default=16)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=True
    started=time.time()
    def status(phase,**kwargs):
        row=dict(phase=phase,elapsed_seconds=time.time()-started,**kwargs)
        write_json(args.output/'status.json',row)
        print(json.dumps(row),flush=True)
    checkpoint=ROOT/'outputs/lsun-church-sign-pattern-20260910/assets/best_rfid_slot1_model.pt'
    assert sha256(checkpoint)=='93f19adc8ca2cfae0cf629d48d2d6f5f7039440b1a48ad1c7ba43cf17932a388'
    backbone=FrozenSparseBackbone(checkpoint).to(args.device).eval()
    before=fingerprint(backbone)
    shared=torch.load(ROOT/'outputs/church-scaled-atom-rq-20260912/sweep/scaled-atom-codebooks.pt',weights_only=True)
    models={f'shared{b}':ScaledAtomRQ(backbone.dictionary,shared['levels'][str(b)].to(args.device)) for b in (8,4)}
    for path in args.candidates:
        compact=torch.load(path,map_location='cpu',weights_only=True)
        torch.testing.assert_close(compact['dictionary'],shared['dictionary'],rtol=0,atol=0)
        name=f'adaptive{compact["levels"].shape[1]}'
        assert name not in models
        models[name]=AdaptiveScaledAtomRQ(backbone.dictionary,compact['levels'].to(args.device))
    transform=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(256),
        transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)])
    train=LSUNClass('/tmp/laser-sign-data','church',transform)
    if 'church_val' not in LSUNClass.valid_categories:
        LSUNClass.valid_categories.append('church_val')
    validation=LSUNClass('/tmp/laser-sign-data','church_val',transform)
    cache=np.load(ROOT/'outputs/church-scaled-atom-stage2-20260913/cache/latents-fp32.npy',mmap_mode='r')
    inception=get_inception_model().eval().requires_grad_(False).to(args.device)
    reference=np.load(ROOT/'outputs/lsun-church-bar-20260910/assets/lsun_256_church.npz')
    results={}
    cache_parity={}
    for split,dataset,count in [('validation',validation,300),('screen',train,args.images)]:
        fid=split=='screen'
        totals={name:dict(pixel_mse=0.,latent_mse=0.) for name in models}
        features={name:[] for name in ['original',*models]} if fid else {}
        loader=DataLoader(Subset(dataset,list(range(count))),batch_size=args.batch_size,
                          num_workers=4,pin_memory=True,shuffle=False)
        done=0
        for images,_ in loader:
            images=images.to(args.device,non_blocking=True)
            original=(images*.5+.5).clamp(0,1)
            z=(torch.tensor(cache[done:done+len(images)].copy(),device=args.device) if fid
               else backbone.encode(images))
            if fid and done==0:
                fresh=backbone.encode(images)
                torch.testing.assert_close(fresh,z,atol=1e-4,rtol=1e-4)
                cache_parity=dict(images=len(images),max_absolute_latent_difference=(fresh-z).abs().max().item())
            if fid:
                features['original'].append(inception(original).cpu().numpy())
            contact={'original':original.cpu()} if done==0 else None
            for name,quantizer in models.items():
                value=quantizer.quantize(z)['quantized']
                decoded=(backbone.decode(value)*.5+.5).clamp(0,1)
                totals[name]['pixel_mse']+=(original-decoded).square().flatten(1).mean(1).sum().item()
                totals[name]['latent_mse']+=(z-value).square().flatten(1).mean(1).sum().item()
                if fid:
                    features[name].append(inception(decoded).cpu().numpy())
                if contact is not None:
                    contact[name]=decoded.cpu()
            if contact is not None:
                make_contact_sheet(contact,args.output/f'{split}-reconstructions.png')
            done+=len(images)
            if done%256==0 or done==count:
                status('reconstructing',split=split,images=done,total=count)
        metrics={}
        for name,values in totals.items():
            metrics[name]={key:value/count for key,value in values.items()}
            metrics[name].update(vocab_size=models[name].vocab_size,
                psnr_from_mean_mse=float(-10*np.log10(metrics[name]['pixel_mse'])))
        results[split]=dict(images=count,metrics=metrics)
        write_json(args.output/'metrics.json',results)
        if fid:
            statistics={}
            for name,parts in features.items():
                values=np.concatenate(parts)
                np.save(args.output/f'{name}-features.npy',values)
                statistics[name]=(values.mean(0),np.cov(values,rowvar=False))
            for name in models:
                status('computing_fid',variant=name)
                mu,sigma=statistics[name]
                metrics[name]['rfid_matched']=float(frechet_distance(mu,sigma,*statistics['original']))
                metrics[name]['rfid_published_reference']=float(frechet_distance(mu,sigma,reference['mu'],reference['sigma']))
                write_json(args.output/'metrics.json',results)
    after=fingerprint(backbone)
    assert before==after
    report=dict(checkpoint_sha256=sha256(checkpoint),backbone_frozen=True,
        frozen_state_before_sha256=before,frozen_state_after_sha256=after,
        cache_parity=cache_parity,code_shape=[8,8,4],precision='FP32; TF32 disabled',
        screen_indices=[0,args.images],validation_images=300,
        candidate_hashes={str(path):sha256(path) for path in args.candidates},
        source_hashes={str(path.relative_to(ROOT)):sha256(path)
            for path in (Path(__file__),ROOT/'src/adaptive_scaled_atom_rq.py')},
        elapsed_seconds=time.time()-started)
    write_json(args.output/'manifest.json',report)
    status('complete',metrics=results['screen']['metrics'])


if __name__=='__main__':
    main()
