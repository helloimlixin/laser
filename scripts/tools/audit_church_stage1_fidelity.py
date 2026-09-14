#!/usr/bin/env python3
"""Matched reconstruction screen of the published and locally finetuned RQ-VAEs."""
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
UPSTREAM=ROOT/'outputs/church-rq-baseline-scratch-20260912/upstream-source'
sys.path[:0]=[str(UPSTREAM),str(ROOT)]
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from torchvision.utils import save_image
from rqvae.img_datasets.lsun import LSUNClass
from rqvae.img_datasets.transforms import create_transforms
from rqvae.metrics.fid import get_inception_model,mean_covar_numpy,frechet_distance
from src.original_rq_training import atomic_json,file_sha256,state_sha256
from audit_published_church_rq import load_frozen


@torch.inference_mode()
def main():
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=True
    device='cuda:0'
    out=ROOT/'outputs/church-published-audit-20260913/tokenizer-screen4096'
    out.mkdir(parents=True,exist_ok=False)
    started=time.time()
    pair=Path('/workspace/tmp/original-rqvae-church-published/extracted/church')
    local=json.loads((ROOT/'outputs/church-rq-baseline-scratch-20260912/stage1/complete.json').read_text())
    specs={'published':(pair/'stage1/model.pt',pair/'stage1/config.yaml'),
           'local_one_epoch_lr4e5':(Path(local['checkpoint']),Path(local['config']))}
    models={};configs={};hashes={}
    for name,(checkpoint,config) in specs.items():
        models[name],configs[name]=load_frozen(checkpoint,config,device)
        hashes[name]=state_sha256(models[name])
    inception=get_inception_model().eval().requires_grad_(False).to(device)
    dataset=LSUNClass('/tmp/laser-sign-data','church',create_transforms(configs['published'].dataset,split='train'))
    loader=DataLoader(Subset(dataset,range(4096)),batch_size=16,num_workers=4,pin_memory=True)
    original_features_path=ROOT/'outputs/church-compact-scaled-rq-20260913/reconstruction/original-features.npy'
    original_features=np.load(original_features_path)
    assert original_features.shape==(4096,2048)
    features={name:np.empty((4096,2048),dtype=np.float32) for name in models}
    squared_error={name:0. for name in models}
    offset=0
    original_check=None
    for images,_ in loader:
        images=images.to(device)
        original=images.mul(.5).add(.5).clamp(0,1)
        grid=[original[:8]]
        if offset==0:
            fresh=inception(original).cpu()
            cached=torch.from_numpy(original_features[:len(images)])
            torch.testing.assert_close(fresh,cached,atol=1e-3,rtol=1e-3)
            original_check=float((fresh-cached).abs().max())
        for name,model in models.items():
            z=model.encode(images)
            quantized,_,_=model.quantizer(z)
            recon=model.decode(quantized).mul(.5).add(.5).clamp(0,1)
            assert torch.isfinite(recon).all()
            squared_error[name]+=float((original-recon).square().mean())*len(images)
            features[name][offset:offset+len(images)]=inception(recon).cpu().numpy()
            if offset==0:grid.append(recon[:8])
        if offset==0:save_image(torch.cat(grid),out/'original-published-local.png',nrow=8)
        offset+=len(images)
        if offset%512==0:
            record=dict(phase='reconstructing',completed=offset,total=4096,elapsed_seconds=time.time()-started)
            atomic_json(out/'status.json',record)
            print(json.dumps(record),flush=True)
    dataset.env.close()
    orig_mu,orig_cov=mean_covar_numpy(original_features)
    reference=np.load(ROOT/'outputs/lsun-church-bar-20260910/assets/lsun_256_church.npz')
    results={}
    for name,values in features.items():
        assert state_sha256(models[name])==hashes[name]
        np.save(out/f'{name}-features.npy',values)
        mu,cov=mean_covar_numpy(values)
        mse=squared_error[name]/4096
        results[name]=dict(matched_rfid4096=float(frechet_distance(mu,cov,orig_mu,orig_cov)),
            reference_rfid4096=float(frechet_distance(mu,cov,reference['mu'],reference['sigma'])),
            pixel_mse=mse,psnr_from_mean_mse=float(-10*np.log10(mse)),
            checkpoint=str(specs[name][0]),checkpoint_sha256=file_sha256(specs[name][0]))
    report=dict(images=4096,indices='training 0..4095',results=results,
        original_features=str(original_features_path),original_features_sha256=file_sha256(original_features_path),
        first16_fresh_original_features_max_abs_error=original_check,tokenizers_unchanged=True,
        precision='FP32, TF32 disabled',elapsed_seconds=time.time()-started,
        script_sha256=file_sha256(__file__))
    atomic_json(out/'result.json',report)
    atomic_json(out/'status.json',dict(phase='complete',elapsed_seconds=time.time()-started))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
