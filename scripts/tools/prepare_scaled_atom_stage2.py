#!/usr/bin/env python3
"""Cache frozen FP32 LASER Church latents for the expanded RQ prior."""
import argparse
from datetime import timedelta
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
UPSTREAM=ROOT/'outputs/church-rq-baseline-scratch-20260912/upstream-source'
sys.path[:0]=[str(UPSTREAM),str(ROOT)]

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from rqvae.img_datasets.lsun import LSUNClass
from rqvae.img_datasets.transforms import create_transforms
from src.original_rq_training import (load_stage2_config,atomic_json,file_sha256,
    state_sha256,IndexedImages,validation_latents)
from src.scaled_atom_training import FrozenScaledTokenizer


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=64)
    args=p.parse_args()
    rank,world=int(os.environ['RANK']),int(os.environ['WORLD_SIZE'])
    device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=True
    dist.init_process_group('nccl',timeout=timedelta(hours=1))
    out=args.output.resolve()
    if rank==0:
        out.mkdir(parents=True,exist_ok=False)
    dist.barrier()
    checkpoint=ROOT/'outputs/lsun-church-sign-pattern-20260910/assets/best_rfid_slot1_model.pt'
    codebook=ROOT/'outputs/church-scaled-atom-rq-20260912/sweep/scaled-atom-codebooks.pt'
    ckpt_hash=file_sha256(checkpoint)
    book_hash=file_sha256(codebook)
    assert ckpt_hash=='93f19adc8ca2cfae0cf629d48d2d6f5f7039440b1a48ad1c7ba43cf17932a388'
    assert book_hash=='275bc44c7eed0b1b7f14308e49b1031ec8b4b0258a2dcb91ac2bc69ec6b8a1f1'
    tokenizer=FrozenScaledTokenizer(checkpoint,codebook).to(device).eval()
    before=state_sha256(tokenizer)
    config=load_stage2_config(UPSTREAM)
    dataset=LSUNClass('/tmp/laser-sign-data','church',create_transforms(config.dataset,split='train'))
    assert len(dataset)==126227
    path=out/'latents-fp32.npy'
    if rank==0:
        values=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(dataset),8,8,256))
        del values
    dist.barrier()
    values=np.load(path,mmap_mode='r+')
    indices=list(range(rank,len(dataset),world))
    loader=DataLoader(Subset(IndexedImages(dataset),indices),batch_size=args.batch_size,
                      num_workers=8,pin_memory=True,shuffle=False)
    started=time.time()
    seen=0
    for batch,(images,index) in enumerate(loader):
        z=tokenizer.encode(images.to(device,non_blocking=True))
        assert z.dtype==torch.float32 and torch.isfinite(z).all()
        values[index.numpy()]=z.cpu().numpy()
        seen+=len(images)
        if rank==0 and batch%20==0:
            status=dict(phase='caching_frozen_latents',images_per_rank=seen,
                total_images=len(dataset),elapsed_seconds=time.time()-started,updated_unix=time.time())
            atomic_json(out/'status.json',status)
            print(status,flush=True)
    values.flush()
    del values
    dataset.env.close()
    heldout=validation_latents(tokenizer,config,device,rank,world)
    torch.save(heldout,out/f'validation-rank{rank}.pt')
    assert before==state_sha256(tokenizer)
    dist.barrier()
    if rank==0:
        receipt=dict(phase='complete',images=len(dataset),dtype='float32',shape=[len(dataset),8,8,256],
            world_size=world,checkpoint=str(checkpoint),checkpoint_sha256=ckpt_hash,
            codebook=str(codebook),codebook_sha256=book_hash,levels=8,vocab_size=131073,
            latent_cache=str(path),cache_sha256=file_sha256(path),
            frozen_state_sha256=before,frozen_state_unchanged=True,
            hard_codes_cached=False,stochastic_targets_recomputed_each_visit=True,
            transform='official Church resize256/center crop/normalize[-1,1]',
            precision='FP32; CUDA matmul and cuDNN TF32 disabled',
            elapsed_seconds=time.time()-started,updated_unix=time.time())
        atomic_json(out/'complete.json',receipt)
        atomic_json(out/'status.json',receipt)
        print(receipt,flush=True)
    dist.destroy_process_group()


if __name__=='__main__':
    main()
