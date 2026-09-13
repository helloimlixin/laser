#!/usr/bin/env python3
"""Measure FP32 cache batch sizes with a bounded pause of one cache worker."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[2]
UPSTREAM=ROOT/'outputs/church-rq-baseline-scratch-20260912/upstream-source'
sys.path[:0]=[str(ROOT),str(UPSTREAM)]
import numpy as np
import torch
from torch.utils.data import DataLoader
from rqvae.img_datasets.transforms import create_transforms
from src.imagenet_scaled_stage2 import ManifestImages,load_imagenet_config
from src.scaled_atom_training import FrozenScaledTokenizer


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--worker-pid',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    base=args.base.resolve()
    command=Path(f'/proc/{args.worker_pid}/cmdline').read_bytes()
    assert str(base).encode() in command and b'train_imagenet_scaled_stage2.py' in command
    torch.set_num_threads(8)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=True
    tokenizer=FrozenScaledTokenizer(base/'assets/best_rfid_slot1_model.pt',base/'scaled-atom-codebook.pt').cuda().eval()
    manifest=json.loads((base/'train-manifest.json').read_text())
    dataset=ManifestImages('/workspace/Projects/data/imagenet2012/train',manifest,
        create_transforms(load_imagenet_config(UPSTREAM).dataset,split='train'),view=1,seed=421,
        indices=range(256))
    images=torch.cat([batch for batch,_,_ in DataLoader(dataset,batch_size=64,num_workers=4)]).cuda()
    rows=[]
    timer=None
    pause_started=time.time()
    def resume_worker():
        os.kill(args.worker_pid,signal.SIGCONT)
    os.kill(args.worker_pid,signal.SIGSTOP)
    try:
        timer=threading.Timer(45,resume_worker)
        timer.start()
        # Let already queued production kernels finish before timing.
        time.sleep(.5)
        def encode(batch):
            z=tokenizer.encode(batch)
            codes=tokenizer.quantizer.quantize(z)['codes']
            return z,codes
        reference_z,reference_codes=[],[]
        for x in images.split(64):
            z,codes=encode(x);reference_z.append(z);reference_codes.append(codes)
        reference_z,reference_codes=torch.cat(reference_z),torch.cat(reference_codes)
        stored=torch.from_numpy(np.load(base/'cache/train-view1-latents.npy',mmap_mode='r')[:256].copy()).cuda()
        torch.testing.assert_close(reference_z,stored,rtol=2e-5,atol=2e-5)
        for size in [64,128,256]:
            encode(images[:size])
            torch.cuda.synchronize()
            durations=[]
            torch.cuda.reset_peak_memory_stats()
            for _ in range(3):
                start=time.perf_counter()
                zs,codes=[],[]
                for x in images.split(size):
                    z,c=encode(x);zs.append(z);codes.append(c)
                torch.cuda.synchronize()
                durations.append(time.perf_counter()-start)
            z,c=torch.cat(zs),torch.cat(codes)
            row=dict(batch_size=size,images=256,
                images_per_second=256/float(np.median(durations)),seconds=durations,
                max_latent_difference=float((z-reference_z).abs().max()),
                latent_rms_difference=float((z-reference_z).square().mean().sqrt()),
                token_mismatch_fraction=float((c!=reference_codes).float().mean()),
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            rows.append(row);print(json.dumps(row),flush=True)
            if time.time()-pause_started>40:raise RuntimeError('Benchmark exceeded bounded pause')
    finally:
        if timer:timer.cancel()
        resume_worker()
    result=dict(precision='FP32; TF32 disabled',pause_seconds=time.time()-pause_started,
        worker_resumed=True,benchmarks=rows,baseline_matches_persisted_cache=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
