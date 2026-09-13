#!/usr/bin/env python3
"""Fit a smaller atom-specific scaled book using frozen training latents."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.scaled_atom_rq import ScaledAtomRQ
from src.adaptive_scaled_atom_rq import AdaptiveScaledAtomRQ,fit_atom_levels


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fit-images',type=int,default=4096)
    parser.add_argument('--screen-images',type=int,default=1024)
    parser.add_argument('--passes',type=int,default=4)
    parser.add_argument('--levels',type=int,choices=[2,4],default=4)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    started=time.time()
    def status(value):
        row=dict(elapsed_seconds=time.time()-started,**value)
        (args.output/'status.json').write_text(json.dumps(row,indent=2)+'\n')
        print(json.dumps(row),flush=True)
    cache_path=ROOT/'outputs/church-scaled-atom-stage2-20260913/cache/latents-fp32.npy'
    source=ROOT/'outputs/church-scaled-atom-rq-20260912/sweep/scaled-atom-codebooks.pt'
    spec=torch.load(source,map_location='cpu',weights_only=True)
    dictionary=spec['dictionary'].to(args.device)
    values=np.load(cache_path,mmap_mode='r')
    calibration=np.array(values[64000:64000+args.fit_images],copy=True)
    assert len(calibration)==args.fit_images and args.screen_images<64000
    quantizer=AdaptiveScaledAtomRQ(dictionary,spec['levels'][str(args.levels)].to(args.device).expand(16384,-1))
    status(dict(phase='fitting',images=args.fit_images,passes=args.passes))
    trace=fit_atom_levels(quantizer,calibration,passes=args.passes,batch_size=16,prior_weight=4.,
                         callback=lambda row:status(dict(phase='fitting',**row)))
    compact=dict(format_version=1,kind='adaptive_scaled_atom_rq',dictionary=quantizer.dictionary.cpu(),
        levels=quantizer.levels.cpu(),depth=4,zero_token=0,code_shape=[8,8,4],
        source_stage1_checkpoint=spec['source_stage1_checkpoint'],
        source_stage1_checkpoint_sha256=spec['source_stage1_checkpoint_sha256'])
    torch.save(compact,args.output/'compact-codebook.pt')
    models=dict(shared8=ScaledAtomRQ(dictionary,spec['levels']['8'].to(args.device)),
                shared4=ScaledAtomRQ(dictionary,spec['levels']['4'].to(args.device)))
    models[f'adaptive{args.levels}']=quantizer
    metrics={}
    for name,model in models.items():
        total=energy=0.
        for start in range(0,args.screen_images,16):
            z=torch.tensor(values[start:min(start+16,args.screen_images)].copy(),device=args.device)
            reconstruction=model.quantize(z)['quantized']
            total+=(z-reconstruction).square().sum().item()
            energy+=z.square().sum().item()
        metrics[name]=dict(latent_mse=total/(args.screen_images*8*8*256),relative_latent_mse=total/energy,
                           vocab_size=model.vocab_size)
        status(dict(phase='latent_screen',variant=name,**metrics[name]))
    report=dict(fit_images=args.fit_images,fit_index_start=64000,fit_index_end_exclusive=64000+args.fit_images,
        screen_images=args.screen_images,screen_index_start=0,passes=args.passes,prior_weight=4.,levels=args.levels,
        source_checkpoint_sha256=spec['source_stage1_checkpoint_sha256'],latent_cache=str(cache_path),
        all_backbone_weights_frozen=True,atom_directions_unchanged=True,code_shape=[8,8,4],
        trace=trace,metrics=metrics,elapsed_seconds=time.time()-started,
        source_hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__),ROOT/'src/adaptive_scaled_atom_rq.py')})
    (args.output/'fit-report.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'fit-source.py').write_text(Path(__file__).read_text())
    status(dict(phase='complete',metrics=metrics))


if __name__=='__main__':
    main()
