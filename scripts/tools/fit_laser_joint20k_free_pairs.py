#!/usr/bin/env python3
"""Improve reconstruction at 20,481 pair IDs with frozen encoder and decoder."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path('/workspace/Projects/laser')
SOURCE=Path('/tmp/laser-rq-joint20k-imagenet-20261017')
BASE=Path('/tmp/laser-joint20k-freepairs-rfid-20261017')
OUT=ROOT/'outputs/imagenet-laser-joint20k-freepairs-rfid-20261017'
PRIOR=ROOT/'outputs/imagenet-laser-adaptive-joint20k-vocabulary-20261017'
CACHE=Path('/tmp/laser-rq-joint-vocabulary-study-20261015')
FIT=Path('/tmp/laser-compact-joint-vocabulary-20261016/fit-latents.npy')
os.environ['RQ_UPSTREAM_ROOT']=str(SOURCE/'upstream')
os.environ.setdefault('TORCH_HOME',str(SOURCE/'torch-cache'))
sys.path[:0]=[str(ROOT),str(SOURCE/'upstream'),str(ROOT/'runtime')]
import numpy as np
import torch
from src.models.laser_joint_tokenizer import JointAtomCoefficientRQ,FrozenJointLaserTokenizer
from src.training.laser_pair_prototype_fit import untie_pair_directions,refit_pair_prototypes


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(name,**values):
    p=OUT/name;tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(dict(values,time=time.time()),indent=2,allow_nan=False)+'\n');tmp.replace(p)


def progress(phase,**values):
    write('progress.json',phase=phase,**values);print(json.dumps(dict(phase=phase,**values)),flush=True)


def build(path,device):
    spec=torch.load(path,map_location='cpu',weights_only=True)
    return JointAtomCoefficientRQ(spec['dictionary'],spec['atom_ids'],spec['coefficients']).to(device)


@torch.inference_mode()
def fit(args,device):
    calibration=json.loads((PRIOR/'calibration-data.json').read_text())
    selected=json.loads((PRIOR/'selected.json').read_text())
    assert sha(FIT)==calibration['fit_latents_sha256']
    assert sha(CACHE/'temperature-latents.npy')==calibration['heldout_latents_sha256']
    path=Path(selected['codebook_path']);assert sha(path)==selected['codebook_sha256']
    original=build(path,device);q=untie_pair_directions(original)
    for d in range(4):torch.testing.assert_close(q.expanded_codebook(d),original.expanded_codebook(d),rtol=0,atol=0)
    spec=torch.load(path,map_location='cpu',weights_only=True)
    x=torch.from_numpy(np.load(FIT).reshape(-1,256)).to(device)
    held=torch.from_numpy(np.load(CACHE/'temperature-latents.npy').reshape(-1,256)).to(device)
    original_mse=float((held-original.quantize(held,chunk_size=1024)['quantized']).square().mean())
    del original
    best,best_error,trace=None,float('inf'),[]
    for iteration in range(args.passes+1):
        result=q.quantize(x,chunk_size=1024)
        error=float((x-result['quantized']).square().mean())
        held_error=float((held-q.quantize(held,chunk_size=1024)['quantized']).square().mean())
        if held_error<best_error:
            best_error=held_error
            best={k:getattr(q,k).cpu().clone() for k in ['dictionary','atom_ids','coefficients']}
        row=dict(iteration=iteration,training_latent_mse=error,heldout_training_latent_mse=held_error)
        if iteration in [6,12,args.passes]:
            name=f'freepairs-pass{iteration:02d}'
            book=dict(spec,**best,pair_directions_untied=True,one_direction_per_pair_id=True,
                source_pair_codebook_sha256=selected['codebook_sha256'],vocabulary=20481,
                fitting_split='train',fitting_views=8192,encoder_decoder_frozen=True,
                coefficient_gain=1.,selection='heldout training latent MSE',
                fitting_iteration=iteration,best_heldout_mse=best_error)
            target=OUT/(name+'-codebook.pt');assert not target.exists()
            torch.save(book,target)
            if iteration==args.passes:
                torch.save(dict(book,coefficients=book['coefficients']*1.05,coefficient_gain=1.05),
                    OUT/(name+'-gain105-codebook.pt'))
        if iteration<args.passes:row.update(refit_pair_prototypes(q,x,result['codes'],cg_iterations=args.cg_iterations))
        trace.append(row);progress('fitting_free_pair_vectors',**row)
        write('fit-report.json',completed=iteration==args.passes,trace=trace,best_heldout_mse=best_error,
            baseline_heldout_mse=original_mse,baseline_full_rfid=4.689409482068868,
            vocabulary=20481,encoder_decoder_frozen=True,one_direction_per_pair_id=True,
            source_codebook_sha256=selected['codebook_sha256'],training_only_fitting=True,
            full_50000_image_rfid_required=True,peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30)
        del result


@torch.inference_mode()
def screen(args,device):
    from src import rqvae_metrics as metrics
    metrics.INCEPTION_PATH=SOURCE/'upstream/rqvae/metrics/inception.py'
    network=metrics.OriginalRQVAEInception().eval().requires_grad_(False).to(device)
    aux=FrozenJointLaserTokenizer(SOURCE/'inputs/stage1-model.pt',SOURCE/'inputs/joint-codebook.pt').to(device)
    real=np.load(CACHE/'screen-real-features.npy');latents=np.load(CACHE/'screen-latents.npy',mmap_mode='r')
    assert real.shape==(2048,2048) and len(latents)==2048
    results={}
    names=['freepairs-pass06','freepairs-pass12',f'freepairs-pass{args.passes:02d}',f'freepairs-pass{args.passes:02d}-gain105']
    for name in dict.fromkeys(names):
        path=OUT/(name+'-codebook.pt');q=build(path,device);features=[];error=0.
        for first in range(0,len(latents),8):
            x=torch.from_numpy(latents[first:first+8].copy()).to(device)
            z=q.quantize(x,chunk_size=512)['quantized'];image=aux.decode(z)
            value=network(image.mul(.5).add(.5).clamp(0,1))
            features.append((value[0] if isinstance(value,tuple) else value).cpu().numpy())
            error+=float((x-z).square().sum())
            if first%512==0:progress('screening_free_pair_vectors',candidate=name,images=first+len(x))
        fake=np.concatenate(features)
        fid=float(metrics.frechet_distance(fake.mean(0),np.cov(fake,rowvar=False),real.mean(0),np.cov(real,rowvar=False)))
        results[name]=dict(matched_subset_rfid=fid,images=2048,vocabulary=20481,
            codebook_path=str(path),codebook_sha256=sha(path),latent_mse=error/latents.size)
        write('screen-results.json',results=results,selection_only=True,baseline_subset_rfid=19.534767882996107)
        progress('screen_result',candidate=name,**results[name])
        del q
    selected=min(results,key=lambda k:results[k]['matched_subset_rfid'])
    write('selected.json',candidate=selected,**results[selected],selection_only=True,
        improved_subset_rfid=results[selected]['matched_subset_rfid']<19.534767882996107)


@torch.inference_mode()
def full(args,device):
    from torch.utils.data import DataLoader,Subset
    from rqvae.utils.config import load_config
    from rqvae.img_datasets.transforms import create_transforms
    from src.training.rq_imagenet_runtime import TarImageNet
    from src import rqvae_metrics as metrics
    import torch.distributed as dist
    rank,world=int(os.environ.get('RANK',0)),int(os.environ.get('WORLD_SIZE',1))
    if world>1:dist.init_process_group('nccl')
    selected=json.loads((OUT/'selected.json').read_text());path=Path(selected['codebook_path'])
    assert sha(path)==selected['codebook_sha256']
    metrics.INCEPTION_PATH=SOURCE/'upstream/rqvae/metrics/inception.py'
    aux=FrozenJointLaserTokenizer(SOURCE/'inputs/stage1-model.pt',path).to(device)
    config=load_config(SOURCE/'inputs/stage2-training-config.yaml')
    data=TarImageNet(SOURCE/'data','val',create_transforms(config.dataset,split='val'));assert len(data)==50000
    loader=DataLoader(Subset(data,range(rank,len(data),world)),batch_size=args.batch_size,num_workers=2,pin_memory=True)
    metric=metrics.DistributedOriginalRQVAEMetrics(device);count=0
    for images,_ in loader:
        images=images.to(device);reconstruction,_,_=aux(images)
        metric.update(images.mul(.5).add(.5).clamp(0,1),real=True)
        metric.update(reconstruction.mul(.5).add(.5).clamp(0,1),real=False)
        count+=len(images)
        if rank==0 and count%512==0:progress('full_reconstruction_fid',images_on_rank=count,target_on_rank=len(range(rank,len(data),world)))
    fid,_,_=metric.compute();assert int(metric.real_count)==int(metric.fake_count)==50000
    if rank==0:
        write('full-rfid.json',completed=True,measured_rfid=fid,vocabulary=20481,
            baseline_rfid=4.689409482068868,change_from_baseline=fid-4.689409482068868,
            improved=fid<4.689409482068868,source_full_rfid=4.214186495529759,
            source_rfid_not_recomputed=True,candidate=selected['candidate'],
            codebook_path=str(path),codebook_sha256=sha(path),real_images=50000,reconstructed_images=50000,
            encoder_decoder_frozen=True,reference='matched full validation, original RQ Inception',
            current_stage2_training_unchanged=True,stage2_launched=False)
    if world>1:dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['fit','screen','full'])
    parser.add_argument('--passes',type=int,default=24);parser.add_argument('--cg-iterations',type=int,default=16)
    parser.add_argument('--memory-fraction',type=float,default=.15);parser.add_argument('--batch-size',type=int,default=8)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True);BASE.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda',int(os.environ.get('LOCAL_RANK',0)));torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction,device);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.action=='fit':fit(args,device)
    elif args.action=='screen':screen(args,device)
    else:full(args,device)


if __name__=='__main__':main()
