#!/usr/bin/env python3
"""Evaluate compact-latent decoder corrections without changing the active AR run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path('/workspace/Projects/laser')
SOURCE=Path('/tmp/laser-rq-joint20k-imagenet-20261017')
OUT=ROOT/'outputs/imagenet-laser-joint20k-decoder-calibration-20261017'
PRIOR=ROOT/'outputs/imagenet-laser-adaptive-joint20k-vocabulary-20261017'
FIT=Path('/tmp/laser-compact-joint-vocabulary-20261016/fit-latents.npy')
CACHE=Path('/tmp/laser-rq-joint-vocabulary-study-20261015')
os.environ['RQ_UPSTREAM_ROOT']=str(SOURCE/'upstream');os.environ.setdefault('TORCH_HOME',str(SOURCE/'torch-cache'))
sys.path[:0]=[str(ROOT),str(SOURCE/'upstream'),str(ROOT/'runtime')]
import numpy as np
import torch
from src.models.laser_joint_tokenizer import JointAtomCoefficientRQ,FrozenJointLaserTokenizer
from src.training.laser_latent_decoder_calibration import (
    latent_regression_statistics,select_center_statistics,solve_latent_correction,correct_latent,
)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(name,**values):
    p=OUT/name;tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(dict(values,time=time.time()),indent=2,allow_nan=False)+'\n');tmp.replace(p)


def progress(phase,**values):
    write('progress.json',phase=phase,**values);print(json.dumps(dict(phase=phase,**values)),flush=True)


def get_book(device):
    report=json.loads((PRIOR/'selected.json').read_text());path=Path(report['codebook_path'])
    assert sha(path)==report['codebook_sha256']
    spec=torch.load(path,map_location='cpu',weights_only=True)
    return JointAtomCoefficientRQ(spec['dictionary'],spec['atom_ids'],spec['coefficients']).to(device),report


@torch.inference_mode()
def fit(device):
    metadata=json.loads((PRIOR/'calibration-data.json').read_text())
    assert sha(FIT)==metadata['fit_latents_sha256']
    assert sha(CACHE/'temperature-latents.npy')==metadata['heldout_latents_sha256']
    q,book=get_book(device)
    x=torch.from_numpy(np.load(FIT)).to(device)
    held=torch.from_numpy(np.load(CACHE/'temperature-latents.npy')).to(device)
    z=q.quantize(x,chunk_size=1024)['quantized'].permute(0,3,1,2).contiguous()
    hz=q.quantize(held,chunk_size=1024)['quantized'].permute(0,3,1,2).contiguous()
    x=x.permute(0,3,1,2).contiguous();held=held.permute(0,3,1,2).contiguous()
    baseline=float((held-hz).square().mean())
    progress('fitting_decoder_input_correction',training_views=len(x),heldout_training_images=len(held),baseline_heldout_mse=baseline)
    spatial=latent_regression_statistics(z,x)
    trace=[]
    for kernel,statistics in [(1,select_center_statistics(spatial)),(3,spatial)]:
        best,best_error=None,float('inf')
        for ridge in [.001,.01,.1,1.]:
            adapter=solve_latent_correction(statistics,ridge_fraction=ridge)
            error=float((held-correct_latent(hz,adapter)).square().mean())
            trace.append(dict(kernel=kernel,ridge_fraction=ridge,heldout_training_latent_mse=error))
            if error<best_error:best,best_error=adapter,error
        name=f'decoder-kernel{kernel}'
        spec=dict(kind='laser_compact_latent_decoder_correction',format_version=1,
            **{k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in best.items()},
            source_stage1_sha256=metadata['source_stage1_sha256'],source_codebook_sha256=book['codebook_sha256'],
            fitting_split='train',training_views=len(x),unique_training_images=4096,
            heldout_training_images=len(held),vocabulary=20481,integer_codes_unchanged=True,
            transformer_input_embeddings_unchanged=True,frozen_cnn_weights=True,
            objective='ridge regression to frozen pre-quantization encoder latents',
            application='residual spatial correction after summing four pair codewords, before frozen CNN decoder',
            baseline_heldout_mse=baseline,heldout_mse=best_error)
        path=OUT/(name+'.pt');assert not path.exists();torch.save(spec,path)
        progress('decoder_correction_fitted',candidate=name,heldout_mse=best_error,baseline_heldout_mse=baseline,ridge_fraction=best['ridge_fraction'])
    write('fit-report.json',completed=True,trace=trace,baseline_heldout_mse=baseline,
        vocabulary=20481,integer_codes_and_embeddings_unchanged=True,cnn_weights_frozen=True,
        fitting_training_only=True,peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30)


@torch.inference_mode()
def screen(device):
    from src import rqvae_metrics as metrics
    metrics.INCEPTION_PATH=SOURCE/'upstream/rqvae/metrics/inception.py'
    network=metrics.OriginalRQVAEInception().eval().requires_grad_(False).to(device)
    aux=FrozenJointLaserTokenizer(SOURCE/'inputs/stage1-model.pt',SOURCE/'inputs/joint-codebook.pt').to(device)
    real=np.load(CACHE/'screen-real-features.npy');latents=np.load(CACHE/'screen-latents.npy',mmap_mode='r')
    results={}
    for kernel in [1,3]:
        name=f'decoder-kernel{kernel}';path=OUT/(name+'.pt');adapter=torch.load(path,map_location=device,weights_only=True)
        features=[];error=0.
        for first in range(0,len(latents),8):
            x=torch.from_numpy(latents[first:first+8].copy()).to(device)
            z=aux.quantizer.quantize(x,chunk_size=512)['quantized']
            corrected=correct_latent(z.permute(0,3,1,2).contiguous(),adapter).permute(0,2,3,1).contiguous()
            image=aux.decode(corrected)
            value=network(image.mul(.5).add(.5).clamp(0,1))
            features.append((value[0] if isinstance(value,tuple) else value).cpu().numpy())
            error+=float((x-corrected).square().sum())
            if first%512==0:progress('screening_decoder_correction',candidate=name,images=first+len(x))
        fake=np.concatenate(features)
        fid=float(metrics.frechet_distance(fake.mean(0),np.cov(fake,rowvar=False),real.mean(0),np.cov(real,rowvar=False)))
        results[name]=dict(matched_subset_rfid=fid,images=2048,vocabulary=20481,
            adapter_path=str(path),adapter_sha256=sha(path),latent_mse=error/latents.size)
        write('screen-results.json',results=results,selection_only=True,baseline_subset_rfid=19.534767882996107)
        progress('screen_result',candidate=name,**results[name])
    selected=min(results,key=lambda k:results[k]['matched_subset_rfid'])
    write('selected.json',candidate=selected,**results[selected],selection_only=True)


@torch.inference_mode()
def full(args,device):
    import torch.distributed as dist
    from torch.utils.data import DataLoader,Subset
    from rqvae.utils.config import load_config
    from rqvae.img_datasets.transforms import create_transforms
    from src.training.rq_imagenet_runtime import TarImageNet
    from src import rqvae_metrics as metrics
    rank,world=int(os.environ.get('RANK',0)),int(os.environ.get('WORLD_SIZE',1))
    if world>1:dist.init_process_group('nccl')
    chosen=json.loads((OUT/'selected.json').read_text());path=Path(chosen['adapter_path'])
    assert sha(path)==chosen['adapter_sha256']
    adapter=torch.load(path,map_location=device,weights_only=True)
    metrics.INCEPTION_PATH=SOURCE/'upstream/rqvae/metrics/inception.py'
    aux=FrozenJointLaserTokenizer(SOURCE/'inputs/stage1-model.pt',SOURCE/'inputs/joint-codebook.pt').to(device)
    config=load_config(SOURCE/'inputs/stage2-training-config.yaml')
    data=TarImageNet(SOURCE/'data','val',create_transforms(config.dataset,split='val'));assert len(data)==50000
    loader=DataLoader(Subset(data,range(rank,len(data),world)),batch_size=args.batch_size,num_workers=2,pin_memory=True)
    metric=metrics.DistributedOriginalRQVAEMetrics(device);count=0
    for images,_ in loader:
        images=images.to(device);x=aux.encode(images);z=aux.quantizer.quantize(x,chunk_size=512)['quantized']
        corrected=correct_latent(z.permute(0,3,1,2).contiguous(),adapter).permute(0,2,3,1).contiguous()
        reconstructed=aux.decode(corrected)
        metric.update(images.mul(.5).add(.5).clamp(0,1),real=True)
        metric.update(reconstructed.mul(.5).add(.5).clamp(0,1),real=False)
        count+=len(images)
        if rank==0 and count%512==0:progress('full_decoder_reconstruction_fid',images_on_rank=count,target_on_rank=len(range(rank,len(data),world)))
    fid,_,_=metric.compute();assert int(metric.real_count)==int(metric.fake_count)==50000
    if rank==0:write('full-rfid.json',completed=True,measured_rfid=fid,vocabulary=20481,
        baseline_rfid=4.689409482068868,change_from_baseline=fid-4.689409482068868,
        improved=fid<4.689409482068868,source_full_rfid=4.214186495529759,source_rfid_not_recomputed=True,
        candidate=chosen['candidate'],adapter_path=str(path),adapter_sha256=sha(path),
        real_images=50000,reconstructed_images=50000,reference='matched full validation, original RQ Inception',
        frozen_cnn_weights=True,integer_codes_and_embeddings_unchanged=True,
        current_training_decoder_unchanged=True,stage2_launched=False)
    if world>1:dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['fit','screen','full'])
    parser.add_argument('--memory-fraction',type=float,default=.15);parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--device',type=int,default=1)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda',int(os.environ.get('LOCAL_RANK',args.device)));torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction,device);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.action=='fit':fit(device)
    elif args.action=='screen':screen(device)
    else:full(args,device)


if __name__=='__main__':main()
