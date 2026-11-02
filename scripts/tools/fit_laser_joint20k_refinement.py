#!/usr/bin/env python3
"""Measure the fidelity/rate tradeoff of five or six native joint pair tokens."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path('/workspace/Projects/laser');SOURCE=Path('/tmp/laser-rq-joint20k-imagenet-20261017')
OUT=ROOT/'outputs/imagenet-laser-joint20k-refinement-rfid-20261017'
PRIOR=ROOT/'outputs/imagenet-laser-adaptive-joint20k-vocabulary-20261017'
FIT=Path('/tmp/laser-compact-joint-vocabulary-20261016/fit-latents.npy')
CACHE=Path('/tmp/laser-rq-joint-vocabulary-study-20261015')
os.environ['RQ_UPSTREAM_ROOT']=str(SOURCE/'upstream');os.environ.setdefault('TORCH_HOME',str(SOURCE/'torch-cache'))
sys.path[:0]=[str(ROOT),str(SOURCE/'upstream'),str(ROOT/'runtime')]
import numpy as np
import torch
from src.models.laser_joint_tokenizer import JointAtomCoefficientRQ,FrozenJointLaserTokenizer
from src.training.laser_joint_refinement import append_refinement_depth,refit_last_depth


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(name,**values):
    p=OUT/name;tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(dict(values,time=time.time()),indent=2,allow_nan=False)+'\n');tmp.replace(p)


def progress(phase,**values):
    write('progress.json',phase=phase,**values);print(json.dumps(dict(phase=phase,**values)),flush=True)


def build(path,device):
    s=torch.load(path,map_location='cpu',weights_only=True)
    return JointAtomCoefficientRQ(s['dictionary'],s['atom_ids'],s['coefficients']).to(device)


@torch.inference_mode()
def fit(args,device):
    calibration=json.loads((PRIOR/'calibration-data.json').read_text());chosen=json.loads((PRIOR/'selected.json').read_text())
    assert sha(FIT)==calibration['fit_latents_sha256']
    assert sha(CACHE/'temperature-latents.npy')==calibration['heldout_latents_sha256']
    path=Path(chosen['codebook_path']);assert sha(path)==chosen['codebook_sha256']
    q=build(path,device);original_coefficients=q.coefficients.clone();spec=torch.load(path,map_location='cpu',weights_only=True)
    x=torch.from_numpy(np.load(FIT).reshape(-1,256)).to(device)
    held=torch.from_numpy(np.load(CACHE/'temperature-latents.npy').reshape(-1,256)).to(device)
    baseline=float((held-q.quantize(held,chunk_size=1024)['quantized']).square().mean())
    results=[]
    for depth in [5,6]:
        q=append_refinement_depth(q,scale=.75)
        best,best_error,trace=None,float('inf'),[]
        # The earlier levels never move. Fit the new row on its exact incoming
        # residual; there is no retrospective coefficient refit during coding.
        for iteration in range(args.passes+1):
            codes=q.quantize(x,chunk_size=1024)['codes']
            error=float((held-q.quantize(held,chunk_size=1024)['quantized']).square().mean())
            if error<best_error:best,best_error=q.coefficients[-1].cpu().clone(),error
            row=dict(depth=depth,iteration=iteration,heldout_training_latent_mse=error)
            if iteration<args.passes:row.update(refit_last_depth(q,x,codes))
            trace.append(row);progress('fitting_refinement_row',**row)
        q.coefficients[-1].copy_(best.to(device))
        torch.testing.assert_close(q.coefficients[:4],original_coefficients,rtol=0,atol=0)
        target=OUT/f'joint20k-depth{depth}-codebook.pt';assert not target.exists()
        book=dict(spec,dictionary=q.dictionary.cpu(),atom_ids=q.atom_ids.cpu(),coefficients=q.coefficients.cpu(),
            source_pair_codebook_sha256=chosen['codebook_sha256'],residual_depth=depth,vocabulary=20481,
            first_four_depths_exactly_unchanged=True,dictionary_encoder_decoder_frozen=True,
            native_greedy_residual_codes=True,retrospective_encoding_refit=False,training_only_fitting=True)
        torch.save(book,target)
        results.append(dict(depth=depth,tokens_per_image=64*depth,baseline_tokens_per_image=256,
            heldout_mse=best_error,baseline_heldout_mse=baseline,codebook_path=str(target),codebook_sha256=sha(target),trace=trace))
        write('fit-report.json',completed=depth==6,results=results,vocabulary=20481,
            baseline_full_rfid=4.689409482068868,first_four_depths_unchanged=True)


@torch.inference_mode()
def screen(device):
    from src import rqvae_metrics as metrics
    metrics.INCEPTION_PATH=SOURCE/'upstream/rqvae/metrics/inception.py'
    net=metrics.OriginalRQVAEInception().eval().requires_grad_(False).to(device)
    aux=FrozenJointLaserTokenizer(SOURCE/'inputs/stage1-model.pt',SOURCE/'inputs/joint-codebook.pt').to(device)
    real=np.load(CACHE/'screen-real-features.npy');latents=np.load(CACHE/'screen-latents.npy',mmap_mode='r')
    results={}
    for depth in [5,6]:
        path=OUT/f'joint20k-depth{depth}-codebook.pt';q=build(path,device);features=[];error=0.
        for first in range(0,len(latents),8):
            x=torch.from_numpy(latents[first:first+8].copy()).to(device);z=q.quantize(x,chunk_size=512)['quantized']
            image=aux.decode(z);value=net(image.mul(.5).add(.5).clamp(0,1))
            features.append((value[0] if isinstance(value,tuple) else value).cpu().numpy())
            error+=float((x-z).square().sum())
            if first%512==0:progress('screening_refinement',depth=depth,images=first+len(x))
        fake=np.concatenate(features)
        fid=float(metrics.frechet_distance(fake.mean(0),np.cov(fake,rowvar=False),real.mean(0),np.cov(real,rowvar=False)))
        results[str(depth)]=dict(matched_subset_rfid=fid,images=2048,depth=depth,vocabulary=20481,
            tokens_per_image=depth*64,codebook_path=str(path),codebook_sha256=sha(path),latent_mse=error/latents.size)
        write('screen-results.json',results=results,selection_only=True,baseline_subset_rfid=19.534767882996107)
        progress('screen_result',**results[str(depth)])


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
    path=OUT/'joint20k-depth6-codebook.pt';metrics.INCEPTION_PATH=SOURCE/'upstream/rqvae/metrics/inception.py'
    aux=FrozenJointLaserTokenizer(SOURCE/'inputs/stage1-model.pt',path).to(device)
    five_book=torch.load(OUT/'joint20k-depth5-codebook.pt',map_location='cpu',weights_only=True)
    torch.testing.assert_close(aux.quantizer.coefficients[:5].cpu(),five_book['coefficients'],rtol=0,atol=0)
    torch.testing.assert_close(aux.quantizer.dictionary.cpu(),five_book['dictionary'],rtol=0,atol=0)
    network=metrics.OriginalRQVAEInception().eval().requires_grad_(False).to(device)
    five=metrics.DistributedOriginalRQVAEMetrics(device,inception=network)
    six=metrics.DistributedOriginalRQVAEMetrics(device,inception=network)
    config=load_config(SOURCE/'inputs/stage2-training-config.yaml')
    data=TarImageNet(SOURCE/'data','val',create_transforms(config.dataset,split='val'));assert len(data)==50000
    loader=DataLoader(Subset(data,range(rank,len(data),world)),batch_size=args.batch_size,num_workers=2,pin_memory=True)
    count=0
    for images,_ in loader:
        images=images.to(device);x=aux.encode(images);codes=aux.quantizer.quantize(x,chunk_size=512)['codes']
        vectors=aux.quantizer.embed(codes);z5=vectors[...,:5,:].sum(-2);z6=vectors.sum(-2)
        five.update(images.mul(.5).add(.5).clamp(0,1),real=True)
        for metric,z in [(five,z5),(six,z6)]:metric.update(aux.decode(z).mul(.5).add(.5).clamp(0,1),real=False)
        count+=len(images)
        if rank==0 and count%512==0:progress('full_refinement_reconstruction_fid',images_on_rank=count,target_on_rank=len(range(rank,len(data),world)))
    # Copy local real moments before either metric performs its reduction.
    six.real_sum.copy_(five.real_sum);six.real_cross.copy_(five.real_cross);six.real_count.copy_(five.real_count)
    result={}
    for depth,metric in [(5,five),(6,six)]:
        fid,_,_=metric.compute();assert int(metric.real_count)==int(metric.fake_count)==50000
        result[str(depth)]=dict(measured_rfid=fid,depth=depth,vocabulary=20481,tokens_per_image=64*depth,
            baseline_full_rfid=4.689409482068868,baseline_depth=4,baseline_tokens_per_image=256,
            change_from_baseline=fid-4.689409482068868,codebook_path=str(OUT/f'joint20k-depth{depth}-codebook.pt'),
            codebook_sha256=sha(OUT/f'joint20k-depth{depth}-codebook.pt'))
    if rank==0:write('full-rfid.json',completed=True,results=result,real_images=50000,reconstructed_images_per_candidate=50000,
        reference='matched full validation, original RQ Inception',encoder_decoder_dictionary_frozen=True,
        first_four_depths_unchanged=True,vocabulary=20481,longer_sequence_required=True,current_training_unchanged=True)
    if world>1:dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['fit','screen','full'])
    parser.add_argument('--passes',type=int,default=16);parser.add_argument('--device',type=int,default=2)
    parser.add_argument('--memory-fraction',type=float,default=.14);parser.add_argument('--batch-size',type=int,default=8)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda',int(os.environ.get('LOCAL_RANK',args.device)));torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction,device);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.action=='fit':fit(args,device)
    elif args.action=='screen':screen(device)
    else:full(args,device)


if __name__=='__main__':main()
