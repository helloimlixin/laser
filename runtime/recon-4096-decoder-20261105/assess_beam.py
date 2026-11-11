"""Matched frozen-CNN reconstruction screen and full ImageNet validation."""
import argparse
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path('/workspace/Projects/laser')
SOURCE=Path('/tmp/laser-j32769-a8192-20261105/frozen')
BASE=Path('/tmp/laser-recon-4096-beam-20261105')
PUBLIC=ROOT/'outputs/imagenet-native4096-decoder-quality-20261105'
sys.path[:0]=[str(SOURCE/'source'),str(SOURCE)]
os.environ['RQ_UPSTREAM_ROOT']=str(SOURCE/'upstream')
os.environ['TORCH_HOME']=str(BASE/'torch-cache')
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader,Subset
from src import rq_upstream
from src.models.laser_joint_tokenizer import FrozenJointLaserTokenizer,JointAtomCoefficientRQ
from src.training.rq_imagenet_runtime import TarImageNet
from src import rqvae_metrics as metrics
from rqvae.utils.config import load_config
from rqvae.img_datasets.transforms import create_transforms
from rqvae.losses.vqgan.lpips import LPIPS
spec=importlib.util.spec_from_file_location('recon_beam',ROOT/'src/models/laser_joint_search.py')
beam_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(beam_module)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.pending');tmp.write_text(json.dumps(value,indent=2,default=str)+'\n');tmp.replace(path)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def book(path,device):
    spec=torch.load(path,map_location='cpu',weights_only=True)
    q=JointAtomCoefficientRQ(spec['dictionary'],spec['atom_ids'],spec['coefficients']).to(device)
    assert q.depth==4 and q.vocab_size==4*q.dictionary.shape[1]+1
    assert not any(p.requires_grad for p in q.parameters())
    return q


def module_sha(module):
    h=hashlib.sha256()
    for name,value in module.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.inference_mode()
def evaluate(args):
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    device=torch.device('cuda',int(os.environ['LOCAL_RANK']));torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    dist.init_process_group('nccl',device_id=device)
    manifest=json.loads((BASE/'candidates.json').read_text())
    if args.phase=='full':
        selected=json.loads((BASE/'screen-results.json').read_text())['selected']
        names=[selected]
        if selected!='compact4096-beam2':names.append('compact4096-beam2')
    else:names=list(manifest['candidates'])
    paths={n:Path(manifest['candidates'][n]['path']) for n in names}
    qs={n:book(p,device) for n,p in paths.items()}
    aux=FrozenJointLaserTokenizer(SOURCE/'inputs/stage1-model.pt',paths[names[0]]).to(device).eval().requires_grad_(False)
    before=module_sha(aux.backbone)
    perceptual=LPIPS().to(device).eval().requires_grad_(False)
    metrics.INCEPTION_PATH=SOURCE/'upstream/rqvae/metrics/inception.py'
    inception=metrics.OriginalRQVAEInception().to(device).eval().requires_grad_(False)
    accum={n:metrics.DistributedOriginalRQVAEMetrics(device,inception=inception) for n in names}
    sums={n:torch.zeros(5,dtype=torch.float64,device=device) for n in names}
    config=load_config(SOURCE/'inputs/stage2-training-config.yaml')
    if args.phase=='full':
        data=TarImageNet(BASE/'data','val',create_transforms(config.dataset,split='val'))
        indices=list(range(len(data)));assert len(indices)==50000
    else:
        data=TarImageNet(BASE/'data','train',create_transforms(config.dataset,split='val'))
        indices=manifest['screen_indices']
    loader=DataLoader(Subset(data,indices[rank::world]),batch_size=args.batch_size,
        num_workers=4,pin_memory=True,persistent_workers=True,prefetch_factor=1,
        multiprocessing_context='spawn')
    seen=0;start=time.monotonic()
    for batch,(images,_) in enumerate(loader):
        images=images.to(device,non_blocking=True)
        latent=aux.encode(images)
        real=images.mul(.5).add(.5).clamp(0,1)
        features,_=inception(real)
        features=features.double()
        total=features.sum(0);cross=features.T@features
        greedy_cache={}
        for name in names:
            q=qs[name]
            key=str(paths[name])
            if key not in greedy_cache:greedy_cache[key]=q.quantize(latent,chunk_size=1024)
            width=int(manifest['candidates'][name].get('beam_width',1))
            encoded=greedy_cache[key] if width==1 else beam_module.beam_quantize(q,latent,beam_width=width,chunk_size=256,greedy=greedy_cache[key])
            quantized=encoded['quantized']
            reconstructed=aux.decode(quantized).float().clamp(-1,1)
            fake=reconstructed.mul(.5).add(.5)
            metric=accum[name]
            metric.real_sum.add_(total);metric.real_cross.add_(cross);metric.real_count.add_(len(images))
            metric.update(fake,real=False)
            error=(fake-real).square().mean((1,2,3))
            psnr=-10*error.clamp_min(1e-12).log10()
            lpips=perceptual(reconstructed,images,reduction='none').flatten()
            sums[name]+=torch.stack([torch.tensor(len(images),device=device),error.sum(),psnr.sum(),lpips.sum(),
                (latent-quantized).square().mean((1,2,3)).sum()]).double()
        seen+=len(images)
        if rank==0 and (batch==0 or batch%16==0):
            row=dict(phase=args.phase,images_on_rank=seen,target_on_rank=len(indices[rank::world]),
                candidates=names,elapsed_seconds=time.monotonic()-start,time=time.time())
            write(BASE/'status.json',row);print(json.dumps(row),flush=True)
    result={}
    for name in names:
        dist.all_reduce(sums[name])
        count=int(sums[name][0]);assert count==len(indices)
        fid,_,_=accum[name].compute()
        assert int(accum[name].fake_count)==int(accum[name].real_count)==count
        result[name]=dict(rfid=fid,pixel_mse=float(sums[name][1]/count),
            psnr_db=float(sums[name][2]/count),lpips=float(sums[name][3]/count),
            latent_mse=float(sums[name][4]/count),images=count,
            atoms=qs[name].dictionary.shape[1],vocabulary=qs[name].vocab_size,
            codebook_path=str(paths[name]),codebook_sha256=sha(paths[name]),reconstruction_beam_width=manifest['candidates'][name].get('beam_width',1))
        if rank==0:print(json.dumps(dict(candidate=name,**result[name])),flush=True)
    assert module_sha(aux.backbone)==before
    if rank==0:
        report=dict(phase=args.phase,results=result,encoder_decoder_frozen_and_unchanged=True,
            source_checkpoint_sha256=sha(SOURCE/'inputs/stage1-model.pt'),
            images=len(indices),real_and_reconstructed_same_images=True,
            split='validation' if args.phase=='full' else 'heldout_training',
            transforms='Resize256 CenterCrop256 Resize256',precision='float32',
            metric_backend='Original RQ-VAE TensorFlow-FID-compatible Inception',
            lpips_backend='Original RQ-VAE VGG LPIPS',world_size=world,
            elapsed_seconds=time.monotonic()-start,time=time.time(),transformer_training_started=False)
        if args.phase=='screen':
            candidates=[n for n in names if n!='control8192']
            report['selected']=min(candidates,key=lambda n:result[n]['rfid'])
            report['selection_only']=True
            report['validation_used_for_selection']=False
        else:
            compact=result[names[0]]
            report.update(selected=names[0],threshold_rfid=4.73,beats_original_rq=compact['rfid']<4.73,
                native_stochastic_soft_teacher_unchanged=True,
                reconstruction_encoding_only=True,physical_codewords_unchanged=True,
                reference_8192_rfid=4.6317537087924165,reference_4096_greedy_rfid=4.908170663058399)
        write(BASE/(args.phase+'-results.json'),report)
        write(PUBLIC/(args.phase+'-results.json'),report)
        write(BASE/'status.json',dict(phase=args.phase+'_completed',time=time.time()))
    dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['screen','full'])
    p.add_argument('--batch-size',type=int,default=16)
    evaluate(p.parse_args())
