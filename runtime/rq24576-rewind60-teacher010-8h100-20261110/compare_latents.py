"""Matched-image latent error and paired reconstruction FID against original RQ."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

BASE=Path('/tmp/laser-rq24576-native-8h100-20261108')
NEW=Path('/tmp/laser-rq24576-rewind60-teacher010-8h100-20261110')
os.environ.update(RQ_UPSTREAM_ROOT=str(BASE/'upstream'),TORCH_HOME=str(BASE/'torch-cache'))
sys.path[:0]=[str(BASE/'source'),str(BASE)]
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from src import rq_upstream,rqvae_metrics as metrics
from src.training.rq_imagenet_recipes import RQRecipe
from src.training.rq_imagenet_runtime import TarImageNet
from runtime.tokenizer import NativeVectorTokenizer,module_digest
from rqvae.img_datasets.transforms import create_transforms
from rqvae.utils.config import load_config


def write(path,value):
    path=Path(path);pending=path.with_suffix('.pending')
    pending.write_text(json.dumps(value,indent=2)+'\n');pending.replace(path)


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--rank',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--validation',action='store_true')
    args=parser.parse_args();out=args.output
    suffix='latent-comparison-val50k-matched' if os.environ.get('LASER_COMPARISON_TEMPERATURE') else ('latent-comparison-val50k' if args.validation else 'latent-comparison')
    local=NEW/suffix;local.mkdir(exist_ok=True)
    rank=args.rank;torch.set_num_threads(2);torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('LASER_COMPARISON_MEMORY_GIB','10'))*2**30/torch.cuda.get_device_properties(0).total_memory)
    def status(phase,**kw):write(out/f'worker-{rank}.json',dict(rank=rank,phase=phase,time=time.time(),**kw))
    status('loading_tokenizers')
    original,_=RQRecipe(NEW/'original-rq').tokenizer(torch.device('cuda'))
    laser=NativeVectorTokenizer(BASE/'inputs/stage1-model.pt',BASE/'inputs/joint-codebook.pt').cuda()
    models={'original_rq':original,'laser_rq24576':laser}
    before={k:module_digest(v) for k,v in models.items()}
    metrics.INCEPTION_PATH=BASE/'upstream/rqvae/metrics/inception.py'
    inception=metrics.OriginalRQVAEInception().cuda().eval().requires_grad_(False)
    indices=(np.arange(50000) if args.validation else
        np.load('/workspace/Projects/laser/outputs/imagenet-rq24576-tokenizer-20261108/heldout-training-indices.npy'))
    config=load_config(BASE/'inputs/stage2-training-config.yaml')
    dataset=TarImageNet(BASE/'data','val' if args.validation else 'train',create_transforms(config.dataset,split='val'),seed=26110824576)
    assert len(indices)==(50000 if args.validation else 10000)
    assert np.array_equal(np.bincount(dataset.labels[indices]),np.full(1000,50 if args.validation else 10))
    part=indices[rank::8].tolist();loader=DataLoader(Subset(dataset,part),batch_size=8,num_workers=2,
        pin_memory=True,prefetch_factor=1,multiprocessing_context='spawn')
    latent={k:[] for k in models};real_features=[];input_hash=hashlib.sha256();image_count=0
    status('encoding_identical_images',target=len(part))
    for images,_ in loader:
        input_hash.update(images.numpy().tobytes());images=images.cuda(non_blocking=True).float()
        for name,model in models.items():latent[name].append(model.encode(images).cpu())
        f,_=inception(images.mul(.5).add(.5).clamp(0,1));real_features.append(f.cpu())
        image_count+=len(images)
        if image_count%200==0:status('encoding_identical_images',images=image_count,target=len(part))
    latent={k:torch.cat(v) for k,v in latent.items()};real_features=torch.cat(real_features).double().cuda()
    assert image_count==len(part) and all(v.shape==(len(part),8,8,256) for v in latent.values())
    real_total=real_features.sum(0).cpu().numpy();real_cross=(real_features.T@real_features).cpu().numpy()
    np.savez(local/f'real-rank-{rank}.npz',total=real_total,cross=real_cross,count=image_count)
    profiles=[('original_rq','greedy',0.),('original_rq','teacher050',.5),
        ('laser_rq24576','greedy',0.),('laser_rq24576','teacher050',.5),
        ('laser_rq24576','teacher025',.25),('laser_rq24576','teacher010',.1)]
    if args.validation:profiles=[p for p in profiles if p[1] not in ['teacher025'] and (p[0]=='original_rq' or p[1]!='teacher050')]
    if os.environ.get('LASER_COMPARISON_TEMPERATURE'):
        profiles=[(model, 'teacher_matched' if model=='laser_rq24576' and name=='teacher010' else name,
            float(os.environ['LASER_COMPARISON_TEMPERATURE']) if model=='laser_rq24576' and name=='teacher010' else temp)
            for model,name,temp in profiles]
    for model_name,label,temp in profiles:
        key=model_name+'_'+label;model=models[model_name];q=model.quantizer
        torch.manual_seed(771001+rank);torch.cuda.manual_seed(771001+rank)
        total=torch.zeros(2048,dtype=torch.float64,device='cuda');cross=torch.zeros(2048,2048,dtype=torch.float64,device='cuda')
        sums=torch.zeros(15,dtype=torch.float64,device='cuda');channels=torch.zeros(256,dtype=torch.float64,device='cuda')
        per_image=[];count=0
        status('measuring',profile=key,images=0,target=len(part))
        for x_cpu in latent[model_name].split(8):
            x=x_cpu.cuda();_,hard_codes=q.quantize(x)
            if temp:
                soft,codes=q.get_soft_codes(x,temp=temp,stochastic=True)
                for d in range(4):
                    p=soft[...,d,:];sums[3+d]+=-(p*p.clamp_min(1e-30).log()).sum().double()
                del soft,p
            else:codes=hard_codes
            recon=q.embed_code(codes)
            error=(x-recon).square().flatten(1).sum(1).double()
            energy=x.square().flatten(1).sum(1).double()
            per_image.append(torch.stack([error,energy],1).cpu())
            sums[0]+=error.sum();sums[1]+=energy.sum();sums[2]+=x.numel()
            channels+=x.double().sum((0,1,2))
            residual=x.clone()
            for d in range(4):
                residual-=q.codebooks[d].embed(codes[...,d]);sums[7+d]+=residual.square().sum().double()
                sums[11+d]+=(codes[...,d]!=hard_codes[...,d]).sum().double()
            images=model.decode_code(codes).float().mul(.5).add(.5).clamp(0,1)
            features,_=inception(images);features=features.double()
            total+=features.sum(0);cross.addmm_(features.T,features)
            if count==0:
                from PIL import Image
                from torchvision.utils import make_grid
                a=make_grid(images,nrow=len(images),padding=2).mul(255).round().byte().permute(1,2,0).cpu().numpy()
                Image.fromarray(a).save(local/f'{key}-rank-{rank}.png')
            count+=len(x)
            if count%200==0:status('measuring',profile=key,images=count,target=len(part))
        values=sums.cpu().numpy()
        np.savez(local/f'{key}-rank-{rank}.npz',total=total.cpu().numpy(),cross=cross.cpu().numpy(),
            count=count,sums=values,channel_sum=channels.cpu().numpy(),per_image=torch.cat(per_image).numpy())
        write(out/f'{key}-rank-{rank}.json',dict(rank=rank,images=count,profile=key,
            model=model_name,teacher_temperature=temp,latent_mse=float(values[0]/values[2]),
            entropy_by_depth=(values[3:7]/(count*64)).tolist()))
        print(json.dumps(dict(rank=rank,profile=key,latent_mse=float(values[0]/values[2]))),flush=True)
        del total,cross,sums;torch.cuda.empty_cache()
    assert all(module_digest(model)==before[name] for name,model in models.items())
    write(out/f'verification-rank-{rank}.json',dict(passed=True,images=image_count,
        identical_input_images_for_both_encoders=True,input_tensor_sha256=input_hash.hexdigest(),
        models_frozen_and_unchanged=True,tokenizer_digests=before))
    status('completed')


if __name__=='__main__':main()
