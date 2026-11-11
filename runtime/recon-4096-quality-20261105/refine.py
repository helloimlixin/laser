"""Refine 4096 codebook atoms/levels through a strictly frozen CNN decoder."""
import argparse
import json
import math
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Subset,DistributedSampler
from evaluate import BASE,SOURCE,PUBLIC,ROOT,book,module_sha,write,sha
from src.models.laser_joint_tokenizer import FrozenJointLaserTokenizer
from src.training.rq_imagenet_runtime import TarImageNet
from rqvae.utils.config import load_config
from rqvae.img_datasets.transforms import create_transforms
from rqvae.losses.vqgan.lpips import LPIPS


class TrainableBook(nn.Module):
    def __init__(self,spec):
        super().__init__()
        self.dictionary=nn.Parameter(spec['dictionary'].clone())
        self.coefficients=nn.Parameter(spec['coefficients'].clone())
        self.register_buffer('atom_ids',spec['atom_ids'].clone())
        self.register_buffer('prior_dictionary',spec['dictionary'].clone())
        self.register_buffer('prior_coefficients',spec['coefficients'].clone())
        prior=spec['coefficients']
        lower=torch.minimum(prior*.8,prior*1.2)
        upper=torch.maximum(prior*.8,prior*1.2)
        same=spec['atom_ids'][:,1:-1]==spec['atom_ids'][:,2:]
        midpoint=(prior[:,1:-1]+prior[:,2:])*.5
        upper[:,1:-1]=torch.minimum(upper[:,1:-1],torch.where(same,midpoint-1e-7,torch.inf))
        lower[:,2:]=torch.maximum(lower[:,2:],torch.where(same,midpoint+1e-7,-torch.inf))
        self.register_buffer('lower',lower);self.register_buffer('upper',upper)

    def forward(self,codes):
        dictionary=F.normalize(self.dictionary,dim=0)
        return sum(dictionary.T[self.atom_ids[d,codes[...,d]]]*self.coefficients[d,codes[...,d],None]
                   for d in range(4))

    @torch.no_grad()
    def project(self):
        self.dictionary.div_(self.dictionary.norm(dim=0).clamp_min(1e-8))
        self.coefficients.copy_(self.coefficients.maximum(self.lower).minimum(self.upper))
        self.coefficients[:,0]=0

    @torch.no_grad()
    def refresh(self,q):
        q.dictionary.copy_(F.normalize(self.dictionary,dim=0))
        q.coefficients.copy_(self.coefficients)
        q.norms.copy_(q.dictionary.square().sum(0))


def main(args):
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    device=torch.device('cuda',int(os.environ['LOCAL_RANK']));torch.cuda.set_device(device)
    torch.set_num_threads(4);torch.manual_seed(2611054096+rank)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dist.init_process_group('nccl',device_id=device)
    manifest=json.loads((BASE/'candidates.json').read_text())
    baseline=json.loads((BASE/'screen-results.json').read_text())['selected']
    path=Path(manifest['candidates'][baseline]['path'])
    spec=torch.load(path,map_location='cpu',weights_only=True)
    assert spec['dictionary'].shape==(256,4096)
    learner=TrainableBook(spec).to(device)
    model=DDP(learner,device_ids=[device.index],broadcast_buffers=False)
    q=book(path,device)
    aux=FrozenJointLaserTokenizer(SOURCE/'inputs/stage1-model.pt',path).to(device).eval().requires_grad_(False)
    before=module_sha(aux.backbone)
    perceptual=LPIPS().to(device).eval().requires_grad_(False)
    optimizer=torch.optim.Adam([
        dict(params=[learner.dictionary],lr=.0003),dict(params=[learner.coefficients],lr=.00003)],betas=(.9,.99))
    fit=json.loads((ROOT/'outputs/imagenet-laser-compact-joint-vocabulary-20261016/calibration-data.json').read_text())['fit_indices']
    assert not set(fit)&set(manifest['screen_indices'])
    config=load_config(SOURCE/'inputs/stage2-training-config.yaml')
    dataset=TarImageNet(BASE/'data','train',create_transforms(config.dataset,split='train'),seed=261001)
    data=Subset(dataset,fit)
    sampler=DistributedSampler(data,num_replicas=world,rank=rank,shuffle=True,seed=2611054096)
    loader=DataLoader(data,batch_size=args.batch_size,sampler=sampler,num_workers=4,
        pin_memory=True,drop_last=True,persistent_workers=True,prefetch_factor=1,multiprocessing_context='spawn')
    step=epoch=0;trace=[];started=time.monotonic();snapshots=[]
    while step<args.steps:
        sampler.set_epoch(epoch);dataset.set_epoch(epoch)
        for images,_ in loader:
            images=images.to(device,non_blocking=True)
            with torch.no_grad():
                latent=aux.encode(images)
                learner.refresh(q)
                codes=q.quantize(latent,chunk_size=1024)['codes']
            optimizer.zero_grad(set_to_none=True)
            quantized=model(codes)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                decoded=aux.decode(quantized)
                pixel=(decoded.float()-images).abs().mean()
                percept=perceptual(decoded,images)
                latent_error=(quantized-latent).square().mean()
                regularization=(learner.dictionary-learner.prior_dictionary).square().mean()
                loss=pixel+percept+latent_error+.01*regularization
            assert torch.isfinite(loss)
            loss.backward()
            learner.coefficients.grad[:,0]=0
            norm=torch.nn.utils.clip_grad_norm_(learner.parameters(),1.)
            assert torch.isfinite(norm)
            optimizer.step();learner.project();step+=1
            factor=.2+.8*.5*(1+math.cos(math.pi*step/args.steps))
            optimizer.param_groups[0]['lr']=.0003*factor;optimizer.param_groups[1]['lr']=.00003*factor
            if step==1 or step%20==0:
                values=torch.stack([loss.detach(),pixel.detach(),percept.detach(),latent_error.detach()]).float()
                dist.all_reduce(values);values/=world
                row=dict(step=step,loss=float(values[0]),pixel_l1=float(values[1]),lpips=float(values[2]),
                    latent_mse=float(values[3]),elapsed_seconds=time.monotonic()-started)
                trace.append(row)
                if rank==0:write(BASE/'status.json',dict(phase='compact_codebook_refinement',**row,time=time.time()));print(json.dumps(row),flush=True)
            if step in [50,150,args.steps]:
                learner.refresh(q)
                if rank==0:
                    name=f'compact4096-refined-step{step:04d}';p=BASE/'books'/(name+'.pt')
                    saved=dict(spec,dictionary=q.dictionary.cpu().clone(),coefficients=q.coefficients.cpu().clone(),
                        image_perceptual_refinement=True,refinement_updates=step,refinement_training_images=len(fit),
                        refinement_loss='LPIPS + image L1 + latent MSE + dictionary prior',
                        encoder_decoder_frozen=True,validation_used_for_fitting=False)
                    torch.save(saved,p)
                    manifest['candidates'][name]=dict(path=str(p),atoms=4096,refinement_updates=step)
                    snapshots.append(name)
                dist.barrier()
            if step>=args.steps:break
        epoch+=1
    assert module_sha(aux.backbone)==before
    if rank==0:
        write(BASE/'candidates.json',manifest)
        report=dict(passed=True,updates=step,snapshots=snapshots,trace=trace,
            encoder_decoder_weights_bitwise_unchanged=True,transformer_training_started=False,
            optimized_parameters=['dictionary directions','four coefficient values per atom per depth'],
            atoms=4096,vocabulary=16385,depth=4,training_images=len(fit),
            validation_images_used=0,heldout_selection_images_disjoint_from_fitting=True,
            initial_codebook_sha256=sha(path),world_size=world,precision='BF16 decoder training; FP32 codebook and final evaluation')
        write(BASE/'refinement-verification.json',report);write(PUBLIC/'refinement-verification.json',report)
    dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--steps',type=int,default=300);p.add_argument('--batch-size',type=int,default=8)
    main(p.parse_args())
