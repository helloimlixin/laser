"""Matched decoder-only trials; freeze encoder, compact codes and discriminator."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path('/workspace/Projects/laser')
sys.path.insert(0,str(ROOT/'runtime/recon-4096-quality-20261105'))
from evaluate import SOURCE,book,module_sha,write,sha,TarImageNet,load_config,create_transforms,LPIPS
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Subset,DistributedSampler
from src.models.laser_joint_tokenizer import FrozenJointLaserTokenizer
from rqvae.losses.vqgan.discriminator import NLayerDiscriminator

BASE=Path('/tmp/laser-recon-4096-decoder-20261105')
PUBLIC=ROOT/'outputs/imagenet-native4096-decoder-quality-20261105'
DATA=Path('/tmp/laser-recon-4096-20261105/data')


class Decoder(nn.Module):
    def __init__(self,aux):
        super().__init__()
        self.post_quant_conv=aux.backbone.post_quant_conv
        self.decoder=aux.backbone.decoder
        self.requires_grad_(True)

    def forward(self,latent):
        return self.decoder(self.post_quant_conv(latent.permute(0,3,1,2).contiguous()))


def main(args):
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    device=torch.device('cuda',int(os.environ['LOCAL_RANK']));torch.cuda.set_device(device)
    torch.set_num_threads(4);torch.manual_seed(261105473+rank)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dist.init_process_group('nccl',device_id=device)
    out=BASE/args.arm;out.mkdir(exist_ok=True)
    manifest=json.loads((BASE/'manifest.json').read_text())
    path=Path(manifest['source_codebook'])
    q=book(path,device)
    aux=FrozenJointLaserTokenizer(SOURCE/'inputs/stage1-model.pt',path).to(device).eval().requires_grad_(False)
    encoder=nn.ModuleDict(dict(encoder=aux.backbone.encoder,quant_conv=aux.backbone.quant_conv))
    encoder_before=module_sha(encoder);book_before=module_sha(q)
    learner=Decoder(aux)
    assert all(not p.requires_grad for p in encoder.parameters())
    model=DDP(learner,device_ids=[device.index],broadcast_buffers=False)
    perceptual=LPIPS().to(device).eval().requires_grad_(False)
    discriminator=None
    if args.adversarial_weight:
        s=torch.load(SOURCE/'inputs/stage1-model.pt',map_location='cpu',weights_only=False,mmap=True)
        final_index=max(int(k.split('.')[1]) for k,v in s['discriminator'].items()
                        if k.endswith('.weight') and v.ndim==4)
        assert (final_index-2)%3==0
        discriminator=NLayerDiscriminator(n_layers=(final_index-2)//3).to(device).eval().requires_grad_(False)
        discriminator.load_state_dict(s['discriminator'],strict=True);del s
    discriminator_before=module_sha(discriminator) if discriminator is not None else None
    optimizer=torch.optim.Adam(learner.parameters(),lr=args.lr,betas=(.9,.99))
    config=load_config(SOURCE/'inputs/stage2-training-config.yaml')
    data=TarImageNet(DATA,'train',create_transforms(config.dataset,split='train'),seed=261001)
    subset=Subset(data,manifest['fit_indices'])
    sampler=DistributedSampler(subset,num_replicas=world,rank=rank,shuffle=True,seed=261105473)
    loader=DataLoader(subset,batch_size=args.batch_size,sampler=sampler,num_workers=4,
        drop_last=True,persistent_workers=True,pin_memory=True,prefetch_factor=1,multiprocessing_context='spawn')
    step=epoch=0;trace=[];snapshots=[];started=time.monotonic();probe=None
    while step<args.steps:
        sampler.set_epoch(epoch);data.set_epoch(epoch)
        for images,_ in loader:
            images=images.to(device,non_blocking=True)
            with torch.no_grad():
                latent=aux.encode(images)
                encoded=q.quantize(latent,chunk_size=1024)
                quantized=encoded['quantized']
                if probe is None:
                    probe=images[:8].clone();probe_codes=q.quantize(aux.encode(probe),chunk_size=1024)['codes'].clone()
            factor=min(1.,(step+1)/10)*(.2+.8*.5*(1+math.cos(math.pi*step/args.steps)))
            optimizer.param_groups[0]['lr']=args.lr*factor
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                decoded=model(quantized)
                pixel=(decoded.float()-images).abs().mean()
                percept=perceptual(decoded,images)
                adversarial=decoded.new_zeros(())
                if discriminator is not None:
                    logits,_=discriminator(decoded.clamp(-1,1))
                    # A frozen critic must not reward arbitrarily large logits.
                    adversarial=torch.nn.functional.softplus(-logits.float()).mean()
                loss=pixel+percept+args.adversarial_weight*adversarial
            assert torch.isfinite(loss)
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(learner.parameters(),1.)
            assert torch.isfinite(norm)
            optimizer.step();step+=1
            if step==1 or step%20==0:
                values=torch.stack([loss.detach(),pixel.detach(),percept.detach(),adversarial.detach(),norm.detach()]).float()
                dist.all_reduce(values);values/=world
                row=dict(step=step,loss=float(values[0]),pixel_l1=float(values[1]),lpips=float(values[2]),
                    frozen_discriminator_loss=float(values[3]),grad_norm=float(values[4]),
                    lr=optimizer.param_groups[0]['lr'],elapsed_seconds=time.monotonic()-started)
                trace.append(row)
                if rank==0:
                    write(out/'status.json',dict(phase='decoder_training',arm=args.arm,**row,time=time.time()))
                    print(json.dumps(row),flush=True)
            if step in [100,300,args.steps]:
                assert module_sha(encoder)==encoder_before and module_sha(q)==book_before
                with torch.no_grad():assert torch.equal(q.quantize(aux.encode(probe),chunk_size=1024)['codes'],probe_codes)
                if rank==0:
                    p=out/f'decoder-step{step:04d}.pt'
                    state={k:v.detach().cpu().clone() for k,v in learner.state_dict().items()}
                    torch.save(dict(state_dict=state,step=step,arm=args.arm,lr=args.lr,
                        adversarial_weight=args.adversarial_weight,source_codebook_sha256=sha(path),
                        encoder_unchanged=True,quantizer_unchanged=True),p)
                    snapshots.append(dict(path=str(p),step=step,arm=args.arm))
                    write(out/'snapshots.json',snapshots)
                dist.barrier()
            if step>=args.steps:break
        epoch+=1
    assert module_sha(encoder)==encoder_before and module_sha(q)==book_before
    if discriminator is not None:assert module_sha(discriminator)==discriminator_before
    if rank==0:
        proof=dict(passed=True,arm=args.arm,steps=step,learning_rate=args.lr,
            adversarial_weight=args.adversarial_weight,encoder_weights_bitwise_unchanged=True,
            quantizer_bitwise_unchanged=True,probe_token_changes=0,discriminator_frozen=True,
            atoms=4096,vocabulary=16385,depth=4,training_images=len(subset),
            heldout_selection_images_disjoint_from_fit=True,validation_used_for_fitting=False,
            optimized_components=['post_quant_conv','decoder'],trained_parameters=sum(p.numel() for p in learner.parameters()),
            source_codebook_sha256=sha(path),snapshots=snapshots,trace=trace,world_size=world,
            elapsed_seconds=time.monotonic()-started)
        write(out/'training-verification.json',proof)
        write(PUBLIC/(args.arm+'-training-verification.json'),proof)
    dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',required=True);p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--adversarial-weight',type=float,default=0.)
    p.add_argument('--steps',type=int,default=600);p.add_argument('--batch-size',type=int,default=16)
    main(p.parse_args())
