"""Select decoder checkpoints on training holdout; verify winner on full validation."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path('/workspace/Projects/laser')
sys.path.insert(0,str(ROOT/'runtime/recon-4096-quality-20261105'))
from evaluate import SOURCE,book,module_sha,write,sha,TarImageNet,load_config,create_transforms,LPIPS,metrics
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader,Subset
from src.models.laser_joint_tokenizer import FrozenJointLaserTokenizer

BASE=Path('/tmp/laser-recon-4096-decoder-20261105')
PUBLIC=ROOT/'outputs/imagenet-native4096-decoder-quality-20261105'
DATA=Path('/tmp/laser-recon-4096-20261105/data')


class Decoder(nn.Module):
    def __init__(self,aux):
        super().__init__()
        self.post_quant_conv=copy.deepcopy(aux.backbone.post_quant_conv)
        self.decoder=copy.deepcopy(aux.backbone.decoder)
        self.eval().requires_grad_(False)

    def forward(self,latent):
        return self.decoder(self.post_quant_conv(latent.permute(0,3,1,2).contiguous()))


@torch.inference_mode()
def main(args):
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    device=torch.device('cuda',int(os.environ['LOCAL_RANK']));torch.cuda.set_device(device)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dist.init_process_group('nccl',device_id=device)
    manifest=json.loads((BASE/'manifest.json').read_text())
    candidates={'baseline4096':dict(path=None)}
    for arm in args.arms:
        proof=json.loads((BASE/arm/'training-verification.json').read_text());assert proof['passed']
        for x in proof['snapshots']:candidates[f"{arm}-step{x['step']:04d}"]=x
    if args.phase=='full':
        selected=json.loads((BASE/'screen-results.json').read_text())['selected']
        names=[selected]
    else:names=list(candidates)
    q=book(Path(manifest['source_codebook']),device)
    aux=FrozenJointLaserTokenizer(SOURCE/'inputs/stage1-model.pt',manifest['source_codebook']).to(device).eval().requires_grad_(False)
    encoder=nn.ModuleDict(dict(encoder=aux.backbone.encoder,quant_conv=aux.backbone.quant_conv))
    before=module_sha(encoder);qbefore=module_sha(q)
    decoders={}
    for name in names:
        d=Decoder(aux)
        if candidates[name]['path']:
            state=torch.load(candidates[name]['path'],map_location='cpu',weights_only=False,mmap=True)
            d.load_state_dict(state['state_dict'],strict=True);del state
        decoders[name]=d
    perceptual=LPIPS().to(device).eval().requires_grad_(False)
    metrics.INCEPTION_PATH=SOURCE/'upstream/rqvae/metrics/inception.py'
    inception=metrics.OriginalRQVAEInception().to(device).eval().requires_grad_(False)
    accum={n:metrics.DistributedOriginalRQVAEMetrics(device,inception=inception) for n in names}
    sums={n:torch.zeros(5,dtype=torch.float64,device=device) for n in names}
    config=load_config(SOURCE/'inputs/stage2-training-config.yaml')
    split='val' if args.phase=='full' else 'train'
    data=TarImageNet(DATA,split,create_transforms(config.dataset,split='val'))
    indices=list(range(len(data))) if args.phase=='full' else manifest['screen_indices']
    if args.phase=='full':assert len(indices)==50000
    loader=DataLoader(Subset(data,indices[rank::world]),batch_size=args.batch_size,num_workers=4,
        pin_memory=True,persistent_workers=True,prefetch_factor=1,multiprocessing_context='spawn')
    seen=0;start=time.monotonic()
    for batch,(images,_) in enumerate(loader):
        images=images.to(device,non_blocking=True)
        latent=aux.encode(images);quantized=q.quantize(latent,chunk_size=1024)['quantized']
        real=images.mul(.5).add(.5).clamp(0,1)
        features,_=inception(real);features=features.double();total=features.sum(0);cross=features.T@features
        if batch==0 and 'baseline4096' in names:
            torch.testing.assert_close(decoders['baseline4096'](quantized),aux.decode(quantized),rtol=0,atol=0)
        for name in names:
            reconstructed=decoders[name](quantized).float().clamp(-1,1)
            fake=reconstructed.mul(.5).add(.5);metric=accum[name]
            metric.real_sum.add_(total);metric.real_cross.add_(cross);metric.real_count.add_(len(images))
            metric.update(fake,real=False)
            error=(fake-real).square().mean((1,2,3));psnr=-10*error.clamp_min(1e-12).log10()
            percept=perceptual(reconstructed,images,reduction='none').flatten()
            sums[name]+=torch.stack([torch.tensor(len(images),device=device),error.sum(),psnr.sum(),percept.sum(),
                (latent-quantized).square().mean((1,2,3)).sum()]).double()
        seen+=len(images)
        if rank==0 and (batch==0 or batch%16==0):
            row=dict(phase=args.phase,images_on_rank=seen,target_on_rank=len(indices[rank::world]),
                candidates=names,elapsed_seconds=time.monotonic()-start,time=time.time())
            write(BASE/'status.json',row);print(json.dumps(row),flush=True)
    results={}
    for name in names:
        dist.all_reduce(sums[name]);count=int(sums[name][0]);assert count==len(indices)
        fid,_,_=accum[name].compute()
        assert int(accum[name].real_count)==int(accum[name].fake_count)==count
        results[name]=dict(rfid=fid,images=count,pixel_mse=float(sums[name][1]/count),
            psnr_db=float(sums[name][2]/count),lpips=float(sums[name][3]/count),
            latent_mse=float(sums[name][4]/count),decoder_checkpoint=candidates[name]['path'])
        if rank==0:print(json.dumps(dict(candidate=name,**results[name])),flush=True)
    assert module_sha(encoder)==before and module_sha(q)==qbefore
    if rank==0:
        report=dict(results=results,phase=args.phase,atoms=4096,vocabulary=16385,depth=4,
            encoder_and_quantizer_bitwise_unchanged=True,transformer_training_started=False,
            split='validation' if args.phase=='full' else 'heldout_training',
            validation_used_for_training_or_selection=False,images=len(indices),
            metric_backend='Original RQ-VAE Inception; same validation and FP32 inference protocol',
            source_codebook=manifest['source_codebook'],source_codebook_sha256=sha(manifest['source_codebook']),
            source_checkpoint_sha256=sha(SOURCE/'inputs/stage1-model.pt'),
            elapsed_seconds=time.monotonic()-start,time=time.time())
        if args.phase=='screen':report['selected']=min(names,key=lambda n:results[n]['rfid'])
        else:
            report.update(selected=names[0],threshold_rfid=4.73,
                beats_original_rq=results[names[0]]['rfid']<4.73,
                baseline_4096_rfid=4.908170663058399)
        tag=args.tag or args.phase
        write(BASE/(tag+'-results.json'),report);write(PUBLIC/(tag+'-results.json'),report)
        write(BASE/'status.json',dict(phase=args.phase+'_completed',time=time.time()))
    dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['screen','full']);p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--arms',nargs='+',default=['perceptual','texture']);p.add_argument('--tag')
    main(p.parse_args())
