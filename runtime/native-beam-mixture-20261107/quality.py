"""Measure full-support teacher mixtures before allocating AR training."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path('/tmp/laser-native-beam-mixture-20261107');BASE=ROOT/'frozen-mixture';OUT=ROOT/'quality'
sys.path[:0]=[str(BASE/'source'),str(BASE)]
os.environ['RQ_UPSTREAM_ROOT']=str(BASE/'upstream');os.environ['TORCH_HOME']=str(ROOT/'torch-cache')
os.environ['LASER_EXACT_BEAM_KERNELS']='1';os.environ['LASER_FAST_BEAM_TOPK']='0'
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader,Subset
from omegaconf import OmegaConf
from src import rq_upstream,rqvae_metrics as metrics
from src.models.rfid_preserved_joint_tokenizer import RFIDPreservedJointTokenizer
from src.training.rq_imagenet_runtime import TarImageNet
from rqvae.img_datasets.transforms import create_transforms
from rqvae.utils.config import load_config
from runtime.speed.exact_beam import install
install()
from runtime.mixture_teacher import get_targets,canonical_beam_codes,verify_bayes

WEIGHTS=[0.,.1,.15,.2,.25,.5,1.]


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.pending');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


@torch.inference_mode()
def main(args):
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    device=torch.device('cuda',int(os.environ['LOCAL_RANK']));torch.cuda.set_device(device);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    torch.set_float32_matmul_precision('highest');dist.init_process_group('nccl',device_id=device)
    torch.manual_seed(args.seed+rank);torch.cuda.manual_seed(args.seed+rank)
    assert verify_bayes()['passed']
    aux=RFIDPreservedJointTokenizer(BASE/'inputs/stage1-model.pt',BASE/'inputs/joint-codebook.pt',
                                   BASE/'inputs/reconstruction-codebook.pt').to(device).eval().requires_grad_(False)
    config=load_config(BASE/'inputs/stage2-training-config.yaml')
    split='train' if args.phase=='screen' else 'val'
    dataset=TarImageNet(BASE/'data',split,create_transforms(config.dataset,split='val'))
    if args.phase=='screen':
        indices=np.random.default_rng(261107).choice(len(dataset),2048,replace=False).tolist()
        if rank==0:write(OUT/'screen-indices.json',dict(split='train',indices=indices,seed=261107))
    else:indices=list(range(50000));assert len(dataset)==50000
    loader=DataLoader(Subset(dataset,indices[rank::world]),batch_size=args.batch_size,num_workers=4,
        pin_memory=True,persistent_workers=True,prefetch_factor=1,multiprocessing_context='spawn')
    metrics.INCEPTION_PATH=BASE/'upstream/rqvae/metrics/inception.py'
    inception=metrics.OriginalRQVAEInception().to(device).eval().requires_grad_(False)
    accum={str(a):metrics.DistributedOriginalRQVAEMetrics(device,inception=inception) for a in WEIGHTS}
    sums={str(a):torch.zeros(9,dtype=torch.float64,device=device) for a in WEIGHTS}
    proof=None;seen=0;start=time.monotonic()
    for batch,(images,_) in enumerate(loader):
        images=images.to(device,non_blocking=True);latent=aux.encode(images)
        beam=canonical_beam_codes(aux,latent)
        if batch==0:
            reference=aux.quantize_reconstruction(latent[:2])['codes']
            assert torch.equal(beam[:2],reference)
            with torch.random.fork_rng(devices=[device.index]):
                torch.cuda.manual_seed(261107);p0,c0=aux.quantizer.get_soft_codes(latent[:2],temp=.5,stochastic=True)
            with torch.random.fork_rng(devices=[device.index]):
                torch.cuda.manual_seed(261107);p1,c1=get_targets(aux,latent[:2],1.,beam_codes=beam[:2])
            assert torch.equal(p0,p1) and torch.equal(c0,c1)
            for a in [.1,.25,.5]:
                p,c,detail=get_targets(aux,latent[:2],a,beam_codes=beam[:2],return_details=True)
                assert torch.isfinite(p).all() and bool((p>=0).all())
                torch.testing.assert_close(p.sum(-1),torch.ones_like(p[...,0]),atol=2e-5,rtol=2e-5)
                torch.testing.assert_close(detail['joint_probability'],detail['expected_joint_probability'],atol=3e-7,rtol=3e-5)
            proof=dict(passed=True,native_endpoint_bitwise_equal=True,beam_endpoint_equals_saved_encoder=True,
                exact_mixture_joint_probability_verified=True,full_vocabulary_native_component=True,
                finite_beam_support_limit=False,near_best_error_mask=False,math=verify_bayes())
            if rank==0:write(OUT/'mixture-verification.json',proof)
            del p0,p1,c0,c1,p,c,detail
        real=images.mul(.5).add(.5).clamp(0,1)
        features,_=inception(real);features=features.double();total=features.sum(0);cross=features.T@features
        for a in WEIGHTS:
            name=str(a)
            # Identical seeds couple the fixed candidate weights for comparison.
            with torch.random.fork_rng(devices=[device.index]):
                torch.cuda.manual_seed(args.seed+rank+1000003*batch)
                targets,codes,detail=get_targets(aux,latent,a,beam_codes=beam,return_details=True)
            if a not in (0.,1.):
                torch.testing.assert_close(detail['joint_probability'],detail['expected_joint_probability'],atol=3e-7,rtol=3e-5)
            flat=targets.reshape(-1,16384)
            entropies=torch.empty(len(flat),device=device)
            for first in range(0,len(flat),256):
                p=flat[first:first+256]
                entropies[first:first+len(p)]=-(p*p.clamp_min(1e-30).log()).sum(-1)
            by_depth=entropies.reshape(*codes.shape).mean((1,2)).sum(0)
            fake=aux.decode_code(codes).float().mul(.5).add(.5).clamp(0,1)
            metric=accum[name];metric.real_sum.add_(total);metric.real_cross.add_(cross);metric.real_count.add_(len(images))
            metric.update(fake,real=False)
            s=sums[name];s[0]+=len(images);s[1]+=entropies.sum();s[2]+=entropies.numel()
            s[3]+=(fake-real).square().mean((1,2,3)).sum()
            s[4]+=detail['native_branch'].float().sum()
            s[5:9]+=by_depth.double()
            del targets,codes,detail,flat,entropies,fake,p
        seen+=len(images)
        if rank==0 and (batch==0 or batch%16==0):
            row=dict(phase=args.phase,images_on_rank=seen,target_on_rank=len(indices[rank::world]),
                elapsed_seconds=time.monotonic()-start,time=time.time(),weights=WEIGHTS)
            write(OUT/'status.json',row);print(json.dumps(row),flush=True)
    results={}
    for name,metric in accum.items():
        fid,_,_=metric.compute();dist.all_reduce(sums[name]);s=sums[name]
        results[name]=dict(rfid=float(fid),images=int(s[0]),teacher_entropy=float(s[1]/s[2]),
            pixel_mse=float(s[3]/s[0]),native_fraction_observed=float(s[4]/(64*s[0])),
            entropy_by_depth=(s[5:9]/s[0]).tolist())
    if rank==0:
        report=dict(phase=args.phase,results=results,predeclared_native_weights=WEIGHTS,
            vocabulary=16384,native_temperature=.5,mixture='a native + (1-a) beam-best',
            prefixes_and_labels_same_distribution=True,full_vocabulary_native_component=True,
            CNN_dictionary_and_coefficients_unchanged=True,measurement_seed=args.seed,
            independent_replication=bool(args.report_name))
        if args.phase=='full':
            eligible=[a for a in WEIGHTS if a>0 and results[str(a)]['rfid']<4.73 and results[str(a)]['teacher_entropy']>=.2]
            report.update(passed=bool(eligible),selected_native_weight=max(eligible) if eligible else None,
                selection_gates=dict(actual_teacher_rfid_below=4.73,teacher_entropy_at_least=.2),
                validation_images=50000,validation_used_for_predeclared_weight_selection=True,
                validation_used_for_gradient_training=False)
            if eligible and not args.report_name:
                chosen=max(eligible);record=results[str(chosen)]
                recipe=OmegaConf.load(BASE/'inputs/recipe.yaml')
                recipe.targets.native_weight=chosen;recipe.targets.measured_entropy=record['teacher_entropy']
                recipe.targets.calibration='predeclared weight screen, full50k rFID and entropy gates'
                recipe.tokenizer.teacher_sampled_rfid=record['rfid']
                OmegaConf.save(recipe,BASE/'inputs/recipe.yaml')
        write(OUT/(args.report_name or args.phase+'-results.json'),report);print(json.dumps(report),flush=True)
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['screen','full'],required=True)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--seed',type=int,default=261107)
    parser.add_argument('--weights',type=float,nargs='+')
    parser.add_argument('--report-name')
    args=parser.parse_args()
    if args.weights is not None:WEIGHTS=args.weights
    main(args)
