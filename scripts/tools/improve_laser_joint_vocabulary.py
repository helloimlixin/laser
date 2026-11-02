#!/usr/bin/env python3
"""Audit and fit an adaptive joint book with consistent IDs at every depth."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/workspace/Projects/laser')
sys.path.insert(0,str(ROOT))
BASE = Path('/tmp/laser-rq-joint-vocabulary-study-20261015')
PREVIOUS = ROOT/'outputs/imagenet-laser-joint-vocabulary-study-20261015'
OUT = ROOT/'outputs/imagenet-laser-joint-vocabulary-improvement-20261016'


def record(name, value):
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/name;temporary=p.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(p)


def initialize():
    import torch
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    return torch


def identity_report(torch, spec):
    atoms,values=spec['atom_ids'],spec['coefficients']
    return dict(vocabulary=atoms.shape[1],
        fraction_ids_with_different_atoms_across_depth=float((atoms[1:]!=atoms[:1]).any(0).float().mean()),
        fraction_ids_with_different_signs_across_depth=float((values[1:].sign()!=values[:1].sign()).any(0).float().mean()))


def audit():
    torch=initialize()
    import numpy as np
    from src.models.laser_joint_tokenizer import JointAtomCoefficientRQ
    reports={}
    for name in ['uniform64k-gain110','adaptive64k-gain110','uniform128k']:
        spec=torch.load(PREVIOUS/f'{name}-codebook.pt',map_location='cpu',weights_only=True)
        reports[name]=identity_report(torch,spec)
    spec=torch.load(PREVIOUS/'uniform64k-gain110-codebook.pt',weights_only=True)
    q=JointAtomCoefficientRQ(spec['dictionary'],spec['atom_ids'],spec['coefficients']).cuda()
    latents=np.load(BASE/'temperature-latents.npy',mmap_mode='r')
    sums=torch.zeros(4,6,dtype=torch.float64)
    with torch.inference_mode():
        generator=torch.Generator(device='cuda').manual_seed(261016)
        for first in range(0,len(latents),4):
            x=torch.from_numpy(latents[first:first+4].copy()).cuda()
            residual=x.reshape(-1,256).clone()
            for depth in range(4):
                for start in range(0,len(residual),128):
                    r=residual[start:start+128]
                    scores=q.scores(r,depth)
                    p=(scores/.0625).softmax(-1)
                    hard=scores.argmax(-1)
                    chosen=torch.multinomial(p,1,generator=generator).squeeze(-1)
                    hard_atom=q.atom_ids[depth,hard]
                    same_atom=q.atom_ids[depth][None,:]==hard_atom[:,None]
                    entropy=-(p*p.clamp_min(1e-30).log()).sum(-1)
                    closest_error=(r.square().sum(-1)-scores.max(-1).values).clamp_min(1e-8)
                    extra=(scores.max(-1).values-(p*scores).sum(-1)).clamp_min(0)
                    values=torch.stack((entropy.sum(),p.max(-1).values.sum(),
                        (p*same_atom).sum(),(p>0).float().sum(),closest_error.sum(),extra.sum())).double().cpu()
                    sums[depth]+=values
                    r.sub_(q.embed_depth(chosen,depth))
    n=len(latents)*64
    report=dict(codebook_identity=reports,temperature=.0625,training_images=len(latents),
        disjoint_from_fitting=True,stochastic_residual_prefixes=True,
        target_profiles=[dict(depth=d+1,mean_entropy=float(sums[d,0]/n),
            mean_maximum_probability=float(sums[d,1]/n),
            mean_mass_on_nearest_atom=float(sums[d,2]/n),
            mean_nonzero_probabilities=float(sums[d,3]/n),
            expected_extra_error_over_nearest_error=float(sums[d,5]/sums[d,4])) for d in range(4)],
        dense_target_bytes_at_microbatch64=64*64*4*65537*4,
        limitations=['These target statistics use128 held-out training images.',
            'Target statistics and vocabulary identity do not establish a change in generation FID.'])
    record('pipeline-audit.json',report)
    print(json.dumps(report),flush=True)


def fit(passes):
    torch=initialize()
    import numpy as np
    from src.models.laser_joint_tokenizer import JointAtomCoefficientRQ
    from src.scaled_atom_rq import continuous_matching_pursuit
    from src.training.laser_joint_vocabulary import (
        fit_coefficient_candidates,allocate_shared_coefficients,pack_codebook,
    )
    dictionary=torch.load(BASE/'dictionary.pt',weights_only=True).cuda()
    latents=np.load(BASE/'fit-latents.npy',mmap_mode='r')
    atoms,coefficients=[],[]
    started=time.monotonic()
    with torch.inference_mode():
        for first in range(0,len(latents),8):
            x=torch.from_numpy(latents[first:first+8].copy()).cuda()
            values=continuous_matching_pursuit(x,dictionary,depth=4)
            atoms.append(values['atoms']);coefficients.append(values['coefficients'])
        atoms,coefficients=torch.cat(atoms),torch.cat(coefficients)
        books,errors=[],[]
        for depth in range(4):
            b,e=fit_coefficient_candidates(atoms[...,depth],coefficients[...,depth],16384)
            books.append(b);errors.append(e)
            record('progress.json',dict(phase='initial_coefficients',depth=depth+1,time=time.time()))
        allocation=allocate_shared_coefficients(errors,65536)
        spec=pack_codebook(dictionary,books,[allocation]*4)
        q=JointAtomCoefficientRQ(spec['dictionary'],spec['atom_ids'],spec['coefficients']).cuda()
        prior=q.coefficients.clone()
        trace=[]
        for iteration in range(passes):
            count=torch.zeros_like(q.coefficients,dtype=torch.float64)
            total=torch.zeros_like(count)
            error=0.
            for first in range(0,len(latents),8):
                x=torch.from_numpy(latents[first:first+8].copy()).cuda()
                result=q.quantize(x,chunk_size=128,return_projections=True)
                for depth in range(4):
                    code=result['codes'][...,depth].flatten()
                    count[depth]+=torch.bincount(code,minlength=q.vocab_size)
                    total[depth]+=torch.bincount(code,weights=result['projections'][...,depth].flatten().double(),minlength=q.vocab_size)
                error+=float((x-result['quantized']).square().sum())
            updated=((total+4.*prior)/(count+4.)).float()
            updated[:,0]=0
            assert (updated[:,1:]*prior[:,1:]>0).all()
            q.coefficients.copy_(updated)
            row=dict(iteration=iteration+1,latent_mse_before_update=error/latents.size,
                     used_entries_by_depth=(count>0).sum(-1).tolist())
            trace.append(row);record('progress.json',dict(phase='lloyd',**row,time=time.time()))
            print(json.dumps(row),flush=True)
        manifest=json.loads((PREVIOUS/'calibration-data.json').read_text())
        spec.update(coefficients=q.coefficients.cpu(),
            source_stage1_checkpoint_sha256=manifest['stage1_sha256'],
            fitting_images=len(latents),fitting_split='train',allocation='shared-adaptive64k',
            allocation_counts={str(k):allocation.count(k) for k in [2,4,6,8]},
            same_atom_and_sign_meaning_at_every_depth=True)
        for gain in [1.,1.1]:
            candidate=dict(spec,coefficients=spec['coefficients']*gain,coefficient_gain=gain)
            identity=identity_report(torch,candidate)
            assert identity['fraction_ids_with_different_atoms_across_depth']==0
            assert identity['fraction_ids_with_different_signs_across_depth']==0
            name='shared-adaptive64k'+('' if gain==1. else '-gain110')
            torch.save(candidate,OUT/f'{name}-codebook.pt')
        record('shared-adaptive64k-fit.json',dict(trace=trace,passes=passes,prior_weight=4.,
            allocation_counts=spec['allocation_counts'],training_only=True,
            integer_identity_consistent_across_depths=True,elapsed_seconds=time.monotonic()-started))
        record('calibration-data.json',manifest)
    print('Shared-ID adaptive codebook fitted',flush=True)


def reference():
    torch=initialize()
    from src import rq_upstream
    from src.training.rq_imagenet_recipes import RQRecipe
    from src.training.rq_imagenet_runtime import TarImageNet
    from rqvae.utils.config import load_config
    from rqvae.img_datasets.transforms import create_transforms
    source=Path('/tmp/laser-original-rq-imagenet-20261014')
    aux,_=RQRecipe(source).tokenizer(torch.device('cuda:0'))
    config=load_config(source/'inputs/stage2-training-config.yaml')
    data=TarImageNet(source/'data','train',create_transforms(config.dataset,split='val'))
    indices=json.loads((PREVIOUS/'calibration-data.json').read_text())['temperature_indices']
    sums=torch.zeros(4,4,dtype=torch.float64)
    with torch.inference_mode():
        for first in range(0,len(indices),4):
            images=torch.stack([data[i][0] for i in indices[first:first+4]]).cuda()
            x=aux.encode(images)
            torch.manual_seed(261016+first)
            probabilities,codes=aux.quantizer.get_soft_codes(x,temp=.5,stochastic=True)
            residual=x.clone()
            book=aux.quantizer.codebooks[0]
            for depth in range(4):
                p=probabilities[...,depth,:]
                distances=book.compute_distances(residual)
                best=distances.min(-1).values.clamp_min(1e-8)
                extra=((p*distances).sum(-1)-best).clamp_min(0)
                sums[depth]+=torch.stack((-(p*p.clamp_min(1e-30).log()).sum(),
                    p.max(-1).values.sum(),best.sum(),extra.sum())).double().cpu()
                residual.sub_(book.embed(codes[...,depth]))
    n=len(indices)*64
    original=[dict(depth=d+1,mean_entropy=float(sums[d,0]/n),
        mean_maximum_probability=float(sums[d,1]/n),
        expected_extra_error_over_nearest_error=float(sums[d,3]/sums[d,2])) for d in range(4)]
    report=dict(training_images=len(indices),same_images_and_transforms=True,
        original_rq_temperature=.5,original_rq_target_profiles=original,
        joint=json.loads((OUT/'pipeline-audit.json').read_text())['target_profiles'],
        caveat='Depth-dependent uncertainty is present in the original recipe too; unequal entropy alone is not a defect.')
    record('target-profile-comparison.json',report)
    print(json.dumps(report),flush=True)


def temperatures():
    torch=initialize()
    import numpy as np
    from src.models.laser_joint_tokenizer import JointAtomCoefficientRQ
    from src.training.rq_imagenet_runtime import sha256
    spec_path=PREVIOUS/'uniform64k-gain110-codebook.pt'
    spec=torch.load(spec_path,weights_only=True)
    q=JointAtomCoefficientRQ(spec['dictionary'],spec['atom_ids'],spec['coefficients']).cuda()
    native=json.loads((OUT/'target-profile-comparison.json').read_text())['original_rq_target_profiles']
    x=torch.from_numpy(np.load(BASE/'temperature-latents.npy').copy()).cuda()
    residual=x.reshape(-1,256).clone()
    generator=torch.Generator(device='cuda').manual_seed(261016)
    chosen_temperatures,trace=[],[]
    with torch.inference_mode():
        for depth in range(4):
            scores=q.scores(residual,depth)
            maximum=scores.max(-1).values
            hard_error=(residual.square().sum(-1)-maximum).clamp_min(1e-8).sum()
            gap=maximum[:,None]-scores
            del scores,maximum
            target=native[depth]['expected_extra_error_over_nearest_error']
            def ratio(temperature):
                p=(-gap/temperature).softmax(-1)
                return float((p*gap).sum()/hard_error)
            low,high=.0001,.5
            assert ratio(low) < target < ratio(high)
            for _ in range(16):
                midpoint=(low+high)/2
                if ratio(midpoint)<target:low=midpoint
                else:high=midpoint
            temperature=(low+high)/2
            p=(-gap/temperature).softmax(-1)
            codes=torch.multinomial(p,1,generator=generator).squeeze(-1)
            residual.sub_(q.embed_depth(codes,depth))
            row=dict(depth=depth+1,temperature=temperature,
                reference_extra_error_ratio=target,calibrated_extra_error_ratio=ratio(temperature),
                mean_entropy=float(-(p*p.clamp_min(1e-30).log()).sum()/len(p)))
            chosen_temperatures.append(temperature);trace.append(row)
            del gap,p
            record('temperature-progress.json',dict(depths=trace))
            print(json.dumps(row),flush=True)
    report=dict(candidate='uniform64k-gain110',codebook_sha256=sha256(spec_path),
        temperatures=chosen_temperatures,old_global_temperature=.0625,
        calibration_images=len(x),calibration_split='held-out training images',
        matching='Per-depth expected quantization-error increase divided by nearest-code error, matching original RQ temperature0.5',
        sequential_stochastic_prefix_calibration=True,trace=trace,
        hard_tokenizer_unchanged=True,model_architecture_unchanged=True,
        generation_fid_improvement_measured=False,
        needs_independent_training_view_validation=True)
    record('per-depth-temperature-proposal.json',report)
    print(json.dumps(report),flush=True)


def validate_temperatures():
    torch=initialize()
    import math
    import numpy as np
    from src import rq_upstream
    from src.models.laser_joint_tokenizer import JointAtomCoefficientRQ
    from src.training.rq_imagenet_recipes import RQRecipe
    from src.training.rq_imagenet_runtime import TarImageNet
    from rqvae.utils.config import load_config
    from rqvae.img_datasets.transforms import create_transforms
    from scripts.tools.prepare_laser_joint_vocabulary import tokenizer
    from scripts.tools import prepare_laser_joint_vocabulary as study
    manifest=json.loads((PREVIOUS/'calibration-data.json').read_text())
    excluded=set(manifest['fit_indices']+manifest['temperature_indices'])
    indices=[int(i) for i in np.random.default_rng(26101665537).permutation(1281167)
             if int(i) not in excluded][:128]
    config=load_config(study.SOURCE/'inputs/stage2-training-config.yaml')
    data=TarImageNet(study.SOURCE/'data','train',create_transforms(config.dataset,split='train'),seed=261001)
    data.set_epoch(0)
    laser=tokenizer(torch.device('cuda:0'))
    native,_=RQRecipe('/tmp/laser-original-rq-imagenet-20261014').tokenizer(torch.device('cuda:0'))
    spec=torch.load(PREVIOUS/'uniform64k-gain110-codebook.pt',weights_only=True)
    q=JointAtomCoefficientRQ(spec['dictionary'],spec['atom_ids'],spec['coefficients']).cuda()
    proposal=json.loads((OUT/'per-depth-temperature-proposal.json').read_text())
    sums={name:torch.zeros(4,4,dtype=torch.float64) for name in ['original_rq','joint_global','joint_per_depth']}
    final_errors={name:torch.zeros(2,dtype=torch.float64) for name in sums}
    with torch.inference_mode():
        for first in range(0,len(indices),4):
            images=torch.stack([data[i][0] for i in indices[first:first+4]]).cuda()
            x=laser.quant_conv(laser.encoder(images)).permute(0,2,3,1).contiguous()
            original_x=native.encode(images)
            original_hard=native.quantizer(original_x)[0]
            joint_hard=q.quantize(x)['quantized']
            for mode in sums:
                torch.manual_seed(261016+first)
                if mode=='original_rq':
                    p,codes=native.quantizer.get_soft_codes(original_x,temp=.5,stochastic=True)
                    residual=original_x.clone()
                else:
                    p,codes=q.get_soft_codes(x,temp=.0625 if mode=='joint_global' else proposal['temperatures'])
                    residual=x.clone()
                for depth in range(4):
                    probabilities=p[...,depth,:].reshape(-1,p.shape[-1])
                    if mode=='original_rq':
                        book=native.quantizer.codebooks[0]
                        distance=book.compute_distances(residual).reshape_as(probabilities)
                        best=distance.min(-1).values
                        gap=distance-best[:,None]
                        update=book.embed(codes[...,depth])
                    else:
                        score=q.scores(residual.reshape(-1,256),depth)
                        maximum=score.max(-1).values
                        best=residual.reshape(-1,256).square().sum(-1)-maximum
                        gap=maximum[:,None]-score
                        update=q.embed_depth(codes[...,depth],depth)
                    sums[mode][depth]+=torch.stack((
                        -(probabilities*probabilities.clamp_min(1e-30).log()).sum(),
                        probabilities.max(-1).values.sum(),best.clamp_min(1e-8).sum(),
                        (probabilities*gap).sum())).double().cpu()
                    residual.sub_(update)
                hard_error=((original_x-original_hard) if mode=='original_rq' else (x-joint_hard)).square().sum()
                final_errors[mode]+=torch.stack((hard_error,residual.square().sum())).double().cpu()
    profiles={mode:[dict(depth=d+1,mean_entropy=float(values[d,0]/(len(indices)*64)),
        mean_maximum_probability=float(values[d,1]/(len(indices)*64)),
        expected_extra_error_over_nearest_error=float(values[d,3]/values[d,2])) for d in range(4)]
        for mode,values in sums.items()}
    distance={mode:sum(abs(math.log(profiles[mode][d]['expected_extra_error_over_nearest_error']/
        profiles['original_rq'][d]['expected_extra_error_over_nearest_error'])) for d in range(4))
        for mode in ['joint_global','joint_per_depth']}
    report=dict(validation_images=len(indices),validation_indices=indices,
        disjoint_from_codebook_fitting_and_temperature_calibration=True,
        transforms='Actual stage2 training RandomCrop/RandomHorizontalFlip, augmentation seed261001 epoch0',
        profiles=profiles,sum_absolute_log_error_to_reference=distance,
        final_sampled_to_hard_latent_mse_ratio={mode:float(v[1]/v[0]) for mode,v in final_errors.items()},
        passed=distance['joint_per_depth']<distance['joint_global'],
        conclusion_scope='Validation of the corruption profile only; generation FID improvement is untested')
    record('per-depth-temperature-validation.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='validation_indices'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['audit','fit','reference','temperatures','validate-temperatures'])
    parser.add_argument('--passes',type=int,default=4)
    args=parser.parse_args()
    if args.action=='audit':audit()
    elif args.action=='fit':fit(args.passes)
    elif args.action=='reference':reference()
    elif args.action=='temperatures':temperatures()
    else:validate_temperatures()
