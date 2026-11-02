#!/usr/bin/env python3
"""Test constrained likelihood-ratio sampling with two frozen RQ checkpoints."""
from datetime import timedelta
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path('/workspace/Projects/laser')
BASE=Path('/tmp/laser-rq-joint20k-depth5-rqnoise-k2048-rngfix-8h100-imagenet-20261017')
OUT=ROOT/'outputs/imagenet-laser-joint20k-depth5-contrastive-test-20261017'
LOCAL=Path('/tmp/laser-joint20k-contrastive-test-20261017')


def main():
    os.environ['RQ_UPSTREAM_ROOT']=str(BASE/'upstream')
    os.environ['TORCH_HOME']=str(BASE/'torch-cache')
    sys.path[:0]=[str(OUT/'source'),str(BASE/'source'),str(BASE/'upstream'),str(BASE),str(BASE/'runtime')]
    import numpy as np
    import torch
    import torch.distributed as dist
    from omegaconf import OmegaConf
    from src.training.laser_joint_recipe import JointLaserRecipe
    from src.training.rq_imagenet_runtime import write_json
    from src import rqvae_metrics as metrics
    from sweep_compound512_samplers import moment_mmd
    spec=importlib.util.spec_from_file_location('frozen_contrastive_sampling',OUT/'source/rq_contrastive_sampling.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rank,world,local=[int(os.environ[k]) for k in ('RANK','WORLD_SIZE','LOCAL_RANK')]
    assert world==8
    device=torch.device('cuda',local);torch.cuda.set_device(device);torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(24*2**30/torch.cuda.get_device_properties(device).total_memory,device)
    dist.init_process_group('nccl',device_id=device,timeout=timedelta(minutes=60))
    plan=json.loads((OUT/'plan.json').read_text())
    profile=JointLaserRecipe(BASE)
    models=[];identities=[]
    for name in ['expert','amateur']:
        saved=torch.load(plan[name]['checkpoint'],map_location='cpu',mmap=True,weights_only=False)
        profile.verify_resume(saved)
        assert saved['global_step']==plan[name]['step']
        identities.append((saved['joint_codebook_sha256'],saved['stage1_checkpoint_sha256']))
        with torch.device('meta'):model=profile.model(OmegaConf.create(saved['config']['arch']))
        model.load_state_dict(saved['state_dict'],strict=True,assign=True)
        models.append(model.requires_grad_(False).eval().to(device))
        del saved
    assert identities[0]==identities[1]
    expert,amateur=models
    aux,_=profile.tokenizer(device)
    profile.adapt(expert,aux);profile.adapt(amateur,aux)
    versions=[[p._version for p in model.parameters()] for model in models]
    frozen=profile.digest(aux)
    with torch.inference_mode():
        labels=torch.tensor([269,612],device=device)
        torch.manual_seed(261018231+rank)
        expected=profile.sample(expert,aux,2,labels)
        wrapper=module.ContrastiveCachedPrior(expert,amateur,top_k=2048)
        torch.manual_seed(261018231+rank)
        actual=profile.sample(expert,aux,2,labels)
        assert torch.equal(actual,expected)
        write_json(OUT/f'zero-guidance-parity-rank{rank}.json',dict(passed=True,
            generated_codes_exact=True,events_verified=640))
    metrics.INCEPTION_PATH=BASE/'upstream/rqvae/metrics/inception.py'
    inception=metrics.OriginalRQVAEInception().eval().requires_grad_(False).to(device)
    mean,cov=metrics.load_reference_statistics(BASE/'inputs/imagenet_256_train.npz')
    std=np.sqrt(np.maximum(np.diag(cov),1.e-8))
    reference_second=cov/np.outer(std,std)*(1281166/1281167)
    mean_t=torch.as_tensor(mean,device=device,dtype=torch.float64)
    std_t=torch.as_tensor(std,device=device,dtype=torch.float64)
    run=None
    if rank==0:
        import wandb
        os.environ['WANDB_API_KEY']=Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
        run=wandb.init(entity='helloimlixin-rutgers',project='laser',id=OUT.name,
            name='LASER depth5 | frozen expert/amateur contrastive sampling',resume='allow',
            dir=str(LOCAL/'wandb'),config=dict(plan,wandb_id=OUT.name),job_type='fixed-checkpoint-contrastive-ablation')
        run.define_metric('screen/index');run.define_metric('screen/*',step_metric='screen/index')
        run.define_metric('confirm/index');run.define_metric('confirm/fid50k',step_metric='confirm/index')
        write_json(OUT/'wandb.json',dict(url=run.url,id=run.id))
    dist.barrier()
    screen=[plan['baseline_screen']]
    with torch.inference_mode():
        for index,strength in enumerate(plan['strengths'],1):
            wrapper.strength=strength
            torch.manual_seed(plan['screen_seed']+rank);torch.cuda.manual_seed(plan['screen_seed']+rank)
            total=torch.zeros(2048,device=device,dtype=torch.float64)
            cross=torch.zeros(2048,2048,device=device,dtype=torch.float64)
            fourth=torch.zeros((),device=device,dtype=torch.float64)
            probabilities=[]
            local_count=plan['screen_images']//world
            for first in range(0,local_count,128):
                count=min(128,local_count-first)
                labels=(torch.arange(first,first+count,device=device)*world+rank)%1000
                codes=profile.sample(expert,aux,count,labels)
                for part in codes.split(8):
                    images=profile.decode(aux,part).float().mul(.5).add(.5).clamp(0,1)
                    features,logits=inception(images,return_logits=True)
                    z=(features.double()-mean_t)/std_t
                    total+=z.sum(0);cross.addmm_(z.T,z);fourth+=z.square().sum(-1).square().sum()
                    probabilities.append(logits.softmax(-1))
                if rank==0:write_json(OUT/'status.json',dict(phase='screening',strength=strength,
                    generated=(first+count)*world,target=plan['screen_images'],time=time.time()))
            for value in [total,cross,fourth]:dist.all_reduce(value)
            p=torch.cat(probabilities);gathered=[torch.empty_like(p) for _ in range(world)]
            dist.all_gather(gathered,p);p=torch.cat(gathered)
            torch.manual_seed(plan['screen_seed'])
            ismean,isstd=metrics.inception_score(p[torch.randperm(len(p),device=device)])
            row=dict(strength=strength,not_fid=True,images=plan['screen_images'],inception_score=ismean,
                inception_score_std=isstd,moment_mmd=moment_mmd(total.cpu().numpy(),cross.cpu().numpy(),
                    float(fourth),plan['screen_images'],reference_second))
            screen.append(row)
            if rank==0:
                write_json(OUT/f'screen/strength{strength}.json',row)
                run.log({'screen/index':index,'screen/strength':strength,'screen/moment_mmd':row['moment_mmd'],
                         'screen/inception_score':ismean})
            del total,cross,fourth,probabilities,p,gathered
        challengers=screen[1:]
        best_mmd=min(challengers,key=lambda r:r['moment_mmd'])['strength']
        best_is=max((r for r in challengers if r['strength']!=best_mmd),key=lambda r:r['inception_score'])['strength']
        selected=[best_mmd,best_is]
        if rank==0:write_json(OUT/'screen-summary.json',dict(results=screen,selected=selected,not_fid=True))

        def confirm(strength,seed,index):
            wrapper.strength=strength
            tracker=metrics.DistributedOriginalRQVAEMetrics(device,compute_inception_score=True,
                reference_stats_path=BASE/'inputs/imagenet_256_train.npz',inception=inception)
            torch.manual_seed(seed+rank);torch.cuda.manual_seed(seed+rank)
            local_count=50000//world+(rank<50000%world)
            started=time.monotonic()
            for first in range(0,local_count,128):
                count=min(128,local_count-first)
                labels=(torch.arange(first,first+count,device=device)*world+rank)%1000
                codes=profile.sample(expert,aux,count,labels)
                for part in codes.split(8):
                    images=profile.decode(aux,part).float().mul(.5).add(.5).clamp(0,1)
                    tracker.update(images,real=False)
                if rank==0:write_json(OUT/'status.json',dict(phase='fid50k',strength=strength,seed=seed,
                    generated=(first+count)*world,target=50000,time=time.time()))
            torch.manual_seed(seed)
            fid,ismean,isstd=tracker.compute(shuffle_for_inception=True)
            row=dict(strength=strength,seed=seed,fid=fid,inception_score=ismean,inception_score_std=isstd,
                generated_images=50000,reference_images=1281167,rejection_sampling=False,
                training_updates_added=0,elapsed_seconds=time.monotonic()-started)
            if rank==0:
                write_json(OUT/f'confirm/seed{seed}-strength{strength}.json',row)
                run.log({'confirm/index':index,'confirm/strength':strength,'confirm/seed':seed,'confirm/fid50k':fid})
            return row

        confirmations=[dict(strength=0.,seed=plan['confirmation_seed'],fid=plan['baseline_fid'],
            generated_images=50000,reference_images=1281167,reused_after_exact_zero_strength_parity=True)]
        for index,strength in enumerate(selected,1):confirmations.append(confirm(strength,plan['confirmation_seed'],index))
        best=min(confirmations,key=lambda r:r['fid'])
        gain=plan['baseline_fid']-best['fid'];replication=[]
        if best['strength']!=0 and gain>=.5:
            replication=[confirm(0.,plan['replication_seed'],3),confirm(best['strength'],plan['replication_seed'],4)]
        second_gain=replication[0]['fid']-replication[1]['fid'] if replication else None
        validated=bool(replication and second_gain>=.5)
        assert versions==[[p._version for p in model.parameters()] for model in models]
        assert profile.digest(aux)==frozen
        wrapper.restore()
        result=dict(completed=True,screen=screen,confirmation=confirmations,replication=replication,
            selected_strength=best['strength'] if validated else 0.,first_seed_improvement=gain,
            second_seed_improvement=second_gain,validated_improvement=validated,
            model_weights_unchanged=True,tokenizer_unchanged=True,training_updates_added=0,
            limits='One epoch-four expert and one epoch-two amateur; two generation seeds for a promising candidate. Experimental adaptation of contrastive decoding to joint image codes.')
        if rank==0:
            write_json(OUT/'report.json',result)
            run.summary.update({'evaluation/validated_improvement':validated,'evaluation/first_seed_gain':gain,
                'evaluation/second_seed_gain':second_gain,'evaluation/selected_strength':result['selected_strength']})
            run.save(str(OUT/'report.json'),base_path=str(OUT),policy='now');run.finish()
            write_json(OUT/'status.json',dict(phase='completed',validated_improvement=validated,time=time.time()))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
