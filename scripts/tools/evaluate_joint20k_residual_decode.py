#!/usr/bin/env python3
"""Compare residual rendering on shared generated codes, then confirm FID50k."""
from datetime import timedelta
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path('/workspace/Projects/laser')
BASE=Path('/tmp/laser-rq-joint20k-depth5-rqnoise-k2048-rngfix-8h100-imagenet-20261017')
OUT=ROOT/'outputs/imagenet-laser-joint20k-depth5-residual-decode-test-20261017'
LOCAL=Path('/tmp/laser-joint20k-residual-decode-test-20261017')


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
    spec=importlib.util.spec_from_file_location('frozen_residual_decode',OUT/'source/joint_residual_decode.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    decode=module.decode_residuals
    rank,world,local=[int(os.environ[k]) for k in ('RANK','WORLD_SIZE','LOCAL_RANK')]
    assert world==8
    device=torch.device('cuda',local);torch.cuda.set_device(device);torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(16*2**30/torch.cuda.get_device_properties(device).total_memory,device)
    dist.init_process_group('nccl',device_id=device,timeout=timedelta(minutes=60))
    plan=json.loads((OUT/'plan.json').read_text())
    profile=JointLaserRecipe(BASE)
    saved=torch.load(plan['checkpoint'],map_location='cpu',mmap=True,weights_only=False)
    profile.verify_resume(saved)
    assert saved['global_step']==2504 and saved['next_epoch']==4
    with torch.device('meta'):model=profile.model(OmegaConf.create(saved['config']['arch']))
    model.load_state_dict(saved['state_dict'],strict=True,assign=True)
    del saved
    model=model.requires_grad_(False).eval().to(device)
    aux,_=profile.tokenizer(device);profile.adapt(model,aux)
    versions=[p._version for p in model.parameters()]
    frozen=profile.digest(aux)
    metrics.INCEPTION_PATH=BASE/'upstream/rqvae/metrics/inception.py'
    inception=metrics.OriginalRQVAEInception().eval().requires_grad_(False).to(device)
    policies=plan['policies']
    run=None
    if rank==0:
        import wandb
        os.environ['WANDB_API_KEY']=Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
        run=wandb.init(entity='helloimlixin-rutgers',project='laser',id=OUT.name,
            name='LASER depth5 | fixed weights | residual decoding test',resume='allow',dir=str(LOCAL/'wandb'),
            config=dict(plan,wandb_id=OUT.name),job_type='fixed-checkpoint-decoder-ablation')
        run.define_metric('screen/index');run.define_metric('screen/*',step_metric='screen/index')
        run.define_metric('confirm/index');run.define_metric('confirm/fid50k',step_metric='confirm/index')
        write_json(OUT/'wandb.json',dict(url=run.url,id=run.id))
        for f in (OUT/'source').iterdir():run.save(str(f),base_path=str(OUT),policy='now')
    dist.barrier()
    mean,cov=metrics.load_reference_statistics(BASE/'inputs/imagenet_256_train.npz')
    std=np.sqrt(np.maximum(np.diag(cov),1.e-8))
    refsecond=cov/np.outer(std,std)*(1281166/1281167)
    mean_t=torch.as_tensor(mean,device=device,dtype=torch.float64)
    std_t=torch.as_tensor(std,device=device,dtype=torch.float64)
    per_rank=plan['screen_images']//world
    screen={name:dict(total=torch.zeros(2048,device=device,dtype=torch.float64),
        cross=torch.zeros(2048,2048,device=device,dtype=torch.float64),
        fourth=torch.zeros((),device=device,dtype=torch.float64),probs=[]) for name in policies}
    torch.manual_seed(plan['screen_seed']+rank);torch.cuda.manual_seed(plan['screen_seed']+rank)
    code_digest=__import__('hashlib').sha256()
    with torch.inference_mode():
        for first in range(0,per_rank,128):
            count=min(128,per_rank-first)
            labels=(torch.arange(first,first+count,device=device)*world+rank)%1000
            codes=profile.sample(model,aux,count,labels)
            code_digest.update(codes.cpu().numpy().tobytes())
            if first==0:
                native=aux.decode_code(codes[:2])
                identity=decode(aux,codes[:2],policies['baseline'])
                assert torch.equal(native,identity)
                write_json(OUT/f'identity-rank{rank}.json',dict(passed=True,decoder_pixels_exact=True))
            for name,scales in policies.items():
                state=screen[name]
                for part in codes.split(8):
                    images=decode(aux,part,scales).float().mul(.5).add(.5).clamp(0,1)
                    features,logits=inception(images,return_logits=True)
                    z=(features.double()-mean_t)/std_t
                    state['total']+=z.sum(0);state['cross'].addmm_(z.T,z)
                    state['fourth']+=z.square().sum(-1).square().sum()
                    state['probs'].append(logits.softmax(-1))
                if first==0 and rank==0:
                    from torchvision.utils import save_image
                    path=OUT/f'screen/{name}/samples.png';path.parent.mkdir(parents=True,exist_ok=True)
                    save_image(images[:8],path,nrow=4)
            if rank==0:write_json(OUT/'status.json',dict(phase='screening',generated=(first+count)*world,
                target=plan['screen_images'],policies=len(policies),time=time.time()))
        results=[]
        for index,(name,state) in enumerate(screen.items()):
            for tensor in [state['total'],state['cross'],state['fourth']]:dist.all_reduce(tensor)
            probs=torch.cat(state['probs'])
            gathered=[torch.empty_like(probs) for _ in range(world)]
            dist.all_gather(gathered,probs)
            probs=torch.cat(gathered)
            torch.manual_seed(plan['screen_seed'])
            ismean,isstd=metrics.inception_score(probs[torch.randperm(len(probs),device=device)])
            row=dict(policy=name,scales=policies[name],images=plan['screen_images'],not_fid=True,
                inception_score=ismean,inception_score_std=isstd,
                moment_mmd=moment_mmd(state['total'].cpu().numpy(),state['cross'].cpu().numpy(),
                    float(state['fourth']),plan['screen_images'],refsecond))
            results.append(row)
            if rank==0:
                write_json(OUT/f'screen/{name}/metrics.json',row)
                run.log({'screen/index':index,'screen/policy':name,'screen/moment_mmd':row['moment_mmd'],
                         'screen/inception_score':ismean})
        challengers=[r for r in results if r['policy']!='baseline']
        best_mmd=min(challengers,key=lambda r:r['moment_mmd'])['policy']
        best_is=max((r for r in challengers if r['policy']!=best_mmd),key=lambda r:r['inception_score'])['policy']
        selected=[best_mmd,best_is]
        write_json(OUT/f'codes-rank{rank}.json',dict(sha256=code_digest.hexdigest(),shared_by_all_policies=True))
        if rank==0:write_json(OUT/'screen-summary.json',dict(results=results,selected=selected,not_fid=True))
        del screen,state,features,logits,z,probs,gathered
        torch.cuda.empty_cache()

        def confirm(names,seed,index_start):
            trackers={name:metrics.DistributedOriginalRQVAEMetrics(device,compute_inception_score=True,
                reference_stats_path=BASE/'inputs/imagenet_256_train.npz',inception=inception) for name in names}
            torch.manual_seed(seed+rank);torch.cuda.manual_seed(seed+rank)
            count_rank=50000//world+(rank<50000%world)
            started=time.monotonic()
            for first in range(0,count_rank,128):
                count=min(128,count_rank-first)
                labels=(torch.arange(first,first+count,device=device)*world+rank)%1000
                codes=profile.sample(model,aux,count,labels)
                for name,tracker in trackers.items():
                    for part in codes.split(8):
                        images=decode(aux,part,policies[name]).float().mul(.5).add(.5).clamp(0,1)
                        tracker.update(images,real=False)
                if rank==0:write_json(OUT/'status.json',dict(phase='fid50k',seed=seed,policies=names,
                    generated=(first+count)*world,target=50000,time=time.time()))
            rows=[]
            for index,(name,tracker) in enumerate(trackers.items(),index_start):
                assert int(tracker.fake_count)==count_rank
                # Give every policy the same IS partition; FID uses no RNG.
                torch.manual_seed(seed)
                fid,ismean,isstd=tracker.compute(shuffle_for_inception=True)
                row=dict(policy=name,scales=policies[name],seed=seed,fid=fid,
                    inception_score=ismean,inception_score_std=isstd,generated_images=50000,
                    reference_images=1281167,rejection_sampling=False,weights_modified=False,
                    training_updates_added=0,elapsed_seconds=time.monotonic()-started)
                rows.append(row)
                if rank==0:
                    write_json(OUT/f'confirm/{seed}/{name}.json',row)
                    run.log({'confirm/index':index,'confirm/seed':seed,'confirm/policy':name,'confirm/fid50k':fid})
            return rows

        confirmations=confirm(['baseline',*selected],plan['confirmation_seed'],0)
        baseline=confirmations[0]['fid']
        if abs(baseline-plan['baseline_fid'])>1.e-4:
            raise RuntimeError(f'Baseline FID did not reproduce: {baseline} vs {plan["baseline_fid"]}')
        best=min(confirmations,key=lambda r:r['fid'])
        first_gain=baseline-best['fid']
        second=[]
        if best['policy']!='baseline' and first_gain>=.5:
            second=confirm(['baseline',best['policy']],plan['replication_seed'],len(confirmations))
        second_gain=second[0]['fid']-second[1]['fid'] if second else None
        validated=bool(second and second_gain>=.5)
        assert versions==[p._version for p in model.parameters()]
        assert profile.digest(aux)==frozen
        result=dict(completed=True,screen=results,confirmation=confirmations,replication=second,
            baseline_fid_reproduced=True,observed_best_policy=best['policy'],
            first_seed_improvement=first_gain,second_seed_improvement=second_gain,
            validated_improvement=validated,selected_scales=best['scales'] if validated else None,
            model_weights_unchanged=True,tokenizer_unchanged=True,training_updates_added=0,
            limits='One checkpoint. Two generation seeds for a candidate passing the first FID check; no claim about later checkpoints.')
        if rank==0:
            write_json(OUT/'report.json',result)
            run.summary.update({'evaluation/validated_improvement':validated,
                'evaluation/first_seed_gain':first_gain,'evaluation/second_seed_gain':second_gain,
                'evaluation/best_policy':best['policy']})
            run.save(str(OUT/'report.json'),base_path=str(OUT),policy='now');run.finish()
            write_json(OUT/'status.json',dict(phase='completed',validated_improvement=validated,time=time.time()))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
