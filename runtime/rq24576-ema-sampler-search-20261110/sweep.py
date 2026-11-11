"""Screen EMA samplers, confirm with FID50k, and replicate on matched seeds."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_name(path.name+f'.{os.getpid()}.pending')
    pending.write_text(json.dumps(value,indent=2)+'\n');pending.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    plan=read(parser.parse_args().plan);local=Path(plan['local']);out=Path(plan['output']);base=Path(plan['base'])
    lock=(local/'sweep.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert read(local/'memory-preflight.json')['passed']
    os.environ.update(RQ_UPSTREAM_ROOT=str(base/'upstream'),TORCH_HOME=str(base/'torch-cache'),
        OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',WANDB_BASE_URL='https://api.wandb.ai',
        WANDB_DISABLE_GIT='true',WANDB_DIR=str(local/'wandb'),WANDB_CACHE_DIR=str(local/'wandb-cache'),
        WANDB_API_KEY=Path('/root/.config/laser/wandb.key').read_text().strip())
    sys.path[:0]=[str(base/'source'),str(base),str(base/'upstream')]
    import numpy as np
    import torch
    import wandb
    from src import rqvae_metrics as metrics
    from image_log import ImageLogger
    torch.set_num_threads(4)
    assert hashlib.sha256((base/'inputs/imagenet_256_train.npz').read_bytes()).hexdigest()==plan['reference_sha256']
    mean,covariance=metrics.load_reference_statistics(base/'inputs/imagenet_256_train.npz')
    std=np.sqrt(np.maximum(np.diag(covariance),1e-8))
    ref_second=covariance/np.outer(std,std)*(plan['reference_images']-1)/plan['reference_images']
    children=[];streams=[];run=None;logger=None

    def persist(name,value):
        write(local/name,value);write(out/name,value)

    def status(phase,**fields):
        workers=[read(local/f'worker-{r}.json') for r in range(8) if (local/f'worker-{r}.json').exists()]
        persist('status.json',dict(phase=phase,pid=os.getpid(),source_epoch=plan['source_epoch'],
            weights='ema',time=time.time(),workers=workers,**fields))

    def refresh():
        rows=[read(p) for p in (local/'confirm').glob('*.json')]
        if rows:
            best=min(rows,key=lambda r:r['fid'])
            record=dict(best_observed_single_seed=best,completed_full50k_evaluations=len(rows),
                independent_seed_validation_complete=False,published_target_fid=11.56,
                earlier_original_best_fid=12.65257290553177,run_url=run.url)
            persist('current-best.json',record)
            run.summary.update(dict(best_observed_fid50k=best['fid'],corresponding_inception_score=best['inception_score'],
                corresponding_is_split_std=best['inception_score_std'],best_observed_policy=best['base_policy']))

    def aggregate(stage,name):
        target=local/stage/(name+'.json')
        if target.exists():
            row=read(target)
            assert row['checkpoint_sha256']==plan['checkpoint_sha256'] and row['sampling']==plan['policies'][name]
            assert row['generated_images']==plan[stage+'_images']
            assert row['seed']==plan['policy_seeds'][name] and row['reference_sha256']==plan['reference_sha256']
            return row
        total=np.zeros(2048);cross=np.zeros((2048,2048));count=0;fourth=0.;probabilities=[];receipts=[]
        seed=plan['policy_seeds'][name]
        for rank in range(8):
            p=local/stage/name/f'rank-{rank}.npz';receipt=read(p.with_suffix('.json'))
            assert receipt['checkpoint_sha256']==plan['checkpoint_sha256'] and receipt['seed']==seed
            assert receipt['sampling']==plan['policies'][name]
            with np.load(p) as data:
                total+=data['total'];cross+=data['cross'];count+=int(data['count']);fourth+=float(data['fourth'])
                probabilities.append(torch.from_numpy(data['probabilities'].copy()))
            receipts.append(receipt)
        assert count==plan[stage+'_images']
        probabilities=torch.cat(probabilities)
        order=torch.randperm(count,generator=torch.Generator().manual_seed(plan['is_shuffle_seed']))
        is_mean,is_std=metrics.inception_score(probabilities[order])
        fake_mean=total/count;fake_cov=(cross-np.outer(total,total)/count)/(count-1)
        fid=metrics.frechet_distance(mean,covariance,fake_mean,fake_cov)
        assert np.isfinite(fid)
        row=dict(policy=name,base_policy=plan['case_base_policy'][name],sampling=plan['policies'][name],
            source_epoch=plan['source_epoch'],source_step=plan['source_step'],weights='ema',ema_decay=plan['ema_decay'],
            ema_updates=plan['ema_updates'],checkpoint_sha256=plan['checkpoint_sha256'],seed=seed,
            generated_images=count,reference_images=1281167,reference_split='train',reference_sha256=plan['reference_sha256'],
            inception_score=is_mean,inception_score_std=is_std,is_shuffle_seed=261002,world_size=8,
            sampling_batch_per_rank=plan['batch'],weights_frozen=True,rejection_sampling=False,
            initial_cuda_rng_sha256_by_rank=[r['initial_cuda_rng_sha256'] for r in receipts],
            max_worker_peak_gib=max(r['peak_allocated_gib'] for r in receipts))
        if stage=='screen':
            normalized_total=(total-count*mean)/std
            normalized_cross=(cross-np.outer(total,mean)-np.outer(mean,total)+count*np.outer(mean,mean))/np.outer(std,std)
            dimension=2048
            linear=2*(normalized_total@normalized_total-np.trace(normalized_cross))/(dimension*count*(count-1))
            quadratic=(np.square(normalized_cross).sum()-fourth)/(dimension**2*count*(count-1))
            mmd=linear+quadratic+np.square(ref_second).sum()/dimension**2-2*np.sum(normalized_cross*ref_second)/(dimension**2*count)
            row.update(screen_only_not_benchmark=True,fid10k_not_benchmark=fid,moment_mmd=float(mmd))
            run.log({'screen/config':name,'screen/fid10k_not_benchmark':fid,'screen/moment_mmd':float(mmd),
                'screen/inception_score10k':is_mean})
        else:
            row['fid']=fid
            if row['base_policy']=='current' and seed==261001:
                difference=abs(fid-plan['logged_baseline']['fid'])
                row['baseline_reproduction_absolute_difference']=difference
                assert difference<=.02,f'Baseline generation differs from recorded EMA FID: {difference}'
            run.log({'confirm/config':name,'confirm/fid50k':fid,'confirm/inception_score50k':is_mean,
                'confirm/is_split_std':is_std})
        persist(f'{stage}/{name}.json',row)
        if stage=='confirm':refresh()
        print(json.dumps(dict(event='aggregated',stage=stage,**row)),flush=True)
        return row

    def wait_for(stage,names):
        rows={}
        while len(rows)<len(names):
            logger.ready()
            for rank,child in enumerate(children):
                if child.poll() is not None and child.returncode!=0:raise RuntimeError(f'Worker {rank} exited {child.returncode}')
            for name in names:
                if name not in rows and all((local/stage/name/f'rank-{rank}.json').exists() for rank in range(8)):
                    status('computing_'+stage,policy=name);rows[name]=aggregate(stage,name)
            status(stage,completed_configs=list(rows),configs=len(names))
            if len(rows)<len(names):time.sleep(10)
        return [rows[name] for name in names]

    def stop(sig,_):raise InterruptedError(f'Sampler study received signal {sig}')
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,stop)
    try:
        out.mkdir(exist_ok=True);(local/'wandb').mkdir(exist_ok=True)
        for name in ['worker.py','sweep.py','image_log.py','prepare.py']:
            shutil.copyfile(HERE/name,out/name)
        run=wandb.init(entity='helloimlixin-rutgers',project='laser',id=out.name,resume='allow',
            name=f"RQ24576 EMA epoch{plan['source_epoch']} | {plan['study_kind']} | sampler search",
            job_type='ema-sampling-config-search',config=plan,dir=str(local/'wandb'),settings=wandb.Settings(init_timeout=60))
        persist('wandb.json',dict(url=run.url,id=run.id));logger=ImageLogger(plan,run)
        run.use_artifact('helloimlixin-rutgers/laser/imagenet-native-rq24576-tokenizer-20261108:v0')
        for name in ['plan.json','worker.py','sweep.py','image_log.py','prepare.py','memory-preflight.json','checkpoint-retention.json']:
            run.save(str(out/name),base_path=str(out),policy='now')
        persist('evaluator-sha256.json',{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ['worker.py','sweep.py','image_log.py']})
        for rank in range(8):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
            stream=(local/f'worker-{rank}.log').open('ab',buffering=0);streams.append(stream)
            children.append(subprocess.Popen([sys.executable,'-u',str(HERE/'worker.py'),'--plan',str(local/'plan.json'),'--rank',str(rank)],
                env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True))
        persist('worker-pids.json',dict(coordinator_pid=os.getpid(),workers=[c.pid for c in children]))
        screens=wait_for('screen',plan['screen_policies'])
        if screens:
            ordered=sorted(screens,key=lambda r:r['fid10k_not_benchmark'])
            chosen=['current']
            for row in ordered:
                if row['base_policy'] not in chosen:chosen.append(row['base_policy'])
                if len(chosen)==5:break
            special=[min(screens,key=lambda r:r['moment_mmd'])['base_policy'],
                max(screens,key=lambda r:r['inception_score'])['base_policy'],'depth_ramp']
            for name in special:
                if name not in chosen and len(chosen)<8:chosen.append(name)
        else:chosen=plan['prespecified_primary_policies']
        primary_names=[name+'__s261001' for name in chosen]
        persist('primary-selection.json',dict(policies=primary_names,checkpoint_sha256=plan['checkpoint_sha256']))
        primary=wait_for('confirm',primary_names)
        current=next(r for r in primary if r['base_policy']=='current')
        candidates=[r for r in primary if r['base_policy']!='current']
        minimum=min(candidates,key=lambda r:r['fid'])
        def balance(row):return min((current['fid']-row['fid'])/current['fid'],(row['inception_score']-current['inception_score'])/current['inception_score'])
        balanced=max(candidates,key=balance)
        selected=list(dict.fromkeys([minimum['base_policy'],balanced['base_policy']]))
        persist('primary-summary.json',dict(results=primary,lowest_fid=minimum,strongest_balance=balanced,selected_challengers=selected))
        names=[name+'__s'+str(seed) for seed in plan['replication_seeds'] for name in ['current']+selected]
        persist('replicate-selection.json',dict(policies=names,checkpoint_sha256=plan['checkpoint_sha256']))
        fresh=wait_for('confirm',names);all_rows=primary+fresh
        groups={}
        for name in ['current']+selected:
            rows=[r for r in all_rows if r['base_policy']==name];by_seed={r['seed']:r for r in rows}
            assert sorted(by_seed)==sorted(plan['generation_seeds'])
            groups[name]=dict(policy=name,sampling=plan['policies'][name],results=rows,
                fid_mean=float(np.mean([r['fid'] for r in rows])),fid_seed_std=float(np.std([r['fid'] for r in rows],ddof=1)),
                inception_score_mean=float(np.mean([r['inception_score'] for r in rows])),
                inception_score_seed_std=float(np.std([r['inception_score'] for r in rows],ddof=1)),
                within_seed_is_split_stds=[r['inception_score_std'] for r in rows])
        baseline={r['seed']:r for r in groups['current']['results']};decisions=[]
        for name in selected:
            pairs=[]
            for row in groups[name]['results']:
                control=baseline[row['seed']]
                assert row['initial_cuda_rng_sha256_by_rank']==control['initial_cuda_rng_sha256_by_rank']
                pairs.append(dict(seed=row['seed'],fid_improvement=control['fid']-row['fid'],is_improvement=row['inception_score']-control['inception_score']))
            matched=[r for r in pairs if r['seed'] in plan['replication_seeds']]
            fid_gain=float(np.mean([r['fid_improvement'] for r in matched]));is_gain=float(np.mean([r['is_improvement'] for r in matched]))
            passed=fid_gain>=.05 and all(r['fid_improvement']>0 for r in matched) and groups[name]['fid_mean']<groups['current']['fid_mean']
            decisions.append(dict(policy=name,paired=pairs,fresh_mean_fid_improvement=fid_gain,
                fresh_mean_is_improvement=is_gain,fid_improvement_confirmed=bool(passed),
                joint_improvement_confirmed=bool(passed and all(r['is_improvement']>0 for r in matched))))
        valid=[groups[d['policy']] for d in decisions if d['fid_improvement_confirmed']]
        recommended=min(valid,key=lambda r:r['fid_mean']) if valid else groups['current']
        best=min(all_rows,key=lambda r:r['fid'])
        for rank,child in enumerate(children):
            assert child.wait(timeout=120)==0
            for stem in ['verification','adapter-verification']:
                proof=read(local/f'{stem}-rank-{rank}.json');assert proof['passed'];persist(f'{stem}-rank-{rank}.json',proof)
        logger.table(screens,all_rows)
        report=dict(plan=plan,screen=screens,primary_confirmation=primary,all_completed_evaluations=all_rows,
            current=groups['current'],confirmed_challenger_summaries=[groups[n] for n in selected],decisions=decisions,
            recommended_policy=recommended['policy'],recommended_sampling=recommended['sampling'],
            recommended_fid_mean=recommended['fid_mean'],recommended_fid_seed_std=recommended['fid_seed_std'],
            recommended_inception_score_mean=recommended['inception_score_mean'],
            recommended_inception_score_seed_std=recommended['inception_score_seed_std'],best_observed_single_seed=best,
            image_logging=read(local/'image-logging.json'),weights='ema',weights_unmodified=True,tokenizer_unmodified=True,
            active_training_sampler_changed=False,independent_seed_validation_complete=True,published_target_fid=11.56,
            target_reached_three_seed_mean=recommended['fid_mean']<11.56,
            limits='10k screen FID and MMD only rank configurations; benchmark comparisons use full50k. IS split std and across-seed std are distinct. Three-seed variation is descriptive.')
        persist('report.json',report)
        run.summary.update(dict(recommended_sampling=recommended['sampling'],recommended_fid_mean=recommended['fid_mean'],
            recommended_fid_seed_std=recommended['fid_seed_std'],recommended_inception_score_mean=recommended['inception_score_mean'],
            recommended_inception_score_seed_std=recommended['inception_score_seed_std'],
            independent_seed_validation_complete=True,best_observed_fid50k=best['fid']))
        for name in ['report.json','primary-summary.json','image-logging.json']:run.save(str(out/name),base_path=str(out),policy='now')
        run.finish();run=None
        media=[]
        for attempt in range(6):
            online=wandb.Api(timeout=45).run('helloimlixin-rutgers/laser/'+out.name)
            media=[f.name for f in online.files() if f.name.startswith('media/images/')]
            if len(media)>=len(logger.records):break
            time.sleep(10)
        assert len(media)>=len(logger.records)
        persist('wandb-media-verification.json',dict(passed=True,uploaded_image_files=len(media),expected_panels=len(logger.records)))
        status('completed',best_fid=best['fid'],recommended_fid_mean=recommended['fid_mean'])
    except BaseException as error:
        status('failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill()
        for stream in streams:stream.close()
        if run is not None:run.finish(exit_code=1)


if __name__=='__main__':main()
