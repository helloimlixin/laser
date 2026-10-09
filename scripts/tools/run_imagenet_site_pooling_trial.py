"""Queue a matched pooling trial after baseline epoch 12, then resume plain AR."""
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

BASE = Path('/tmp/laser-imagenet-site-pooling-20261007')
OUT = Path('/workspace/Projects/laser/outputs/imagenet-site-pooling-trial-20261007')
ROOT = Path('/workspace/Projects/laser')
SOURCE = Path('/tmp/laser-imagenet-pair-memory-investigation-20261007/source')
PREVIOUS = Path('/tmp/laser-imagenet-pair-memory-investigation-20261007')
ASSETS = Path('/tmp/laser-imagenet-classcond-20261007')
PRODUCTION = ROOT / 'outputs/imagenet-rfid421-classcond-8h100-20261007'
CACHE = Path('/tmp/laser-imagenet-pair-memory-20261007/checkpoint-cache')
BRANCHES = ('sum','mlp','attention')
SEEDS = (261001,271001)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def record(**value):
    value.update(time=time.time(),supervisor_pid=os.getpid())
    write(OUT/'launch-status.json',value)
    with (OUT/'launch-history.jsonl').open('a') as stream:
        stream.write(json.dumps(value)+'\n')
    print(json.dumps(value),flush=True)


def prepare():
    import torch
    import yaml
    BASE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    for name in ('wandb','checkpoint-staging','matplotlib'):(BASE/name).mkdir(exist_ok=True)
    assert not (BASE/'source').exists(), 'Preparation is single-use; inspect the existing trial.'
    shutil.copytree(SOURCE,BASE/'source',copy_function=shutil.copyfile,
                    ignore=shutil.ignore_patterns('__pycache__'))
    # Change only the pooling interface in the exact source used by the baseline.
    path=BASE/'source/src/models/rqtransformer/transformers.py'
    code=path.read_text()
    marker='    def classify_head_outputs(self, head_outputs):'
    assert code.count(marker)==1
    code=code.replace(marker,'    def pool_spatial_pairs(self, pair_embeddings):\n'
        '        """Summarize a completed site before its shifted spatial input."""\n'
        '        return pair_embeddings.sum(dim=-2)\n\n'+marker)
    old='xs_emb = xs_emb.sum(dim=-2) + self.pos_emb_hw[:, :seq_len, :]'
    assert code.count(old)==2
    path.write_text(code.replace(old,'xs_emb = self.pool_spatial_pairs(xs_emb) + self.pos_emb_hw[:, :seq_len, :]'))
    path=BASE/'source/src/training/rqtransformer.py';code=path.read_text()
    old='spatial_inputs = xs_emb.sum(dim=-2) + self.pos_emb_hw[:, :seq_len]'
    assert code.count(old)==1
    path.write_text(code.replace(old,'spatial_inputs = self.pool_spatial_pairs(xs_emb) + self.pos_emb_hw[:, :seq_len]'))
    for relative in ['src/models/site_pair_pooling.py','src/training/site_pair_pooling.py']:
        shutil.copyfile(ROOT/relative,BASE/'source'/relative)
    for name in ['imagenet_site_pooling_entry.py','run_imagenet_site_pooling_trial.py']:
        shutil.copyfile(ROOT/'scripts/tools'/name,BASE/name)
        shutil.copyfile(ROOT/'scripts/tools'/name,OUT/name)
    sys.path[:0]=[str(BASE/'source'),str(BASE/'source/runtime')]
    from src.models.site_pair_pooling import SitePairPooling
    torch.manual_seed(261007)
    attention=SitePairPooling(1536,4,width=480,heads=8,mode='attention')
    mlp=SitePairPooling(1536,4,width=480,heads=8,mode='mlp')
    shared=set(attention.state_dict()) & set(mlp.state_dict())
    state=mlp.state_dict()
    for name in shared:state[name]=attention.state_dict()[name].clone()
    mlp.load_state_dict(state,strict=True)
    counts={name:sum(p.numel() for p in model.parameters()) for name,model in [('attention',attention),('mlp',mlp)]}
    assert counts['attention']==counts['mlp']
    for name,model in [('attention',attention),('mlp',mlp)]:
        torch.save(model.state_dict(),BASE/f'initial-{name}.pt')
    plan=dict(status='prepared',minimum_anchor_epoch=12,target_anchor_step=7512,
        anchor_step=None,updates=2048,endpoint_step=None,width=480,heads=8,
        added_parameters=counts,common_initial_parameter_tensors=len(shared),
        fid_samples=50000,fid_seeds=list(SEEDS),world_size=8,local_batch=64,
        accumulation_steps=4,global_batch=2048,training_seed=261001,
        branches=list(BRANCHES),attention_sum_only_ablation_seeds=list(SEEDS),
        no_intermediate_training_FID=True,no_added_dropout=True,
        zero_initial_summary=True,no_automatic_promotion=True,
        production_resumes_from_sum_control=True,independent_training_replicates=False)
    write(OUT/'plan.json',plan)
    template=yaml.safe_load((PREVIOUS/'production-train.yaml').read_text())
    (BASE/'production-template.yaml').write_text(yaml.safe_dump(template,sort_keys=False))
    for branch in BRANCHES:
        config=yaml.safe_load(yaml.safe_dump(template))
        folder=OUT/branch/'train'
        config['options'].update(output=str(folder),checkpoint_dir=str(folder/'checkpoints'),
            resume_checkpoint=str(BASE/'anchor.pt'),fid_every=0,fid_early_epochs=0,
            sample_grid_every=626,sample_grid_on_start=False,upload_checkpoints=False,
            save_step_freq=512,save_ckpt_freq=100,max_optimizer_steps=2048,
            wandb_id=f'imagenet-rfid421-site-pool-{branch}-20261007',
            wandb_name=f'ImageNet rFID4.21 | site pooling {branch} | matched 2048 updates')
        text=yaml.safe_dump(config,sort_keys=False)
        (BASE/f'{branch}-train.yaml').write_text(text);(OUT/f'{branch}-train.yaml').write_text(text)
        (ROOT/f'configs/stage2/imagenet-rfid421-site-pooling-{branch}-trial-20261007.yaml').write_text(text)
        for seed in SEEDS:
            phases=[f'seed-{seed}']+([f'seed-{seed}-sum-only'] if branch=='attention' else [])
            for phase in phases:
                evaluation=yaml.safe_load(text);folder=OUT/branch/phase
                evaluation['options'].update(output=str(folder),checkpoint_dir=str(folder/'checkpoints'),
                    resume_checkpoint=str(OUT/branch/'train/checkpoints/last.pt'),
                    fid_only=True,fid_seed=seed,max_optimizer_steps=0,save_step_freq=0,
                    sample_grid_every=0,sample_grid_on_start=seed==SEEDS[0],
                    wandb_id=f'imagenet-rfid421-site-pool-{branch}-{phase}-20261007',
                    wandb_name=f'ImageNet rFID4.21 | site pooling {branch} | {phase}')
                text_eval=yaml.safe_dump(evaluation,sort_keys=False)
                (BASE/f'{branch}-{phase}.yaml').write_text(text_eval)
                (OUT/f'{branch}-{phase}.yaml').write_text(text_eval)
    files=['src/models/rqtransformer/transformers.py','src/training/rqtransformer.py',
           'src/models/site_pair_pooling.py','src/training/site_pair_pooling.py']
    hashes={name:hashlib.sha256((BASE/'source'/name).read_bytes()).hexdigest() for name in files}
    write(OUT/'source-manifest.json',dict(source_baseline=str(SOURCE),sha256=hashes,
        initial_states={m:hashlib.sha256((BASE/f'initial-{m}.pt').read_bytes()).hexdigest() for m in ('attention','mlp')}))
    write(OUT/'launch-status.json',dict(phase='prepared',time=time.time()))
    print(json.dumps(plan),flush=True)


def environment():
    return dict(os.environ,
        WANDB_API_KEY=Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip(),
        WANDB_DIR=str(BASE/'wandb'),WANDB_CACHE_DIR=str(BASE/'wandb-cache'),
        WANDB_DATA_DIR=str(BASE/'wandb-data'),WANDB_ARTIFACT_DIR=str(BASE/'wandb-artifacts'),
        TORCH_HOME=str(ASSETS/'torch-cache'),TORCHINDUCTOR_CACHE_DIR=str(ASSETS/'inductor-cache'),
        TORCHINDUCTOR_COMPILE_THREADS='4',MPLCONFIGDIR=str(BASE/'matplotlib'),
        PYTHONPATH=str(BASE/'source')+':'+str(BASE/'source/runtime'),
        CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',LASER_ACCUMULATION='4',
        LASER_SITE_POOLING_BASE=str(BASE),LASER_SITE_POOLING_OUTPUT=str(OUT),
        LASER_CHECKPOINT_STAGING_DIR=str(BASE/'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(CACHE),LASER_CHECKPOINT_IMMUTABLE_FILES='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',NCCL_NVLS_ENABLE='0')


def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False


def run(env,branch,phase,wb,*,production=False,resume_step=None):
    config=BASE/('production-train.yaml' if production else f'{branch}-{phase}.yaml')
    child_env=dict(env,LASER_BRANCH=branch,LASER_PHASE=phase)
    if resume_step is not None:child_env['LASER_RESUME_STEP']=str(resume_step)
    log_path=OUT/f'{branch}-{phase}.log'
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8',
        str(BASE/'imagenet_site_pooling_entry.py'),'--config',str(config)]
    with log_path.open('a') as log:
        process=subprocess.Popen(command,cwd=BASE/'source',env=child_env,stdout=log,stderr=subprocess.STDOUT)
        record(phase='running',branch=branch,operation=phase,training_pid=process.pid,
               production=production,log=str(log_path))
        if wb is not None:
            wb.summary.update({'trial/status':'production_resumed' if production else 'running',
                               'trial/branch':branch,'trial/operation':phase})
        if production:
            write(PRODUCTION/'launch-status.json',dict(phase='running',branch='baseline',
                supervisor_pid=os.getpid(),training_pid=process.pid,log=str(log_path),
                resume_step=resume_step,time=time.time()))
        result=process.wait()
    record(phase='completed' if result==0 else 'failed',branch=branch,operation=phase,returncode=result)
    if production:
        write(PRODUCTION/'launch-status.json',dict(phase='completed' if result==0 else 'failed',
            branch='baseline',returncode=result,supervisor_pid=os.getpid(),
            training_pid=process.pid,log=str(log_path),time=time.time()))
    if result:raise RuntimeError(f'{branch} {phase} exited {result}')


def ready_for_anchor(payload,log_tail):
    return (payload.get('epoch',0)>=12 and payload.get('global_step',0)>=7512
        and 'Epoch 12: FID=' in log_tail
        and all(Path(path).is_file() for name in ('best_fid','best_inception')
                for _,path in payload.get(name,[])))


def wait_and_pin(io,torch,wb):
    previous_source=None
    while True:
        status=json.loads((PRODUCTION/'launch-status.json').read_text())
        assert status['phase']=='running' and alive(status['training_pid']), 'Baseline is not running'
        source=(PRODUCTION/'train/checkpoints/last.pt').resolve()
        if source!=previous_source:
            local=io._checkpoint_upload_source(source)
            if local!=source:
                payload=torch.load(local,map_location='cpu',mmap=True,weights_only=False)
                with Path(status['log']).open('rb') as stream:
                    stream.seek(max(0,Path(status['log']).stat().st_size-131072))
                    tail=stream.read().decode(errors='replace')
                if ready_for_anchor(payload,tail):
                    assert len(payload['optimizer']['state'])==798
                    step=payload['global_step']
                    assert payload['scheduler']['last_epoch']==step
                    assert payload['scheduler']['T_max']==62600
                    assert {int(v['step']) for v in payload['optimizer']['state'].values()}=={step}
                    assert payload['checkpoint_world_size']==len(payload['rng_state_by_rank'])==8
                    assert not any(k.startswith(('site_pooling.','pair_memory_queries.')) for k in payload['state_dict'])
                    try:os.link(local,BASE/'anchor.pt')
                    except FileNotFoundError:continue
                    (OUT/'anchor.pt').symlink_to(source)
                    plan=json.loads((OUT/'plan.json').read_text())
                    plan.update(status='pinned',anchor_step=step,endpoint_step=step+plan['updates'],
                        anchor_epoch=payload['epoch'],anchor_batch_idx=payload.get('batch_idx',0))
                    write(OUT/'plan.json',plan)
                    wb.config.update(plan,allow_val_change=True)
                    write(OUT/'anchor-verification.json',dict(passed=True,global_step=step,
                        optimizer_states=798,scheduler_age=step,rng_ranks=8,
                        persistent_source=str(source),local_serialization=str(BASE/'anchor.pt'),
                        post_epoch12_FID_save_completed=True,bytes=local.stat().st_size))
                    for phase in ('train',*(f'seed-{s}' for s in SEEDS)):
                        folder=OUT/'sum'/phase/'checkpoints';folder.mkdir(parents=True,exist_ok=True)
                        for field in ('best_fid','best_inception'):
                            for _,path in payload.get(field,[]):
                                target=folder/Path(path).name
                                if not target.exists():target.symlink_to(Path(path).resolve())
                    return plan,status
                # Recheck the same payload while best snapshots finish copying.
                if payload.get('epoch',0)<12:previous_source=source
                wb.log({'baseline/checkpoint_step':payload['global_step'],
                        'baseline/checkpoint_epoch':payload['epoch']})
                del payload
        record(phase='waiting_for_epoch12',production_training_pid=status['training_pid'],
               target_anchor_step=7512,baseline_continues=True)
        time.sleep(15)


def stop_baseline(status):
    supervisor,runner=status['supervisor_pid'],status['training_pid']
    expected=str(PREVIOUS).encode()
    for pid in (supervisor,runner):
        assert expected in Path(f'/proc/{pid}/cmdline').read_bytes(), f'Unrecognized process {pid}'
    workers=[int(x) for x in subprocess.check_output(['pgrep','-P',str(runner)],text=True).split()]
    assert len(workers)==8
    for pid in workers:assert expected in Path(f'/proc/{pid}/cmdline').read_bytes()
    write(OUT/'handoff-started.json',dict(pids=[supervisor,runner,*workers],time=time.time()))
    for pid in (supervisor,runner):
        if alive(pid):os.kill(pid,signal.SIGTERM)
    deadline=time.monotonic()+30
    while any(alive(pid) for pid in (runner,*workers)) and time.monotonic()<deadline:time.sleep(1)
    for pid in (runner,*workers):
        if alive(pid):os.kill(pid,signal.SIGKILL)
    deadline=time.monotonic()+10
    while any(alive(pid) for pid in (runner,*workers)) and time.monotonic()<deadline:time.sleep(1)
    assert not any(alive(pid) for pid in (runner,*workers)), 'Old workers have not exited'
    write(OUT/'baseline-handoff.json',dict(stopped_pids=[supervisor,runner,*workers],
          anchor_step=json.loads((OUT/'plan.json').read_text())['anchor_step'],time=time.time()))
    write(PRODUCTION/'launch-status.json',dict(phase='site_pooling_trial',supervisor_pid=os.getpid(),
          trial_status=str(OUT/'launch-status.json'),baseline_resume_after_trial=True,time=time.time()))


def verify_endpoint(branch,plan,io,torch):
    checkpoint=OUT/branch/'train/checkpoints/last.pt'
    local=io._checkpoint_upload_source(checkpoint.resolve());assert local!=checkpoint.resolve()
    payload=torch.load(local,map_location='cpu',mmap=True,weights_only=False)
    assert payload['global_step']==payload['scheduler']['last_epoch']==plan['endpoint_step']
    assert payload['checkpoint_world_size']==len(payload['rng_state_by_rank'])==8
    states=payload['optimizer']['state']
    assert {int(states[i]['step']) for i in range(798)}=={plan['endpoint_step']}
    added={k:v for k,v in payload['state_dict'].items() if k.startswith('site_pooling.')}
    if branch=='sum':assert not added and len(states)==798
    else:
        assert len(states)==798+len(added)
        assert {int(states[i]['step']) for i in range(798,len(states))}=={plan['updates']}
        assert sum(v.numel() for v in added.values())==plan['added_parameters'][branch]
        assert all(torch.isfinite(v).all() for v in added.values())
        assert added['site_pooling.output.weight'].abs().sum()>0
    write(OUT/f'{branch}-endpoint-verification.json',dict(passed=True,global_step=payload['global_step'],
        old_adam_age=plan['endpoint_step'],new_adam_age=plan['updates'] if branch!='sum' else None,
        scheduler_age=payload['scheduler']['last_epoch'],rng_ranks=8,
        checkpoint=str(checkpoint.resolve()),local_serialization=str(local)))


def compare(plan,wb):
    metrics={};streams={};ablations={}
    for branch in BRANCHES:
        metrics[branch]={};streams[branch]={}
        for seed in SEEDS:
            phase=f'seed-{seed}'
            value=json.loads((OUT/branch/phase/f'evaluations/fid_step_{plan["endpoint_step"]:07d}.json').read_text())
            assert value['global_step']==plan['endpoint_step'] and value['num_generated_samples']==50000
            assert value['fid_seed']==seed and value['metric_backend']=='original-rqvae'
            ranks=[json.loads((OUT/'verification'/branch/phase/f'fid-sampling-rank{r}.json').read_text()) for r in range(8)]
            assert all(x['completed'] and x['training_rng_restored'] for x in ranks)
            metrics[branch][str(seed)]=value
            streams[branch][seed]=[x['cuda_rng_sha256'] for x in ranks]
    for seed in SEEDS:
        assert streams['sum'][seed]==streams['mlp'][seed]==streams['attention'][seed]
        phase=f'seed-{seed}-sum-only'
        value=json.loads((OUT/'attention'/phase/f'evaluations/fid_step_{plan["endpoint_step"]:07d}.json').read_text())
        assert value['global_step']==plan['endpoint_step'] and value['num_generated_samples']==50000
        assert value['fid_seed']==seed and value['metric_backend']=='original-rqvae'
        ranks=[json.loads((OUT/'verification/attention'/phase/f'fid-sampling-rank{r}.json').read_text()) for r in range(8)]
        assert all(x['completed'] and x['training_rng_restored'] for x in ranks)
        assert [x['cuda_rng_sha256'] for x in ranks]==streams['attention'][seed]
        ablations[str(seed)]=value
    report=dict(anchor_step=plan['anchor_step'],endpoint_step=plan['endpoint_step'],
        updates=plan['updates'],fid_samples=50000,metrics=metrics,attention_sum_only=ablations,
        attention_sum_only_minus_full={str(s):ablations[str(s)]['fid']-metrics['attention'][str(s)]['fid'] for s in SEEDS},
        matched_sampling_rng_verified=True,no_automatic_promotion=True,independent_training_replicates=False,
        limits=['Two sampling seeds evaluate one training run per architecture.',
                'The sum-only intervention is an inference ablation, not a retrained control.'])
    write(OUT/'comparison.json',report)
    for branch in BRANCHES:
        for seed in SEEDS:wb.summary[f'results/{branch}/fid_seed_{seed}']=metrics[branch][str(seed)]['fid']
    for seed in SEEDS:wb.summary[f'results/attention_sum_only/fid_seed_{seed}']=ablations[str(seed)]['fid']
    wb.save(str(OUT/'comparison.json'),base_path=str(OUT),policy='now')
    return report


def evaluated_plain_payload(payload,evaluation,ready_source):
    """Only evaluation metadata changes; all native training objects survive."""
    step=payload['global_step']
    assert evaluation['global_step']==step and evaluation['fid_seed']==SEEDS[0]
    names={
        'fid':str(PRODUCTION/f'train/checkpoints/best_fid_{evaluation["fid"]:.4f}_step_{step:07d}.pt'),
        'inception_score':str(PRODUCTION/f'train/checkpoints/best_is_{evaluation["inception_score"]:.4f}_step_{step:07d}.pt')}
    best_fid=sorted([*payload.get('best_fid',[]),(evaluation['fid'],names['fid'])],key=lambda x:x[0])[:1]
    best_is=sorted([*payload.get('best_inception',[]),(evaluation['inception_score'],names['inception_score'])],
                   key=lambda x:x[0],reverse=True)[:1]
    updated=dict(payload,fid=evaluation['fid'],inception_score=evaluation['inception_score'],
        inception_score_std=evaluation['inception_score_std'],best_fid=best_fid,best_inception=best_is)
    for key in ('state_dict','optimizer','scheduler','rng_state_by_rank','config'):
        assert updated[key] is payload[key]
    links={Path(path):ready_source for field in ('best_fid','best_inception')
           for _,path in updated[field] if path in names.values()}
    return updated,links


def replace_link(target,source):
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_name(target.name+'.site-pool.tmp')
    temporary.unlink(missing_ok=True);temporary.symlink_to(source);temporary.replace(target)


def prepare_production(io,torch,report):
    import yaml
    candidate=OUT/'sum/train/checkpoints/last.pt'
    persistent=candidate.resolve() if candidate.is_file() else (OUT/'anchor.pt').resolve()
    local=io._checkpoint_upload_source(persistent)
    if local==persistent and not candidate.is_file():local=BASE/'anchor.pt'
    assert str(local).startswith('/tmp/'), 'Production must load a retained local checkpoint'
    payload=torch.load(local,map_location='cpu',mmap=True,weights_only=False)
    assert len(payload['optimizer']['state'])==798
    assert {int(v['step']) for v in payload['optimizer']['state'].values()}=={payload['global_step']}
    assert payload['scheduler']['last_epoch']==payload['global_step']
    assert payload['checkpoint_world_size']==len(payload['rng_state_by_rank'])==8
    assert not any(k.startswith('site_pooling.') for k in payload['state_dict'])
    if report is not None:
        evaluation=report['metrics']['sum'][str(SEEDS[0])]
        ready=OUT/'sum/continuation-ready.pt'
        updated,links=evaluated_plain_payload(payload,evaluation,ready)
        io.atomic_torch_save(updated,ready)
        persistent=ready.resolve();local=io._checkpoint_upload_source(persistent)
        assert local!=persistent
        for target in links:replace_link(target,persistent)
        write(OUT/'production-best-verification.json',dict(passed=True,primary_fid_seed=SEEDS[0],
            fid=evaluation['fid'],training_objects_preserved=True,checkpoint=str(persistent),
            local_serialization=str(local),best_fid=updated['best_fid'],best_inception=updated['best_inception']))
    replace_link(PRODUCTION/'train/checkpoints/last.pt',persistent)
    config=yaml.safe_load((BASE/'production-template.yaml').read_text())
    config['options'].update(resume_checkpoint=str(local),max_optimizer_steps=0)
    text=yaml.safe_dump(config,sort_keys=False)
    (BASE/'production-train.yaml').write_text(text);(OUT/'production-train.yaml').write_text(text)
    return payload['global_step']


def main():
    import torch
    sys.path[:0]=[str(BASE/'source'),str(BASE/'source/runtime')]
    from src.training import k4_checkpoint_io as io
    assert json.loads((OUT/'setup-verification.json').read_text())['passed']
    lock=(BASE/'launch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=environment()
    os.environ.update({k:v for k,v in env.items() if k.startswith('LASER_CHECKPOINT_') or k=='WANDB_API_KEY'})
    import wandb
    wb=wandb.init(entity='helloimlixin-rutgers',project='laser',
        id='imagenet-rfid421-site-pooling-controller-20261007',
        name='ImageNet rFID4.21 | site pooling trial | queue after epoch12',
        config=json.loads((OUT/'plan.json').read_text()),resume='allow',mode='online',
        dir=str(BASE/'wandb'),job_type='experiment-supervisor')
    write(OUT/'wandb.json',dict(id=wb.id,url=wb.url,online=wb.settings.mode=='online'))
    wb.summary.update({'trial/status':'waiting_for_epoch12','trial/no_automatic_promotion':True})
    plan,status=wait_and_pin(io,torch,wb)
    report=None;error=None
    try:
        stop_baseline(status)
        with (BASE/'anchor.pt').open('rb') as stream:
            anchor_hash=hashlib.file_digest(stream,'sha256').hexdigest()
        write(OUT/'anchor-sha256.json',dict(sha256=anchor_hash,global_step=plan['anchor_step']))
        for branch in BRANCHES:
            run(env,branch,'train',wb)
            verify_endpoint(branch,plan,io,torch)
        for branch in BRANCHES:
            for seed in SEEDS:run(env,branch,f'seed-{seed}',wb)
        for seed in SEEDS:run(env,'attention',f'seed-{seed}-sum-only',wb)
        report=compare(plan,wb)
        wb.summary['trial/status']='complete'
    except Exception as exception:
        error=repr(exception)
        record(phase='trial_failed',error=error,baseline_recovery_pending=True)
        wb.summary.update({'trial/status':'failed','trial/error':error})
    finally:
        if (OUT/'handoff-started.json').exists():
            pids=json.loads((OUT/'handoff-started.json').read_text())['pids']
            assert not any(alive(pid) for pid in pids), 'Refuse to overlap production with stopped workers'
            try:step=prepare_production(io,torch,report)
            except Exception as exception:
                record(phase='evaluated_resume_metadata_failed',error=repr(exception),raw_plain_recovery=True)
                step=prepare_production(io,torch,None)
            wb.summary['trial/baseline_resume_step']=step
            wb.finish()
            record(phase='resuming_production',resume_step=step,trial_error=error)
            run(env,'sum','production',None,production=True,resume_step=step)
        else:wb.finish()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if args.prepare_only:prepare()
    else:main()
