"""Screen beam-prior settings, confirm at FID50k, and always resume training."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path('/workspace/Projects/laser')
BASE=Path('/tmp/laser-beam451-stage2-4h100-20261107')
OUT=ROOT/'outputs/imagenet-beam451-aligned-joint16k-scratch100-4h100-20261107'
EXPERIMENT=OUT/'sampler-calibration-20261108'
LOCAL=Path('/tmp/laser-beam451-sampler-20261108')
HERE=Path(__file__).resolve().parent
SUPERVISOR=ROOT/'runtime/launch-imagenet-beam451-4h100-20261107/supervise.py'


def read(path):return json.loads(path.read_text())


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.pending')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def policy(name,t=1.,k=1024,p=.95):return dict(name=name,sampling=dict(temperature=t,top_k=k,top_p=p))


def run_evaluation(stage,policies,samples,checkpoint,sha,seed=None):
    destination=LOCAL/stage
    plan=dict(output=str(destination),checkpoint=str(checkpoint),checkpoint_sha256=sha,
        policies=policies,samples=samples,seed=seed if seed is not None else (26110871 if samples<50000 else 26110872))
    plan_path=LOCAL/(stage+'-plan.json');write(plan_path,plan)
    spec=importlib.util.spec_from_file_location('supervisor',SUPERVISOR);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    env=module.environment()
    command=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc-per-node=4',str(HERE/'evaluate.py'),'--plan',str(plan_path)]
    with (EXPERIMENT/(stage+'.log')).open('ab',buffering=0) as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while process.poll() is None:
            progress=read(destination/'progress.json') if (destination/'progress.json').exists() else None
            write(EXPERIMENT/'status.json',dict(phase=stage,pid=process.pid,progress=progress,time=time.time()))
            # Preserve every finished policy promptly.
            if destination.exists():
                for p in destination.glob('*.json'):
                    if p.name!='progress.json':write(EXPERIMENT/stage/p.name,read(p))
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:pass
    assert process.returncode==0,f'{stage} evaluator failed; training will resume with unchanged sampler'
    result=read(destination/'results.json');assert result['passed'] and result['weights_unmodified']
    write(EXPERIMENT/stage/'results.json',result)
    return result['results']


def resume_training():
    with (OUT/'supervisor.log').open('ab',buffering=0) as log:
        process=subprocess.Popen([sys.executable,'-u',str(SUPERVISOR)],cwd=ROOT,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(EXPERIMENT/'training-resumed.json',dict(supervisor_pid=process.pid,time=time.time()))
    return process.pid


def main():
    EXPERIMENT.mkdir(exist_ok=True);LOCAL.mkdir(exist_ok=True)
    lock=(LOCAL/'sweep.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while not (EXPERIMENT/'pause.json').exists() or read(EXPERIMENT/'pause.json')['phase']!='paused_for_fixed_checkpoint_calibration':
        write(EXPERIMENT/'status.json',dict(phase='waiting_for_persisted_training_boundary',time=time.time()));time.sleep(15)
    paused=read(EXPERIMENT/'pause.json');checkpoint=Path(paused['checkpoint']['path'])
    policies=[policy('baseline'),policy('t095',.95),policy('t090',.9),policy('t085',.85),
        policy('k512',k=512),policy('k2048',k=2048),policy('p100',p=1.),
        policy('depth_tight_t095',.95,[1024,1024,512,256],[.95,.95,.9,.85]),
        policy('depth_wide_t095',.95,[1024,1024,2048,4096],[.95,.95,.98,.99])]
    report=None
    try:
        with checkpoint.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
        write(EXPERIMENT/'plan.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=sha,checkpoint_step=paused['target_step'],
            policies=policies,screen_samples=2000,confirmation_samples=50000,confirmation_challengers=2,
            fixed_weights=True,training_updates_added=0,separate_screen_and_confirmation_seeds=True,
            selection='Full FID50k confirmation of two best non-baseline screen candidates; report IS alongside FID'))
        screen=run_evaluation('screen',policies,2000,checkpoint,sha)
        finalists=sorted([r for r in screen if r['policy']!='baseline'],key=lambda r:r['fid'])[:2]
        selected_names={'baseline',*[r['policy'] for r in finalists]}
        confirmation=run_evaluation('confirm',[p for p in policies if p['name'] in selected_names],50000,checkpoint,sha)
        baseline=next(r for r in confirmation if r['policy']=='baseline')
        eligible=[r for r in confirmation if r['inception_score']>=.95*baseline['inception_score']]
        winner=min(eligible,key=lambda r:r['fid'])
        minimum_improvement=max(.5,.01*baseline['fid'])
        improved=winner['fid']<=baseline['fid']-minimum_improvement
        if not improved:winner=baseline
        report=dict(passed=True,source_run=OUT.name,source_step=paused['target_step'],checkpoint_sha256=sha,
            weights_unmodified=True,training_updates_added=0,generated_images=50000,reference_images=1281167,reference_split='train',
            baseline_fid=baseline['fid'],baseline_inception_score=baseline['inception_score'],fid=winner['fid'],
            inception_score=winner['inception_score'],source_sampling=baseline['sampling'],sampling=winner['sampling'],
            selected_policy=winner['policy'],improved=improved,minimum_fid_improvement=minimum_improvement,
            training_recipe_sha256=hashlib.sha256((BASE/'inputs/recipe.yaml').read_bytes()).hexdigest(),
            codebook_sha256='a261b0aeb7470b0bf005164ee047df77b56bb78eed40eec7376d7eddbc3f8eba',
            screen=screen,confirmation=confirmation,tokenizer_rfid_unchanged=4.514300868542591,
            limit='A fixed checkpoint and one held-out confirmation seed; optimum can change as the prior trains.')
        write(EXPERIMENT/'report.json',report)
        if report['improved']:
            from install_override import install
            install(report)
    except BaseException as error:
        write(EXPERIMENT/'failure.json',dict(error=str(error),time=time.time()));raise
    finally:
        pid=resume_training()
        write(EXPERIMENT/'status.json',dict(phase='resumed_training',supervisor_pid=pid,
            calibration_passed=report is not None,selected_policy=report['selected_policy'] if report else 'baseline',time=time.time()))


if __name__=='__main__':main()
