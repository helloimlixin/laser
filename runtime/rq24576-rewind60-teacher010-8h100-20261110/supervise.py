"""Supervise the verified single eight-H100 continuation and durable checkpoints."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
PLAN=json.loads((HERE/'plan.json').read_text());BASE=Path(PLAN['base']);OUT=Path(PLAN['output'])
CHILD=None


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);pending=path.with_suffix('.pending')
    pending.write_text(json.dumps(value,indent=2)+'\n');pending.replace(path)


def environment():
    spec=importlib.util.spec_from_file_location('parent_env',HERE.parent/'rq24576-native-8h100-20261108/environment.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    env=module.environment()
    env.update(LASER_STAGE2_OUTPUT=str(OUT),LASER_BEAM_BASE=str(BASE),WANDB_CACHE_DIR=str(BASE/'wandb-cache'))
    env.pop('LASER_TEST_FINAL_TEACHER',None);env.pop('LASER_PRIVATE_SMOKE_DIR',None)
    return env


def status(phase,**values):
    progress=BASE/'live-training/train/progress-rank0.json'
    write(OUT/'status.json',dict(phase=phase,run_id=PLAN['run_id'],supervisor_pid=os.getpid(),
        child_pid=CHILD.pid if CHILD else None,world_size=8,source_epoch=PLAN['source_epoch'],
        target_epoch=PLAN['target_epoch'],teacher_temperature_after_rewind=PLAN['teacher']['final'],fid_target=11.56,progress=read(progress) if progress.exists() else None,
        time=time.time(),**values))


def stop(sig,_):
    if CHILD is not None and CHILD.poll() is None:
        try:os.killpg(CHILD.pid,signal.SIGTERM)
        except ProcessLookupError:pass
    resume_samplers('training_stopped');status('stopped_by_signal');raise SystemExit(128+sig)


def resume_samplers(reason):
    path=OUT/'sampler-pause.json'
    if not path.exists():return
    state=read(path)
    if state['resumed']:return
    resumed=[]
    for pid in state['paused_workers']:
        cmd=Path(f'/proc/{pid}/cmdline')
        if not cmd.exists():continue
        text=cmd.read_bytes().replace(b'\0',b' ').decode()
        if 'rq24576-ema-sampler-search-20261110/worker.py' not in text:continue
        os.kill(pid,signal.SIGCONT);resumed.append(pid)
    state.update(resumed=True,resumed_workers=resumed,resumed_at=time.time(),resume_reason=reason)
    write(path,state)
    write(OUT.parent/'imagenet-rq24576-ema-sampler-search-20261110/paused-for-rewind60.json',state)


def audit_replay():
    old=OUT.parent/PLAN['parent_run_id']/'train/metrics.jsonl';new=OUT/'train/metrics.jsonl'
    if not old.exists() or not new.exists():return
    source={}
    for line in old.read_text().splitlines():
        row=json.loads(line);step=row.get('train/global_step',-1)
        if PLAN['source_step']<step<=PLAN['requested_rewind_step']:source[step]=row
    rows=[]
    for line in new.read_text().splitlines():
        row=json.loads(line);step=row.get('train/global_step',-1)
        if step in source:rows.append(dict(step=step,loss_difference=abs(row['train/loss']-source[step]['train/loss']),
            lr_difference=abs(row['train/lr']-source[step]['train/lr'])))
    if rows:write(OUT/'replay-audit.json',dict(compared_updates=len(rows),bitwise_reproduction_claimed=False,
        max_loss_difference=max(r['loss_difference'] for r in rows),max_lr_difference=max(r['lr_difference'] for r in rows),rows=rows))


def main():
    global CHILD
    lock=(BASE/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,stop)
    for name in ['resume-preflight-verification.json','policy-verification.json','source-checkpoint-verification.json']:
        assert read(OUT/name)['passed']
    assert not PLAN.get('source_selection_pending',True)
    archive=OUT/'reproducibility';archive.mkdir(exist_ok=True)
    for path in HERE.glob('*.py'):shutil.copyfile(path,archive/path.name)
    for name in ['recipe.yaml','parent-recipe.yaml','late-refinement.json','teacher-calibration.json','execution.json','stage2-training-config.yaml']:
        shutil.copyfile(BASE/'inputs'/name,archive/name)
    for name in ['teacher010-matched-10k-comparison.json','teacher010-matched-50k-validation.json']:
        shutil.copyfile(BASE/'inputs'/name,archive/name)
    for name in ['resume-preflight-verification.json','policy-verification.json','source-checkpoint-verification.json']:
        shutil.copyfile(OUT/name,archive/name)
    attempts={}
    while True:
        source=Path(PLAN['source_checkpoint']);step=PLAN['source_step'];new_run=True
        if (OUT/'checkpoint.json').exists():
            receipt=read(OUT/'checkpoint.json');source=Path(receipt['path'])
            if not source.exists():source=Path(receipt['persistent_path'])
            assert source.stat().st_size==receipt['bytes'];step=receipt['global_step'];new_run=False
        attempts[step]=attempts.get(step,0)+1
        assert attempts[step]<=3,f'Repeated training failure at {step}'
        command=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc-per-node=8',str(BASE/'train.py'),
            '--epochs',str(PLAN['target_epoch']),'--microbatch','64','--resume',str(source)]
        if new_run:command.append('--new-wandb-run')
        with (OUT/'training.log').open('ab',buffering=0) as log:
            CHILD=subprocess.Popen(command,cwd=BASE,env=environment(),stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(OUT/'launch.json',dict(command=command,pid=CHILD.pid,supervisor_pid=os.getpid(),
            world_size=8,resume_step=step,source_epoch=PLAN['source_epoch'],time=time.time()))
        last_audit=0
        while CHILD.poll() is None:
            progress=BASE/'live-training/train/progress-rank0.json'
            step=read(progress)['train/global_step'] if progress.exists() else PLAN['source_step']
            if step>=PLAN['requested_rewind_step']:resume_samplers('replay_reached_epoch60')
            status('replaying_epoch58_to60' if step<PLAN['requested_rewind_step'] else 'training_calibrated_teacher')
            if time.monotonic()-last_audit>60:audit_replay();last_audit=time.monotonic()
            try:CHILD.wait(timeout=10)
            except subprocess.TimeoutExpired:pass
        if CHILD.returncode==0:
            assert read(OUT/'completed.json')['epochs']==PLAN['target_epoch']
            status('completed');return
        status('recovering',exit_code=CHILD.returncode,resume_step=step)


if __name__=='__main__':
    try:main()
    except Exception as error:
        resume_samplers('training_failed');status('failed',error_type=type(error).__name__,error=str(error));raise
