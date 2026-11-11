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
        target_epoch=PLAN['target_epoch'],fid_target=11.56,progress=read(progress) if progress.exists() else None,
        time=time.time(),**values))


def stop(sig,_):
    if CHILD is not None and CHILD.poll() is None:
        try:os.killpg(CHILD.pid,signal.SIGTERM)
        except ProcessLookupError:pass
    status('stopped_by_signal');raise SystemExit(128+sig)


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
        while CHILD.poll() is None:
            status('training')
            try:CHILD.wait(timeout=10)
            except subprocess.TimeoutExpired:pass
        if CHILD.returncode==0:
            assert read(OUT/'completed.json')['epochs']==PLAN['target_epoch']
            status('completed');return
        status('recovering',exit_code=CHILD.returncode,resume_step=step)


if __name__=='__main__':
    try:main()
    except Exception as error:
        status('failed',error_type=type(error).__name__,error=str(error));raise
