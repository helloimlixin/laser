"""Checkpointed controller upgrade, then keep the continuous trainer supervised."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

HERE=Path(__file__).resolve().parent
BASE=Path('/tmp/laser-s1-continuous-b200')
LOCAL=BASE/'run-output'
OUT=HERE.parents[1]/'outputs/imnet-s1-laser-best2465-continuous-1b200-20261104'


def write(path,value):
    temp=path.with_suffix(path.suffix+'.pending')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    import torch
    torch.set_num_threads(4)
    lock=(OUT/'continuous-runner.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=json.loads((OUT/'supervisor-status.json').read_text())
    old_pid=old.get('child_pid')
    # The upgrade waits for a full state checkpoint; there is no forced kill.
    (LOCAL/'STOP').touch()
    deadline=time.monotonic()+180
    while old_pid and Path(f'/proc/{old_pid}').exists():
        if time.monotonic()>deadline:raise RuntimeError('Checkpointed upgrade timed out')
        time.sleep(1)
    runtime=BASE/'runtime-live'
    shutil.copytree(BASE/'runtime',runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copyfile(HERE/'continuous.py',runtime/'continuous.py')
    cfg=json.loads((LOCAL/'config.json').read_text())
    previous_contract=torch.load(LOCAL/'tokenizer-last.pt',map_location='cpu',weights_only=False,mmap=True)
    cfg['runtime_sha256']={name:sha(runtime/name) for name in cfg['runtime_sha256']}
    contract={k:v for k,v in cfg.items() if k not in {'output','wandb','workers','checkpoint_every','preview_every','source_bundle'}}
    contract_sha=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    migrated=dict(previous_contract,contract_sha256=contract_sha)
    temporary=LOCAL/'upgrade-checkpoint.pt'
    torch.save(migrated,temporary)
    check=torch.load(temporary,map_location='cpu',weights_only=False,mmap=True)
    sys.path.insert(0,str(runtime))
    from migrate_stage1_world_size import assert_equal
    for field in ['model','discriminator','optimizer','discriminator_optimizer','rng','progress','learning_dynamics']:
        assert_equal(previous_contract[field],check[field])
    del check,previous_contract,migrated
    temporary.replace(LOCAL/'tokenizer-last.pt')
    write(LOCAL/'config.json',cfg);write(OUT/'config.json',cfg)
    write(LOCAL/'contract.json',contract)
    with tarfile.open(BASE/'source.tar.gz','w:gz') as archive:
        for path in runtime.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                archive.add(path,arcname='runtime/'+str(path.relative_to(runtime)))
    write(OUT/'controller-upgrade.json',dict(passed=True,full_state_bit_exact=True,
        runtime=str(runtime),time=time.time(),blocking_initial_republication_removed=True))
    (LOCAL/'STOP').unlink()
    command=[sys.executable,'-u','-m','torch.distributed.run','--standalone',
        '--nproc-per-node=1',str(runtime/'continuous.py'),'--config',str(LOCAL/'config.json')]
    env=dict(os.environ,LASER_VGG16_WEIGHTS='/tmp/laser-s1-best-b200/weights/vgg16-397923af.pth',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        WANDB_DISABLE_GIT='true',WANDB_MODE='online',PYTHONUNBUFFERED='1',
        PYTORCH_ALLOC_CONF='expandable_segments:True')
    failures=0
    while True:
        with (BASE/'training.log').open('ab',buffering=0) as stream:
            child=subprocess.Popen(command,cwd=runtime,env=env,stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        while child.poll() is None:
            write(OUT/'supervisor-status.json',dict(phase='training',continuous=True,
                supervisor_pid=os.getpid(),child_pid=child.pid,failures=failures,time=time.time()))
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:pass
        if child.returncode==0 or (LOCAL/'STOP').exists():
            write(OUT/'supervisor-status.json',dict(phase='user_stop_checkpointed',time=time.time()))
            return
        failures+=1
        write(OUT/'supervisor-status.json',dict(phase='restarting_after_error',failures=failures,time=time.time()))
        if failures>=3:raise RuntimeError('Repeated runtime failure; inspect training.log')
        time.sleep(2)


if __name__=='__main__':main()
