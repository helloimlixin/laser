"""Launch and supervise the continuous B200 job from the current best."""
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
SOURCE=Path('/tmp/laser-s1-best2486-lr1e6-b200')
BASE=Path('/tmp/laser-s1-continuous-b200')
RUNTIME=BASE/'runtime'
RUN='imnet-s1-laser-best2465-continuous-1b200-20261104'
OUT=HERE.parents[1]/'outputs'/RUN


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.pending')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    import torch
    torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True);BASE.mkdir(exist_ok=True)
    lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'launch.json').exists(),'Existing continuous launch; inspect before restarting'
    shutil.copytree(SOURCE/'runtime',RUNTIME,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    text=(RUNTIME/'train_matched_stage1.py').read_text()
    assert text.count("for epoch in range(self.progress['epoch'],cfg['epochs']):")==1
    text=text.replace('import argparse,contextlib,','import itertools\nimport argparse,contextlib,')
    text=text.replace("for epoch in range(self.progress['epoch'],cfg['epochs']):",
                      "for epoch in itertools.count(self.progress['epoch']):")
    (RUNTIME/'train_matched_stage1.py').write_text(text)
    shutil.copyfile(HERE/'continuous.py',RUNTIME/'continuous.py')
    cfg=json.loads((SOURCE/'run-output/config.json').read_text())
    best=json.loads((SOURCE/'run-output/best-reconstruction.json').read_text())
    source=SOURCE/'run-output/tokenizer-best-rfid.pt'
    assert sha(source)==best['checkpoint']['checkpoint_sha256']
    assert best['step']==42636 and abs(best['rfid']-2.4651174750250675)<1e-12
    state=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
    local=BASE/'run-output';local.mkdir(exist_ok=True)
    cfg.update(output=str(local),source_bundle=str(BASE/'source.tar.gz'),lr=3e-7,
        wandb=dict(id=RUN,name='ImageNet stage1 | best2.465 | continuous best-rewind | B200'))
    policy=cfg['continuation']
    policy.update(source_run='helloimlixin-rutgers/laser/'+json.loads((SOURCE/'run-output/config.json').read_text())['wandb']['id'],
        source_artifact=best['checkpoint']['artifact'],source_step=42636,
        source_checkpoint_sha256=best['checkpoint']['checkpoint_sha256'],source_rfid=best['rfid'],
        durable=str(OUT),continuous=True,automatic_recovery_instead_of_stop=True,
        recovery_preserves_stream_cursor_and_restores_adam_age=True)
    policy.pop('independent_full_baseline_control',None)
    policy['learning_rate'].update(maximum=3e-7,minimum=1e-7,absolute_floor=1e-8,
        start_step=42636,decay_updates=2000)
    cfg['runtime_sha256']={name:sha(RUNTIME/name) for name in cfg['runtime_sha256']}
    cfg['runtime_sha256']['continuous.py']=sha(RUNTIME/'continuous.py')
    contract={k:v for k,v in cfg.items() if k not in {'output','wandb','workers','checkpoint_every','preview_every','source_bundle'}}
    contract_sha=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    payload=dict(state,contract_sha256=contract_sha,
        learning_dynamics=dict(lr_factor=1.,nonimproving_checks=0,evaluations=0,reductions=0,recoveries=0))
    torch.save(payload,local/'tokenizer-last.pt')
    recovered=torch.load(local/'tokenizer-last.pt',map_location='cpu',weights_only=False,mmap=True)
    sys.path.insert(0,str(RUNTIME))
    from migrate_stage1_world_size import assert_equal
    for field in ['model','discriminator','optimizer','discriminator_optimizer','progress','rng']:
        assert_equal(state[field],recovered[field])
    proof=dict(passed=True,source_sha256=best['checkpoint']['checkpoint_sha256'],
        source_artifact=best['checkpoint']['artifact'],source_rfid=best['rfid'],
        model_both_optimizers_rng_and_cursor_identical=True,progress=state['progress'])
    for name,value in [('config.json',cfg),('initialization.json',proof),
                       ('resume-verification.json',proof),('best-reconstruction.json',best)]:
        write(local/name,value);write(OUT/name,value)
    write(local/'checkpoint.json',dict(progress=state['progress'],contract_sha256=contract_sha,
        checkpoint_sha256=sha(local/'tokenizer-last.pt'),full_state=True))
    with tarfile.open(BASE/'source.tar.gz','w:gz') as archive:
        for path in RUNTIME.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                archive.add(path,arcname='runtime/'+str(path.relative_to(RUNTIME)))
    command=[sys.executable,'-u','-m','torch.distributed.run','--standalone',
        '--nproc-per-node=1',str(RUNTIME/'continuous.py'),'--config',str(local/'config.json')]
    env=dict(os.environ,LASER_VGG16_WEIGHTS='/tmp/laser-s1-best-b200/weights/vgg16-397923af.pth',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        WANDB_DISABLE_GIT='true',WANDB_MODE='online',PYTHONUNBUFFERED='1',
        PYTORCH_ALLOC_CONF='expandable_segments:True')
    launch=dict(run_id=RUN,source=proof,continuous=True,quality_recovery='rewind_best_halve_lr_keep_data_cursor',
        generator_lr=3e-7,discriminator_lr=3e-7,gpu='NVIDIA B200',global_batch=128,
        microbatch=64,accumulation=2,command=command,time=time.time())
    write(OUT/'launch.json',launch)
    failures=0
    while True:
        with (BASE/'training.log').open('ab',buffering=0) as log:
            child=subprocess.Popen(command,cwd=RUNTIME,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while child.poll() is None:
            write(OUT/'supervisor-status.json',dict(phase='training',time=time.time(),
                supervisor_pid=os.getpid(),child_pid=child.pid,continuous=True,failures=failures))
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:pass
        if child.returncode==0 or (local/'STOP').exists():
            write(OUT/'supervisor-status.json',dict(phase='user_stop_checkpointed',time=time.time()))
            return
        failures+=1
        write(OUT/'supervisor-status.json',dict(phase='restarting_after_error',
            failures=failures,returncode=child.returncode,time=time.time()))
        if failures>=3:raise RuntimeError('Three runtime failures: inspect training.log')
        # Reuse the last full checkpoint; the trainer's quality recovery stays
        # inside the process and does not consume these infrastructure retries.
        time.sleep(2)


if __name__=='__main__':main()
