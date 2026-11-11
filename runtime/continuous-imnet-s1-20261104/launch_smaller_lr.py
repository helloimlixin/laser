"""Stage a tenfold-smaller-LR restart before switching the active GPU job."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time

HERE=Path(__file__).resolve().parent
OLD=Path('/tmp/laser-s1-continuous-b200')
OLD_OUT=HERE.parents[1]/'outputs/imnet-s1-laser-best2465-continuous-1b200-20261104'
BASE=Path('/tmp/laser-s1-lr1e9-b200-20261105')
RUN='imnet-s1-laser-best2465-lr1e9-continuous-1b200-20261105'
OUT=HERE.parents[1]/'outputs'/RUN


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.pending')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    import torch
    torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True);BASE.mkdir(exist_ok=True)
    lock=(OUT/'supervisor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'launch.json').exists(),'Refusing to duplicate a running restart'
    best=json.loads((OLD/'run-output/best-reconstruction.json').read_text())
    source=OLD/'run-output/tokenizer-best-rfid.pt'
    assert sha(source)==best['checkpoint']['checkpoint_sha256']
    payload=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
    runtime=BASE/'runtime'
    shutil.copytree(OLD/'runtime-live',runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    local=BASE/'run-output';local.mkdir(exist_ok=True)
    cfg=json.loads((OLD/'run-output/config.json').read_text())
    previous_lr=1e-8
    cfg.update(output=str(local),source_bundle=str(BASE/'source.tar.gz'),lr=1e-9,
        wandb=dict(id=RUN,name='ImageNet stage1 | best2.465 | constant LR1e-9 | continuous B200'))
    policy=cfg['continuation']
    policy.update(source_run='helloimlixin-rutgers/laser/'+json.loads((OLD/'run-output/config.json').read_text())['wandb']['id'],
        source_artifact=best['checkpoint']['artifact'],source_step=payload['progress']['step'],
        source_checkpoint_sha256=best['checkpoint']['checkpoint_sha256'],source_rfid=best['rfid'],
        durable=str(OUT),minimum_improvement=1e-4,initial_lr_compared_with_previous_live_lr=.1,
        source_baseline_inherited=True)
    policy['learning_rate'].update(maximum=1e-9,minimum=1e-9,absolute_floor=1e-11,
        start_step=payload['progress']['step'],decay_updates=2000,
        kind='constant_with_plateau_and_recovery_reductions')
    cfg['runtime_sha256']={name:sha(runtime/name) for name in cfg['runtime_sha256']}
    contract={k:v for k,v in cfg.items() if k not in {'output','wandb','workers','checkpoint_every','preview_every','source_bundle'}}
    contract_sha=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    state=dict(payload,contract_sha256=contract_sha,
        learning_dynamics=dict(lr_factor=1.,nonimproving_checks=0,evaluations=0,reductions=0,recoveries=0))
    torch.save(state,local/'tokenizer-last.pt')
    restored=torch.load(local/'tokenizer-last.pt',map_location='cpu',weights_only=False,mmap=True)
    sys.path.insert(0,str(runtime))
    from migrate_stage1_world_size import assert_equal
    from dynamics import learning_rate,observe
    for key in ['model','discriminator','optimizer','discriminator_optimizer','rng','progress']:
        assert_equal(payload[key],restored[key])
    for step in [payload['progress']['step'],payload['progress']['step']+100,1000000]:
        assert abs(learning_rate(policy['learning_rate'],state['learning_dynamics'],step)-1e-9)<1e-20
    _,decision=observe(policy,state['learning_dynamics'],best['rfid'],best['rfid']-.0002)
    assert decision['significant_improvement']
    ages={name:sorted({int(v['step']) for v in payload[name]['state'].values() if 'step' in v})
          for name in ['optimizer','discriminator_optimizer']}
    assert ages['optimizer']==[42636] and ages['discriminator_optimizer']==[32636]
    # Preserve the original immutable best byte-for-byte, including its remote
    # artifact identity; only the new resumable checkpoint has a new contract.
    os.link(source,local/'tokenizer-best-rfid.pt')
    proof=dict(passed=True,source_rfid=best['rfid'],source_step=payload['progress']['step'],
        source_artifact=best['checkpoint']['artifact'],source_sha256=best['checkpoint']['checkpoint_sha256'],
        model_both_optimizers_rng_and_cursor_bit_exact=True,optimizer_ages=ages,
        previous_live_lr=previous_lr,new_generator_lr=1e-9,new_discriminator_lr=1e-9,
        constant_lr_endpoints_passed=True,small_improvement_detection_passed=True)
    for name,value in [('config.json',cfg),('initialization.json',proof),('resume-verification.json',proof),
                       ('best-reconstruction.json',best),('contract.json',contract)]:
        write(local/name,value);write(OUT/name,value)
    write(local/'checkpoint.json',dict(progress=state['progress'],full_state=True,
        checkpoint_sha256=sha(local/'tokenizer-last.pt'),contract_sha256=contract_sha))
    with tarfile.open(BASE/'source.tar.gz','w:gz') as archive:
        for path in runtime.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                archive.add(path,arcname='runtime/'+str(path.relative_to(runtime)))
    # Everything is ready before stopping the old process. Its immutable best
    # and recent full checkpoints are already mirrored and published. We are
    # explicitly restarting from best, not preserving the regressed live state.
    current_best=json.loads((OLD/'run-output/best-reconstruction.json').read_text())
    assert current_best['checkpoint']['checkpoint_sha256']==proof['source_sha256'],'Best changed during staging'
    old_status=json.loads((OLD_OUT/'supervisor-status.json').read_text())
    supervisor=old_status['supervisor_pid'];child_pid=old_status['child_pid']
    assert b'runner.py' in Path(f'/proc/{supervisor}/cmdline').read_bytes()
    assert os.getpgid(child_pid)==child_pid
    # torchrun workers create their own sessions. Capture descendants before
    # terminating the launcher so no CUDA worker becomes an untracked orphan.
    parents={}
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():continue
        try:
            fields=(process/'stat').read_text().split()
            parents[int(process.name)]=int(fields[3])
        except (OSError,ValueError):pass
    descendants={child_pid}
    while True:
        expanded=descendants|{pid for pid,parent in parents.items() if parent in descendants}
        if expanded==descendants:break
        descendants=expanded
    handoff=dict(time=time.time(),superseded_by=RUN,reason='User requested smaller LR from current best',
        preserved_best=proof,previous_live_status=old_status)
    write(OLD_OUT/'smaller-lr-handoff.json',handoff)
    os.kill(supervisor,signal.SIGTERM)
    for pid in sorted(descendants,reverse=True):
        try:
            if os.getpgid(pid)==pid:os.killpg(pid,signal.SIGKILL)
            else:os.kill(pid,signal.SIGKILL)
        except ProcessLookupError:pass
    write(OLD_OUT/'supervisor-status.json',dict(phase='superseded_smaller_lr',run_id=RUN,time=time.time()))
    command=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc-per-node=1',
        str(runtime/'continuous.py'),'--config',str(local/'config.json')]
    env=dict(os.environ,LASER_VGG16_WEIGHTS='/tmp/laser-s1-best-b200/weights/vgg16-397923af.pth',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        WANDB_DISABLE_GIT='true',WANDB_MODE='online',PYTHONUNBUFFERED='1',
        PYTORCH_ALLOC_CONF='expandable_segments:True')
    write(OUT/'launch.json',dict(time=time.time(),run_id=RUN,source=proof,continuous=True,
        action_on_regression='restore_best_reduce_lr_continue',gpu='NVIDIA B200',
        microbatch=64,accumulation=2,global_batch=128,first_fid_step=42736,
        first_fid_updates=100,full_fid_every_updates=500,full_fid_images=50000,
        command=command,local_log=str(BASE/'training.log')))
    failures=0
    while True:
        with (BASE/'training.log').open('ab',buffering=0) as stream:
            child=subprocess.Popen(command,cwd=runtime,env=env,stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        while child.poll() is None:
            write(OUT/'supervisor-status.json',dict(phase='training',continuous=True,
                time=time.time(),supervisor_pid=os.getpid(),child_pid=child.pid,failures=failures))
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:pass
        if child.returncode==0 or (local/'STOP').exists():
            write(OUT/'supervisor-status.json',dict(phase='user_stop_checkpointed',time=time.time()))
            return
        failures+=1
        write(OUT/'supervisor-status.json',dict(phase='restarting_after_error',failures=failures,time=time.time()))
        if failures>=5:raise RuntimeError('Repeated runtime failure; inspect training.log')
        time.sleep(2)


if __name__=='__main__':main()
