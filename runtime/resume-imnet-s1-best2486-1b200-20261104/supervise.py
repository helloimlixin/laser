"""Preflight real resumed updates, then launch the guarded job in the background."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from prepare import BASE,CONTROL,DURABLE,FROZEN,HERE,OLD,RUN,build_runtime,prepare,write

CHILD=None


def status(phase,**extra):
    write(DURABLE/'supervisor-status.json',dict(phase=phase,time=time.time(),
        supervisor_pid=os.getpid(),run_id=RUN,**extra))


def run(cfg,smoke=False):
    global CHILD
    command=[sys.executable,'-u','-m','torch.distributed.run','--standalone',
        '--nproc-per-node=1',str(FROZEN/'continuation.py'),
        '--config',str(Path(cfg['output'])/'config.json')]
    if smoke:command+=['--smoke','--max-steps','42539']
    log=BASE/('preflight.log' if smoke else 'training.log')
    env=dict(os.environ,LASER_VGG16_WEIGHTS=str(OLD/'weights/vgg16-397923af.pth'),
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        WANDB_DISABLE_GIT='true',WANDB_MODE='online',PYTHONUNBUFFERED='1',
        PYTORCH_ALLOC_CONF='expandable_segments:True')
    with log.open('ab',buffering=0) as stream:
        CHILD=subprocess.Popen(command,cwd=FROZEN,env=env,stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    while CHILD.poll() is None:
        status('preflight' if smoke else 'training',child_pid=CHILD.pid,log=str(log))
        try:CHILD.wait(timeout=15)
        except subprocess.TimeoutExpired:pass
    return CHILD.returncode,log


def stop(signum,_frame):
    if CHILD is not None and CHILD.poll() is None:
        (BASE/'run-output/STOP').touch()
    status('stop_requested',signal=signum)
    raise SystemExit(128+signum)


def main():
    DURABLE.mkdir(parents=True,exist_ok=True)
    lock=(DURABLE/'supervisor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (DURABLE/'launch.json').exists(),'Inspect the production checkpoint before restarting'
    for signum in [signal.SIGTERM,signal.SIGINT]:signal.signal(signum,stop)
    for name in ['train','val']:
        receipt=json.loads((OLD/f'data/{name}.verified.json').read_text())
        assert receipt['verified'] and (OLD/f'data/{name}.tar').stat().st_size==receipt['bytes']
    if CONTROL:
        deadline=time.monotonic()+1200
        while not Path(CONTROL).exists():
            assert time.monotonic()<deadline,'Independent baseline verification timed out'
            status('waiting_for_independent_baseline_control')
            time.sleep(10)
        assert json.loads(Path(CONTROL).read_text())['passed'],'Best-checkpoint baseline control failed'
    if not FROZEN.exists():build_runtime()
    cfg=prepare(smoke=True)
    code,log=run(cfg,smoke=True)
    assert code==0,f'Private preflight failed: {log}'
    output=BASE/'preflight'
    proof=json.loads((output/'smoke-complete.json').read_text())
    assert proof['completed'] and proof['full_state_roundtrip']
    assert proof['model_bit_exact'] and proof['rng_bit_exact'] and proof['optimizer_age']==[42539]
    import torch
    state=torch.load(output/'tokenizer-last.pt',map_location='cpu',weights_only=False,mmap=True)
    ages=sorted({int(v['step']) for v in state['discriminator_optimizer']['state'].values() if 'step' in v})
    assert ages==[32539]
    assert state['progress']==dict(epoch=4,batch=5006,step=42539)
    assert state['learning_dynamics']==dict(lr_factor=1.,nonimproving_checks=0,evaluations=0,reductions=0)
    rate=cfg['continuation']['learning_rate']['maximum']
    for name in ['optimizer','discriminator_optimizer']:
        assert all(rate*.999<g['lr']<=rate*(1.+1e-12) for g in state[name]['param_groups'])
    proof.update(discriminator_optimizer_age=ages,real_imagenet_cursor_preserved=True,gan_exercised=True,
        both_new_learning_rates_verified=True,learning_dynamics_checkpoint_roundtrip=True)
    write(DURABLE/'preflight-verification.json',proof)
    del state
    cfg=prepare()
    launch=dict(run_id=RUN,source_artifact=cfg['continuation']['source_artifact'],
        source_step=42536,source_rfid=cfg['continuation']['source_rfid'],
        source_epoch=4,source_microbatch_cursor=5000,
        world_size=1,gpu='NVIDIA B200',microbatch=64,accumulation=2,global_batch=128,
        generator_initial_lr=rate,discriminator_initial_lr=rate,
        cosine_floor=cfg['continuation']['learning_rate']['minimum'],
        cosine_decay_updates=cfg['continuation']['learning_rate']['decay_updates'],
        first_new_evaluation_step=42636,every_updates=500,
        full_validation_images=50000,maximum_relative_regression=.02,
        halve_lr_after_nonimproving_checks=2,stop_after_nonimproving_checks=4,
        guard_state_saved_inside_full_checkpoints=True,
        inherited_source_baseline=True,target_epochs=10,
        log=str(BASE/'training.log'),started_at=time.time())
    write(DURABLE/'launch.json',launch)
    code,log=run(cfg)
    if code:raise RuntimeError(f'Training exited {code}: {log}')
    checkpoint=json.loads((BASE/'run-output/checkpoint.json').read_text())
    decision=BASE/'run-output/guard-decision.json'
    action=json.loads(decision.read_text())['action'] if decision.exists() else None
    phase=action if action and action.startswith('stop_') else (
        'completed' if checkpoint['progress']['epoch']==10 else 'paused_after_checkpoint')
    status(phase,progress=checkpoint['progress'],guard_action=action)


if __name__=='__main__':
    try:main()
    except Exception as error:
        status('failed',error=str(error))
        raise
