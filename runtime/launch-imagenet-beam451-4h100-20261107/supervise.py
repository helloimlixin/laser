"""Keep one four-GPU worker alive, recovering only complete checkpoints."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

BASE = Path('/tmp/laser-beam451-stage2-4h100-20261107')
RUN = 'imagenet-beam451-aligned-joint16k-scratch100-4h100-20261107'
OUT = Path('/workspace/Projects/laser/outputs') / RUN
CHILD = None
STOP = False


def environment():
    env = dict(os.environ)
    for name in ('WANDB_SERVICE', '_WANDB_SERVICE'): env.pop(name, None)
    env.update(WANDB_API_KEY=Path('/root/.config/laser/wandb.key').read_text().strip(),
        WANDB_BASE_URL='https://api.wandb.ai', WANDB_MODE='online', WANDB_CONSOLE='off',
        WANDB_DISABLE_GIT='true', CUDA_VISIBLE_DEVICES='0,1,2,3',
        LASER_STAGE2_OUTPUT=str(OUT), LASER_BEAM_BASE=str(BASE), TORCH_HOME=str(BASE/'torch-cache'),
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
        PYTHONUNBUFFERED='1', NCCL_DEBUG='WARN', NCCL_NVLS_ENABLE='0',
        LASER_FUSED_ADAMW='1', LASER_TRAIN_SDPA='1', LASER_SDPA_MIN_SEQ='16',
        LASER_COMPILE_NATIVE_LOSS='1', LASER_COMPILE_LOSS_DYNAMIC='1',
        LASER_EXACT_BEAM_KERNELS='1', LASER_FAST_BEAM_TOPK='0',
        TORCHINDUCTOR_COMPILE_THREADS='2', TORCHINDUCTOR_CACHE_DIR=str(BASE/'compile-cache'),
        PYTORCH_ALLOC_CONF='expandable_segments:True')
    return env


def write(path, value):
    tmp=path.with_suffix('.pending');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def status(phase, **extra):
    local = BASE / 'live-training/train/progress-rank0.json'
    progress = json.loads(local.read_text()) if local.exists() else None
    write(OUT/'status.json', dict(phase=phase,run_id=RUN,supervisor_pid=os.getpid(),
        child_pid=CHILD.pid if CHILD else None,world_size=4,hardware='4 H100 80GB',
        global_batch=2048,epochs=100,latest_progress=progress,time=time.time(),**extra))
    for rank in range(4):
        for prefix in ('startup', 'targets', 'progress'):
            source=BASE/f'live-training/train/{prefix}-rank{rank}.json'
            if source.exists():write(OUT/f'{prefix}-rank{rank}.json',json.loads(source.read_text()))
    log=BASE/'production.log'
    if log.exists():
        with log.open('rb') as stream:
            stream.seek(max(0,log.stat().st_size-64000))
            (OUT/'production-tail.log').write_bytes(stream.read())


def stop(sig, _):
    global STOP
    STOP=True
    if CHILD is not None:
        try:os.killpg(CHILD.pid,signal.SIGTERM)
        except ProcessLookupError:pass


def main():
    global CHILD
    lock=(BASE/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    proof=json.loads((BASE/'preflight-verification.json').read_text())
    assert proof['passed'] and proof['four_gpu_training'] and proof['full_state_roundtrip']
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    while not (BASE/'data/ready.json').exists():
        if STOP:status('stopped_by_signal');return
        status('staging_data')
        time.sleep(15)
    ready=json.loads((BASE/'data/ready.json').read_text())
    assert ready['ready'] and ready['training_images']==1281167 and ready['classes']==1000
    attempts={}
    while not STOP:
        command=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc-per-node=4',str(BASE/'train.py'),'--microbatch','128','--epochs','100']
        receipt=OUT/'checkpoint.json'
        step=0
        if receipt.exists():
            checkpoint=json.loads(receipt.read_text());path=Path(checkpoint['path'])
            if not path.exists():path=Path(checkpoint['persistent_path'])
            assert path.stat().st_size==checkpoint['bytes']
            step=checkpoint['global_step'];command+=['--resume',str(path)]
        attempts[step]=attempts.get(step,0)+1
        if attempts[step]>3:raise RuntimeError('Repeated failure at same complete checkpoint')
        with (BASE/'production.log').open('ab',buffering=0) as log:
            CHILD=subprocess.Popen(command,cwd=BASE,env=environment(),stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(OUT/'launch.json',dict(command=command,supervisor_pid=os.getpid(),child_pid=CHILD.pid,
            run_id=RUN,resume_step=step,world_size=4,microbatch=128,accumulation_steps=4,
            global_batch=2048,epochs=100,teacher='beam16 aligned conditional posterior T2.0',time=time.time()))
        while CHILD.poll() is None:
            status('training',resume_step=step)
            try:CHILD.wait(timeout=15)
            except subprocess.TimeoutExpired:pass
        if STOP:status('stopped_by_signal');return
        if CHILD.returncode==0:
            assert (OUT/'completed.json').exists()
            status('completed');return
        if not receipt.exists():
            raise RuntimeError('Worker failed before first complete checkpoint; inspect production-tail.log')
        status('recovering',exit_code=CHILD.returncode)


if __name__=='__main__':
    try:main()
    except Exception as error:status('failed',error=str(error));raise
