"""Choose the measured B200 batch size and run a durable stage-one continuation."""
import fcntl
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

from prepare import BASE, DURABLE, FROZEN, HERE, RUN, patch_runtime, prepare, write

CHILD = None


def status(phase, **extra):
    write(DURABLE/'supervisor-status.json',dict(
        phase=phase,time=time.time(),supervisor_pid=os.getpid(),run_id=RUN,**extra))


def environment():
    return dict(os.environ, LASER_VGG16_WEIGHTS=str(BASE/'weights/vgg16-397923af.pth'),
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        WANDB_DISABLE_GIT='true',WANDB_MODE='online',PYTHONUNBUFFERED='1',
        PYTORCH_ALLOC_CONF='expandable_segments:True')


def run(cfg, smoke=False):
    global CHILD
    command=[sys.executable,'-u','-m','torch.distributed.run','--standalone',
             '--nproc-per-node=1',str(FROZEN/'continuation.py'),
             '--config',str(Path(cfg['output'])/'config.json')]
    if smoke:
        command+=['--smoke','--max-steps','40041']
    log=BASE/(f"benchmark-mb{cfg['microbatch']}.log" if smoke else 'training.log')
    with log.open('ab',buffering=0) as handle:
        CHILD=subprocess.Popen(command,cwd=FROZEN,env=environment(),stdin=subprocess.DEVNULL,
                               stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
    while CHILD.poll() is None:
        status('benchmark' if smoke else 'training',child_pid=CHILD.pid,log=str(log),
               microbatch=cfg['microbatch'],accumulation=cfg['accumulation'])
        try:
            CHILD.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass
    return CHILD.returncode,log


def benchmark_result(microbatch):
    output=BASE/f'benchmark-mb{microbatch}'
    proof=json.loads((output/'smoke-complete.json').read_text())
    assert proof['completed'] and proof['full_state_roundtrip'] and proof['optimizer_age']==[40041]
    import torch
    state=torch.load(output/'tokenizer-last.pt',map_location='cpu',weights_only=False,mmap=True)
    d_age={int(v['step']) for v in state['discriminator_optimizer']['state'].values() if 'step' in v}
    assert d_age=={30041}
    metrics=[json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines()]
    rows=[m for m in metrics if m['phase']=='train'][-3:]
    return dict(microbatch=microbatch,accumulation=128//microbatch,
        median_images_per_second=statistics.median(m['images_per_second'] for m in rows),
        peak_memory_gib=max(m['peak_memory_gib'] for m in rows),
        optimizer_age=proof['optimizer_age'],discriminator_optimizer_age=sorted(d_age),
        model_and_rng_roundtrip_passed=proof['model_bit_exact'] and proof['rng_bit_exact'])


def stop(signum,_frame):
    if CHILD is not None and CHILD.poll() is None:
        os.killpg(CHILD.pid,signal.SIGINT)
    status('interrupted',signal=signum)
    raise SystemExit(128+signum)


def main():
    DURABLE.mkdir(parents=True,exist_ok=True)
    lock=(DURABLE/'supervisor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for signum in [signal.SIGTERM,signal.SIGINT]:signal.signal(signum,stop)
    assert not (DURABLE/'launch.json').exists(),'Inspect production checkpoint before relaunch'
    patch_runtime()
    results=[]
    for microbatch in [32,64]:
        if not (BASE/f'benchmark-mb{microbatch}'/'smoke-complete.json').exists():
            cfg=prepare(microbatch,smoke=True)
            code,log=run(cfg,smoke=True)
            if code:
                tail=log.read_text()[-12000:]
                if 'out of memory' in tail.lower():
                    results.append(dict(microbatch=microbatch,out_of_memory=True))
                    continue
                raise RuntimeError(f'Execution preflight failed: {log}\n{tail}')
        results.append(benchmark_result(microbatch))
        write(DURABLE/'execution-benchmarks.json',results)
    selected=max([r for r in results if not r.get('out_of_memory')],
                 key=lambda r:r['median_images_per_second'])
    write(DURABLE/'selected-execution.json',selected)
    while not all((BASE/f'data/{name}.verified.json').exists() for name in ['train','val']):
        status('waiting_for_archive_verification',selected_execution=selected)
        time.sleep(10)
    cfg=prepare(selected['microbatch'])
    write(DURABLE/'launch.json',dict(
        run_id=RUN,source_run=cfg['continuation']['source_run'],
        source_step=40036,source_epoch=4,source_rfid=2.9342634986184635,
        world_size=1,gpu='NVIDIA B200',global_batch=128,
        selected_execution=selected,learning_rate=cfg['continuation']['learning_rate'],
        target_epoch=10,target_step=100090,log=str(BASE/'training.log'),
        first_new_full_evaluation_step=40536,evaluation_images=50000,
        started_at=time.time()))
    code,log=run(cfg)
    if code:
        raise RuntimeError(f'Training exited {code}: {log}')
    progress=json.loads((BASE/'run-output/checkpoint.json').read_text())['progress']
    status('completed' if progress['epoch']==10 else 'paused_after_checkpoint',progress=progress)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        status('failed',error=str(error))
        raise
