"""Supervise fresh beam16 joint16k training and recover complete checkpoints."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

RUN = 'imagenet-beam16-joint1694m-native-scratch200-4h200-20261107'
BASE = Path('/tmp') / RUN
OUT = Path('/workspace/Projects/laser/outputs') / RUN
CHILD = None
STOPPING = False
MIRRORED = {}


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def status(phase, **extra):
    progress = read(BASE / 'live-training/train/progress-rank0.json')
    write(OUT / 'status.json', dict(phase=phase, run_id=RUN, supervisor_pid=os.getpid(),
        child_pid=CHILD.pid if CHILD else None, target_epochs=200, target_generation_fid=11.,
        latest_progress=progress, log=str(BASE/'production.log'), time=time.time(), **extra))
    for rank in range(4):
        path = BASE / f'live-training/train/startup-rank{rank}.json'
        value = read(path)
        if value is not None and MIRRORED.get(rank) != value:
            write(OUT / f'startup-rank{rank}.json', value); MIRRORED[rank] = value
    source = BASE / 'production.log'
    if source.exists():
        with source.open('rb') as stream:
            stream.seek(max(0, source.stat().st_size-256*1024))
            (OUT/'production-tail.log').write_bytes(stream.read())


def stop(signum, _):
    global STOPPING
    STOPPING = True
    (BASE/'STOP').touch()
    if CHILD is not None:
        try: os.killpg(CHILD.pid, signal.SIGTERM)
        except ProcessLookupError: pass
    status('stopping_at_user_request', signal=signum)


def environment():
    env = dict(os.environ)
    for name in ('WANDB_SERVICE', '_WANDB_SERVICE'): env.pop(name, None)
    env.update(PYTHONPATH='/tmp/laser-stage2-workspace-deps:/tmp/laser-wandb-logging-deps:/tmp/laser-coefficient-audit-deps',
        WANDB_API_KEY=Path('/root/.config/laser/wandb.key').read_text().strip(),
        WANDB_BASE_URL='https://api.wandb.ai', WANDB_MODE='online', WANDB_DISABLE_GIT='true',
        WANDB_CONSOLE='off', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
        PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='0,1,2,3', NCCL_DEBUG='WARN',
        LASER_STAGE2_OUTPUT=str(OUT), LASER_GENERATION_BATCH='256', LASER_DATA_WORKERS='8',
        LASER_TRAIN_SDPA='1', LASER_FUSED_ADAMW='1', LASER_COMPILE_NATIVE_LOSS='1',
        LASER_EVAL_EVERY_EPOCHS='2', LASER_LR_AMPLITUDE_FACTOR='1.0',
        LASER_PRESERVE_INCEPTION_MIN='0', LASER_QUALITY_EARLY_STOP='0',
        LASER_MAX_RECOVERY_LOSS='inf',
        LASER_CHECKPOINT_STAGING_DIR=str(BASE/'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(BASE/'checkpoint-cache'),
        TORCHINDUCTOR_COMPILE_THREADS='2', TORCHINDUCTOR_CACHE_DIR=str(BASE/'compile-cache/production'),
        PYTORCH_ALLOC_CONF='expandable_segments:True')
    return env


def main():
    global CHILD
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    lock = (BASE/'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (OUT/'launch.json').exists() or args.resume is not None, 'Inspect an existing launch before starting another supervisor'
    if args.resume is not None:
        restart=read(BASE/'runtime-relaunch-verification.json')
        assert restart['passed'] and Path(restart['checkpoint'])==args.resume
        assert args.resume.stat().st_size==restart['bytes'] and restart['global_step']==250
        (BASE/'STOP').unlink(missing_ok=True)
    assert not (BASE/'STOP').exists()
    proof = read(BASE/'preflight-verification.json')
    assert proof['passed'] and proof['four_gpu_training'] and proof['full_state_roundtrip']
    assert proof['frozen_tokenizer_unchanged'] and proof['native_sampling_finite']
    assert proof['parameter_count'] == 1692336000 and proof['global_batch'] == 2048
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, stop)
    command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
        '--nproc-per-node=4', str(BASE/'train.py'), '--microbatch', '256', '--epochs', '200']
    base_command=list(command)
    if args.resume is not None:command+=['--resume',str(args.resume)]
    initial = dict(command=command, run_id=RUN, fresh_stage2=True, source_ar_weights_loaded=False,
        parameter_count=1692336000, world_size=4, microbatch=256, accumulation_steps=2,
        global_batch=2048, training_epochs=200, cosine_schedule_epochs=200,
        reconstruction_tokenizer='verified beam16 joint16384, rFID4.5143008685 on full validation',
        objective='native stochastic soft target CE, T0.5',
        full50k_generation_evaluation_epochs=[1,2,4,6], generation_evaluation_interval_after_epoch_one=2,
        preview_first_step=10, preview_every_steps=250, automatic_checkpoint_recovery=True,
        supervisor_pid=os.getpid(), started_at=time.time())
    if args.resume is None:write(OUT/'launch.json', initial)
    else:write(OUT/'runtime-relaunch.json',dict(initial,resume_step=250,reason='Separate checkpoint persistence and optional cloud model uploads'))
    retry_counts = {}
    while not STOPPING and not (BASE/'STOP').exists():
        with (BASE/'production.log').open('ab', buffering=0) as log:
            CHILD = subprocess.Popen(command, cwd=BASE, env=environment(), stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while CHILD.poll() is None:
            status('training', command=command)
            try: CHILD.wait(timeout=15)
            except subprocess.TimeoutExpired: pass
        if STOPPING:
            status('stopped_by_user', exit_code=CHILD.returncode); return
        if CHILD.returncode == 0:
            completed = read(OUT/'completed.json')
            assert completed and completed['epochs'] == 200 and completed['global_step'] == 125200
            status('completed', completion=completed); return
        receipt = read(OUT/'checkpoint.json')
        if receipt is None:
            status('failed_before_first_complete_checkpoint', exit_code=CHILD.returncode)
            raise RuntimeError('Training exited before its first complete checkpoint; refusing an unlogged fresh restart')
        path = Path(receipt['path'])
        assert receipt['state'] in ('COMMITTED','PERSISTED') and path.is_file() and path.stat().st_size == receipt['bytes']
        if receipt['state']=='COMMITTED':assert receipt['artifact'].split('/')[-1].startswith(RUN+'-checkpoint:')
        else:assert receipt['persistent_checkpoint_complete']
        if args.resume is not None and receipt['global_step']<250:
            path=args.resume;receipt=dict(global_step=250)
        step = receipt['global_step']; retry_counts[step] = retry_counts.get(step, 0) + 1
        if retry_counts[step] > 3:
            status('failed_after_repeated_recovery', checkpoint_step=step)
            raise RuntimeError(f'Three recoveries failed at step {step}')
        status('recovering_from_complete_checkpoint', checkpoint_step=step, retry=retry_counts[step])
        command = base_command + ['--resume', str(path)]


if __name__ == '__main__':
    try: main()
    except Exception as error:
        status('failed', error=str(error)); raise
