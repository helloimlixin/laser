"""Resume from NVMe; publish full checkpoints directly to W&B; restart failures."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

BASE = Path('/tmp/laser-beam16-a100-20261105/frozen')
OUT = BASE.parent/'production-nvme'
RUN = 'imnet-j16384-a4096-beam16-rfid451-scratch100-2b300-20261104'
CHILD = None
STOPPING = False


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2, default=str)+'\n')
    temporary.replace(path)


def status(phase, **extra):
    write(OUT/'status.json', dict(phase=phase, time=time.time(),
        supervisor_pid=os.getpid(), run_id=RUN, world_size=4,
        hardware='4 NVIDIA A100-SXM4-80GB', global_batch=2048,
        target_epoch=100, target_step=62600,
        progress=read(OUT/'train/progress-rank0.json'),
        checkpoint=read(OUT/'checkpoint.json'),
        storage='Local NVMe for all active files; verified W&B cloud checkpoints',
        **extra))


def stop(signum, _):
    global STOPPING
    STOPPING = True
    if CHILD is not None and CHILD.poll() is None:
        os.killpg(CHILD.pid, signal.SIGTERM)
    status('interrupted', signal=signum)
    raise SystemExit(128+signum)


def checkpoint():
    import torch
    candidates = sorted((OUT/'persistent-train/checkpoints').glob('step-*.pt'), reverse=True)
    candidates += [BASE.parent/'recovered-step-0011010.pt', BASE/'inputs/resume-checkpoint.pt']
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
            step = int(payload['global_step'])
            assert payload['run_id'] == RUN
            assert len(payload['optimizer']['state']) == 780
            assert all(int(v['step']) == step for v in payload['optimizer']['state'].values())
            assert payload['scheduler']['after']['last_epoch'] == step
            assert payload['scheduler']['after']['T_max'] == 62600
            del payload
            return path, step
        except Exception as error:
            write(OUT/'rejected-checkpoints'/f'{path.stem}.json', dict(path=str(path), error=str(error)))
    raise RuntimeError('No valid continuation checkpoint')


def main():
    global CHILD
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/'watchdog.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, stop)
    env = dict(os.environ)
    for key in ['WANDB_SERVICE', '_WANDB_SERVICE']:
        env.pop(key, None)
    env.update(WANDB_API_KEY=Path('/root/.config/laser/wandb.key').read_text().strip(),
        WANDB_BASE_URL='https://api.forge.coreweave.com', WANDB_MODE='online',
        CUDA_VISIBLE_DEVICES='0,1,2,3', PYTHONUNBUFFERED='1',
        LASER_STAGE2_OUTPUT=str(OUT), WANDB_CONSOLE='off',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        LASER_MAX_RECOVERY_LOSS='20.0', LASER_LR_AMPLITUDE_FACTOR='1.0',
        LASER_GENERATION_BATCH='64', LASER_EVAL_EVERY_EPOCHS='2',
        LASER_PRESERVE_INCEPTION_MIN='0.0', LASER_QUALITY_EARLY_STOP='0',
        LASER_COMPILE_NATIVE_LOSS='0', LASER_FUSED_ADAMW='1', LASER_TRAIN_SDPA='0',
        PYTORCH_ALLOC_CONF='expandable_segments:True', NCCL_NVLS_ENABLE='0')
    attempt = 0
    failures = 0
    while not STOPPING:
        source, step = checkpoint()
        execution = read(BASE/'inputs/execution.json')
        execution.update(initial_global_step=step, source_step=step, source_epoch=step//626,
            initial_epoch=step//626, resume_world_size=4,
            checkpoint_storage='Local NVMe atomic writes + verified W&B cloud commits',
            active_output=str(OUT), automatic_restart=True)
        write(BASE/'inputs/execution.json', execution)
        cmd = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
            '--nproc-per-node=4', str(BASE/'train.py'), '--microbatch', '128',
            '--epochs', '100', '--resume', str(source)]
        launch = dict(command=cmd, run_id=RUN, start_step=step, source_checkpoint=str(source),
            world_size=4, microbatch=128, accumulation_steps=4, global_batch=2048,
            training_attention='native bmm', hardware='4 NVIDIA A100-SXM4-80GB',
            scheduler_horizon=62600, target_epoch=100, attempt=attempt, started_at=time.time(),
            storage=str(OUT), log=str(OUT/'training.log'), new_wandb_run=False)
        write(OUT/'launch.json', launch)
        write(OUT/'attempts'/f'{attempt:04d}.json', launch)
        started = time.time()
        with (OUT/'training.log').open('ab', buffering=0) as log:
            CHILD = subprocess.Popen(cmd, cwd=BASE, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while CHILD.poll() is None:
                status('training', child_pid=CHILD.pid, source_step=step,
                    attempt=attempt, restarts=attempt)
                activity = [OUT/'train/progress-rank0.json', OUT/'generation-progress.json', OUT/'training.log']
                last = max([started]+[p.stat().st_mtime for p in activity if p.exists()])
                if time.time()-last > 1800:
                    os.killpg(CHILD.pid, signal.SIGTERM)
                    try:
                        CHILD.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(CHILD.pid, signal.SIGKILL)
                try:
                    CHILD.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
        code = CHILD.returncode
        complete = read(OUT/'completed.json')
        if code == 0 and complete and complete['global_step'] == 62600:
            status('completed', completion=complete)
            return
        progress = read(OUT/'train/progress-rank0.json') or {}
        progress_path = OUT/'train/progress-rank0.json'
        made_progress = (progress_path.exists() and progress_path.stat().st_mtime >= started
                         and progress.get('train/global_step', step) > step)
        failures = 0 if made_progress else failures+1
        write(OUT/'attempts'/f'{attempt:04d}-exit.json',
            dict(returncode=code, time=time.time(), source_step=step, progress=progress))
        if failures >= 5:
            status('failed', returncode=code, error='Five consecutive startup failures; inspect local log')
            return
        attempt += 1
        status('restarting', returncode=code, attempt=attempt)
        time.sleep(min(120, 15*max(1, failures)))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error=str(error))
        raise
