"""Run the bounded direction search and preserve live progress on shared storage."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

BASE = Path('/tmp/laser-direction-search-20261104/frozen')
RUN = 'imnet-j16384-best14574-direction-search-1b300-20261104'
OUT = Path('/workspace/Projects/laser/outputs') / RUN
LOCAL = BASE / 'run-output'
CHILD = None


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def status(phase, **extra):
    write(OUT / 'status.json', dict(phase=phase, time=time.time(), supervisor_pid=os.getpid(),
        child_pid=CHILD.pid if CHILD else None, run_id=RUN,
        stage=read(LOCAL / 'stage.json'), training=read(LOCAL / 'training-progress.json'),
        generation=read(LOCAL / 'generation-progress.json'), **extra))
    for name in ['source-artifact.json', 'wandb.json']:
        value = read(LOCAL / name)
        if value is not None:
            write(OUT / name, value)


def stop(signum, _frame):
    (OUT / 'STOP').touch()
    if CHILD is not None and CHILD.poll() is None:
        os.killpg(CHILD.pid, signal.SIGINT)
    status('stopped_by_user', signal=signum)
    raise SystemExit(128 + signum)


def main():
    global CHILD
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, stop)
    assert not (LOCAL / 'completed.json').exists(), 'Search already completed'
    env = dict(os.environ)
    for key in ['WANDB_SERVICE', '_WANDB_SERVICE']:
        env.pop(key, None)
    env.update(WANDB_API_KEY=Path('/root/.config/laser/wandb.key').read_text().strip(),
        WANDB_BASE_URL='https://api.forge.coreweave.com', WANDB_MODE='online', WANDB_DISABLE_GIT='true',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1',
        PYTORCH_ALLOC_CONF='expandable_segments:True', TORCHINDUCTOR_COMPILE_THREADS='4')
    command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
               '--nproc-per-node=1', str(BASE / 'search.py')]
    write(OUT / 'launch.json', dict(command=command, time=time.time(), baseline_source_step=45698,
        protocol="fast-screen-v2", screen_images=2000, screen_batch=256,
        full50k_evaluations_maximum=1, independent_trials=4, steps_per_trial=8,
        automatic_long_training=False, best_model_preservation=True,
        url='https://forge.coreweave.com/wandb/helloimlixin-rutgers/laser/runs/' + RUN))
    for attempt in range(1, 4):
        if (OUT / 'STOP').exists():
            status('stopped_by_user')
            return
        log = BASE / f'search-attempt-{attempt}.log'
        with log.open('ab', buffering=0) as handle:
            CHILD = subprocess.Popen(command, cwd=BASE, env=env, stdin=subprocess.DEVNULL,
                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            while CHILD.poll() is None:
                status('running_search', attempt=attempt, log=str(log))
                try:
                    CHILD.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        if (LOCAL / 'completed.json').exists():
            status('completed', completion=read(LOCAL / 'completed.json'))
            return
        if (OUT / 'STOP').exists():
            status('stopped_by_user', exit_code=CHILD.returncode)
            return
        # Initialization/preflight failures require inspection; only production
        # evaluation interruption is retried from completed experiment records.
        if not (LOCAL / 'preflight.json').exists():
            raise RuntimeError(f'Preflight failed; inspect {log}')
        status('retrying_search', exit_code=CHILD.returncode, attempt=attempt)
    raise RuntimeError('Three attempts failed; original best remains intact')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error_type=type(error).__name__, error=str(error))
        raise
