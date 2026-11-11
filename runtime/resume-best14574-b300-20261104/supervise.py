"""Verify three discarded resumed updates, then supervise detached training."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

BASE = Path('/tmp/laser-best-b300-20261104/frozen')
RUN = 'imnet-j16384-best14574-continue100-1b300-20261104'
OUT = Path('/workspace/Projects/laser/outputs') / RUN
LOCAL = BASE / 'run-output'
CHILD = None
STOPPING = False


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def status(phase, **extra):
    pause = read(OUT / 'manual-pause.json')
    if pause and pause.get('paused') and phase == 'training':
        phase = 'paused_for_investigation'
        extra['pause_reason'] = pause['reason']
    write(OUT / 'status.json', dict(phase=phase, run_id=RUN, time=time.time(),
        supervisor_pid=os.getpid(), child_pid=CHILD.pid if CHILD else None,
        target_epoch=100, target_step=62600,
        latest_progress=read(BASE / 'live-training/train/progress-rank0.json'), **extra))
    for name in ['checkpoint.json', 'generation-progress.json', 'completed.json', 'wandb.json',
                 'matched-baseline.json', 'source-artifact.json', 'best-inception.json', 'workspace.json']:
        value = read(LOCAL / name)
        if value is not None:
            write(OUT / name, value)
    for directory in ['train', 'evaluations']:
        source = LOCAL / directory
        if source.exists():
            for path in source.glob('*'):
                if path.is_file() and path.suffix in {'.json', '.jsonl'}:
                    target = OUT / directory / path.name
                    target.parent.mkdir(exist_ok=True)
                    if not target.exists() or path.stat().st_mtime_ns != target.stat().st_mtime_ns:
                        temporary = target.with_suffix('.pending')
                        # This object-backed shared mount does not support
                        # copystat/chmod; copy contents without local metadata.
                        shutil.copyfile(path, temporary)
                        temporary.replace(target)
    best = read(LOCAL / 'persistent-train/checkpoints/best.json')
    if best is not None:
        write(OUT / 'best-checkpoint.json', best)


def environment():
    env = dict(os.environ)
    for key in ['WANDB_SERVICE', '_WANDB_SERVICE']:
        env.pop(key, None)
    env.update(WANDB_API_KEY=Path('/root/.config/laser/wandb.key').read_text().strip(),
        WANDB_BASE_URL='https://api.forge.coreweave.com', WANDB_MODE='online',
        WANDB_DISABLE_GIT='true', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        PYTHONUNBUFFERED='1', LASER_MAX_RECOVERY_LOSS='8.25', LASER_LR_AMPLITUDE_FACTOR='0.0375',
        LASER_GENERATION_BATCH='64', LASER_EVAL_EVERY_EPOCHS='1',
        LASER_PRESERVE_INCEPTION_MIN='93.50654602050781', LASER_QUALITY_EARLY_STOP='0',
        LASER_STAGE2_OUTPUT=str(LOCAL), LASER_COMPILE_NATIVE_LOSS='1', LASER_FUSED_ADAMW='1',
        LASER_TRAIN_SDPA='0', LASER_DATA_WORKERS='12', LASER_INHERIT_SOURCE_EVALUATION='1',
        PYTORCH_ALLOC_CONF='expandable_segments:True', TORCHINDUCTOR_COMPILE_THREADS='4')
    return env


def command(checkpoint):
    return [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
        '--nproc-per-node=1', str(BASE / 'train.py'), '--microbatch', '256',
        '--epochs', '100', '--resume', str(checkpoint)]


def child(cmd, phase):
    global CHILD
    with (BASE / (phase + '.log')).open('ab', buffering=0) as handle:
        CHILD = subprocess.Popen(cmd, cwd=BASE, env=environment(), stdin=subprocess.DEVNULL,
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        while CHILD.poll() is None:
            status(phase, log=str(BASE / (phase + '.log')))
            try:
                CHILD.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
    return CHILD.returncode


def stop(signum, _frame):
    global STOPPING
    STOPPING = True
    if CHILD is not None and CHILD.poll() is None:
        os.killpg(CHILD.pid, signal.SIGINT)
    status('interrupted', signal=signum)
    raise SystemExit(128 + signum)


class AdoptedChild:
    """Monitor a surviving detached torchrun after supervisor-only failure."""
    def __init__(self, pid):
        self.pid = pid
        assert self.poll() is None, 'Requested trainer is not alive'
        cmdline = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        assert str(BASE / 'train.py').encode() in cmdline

    def poll(self):
        path = Path(f'/proc/{self.pid}/stat')
        try:
            running = path.read_text().split(')', 1)[1].split()[0] not in {'Z', 'X'}
        except FileNotFoundError:
            running = False
        if running:
            return None
        return 0 if (LOCAL / 'completed.json').exists() else 1

    def wait(self, timeout):
        time.sleep(timeout)
        code = self.poll()
        if code is None:
            raise subprocess.TimeoutExpired(str(self.pid), timeout)
        return code


def finish(code):
    retry_step = None
    retries = 0
    while code and not STOPPING:
        pause = read(OUT / 'manual-pause.json')
        if pause and pause.get('paused'):
            status('paused_trainer_exited', exit_code=code)
            return
        receipt = read(LOCAL / 'checkpoint.json')
        if not receipt or receipt.get('state') != 'COMMITTED':
            raise RuntimeError(f'Training exited {code} before a durable recovery checkpoint')
        path = Path(receipt['path'])
        assert path.is_file() and path.stat().st_size == receipt['bytes']
        step = receipt['global_step']
        retries = retries + 1 if step == retry_step else 1
        retry_step = step
        if retries > 3:
            raise RuntimeError('Three recovery attempts failed at the same checkpoint')
        status('recovering', recovery_step=step, retry_count=retries)
        code = child(command(path), f'recovery-{step}-{retries}')
    complete = read(LOCAL / 'completed.json')
    assert complete and complete['epochs'] == 100 and complete['global_step'] == 62600
    status('completed', completion=complete)


def main():
    global CHILD
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, stop)
    if os.environ.get('LASER_ADOPT_PID'):
        assert (OUT / 'launch.json').exists()
        CHILD = AdoptedChild(int(os.environ['LASER_ADOPT_PID']))
        while CHILD.poll() is None:
            status('training', adopted_existing_trainer=True, log=str(BASE / 'training.log'))
            try:
                CHILD.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        finish(CHILD.poll())
        return
    assert not (OUT / 'launch.json').exists(), 'Inspect existing launch before relaunch'
    while not (BASE / 'data/train-ready.json').exists():
        status('preparing_local_training_data', data_status=read(BASE / 'data/status.json'))
        time.sleep(10)
    ready = read(BASE / 'data/train-ready.json')
    assert ready['ready'] and ready['training_images'] == 1281167
    initial = BASE / 'inputs/resume-checkpoint.pt'
    if child(command(initial) + ['--smoke-updates', '45701'], 'preflight'):
        raise RuntimeError('Resumed-update preflight failed; see preflight.log')
    proof = read(LOCAL / 'smoke/verification.json')
    assert proof['passed'] and proof['global_step'] == proof['all_adam_steps'] == 45701
    assert proof['optimizer_state_count'] == 780 and proof['frozen_tokenizer_unchanged']
    init = proof['initialization']
    assert init['global_step'] == init['restored_optimizer_age'] == 45698
    assert not init['fresh_stage2'] and not init['fresh_optimizer']
    assert init['global_batch'] == 2048 and init['microbatch'] == 256 and init['accumulation_steps'] == 8
    assert init['resume_memory']['source_mmap_released']
    assert init['lr_recovery']['applied_ratio'] == 0.25
    assert init['lr_recovery']['scheduler_step'] == 45698
    assert init['lr_recovery']['scheduler_horizon'] == 62600
    assert abs(init['restored_lr'] - 3.175201262590684e-6) < 1e-17
    roundtrip = read(LOCAL / 'smoke/checkpoint-roundtrip.json')
    assert roundtrip['passed'] and roundtrip['global_step'] == 45701
    write(OUT / 'preflight.json', proof)
    write(OUT / 'reproducibility/checkpoint-roundtrip.json', roundtrip)
    write(BASE / 'full-size-smoke-verification.json', proof)
    launch = dict(command=command(initial) + ['--new-wandb-run'], run_id=RUN,
        source_step=45698, source_epoch=73, target_epoch=100, target_step=62600,
        initial_lr=init['restored_lr'], checkpoint_and_adam_verified=True,
        preflight_updates_discarded=True, quality_early_stop=False,
        hardware='1 B300', global_batch=2048, microbatch=256, accumulation_steps=8,
        log=str(BASE / 'training.log'), local_output=str(LOCAL), started_at=time.time(),
        url='https://forge.coreweave.com/wandb/helloimlixin-rutgers/laser/runs/' + RUN)
    write(OUT / 'launch.json', launch)
    code = child(launch['command'], 'training')
    # Only crash recovery from committed, complete checkpoints, never from a
    # failed pilot, arbitrary local file, or a quality-stop result.
    finish(code)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error=str(error))
        raise
