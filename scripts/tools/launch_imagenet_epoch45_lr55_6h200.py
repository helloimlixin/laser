"""Supervise the resumed six-H200 ImageNet launch and its durable checkpoint."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

BASE = Path('/tmp/laser-imagenet-epoch45-lr55-stage2')
EVIDENCE = Path('/workspace/Projects/laser/outputs/imagenet-rfid421-epoch45-lr55-5h200-20261003')
RUN_ID = 'imagenet-rfid421-epoch45-lr55-5h200-20261003'
RUN_URL = f'https://wandb.ai/helloimlixin-rutgers/laser/runs/{RUN_ID}'
PYTHON = '/opt/laser-venv/bin/python'
STOPPING = False


def record(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def stop(_signum, _frame):
    global STOPPING
    STOPPING = True


def supervise():
    lock = (BASE / 'production.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, stop)
    assert json.loads((EVIDENCE / 'preflight-summary.json').read_text())['passed']
    assert json.loads((EVIDENCE / 'fresh-augmentation-audit.json').read_text())['acceptable']
    ready = json.loads(Path('/tmp/laser-imagenet-stage2/imagenet/training-ready.json').read_text())
    assert ready['training_images'] == 1281167 and ready['md5'] == '1d675b47d978889d74fa0da5fadfb00e'
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES='0,1,2,3,4,5', LASER_RUNTIME_ROOT=str(BASE / 'runtime/runtime'),
        LASER_RUN_BASE=str(BASE), LASER_PERSISTENT_BASE=str(EVIDENCE),
        LASER_ALLOW_OPTIMIZER_BOUNDARY_WORLD_SIZE_CHANGE='1',
        LASER_LAYOUT_MIGRATION_DIR=str(EVIDENCE / 'resume-6h200-20261003/rng'),
        LASER_CHECKPOINT_DIRECT_UPLOAD='1',
        LASER_PHASE='production', LASER_ACCUMULATION='2', LASER_COMPILE_BLOCKS='1',
        LASER_COMPILE_OBJECTIVE='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
        OPENBLAS_NUM_THREADS='8', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        TORCHINDUCTOR_COMPILE_THREADS='2', TORCHINDUCTOR_CACHE_DIR=str(BASE / 'inductor-cache'),
        TORCH_HOME=str(BASE / 'torch-cache'), NCCL_NVLS_ENABLE='0', PYTHONUNBUFFERED='1',
        LASER_CHECKPOINT_STAGING_DIR='/dev/shm/laser-imagenet-checkpoint-staging',
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR='/dev/shm/laser-imagenet-checkpoint-upload-cache',
        LASER_CHECKPOINT_IMMUTABLE_FILES='1', WANDB_DIR=str(BASE / 'wandb'),
        WANDB_CACHE_DIR='/dev/shm/laser-wandb-cache', WANDB_DATA_DIR='/dev/shm/laser-wandb-data',
        WANDB_CONFIG_DIR=str(BASE / 'wandb-config'),
    )
    env['WANDB_API_KEY'] = Path('/root/.config/laser/wandb.key').read_text().strip()
    for directory in ('WANDB_DIR', 'WANDB_CACHE_DIR', 'WANDB_DATA_DIR', 'WANDB_CONFIG_DIR'):
        Path(env[directory]).mkdir(parents=True, exist_ok=True)
    assert json.loads((EVIDENCE / 'resume-6h200-20261003/resume-tests.json').read_text())['passed']
    def recovery_checkpoint():
        receipt = EVIDENCE / 'train/checkpoints/latest-checkpoint.json'
        if receipt.exists():
            candidate = Path(json.loads(receipt.read_text())['payload'])
            if candidate.is_file() and candidate.stat().st_size > 10_000_000_000:
                return candidate
        local = Path('/dev/shm/laser-imagenet-recovery/last.pt')
        if local.is_file() and local.stat().st_size == 16703442049:
            return local
        raise RuntimeError('No complete recovery checkpoint available')
    checkpoint = recovery_checkpoint()
    base_command = [PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=6',
                    str(Path(__file__).with_name('imagenet_epoch45_lr55_6h200_entry.py')), '--config', str(Path(__file__).resolve().parents[2] / 'configs/stage2/imagenet-rfid421-epoch45-lr55-6h200.yaml')]
    started = time.time()
    for attempt in range(3):
        if STOPPING:
            break
        checkpoint = recovery_checkpoint()
        command = list(base_command)
        command.append('options.resume_checkpoint=' + str(checkpoint))
        if attempt:
            if not checkpoint.is_file():
                raise RuntimeError('Recovery requires this run\'s durable full checkpoint')
            command.append('options.resume=true')
        elif checkpoint.exists():
            import yaml
            if yaml.safe_load((BASE / 'recipe.yaml').read_text())['options'].get('resume') is True:
                command.append('options.resume=true')
            else:
                raise RuntimeError('Fresh launch refuses an existing production checkpoint')
        log = BASE / f'production-{os.getpid()}-attempt-{attempt}.log'
        log_stream = log.open('w')
        process = subprocess.Popen(command, env=env, cwd=BASE / 'runtime/runtime', stdout=log_stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        state = dict(supervisor_pid=os.getpid(), torchrun_pid=process.pid, attempt=attempt,
                     started_unix=started, attempt_started_unix=time.time(), run_url=RUN_URL,
                     command=command, local_log=str(log), persistent_log=str(EVIDENCE / 'training.log'),
                     state='starting', global_batch=2048, world_size=6, accumulation=2, resumed_from_step=30000)
        sent_stop = False
        cursor = 0
        last_sync = 0
        while process.poll() is None:
            if STOPPING and not sent_stop:
                # Let rank zero request a checkpoint at a coordinated optimizer
                # boundary. Signalling torchrun itself gives workers too little
                # time to serialize a full checkpoint.
                children = Path(f'/proc/{process.pid}/task/{process.pid}/children')
                if children.exists():
                    ranks = children.read_text().split()
                    for rank_pid in ranks:
                        rank_env = Path(f'/proc/{rank_pid}/environ')
                        if rank_env.exists() and b'RANK=0' in rank_env.read_bytes().split(b'\0'):
                            os.kill(int(rank_pid), signal.SIGTERM)
                            sent_stop = True
                            break
            with log.open() as stream:
                stream.seek(cursor)
                for line in stream:
                    if line.startswith('{'):
                        try:
                            progress = json.loads(line)
                        except ValueError:
                            continue
                        if progress.get('phase') in ('training', 'fid_progress'):
                            state.update(state='running', progress=progress, progress_unix=time.time())
                cursor = stream.tell()
            durable = BASE / 'production/durable-checkpoint.json'
            if durable.is_file():
                state['durable_checkpoint'] = json.loads(durable.read_text())
            if time.time() - last_sync >= 30:
                state['heartbeat_unix'] = time.time()
                record(BASE / 'production-status.json', state)
                record(EVIDENCE / 'status.json', state)
                temporary = EVIDENCE / 'training.log.tmp'
                temporary.write_bytes(log.read_bytes())
                temporary.replace(EVIDENCE / 'training.log')
                last_sync = time.time()
            time.sleep(5)
        log_stream.close()
        state.update(exit_code=process.returncode, state='stopped' if STOPPING else
                     ('finished' if process.returncode == 0 else 'failed'), heartbeat_unix=time.time())
        record(BASE / 'production-status.json', state)
        record(EVIDENCE / 'status.json', state)
        (EVIDENCE / f'training-{os.getpid()}-attempt-{attempt}.log').write_bytes(log.read_bytes())
        if process.returncode == 0 or STOPPING:
            break
        tail = log.read_text()[-24000:]
        if any(error in tail for error in ('non-finite', 'nonfinite', 'illegal memory access',
                                          'out of memory', 'AssertionError')):
            break
        if not checkpoint.is_file():
            break
        time.sleep(15)


if __name__ == '__main__':
    if '--supervise' in sys.argv:
        supervise()
    else:
        status = BASE / 'production-status.json'
        if status.exists():
            previous = json.loads(status.read_text())
            if (previous.get('state') in ('starting', 'running')
                    and Path(f"/proc/{previous['supervisor_pid']}").exists()):
                print(json.dumps(previous, indent=2))
                raise SystemExit(0)
        with (BASE / 'supervisor.log').open('a') as output:
            process = subprocess.Popen([PYTHON, str(Path(__file__).resolve()), '--supervise'],
                                       stdout=output, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        print(json.dumps(dict(supervisor_pid=process.pid, run_url=RUN_URL)))
