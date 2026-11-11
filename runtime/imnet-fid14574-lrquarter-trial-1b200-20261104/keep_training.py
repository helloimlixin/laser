"""Supervise the explicitly resumed run and stop this Pod after durable completion."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

BASE = Path('/tmp/laser-fid14574-lrquarter/frozen')
LOCAL = BASE / 'run-output'
OUT = Path('/workspace/Projects/laser/outputs/imnet-j16384-a4096-fid14574-lrquarter-trial64-1b200-20261104')
POD_ID = os.environ['RUNPOD_POD_ID']
STOPPING = False
PID = 11256


def write(path, value):
    tmp = path.with_suffix('.pending')
    tmp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def alive(pid):
    path = Path(f'/proc/{pid}/stat')
    return path.exists() and path.read_text().split(')', 1)[1].split()[0] not in {'Z', 'X'}


def status(phase, **extra):
    progress = read(BASE / 'live-training/train/progress-rank0.json')
    write(OUT / 'status.json', dict(
        phase=phase, time=time.time(), supervisor_pid=os.getpid(), child_pid=PID,
        user_requested_continuation=True, automatic_quality_stop=False,
        target_epoch=100, target_step=62600, latest_progress=progress,
        automatic_pod_stop_after_completion=True, **extra))
    for name in ['checkpoint.json', 'generation-progress.json', 'completed.json', 'wandb.json']:
        value = read(LOCAL / name)
        if value is not None:
            write(OUT / name, value)


def environment():
    env = dict(os.environ)
    for key in ['WANDB_SERVICE', '_WANDB_SERVICE']:
        env.pop(key, None)
    env.update(
        WANDB_API_KEY=Path('/root/.config/laser/wandb.key').read_text().strip(),
        WANDB_BASE_URL='https://api.forge.coreweave.com', WANDB_MODE='online', WANDB_DISABLE_GIT='true',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1',
        LASER_MAX_RECOVERY_LOSS='8.25', LASER_LR_AMPLITUDE_FACTOR='0.0375',
        LASER_GENERATION_BATCH='64', LASER_EVAL_EVERY_EPOCHS='1',
        LASER_PRESERVE_INCEPTION_MIN='93.50654602050781', LASER_QUALITY_EARLY_STOP='0',
        LASER_STAGE2_OUTPUT=str(LOCAL), LASER_COMPILE_NATIVE_LOSS='1', LASER_FUSED_ADAMW='1',
        LASER_TRAIN_SDPA='0', LASER_DATA_WORKERS='12', LASER_INHERIT_SOURCE_EVALUATION='1',
        PYTORCH_ALLOC_CONF='expandable_segments:True', TORCHINDUCTOR_COMPILE_THREADS='4')
    return env


def stop_current_pod(reason):
    # Never stop an instance while another GPU workload is using it.
    import requests
    active = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    if active:
        status('billing_stop_deferred_other_gpu_work', active_gpu_pids=active, reason=reason)
        return
    status('stopping_paid_instance', reason=reason, pod_id=POD_ID)
    query = 'mutation { podStop(input: {podId: ' + json.dumps(POD_ID) + '}) { id desiredStatus } }'
    for attempt in range(3):
        try:
            response = requests.post('https://api.runpod.io/graphql',
                params={'api_key': os.environ['RUNPOD_API_KEY']}, json={'query': query}, timeout=20)
            if response.ok:
                result = response.json()
                value = (result.get('data') or {}).get('podStop')
                if value and value.get('id') == POD_ID and value.get('desiredStatus') != 'RUNNING':
                    write(OUT / 'pod-stop-receipt.json', dict(result=value, reason=reason, time=time.time()))
                    return
            status('retrying_paid_instance_stop', attempt=attempt + 1, http_status=response.status_code)
        except requests.RequestException as error:
            status('retrying_paid_instance_stop', attempt=attempt + 1, error_type=type(error).__name__)
        time.sleep(5)
    status('billing_stop_failed', reason=reason, pod_id=POD_ID)


def stop(signum, _frame):
    global STOPPING
    STOPPING = True
    if alive(PID):
        os.kill(PID, signal.SIGINT)
    status('interrupted_by_user', signal=signum)


def main():
    global PID
    lock = (OUT / 'continuation-supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, stop)
    retries = 0
    retry_step = None
    while not STOPPING:
        if alive(PID):
            status('training', retry_count=retries)
            time.sleep(10)
            continue
        progress = read(BASE / 'live-training/train/progress-rank0.json') or {}
        worker = progress.get('pid')
        if worker and alive(worker):
            cmdline = Path(f'/proc/{worker}/cmdline').read_bytes()
            if str(BASE / 'train.py').encode() in cmdline.split(b'\0'):
                PID = worker
                continue
        completed = read(LOCAL / 'completed.json')
        receipt = read(LOCAL / 'checkpoint.json')
        if (completed and completed['global_step'] == 62600 and completed['epochs'] == 100
                and receipt and receipt.get('state') == 'COMMITTED' and receipt['global_step'] == 62600):
            status('completed', completion=completed)
            stop_current_pod('Epoch100 finished and its full checkpoint is committed in W&B')
            return
        try:
            if not receipt or receipt.get('state') != 'COMMITTED':
                raise RuntimeError('No committed recovery checkpoint')
            checkpoint = Path(receipt['path'])
            if not checkpoint.is_file() or checkpoint.stat().st_size != receipt['bytes']:
                raise RuntimeError('Recovery checkpoint does not match its receipt')
            current_step = int(receipt['global_step'])
            retries = retries + 1 if current_step == retry_step else 1
            retry_step = current_step
            if retries > 3:
                raise RuntimeError('Three attempts failed at the same recovery checkpoint')
            cmd = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
                   '--nproc-per-node=1', str(BASE / 'train.py'), '--microbatch', '256',
                   '--epochs', '100', '--resume', str(checkpoint)]
            log = BASE / f'recovery-{current_step}-{retries}.log'
            with log.open('ab', buffering=0) as handle:
                child = subprocess.Popen(cmd, cwd=BASE, env=environment(), stdin=subprocess.DEVNULL,
                    stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            PID = child.pid
            status('restarting_from_checkpoint', recovery_step=current_step,
                   retry_count=retries, log=str(log))
        except Exception as error:
            status('unrecoverable_training_failure', error=str(error))
            stop_current_pod('Training cannot safely restart; stop idle GPU compute charges')
            return


if __name__ == '__main__':
    main()
