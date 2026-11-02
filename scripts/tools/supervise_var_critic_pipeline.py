"""Resume both critic-repair tokenizers, then jointly release class-conditional VAR.

Run in a detached process. Each stage has bounded restart supervision, a single
writer lock, durable full-state checkpoints, and an explicit joint epoch-10 gate.
"""
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

STOP = threading.Event()
CHILDREN = {}


def atomic(path, value):
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}-{threading.get_ident()}')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def stage2_gate(base):
    runs = {}
    for arm in ('laser', 'vq'):
        marker = base / 'stage1' / arm / 'complete.json'
        if not marker.exists():
            return None
        receipt = json.loads(marker.read_text())
        if receipt['progress']['epoch'] != 10 or receipt['progress']['batch'] != 0:
            raise ValueError('Epoch-10 gate received an incomplete tokenizer')
        runs[arm] = receipt
    return dict(ready=True, runs=runs, opened_unix=time.time())


def stop(*_):
    STOP.set()
    for child in list(CHILDREN.values()):
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)


def run_arm(base, source, local, entry, stage, arm, devices, gate=None):
    durable = base / stage / arm
    config = durable / 'launch-config.yaml'
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               f'--nproc_per_node={len(devices.split(","))}', str(entry),
               '--source', str(source), '--config', str(config), '--durable', str(durable)]
    if gate:
        command += ['--gate', str(gate)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices, OMP_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
               TORCH_HOME='/tmp/laser-imagenet-var/assets/torch',
               LASER_VGG16_WEIGHTS='/tmp/laser-imagenet-var/assets/vgg16-397923af.pth',
               NCCL_NVLS_ENABLE='0', WANDB_CONSOLE='off', PYTHONUNBUFFERED='1')
    for attempt in range(1, 7):
        if STOP.is_set():
            return False
        if (durable / 'complete.json').exists():
            return True
        # If the durable mirror is newer after a process failure, restore it
        # atomically. Otherwise the latest local atomic checkpoint is retained.
        name = 'tokenizer-last.pt' if stage == 'stage1' else 'prior-last.pt'
        remote, target = durable / name, local / stage / arm / name
        if remote.exists() and (not target.exists() or remote.stat().st_mtime > target.stat().st_mtime):
            import shutil
            temporary = target.with_suffix('.recovering')
            shutil.copyfile(remote, temporary)
            temporary.replace(target)
        with (durable / 'training.log').open('a', buffering=1) as log:
            child = subprocess.Popen(command, cwd=source, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            CHILDREN[(stage, arm)] = child
            receipt = dict(phase='running', stage=stage, arm=arm, devices=devices,
                           supervisor_pid=os.getpid(), torchrun_pid=child.pid,
                           attempt=attempt, command=command, started_unix=time.time())
            atomic(durable / 'supervisor-status.json', receipt)
            while child.poll() is None:
                for p in (local / stage / arm).glob('*'):
                    if p.is_file() and p.suffix in ('.json', '.yaml', '.png'):
                        dest = durable / p.name
                        if not dest.exists() or p.stat().st_mtime > dest.stat().st_mtime:
                            import shutil
                            temporary = dest.with_name(dest.name + '.copying')
                            shutil.copyfile(p, temporary)
                            temporary.replace(dest)
                STOP.wait(10)
            complete = child.returncode == 0 and (durable / 'complete.json').exists()
            receipt.update(phase='complete' if complete else 'retrying',
                           returncode=child.returncode, updated_unix=time.time())
            atomic(durable / 'supervisor-status.json', receipt)
            if complete:
                return True
        if STOP.is_set():
            return False
        if attempt < 6:
            STOP.wait(30)
    receipt['phase'] = 'failed'
    atomic(durable / 'supervisor-status.json', receipt)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--local', required=True)
    parser.add_argument('--entry', required=True)
    args = parser.parse_args()
    base, source, local, entry = [Path(p).resolve() for p in
                                 (args.base, args.source, args.local, args.entry)]
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with (base / '.supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic(base / 'supervisor-status.json', dict(
            phase='stage1', supervisor_pid=os.getpid(), target_tokenizer_epochs=10,
            stage2_joint_epoch10_gate=True, time=time.time()))
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(run_arm, base, source, local, entry, 'stage1', arm, devices)
                    for arm, devices in [('laser', '0,1'), ('vq', '2,3')]]
            complete = all([job.result() for job in jobs])
        if not complete or STOP.is_set():
            atomic(base / 'supervisor-status.json', dict(phase='stage1_incomplete', time=time.time()))
            return
        gate = stage2_gate(base)
        if gate is None:
            raise RuntimeError('Both tokenizer processes exited without opening the gate')
        gate_path = base / 'stage2-gate.json'
        atomic(gate_path, gate)
        import shutil
        for arm in ('laser', 'vq'):
            src = local / 'stage1' / arm / 'tokenizer-last.pt'
            dest = local / 'stage2' / arm / 'tokenizer-last.pt'
            if not dest.exists():
                shutil.copyfile(src, dest)
            atomic(base / 'stage2' / arm / 'recovery.json', dict(
                artifact=f'epoch10-local:{gate["runs"][arm]["checkpoint_sha256"]}',
                progress=gate['runs'][arm]['progress']))
        atomic(base / 'supervisor-status.json', dict(phase='stage2', gate=str(gate_path), time=time.time()))
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(run_arm, base, source, local, entry, 'stage2', arm, devices, gate_path)
                    for arm, devices in [('laser', '0,1,2,3'), ('vq', '4,5,6,7')]]
            complete = all([job.result() for job in jobs])
        atomic(base / 'supervisor-status.json', dict(
            phase='complete' if complete else 'stage2_incomplete', time=time.time()))


if __name__ == '__main__':
    main()
