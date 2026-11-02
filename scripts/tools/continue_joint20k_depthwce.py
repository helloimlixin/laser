#!/usr/bin/env python3
"""Resume the saved depth-weighted ImageNet trial to 100 total epochs."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile

ROOT = Path('/workspace/Projects/laser')
RUN = 'imagenet-laser-joint20k-depth5-depthwce-epoch1to3-8h100-20261018'
OUT = ROOT / 'outputs' / RUN
BASE = Path('/tmp/laser-joint20k-depthwce-100ep-4b200-20261018')
RECORD = OUT / 'continuation-100ep-4b200-20261018'
KEY = Path('/root/.config/laser/wandb_api_key')
CHILD = None


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stop(signum, _frame):
    if CHILD is not None and CHILD.poll() is None:
        os.killpg(CHILD.pid, signal.SIGINT)
    raise SystemExit(128 + signum)


def status(phase, **fields):
    value = dict(phase=phase, supervisor_pid=os.getpid(), run_id=RUN,
        epochs=100, world_size=4, gpu='NVIDIA B200', time=time.time(), **fields)
    write(RECORD / 'status.json', value)
    if phase in ('training', 'completed', 'failed'):
        write(OUT / 'status.json', value)


def run_child(command, env, log, phase, microbatch):
    global CHILD
    with log.open('ab', buffering=0) as stream:
        CHILD = subprocess.Popen(command, cwd=BASE, env=env,
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True)
        while True:
            progress = {}
            path = BASE / 'live-training/train/progress-rank0.json'
            if phase == 'training' and path.exists():
                progress = json.loads(path.read_text())
            status(phase, child_pid=CHILD.pid, command=command, log=str(log),
                microbatch=microbatch, accumulation_steps=2048 // (4 * microbatch),
                global_batch=2048, source_epoch=2, source_step=1252,
                latest_progress=progress)
            try:
                return CHILD.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass


def verify_smoke(folder, microbatch):
    proof = json.loads((folder / 'verification.json').read_text())
    assert proof['passed'] and proof['global_step'] == 1253
    assert proof['all_adam_steps'] == 1253
    assert proof['optimizer_state_count'] == 780 and proof['frozen_tokenizer_unchanged']
    for rank in range(4):
        startup = json.loads((folder / f'startup-rank{rank}.json').read_text())
        assert startup['world_size'] == 4 and startup['microbatch'] == microbatch
        assert startup['accumulation_steps'] * 4 * microbatch == 2048
        assert startup['global_step'] == 1252
        assert not startup['fresh_optimizer'] and not startup['fresh_stage2']
        assert all(math.isclose(a, b, rel_tol=1.e-12) for a, b in zip(
            startup['depth_loss_weights'], [1.5, 1.25, 1., .75, .5]))
        pairing = json.loads((folder / f'loss-pairing-rank{rank}.json').read_text())
        assert pairing['passed'] and pairing['cuda'] and pairing['training_rng_unchanged']
        preview = json.loads((folder / f'preview-verification-rank{rank}.json').read_text())
        assert preview['passed'] and preview['training_rng_unchanged']
    return proof


def publish_execution(microbatch, receipt):
    import wandb
    os.environ['WANDB_API_KEY'] = KEY.read_text().strip()
    run = wandb.Api(timeout=60).run(f'helloimlixin-rutgers/laser/{RUN}')
    migration = dict(source_checkpoint_sha256=receipt['sha256'], source_epoch=2,
        source_step=1252, previous_logged_step=int(run.summary.get('train/global_step', 0)),
        total_epochs=100, target_step=62600, world_size=4, gpu='NVIDIA B200',
        microbatch=microbatch, accumulation_steps=2048 // (4 * microbatch),
        global_batch=2048, weights_optimizer_scheduler_scaler_unchanged=True,
        checkpoint_byte_identical=True, retained_rank_rng_streams=[0, 1, 2, 3],
        exact_trajectory_preserved=False, replay_epoch=3,
        reason='User requested 100 epochs scaled to available hardware')
    run.config.update(training_epochs=100, world_size=4, microbatch=microbatch,
        accumulation_steps=migration['accumulation_steps'], hardware_continuation=migration)
    run.name = 'LASER joint20k | weighted CE | continue100 | 4 B200'
    run.update()
    run.summary.update({'training/stop_after_epochs':100,
        'training/hardware_continuation':migration})
    write(RECORD / 'plan.json', migration)


def archive_stale_aliases():
    expected = {'train':BASE / 'live-training/train',
                'checkpoints':OUT / 'persistent-train/checkpoints'}
    for name in ('train', 'checkpoints'):
        path = OUT / name
        if path.is_symlink():
            if path.resolve() != expected[name].resolve():
                path.rename(RECORD / f'previous-{name}-link-{time.time_ns()}')
        elif path.is_file():
            if path.stat().st_size:
                raise RuntimeError(f'Unexpected nonempty materialized alias: {path}')
            path.rename(RECORD / f'previous-materialized-{name}-{time.time_ns()}')


def main():
    RECORD.mkdir(parents=True, exist_ok=True)
    with (BASE / 'continuation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not (BASE / 'data/val-index.npz').exists() or not (BASE / 'source-receipt.json').exists():
            status('preparing_data_and_checkpoint')
            # The copy process writes this only after a full byte/hash check.
            time.sleep(10)
        receipt = json.loads((BASE / 'source-receipt.json').read_text())
        assert receipt['epoch'] == 2 and receipt['global_step'] == 1252
        assert (BASE / 'source-epoch002.pt').stat().st_size == receipt['bytes']
        occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
            '--format=csv,noheader,nounits'], text=True).strip()
        if occupied:
            raise RuntimeError(f'GPUs are occupied: {occupied}')
        import torch
        torch.set_num_threads(4)
        saved = torch.load(BASE / 'source-epoch002.pt', map_location='cpu', mmap=True, weights_only=False)
        assert saved['run_id'] == RUN and saved['global_step'] == 1252
        assert saved['next_epoch'] == 2 and saved['next_microbatch'] == 0
        assert len(saved['rng_state_by_rank']) == 8
        assert all(int(v['step']) == 1252 for v in saved['optimizer']['state'].values())
        write(RECORD / 'checkpoint-verification.json', dict(passed=True,
            source_checkpoint_sha256=receipt['sha256'], source_step=1252,
            optimizer_state_count=len(saved['optimizer']['state']),
            all_adam_steps=1252, checkpoint_byte_identical=True,
            retained_rank_rng_streams=[0, 1, 2, 3], time=time.time()))
        del saved
        env = dict(os.environ, **json.loads((BASE / 'environment.json').read_text()),
            WANDB_API_KEY=KEY.read_text().strip())
        distributed = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
            '--nproc-per-node=4', str(BASE / 'train.py'),
            '--resume', str(BASE / 'source-epoch002.pt'), '--epochs', '100']
        for microbatch in (256, 128, 64):
            folder = BASE / f'private-preflight-micro{microbatch}'
            env['LASER_PRIVATE_SMOKE_DIR'] = str(folder)
            log = RECORD / f'preflight-micro{microbatch}.log'
            result = 0 if (folder / 'verification.json').exists() else run_child(
                [*distributed, '--microbatch', str(microbatch), '--smoke-updates', '1'],
                env, log, 'preflight', microbatch)
            if result == 0:
                proof = verify_smoke(folder, microbatch)
                write(RECORD / 'preflight.json', dict(passed=True, microbatch=microbatch,
                    world_size=4, global_batch=2048, smoke=proof,
                    preflight_weights_discarded=True, time=time.time()))
                break
            if 'out of memory' not in log.read_text().lower() or microbatch == 64:
                raise RuntimeError(f'Preflight failed with exit {result}: {log}')
        while not (BASE / 'data/ready.json').exists():
            status('waiting_for_training_data', microbatch=microbatch,
                preflight_passed=True)
            time.sleep(10)
        assert json.loads((BASE / 'data/ready.json').read_text())['ready']
        publish_execution(microbatch, receipt)
        archive_stale_aliases()
        # Save the runtime used for this launch beside the original immutable snapshot.
        with zipfile.ZipFile(RECORD / 'runtime.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for directory in ('source', 'runtime', 'upstream'):
                for path in (BASE / directory).rglob('*.py'):
                    if '__pycache__' not in path.parts:
                        archive.write(path, path.relative_to(BASE))
            for name in ('train.py', 'environment.json', 'prepare_data.py'):
                archive.write(BASE / name, name)
        env.pop('LASER_PRIVATE_SMOKE_DIR')
        command = [*distributed, '--microbatch', str(microbatch)]
        write(ROOT / 'outputs/imagenet-laser-joint20k-active.json', dict(
            run_id=RUN, url=f'https://wandb.ai/helloimlixin-rutgers/laser/runs/{RUN}',
            base=str(BASE), recipe='Depth-weighted CE continuation to 100 total epochs',
            depth_loss_weights=[1.5, 1.25, 1., .75, .5], world_size=4,
            hardware='4 NVIDIA B200', updated=time.time()))
        result = run_child(command, env, RECORD / 'training.log', 'training', microbatch)
        if result:
            raise RuntimeError(f'Training exited {result}; inspect {RECORD / "training.log"}')
        completed = json.loads((OUT / 'completed.json').read_text())
        assert completed['epochs'] == 100 and completed['global_step'] == 62600
        status('completed', completion=completed)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        main()
    except BaseException as error:
        status('failed', error_type=type(error).__name__, error=str(error))
        raise
