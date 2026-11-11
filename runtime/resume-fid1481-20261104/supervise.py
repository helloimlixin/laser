"""Benchmark four-H100 execution, validate full-state resume, then train to100.

Credentials are read from a private file outside the repository. Training and
checkpoint writes use local disk; verified W&B commits retain durable payloads.
"""
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time

BASE = Path('/tmp/laser-fid1481-resume/frozen')
RUN = 'imnet-j16384-a4096-fid1481-resume100-4h100-20261104'
OUT = Path('/workspace/Projects/laser/outputs') / RUN
LOCAL = BASE / 'run-output'
SOURCE_STEP = 40754
CHILD = None


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.pending')
    tmp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    tmp.replace(path)


def status(phase, **extra):
    progress = BASE / 'live-training/train/progress-rank0.json'
    latest = json.loads(progress.read_text()) if progress.exists() else None
    write(OUT / 'status.json', dict(phase=phase, run_id=RUN,
        supervisor_pid=os.getpid(), time=time.time(), source_step=SOURCE_STEP,
        source_fid=14.812355891611958, target_epoch=100, target_step=62600,
        latest_progress=latest, **extra))
    for name in ['checkpoint.json', 'matched-baseline.json', 'wandb.json',
                 'generation-progress.json', 'completed.json', 'best-inception.json']:
        path = LOCAL / name
        if path.exists():
            write(OUT / name, json.loads(path.read_text()))


def stop(signum, _):
    if CHILD is not None and CHILD.poll() is None:
        os.killpg(CHILD.pid, signal.SIGINT)
    status('interrupted', signal=signum)
    raise SystemExit(128 + signum)


def environment():
    env = dict(os.environ)
    for key in ['WANDB_SERVICE', '_WANDB_SERVICE']:
        env.pop(key, None)
    env.update(WANDB_API_KEY=Path('/root/.config/laser/wandb.key').read_text().strip(),
        WANDB_BASE_URL='https://api.forge.coreweave.com', WANDB_MODE='online',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        PYTHONUNBUFFERED='1', LASER_MAX_RECOVERY_LOSS='8.25',
        LASER_LR_AMPLITUDE_FACTOR='0.15', LASER_GENERATION_BATCH='64',
        LASER_EVAL_EVERY_EPOCHS='1', LASER_PRESERVE_INCEPTION_MIN='91.6950912475586',
        LASER_QUALITY_EARLY_STOP='0', LASER_STAGE2_OUTPUT=str(LOCAL),
        LASER_COMPILE_NATIVE_LOSS='1', LASER_FUSED_ADAMW='1',
        LASER_DATA_WORKERS='8', LASER_INHERIT_SOURCE_EVALUATION='1',
        PYTORCH_ALLOC_CONF='expandable_segments:True',
        TORCHINDUCTOR_COMPILE_THREADS='4')
    return env


def command(microbatch):
    return [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
        '--nproc-per-node=4', str(BASE/'train.py'), '--microbatch', str(microbatch),
        '--epochs', '100', '--resume', str(BASE/'inputs/resume-checkpoint.pt')]


def child(cmd, env, phase):
    global CHILD
    log = BASE / (phase + '.log')
    with log.open('ab', buffering=0) as handle:
        CHILD = subprocess.Popen(cmd, cwd=BASE, env=env, stdin=subprocess.DEVNULL,
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        while CHILD.poll() is None:
            status(phase, child_pid=CHILD.pid, log=str(log))
            try:
                CHILD.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
    return CHILD.returncode, log


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for signum in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(signum, stop)
    assert not (OUT/'launch.json').exists(), 'Inspect existing production checkpoint before relaunch'
    while not ((BASE/'data/train-ready.json').exists() or (BASE/'data/benchmark-ready.json').exists()):
        status('waiting_for_indexed_training_data')
        time.sleep(10)
    env = environment()
    results = []
    for microbatch, sdpa in [(128, False), (256, False), (128, True), (256, True)]:
        phase = f'benchmark-mb{microbatch}-sdpa{int(sdpa)}'
        folder = BASE / phase
        trial_env = dict(env, LASER_TRAIN_SDPA=str(int(sdpa)),
            LASER_BENCHMARK_ONLY='1', LASER_PRIVATE_SMOKE_DIR=str(folder))
        code, log = child(command(microbatch) + ['--smoke-updates', str(SOURCE_STEP+6)],
            trial_env, phase)
        row = dict(microbatch=microbatch, sdpa=sdpa, returncode=code, log=str(log))
        if code == 0:
            proof = json.loads((folder/'verification.json').read_text())
            assert proof['passed'] and proof['all_adam_steps'] == SOURCE_STEP+6
            assert proof['optimizer_state_count'] == 780 and proof['frozen_tokenizer_unchanged']
            startup = proof['initialization']
            assert startup['global_step'] == startup['restored_optimizer_age'] == SOURCE_STEP
            assert not startup['fresh_stage2'] and not startup['fresh_optimizer']
            assert startup['lr_recovery']['applied_ratio'] == 1
            assert startup['lr_recovery']['scheduler_step'] == SOURCE_STEP
            assert startup['lr_recovery']['scheduler_horizon'] == 62600
            metrics = [json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
            speed = [m['train/images_per_second'] for m in metrics][-3:]
            row.update(median_images_per_second=statistics.median(speed),
                measured_images_per_second=speed,
                peak_allocated_gib=json.loads((folder/'progress-rank0.json').read_text())['peak_allocated_gib'],
                verification=proof)
        else:
            tail = log.read_text()[-12000:]
            row['out_of_memory'] = 'out of memory' in tail.lower() or 'OutOfMemoryError' in tail
            row['error_tail'] = tail[-3000:]
            if not row['out_of_memory']:
                write(OUT/'failed-benchmark.json', row)
                raise RuntimeError(f'Benchmark failed unexpectedly: {log}')
        results.append(row)
        write(OUT/'execution-benchmarks.json', results)
    passed = [r for r in results if r['returncode'] == 0]
    assert passed
    while not (BASE/'data/train-ready.json').exists():
        status('waiting_for_verified_training_data')
        time.sleep(10)
    ready = json.loads((BASE/'data/train-ready.json').read_text())
    assert ready['ready'] and ready['training_images'] == 1281167
    selected = max(passed, key=lambda r: r['median_images_per_second'])
    mb, sdpa = selected['microbatch'], selected['sdpa']
    write(OUT/'selected-execution.json', selected)
    execution_path = BASE/'inputs/execution.json'
    execution = json.loads(execution_path.read_text())
    execution.update(microbatch=mb, accumulation_steps=2048//(4*mb), world_size=4,
        training_attention='SDPA' if sdpa else 'native bmm', loader_workers_per_rank=8,
        benchmark_median_images_per_second=selected['median_images_per_second'],
        peak_allocated_gib=selected['peak_allocated_gib'])
    write(execution_path, execution)
    shutil.copyfile(OUT/'execution-benchmarks.json', BASE/'execution-benchmark.json')
    env['LASER_TRAIN_SDPA'] = str(int(sdpa))
    # Save and verify a real three-update smoke checkpoint; production reloads
    # the unchanged source checkpoint, so benchmark updates never enter training.
    code, log = child(command(mb)+['--smoke-updates', str(SOURCE_STEP+3)], env, 'preflight')
    if code:
        raise RuntimeError(f'Preflight failed: {log}')
    proof = json.loads((LOCAL/'smoke/verification.json').read_text())
    assert proof['passed'] and proof['all_adam_steps'] == SOURCE_STEP+3
    write(OUT/'preflight.json', proof)
    shutil.copyfile(OUT/'preflight.json', BASE/'full-size-smoke-verification.json')
    for path in (LOCAL/'smoke/checkpoints').glob('*.pt'):
        path.unlink()
    launch = dict(run_id=RUN, command=command(mb)+['--new-wandb-run'],
        source_checkpoint='helloimlixin-rutgers/laser/imnet-j16384-a4096-is92-lr015-repair-cont100-2b300-20261103-checkpoint:v1',
        source_step=SOURCE_STEP, source_fid=14.812355891611958,
        source_inception_score=92.1950912475586, world_size=4,
        microbatch=mb, accumulation_steps=2048//(4*mb), global_batch=2048,
        training_attention=execution['training_attention'],
        model_optimizer_scheduler_preserved=True, lr_amplitude_unchanged=True,
        initial_lr=2.0368127469368654e-5, target_epoch=100, target_step=62600,
        initial_evaluation_inherited=True, global_sample_cursor_preserved=True,
        topology_change_is_not_bitwise_replay=True, quality_early_stop=False,
        log=str(BASE/'training.log'), local_output=str(LOCAL), started_at=time.time(),
        url='https://forge.coreweave.com/wandb/helloimlixin-rutgers/laser/runs/'+RUN)
    write(OUT/'launch.json', launch)
    code, log = child(launch['command'], env, 'training')
    if code:
        raise RuntimeError(f'Training failed: {log}')
    complete = json.loads((LOCAL/'completed.json').read_text())
    assert complete['epochs'] == 100 and complete['global_step'] == 62600
    status('completed', completion=complete)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error=str(error))
        raise
