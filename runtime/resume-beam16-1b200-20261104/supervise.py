"""Verify and resume the same W&B beam16 run on the available single B200."""
import fcntl
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

BASE = Path('/tmp/laser-beam16-resume/frozen')
RUN = 'imnet-j16384-a4096-beam16-rfid451-scratch100-2b300-20261104'
OUT = Path('/workspace/Projects/laser/outputs') / RUN
SOURCE_STEP = 6260
CHILD = None


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def status(phase, **extra):
    write(OUT / 'status.json', dict(phase=phase, time=time.time(),
        supervisor_pid=os.getpid(), run_id=RUN, source_step=SOURCE_STEP,
        source_epoch=10, target_epoch=100, target_step=62600,
        world_size=1, hardware='NVIDIA B200', global_batch=2048,
        progress=read(OUT/'train/progress-rank0.json'), **extra))


def interrupt(signum, _):
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
        PYTHONUNBUFFERED='1', LASER_STAGE2_OUTPUT=str(OUT),
        LASER_MAX_RECOVERY_LOSS='20.0', LASER_LR_AMPLITUDE_FACTOR='1.0',
        LASER_GENERATION_BATCH='64', LASER_EVAL_EVERY_EPOCHS='2',
        LASER_PRESERVE_INCEPTION_MIN='0.0', LASER_QUALITY_EARLY_STOP='0',
        LASER_COMPILE_NATIVE_LOSS='0', LASER_FUSED_ADAMW='1', LASER_TRAIN_SDPA='0',
        PYTORCH_ALLOC_CONF='expandable_segments:True')
    return env


def command(microbatch):
    return [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
        '--nproc-per-node=1', str(BASE/'train.py'), '--microbatch', str(microbatch),
        '--epochs', '100', '--resume', str(BASE/'inputs/resume-checkpoint.pt')]


def child(cmd, env, phase):
    global CHILD
    log = OUT / (phase + '.log')
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


def check_proof(folder, updates):
    proof = read(folder/'verification.json')
    assert proof['passed'] and proof['all_adam_steps'] == SOURCE_STEP + updates
    assert proof['global_step'] == SOURCE_STEP + updates
    assert proof['optimizer_state_count'] == 780 and proof['frozen_tokenizer_unchanged']
    initial = proof['initialization']
    assert initial['global_step'] == initial['restored_optimizer_age'] == SOURCE_STEP
    assert not initial['fresh_stage2'] and not initial['fresh_optimizer']
    assert initial['optimizer_state_count_at_start'] == 780
    assert initial['scheduler_step_at_start'] == SOURCE_STEP and initial['scheduler_horizon'] == 62600
    assert initial['global_batch'] == 2048 and initial['world_size'] == 1
    assert initial['lr_recovery']['applied_ratio'] == 1
    assert initial['lr_amplitude_factor'] == 1 and initial['stage2_checkpoint_loaded']
    assert not initial['soft_labels'] and not initial['stochastic_codes']
    targets = read(folder/'targets-rank0.json')
    assert targets['training_targets_equal_reconstruction_beam_codes']
    assert targets['native_and_physical_code_embeddings_match']
    assert targets['teacher_encoding_beam_width'] == 16
    return proof


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, interrupt)
    assert not (OUT/'launch.json').exists(), 'Inspect existing production checkpoints before relaunch'
    while not (BASE/'data/train-ready.json').exists() or not (BASE/'inputs/resume-checkpoint.pt').exists():
        status('preparing_assets')
        time.sleep(10)
    ready = read(BASE/'data/train-ready.json')
    assert ready['ready'] and ready['training_images'] == 1281167
    env = environment()
    results = []
    for microbatch in [128, 256]:
        phase = f'benchmark-mb{microbatch}'
        folder = BASE / phase
        trial_env = dict(env, LASER_PRIVATE_SMOKE_DIR=str(folder))
        code, log = child(command(microbatch) + ['--smoke-updates', str(SOURCE_STEP+3)], trial_env, phase)
        row = dict(microbatch=microbatch, returncode=code, log=str(log))
        if code == 0:
            proof = check_proof(folder, 3)
            progress = read(folder/'progress-rank0.json')
            row.update(images_per_second=progress['train/images_per_second'],
                peak_allocated_gib=progress['peak_allocated_gib'], verification=proof)
            for path in (OUT/'smoke/checkpoints').glob('*.pt'):
                path.unlink()
        else:
            tail = log.read_text()[-12000:]
            row['out_of_memory'] = 'out of memory' in tail.lower() or 'OutOfMemoryError' in tail
            row['error_tail'] = tail[-4000:]
            if not row['out_of_memory']:
                write(OUT/'failed-benchmark.json', row)
                raise RuntimeError(f'Unexpected preflight failure: {log}')
        results.append(row)
        write(OUT/'execution-benchmarks.json', results)
    passed = [row for row in results if row['returncode'] == 0]
    assert passed, 'Neither batch size fits; inspect diagnostics'
    selected = max(passed, key=lambda row: row['images_per_second'])
    mb = selected['microbatch']
    write(OUT/'selected-execution.json', selected)
    write(BASE/'execution-benchmark.json', results)
    write(BASE/'continuation-verification.json', selected['verification'])
    execution = read(BASE/'inputs/execution.json')
    execution.update(microbatch=mb, accumulation_steps=2048//mb,
        benchmark_images_per_second=selected['images_per_second'],
        peak_allocated_gib=selected['peak_allocated_gib'])
    write(BASE/'inputs/execution.json', execution)
    launch = dict(command=command(mb), run_id=RUN, start_step=SOURCE_STEP,
        source_epoch=10, source_artifact='helloimlixin-rutgers/laser/'+RUN+'-checkpoint:v31',
        prior_logged_step=6600, target_epoch=100, target_step=62600,
        hardware='1 NVIDIA B200', microbatch=mb, accumulation_steps=2048//mb,
        global_batch=2048, world_size=1, training_attention='native bmm',
        model_optimizer_scheduler_restored=True, lr_amplitude_unchanged=True,
        scheduler_horizon=62600, new_wandb_run=False,
        topology_change_is_not_bitwise_replay=True,
        log=str(OUT/'training.log'), started_at=time.time(),
        url='https://forge.coreweave.com/wandb/helloimlixin-rutgers/laser/runs/'+RUN)
    write(OUT/'launch.json', launch)
    code, log = child(launch['command'], env, 'training')
    if code:
        raise RuntimeError(f'Training failed: {log}')
    complete = read(OUT/'completed.json')
    assert complete['epochs'] == 100 and complete['global_step'] == 62600
    status('completed', completion=complete)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error=str(error))
        raise
