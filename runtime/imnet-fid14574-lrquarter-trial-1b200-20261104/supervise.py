"""Rewind best epoch73, test quarter LR for64 updates, and gate continuation."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

BASE = Path('/tmp/laser-fid14574-lrquarter/frozen')
RUN = 'imnet-j16384-a4096-fid14574-lrquarter-trial64-1b200-20261104'
OUT = Path('/workspace/Projects/laser/outputs') / RUN
LOCAL = BASE / 'run-output'
SOURCE_STEP = 45698
PILOT_STEP = SOURCE_STEP + 64
CHILD = None


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def status(phase, **extra):
    progress = BASE / 'live-training/train/progress-rank0.json'
    latest = json.loads(progress.read_text()) if progress.exists() else None
    evaluation = LOCAL / 'generation-progress.json'
    generation = json.loads(evaluation.read_text()) if evaluation.exists() else None
    decision = LOCAL / 'pilot-decision.json'
    pilot = json.loads(decision.read_text()) if decision.exists() else None
    write(OUT / 'status.json', dict(
        phase=phase, run_id=RUN, supervisor_pid=os.getpid(), time=time.time(),
        source_step=SOURCE_STEP, source_epoch=73, source_fid=14.573915136162327,
        lr_multiplier_vs_source=0.25, pilot_step=PILOT_STEP,
        target_epoch=100, target_step=62600, latest_progress=latest,
        generation_progress=generation, pilot_decision=pilot, **extra))
    for name in ['checkpoint.json', 'matched-baseline.json', 'wandb.json',
                 'generation-progress.json', 'completed.json', 'best-inception.json',
                 'source-artifact.json', 'workspace.json', 'pilot-decision.json',
                 'quality-stopped.json', 'quality-stop.json']:
        path = LOCAL / name
        if path.exists():
            write(OUT / name, json.loads(path.read_text()))


def stop(signum, _frame):
    if CHILD is not None and CHILD.poll() is None:
        os.killpg(CHILD.pid, signal.SIGINT)
    status('interrupted', signal=signum)
    raise SystemExit(128 + signum)


def environment():
    env = dict(os.environ)
    for key in ['WANDB_SERVICE', '_WANDB_SERVICE']:
        env.pop(key, None)
    env.update(
        WANDB_API_KEY=Path('/root/.config/laser/wandb.key').read_text().strip(),
        WANDB_BASE_URL='https://api.forge.coreweave.com', WANDB_MODE='online',
        WANDB_DISABLE_GIT='true', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1',
        LASER_MAX_RECOVERY_LOSS='8.25', LASER_LR_AMPLITUDE_FACTOR='0.0375',
        LASER_GENERATION_BATCH='64', LASER_EVAL_EVERY_EPOCHS='1',
        LASER_PRESERVE_INCEPTION_MIN='93.50654602050781',
        LASER_QUALITY_EARLY_STOP='1', LASER_STAGE2_OUTPUT=str(LOCAL),
        LASER_COMPILE_NATIVE_LOSS='1', LASER_FUSED_ADAMW='1', LASER_TRAIN_SDPA='0',
        LASER_DATA_WORKERS='12', LASER_INHERIT_SOURCE_EVALUATION='1',
        PYTORCH_ALLOC_CONF='expandable_segments:True', TORCHINDUCTOR_COMPILE_THREADS='4')
    return env


def command():
    return [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
            '--nproc-per-node=1', str(BASE / 'train.py'), '--microbatch', '256',
            '--epochs', '100', '--resume', str(BASE / 'inputs/resume-checkpoint.pt')]


def child(cmd, env, phase):
    global CHILD
    log = BASE / (phase + '.log')
    with log.open('ab', buffering=0) as handle:
        CHILD = subprocess.Popen(cmd, cwd=BASE, env=env, stdin=subprocess.DEVNULL,
                                 stdout=handle, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        while CHILD.poll() is None:
            status(phase, child_pid=CHILD.pid, log=str(log))
            try:
                CHILD.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
    return CHILD.returncode, log


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for signum in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(signum, stop)
    assert not (OUT / 'launch.json').exists(), 'Inspect existing trial before relaunch'
    ready = json.loads((BASE / 'data/train-ready.json').read_text())
    assert ready['ready'] and ready['training_images'] == 1281167
    env = environment()
    smoke_env = dict(env, LASER_BENCHMARK_ONLY='1')
    code, log = child(command() + ['--smoke-updates', str(SOURCE_STEP + 3)],
                      smoke_env, 'preflight')
    if code:
        raise RuntimeError(f'Preflight failed: {log}')
    proof = json.loads((LOCAL / 'smoke/verification.json').read_text())
    assert proof['passed'] and proof['global_step'] == proof['all_adam_steps'] == SOURCE_STEP + 3
    assert proof['optimizer_state_count'] == 780 and proof['frozen_tokenizer_unchanged']
    initial = proof['initialization']
    assert initial['global_step'] == initial['restored_optimizer_age'] == SOURCE_STEP
    assert initial['global_batch'] == 2048 and initial['world_size'] == 1
    assert initial['microbatch'] == 256 and initial['accumulation_steps'] == 8
    assert not initial['fresh_optimizer'] and not initial['fresh_stage2']
    lr = initial['lr_recovery']
    assert lr['previous_factor'] == 0.15 and lr['desired_factor'] == 0.0375
    assert lr['applied_ratio'] == 0.25 and lr['scheduler_step'] == SOURCE_STEP
    assert lr['scheduler_horizon'] == 62600
    assert abs(initial['restored_lr'] - 3.175201262590684e-6) < 1e-17
    assert initial['resume_memory']['source_mmap_released']
    write(OUT / 'preflight.json', proof)
    shutil.copyfile(OUT / 'preflight.json', BASE / 'full-size-smoke-verification.json')
    launch = dict(
        run_id=RUN, command=command() + ['--new-wandb-run'], source_step=SOURCE_STEP,
        source_epoch=73, source_fid=14.573915136162327, source_is=94.00654602050781,
        initial_lr=3.175201262590684e-6, lr_multiplier=0.25,
        model_and_adam_restored=True, scheduler_step_preserved=True,
        target_epoch=100, target_step=62600, pilot_updates=64, pilot_step=PILOT_STEP,
        pilot_minimum_fid_improvement=0.05, pilot_maximum_is_drop=0.5,
        matched_baseline_reused=True, global_batch=2048,
        microbatch=256, accumulation_steps=8, world_size=1,
        old_regressing_run_stopped=True, existing_imagenet_data_reused=True,
        log=str(BASE / 'trial_training.log'), local_output=str(LOCAL), started_at=time.time(),
        url='https://forge.coreweave.com/wandb/helloimlixin-rutgers/laser/runs/' + RUN)
    write(OUT / 'launch.json', launch)
    code, log = child(launch['command'], env, 'trial_training')
    if code:
        raise RuntimeError(f'Training failed: {log}')
    decision_path = LOCAL / 'pilot-decision.json'
    if decision_path.exists():
        decision = json.loads(decision_path.read_text())
        if not decision['accepted']:
            status('pilot_rejected', decision=decision)
            return
    if (LOCAL / 'quality-stopped.json').exists():
        status('stopped_quality_regression')
        return
    complete = json.loads((LOCAL / 'completed.json').read_text())
    assert complete['epochs'] == 100 and complete['global_step'] == 62600
    status('completed', completion=complete)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error=str(error))
        raise
