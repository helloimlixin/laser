"""Run the verified epoch-65 IS92 checkpoint through epoch100 on two B300s."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

BASE = Path('/tmp/imnet-is92-resume/frozen')
RUN = 'imnet-j16384-a4096-is92-ep65-cont100-2b300-20261103'
OUT = Path('/workspace/Projects/laser/outputs') / RUN
CHILD = None


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def status(phase, **extra):
    progress = BASE / 'live-training/train/progress-rank0.json'
    write(OUT / 'status.json', dict(
        phase=phase, run_id=RUN, supervisor_pid=os.getpid(), time=time.time(),
        target_epoch=100, target_step=62600, source_epoch=65, source_step=40690,
        latest_progress=json.loads(progress.read_text()) if progress.exists() else None,
        **extra))


def stop(signum, _):
    if CHILD is not None and CHILD.poll() is None:
        os.killpg(CHILD.pid, signal.SIGINT)
    status('interrupted', signal=signum)
    raise SystemExit(128 + signum)


def child(command, env, phase):
    global CHILD
    with (OUT / (phase + '.log')).open('ab', buffering=0) as log:
        CHILD = subprocess.Popen(command, cwd=BASE, env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
        while CHILD.poll() is None:
            status(phase, child_pid=CHILD.pid, command=command)
            try:
                CHILD.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
    if CHILD.returncode:
        raise RuntimeError(f'{phase} exited {CHILD.returncode}; see {OUT / (phase + ".log")}')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop)
    assert not (OUT / 'launch.json').exists(), 'Production was already launched; inspect its checkpoint before restarting'
    env = dict(os.environ)
    for key in ('WANDB_SERVICE', '_WANDB_SERVICE'):
        env.pop(key, None)
    env.update(
        WANDB_API_KEY=Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip(),
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        PYTHONUNBUFFERED='1', LASER_MAX_RECOVERY_LOSS='8.25',
        LASER_LR_AMPLITUDE_FACTOR='0.01875', LASER_GENERATION_BATCH='64',
        LASER_EVAL_EVERY_EPOCHS='1', LASER_PRESERVE_INCEPTION_MIN='92.04315185546875',
        LASER_QUALITY_EARLY_STOP='0', LASER_STAGE2_OUTPUT=str(OUT),
        LASER_COMPILE_NATIVE_LOSS='1', LASER_FUSED_ADAMW='1', LASER_TRAIN_SDPA='0',
        PYTORCH_ALLOC_CONF='expandable_segments:True')
    while not (BASE / 'data/train-ready.json').exists():
        status('waiting_for_verified_training_data')
        time.sleep(10)
    ready = json.loads((BASE / 'data/train-ready.json').read_text())
    assert ready['ready'] and ready['training_images'] == 1281167
    command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
        '--nproc-per-node=2', str(BASE / 'train.py'), '--microbatch', '256',
        '--epochs', '100', '--resume', str(BASE / 'inputs/resume-checkpoint.pt')]
    proof_path = OUT / 'preflight.json'
    if not proof_path.exists():
        child(command + ['--smoke-updates', '40693'], env, 'preflight')
        proof = json.loads((OUT / 'smoke/verification.json').read_text())
        assert proof['passed'] and proof['global_step'] == proof['all_adam_steps'] == 40693
        assert proof['optimizer_state_count'] == 780 and proof['frozen_tokenizer_unchanged']
        startup = proof['initialization']
        assert startup['global_step'] == startup['restored_optimizer_age'] == 40690
        assert startup['global_batch'] == 2048 and startup['world_size'] == 2
        assert startup['training_epochs'] == startup['lr_schedule_epochs'] == 100
        assert startup['lr_recovery']['applied_ratio'] == 1
        assert startup['lr_recovery']['scheduler_step'] == 40690
        assert startup['lr_recovery']['scheduler_horizon'] == 62600
        assert not startup['fresh_optimizer'] and not startup['fresh_stage2']
        write(proof_path, proof)
        # The smoke process exits; production loads the original epoch65 state again.
        (OUT / 'smoke/checkpoints/step-0040693.pt').unlink(missing_ok=True)
    write(OUT / 'launch.json', dict(
        command=command + ['--new-wandb-run'], output=str(OUT), run_id=RUN,
        source_checkpoint='helloimlixin-rutgers/laser/imnet-j16384-a4096-bestis65-lr001875-fast-fidguard-4h100-20261103-checkpoint:v0',
        source_epoch=65, source_step=40690, source_is=92.04315185546875,
        source_fid=14.990640506382817, target_epoch=100, target_step=62600,
        quality_early_stop=False, model_optimizer_and_cosine_restored=True,
        initial_lr=2.5594195324708132e-6, global_batch=2048,
        world_size=2, microbatch=256, accumulation_steps=4,
        rng_policy='Restore saved rank0 and rank1 RNG at epoch boundary; new two-rank trajectory',
        url='https://forge.coreweave.com/wandb/helloimlixin-rutgers/laser/runs/' + RUN,
        log=str(OUT / 'training.log'), started_at=time.time()))
    child(command + ['--new-wandb-run'], env, 'training')
    complete = json.loads((OUT / 'completed.json').read_text())
    assert complete['epochs'] == 100 and complete['global_step'] == 62600
    assert not (OUT / 'quality-stopped.json').exists()
    status('completed', completion=complete)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error=str(error))
        raise
