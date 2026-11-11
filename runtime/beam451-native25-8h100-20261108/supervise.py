"""Calibrate, verify a full-state restore, then continue one run to epoch100."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import time
from omegaconf import OmegaConf
from environment import BASE, OUT, HERE, PYTHON, RUN, SOURCE_RUN, environment

CHILD = None
STOP = False


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def status(phase, **extra):
    progress = BASE / 'live-training/train/progress-rank0.json'
    write(OUT / 'status.json', dict(
        phase=phase, supervisor_pid=os.getpid(), child_pid=CHILD.pid if CHILD else None,
        run_id=RUN, source_run=SOURCE_RUN, source_epoch=12, epochs=100, world_size=8,
        global_batch=2048, progress=read(progress) if progress.exists() else None,
        time=time.time(), **extra))


def stop(signum, frame):
    global STOP
    STOP = True
    if CHILD and CHILD.poll() is None:
        try:
            os.killpg(CHILD.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    status('stopped_by_signal')
    raise SystemExit(128 + signum)


def distributed(script, args):
    return [PYTHON, '-u', '-m', 'torch.distributed.run', '--standalone',
            '--nproc-per-node=8', str(script), *args]


def launch(command, phase, extra=None):
    global CHILD
    env = environment()
    env.update(extra or {})
    with (OUT / (phase + '.log')).open('ab', buffering=0) as log:
        CHILD = subprocess.Popen(command, env=env, cwd=BASE, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    while CHILD.poll() is None:
        status(phase)
        try:
            CHILD.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    return CHILD.returncode


def benchmark(microbatch):
    folder = BASE / 'benchmarks' / f'mb{microbatch}'
    folder.mkdir(parents=True, exist_ok=True)
    if not (folder / 'verification.json').exists():
        code = launch(distributed(BASE / 'train.py', [
            '--epochs', '100', '--microbatch', str(microbatch),
            '--resume', str(BASE / 'inputs/source-step-0007512.pt'),
            '--smoke-updates', '7517']), f'benchmark-mb{microbatch}',
            dict(LASER_PRIVATE_SMOKE_DIR=str(folder)))
        if code:
            tail = (OUT / f'benchmark-mb{microbatch}.log').read_text()[-16000:]
            if 'out of memory' in tail.lower():
                return dict(microbatch=microbatch, passed=False, reason='GPU memory limit')
            raise RuntimeError(f'Continuation benchmark failed: {tail[-3500:]}')
    proof = read(folder / 'verification.json')
    assert proof['passed'] and proof['frozen_tokenizer_unchanged']
    assert proof['all_adam_steps'] == 7517 and proof['optimizer_state_count'] == 780
    speeds, peaks = [], []
    for rank in range(8):
        initial = read(folder / f'startup-rank{rank}.json')
        assert not initial['fresh_stage2'] and not initial['fresh_optimizer']
        assert initial['restored_optimizer_age'] == 7512 and initial['scheduler_step_at_start'] == 7512
        assert initial['scheduler_horizon'] == 62600 and initial['world_size'] == 8
        assert initial['training_autocast_dtype'] == 'bfloat16' and not initial['grad_scaler_enabled']
        assert initial['gradient_clip_max_norm'] == .1 and initial['native_weight'] == .25
        assert initial['stage1_preservation_guard_passed']
        target = read(folder / f'targets-rank{rank}.json')
        assert target['exact_prefix_conditionals'] and target['entropy'] > .2
        rows = [json.loads(line) for line in (folder / f'throughput-rank{rank}.jsonl').read_text().splitlines()]
        speeds.append(statistics.median(r['images_per_second'] for r in rows if r['step'] >= 7515))
        peaks.append(read(folder / f'progress-rank{rank}.json')['peak_allocated_gib'])
    return dict(passed=True, microbatch=microbatch, images_per_second=min(speeds),
                peak_allocated_gib=max(peaks), full_state_restore=True,
                optimizer_state_count=780, restored_step=7512, verified_step=7517)


def main():
    lock = (BASE / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, stop)
    assert read(BASE / 'data/ready.json')['training_images'] == 1281167
    # These independent full-validation checks used the exact frozen beam codec.
    for name in ['prior-full-results.json', 'prior-replicate-results.json']:
        report = read(OUT / 'quality' / name)
        assert report['passed'] and report['results']['0.25']['images'] == 50000
        assert report['results']['0.25']['rfid'] < 4.73
    if not (OUT / 'teacher-calibration.json').exists():
        code = launch(distributed(BASE / 'runtime/calibrate.py', []), 'teacher-calibration')
        if code:
            raise RuntimeError('Training-image teacher calibration failed')
    assert read(OUT / 'teacher-calibration.json')['passed']
    while not (BASE / 'inputs/source-step-0007512.pt').exists():
        status('staging_source_checkpoint')
        time.sleep(5)
    assert (BASE / 'inputs/source-step-0007512.pt').stat().st_size == 16651333771
    rows = [benchmark(128), benchmark(64)]
    selected = max((r for r in rows if r['passed']), key=lambda r: r['images_per_second'])
    microbatch = selected['microbatch']
    write(OUT / 'throughput-benchmarks.json', dict(candidates=rows, selected=selected))
    recipe = OmegaConf.load(BASE / 'inputs/recipe.yaml')
    recipe.microbatch = microbatch
    OmegaConf.save(recipe, BASE / 'inputs/recipe.yaml')
    execution = read(BASE / 'inputs/execution.json')
    execution.update(microbatch=microbatch, accumulation_steps=2048 // (8 * microbatch),
                     measured_images_per_second=selected['images_per_second'])
    write(BASE / 'inputs/execution.json', execution)
    write(OUT / 'preflight-verification.json', dict(
        passed=True, full_model_optimizer_scheduler_restore=True, source_step=7512,
        source_epoch=12, source_run=SOURCE_RUN, hardware=selected,
        teacher=read(OUT / 'teacher-calibration.json')))
    repro = OUT / 'reproducibility'
    repro.mkdir(exist_ok=True)
    for path in (BASE / 'inputs').iterdir():
        if path.is_file() and path.stat().st_size < 50 * 1024**2:
            shutil.copyfile(path, repro / path.name)
    for path in HERE.glob('*.py'):
        shutil.copyfile(path, repro / path.name)
    shutil.copyfile(BASE / 'inputs/recipe.yaml', HERE.parent.parent / 'configs/stage2' / (RUN + '.yaml'))
    attempts = {}
    while not STOP:
        step = 7512
        checkpoint = BASE / 'inputs/source-step-0007512.pt'
        args = ['--epochs', '100', '--microbatch', str(microbatch)]
        if (OUT / 'checkpoint.json').exists():
            receipt = read(OUT / 'checkpoint.json')
            checkpoint = Path(receipt['path'])
            if not checkpoint.exists():
                checkpoint = Path(receipt['persistent_path'])
            assert checkpoint.stat().st_size == receipt['bytes']
            step = receipt['global_step']
        else:
            args.append('--new-wandb-run')
        args += ['--resume', str(checkpoint)]
        attempts[step] = attempts.get(step, 0) + 1
        if attempts[step] > 3:
            raise RuntimeError(f'Repeated training failure from complete checkpoint {step}')
        command = distributed(BASE / 'train.py', args)
        write(OUT / 'launch.json', dict(command=command, world_size=8, microbatch=microbatch,
                                       global_batch=2048, epochs=100, resume_step=step, time=time.time()))
        code = launch(command, 'training')
        if code == 0:
            assert read(OUT / 'completed.json')['epochs'] == 100
            status('completed')
            return
        status('recovering', exit_code=code, resume_step=step)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error=str(error))
        raise
