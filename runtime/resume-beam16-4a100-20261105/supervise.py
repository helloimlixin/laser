"""Benchmark and continue the existing beam16 run on all four local A100s."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

BASE = Path('/tmp/laser-beam16-a100-20261105/frozen')
OUT = Path('/workspace/Projects/laser/outputs/imnet-beam16-resume-4a100-20261105')
RUN = 'imnet-j16384-a4096-beam16-rfid451-scratch100-2b300-20261104'
SOURCE_STEP = 11000
CHILD = None


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.pending')
    tmp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    tmp.replace(path)


def status(phase, **extra):
    write(OUT/'status.json', dict(phase=phase, time=time.time(),
        supervisor_pid=os.getpid(), run_id=RUN, source_step=SOURCE_STEP,
        target_epoch=100, target_step=62600, world_size=4,
        hardware='4 NVIDIA A100-SXM4-80GB', global_batch=2048,
        progress=read(OUT/'train/progress-rank0.json'), **extra))


def interrupt(signum, _):
    if CHILD is not None and CHILD.poll() is None:
        os.killpg(CHILD.pid, signal.SIGINT)
    status('interrupted', signal=signum)
    raise SystemExit(128 + signum)


def environment(sdpa):
    env = dict(os.environ)
    for key in ['WANDB_SERVICE', '_WANDB_SERVICE']:
        env.pop(key, None)
    env.update(WANDB_API_KEY=Path('/root/.config/laser/wandb.key').read_text().strip(),
        WANDB_BASE_URL='https://api.forge.coreweave.com', WANDB_MODE='online',
        CUDA_VISIBLE_DEVICES='0,1,2,3',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        PYTHONUNBUFFERED='1', LASER_STAGE2_OUTPUT=str(OUT),
        LASER_MAX_RECOVERY_LOSS='20.0', LASER_LR_AMPLITUDE_FACTOR='1.0',
        LASER_GENERATION_BATCH='64', LASER_EVAL_EVERY_EPOCHS='2',
        LASER_PRESERVE_INCEPTION_MIN='0.0', LASER_QUALITY_EARLY_STOP='0',
        LASER_COMPILE_NATIVE_LOSS='0', LASER_FUSED_ADAMW='1',
        LASER_TRAIN_SDPA=str(int(sdpa)),
        PYTORCH_ALLOC_CONF='expandable_segments:True',
        NCCL_NVLS_ENABLE='0')
    return env


def command(microbatch):
    return [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
        '--nproc-per-node=4', str(BASE/'train.py'), '--microbatch', str(microbatch),
        '--epochs', '100', '--resume', str(BASE/'inputs/resume-checkpoint.pt')]


def child(cmd, env, phase):
    global CHILD
    log = OUT/(phase+'.log')
    with log.open('ab', buffering=0) as handle:
        CHILD = subprocess.Popen(cmd, cwd=BASE, env=env, stdin=subprocess.DEVNULL,
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        while CHILD.poll() is None:
            status(phase, child_pid=CHILD.pid, log=str(log))
            try:
                CHILD.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
    return CHILD.returncode, log


def verify(folder, microbatch):
    proof = read(folder/'verification.json')
    assert proof['passed'] and proof['all_adam_steps'] == SOURCE_STEP+3
    assert proof['global_step'] == SOURCE_STEP+3
    assert proof['optimizer_state_count'] == 780 and proof['frozen_tokenizer_unchanged']
    for rank in range(4):
        initial = read(folder/f'startup-rank{rank}.json')
        assert initial['global_step'] == initial['restored_optimizer_age'] == SOURCE_STEP
        assert not initial['fresh_stage2'] and not initial['fresh_optimizer']
        assert initial['optimizer_state_count_at_start'] == 780
        assert initial['scheduler_step_at_start'] == SOURCE_STEP
        assert initial['scheduler_horizon'] == 62600
        assert initial['global_batch'] == 2048 and initial['world_size'] == 4
        assert initial['microbatch'] == microbatch
        assert initial['lr_recovery']['applied_ratio'] == 1
        assert not initial['soft_labels'] and not initial['stochastic_codes']
        targets = read(folder/f'targets-rank{rank}.json')
        assert targets['training_targets_equal_reconstruction_beam_codes']
        assert targets['native_and_physical_code_embeddings_match']
        assert targets['teacher_encoding_beam_width'] == 16
        assert read(folder/f'replica-verification-rank{rank}.json')['passed']
        cursor = read(folder/f'cursor-conversion-rank{rank}.json')
        assert cursor['completed_updates_in_epoch'] == SOURCE_STEP % 626
        assert cursor['global_sample_cursor_preserved']
    return proof


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, interrupt)
    assert not (OUT/'launch.json').exists(), 'Inspect production checkpoints before relaunch'
    while not (BASE/'data/train-ready.json').exists():
        status('preparing_data')
        time.sleep(10)
    ready = read(BASE/'data/train-ready.json')
    assert ready['ready'] and ready['training_images'] == 1281167
    assert read(BASE.parent/'kernel-verification.json')['passed']
    # Trial payloads are disposable; write them to NVMe rather than the network mount.
    smoke_checkpoints = BASE/'smoke-checkpoints'
    smoke_checkpoints.mkdir(exist_ok=True)
    (OUT/'smoke').mkdir(exist_ok=True)
    if not (OUT/'smoke/checkpoints').is_symlink():
        (OUT/'smoke/checkpoints').symlink_to(smoke_checkpoints, target_is_directory=True)
    results = read(OUT/'execution-benchmarks.json') or []
    for mb, sdpa in [(64, True), (128, True), (64, False), (128, False)]:
        if any(r['microbatch'] == mb and r['sdpa'] == sdpa and r['returncode'] == 0
               for r in results):
            continue
        phase = f'benchmark-mb{mb}-'+('sdpa' if sdpa else 'native')
        folder = BASE/phase
        env = dict(environment(sdpa), LASER_PRIVATE_SMOKE_DIR=str(folder))
        code, log = child(command(mb)+['--smoke-updates', str(SOURCE_STEP+3)], env, phase)
        row = dict(microbatch=mb, sdpa=sdpa, returncode=code, log=str(log))
        if code == 0:
            proof = verify(folder, mb)
            progress = [read(folder/f'progress-rank{rank}.json') for rank in range(4)]
            row.update(images_per_second=progress[0]['train/images_per_second'],
                peak_allocated_gib=max(p['peak_allocated_gib'] for p in progress),
                verification=proof)
        else:
            tail = log.read_text()[-16000:]
            row['out_of_memory'] = 'out of memory' in tail.lower() or 'OutOfMemoryError' in tail
            row['error_tail'] = tail[-6000:]
            if not row['out_of_memory']:
                write(OUT/'failed-benchmark.json', row)
                raise RuntimeError(f'Unexpected preflight failure: {log}')
        results.append(row)
        write(OUT/'execution-benchmarks.json', results)
        for path in (OUT/'smoke/checkpoints').glob('*.pt'):
            path.unlink()
    passed = [r for r in results if r['returncode'] == 0]
    assert passed, 'No benchmark configuration fits'
    selected = max(passed, key=lambda r: r['images_per_second'])
    mb, sdpa = selected['microbatch'], selected['sdpa']
    write(OUT/'selected-execution.json', selected)
    write(BASE/'execution-benchmark.json', results)
    write(BASE/'continuation-verification.json', selected['verification'])
    execution = read(BASE/'inputs/execution.json')
    execution.update(fresh_stage2=False, fresh_optimizer=False, initial_global_step=SOURCE_STEP,
        initial_epoch=17, source_epoch=17, source_step=SOURCE_STEP, world_size=4,
        microbatch=mb, accumulation_steps=2048//(4*mb), global_batch=2048,
        hardware='4 NVIDIA A100-SXM4-80GB', resume_world_size=1,
        training_attention='SDPA' if sdpa else 'native bmm',
        global_sample_cursor_preserved=True, topology_change_is_not_bitwise_replay=True,
        benchmark_images_per_second=selected['images_per_second'],
        peak_allocated_gib=selected['peak_allocated_gib'])
    write(BASE/'inputs/execution.json', execution)
    launch = dict(command=command(mb), run_id=RUN, start_step=SOURCE_STEP,
        source_artifact=read(BASE.parent/'latest-artifact.json')['name'],
        target_epoch=100, target_step=62600, world_size=4,
        hardware='4 NVIDIA A100-SXM4-80GB', microbatch=mb,
        accumulation_steps=2048//(4*mb), global_batch=2048,
        training_attention=execution['training_attention'],
        model_optimizer_scheduler_restored=True, lr_amplitude_unchanged=True,
        scheduler_horizon=62600, new_wandb_run=False,
        topology_change_is_not_bitwise_replay=True,
        log=str(OUT/'training.log'), started_at=time.time(),
        url='https://forge.coreweave.com/wandb/helloimlixin-rutgers/laser/runs/'+RUN)
    write(OUT/'launch.json', launch)
    code, log = child(launch['command'], environment(sdpa), 'training')
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
