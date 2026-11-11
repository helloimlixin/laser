"""Continue the validated aligned-teacher run through epoch 100."""
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

BASE = Path('/tmp/laser-aligned-beam451-20261105/frozen')
OUT = BASE.parent / 'production-nvme'
RUN = 'imnet-j16384-a4096-beam16-aligned-soft-scratch2-4a100-20261105'
TARGET_EPOCHS = 100
TARGET_STEPS = 626 * TARGET_EPOCHS
RESUME_FLOOR = 1252
CHILD = None


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def status(phase, **extra):
    write(OUT / 'status.json', dict(phase=phase, time=time.time(), supervisor_pid=os.getpid(),
        run_id=RUN, world_size=4, hardware='4 NVIDIA A100-SXM4-80GB',
        atoms=4096, vocabulary=16384, global_batch=2048, target_epoch=TARGET_EPOCHS, target_step=TARGET_STEPS,
        scheduler_horizon=62600, progress=read(OUT / 'train/progress-rank0.json'),
        checkpoint=read(OUT / 'checkpoint.json'),
        ar_targets='beam16 conditional soft posterior', ar_beam_width=16, reconstruction_beam_width=16,
        tokenizer_rfid=read(BASE.parent/'quality/full-results.json')['results']['teacher']['rfid'],
        cloud=read(OUT/'cloud-sync-status.json'), **extra))


def stop(sig, _):
    if CHILD is not None and CHILD.poll() is None:
        os.killpg(CHILD.pid, signal.SIGTERM)
    status('interrupted', signal=sig)
    raise SystemExit(128 + sig)


def environment():
    env = dict(os.environ)
    for key in ['WANDB_SERVICE', '_WANDB_SERVICE']: env.pop(key, None)
    env.update(WANDB_API_KEY=Path('/root/.config/laser/wandb.key').read_text().strip(),
        WANDB_BASE_URL='https://api.forge.coreweave.com', WANDB_MODE='online',
        WANDB_CONSOLE='off', WANDB_DISABLE_GIT='true',
        CUDA_VISIBLE_DEVICES='0,1,2,3', PYTHONUNBUFFERED='1', LASER_STAGE2_OUTPUT=str(OUT),
        TORCH_HOME='/tmp/laser-recon-4096-20261105/torch-cache',
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
        LASER_FUSED_ADAMW='1', LASER_EXACT_BEAM_KERNELS='1', LASER_FAST_BEAM_TOPK='0',  TORCHINDUCTOR_CACHE_DIR='/tmp/laser-speed-aligned-20261105/inductor-cache', TORCHINDUCTOR_COMPILE_THREADS='2', LASER_TRAIN_SDPA='1', LASER_SDPA_MIN_SEQ='16', LASER_COMPILE_LOSS_DYNAMIC='1', LASER_COMPILE_NATIVE_LOSS='1',
        NCCL_NVLS_ENABLE='0', PYTORCH_ALLOC_CONF='expandable_segments:True')
    return env


def command(microbatch, source=None):
    cmd = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
           '--nproc-per-node=4', str(BASE / 'train.py'), '--microbatch', str(microbatch),
           '--epochs', str(TARGET_EPOCHS)]
    if source: cmd += ['--resume', str(source)]
    return cmd


def child(cmd, env, phase, attempt=0):
    global CHILD
    with (OUT / (phase + '.log')).open('ab', buffering=0) as log:
        CHILD = subprocess.Popen(cmd, cwd=BASE, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        started = time.time()
        while CHILD.poll() is None:
            state = phase
            generation = OUT / 'generation-progress.json'
            progress = OUT / 'train/progress-rank0.json'
            if phase == 'training' and generation.exists() and progress.exists():
                g = read(generation)
                if generation.stat().st_mtime > progress.stat().st_mtime and g['generated_on_rank'] < g['samples_on_rank']:
                    state = 'evaluating'
            storage=OUT/'checkpoint-storage-wait.json'
            if storage.exists() and read(storage).get('state')=='WAITING':state='waiting_for_storage'
            status(state, child_pid=CHILD.pid, attempt=attempt)
            activity = [progress, generation, storage, OUT / (phase + '.log')]
            last = max([started] + [p.stat().st_mtime for p in activity if p.exists()])
            if time.time() - last > 1800:
                os.killpg(CHILD.pid, signal.SIGTERM)
                try: CHILD.wait(timeout=30)
                except subprocess.TimeoutExpired: os.killpg(CHILD.pid, signal.SIGKILL)
            try: CHILD.wait(timeout=10)
            except subprocess.TimeoutExpired: pass
    return CHILD.returncode


def verify(folder, microbatch):
    proof = read(folder / 'verification.json')
    assert proof['passed'] and proof['global_step'] == proof['all_adam_steps'] == 5
    assert proof['optimizer_state_count'] == 780 and proof['frozen_tokenizer_unchanged']
    for rank in range(4):
        startup = read(folder / f'startup-rank{rank}.json')
        assert startup['fresh_stage2'] and startup['fresh_optimizer']
        assert startup['global_step'] == startup['scheduler_step_at_start'] == 0
        assert startup['optimizer_state_count_at_start'] == 0 and startup['scheduler_horizon'] == 62600
        assert startup['global_batch'] == 2048 and startup['world_size'] == 4 and startup['microbatch'] == microbatch
        assert startup['gradient_clip_max_norm'] == 1. and startup['grad_scaler_enabled']
        assert startup['training_autocast_dtype'] == 'float16'
        assert startup['soft_labels'] and startup['stochastic_codes']
        targets = read(folder / f'targets-rank{rank}.json')
        assert targets['vocabulary'] == 16384 and targets['soft_targets'] and targets['stochastic_prefixes']
        expected=read(BASE.parent/'quality/calibration.json')['sequence_temperature']
        assert targets['temperature']==expected
        assert targets['prefixes_from_same_teacher_distribution']
        assert not targets['native_greedy_soft_teacher_used'] and not targets['one_hot_beam_targets']
        assert targets['entropy'] > 0
        generation = read(folder / f'generation-interface-rank{rank}.json')
        assert generation['passed'] and generation['native_cached_autoregressive_sampling']
        assert not generation['reconstruction_beam_used'] and not generation['generation_quality_measured']
    return proof


def checkpoint():
    import torch
    for path in sorted((OUT / 'persistent-train/checkpoints').glob('step-*.pt'), reverse=True):
        try:
            payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
            step = int(payload['global_step'])
            assert payload['run_id'] == RUN and len(payload['optimizer']['state']) == 780
            assert payload['recipe']['tokenizer']['vocab_size'] == 16384
            assert all(int(value['step']) == step for value in payload['optimizer']['state'].values())
            del payload
            return path, step
        except Exception as error:
            write(OUT / 'rejected-checkpoints' / f'{path.stem}.json', dict(error=str(error)))
    return None, 0


def main():
    lock = (OUT / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for sig in [signal.SIGTERM, signal.SIGINT]: signal.signal(sig, stop)
    gate=read(BASE.parent/'quality/full-results.json')
    assert gate and gate['passed'] and gate['validation_images']==50000
    assert gate['results']['teacher']['rfid']<4.60
    results = read(OUT / 'execution-benchmarks.json') or []
    for microbatch in [64, 96]:
        if any(row['microbatch'] == microbatch for row in results): continue
        phase = f'benchmark-mb{microbatch}'
        folder = BASE / phase
        code = child(command(microbatch) + ['--smoke-updates', '5'],
                     dict(environment(), LASER_PRIVATE_SMOKE_DIR=str(folder)), phase)
        row = dict(microbatch=microbatch, returncode=code, log=str(OUT / (phase + '.log')))
        if code == 0:
            proof = verify(folder, microbatch)
            progress = read(folder / 'progress-rank0.json')
            row.update(images_per_second=progress['train/images_per_second'],
                       peak_allocated_gib=progress['peak_allocated_gib'], verification=proof,
                       generation_interface=read(folder / 'generation-interface-rank0.json'))
        else:
            tail = (OUT / (phase + '.log')).read_text()[-12000:]
            row.update(out_of_memory='out of memory' in tail.lower(), error_tail=tail[-6000:])
            if not row['out_of_memory']:
                write(OUT / 'failed-benchmark.json', row)
                raise RuntimeError(f'Unexpected benchmark failure: {phase}')
        results.append(row); write(OUT / 'execution-benchmarks.json', results)
        # These are verification weights; production always initializes afresh.
        for path in (OUT / 'smoke/checkpoints').glob('*.pt'): path.unlink()
    selected = max((row for row in results if row['returncode'] == 0), key=lambda row: row['images_per_second'])
    assert read(BASE.parent/'quality/cloud-verification.json')['state']=='COMMITTED'
    microbatch = selected['microbatch']; write(OUT / 'selected-execution.json', selected)
    execution = read(BASE / 'inputs/execution.json')
    execution.update(microbatch=microbatch, accumulation_steps=math.ceil(2048 / (4 * microbatch)))
    write(BASE / 'inputs/execution.json', execution)
    attempt = failures = 0
    while True:
        source, step = checkpoint()
        assert source is not None and step>=RESUME_FLOOR, 'Continuation requires the validated epoch-2 state or a later checkpoint'
        cmd = command(microbatch, source)
        launch = dict(command=cmd, source_checkpoint=str(source) if source else None,
            source_step=step, fresh=False, run_id=RUN, target_epoch=TARGET_EPOCHS, target_step=TARGET_STEPS,
            scheduler_horizon=62600, global_batch=2048, world_size=4, microbatch=microbatch,
            accumulation_steps=math.ceil(2048 / (4 * microbatch)), beam_aligned_conditional_soft_teacher=True,
            source_epoch1_fid=89.12546967078106, source_epoch2_fid=73.73770937849014,
            reconstruction_rfid=gate['results']['canonical']['rfid'],teacher_sampled_rfid=gate['results']['teacher']['rfid'], started_at=time.time(), attempt=attempt,
            url='https://forge.coreweave.com/wandb/helloimlixin-rutgers/laser/runs/' + RUN)
        write(OUT / 'launch.json', launch); write(OUT / 'attempts' / f'{attempt:04d}.json', launch)
        started = time.time(); code = child(cmd, environment(), 'training', attempt)
        completed = read(OUT / 'completed.json')
        if code == 0 and completed and completed['global_step'] == TARGET_STEPS and completed['epochs']==TARGET_EPOCHS:
            comparisons = {}
            for epoch, baseline in [(1, 89.12546967078106), (2, 73.73770937849014)]:
                evaluation = read(OUT / f'evaluations/epoch{epoch:03d}.json')
                hard_baseline={1:96.959082102031,2:77.06398779070582}[epoch]
                comparisons[str(epoch)] = dict(evaluation=evaluation, reference_8192_native_fid=baseline,
                    fid_delta_vs_native8192=evaluation['eval/fid']-baseline,
                    reference_same_book_hard_beam_fid=hard_baseline,
                    fid_delta_vs_same_book_hard_beam=evaluation['eval/fid']-hard_baseline,
                    note='Teacher, clipping, precision, and hardware differ; references are not an isolated teacher ablation')
            write(OUT / 'matched-epoch-comparison.json', comparisons)
            status('completed_training', comparison=comparisons,
                   final_evaluation=read(OUT/f'evaluations/epoch{TARGET_EPOCHS:03d}.json')); return
        progress_path = OUT / 'train/progress-rank0.json'
        progress = read(progress_path) or {}
        moved = progress_path.exists() and progress_path.stat().st_mtime >= started and progress.get('train/global_step', step) > step
        failures = 0 if moved else failures + 1
        write(OUT / 'attempts' / f'{attempt:04d}-exit.json', dict(returncode=code, progress=progress, time=time.time()))
        if failures >= 3: raise RuntimeError('Three consecutive startup failures')
        attempt += 1; status('restarting', returncode=code, attempt=attempt)
        time.sleep(min(60, 15 * max(1, failures)))


if __name__ == '__main__':
    try: main()
    except Exception as error:
        status('failed', error=str(error))
        raise
