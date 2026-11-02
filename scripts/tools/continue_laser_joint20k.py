#!/usr/bin/env python3
"""Continue the verified two-epoch joint20k pilot to 100 total epochs."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


ROOT = Path('/workspace/Projects/laser')
BASE = Path('/tmp/laser-rq-joint20k-imagenet-20261017')
RUN = 'imagenet-laser-joint20k-scratch2-4h100-20261017'
OUT = ROOT / 'outputs' / RUN
CONTINUE = OUT / 'continuation-to100'
SOURCE_STEP = 1252
STOP_STEP = 62600


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(dict(value, time=time.time()), indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def prepare():
    import torch
    torch.set_num_threads(4)
    assert not (CONTINUE / 'prepared.json').exists(), 'Continuation already prepared'
    assert json.loads((OUT / 'launch-status.json').read_text())['phase'] == 'completed'
    assert json.loads((OUT / 'completed.json').read_text())['global_step'] == SOURCE_STEP
    assert not subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'
    ], text=True).strip(), 'GPUs occupied'
    manifest = json.loads((OUT / 'reproducibility/code-manifest.json').read_text())['files']
    assert all(sha(BASE / name) == expected for name, expected in manifest.items())
    source = (OUT / 'checkpoints/last.pt').resolve()
    receipts = [json.loads(p.read_text()) for p in
                (OUT / 'persistent-train/checkpoints/.checkpoint-data').glob('*.json')]
    matches = [record for record in receipts if Path(record['persistent']) == source]
    assert len(matches) == 1
    receipt = matches[0]
    assert receipt['global_step'] == SOURCE_STEP
    for key in ('local', 'persistent'):
        path = Path(receipt[key])
        assert path.stat().st_size == receipt['bytes']
        assert sha(path) == receipt['sha256'], key
    saved = torch.load(receipt['local'], map_location='cpu', mmap=True, weights_only=False)
    assert saved['run_id'] == RUN
    assert saved['global_step'] == SOURCE_STEP and saved['next_epoch'] == 2
    assert saved['next_microbatch'] == 0 and len(saved['rng_state_by_rank']) == 4
    assert saved['joint_codebook_sha256'] == '974fdaf42f041a02e93c7321b65d57900088ab7a2339ada6a5ffcc062ee9ac54'
    assert saved['recipe'] == json.loads((OUT / 'prepared.json').read_text())['recipe']
    assert len(saved['optimizer']['state']) == 780
    assert all(int(state['step']) == SOURCE_STEP
               and torch.isfinite(state['exp_avg']).all()
               and torch.isfinite(state['exp_avg_sq']).all()
               for state in saved['optimizer']['state'].values())
    after = saved['scheduler']['after']
    assert after['T_max'] == STOP_STEP and after['last_epoch'] == SOURCE_STEP
    initial_lr = saved['optimizer']['param_groups'][0]['lr']
    assert math.isclose(initial_lr, .0004995066821070667, rel_tol=1e-12)
    baseline = json.loads((OUT / 'evaluations/epoch002.json').read_text())
    assert baseline['generated_images'] == 50000 and baseline['real_images'] == 1281167
    del saved
    preserved = OUT / 'epoch002-preserved.pt'
    local = BASE / 'epoch002-preserved.pt'
    assert not preserved.exists() and not local.exists()
    preserved.symlink_to(source.relative_to(OUT))
    os.link(receipt['local'], local)
    CONTINUE.mkdir(parents=True, exist_ok=True)
    for name in ('completed.json', 'launch-status.json', 'comparison.json'):
        shutil.copyfile(OUT / name, CONTINUE / ('epoch002-' + name))
    for rank in range(4):
        shutil.copyfile(OUT / f'train/startup-rank{rank}.json',
                        CONTINUE / f'epoch002-startup-rank{rank}.json')
    shutil.copyfile(Path(__file__), CONTINUE / Path(__file__).name)
    plan = dict(source_run=RUN, source_epoch=2, source_step=SOURCE_STEP,
                checkpoint=str(preserved), local_checkpoint=str(local),
                checkpoint_sha256=receipt['sha256'], checkpoint_bytes=receipt['bytes'],
                local_and_persistent_sha256_verified=True,
                all_adam_steps=SOURCE_STEP, finite_optimizer_moments=True,
                rank_rng_states=4, resume_lr=initial_lr,
                total_epochs=100, stop_step=STOP_STEP, added_epochs=98,
                added_updates=STOP_STEP-SOURCE_STEP, cosine_total_updates=STOP_STEP,
                learning_rate_schedule_restarted=False, recipe_unchanged=True,
                world_size=4, microbatch=64, accumulation_steps=8, global_batch=2048,
                generated_images_per_fid=50000, reference_images=1281167,
                fid_every_epochs=2, preview_every_updates=200,
                code_manifest_sha256=sha(OUT / 'reproducibility/code-manifest.json'))
    write(CONTINUE / 'prepared.json', plan)
    return plan


def publish(plan):
    import wandb
    run = wandb.Api(timeout=45).run(f'helloimlixin-rutgers/laser/{RUN}')
    assert run.config['codebook_sha256'] == '974fdaf42f041a02e93c7321b65d57900088ab7a2339ada6a5ffcc062ee9ac54'
    run.config.update(training_epochs=100, effective_training_epochs=100,
                      continuation_to100=plan)
    run.summary.update({'training/continued_from_step': SOURCE_STEP,
                        'training/continued_from_epoch': 2,
                        'training/stop_after_epochs': 100,
                        'training/stop_after_updates': STOP_STEP,
                        'training/schedule_restarted': False})
    run.update()
    verified = wandb.Api(timeout=45).run(f'helloimlixin-rutgers/laser/{RUN}')
    assert verified.config['training_epochs'] == 100
    assert verified.summary['training/stop_after_epochs'] == 100
    write(CONTINUE / 'online-verification.json', dict(passed=True,
          training_epochs=100, stop_step=STOP_STEP, same_wandb_run=True))


def main():
    CONTINUE.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write(CONTINUE / 'controller.json', dict(phase='verifying', pid=os.getpid()))
    plan = prepare()
    environment = dict(os.environ, **json.loads((BASE / 'environment.json').read_text()))
    environment['WANDB_API_KEY'] = Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
    os.environ['WANDB_API_KEY'] = environment['WANDB_API_KEY']
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               '--nproc-per-node=4', str(BASE / 'train.py'), '--microbatch', '64',
               '--epochs', '100', '--resume', plan['checkpoint']]
    (OUT / 'completed.json').unlink()
    launched_at = time.time()
    child = None
    cancelled = False

    def stop(*_):
        nonlocal cancelled
        cancelled = True
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with (CONTINUE / 'training.log').open('ab', buffering=0) as log:
        child = subprocess.Popen(command, cwd=BASE, env=environment, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        status = dict(phase='training', pid=os.getpid(), child_pid=child.pid,
                      run_id=RUN, resuming=True, source_step=SOURCE_STEP,
                      stop_epoch=100, stop_step=STOP_STEP, world_size=4,
                      global_batch=2048, microbatch=64, accumulation_steps=8,
                      launched_at=launched_at)
        write(CONTINUE / 'controller.json', status)
        write(OUT / 'launch-status.json', status)
        startup_verified = online_verified = False
        retry_at = 0.
        while child.poll() is None:
            if not startup_verified:
                paths = [OUT / f'train/startup-rank{rank}.json' for rank in range(4)]
                if all(p.exists() and p.stat().st_mtime >= launched_at for p in paths):
                    rows = [json.loads(p.read_text()) for p in paths]
                    assert all(row['global_step'] == SOURCE_STEP
                               and not row['fresh_stage2'] and not row['fresh_optimizer']
                               and row['training_epochs'] == 100
                               and row['lr_schedule_epochs'] == 100
                               and math.isclose(row['initial_optimizer_lr'], plan['resume_lr'], rel_tol=1e-12)
                               and row['world_size'] == 4 and row['global_batch'] == 2048
                               and row['scalar_events_per_site'] == 4
                               for row in rows)
                    write(CONTINUE / 'resume-verification.json', dict(passed=True,
                          global_step=SOURCE_STEP, ranks=4, optimizer_restored=True,
                          lr=plan['resume_lr'], schedule_restarted=False))
                    startup_verified = True
            if startup_verified and not online_verified and time.monotonic() >= retry_at:
                try:
                    publish(plan)
                    online_verified = True
                except Exception as error:
                    write(CONTINUE / 'online-retry.json', dict(error_type=type(error).__name__,
                          error=str(error)))
                    retry_at = time.monotonic() + 60
            time.sleep(3)
        result = child.wait()
    if result or cancelled:
        status.update(phase='cancelled' if cancelled else 'failed', returncode=result)
        write(CONTINUE / 'controller.json', status)
        write(OUT / 'launch-status.json', status)
        raise RuntimeError(f'Training returned {result}')
    completed = json.loads((OUT / 'completed.json').read_text())
    assert completed['epochs'] == 100 and completed['global_step'] == STOP_STEP
    status.update(phase='completed', completed_epochs=100, global_step=STOP_STEP, returncode=0)
    write(CONTINUE / 'controller.json', status)
    write(OUT / 'launch-status.json', status)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(CONTINUE / 'error.json', dict(error_type=type(error).__name__, error=str(error)))
        raise
