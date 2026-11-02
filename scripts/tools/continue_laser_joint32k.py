#!/usr/bin/env python3
"""Test one sampler, then extend the intact compact-token training to epoch six."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path('/workspace/Projects/laser')
BASE = Path('/tmp/laser-rq-native-joint32k-imagenet-20261016')
TRAIN = ROOT/'outputs/imagenet-laser-native-joint32k-scratch2-4h100-20261016'
OUT = ROOT/'outputs/imagenet-laser-joint32k-epoch2-topk512-fid50k-20261016'
CONTINUE = TRAIN/'continuation-to6'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    import torch
    torch.set_num_threads(4)
    BASE.mkdir(parents=True, exist_ok=True)
    lock = (BASE/'improvement-controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (CONTINUE/'prepared.json').exists(), 'Continuation already prepared'
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    assert not active.strip(), 'GPUs occupied'
    assert json.loads((TRAIN/'launch-status.json').read_text())['phase'] == 'completed'
    manifest = json.loads((TRAIN/'reproducibility/code-manifest.json').read_text())
    assert all(digest(BASE/name) == expected for name, expected in manifest.items())
    OUT.mkdir(parents=True, exist_ok=True)
    CONTINUE.mkdir(parents=True, exist_ok=True)
    source = (TRAIN/'checkpoints/last.pt').resolve()
    # A sibling alias also tells checkpoint retention to preserve this payload.
    preserved = source.parent.parent/'epoch002-preserved.pt'
    assert not preserved.exists()
    preserved.symlink_to(source.relative_to(preserved.parent))
    receipts = BASE/'checkpoint-cache/objects'
    matches = [p.with_suffix('.pt') for p in receipts.glob('*.json')
               if json.loads(p.read_text()).get('source') == str(source)]
    assert len(matches) == 1
    local = BASE/'epoch002-preserved.pt'
    os.link(matches[0], local)
    assert local.stat().st_size == source.stat().st_size
    saved = torch.load(local, map_location='cpu', mmap=True, weights_only=False)
    assert saved['global_step'] == 1252 and saved['next_epoch'] == 2 and saved['next_microbatch'] == 0
    assert saved['joint_codebook_sha256'] == 'e378029c56a5a9e9096b0a18b56e0c592ccfe6dff69b5061860578d00c680611'
    assert len(saved['optimizer']['state']) == 780 and len(saved['rng_state_by_rank']) == 4
    assert saved['scheduler']['after']['T_max'] == 62600 and saved['scheduler']['after']['last_epoch'] == 1252
    assert all(int(value['step']) == 1252 and torch.isfinite(value['exp_avg']).all()
               and torch.isfinite(value['exp_avg_sq']).all() for value in saved['optimizer']['state'].values())
    assert saved['recipe'] == json.loads((TRAIN/'prepared.json').read_text())['recipe']
    del saved
    checkpoint_hash = digest(local)
    baseline = json.loads((TRAIN/'evaluations/epoch002.json').read_text())
    assert baseline['generated_images'] == 50000 and baseline['real_images'] == 1281167
    plan = dict(source_run=TRAIN.name, source_step=1252, source_epoch=2,
        checkpoint=str(preserved), local_checkpoint=str(local), checkpoint_sha256=checkpoint_hash,
        reference_fid=baseline['eval/fid'], generated_images=50000, real_images=1281167,
        source_sampling=dict(temperature=1., top_k=[1024], top_p=[.95]),
        candidate_sampling=dict(temperature=1., top_k=[512], top_p=[.95]),
        seed=261001, batch_per_gpu=128, world_size=4, training_updates_added=0,
        training_sampler_will_remain_unchanged=True)
    write(OUT/'plan.json', plan)
    write(CONTINUE/'prepared.json', dict(start_epoch=2, start_step=1252, stop_epoch=6,
        stop_step=3756, added_updates=2504, model_optimizer_rng_restored=True,
        learning_rate_schedule_restarted=False, cosine_total_updates=62600,
        sampling_unchanged=True, source_checkpoint=str(preserved),
        source_checkpoint_sha256=checkpoint_hash, evaluation_epochs=[4,6]))
    for name in ['completed.json', 'launch-status.json', 'comparison.json', 'final-online-verification.json']:
        if (TRAIN/name).exists():shutil.copyfile(TRAIN/name, CONTINUE/('epoch002-'+name))
    for rank in range(4):
        shutil.copyfile(TRAIN/f'train/startup-rank{rank}.json', CONTINUE/f'epoch002-startup-rank{rank}.json')
    shutil.copyfile(Path(__file__), CONTINUE/Path(__file__).name)
    shutil.copyfile(ROOT/'scripts/tools/evaluate_laser_joint32k_sampler.py', OUT/'evaluate.py')
    write(OUT/'source-manifest.json', dict(evaluator_sha256=digest(OUT/'evaluate.py'),
        training_manifest_sha256=digest(TRAIN/'reproducibility/code-manifest.json'),
        checkpoint_sha256=checkpoint_hash))
    environment = dict(os.environ, **json.loads((BASE/'environment.json').read_text()))
    environment['WANDB_API_KEY'] = Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
    child = None
    cancelled = False

    def stop(*_):
        nonlocal cancelled
        cancelled = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def launch(command, phase, logfile):
        nonlocal child
        if cancelled:
            raise RuntimeError('Cancelled')
        with logfile.open('a') as log:
            child = subprocess.Popen(command, cwd=BASE, env=environment,
                stdout=log, stderr=subprocess.STDOUT)
            status = dict(phase=phase, pid=os.getpid(), child_pid=child.pid,
                source_step=1252, world_size=4, global_batch=2048, time=time.time())
            write(CONTINUE/'controller.json', status)
            if phase == 'train':
                write(TRAIN/'launch-status.json', dict(status, run_id=TRAIN.name,
                    resuming=True, stop_epoch=6, stop_step=3756, microbatch=64, accumulation_steps=8))
            result = child.wait()
        if result or cancelled:
            write(CONTINUE/'controller.json', dict(status, phase='cancelled' if cancelled else 'failed',
                returncode=result, finished_at=time.time()))
            raise RuntimeError(f'{phase} returned {result}')

    distributed = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=4']
    launch([*distributed, str(OUT/'evaluate.py')], 'sampler_fid50k', OUT/'evaluation.log')
    result = json.loads((OUT/'fid50k.json').read_text())
    assert result['generated_images'] == 50000 and result['real_images'] == 1281167
    assert json.loads((OUT/'checkpoint-verification.json').read_text())['passed']
    assert digest(local) == checkpoint_hash
    write(CONTINUE/'sampler-comparison.json', result)
    # Keep every point of the main learning curve on its existing sampler.
    # The alternative measurement remains in its own W&B run.
    if (TRAIN/'completed.json').exists():
        (TRAIN/'completed.json').unlink()
    launch([*distributed, str(BASE/'train.py'), '--epochs', '6', '--microbatch', '64',
        '--resume', str(preserved)], 'train', TRAIN/'train.log')
    curves = {}
    reference = ROOT/'outputs/imagenet-original-rq1400m-rfid473-scratch20-4h100-20261014/evaluations'
    for epoch in [1,2,4,6]:
        current = json.loads((TRAIN/'evaluations'/f'epoch{epoch:03d}.json').read_text())
        rq = json.loads((reference/f'epoch{epoch:03d}.json').read_text())
        assert current['train/global_step'] == rq['train/global_step'] == epoch*626
        assert current['generated_images'] == rq['generated_images'] == 50000
        assert current['real_images'] == rq['real_images'] == 1281167
        curves[str(epoch)] = dict(joint32k_fid=current['eval/fid'], original_rq_fid=rq['eval/fid'],
            joint_minus_rq=current['eval/fid']-rq['eval/fid'])
    write(TRAIN/'comparison.json', dict(completed_epochs=6, global_step=3756, curves=curves,
        sampler_unchanged=True, training_restarted=False, continued_beyond_six_epochs=False))
    status = dict(phase='completed', pid=os.getpid(), run_id=TRAIN.name, resuming=True,
        completed_epochs=6, global_step=3756, returncode=0, finished_at=time.time())
    write(CONTINUE/'controller.json', status)
    write(TRAIN/'launch-status.json', status)


if __name__ == '__main__':
    main()
