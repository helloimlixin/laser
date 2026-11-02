#!/usr/bin/env python3
"""Validate and supervise an inference-only sampler sweep beside training."""
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
BASE = Path('/tmp/laser-rq-joint20k-depth5-imagenet-20261017')
OUT = ROOT/'outputs/imagenet-laser-joint20k-depth5-epoch6-sampler-sweep-20261017'
ACTIVE = ROOT/'outputs/imagenet-laser-joint20k-depth5-scratch100-4h100-20261017'
LOCAL = Path('/tmp/laser-joint20k-depth5-sampler-sweep-20261017')


def write(name, **values):
    path = OUT/name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(dict(values, time=time.time()), indent=2)+'\n')
    temporary.replace(path)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    lock = (OUT/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (OUT/'plan.json').exists(), 'Sweep already prepared'
    import torch
    torch.set_num_threads(4)
    write('status.json', phase='verifying_checkpoint', pid=os.getpid())
    records = [json.loads(p.read_text()) for p in
        (ACTIVE/'persistent-train/checkpoints/.checkpoint-data').glob('*.json')]
    records = [r for r in records if r['global_step'] == 3756]
    assert len(records) == 1
    receipt = records[0]
    source = Path(receipt['local'])
    assert source.stat().st_size == Path(receipt['persistent']).stat().st_size == receipt['bytes']
    assert sha(source) == receipt['sha256']
    saved = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
    assert saved['global_step'] == 3756 and saved['next_epoch'] == 6 and saved['next_microbatch'] == 0
    assert saved['run_id'] == ACTIVE.name and len(saved['rng_state_by_rank']) == 4
    assert saved['joint_codebook_sha256'] == '922c33037770287facb9ba8f2c477224087155ff14af3b9d6d7390b5a40485a9'
    assert saved['scheduler']['after']['last_epoch'] == 3756
    assert all(int(s['step']) == 3756 for s in saved['optimizer']['state'].values())
    baseline = json.loads((ACTIVE/'evaluations/epoch006.json').read_text())
    assert saved['evaluation']['fid'] == receipt['fid'] == baseline['eval/fid']
    assert baseline['generated_images'] == 50000 and baseline['real_images'] == 1281167
    del saved
    manifest = json.loads((ACTIVE/'reproducibility/code-manifest.json').read_text())['files']
    assert all(sha(BASE/name) == expected for name, expected in manifest.items())
    LOCAL.mkdir(parents=True, exist_ok=True)
    checkpoint = LOCAL/'epoch006.pt'
    assert not checkpoint.exists()
    os.link(source, checkpoint)
    policies = {'baseline':dict(temperature=1., top_k=1024, top_p=.95)}
    for temperature in (.9, .95, 1.):
        for top_k in (1024, 2048, None):
            if temperature == 1. and top_k == 1024:
                continue
            name = f't{round(temperature*100):03d}_k{top_k or "all"}'
            policies[name] = dict(temperature=temperature, top_k=top_k, top_p=.95)
    policies['fifth_depth_unrestricted'] = dict(temperature=1., top_k=[1024,1024,1024,1024,20481], top_p=.95)
    (OUT/'source').mkdir(exist_ok=True)
    sources = {
        'evaluate.py':ROOT/'scripts/tools/sweep_laser_joint20k_depth5_samplers.py',
        'joint_integer_sampling_audit.py':ROOT/'src/training/joint_integer_sampling_audit.py',
        'sweep_compound512_samplers.py':ROOT/'scripts/tools/sweep_compound512_samplers.py',
        'controller.py':Path(__file__),
    }
    for name, path in sources.items():
        shutil.copyfile(path, OUT/'source'/name)
    write('source-manifest.json', files={name:sha(OUT/'source'/name) for name in sources})
    write('plan.json', wandb_id=OUT.name, source_run=ACTIVE.name, source_epoch=6,
        source_step=3756, checkpoint=receipt, local_checkpoint=str(checkpoint),
        baseline_fid50k=baseline['eval/fid'], baseline_sampling=policies['baseline'],
        reference_sha256=sha(BASE/'inputs/imagenet_256_train.npz'),
        reference_images=1281167, reference_split='train', source_depth=5, vocabulary=20481,
        screen_images=4096, screen_seed=261017206, screen_is_not_fid=True,
        confirmation_images=50000, confirmation_seed=261001, policies=policies,
        selection='Two distinct nonbaseline policies: lowest moment MMD and highest Inception Score',
        weights_unmodified=True, training_updates_added=0, active_training_sampler_changed=False,
        world_size=4, sampling_batch_per_gpu=128, memory_cap_gib_per_gpu=16,
        active_training_recipe_sha256=sha(BASE/'inputs/recipe.yaml'),
        active_training_manifest_sha256=sha(ACTIVE/'reproducibility/code-manifest.json'))
    environment = dict(os.environ, **json.loads((BASE/'environment.json').read_text()))
    environment.update(WANDB_DIR=str(LOCAL/'wandb'), PYTHONPATH='.:runtime:scripts/tools',
        WANDB_API_KEY=Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip())
    (LOCAL/'wandb').mkdir(exist_ok=True)
    child = None

    def stop(*_):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)
        raise KeyboardInterrupt('Sampler sweep interrupted; training remains running')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def launch(command, phase):
        nonlocal child
        with (OUT/(phase+'.log')).open('ab', buffering=0) as log:
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
            write('status.json', phase=phase, pid=os.getpid(), child_pid=child.pid)
            result = child.wait()
        if result:
            raise RuntimeError(f'{phase} returned {result}')

    launch([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
        'tests/test_joint_integer_sampling_audit.py', 'tests/test_compound512_sampler_screen.py',
        '--junitxml='+str(OUT/'tests.xml')], 'cpu_verification')
    launch([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=4',
            str(OUT/'source/evaluate.py')], 'evaluation')
    report = json.loads((OUT/'report.json').read_text())
    assert all(r['generated_images'] == 50000 for r in report['confirmation'])
    plan = report['plan']
    assert sha(BASE/'inputs/recipe.yaml') == plan['active_training_recipe_sha256']
    assert sha(ACTIVE/'reproducibility/code-manifest.json') == plan['active_training_manifest_sha256']
    progress = json.loads((ACTIVE/'train/progress-rank0.json').read_text())
    write('final-verification.json', passed=True, source_step=3756,
        checkpoint_sha256_verified=True, active_training_recipe_and_runtime_unchanged=True,
        training_current_step=progress['train/global_step'],
        selected_policy=report['selected_policy'], baseline_fid50k=report['baseline_fid50k'],
        selected_fid50k=report['selected_fid50k'], improvement=report['improvement'])
    write('status.json', phase='completed', selected_policy=report['selected_policy'],
        fid=report['selected_fid50k'], improvement=report['improvement'])


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        main()
    except BaseException as error:
        write('status.json', phase='failed', error_type=type(error).__name__, error=str(error))
        raise
