#!/usr/bin/env python3
"""Continue the earlier-best LASER recipe on all eight H100s."""
import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile

ROOT = Path('/workspace/Projects/laser')
SOURCE_RUN = 'imagenet-laser-joint20k-depth5-earlierbest-continue100-4h100-20261018'
SOURCE = ROOT/'outputs'/SOURCE_RUN
SOURCE_BASE = Path('/tmp/laser-joint20k-earlier-best-20261018')
RUN = 'imagenet-laser-joint20k-depth5-earlierbest-continue100-8h100-20261018'
OUT = ROOT/'outputs'/RUN
BASE = Path('/tmp/laser-joint20k-earlier-best-8h100-20261018')
KEY = Path('/root/.config/laser/imagenet-stage2-wandb.key')


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str)+'\n')
    temporary.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare():
    import yaml
    if BASE.exists() or OUT.exists():
        raise RuntimeError('Refusing to replace a prepared eight-GPU run')
    request = json.loads((SOURCE/'eight-gpu-migration-request.json').read_text())
    receipt = request['latest_checkpoint']
    manifest = json.loads((SOURCE/'reproducibility/code-manifest.json').read_text())
    for name, expected in manifest.items():
        assert sha(SOURCE_BASE/name) == expected, name
    BASE.mkdir(parents=True)
    OUT.mkdir(parents=True)
    for name in ('source', 'runtime'):
        shutil.copytree(SOURCE_BASE/name, BASE/name,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('upstream', 'data', 'torch-cache'):
        (BASE/name).symlink_to(SOURCE_BASE/name, target_is_directory=True)
    (BASE/'inputs').mkdir()
    for path in (SOURCE_BASE/'inputs').iterdir():
        if path.suffix in ('.pt', '.npz'):
            os.link(path, BASE/'inputs'/path.name)
        else:
            shutil.copyfile(path, BASE/'inputs'/path.name)
    for name in ('train.py', 'verify.py', 'configure_samples.py'):
        shutil.copyfile(SOURCE_BASE/name, BASE/name)
    original = yaml.safe_load((BASE/'inputs/recipe.yaml').read_text())
    recipe = copy.deepcopy(original)
    recipe.update(run_id=RUN, name='LASER joint20k depth5 | earlier best | 8 H100 | continue100',
        lineage=dict(source_run=SOURCE_RUN, source_epoch=receipt['epoch'],
            source_step=receipt['global_step'], checkpoint_sha256=receipt['sha256'],
            reason='Use all eight available H100s; preserve earlier-best training recipe',
            stage2_initialization='continued',
            hardware_migration=dict(old_world_size=4, world_size=8, old_accumulation=8,
                accumulation=4, microbatch=64, global_batch=2048,
                rng_policy='Retain four saved streams; derive four added rank streams from checkpoint SHA and step')))
    before, after = copy.deepcopy(original), copy.deepcopy(recipe)
    for value in (before, after):
        for key in ('run_id', 'name', 'lineage'):
            value.pop(key, None)
    assert before == after, 'Hardware migration must not change the training or sampler recipe'
    (BASE/'inputs/recipe.yaml').write_text(yaml.safe_dump(recipe, sort_keys=False))
    env = {key:value.replace(str(SOURCE_BASE), str(BASE)) for key,value in
        json.loads((SOURCE_BASE/'environment.json').read_text()).items()}
    env['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    write(BASE/'environment.json', env)
    for name in ('wandb', 'wandb-cache', 'wandb-data'):
        (BASE/name).mkdir(exist_ok=True)
    local = Path(receipt['local'])
    assert local.stat().st_size == receipt['bytes'] and sha(local) == receipt['sha256']
    os.link(local, BASE/'source-checkpoint.pt')
    repro = OUT/'reproducibility'
    repro.mkdir()
    shutil.copyfile(Path(__file__), repro/Path(__file__).name)
    shutil.copyfile(ROOT/'src/training/rq_layout_rng.py', repro/'rq_layout_rng.py')
    shutil.copyfile(ROOT/'tests/test_rq_layout_rng.py', repro/'test_rq_layout_rng.py')
    shutil.copyfile(SOURCE/'eight-gpu-migration-request.json', repro/'migration-request.json')
    for name in ('recipe.yaml', 'stage2-training-config.yaml', 'stage2-archive-config.yaml'):
        shutil.copyfile(BASE/'inputs'/name, repro/name)
    files = [BASE/name for name in manifest]
    write(repro/'code-manifest.json', {str(p.relative_to(BASE)):sha(p) for p in files})
    assert all(sha(BASE/name) == expected for name,expected in manifest.items() if name.endswith('.py'))
    with zipfile.ZipFile(repro/'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(BASE))
    write(OUT/'plan.json', dict(run_id=RUN, source_run=SOURCE_RUN, source_checkpoint=receipt,
        protected_best_checkpoint=request['protected_best_checkpoint'],
        source_step=receipt['global_step'], source_epoch=receipt['epoch'],
        url=f'https://wandb.ai/helloimlixin-rutgers/laser/runs/{RUN}',
        world_size=8, global_batch=2048, microbatch=64, accumulation_steps=4,
        epochs=100, lr_schedule_epochs=100, stop_step=62600,
        training_and_sampling_recipe_unchanged=True, original_training_python_byte_identical=True,
        checkpoint_optimizer_scheduler_scaler_preserved=True,
        original_rank_rng_states_preserved=4, added_rank_rng_states=4,
        unsaved_updates_to_replay=request['unsaved_updates_to_replay'],
        experiments_cancelled=True, time=time.time()))
    write(OUT/'status.json', dict(phase='prepared', time=time.time()))
    print(json.dumps(dict(prepared=True, source_step=receipt['global_step'], world_size=8)), flush=True)


def migrate():
    import torch
    from omegaconf import OmegaConf
    torch.set_num_threads(8)
    spec = importlib.util.spec_from_file_location('checkpoint_hash_support',
        ROOT/'scripts/tools/restore_joint20k_earlier_best.py')
    support = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(support)
    rng_spec = importlib.util.spec_from_file_location('layout_rng_support', OUT/'reproducibility/rq_layout_rng.py')
    rng = importlib.util.module_from_spec(rng_spec)
    rng_spec.loader.exec_module(rng)
    plan = json.loads((OUT/'plan.json').read_text())
    saved = torch.load(BASE/'source-checkpoint.pt', map_location='cpu', mmap=True, weights_only=False)
    assert saved['run_id'] == SOURCE_RUN and saved['global_step'] == plan['source_step']
    assert saved['next_microbatch'] == 0 and len(saved['rng_state_by_rank']) == 4
    assert saved['next_epoch'] == plan['source_epoch']
    protected = {name:support.tensor_hash(saved[name]) for name in
        ('state_dict', 'optimizer', 'scheduler', 'scaler')}
    original_rng_hash = support.tensor_hash(saved['rng_state_by_rank'])
    factory = lambda seed: torch.Generator(device='cuda:0').manual_seed(seed).get_state()
    expanded, seeds = rng.expand_four_to_eight(saved['rng_state_by_rank'],
        plan['source_checkpoint']['sha256'], saved['global_step'], factory)
    assert support.tensor_hash(expanded[:4]) == original_rng_hash
    saved['rng_state_by_rank'] = expanded
    saved['run_id'] = RUN
    saved['recipe'] = OmegaConf.to_container(OmegaConf.load(BASE/'inputs/recipe.yaml'), resolve=True)
    saved['runtime'] = copy.deepcopy(saved['runtime'])
    saved['runtime'].update(world_size=8, accumulation_steps=4, microbatch=64,
        fresh_stage2=False, fresh_optimizer=False, global_step=saved['global_step'])
    saved['hardware_migration'] = dict(source_run=SOURCE_RUN,
        source_checkpoint_sha256=plan['source_checkpoint']['sha256'], source_step=saved['global_step'],
        old_world_size=4, world_size=8, global_batch=2048,
        original_rank_streams_preserved=4, new_rank_seeds=seeds, trajectory_bitwise_identical=False)
    destination = BASE/'resume-eight-gpu.pt'
    temporary = destination.with_suffix('.tmp')
    torch.save(saved, temporary)
    temporary.replace(destination)
    del saved
    check = torch.load(destination, map_location='cpu', mmap=True, weights_only=False)
    assert {name:support.tensor_hash(check[name]) for name in protected} == protected
    assert support.tensor_hash(check['rng_state_by_rank'][:4]) == original_rng_hash
    assert len(check['rng_state_by_rank']) == 8
    assert all(int(s['step']) == plan['source_step'] for s in check['optimizer']['state'].values())
    proof = dict(passed=True, checkpoint=str(destination), checkpoint_sha256=sha(destination),
        source_step=plan['source_step'], protected_training_state_hashes=protected,
        weights_optimizer_scheduler_scaler_bit_identical=True,
        original_rank_rng_states_bit_identical=True, rank_rng_states=8,
        added_rank_seeds=seeds, world_size=8, global_batch=2048, accumulation_steps=4, time=time.time())
    write(OUT/'checkpoint-migration.json', proof)
    print(json.dumps(proof), flush=True)


def run():
    lock = (BASE/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    proof = json.loads((OUT/'checkpoint-migration.json').read_text())
    assert proof['passed'] and proof['rank_rng_states'] == 8
    for name, expected in json.loads((OUT/'reproducibility/code-manifest.json').read_text()).items():
        assert sha(BASE/name) == expected, name
    occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True).strip()
    if occupied:
        raise RuntimeError('Old workers must exit before eight-GPU launch')
    if (OUT/'train').exists():
        raise RuntimeError('Refusing to overwrite a production continuation')
    env = dict(os.environ, **json.loads((BASE/'environment.json').read_text()),
        WANDB_API_KEY=KEY.read_text().strip(), LASER_PRIVATE_SMOKE_DIR=str(BASE/'private-smoke'))
    distributed = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8']
    resume = ['--resume', str(BASE/'resume-eight-gpu.pt')]
    child = None
    def stop(*_):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)
        raise KeyboardInterrupt('Eight-GPU earlier-best continuation stopped')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for phase, command in [
        ('eight_gpu_preflight', [*distributed, str(BASE/'train.py'), *resume, '--smoke-updates', '1']),
        ('training', [*distributed, str(BASE/'train.py'), *resume, '--epochs', '100', '--new-wandb-run'])]:
        if phase == 'training':
            smoke = json.loads((BASE/'private-smoke/verification.json').read_text())
            assert smoke['passed'] and smoke['global_step'] == proof['source_step']+1
            assert smoke['all_adam_steps'] == proof['source_step']+1
            for rank in range(8):
                startup = json.loads((BASE/f'private-smoke/startup-rank{rank}.json').read_text())
                assert startup['world_size'] == 8 and startup['accumulation_steps'] == 4
                assert startup['global_batch'] == 2048 and startup['global_step'] == proof['source_step']
                assert not startup['fresh_stage2'] and not startup['fresh_optimizer']
            write(OUT/'eight-gpu-preflight.json', dict(passed=True, optimizer_updates=1,
                preflight_weights_discarded=True, all_eight_ranks_verified=True,
                world_size=8, global_batch=2048, accumulation_steps=4, time=time.time()))
        with (OUT/f'{phase}.log').open('ab', buffering=0) as log:
            child = subprocess.Popen(command, cwd=BASE, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
            write(OUT/'status.json', dict(phase=phase, pid=os.getpid(), child_pid=child.pid,
                command=command, source_step=proof['source_step'], world_size=8, time=time.time()))
            result = child.wait()
        if result:
            raise RuntimeError(f'{phase} exited {result}; see {OUT/phase}.log')
    write(OUT/'status.json', dict(phase='completed', time=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare','migrate','run'])
    args = parser.parse_args()
    try:
        {'prepare':prepare,'migrate':migrate,'run':run}[args.phase]()
    except BaseException as error:
        if OUT.exists():
            write(OUT/'failure.json', dict(error_type=type(error).__name__, error=str(error), time=time.time()))
        raise
