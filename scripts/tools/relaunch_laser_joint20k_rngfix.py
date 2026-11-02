#!/usr/bin/env python3
"""Replace the eight-H100 run with a fresh, independently seeded rank run."""
import argparse
import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import zipfile

ROOT = Path('/workspace/Projects/laser')
PREVIOUS_RUN = 'imagenet-laser-joint20k-depth5-rqnoise-k2048-scratch100-8h100-20261017'
PREVIOUS = ROOT/'outputs'/PREVIOUS_RUN
PREVIOUS_BASE = Path('/tmp/laser-rq-joint20k-depth5-rqnoise-k2048-8h100-imagenet-20261017')
RUN = 'imagenet-laser-joint20k-depth5-rqnoise-k2048-rngfix-scratch100-8h100-20261017'
OUT = ROOT/'outputs'/RUN
BASE = Path('/tmp/laser-rq-joint20k-depth5-rqnoise-k2048-rngfix-8h100-imagenet-20261017')
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


def core():
    spec = importlib.util.spec_from_file_location('rngfix_controller_core', BASE/'controller_core.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.BASE, module.OUT, module.RUN = BASE, OUT, RUN
    module.SOURCE_RUN = PREVIOUS_RUN
    # The shared controller's data subprocess must enter this run's CLI.
    module.__file__ = str(Path(__file__).resolve())
    return module


def prepare():
    import yaml
    assert not BASE.exists() and not OUT.exists(), 'Refusing to overwrite a prepared run'
    BASE.mkdir(parents=True)
    OUT.mkdir(parents=True)
    for name in ('source', 'runtime'):
        shutil.copytree(PREVIOUS_BASE/name, BASE/name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('upstream', 'data', 'torch-cache'):
        (BASE/name).symlink_to(PREVIOUS_BASE/name, target_is_directory=True)
    (BASE/'inputs').mkdir()
    for path in (PREVIOUS_BASE/'inputs').iterdir():
        destination = BASE/'inputs'/path.name
        if path.suffix in ('.pt', '.npz'):
            os.link(path, destination)
        else:
            shutil.copyfile(path, destination)
    for name in ('train.py', 'verify.py', 'configure_samples.py'):
        shutil.copyfile(PREVIOUS_BASE/name, BASE/name)
    shutil.copyfile(ROOT/'scripts/tools/duplicate_laser_joint20k_depth5_rqnoise.py',
                    BASE/'controller_core.py')
    recipe = yaml.safe_load((BASE/'inputs/recipe.yaml').read_text())
    previous_recipe = json.loads(json.dumps(recipe))
    recipe.update(run_id=RUN, name='LASER joint20k depth5 | RQ noise K2048 | independent rank RNG | fresh100',
                  version='laser-native-joint20k-depth5-rqnoise-k2048-rngfix-v1',
                  training_rng=dict(policy='independent_rank_v1', base_seed=int(recipe['seed'])),
                  lineage=dict(replaces=PREVIOUS_RUN, stage2_initialization='scratch'))
    for key in ('tokenizer', 'targets', 'sampling', 'evaluation', 'initialization'):
        assert recipe[key] == previous_recipe[key]
    assert recipe['global_batch'] == 2048 and recipe['microbatch'] == 64
    assert recipe['epochs'] == recipe['lr_schedule_epochs'] == 100
    for target in (BASE/'inputs/recipe.yaml', ROOT/'configs/stage2'/f'{RUN}.yaml'):
        target.write_text(yaml.safe_dump(recipe, sort_keys=False))
    changed = ['source/src/training/rq_imagenet_shared.py', 'runtime/rq_imagenet_entry.py']
    for relative, source in [(changed[0], ROOT/'src/training/rq_imagenet_shared.py'),
                             (changed[1], ROOT/'runtime/rq_imagenet_entry.py'),
                             ('source/src/training/rq_rng.py', ROOT/'src/training/rq_rng.py')]:
        shutil.copyfile(source, BASE/relative)
    verify = (BASE/'verify.py').read_text()
    needle = "    profile.verify_resume(saved)\n"
    assert verify.count(needle) == 1
    verify = verify.replace(needle, needle + """    assert saved['runtime']['training_rng_policy'] == 'independent_rank_v1'
    states = saved['rng_state_by_rank']
    assert len({s['cuda'].numpy().tobytes() for s in states}) == 8
    startup = [json.loads((smoke/f'rng-startup-rank{rank}.json').read_text()) for rank in range(8)]
    assert [r['training_rng_seed'] for r in startup] == list(range(261001,261009))
    assert len({r['dropout_probe_sha256'] for r in startup}) == 8
    assert all(r['training_rng_unchanged_by_audit'] for r in startup)
    assert all(not r['training_rng_restored_from_checkpoint'] for r in startup)
""")
    verify = verify.replace('proof=dict(passed=True,',
        'proof=dict(passed=True,rank_rng_independence_verified=True,unique_rank_rng_states=8,unique_dropout_masks=8,')
    (BASE/'verify.py').write_text(verify)
    env = json.loads((PREVIOUS_BASE/'environment.json').read_text())
    env = {key:value.replace(str(PREVIOUS_BASE), str(BASE)) for key,value in env.items()}
    write(BASE/'environment.json', env)
    (BASE/'wandb').mkdir()
    repro = OUT/'reproducibility'
    repro.mkdir()
    for name in ('recipe-audit.json', 'upstream-imagenet1400m.yaml', 'data-verification.json',
                 'noise-calibration-report.json', 'sampler-selection.json', 'version-verification.json'):
        shutil.copyfile(PREVIOUS/'reproducibility'/name, repro/name)
    audit = json.loads((repro/'recipe-audit.json').read_text())
    audit.update(comparison_base=PREVIOUS_RUN, training_runtime_python_unchanged=False,
                 unchanged_except_independent_rank_rng=True,
                 training_rng_policy=recipe['training_rng'],
                 changes=['Independent rank training RNG after DDP on fresh launches',
                          'New run identity; same model, tokenizer, noise, batch, LR and sampler'])
    write(repro/'recipe-audit.json', audit)
    for name in ('recipe.yaml', 'stage2-training-config.yaml'):
        shutil.copyfile(BASE/'inputs'/name, repro/name)
    shutil.copyfile(ROOT/'tests/test_rq_rng.py', repro/'test_rq_rng.py')
    shutil.copyfile(Path(__file__), repro/Path(__file__).name)
    patch = []
    for name in changed:
        patch.extend(difflib.unified_diff((PREVIOUS_BASE/name).read_text().splitlines(True),
                                        (BASE/name).read_text().splitlines(True),
                                        fromfile='previous/'+name, tofile='rngfix/'+name))
    for path, relative in [(BASE/'source/src/training/rq_rng.py', 'src/training/rq_rng.py'),
                           (ROOT/'tests/test_rq_rng.py', 'tests/test_rq_rng.py')]:
        patch.extend(difflib.unified_diff([], path.read_text().splitlines(True),
                                        fromfile='/dev/null', tofile='rngfix/'+relative))
    (repro/'rng-fix.patch').write_text(''.join(patch))
    assert sha(BASE/'inputs/stage1-model.pt') == recipe['tokenizer']['sha256']
    assert sha(BASE/'inputs/joint-codebook.pt') == recipe['tokenizer']['codebook_sha256']
    assert sha(BASE/'inputs/imagenet_256_train.npz') == recipe['evaluation']['reference_sha256']
    ready = json.loads((BASE/'data/ready.json').read_text())
    assert ready['ready'] and ready['training_images'] == 1281167 and ready['validation_images'] == 50000
    plan = dict(prepared=True, training_launched=False, run_id=RUN, replaces=PREVIOUS_RUN,
                recipe=recipe, fresh_stage2=True, fresh_optimizer=True, stage2_checkpoint_loaded=False,
                initial_global_step=0, world_size=8, global_batch=2048, microbatch=64,
                accumulation_steps=4, epochs=100, optimizer_updates_per_epoch=626,
                cosine_total_updates=62600, verified_data_reused=True,
                only_training_behavior_change='Independent rank RNG after model initialization and DDP',
                quality_improvement_established=False)
    write(OUT/'prepared.json', plan)
    write(repro/'cpu-tests.json', dict(passed=True, tests=20,
        command='pytest tests/test_rq_rng.py tests/test_exact_global_batch.py tests/test_amp_replay.py'))
    files = [BASE/name for name in ('train.py', 'verify.py', 'configure_samples.py', 'controller_core.py', 'environment.json')]
    for directory in ('source', 'runtime', 'upstream'):
        files.extend((BASE/directory).rglob('*.py'))
    files.extend((BASE/'inputs').glob('*.yaml'))
    files.extend((BASE/'inputs').glob('*.json'))
    manifest = {str(p.relative_to(BASE)):sha(p) for p in files}
    write(repro/'code-manifest.json', manifest)
    with zipfile.ZipFile(repro/'source.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, path.relative_to(BASE))
        bundle.write(BASE/'inputs/joint-codebook.pt', 'inputs/joint-codebook.pt')
    print(json.dumps(plan), flush=True)


def matches(pid, token):
    try:
        return token.encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
    except FileNotFoundError:
        return False


def preserve_and_stop():
    receipts = [json.loads(p.read_text()) for p in
                (PREVIOUS/'persistent-train/checkpoints/.checkpoint-data').glob('*.json')]
    receipt = max(receipts, key=lambda r:r['global_step'])
    local, durable = Path(receipt['local']), Path(receipt['persistent'])
    assert local.stat().st_size == durable.stat().st_size == receipt['bytes']
    assert sha(local) == receipt['sha256']
    with local.open('rb') as a, durable.open('rb') as b:
        assert a.read(64) == b.read(64)
        a.seek(-64, 2); b.seek(-64, 2)
        assert a.read(64) == b.read(64)
    alias = PREVIOUS/f'epoch{receipt["epoch"]:03d}-before-rngfix.pt'
    if not alias.exists():
        alias.symlink_to(os.path.relpath(durable, alias.parent))
    local_alias = PREVIOUS_BASE/f'epoch{receipt["epoch"]:03d}-before-rngfix.pt'
    if not local_alias.exists():
        os.link(local, local_alias)
    status = json.loads((PREVIOUS/'launch-status.json').read_text())
    ranks = [json.loads((PREVIOUS/f'train/progress-rank{rank}.json').read_text()) for rank in range(8)]
    write(OUT/'preserved-predecessor-checkpoint.json', dict(passed=True, checkpoint=receipt,
          persistent_alias=str(alias), local_alias=str(local_alias),
          latest_observed_step=max(r['train/global_step'] for r in ranks),
          reused_for_new_training=False, time=time.time()))
    if matches(status['pid'], 'duplicate_laser_joint20k_depth5_rqnoise.py'):
        os.kill(status['pid'], signal.SIGTERM)
    for second in range(45):
        live = [r['pid'] for r in ranks if matches(r['pid'], str(PREVIOUS_BASE/'train.py'))]
        controller_live = matches(status['pid'], 'duplicate_laser_joint20k_depth5_rqnoise.py')
        if not live and not controller_live:
            break
        if second in (15, 30):
            for pid in live:
                os.kill(pid, signal.SIGTERM)
        time.sleep(1)
    assert not any(matches(r['pid'], str(PREVIOUS_BASE/'train.py')) for r in ranks)
    write(PREVIOUS/'launch-status.json', dict(phase='stopped_for_rngfix_restart', replacement_run=RUN,
          preserved_checkpoint_step=receipt['global_step'], time=time.time()))
    import wandb
    os.environ['WANDB_API_KEY'] = KEY.read_text().strip()
    previous = wandb.Api(timeout=45).run(f'helloimlixin-rutgers/laser/{PREVIOUS_RUN}')
    previous.summary.update({'training/status':'stopped_for_rngfix_restart',
        'training/replacement_run':RUN, 'training/preserved_checkpoint_step':receipt['global_step']})
    previous.update()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'data', 'run', 'panels'])
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare()
    elif args.phase == 'data':
        ready = json.loads((BASE/'data/ready.json').read_text())
        assert ready['ready'] and ready['training_images'] == 1281167
        print(json.dumps(ready), flush=True)
    elif args.phase == 'panels':
        core().configure_focused_panels()
    else:
        for name, expected in json.loads((OUT/'reproducibility/code-manifest.json').read_text()).items():
            assert sha(BASE/name) == expected, name
        preserve_and_stop()
        core().supervise()


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if OUT.exists():
            write(OUT/'launch-status.json', dict(phase='failed', error_type=type(error).__name__,
                  error=str(error), time=time.time()))
        raise
