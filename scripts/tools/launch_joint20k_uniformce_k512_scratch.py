#!/usr/bin/env python3
"""Run one fresh ImageNet sparse-integer RQTransformer on all four B200s."""
import argparse
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
import zipfile


ROOT = Path('/workspace/Projects/laser')
SOURCE = Path('/tmp/laser-joint20k-depthwce-100ep-4b200-20261018')
RUN = 'imagenet-laser-joint20k-depth5-uniformce-k512-scratch100-4b200-20261018'
BASE = Path('/tmp/laser-joint20k-depth5-uniformce-k512-scratch100-4b200-20261018')
OUT = ROOT / 'outputs' / RUN
RECIPE = ROOT / 'configs/stage2' / (RUN + '.yaml')
KEY = Path('/root/.config/laser/wandb_api_key')
CHILD = None
STOPPING = False


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.pending')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def status(phase, **values):
    write(OUT / 'status.json', dict(phase=phase, run_id=RUN,
        supervisor_pid=os.getpid(), fresh_stage2=True, fresh_optimizer=True,
        world_size=4, hardware='4 NVIDIA B200', epochs=100,
        global_batch=2048, microbatch=256, accumulation_steps=2,
        objective='native_soft_target_cross_entropy',
        sampling=dict(temperature=1., top_k=512, top_p=.95),
        time=time.time(), **values))


def prepare():
    import yaml

    if BASE.exists() or RECIPE.exists() or (OUT / 'prepared.json').exists():
        raise RuntimeError('Fresh launch already prepared; refusing to overwrite it')
    OUT.mkdir(parents=True, exist_ok=True)
    (BASE / 'inputs').mkdir(parents=True)
    for directory in ('source', 'runtime'):
        shutil.copytree(SOURCE / directory, BASE / directory,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for directory in ('upstream', 'data', 'torch-cache'):
        (BASE / directory).symlink_to(SOURCE / directory, target_is_directory=True)
    for path in (SOURCE / 'inputs').iterdir():
        destination = BASE / 'inputs' / path.name
        if path.suffix in ('.pt', '.npz'):
            os.link(path, destination)
        else:
            shutil.copyfile(path, destination)
    shutil.copyfile(SOURCE / 'train.py', BASE / 'train.py')
    recipe = yaml.safe_load((SOURCE / 'inputs/recipe.yaml').read_text())
    old_recipe = yaml.safe_load((SOURCE / 'inputs/recipe.yaml').read_text())
    recipe.update(version='laser-native-joint20k-depth5-uniformce-k512-v1',
        run_id=RUN, name='LASER joint20k depth5 | uniform CE + K512 | fresh100 | 4 B200',
        epochs=100, lr_schedule_epochs=100, microbatch=256, global_batch=2048)
    # Absence of depth weights selects the existing native uniform-CE path.
    recipe['loss'] = dict(type='soft_target_cross_entropy', reduction='mean')
    recipe['sampling'] = dict(temperature=1., top_k=512, top_p=.95)
    recipe['lineage'] = dict(supersedes_run=old_recipe['run_id'],
        stage2_initialization='fresh_random', stage2_checkpoint_loaded=False,
        optimizer_initialization='fresh', scheduler_initialization='fresh',
        reason='User requested a single fresh uniform-CE top-k512 run using all GPUs')
    assert recipe['targets'] == old_recipe['targets']
    assert recipe['tokenizer'] == old_recipe['tokenizer']
    assert recipe['initialization'] == old_recipe['initialization']
    assert recipe['decoder'] == 'native_joint_integer' and recipe['auxiliary_loss_terms'] == 0
    for path in (RECIPE, BASE / 'inputs/recipe.yaml'):
        path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    config_path = BASE / 'inputs/stage2-training-config.yaml'
    config = yaml.safe_load(config_path.read_text())
    config['loss'].pop('depth_weights', None)
    config['experiment']['batch_size'] = 256
    config['experiment']['epochs'] = 100
    config['experiment']['sample'] = dict(temp=1., top_k=512, top_p=.95)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    archive_path = BASE / 'inputs/stage2-archive-config.yaml'
    archive = yaml.safe_load(archive_path.read_text())
    archive['sampling'] = dict(temp=1., top_k=512, top_p=.95)
    archive_path.write_text(yaml.safe_dump(archive, sort_keys=False))
    # The frozen trial's metadata assumed depth weights were always present.
    # Correct that label while retaining its existing native loss implementation.
    recipe_source = BASE / 'source/src/training/laser_joint_recipe.py'
    source = recipe_source.read_text()
    before = "if self.depth_loss_weights() == [1.] * self.shape[-1] else"
    after = "if self.depth_loss_weights() is None or self.depth_loss_weights() == [1.] * self.shape[-1] else"
    assert source.count(before) == 1
    recipe_source.write_text(source.replace(before, after))
    changed_python = []
    for directory in ('source', 'runtime'):
        for path in (BASE / directory).rglob('*.py'):
            relative = path.relative_to(BASE)
            if digest(path) != digest(SOURCE / relative):
                changed_python.append(str(relative))
    assert changed_python == ['source/src/training/laser_joint_recipe.py']
    env = {key: value.replace(str(SOURCE), str(BASE)) for key, value in
        json.loads((SOURCE / 'environment.json').read_text()).items()}
    env.update(CUDA_VISIBLE_DEVICES='0,1,2,3', LASER_WANDB_CHECKPOINT_UPLOADS='0')
    write(BASE / 'environment.json', env)
    for name in ('wandb', 'wandb-cache', 'wandb-data', 'checkpoint-staging', 'checkpoint-cache'):
        (BASE / name).mkdir(exist_ok=True)
    ready = json.loads((BASE / 'data/ready.json').read_text())
    assert ready['ready'] and ready['training_images'] == 1281167 and ready['validation_images'] == 50000
    assets = dict(stage1_model_sha256=digest(BASE / 'inputs/stage1-model.pt'),
        codebook_sha256=digest(BASE / 'inputs/joint-codebook.pt'),
        fid_reference_sha256=digest(BASE / 'inputs/imagenet_256_train.npz'))
    assert assets['stage1_model_sha256'] == recipe['tokenizer']['sha256']
    assert assets['codebook_sha256'] == recipe['tokenizer']['codebook_sha256']
    assert assets['fid_reference_sha256'] == recipe['evaluation']['reference_sha256']
    plan = dict(run_id=RUN, base=str(BASE), output=str(OUT), recipe=str(RECIPE),
        fresh_stage2=True, fresh_optimizer=True, fresh_scheduler=True,
        stage2_checkpoint_loaded=False, initial_step=0, initial_epoch=0,
        world_size=4, hardware='4 NVIDIA B200', microbatch=256,
        accumulation_steps=2, global_batch=2048, epochs=100,
        optimizer_updates_per_epoch=626, target_step=62600, cosine_updates=62600,
        loss='native uniform soft-target cross entropy over whole integer IDs',
        depth_weights=[1.] * 5, sampling=recipe['sampling'],
        target_temperatures=recipe['targets']['temperature'],
        code_shape=[8, 8, 5], vocabulary=20481, scalar_tokens_per_image=320,
        frozen_tokenizer_unchanged=True, target_policy_unchanged=True,
        training_math_unchanged=True, frozen_python_metadata_only_change=changed_python,
        reused_verified_data=ready, assets=assets, concurrent_training_runs=1,
        supersedes_run=old_recipe['run_id'], time=time.time())
    write(OUT / 'prepared.json', plan)
    repro = OUT / 'reproducibility'
    repro.mkdir(exist_ok=True)
    for path in (RECIPE, config_path, archive_path):
        destination = repro / ('recipe.yaml' if path == RECIPE else path.name)
        shutil.copyfile(path, destination)
    files = [BASE / name for name in ('train.py', 'environment.json')]
    for directory in ('source', 'runtime', 'upstream'):
        files.extend(p for p in (BASE / directory).rglob('*.py') if '__pycache__' not in p.parts)
    files.extend(p for p in (BASE / 'inputs').iterdir() if p.suffix in ('.yaml', '.json'))
    manifest = {str(p.relative_to(BASE)): digest(p) for p in files}
    write(repro / 'code-manifest.json', manifest)
    write(repro / 'adaptation-verification.json', dict(passed=True, **assets,
        training_math_unchanged=True, python_changes=changed_python,
        metadata_change='Correctly label the pre-existing no-depth-weights uniform CE path',
        frozen_tokenizer_unchanged=True, target_temperatures_unchanged=True,
        fresh_stage2=True, stage2_checkpoint_loaded=False))
    with zipfile.ZipFile(repro / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as snapshot:
        for path in files:
            snapshot.write(path, path.relative_to(BASE))
        snapshot.write(Path(__file__), Path(__file__).relative_to(ROOT))
    shutil.copyfile(__file__, OUT / Path(__file__).name)
    status('prepared')
    print(json.dumps(dict(prepared=True, run_id=RUN, world_size=4,
        fresh_stage2=True, sampling=recipe['sampling'])), flush=True)


def stop(signum, _frame):
    global STOPPING
    STOPPING = True
    if CHILD is not None and CHILD.poll() is None:
        os.killpg(CHILD.pid, signal.SIGINT)
    raise SystemExit(128 + signum)


def launch(command, env, phase):
    global CHILD
    log = OUT / (phase + '.log')
    with log.open('ab', buffering=0) as stream:
        CHILD = subprocess.Popen(command, cwd=BASE, env=env,
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True)
        while True:
            progress = {}
            path = BASE / 'live-training/train/progress-rank0.json'
            if phase == 'training' and path.exists():
                progress = json.loads(path.read_text())
            status(phase, child_pid=CHILD.pid, command=command,
                log=str(log), latest_progress=progress)
            try:
                code = CHILD.wait(timeout=20)
                break
            except subprocess.TimeoutExpired:
                pass
    if code:
        raise RuntimeError(f'{phase} exited {code}; inspect {log}')


def verify_startup(folder):
    for rank in range(4):
        row = json.loads((folder / f'startup-rank{rank}.json').read_text())
        assert row['fresh_stage2'] and row['fresh_optimizer'] and row['global_step'] == 0
        assert row['world_size'] == 4 and row['microbatch'] == 256 and row['accumulation_steps'] == 2
        assert row['global_batch'] == 2048 and row['training_epochs'] == row['lr_schedule_epochs'] == 100
        assert math.isclose(row['initial_optimizer_lr'], .0005, rel_tol=1.e-12)
        assert row['objective'] == 'native_soft_target_cross_entropy' and row['depth_loss_weights'] is None
        assert row['native_sampling'] == dict(temperature=1., top_k=[512], top_p=[.95])
        assert row['integer_joint_ids'] and row['scalar_events_per_site'] == 5 and row['vocabulary'] == 20481
        assert row['auxiliary_loss_terms'] == 0
        assert row['parameter_count'] == 1393828353
        proof = json.loads((folder / f'loss-pairing-rank{rank}.json').read_text())
        assert proof['passed'] and proof['cuda'] and proof['training_rng_unchanged']


def run():
    lock = (BASE / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUT / 'train/progress-rank0.json').exists():
        raise RuntimeError('Production has already advanced; refusing to restart from zero')
    occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True).strip()
    assert not occupied, f'GPU processes remain: {occupied}'
    for name, expected in json.loads((OUT / 'reproducibility/code-manifest.json').read_text()).items():
        assert digest(BASE / name) == expected, name
    env = dict(os.environ, **json.loads((BASE / 'environment.json').read_text()),
        WANDB_API_KEY=KEY.read_text().strip())
    env.pop('LASER_PRIVATE_SMOKE_DIR', None)
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
        '--nproc-per-node=4', str(BASE / 'train.py'), '--epochs', '100', '--microbatch', '256']
    smoke = BASE / 'private-preflight'
    launch([*command, '--smoke-updates', '1'],
        dict(env, LASER_PRIVATE_SMOKE_DIR=str(smoke)), 'preflight')
    verify_startup(smoke)
    proof = json.loads((smoke / 'verification.json').read_text())
    assert proof['passed'] and proof['global_step'] == proof['all_adam_steps'] == 1
    assert proof['optimizer_state_count'] == 780 and proof['frozen_tokenizer_unchanged']
    for rank in range(4):
        preview = json.loads((smoke / f'preview-verification-rank{rank}.json').read_text())
        assert preview['passed'] and preview['training_rng_unchanged']
    panels = json.loads((OUT / 'smoke/samples/step000001-panels.json').read_text())
    assert panels['sampler'] == dict(temperature=1., top_k=512, top_p=.95)
    write(OUT / 'preflight.json', dict(passed=True, verification=proof,
        production_starts_from_zero=True, preflight_weights_discarded=True,
        sampler_top_k_verified=512, training_rng_unchanged=True, time=time.time()))
    write(ROOT / 'outputs/imagenet-laser-joint20k-active.json', dict(run_id=RUN,
        url=f'https://forge.coreweave.com/wandb/helloimlixin-rutgers/laser/runs/{RUN}',
        base=str(BASE), recipe='Fresh uniform CE, whole integer tokens, top-k512, 100 epochs',
        depth_loss_weights=[1.] * 5, sampling=dict(temperature=1., top_k=512, top_p=.95),
        world_size=4, hardware='4 NVIDIA B200', fresh_stage2=True, updated=time.time()))
    launch(command, env, 'training')
    completed = json.loads((OUT / 'completed.json').read_text())
    assert completed['epochs'] == 100 and completed['global_step'] == 62600
    status('completed', completion=completed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'run'))
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare()
    else:
        run()


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        main()
    except BaseException as error:
        status('stopped' if STOPPING else 'failed', error_type=type(error).__name__, error=str(error))
        raise
