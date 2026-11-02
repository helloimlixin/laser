#!/usr/bin/env python3
"""Launch the shared RQ backbone on deterministic sparse support/coefficient tokens."""
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
sys.path.insert(0, str(ROOT))
from runtime.rq_imagenet_entry import freeze_shared_runtime

BASE = Path('/tmp/laser-rq-hard-ar-imagenet-20261015')
SOURCE_BASE = Path('/tmp/laser-rq-minimal-imagenet-20261014')
RUN = 'imagenet-laser-rq-hard-ar-rfid421-scratch20-20261015'
OUT = ROOT/'outputs'/RUN
RECIPE_PATH = ROOT/'configs/stage2/imagenet-laser-rq-hard-ar-20ep-20261015.yaml'
NEW_WANDB_RUN = False


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str)+'\n')
    temporary.replace(path)


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def training_microbatch(memory_mib, requested):
    capacity = 64 if min(memory_mib) >= 75000 else 32 if min(memory_mib) >= 45000 else 16
    return min(capacity, int(requested))


def prepare():
    import yaml
    if (OUT/'prepared.json').exists():
        raise RuntimeError('Already prepared; use run')
    recipe_path = RECIPE_PATH
    recipe = yaml.safe_load(recipe_path.read_text())
    assert recipe['run_id'] == RUN and recipe['target_mode'] == 'hard_sparse_tokens'
    assert 'teacher' not in recipe and recipe['auxiliary_loss_terms'] == 0
    assert json.loads((SOURCE_BASE/'data/ready.json').read_text())['ready']
    BASE.mkdir(parents=True, exist_ok=True)
    (BASE/'inputs').mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    freeze_shared_runtime(BASE, ROOT, include_stochastic_teacher=False)
    shutil.copytree(SOURCE_BASE/'upstream', BASE/'upstream',
        ignore=shutil.ignore_patterns('__pycache__', '.git', '*.pyc'))
    for original, name in (('train_laser_rq_minimal_imagenet.py', 'train.py'),
                            ('verify_laser_rq_minimal_checkpoint.py', 'verify.py')):
        shutil.copyfile(ROOT/'scripts/tools'/original, BASE/name)
    # This independent reference is used only by the private GPU verification.
    reference = BASE/'validation/stage1_reference'
    reference.mkdir(parents=True)
    (reference/'__init__.py').touch()
    for name in ('dictionary_learner.py', 'bottleneck_utils.py'):
        shutil.copyfile(ROOT/'src/models'/name, reference/name)
    shutil.copyfile(recipe_path, BASE/'inputs/recipe.yaml')
    shutil.copyfile(ROOT/'configs/rq-upstream-core.json', BASE/'inputs/rq-upstream-core.json')
    source = SOURCE_BASE/'inputs/stage1-model.pt'
    assert digest(source) == recipe['tokenizer']['sha256']
    os.link(source, BASE/'inputs/stage1-model.pt')
    for name in ('stage2-training-config.yaml', 'stage2-archive-config.yaml', 'imagenet_256_train.npz'):
        shutil.copyfile(SOURCE_BASE/'inputs'/name, BASE/'inputs'/name)
    training = yaml.safe_load((BASE/'inputs/stage2-training-config.yaml').read_text())
    training['loss'] = dict(type='cross_entropy', stochastic_codes=False)
    (BASE/'inputs/stage2-training-config.yaml').write_text(yaml.safe_dump(training, sort_keys=False))
    assert digest(BASE/'inputs/imagenet_256_train.npz') == recipe['evaluation']['reference_sha256']
    for name in ('data', 'torch-cache'):
        (BASE/name).symlink_to(SOURCE_BASE/name, target_is_directory=True)
    environment = dict(LASER_RQ_BASE=str(BASE), LASER_CHECKPOINT_STAGING_DIR=str(BASE/'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(BASE/'checkpoint-cache'), LASER_CHECKPOINT_IMMUTABLE_FILES='1',
        TORCH_HOME=str(BASE/'torch-cache'), WANDB_DIR=str(BASE/'wandb'),
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    write(BASE/'environment.json', environment)
    reproducibility = OUT/'reproducibility'
    reproducibility.mkdir(exist_ok=True)
    for path in (BASE/'inputs').iterdir():
        if path.suffix in ('.yaml', '.json'):
            shutil.copyfile(path, reproducibility/path.name)
    shutil.copyfile(SOURCE_BASE/'data/ready.json', reproducibility/'data-verification.json')
    files = [BASE/'train.py', BASE/'verify.py', BASE/'environment.json']
    for folder in ('source', 'runtime', 'upstream', 'validation'):
        files.extend((BASE/folder).rglob('*.py'))
    files.extend((BASE/'inputs').glob('*.yaml'))
    files.append(BASE/'inputs/rq-upstream-core.json')
    manifest = {str(path.relative_to(BASE)): digest(path) for path in files}
    write(reproducibility/'code-manifest.json', manifest)
    with zipfile.ZipFile(reproducibility/'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(BASE))
        archive.write(Path(__file__), 'scripts/tools/'+Path(__file__).name)
        for name in ('test_laser_hard_ar.py', 'test_shared_rq_codebase.py', 'test_amp_replay.py'):
            archive.write(ROOT/'tests'/name, 'tests/'+name)
    plan = dict(prepared=True, run_id=RUN, recipe=recipe, fresh_stage2=True, fresh_optimizer=True,
        train_images=1281167, global_batch=2048, updates_per_epoch=626,
        stop_updates=626*int(recipe['epochs']),
        cosine_total_updates=626*int(recipe['lr_schedule_epochs']), sampled_trajectories=0,
        source_manifest_sha256=digest(reproducibility/'code-manifest.json'), prepared_at=time.time())
    write(OUT/'prepared.json', plan)
    print(json.dumps(plan), flush=True)


def run(resume=False):
    prepared = json.loads((OUT/'prepared.json').read_text())
    assert prepared['prepared'] and json.loads((OUT/'cpu-verification.json').read_text())['passed']
    lock = (BASE/'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    if active.strip():
        raise RuntimeError('GPUs are occupied; stop the superseded run before launching')
    manifest = json.loads((OUT/'reproducibility/code-manifest.json').read_text())
    for name, expected in manifest.items():
        assert digest(BASE/name) == expected, name
    assert digest(BASE/'inputs/stage1-model.pt') == prepared['recipe']['tokenizer']['sha256']
    sizes = [int(value) for value in subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], text=True).splitlines()]
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    world = len(sizes) if visible is None else len(visible.split(','))
    microbatch = training_microbatch(sizes, prepared['recipe'].get('microbatch', 64))
    assert world > 0
    environment = dict(os.environ, **json.loads((BASE/'environment.json').read_text()))
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
        f'--nproc-per-node={world}', str(BASE/'train.py'), '--epochs',
        str(int(prepared['recipe']['epochs'])), '--microbatch', str(microbatch)]
    if resume:
        checkpoint = OUT/'checkpoints/last.pt'
        assert checkpoint.is_file()
        command += ['--resume', str(checkpoint)]
        if NEW_WANDB_RUN and not (OUT/'wandb.json').exists():
            command += ['--new-wandb-run']
    status = dict(pid=os.getpid(), run_id=RUN, resuming=resume, world_size=world,
        microbatch=microbatch, accumulation_steps=math.ceil(2048/(world*microbatch)),
        global_batch=2048, sampled_trajectories=0, auxiliary_loss_terms=0)
    child = None
    cancelled = False

    def stop(_signum, _frame):
        nonlocal cancelled
        cancelled = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def launch(arguments, phase):
        nonlocal child
        with (OUT/f'{phase}.log').open('a') as log:
            child = subprocess.Popen(arguments, cwd=BASE, env=environment, stdout=log, stderr=subprocess.STDOUT)
            status.update(phase=phase, child_pid=child.pid, started_at=time.time())
            write(OUT/'launch-status.json', status)
            result = child.wait()
        if result:
            status.update(phase='cancelled' if cancelled else 'stopped_or_failed',
                          returncode=result, finished_at=time.time())
            write(OUT/'launch-status.json', status)
            raise RuntimeError(f'{phase} returned {result}; inspect {OUT/phase}.log')

    if not resume:
        launch([*command, '--smoke-updates', '2'], 'gpu_preflight')
        if cancelled:
            return 130
        launch([sys.executable, str(BASE/'verify.py')], 'checkpoint_verification')
    verification = OUT/'continuation-verification.json' if resume and (OUT/'continuation-verification.json').exists() else OUT/'smoke-reload-verification.json'
    assert json.loads(verification.read_text())['passed']
    if cancelled:
        return 130
    launch(command, 'train')
    status.update(phase='completed', returncode=0, finished_at=time.time())
    write(OUT/'launch-status.json', status)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--resume', action='store_true')
    arguments = parser.parse_args()
    if arguments.action == 'prepare':
        prepare()
    else:
        sys.exit(run(arguments.resume))
