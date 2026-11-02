#!/usr/bin/env python3
"""Validate, freeze and launch the requested fresh single-integer LASER run."""
import argparse
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
import zipfile

ROOT = Path('/workspace/Projects/laser')
RUN = 'imagenet-laser-pairid256k-grid512-scratch100-4h100-20261017'
OUT = ROOT/'outputs'/RUN
BASE = Path('/tmp/laser-pairid256k-grid512-imagenet-20261017')
SOURCE = Path('/tmp/laser-compound512-origlr-imagenet-20261016-r3')
STUDY = ROOT/'outputs/imagenet-laser-catalog512-vocabulary-20261017'
RECIPE = ROOT/'configs/stage2/imagenet-laser-catalog256k-grid512-scratch100-20261017.yaml'
sys.path.insert(0, str(ROOT))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name, **values):
    value = dict(values, time=time.time())
    tmp = OUT/(name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(OUT/name)
    print(json.dumps(value), flush=True)


def prepare():
    import yaml
    from runtime.rq_imagenet_entry import freeze_shared_runtime
    from src.training.rq_sample_panels import CLASSES
    if (OUT/'prepared.json').exists():
        return
    assert not (BASE/'inputs/recipe.yaml').exists(), 'Preserve any partially prepared runtime'
    report = json.loads((STUDY/'catalog262144-full-rfid.json').read_text())
    assert report['completed'] and report['full_50000_reconstruction_fid']
    assert report['real_images'] == report['reconstructed_images'] == 50000
    assert report['measured_rfid'] <= 4.4, 'Catalog reconstruction drift exceeds the selected gate'
    recipe = yaml.safe_load(RECIPE.read_text())
    recipe['tokenizer']['measured_rfid'] = report['measured_rfid']
    RECIPE.write_text(yaml.safe_dump(recipe, sort_keys=False))
    catalog = STUDY/'catalog262144-grid512.pt'
    assert sha(catalog) == report['catalog_sha256'] == recipe['tokenizer']['catalog_sha256']
    assert sha(SOURCE/'inputs/stage1-model.pt') == recipe['tokenizer']['sha256']
    assert sha(SOURCE/'inputs/imagenet_256_train.npz') == recipe['evaluation']['reference_sha256']
    assert recipe['epochs'] == recipe['lr_schedule_epochs'] == 100 and recipe['global_batch'] == 2048
    (BASE/'inputs').mkdir(parents=True)
    (OUT/'reproducibility').mkdir(exist_ok=True)
    freeze_shared_runtime(BASE, ROOT, include_stochastic_teacher=False)
    for name in ['upstream', 'data', 'torch-cache']:
        (BASE/name).symlink_to((SOURCE/name).resolve(), target_is_directory=True)
    for name in ['stage1-model.pt', 'imagenet_256_train.npz']:
        (BASE/'inputs'/name).symlink_to((SOURCE/'inputs'/name).resolve())
    for name in ['stage2-training-config.yaml', 'stage2-archive-config.yaml', 'rq-upstream-core.json']:
        shutil.copyfile(SOURCE/'inputs'/name, BASE/'inputs'/name)
    shutil.copyfile(catalog, BASE/'inputs/pair-catalog.pt')
    shutil.copyfile(RECIPE, BASE/'inputs/recipe.yaml')
    config = yaml.safe_load((BASE/'inputs/stage2-training-config.yaml').read_text())
    config['loss'] = dict(type='single_global_pair_cross_entropy', coefficient_temperature=.01125,
        stochastic_codes=True, auxiliary_loss_terms=0)
    config['experiment'].update(epochs=100, batch_size=32, total_batch_size=2048,
        save_ckpt_freq=1, test_freq=2)
    assert config['optimizer']['init_lr'] == .0005 and config['optimizer']['warmup']['epoch'] == 0
    (BASE/'inputs/stage2-training-config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    (BASE/'train.py').write_text("from runtime.rq_imagenet_entry import launch\nif __name__=='__main__':\n    launch('laser_catalog_integer')\n")
    shutil.copyfile(ROOT/'scripts/tools/verify_laser_catalog_smoke.py', BASE/'verify.py')
    shutil.copyfile(ROOT/'scripts/tools/configure_rq_sample_panels.py', BASE/'configure_samples.py')
    environment = dict(LASER_RQ_BASE=str(BASE), LASER_RQ_SAFE_CHECKPOINTS='1',
        LASER_CHECKPOINT_STAGING_DIR=str(BASE/'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(BASE/'checkpoint-cache'),
        LASER_CHECKPOINT_IMMUTABLE_FILES='1', TORCH_HOME=str(BASE/'torch-cache'),
        WANDB_DIR=str(BASE/'wandb'), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        PYTHONUNBUFFERED='1', NCCL_NVLS_ENABLE='0', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    (BASE/'environment.json').write_text(json.dumps(environment, indent=2)+'\n')
    files = [BASE/name for name in ['train.py', 'verify.py', 'configure_samples.py', 'environment.json']]
    for directory in ['source', 'runtime', 'upstream']:
        files.extend((BASE/directory).rglob('*.py'))
    files += list((BASE/'inputs').glob('*.yaml')) + list((BASE/'inputs').glob('*.json'))
    files.append(BASE/'inputs/pair-catalog.pt')
    manifest = {str(p.relative_to(BASE)):sha(p) for p in files}
    write('reproducibility/code-manifest.json', files=manifest)
    for p in (BASE/'inputs').glob('*.yaml'):
        shutil.copyfile(p, OUT/'reproducibility'/p.name)
    for name in ['fit-report.json', 'catalog262144-full-rfid.json', 'source-manifest.json']:
        shutil.copyfile(STUDY/name, OUT/'reproducibility'/name)
    shutil.copyfile('/tmp/laser-catalog-integer-tests.xml', OUT/'reproducibility/tests.xml')
    with zipfile.ZipFile(OUT/'reproducibility/source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in files:
            archive.write(p, p.relative_to(BASE))
        for p in [Path(__file__), ROOT/'tests/test_laser_catalog_integer.py']:
            archive.write(p, p.relative_to(ROOT))
    write('prepared.json', prepared=True, training_launched=False, run_id=RUN,
        fresh_stage2=True, fresh_optimizer=True, stage2_checkpoint_loaded=False,
        initial_global_step=0, recipe=recipe, base=str(BASE), world_size=4,
        global_batch=2048, optimizer_updates_per_epoch=626, cosine_total_updates=62600,
        stop_after_updates=62600, measured_reconstruction_rfid=report['measured_rfid'],
        source_reconstruction_rfid=report['source_full_rfid'], catalog_sha256=sha(catalog),
        sampling=dict(temperature=.9, top_k=0, top_p=.95, whole_pair=True),
        sample_classes=[dict(name=name, imagenet_id=label) for name, label in CLASSES],
        code_manifest_sha256=sha(OUT/'reproducibility/code-manifest.json'))


def run():
    lock = (OUT/'controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    child, panel = None, None
    def stop(*_):
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
        if panel is not None and panel.poll() is None:
            panel.terminate()
        raise KeyboardInterrupt('Integer catalog controller interrupted')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    deadline = time.monotonic()+2700
    write('launch-status.json', phase='waiting_for_full_reconstruction_validation', pid=os.getpid())
    while not (STUDY/'catalog262144-full-rfid.json').exists():
        if time.monotonic() > deadline:
            raise TimeoutError('Full reconstruction check has not completed within45min')
        time.sleep(5)
    prepare()
    while True:
        p = OUT/'source-handoff-status.json'
        if p.exists() and json.loads(p.read_text())['phase'] == 'source_stopped_with_verified_checkpoint':
            break
        if time.monotonic() > deadline:
            raise TimeoutError('Source checkpoint handoff did not finish')
        time.sleep(3)
    assert json.loads((OUT/'preserved-source-checkpoint.json').read_text())['passed']
    write('launch-status.json', phase='waiting_for_gpu_release', pid=os.getpid())
    while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        if time.monotonic() > deadline:
            raise TimeoutError('GPUs did not become available')
        time.sleep(3)
    for rel, expected in json.loads((OUT/'reproducibility/code-manifest.json').read_text())['files'].items():
        assert sha(BASE/rel) == expected, rel
    env = dict(os.environ, **json.loads((BASE/'environment.json').read_text()))
    env['WANDB_API_KEY'] = Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
    smoke_env = dict(env, LASER_PRIVATE_SMOKE_DIR=str(BASE/'private-smoke'))
    distributed = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=4']
    def launch(command, name, execution_env):
        nonlocal child
        with (OUT/(name+'.log')).open('ab', buffering=0) as log:
            child = subprocess.Popen(command, env=execution_env, cwd=BASE,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write('launch-status.json', phase=name, pid=os.getpid(), child_pid=child.pid)
            result = child.wait()
        if result:
            write('launch-status.json', phase=name+'_failed', returncode=result)
            raise RuntimeError(f'{name} failed with exit code{result}')
    microbatch = 32
    launch([*distributed, str(BASE/'train.py'), '--microbatch', str(microbatch), '--smoke-updates', '2'],
           'gpu_preflight', smoke_env)
    launch([sys.executable, str(BASE/'verify.py')], 'checkpoint_verification', smoke_env)
    assert json.loads((OUT/'smoke-reload-verification.json').read_text())['passed']
    with (OUT/'sample-workspace.log').open('ab', buffering=0) as log:
        panel = subprocess.Popen([sys.executable, str(BASE/'configure_samples.py'), '--run', RUN, '--watch'],
            env=env, cwd=BASE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    plan = json.loads((OUT/'prepared.json').read_text())
    plan.update(training_launched=True, preflight_passed=True, production_starts_from_zero=True,
        actual_microbatch=microbatch, accumulation_steps=16, launched_at=time.time())
    write('prepared.json', **plan)
    launch([*distributed, str(BASE/'train.py'), '--microbatch', str(microbatch), '--epochs', '100'], 'training', env)
    write('launch-status.json', phase='completed', epochs=100, updates=62600)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run'])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        prepare() if args.action == 'prepare' else run()
    except BaseException as error:
        write('launch-status.json', phase='failed', error=repr(error))
        raise
