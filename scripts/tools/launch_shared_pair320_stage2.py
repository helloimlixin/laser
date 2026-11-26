#!/usr/bin/env python3
"""Freeze, smoke-test and launch the approved five-depth shared-pair prior."""
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

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.training.rq_imagenet_runtime import write_json, sha256

RUN = 'imagenet-sharedpair24577-depth5-scratch100-4gpu-20261112'
BASE = Path('/tmp/laser-sharedpair24577-d5-4gpu-20261112')
OUT = ROOT / 'outputs' / RUN
CHILD = None


def status(phase, **values):
    write_json(OUT / 'status.json', dict(phase=phase, run_id=RUN, supervisor_pid=os.getpid(),
        child_pid=CHILD.pid if CHILD else None, time=time.time(), **values))


def environment():
    env = dict(os.environ)
    for name in ['WANDB_SERVICE', '_WANDB_SERVICE']: env.pop(name, None)
    if not env.get('WANDB_API_KEY'): raise ValueError('WANDB_API_KEY must be supplied in the environment')
    env.update(WANDB_BASE_URL='https://api.forge.coreweave.com', WANDB_MODE='online',
        WANDB_DISABLE_GIT='true', WANDB_DIR=str(BASE / 'wandb'),
        CUDA_VISIBLE_DEVICES='0,1,2,3', SHARED_PAIR320_BASE=str(BASE),
        TORCH_HOME=str(BASE / 'torch-cache'), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1', NCCL_NVLS_ENABLE='0',
        PYTORCH_ALLOC_CONF='expandable_segments:True',
        LASER_RQ_SAFE_CHECKPOINTS='1', LASER_CHECKPOINT_IMMUTABLE_FILES='1',
        LASER_CHECKPOINT_STAGING_DIR=str(BASE / 'staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(BASE / 'checkpoint-cache'),
        LASER_PRIVATE_SMOKE_DIR=str(BASE / 'smoke'),
        LASER_WANDB_CHECKPOINT_UPLOADS='1')
    return env


def freeze():
    report = json.loads((OUT / 'calibration/report.json').read_text())
    assert report['completed'] and report['passed']
    assert report['codebook_sha256'] == sha256(BASE / 'inputs/joint-codebook.pt')
    recipe = yaml.safe_load((BASE / 'inputs/recipe.yaml').read_text())
    recipe['logging'] = dict(wandb=True, create_workspace=False)
    recipe['targets']['temperature'] = report['proposed_temperatures']
    assert recipe['tokenizer']['depth'] == 5 and recipe['tokenizer']['vocab_size'] == 24577
    assert recipe['tokenizer']['shared_codebook_across_depths']
    (BASE / 'inputs/recipe.yaml').write_text(yaml.safe_dump(recipe, sort_keys=False))
    (ROOT / 'configs/stage2' / (RUN + '.yaml')).write_text(yaml.safe_dump(recipe, sort_keys=False))
    source = BASE / 'source/src'
    if source.exists(): raise RuntimeError('Refusing to overwrite an already frozen training source')
    shutil.copytree(ROOT / 'src', source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    upstream = BASE / 'upstream'
    assert upstream.is_symlink()
    upstream.unlink()
    shutil.copytree(ROOT / 'third_party/rq-vae-transformer', upstream,
        ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc'))
    for name in ['amp_replay.py', 'rq_checkpoint_store.py']:
        shutil.copyfile(ROOT / 'runtime' / name, BASE / 'runtime' / name)
    shutil.copyfile(ROOT / 'scripts/tools/train_shared_pair320.py', BASE / 'train.py')
    repro = OUT / 'reproducibility'
    repro.mkdir(exist_ok=True)
    for name in ['recipe.yaml', 'stage2-training-config.yaml', 'stage2-archive-config.yaml',
                 'execution.json', 'joint-codebook.pt', 'target-audit.json', 'imagenet_256_train.npz']:
        shutil.copyfile(BASE / 'inputs' / name, repro / name)
    shutil.copyfile(OUT / 'calibration/report.json', repro / 'teacher-calibration.json')
    shutil.copyfile(ROOT / 'outputs/imagenet-shared-pair24577-improve-20261112/full-rfid.json', repro / 'tokenizer-rfid50k.json')
    files = [BASE / 'train.py']
    for directory in ['source', 'upstream', 'runtime']:
        files.extend(path for path in (BASE / directory).rglob('*.py') if '__pycache__' not in path.parts)
    hashes = {str(path.relative_to(BASE)): sha256(path) for path in files}
    write_json(repro / 'code-manifest.json', hashes)
    with zipfile.ZipFile(repro / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files: archive.write(path, path.relative_to(BASE))
    write_json(repro / 'frozen-inputs.json', dict(stage1_checkpoint_sha256=sha256(BASE / 'inputs/stage1-model.pt'),
        stage1_persistent_source=str(ROOT / 'outputs/imagenet-laser-rq-minimal-rfid421-scratch20-20261014/reproducibility/stage1-model.pt'),
        codebook_sha256=sha256(BASE / 'inputs/joint-codebook.pt'), world_size=4,
        full_state_checkpoints=True, fresh_stage2=True, fresh_optimizer=True))


def run_child(smoke):
    global CHILD
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4', str(BASE / 'train.py')]
    if smoke: command += ['--smoke-updates', '1', '--smoke-split', 'val']
    log_path = OUT / ('smoke.log' if smoke else 'training.log')
    with log_path.open('ab', buffering=0) as log:
        CHILD = subprocess.Popen(command, env=environment(), cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(OUT / ('smoke-launch.json' if smoke else 'training-launch.json'),
            dict(pid=CHILD.pid, command=command, fresh_optimizer=True, fresh_stage2=True))
        while CHILD.poll() is None:
            progress_path = BASE / ('smoke/progress-rank0.json' if smoke else 'live-training/train/progress-rank0.json')
            progress = json.loads(progress_path.read_text()) if progress_path.exists() else None
            status('smoke_testing' if smoke else 'training', progress=progress)
            time.sleep(10)
        if CHILD.returncode:
            raise RuntimeError(f"{'Smoke' if smoke else 'Training'} failed: {CHILD.returncode}; inspect {log_path}")


def stop(signum, frame):
    if CHILD and CHILD.poll() is None: os.killpg(CHILD.pid, signal.SIGTERM)
    status('stopped_by_signal', signal=signum)
    raise SystemExit(128 + signum)


def main():
    environment()
    for sig in [signal.SIGTERM, signal.SIGINT]: signal.signal(sig, stop)
    status('freezing_runtime')
    freeze()
    run_child(smoke=True)
    verification = json.loads((BASE / 'smoke/verification.json').read_text())
    assert verification['passed'] and verification['global_step'] == 1
    assert verification['frozen_tokenizer_unchanged']
    shutil.copyfile(BASE / 'smoke/verification.json', OUT / 'smoke-verification.json')
    status('waiting_for_verified_training_archive')
    while not (BASE / 'data/ready.json').exists(): time.sleep(10)
    data = json.loads((BASE / 'data/ready.json').read_text())
    assert data['ready'] and data['images'] == 1281167
    run_child(smoke=False)
    status('completed')


if __name__ == '__main__':
    try: main()
    except Exception as error:
        status('failed', error_type=type(error).__name__, error=str(error))
        raise
