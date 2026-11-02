#!/usr/bin/env python3
"""Prepare and supervise a fresh, hardware-scaled duplicate of the Oct 8 run."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = Path(os.environ.get('LASER_PROJECT_ROOT', '/workspace/Projects/laser'))
ORIGINAL = ROOT / 'outputs/imagenet-rfid421-tiny-jitter-fresh-8h100-20261008'
SOURCE_RUN = 'helloimlixin-rutgers/laser/' + ORIGINAL.name
RUN_ID = 'imagenet-rfid421-tiny-jitter-fresh-4h100-20261014'
BASE = Path('/tmp/laser-imagenet-duplicate-20261014')
OUT = ROOT / 'outputs' / RUN_ID
KEY = Path('/root/.config/laser/imagenet-stage2-wandb.key')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def environment():
    plan = json.loads((OUT / 'plan.json').read_text())
    return dict(os.environ,
        WANDB_API_KEY=KEY.read_text().strip(),
        WANDB_DIR=str(BASE / 'wandb'), WANDB_CACHE_DIR=str(BASE / 'wandb-cache'),
        WANDB_DATA_DIR=str(BASE / 'wandb-data'), WANDB_ARTIFACT_DIR=str(BASE / 'wandb-artifacts'),
        WANDB_DISABLE_GIT='true',
        TORCH_HOME=str(BASE / 'torch-cache'), TORCHINDUCTOR_CACHE_DIR=str(BASE / 'inductor-cache'),
        TORCHINDUCTOR_COMPILE_THREADS='4', MPLCONFIGDIR=str(BASE / 'matplotlib'),
        PYTHONPATH=str(BASE / 'source') + ':' + str(BASE / 'source/runtime'),
        CUDA_VISIBLE_DEVICES=','.join(str(i) for i in range(plan['world_size'])),
        LASER_EXPECTED_WORLD_SIZE=str(plan['world_size']),
        LASER_ACCUMULATION=str(plan['accumulation_steps']),
        LASER_PAIR_MEMORY_BASE=str(BASE), LASER_PAIR_MEMORY_OUTPUT=str(OUT),
        LASER_CHECKPOINT_STAGING_DIR=str(BASE / 'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(BASE / 'checkpoint-upload-cache'),
        LASER_CHECKPOINT_IMMUTABLE_FILES='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1',
        NCCL_NVLS_ENABLE='0', TORCH_NCCL_ASYNC_ERROR_HANDLING='1')


def prepare(microbatch=128):
    import torch
    import yaml
    world = torch.cuda.device_count()
    assert world == 4, 'This named run is prepared for the four available H100s'
    assert all(torch.cuda.get_device_properties(i).total_memory > 79 * 2**30 for i in range(world))
    assert 2048 % (world * microbatch) == 0
    accumulation = 2048 // (world * microbatch)
    BASE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'train/checkpoints/last.pt').exists(), 'Refuse to overwrite training'
    source = BASE / 'source'
    if not source.exists():
        with tarfile.open(ROOT / 'outputs/imagenet-rfid421-classcond-8h100-20261007/frozen-source.tar.gz') as archive:
            archive.extractall(BASE, filter='data')
    original_manifest = json.loads((ORIGINAL / 'source-manifest.json').read_text())
    for relative, expected in original_manifest['sha256'].items():
        assert digest(source / relative) == expected, relative
    for name in ('coefficient_jitter.py', 'generation_logging.py'):
        assert digest(ORIGINAL / name) == original_manifest['helper_sha256'][name]
        shutil.copyfile(ORIGINAL / name, source / 'src/training' / name)
    entry = (ORIGINAL / 'entry.py').read_text()
    assert digest(ORIGINAL / 'entry.py') == original_manifest['entry_sha256']
    before = "assert world == 8 and backend == 'ddp'"
    assert entry.count(before) == 1
    entry = entry.replace(before, "assert world == int(os.environ['LASER_EXPECTED_WORLD_SIZE']) and backend == 'ddp'")
    # Record run lineage and baseline targets without loading any prior weights.
    before = "'stage1_reconstruction_fid': 4.21, 'noise_calibration': CALIBRATION}"
    assert entry.count(before) == 1
    entry = entry.replace(before,
        "'stage1_reconstruction_fid': 4.21, 'noise_calibration': CALIBRATION, "
        f"'duplicate_of': {SOURCE_RUN!r}, 'rq_baseline_fid': 11.56, "
        "'rq_baseline_inception_score': 112.4, 'stage2_pretrained_weights_loaded': False}")
    before = "('noise-calibration.json', 'plan.json', 'source-manifest.json', 'preflight-verification.json')"
    assert entry.count(before) == 1
    entry = entry.replace(before,
        "('noise-calibration.json', 'plan.json', 'source-manifest.json', 'preflight-verification.json', "
        "'asset-verification.json', 'train.yaml', 'entry.py', 'recipe-audit.json', "
        "'evaluator-verification.json', 'batch-layout-verification.json', 'smoke-verification.json', "
        "'frozen-source.tar.gz', 'duplicate_imagenet_tiny_jitter.py')")
    compile(entry, 'entry.py', 'exec')
    (BASE / 'entry.py').write_text(entry)
    (OUT / 'entry.py').write_text(entry)
    shutil.copyfile(ORIGINAL / 'noise-calibration.json', OUT / 'noise-calibration.json')
    inputs = BASE / 'inputs'
    inputs.mkdir(exist_ok=True)
    assets = ROOT / 'outputs/imagenet-rfid421-combination-soft-20260927/assets'
    specs = [
        (assets / 'laser-tokenizer.pt', 'stage1-tokenizer.pt', 'dd28db9306d526bdc9fbf8016403e106c4f317fd8f5c04792f0953e53310fdab'),
        (assets / 'imagenet_256_train.npz', 'imagenet_256_train.npz', '3f9c92d15755e76ec312964a819e3e19b9cf3aadc618a7ae99f5e8aa96501260'),
        (ROOT / 'outputs/imagenet-rfid421-ffhq-compound-1400m-20260922/compound-cache.pt', 'compound-cache.pt', 'cd0cd8517a69a3e232575edacac6a61be8fedf9fe382f3f4890baa946e28f300'),
    ]
    def verify_asset(spec):
        original, name, expected = spec
        target = inputs / name
        if not target.exists():
            shutil.copyfile(original, target)
        assert digest(target) == expected, name
        return dict(name=name, source=str(original), bytes=target.stat().st_size, sha256=expected)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        verified = list(pool.map(verify_asset, specs))
    cache = torch.load(inputs / 'compound-cache.pt', weights_only=True, mmap=True, map_location='cpu')
    assert cache['atoms'].shape == cache['coeffs'].shape == (1281167, 8, 8, 4)
    assert cache['labels'].shape == (1281167,) and cache['labels'].unique().numel() == 1000
    assert cache['meta']['stage1_checkpoint_sha256'] == specs[0][2]
    assert cache['meta']['clip_coefficients'] is False
    write(OUT / 'asset-verification.json', dict(passed=True, files=verified, cache_meta=cache['meta']))
    recipe = yaml.safe_load((ORIGINAL / 'train.yaml').read_text())
    recipe['options'].update(
        checkpoint=str(inputs / 'stage1-tokenizer.pt'), token_cache=str(inputs / 'compound-cache.pt'),
        data='/workspace/Projects/data/imagenet', output=str(OUT / 'train'),
        checkpoint_dir=str(OUT / 'train/checkpoints'),
        fid_reference_stats=str(inputs / 'imagenet_256_train.npz'),
        batch_size=microbatch, total_batch_size=2048,
        epochs=100, lr_schedule_epochs=100, resume=False, resume_checkpoint=None,
        init_stage2_checkpoint=None, max_optimizer_steps=0,
        wandb_id=RUN_ID, wandb_name='ImageNet rFID4.21 | fresh duplicate, 1% bin jitter | 4 H100')
    text = yaml.safe_dump(recipe, sort_keys=False)
    for target in (BASE / 'train.yaml', OUT / 'train.yaml', ROOT / 'configs/stage2' / (RUN_ID + '.yaml')):
        target.write_text(text)
    for name in ('wandb', 'wandb-cache', 'wandb-data', 'wandb-artifacts', 'checkpoint-staging',
                 'checkpoint-upload-cache', 'inductor-cache', 'matplotlib', 'torch-cache/hub/checkpoints'):
        (BASE / name).mkdir(parents=True, exist_ok=True)
    plan = dict(status='prepared', duplicate_of=SOURCE_RUN, stage2_initialization='scratch', start_step=0,
        world_size=world, microbatch=microbatch, accumulation_steps=accumulation, global_batch=2048,
        epochs=100, scheduler='fresh cosine to zero, no warmup', initial_lr=.0005,
        steps_per_epoch=626, scheduler_total_steps=62600, training_images=1281167,
        stage1_reconstruction_fid=4.210914134979248,
        train_data_mode='original immutable center-crop token cache',
        sigma_bins=.01, cap_bins=.025, fid_samples=50000, fid_every=2, first_fid_epoch=1,
        fid_reference='official ImageNet-256 training statistics', rejection_sampling=False,
        published_baseline_fid=11.56, published_baseline_inception_score=112.4,
        source_run_last_local_observed_step=12490, source_run_last_fid=39.26407722679767,
        wandb_id=RUN_ID, output=str(OUT), work=str(BASE),
        original_archive='https://twg.kakaocdn.net/brainrepo/models/RQVAE/f5cf4e5f3f0b5088d52cbb5e85c1077f/imagenet_1.4B.tar.gz',
        bitwise_original_trajectory_guaranteed=False)
    write(OUT / 'plan.json', plan)
    write(OUT / 'source-manifest.json', dict(original=original_manifest,
        original_core_source_hashes_verified=True, entry_sha256=digest(BASE / 'entry.py'),
        adaptations=['GPU-count assertion: eight to available four', 'new run identity and lineage metadata',
                     'upload reproducibility and validation receipts'],
        pretrained_transformer_weights_loaded=False))
    if (BASE / 'source-run.json').exists(): shutil.copyfile(BASE / 'source-run.json', OUT / 'source-run.json')
    if (BASE / 'reference').exists():
        (OUT / 'reference').mkdir(exist_ok=True)
        for path in (BASE / 'reference').iterdir():
            if path.is_file(): shutil.copyfile(path, OUT / 'reference' / path.name)
    shutil.copyfile(Path(__file__), OUT / 'duplicate_imagenet_tiny_jitter.py')
    with tarfile.open(OUT / 'frozen-source.tar.gz', 'w:gz') as archive:
        archive.add(source, arcname='source', filter=lambda info: None if '__pycache__' in info.name else info)
    print(json.dumps(plan, indent=2), flush=True)


def launch(smoke=False):
    import yaml
    lock = (BASE / 'launch.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert json.loads((OUT / 'preflight-verification.json').read_text())['passed']
    env = environment()
    config = BASE / 'train.yaml'
    phase = 'smoke' if smoke else 'train'
    if smoke:
        recipe = yaml.safe_load(config.read_text())
        recipe['options'].update(output=str(OUT / 'smoke'), checkpoint_dir=str(OUT / 'smoke/checkpoints'),
            max_optimizer_steps=4, wandb_mode='disabled', upload_checkpoints=False,
            save_step_freq=0, sample_grid_every=0, fid_every=0)
        config = BASE / 'smoke.yaml'
        config.write_text(yaml.safe_dump(recipe, sort_keys=False))
        env['LASER_PAIR_MEMORY_OUTPUT'] = str(OUT / 'smoke-evidence')
        evidence = Path(env['LASER_PAIR_MEMORY_OUTPUT'])
        evidence.mkdir(exist_ok=True)
        shutil.copyfile(OUT / 'noise-calibration.json', evidence / 'noise-calibration.json')
    else:
        assert json.loads((OUT / 'smoke-verification.json').read_text())['passed']
        assert not (OUT / 'train/checkpoints/last.pt').exists(), 'Fresh launch refuses existing weights'
    world = json.loads((OUT / 'plan.json').read_text())['world_size']
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               f'--nproc-per-node={world}', str(BASE / 'entry.py'), '--config', str(config)]
    status = dict(phase=phase, supervisor_pid=os.getpid(), start_step=0, wandb_id=RUN_ID,
                  log=str(OUT / (phase + '.log')), command=command, time=time.time())
    with (OUT / (phase + '.log')).open('a') as log:
        process = subprocess.Popen(command, cwd=BASE / 'source', env=env, stdout=log, stderr=subprocess.STDOUT)
        status['training_pid'] = process.pid
        write(OUT / 'launch-status.json', status)
        print(json.dumps(status), flush=True)
        result = process.wait()
    status.update(phase=phase + ('_completed' if result == 0 else '_failed'), returncode=result, time=time.time())
    write(OUT / 'launch-status.json', status)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'smoke', 'train'])
    parser.add_argument('--microbatch', type=int, default=128)
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare(args.microbatch)
    else:
        raise SystemExit(launch(smoke=args.phase == 'smoke'))
