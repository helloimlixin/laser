"""Run verified data indexing, full tokenizer audit, a discarded canary, then training."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = Path('/workspace/Projects/laser')
BASE = Path('/tmp/laser-imagenet-compound512-audit-20261026')
SOURCE = BASE/'source'
RUN = 'imagenet-compound512-audited-8a100-20261026'
OUT = ROOT/'outputs'/RUN
CONFIG = 'configs/stage2/imagenet-compound512-audited-8a100-20261026.yaml'


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, default=str)+'\n')
    temporary.replace(path)


def status(phase, **payload):
    record = dict(phase=phase, supervisor_pid=os.getpid(), run_id=RUN,
                  updated_unix=time.time(), **payload)
    write(BASE/'records/status.json', record)
    # Only the supervisor touches persistent storage, outside GPU collectives.
    write(OUT/'status.json', record)
    print(json.dumps(record), flush=True)


def environment():
    env = dict(os.environ, LASER_RUN_BASE=str(BASE),
        PYTHONPATH=':'.join(map(str, [SOURCE, SOURCE/'runtime', SOURCE/'scripts/tools'])),
        PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='4', LASER_ACCUMULATION='8', LASER_ENCODER_BATCH='16',
        LASER_DATA_WORKERS='4', LASER_COMPILE_BLOCKS='0',
        TORCH_HOME=str(BASE/'torch-cache'), WANDB_DIR=str(BASE/'wandb'),
        WANDB_CACHE_DIR=str(BASE/'wandb-cache'), WANDB_DATA_DIR=str(BASE/'wandb-data'),
        PYTORCH_ALLOC_CONF='expandable_segments:True', NCCL_NVLS_ENABLE='0', MPLBACKEND='Agg')
    env['WANDB_API_KEY'] = Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
    return env


def execute(phase, command):
    log = BASE/(phase+'.log')
    with log.open('a') as handle:
        child = subprocess.Popen(command, cwd=SOURCE, env=environment(),
                                 stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
        status(phase, child_pid=child.pid, log=str(log))
        while child.poll() is None:
            time.sleep(10)
            payload = dict(child_pid=child.pid, log=str(log))
            metrics = BASE/'train/metrics.jsonl'
            if phase == 'training' and metrics.exists():
                lines = metrics.read_text().splitlines()
                if lines:
                    payload['latest_metrics'] = json.loads(lines[-1])
            status(phase, **payload)
        if child.returncode:
            raise RuntimeError(f'{phase} exited {child.returncode}; see {log}')


def main():
    from omegaconf import OmegaConf
    records = BASE/'records'
    records.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if (BASE/'train/metrics.jsonl').exists():
        raise RuntimeError('Fresh output already contains training progress; explicit recovery is required')
    status('waiting_for_verified_training_archive')
    while not (BASE/'data/ILSVRC2012_img_train.verified.json').exists():
        time.sleep(10)
    # A separately started validation repair may still be completing.
    while not (BASE/'data/ILSVRC2012_img_val.verified.json').exists():
        time.sleep(10)
    if not (BASE/'data/ready.json').exists():
        execute('indexing', [sys.executable, str(SOURCE/'scripts/tools/prepare_imagenet_compound_audited.py'), 'index'])
    ready = json.loads((BASE/'data/ready.json').read_text())
    assert ready['training_images'] == 1281167 and ready['validation_images'] == 50000
    assert json.loads((records/'cpu-tests.json').read_text())['passed']
    recipe = SOURCE/CONFIG
    shutil.copyfile(recipe, records/'recipe.yaml')
    manifest = {}
    for folder in ['src', 'runtime', 'scripts', 'configs', 'third_party/rq-vae-transformer']:
        for path in (SOURCE/folder).rglob('*'):
            if path.is_file() and path.suffix in {'.py', '.yaml', '.json'}:
                manifest[str(path.relative_to(SOURCE))] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest['train.py'] = hashlib.sha256((SOURCE/'train.py').read_bytes()).hexdigest()
    write(records/'source-manifest.json', manifest)
    with tarfile.open(records/'source.tar.gz', 'w:gz') as archive:
        for name in manifest:
            archive.add(SOURCE/name, arcname=name)
    for name in ['audit.json', 'tests.xml', 'source-manifest.json', 'source.tar.gz', 'recipe.yaml', 'cpu-tests.json']:
        shutil.copyfile(records/name, OUT/name)
    run_command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8']
    execute('tokenizer_verification', run_command+[str(SOURCE/'scripts/tools/verify_imagenet_compound_audited.py')])
    assert json.loads((records/'quantization.json').read_text())['passed']
    shutil.copyfile(records/'quantization.json', OUT/'quantization.json')
    cfg = OmegaConf.load(recipe)
    canary = OmegaConf.create(OmegaConf.to_container(cfg))
    canary.options.update(output=str(BASE/'canary'), checkpoint_dir=str(BASE/'canary/checkpoints'),
        wandb_mode='disabled', upload_checkpoints=False, fid_every=0, fid_early_epochs=0,
        sample_grid_every=0, save_step_freq=0, max_optimizer_steps=12)
    canary_path = BASE/'canary.yaml'
    OmegaConf.save(canary, canary_path)
    execute('canary', run_command+[str(SOURCE/'train.py'), '--config', str(canary_path)])
    lines = [json.loads(x) for x in (BASE/'canary/metrics.jsonl').read_text().splitlines()]
    latest = json.loads((BASE/'canary/checkpoints/latest.json').read_text())
    assert latest['global_step'] == 12 and latest['rng_ranks'] == 8
    assert lines and 0 < lines[-1]['train/loss'] < 20
    checkpoint = latest['path']
    # Verify optimizer/scheduler age and the exact epoch cursor before throwing
    # away every canary weight; production is a separate fresh initialization.
    import torch
    snapshot = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert snapshot['global_step'] == 12 and snapshot['batch_idx'] == 96
    assert snapshot['scheduler']['kind'] == 'warmup-cosine-v1'
    assert snapshot['scheduler']['last_epoch'] == 12
    assert snapshot['scheduler']['policy']['warmup_steps'] == 313
    ages = {int(state['step']) for state in snapshot['optimizer']['state'].values()}
    assert ages == {12}
    report = dict(passed=True, steps=12, optimizer_ages=sorted(ages), rng_ranks=8,
        global_batch=2048, gpu_count=8, accumulation=8, microbatch=32,
        final_loss=lines[-1]['train/loss'], grad_norm=lines[-1]['train/grad_norm'],
        images_per_second=lines[-1]['train/images_per_second'],
        fresh_production_initialization=True, canary_weights_reused=False)
    write(records/'canary.json', report)
    shutil.copyfile(records/'canary.json', OUT/'canary.json')
    del snapshot
    shutil.rmtree(BASE/'canary/checkpoints')
    execute('training', run_command+[str(SOURCE/'train.py'), '--config', str(recipe)])
    status('completed')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        status('failed', error=repr(error))
        raise
