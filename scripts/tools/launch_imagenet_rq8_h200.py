#!/usr/bin/env python3
"""Resume ImageNet RQ8 on six H200s using a verified stage-1-compatible cache."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/imagenet-rfid421-rq8-refit-20260913'
RECOVERY = BASE / 'best3-20260918'
OUTPUT = BASE / 'train-h200x6-20260918'
RUN_ID = 'imagenet-rfid421-rq8-refit-480m-20260913'


def ensure_ram_cache(verification):
    report = RECOVERY / 'cache-verification.json'
    assert hashlib.sha256(report.read_bytes()).hexdigest() == verification['cache_report_sha256']
    proof = json.loads(report.read_text())
    source, target = Path(proof['source']), Path(proof['cache'])
    assert source.is_absolute() and proof['passed']
    if (target / 'verification.json').exists():
        assert (target / 'verification.json').read_bytes() == report.read_bytes()
        assert all((target / name).stat().st_size == record['bytes']
                   for name, record in proof['files'].items())
        return target
    target.mkdir(parents=True, exist_ok=True)
    def stage(item):
        name, record = item
        digest = hashlib.sha256()
        temporary = (target / name).with_suffix('.partial')
        with (source / name).open('rb') as inp, temporary.open('wb') as out:
            while block := inp.read(16 * 1024**2):
                digest.update(block)
                out.write(block)
        assert temporary.stat().st_size == record['bytes'], name
        assert digest.hexdigest() == record['sha256'], name
        temporary.replace(target / name)
        print(f'Restored cache file: {name}', flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(stage, proof['files'].items()))
    complete = (source / 'complete.json').read_bytes()
    assert hashlib.sha256(complete).hexdigest() == proof['complete_spec_sha256']
    (target / 'complete.json').write_bytes(complete)
    (target / 'verification.json').write_bytes(report.read_bytes())
    return target


def main():
    verification = json.loads((RECOVERY / 'verification.json').read_text())
    assert verification['passed']
    runtime = RECOVERY / 'runtime'
    manifest = RECOVERY / 'runtime-manifest.json'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == verification['runtime_manifest_sha256']
    for name, digest in json.loads(manifest.read_text()).items():
        assert hashlib.sha256((runtime / name).read_bytes()).hexdigest() == digest, name
    checkpoint = OUTPUT / 'checkpoints/last.pt'
    assert checkpoint.is_file()
    receipt_path = OUTPUT / 'launch.json'
    for receipt in (receipt_path, OUTPUT / 'status.json'):
        if receipt.exists():
            pid = json.loads(receipt.read_text()).get('pid')
            cmdline = Path(f'/proc/{pid}/cmdline')
            if pid and cmdline.exists() and str(OUTPUT).encode() in cmdline.read_bytes():
                raise RuntimeError(f'Training is already active as PID {pid}')
    cache = ensure_ram_cache(verification)
    python = Path(verification['python'])
    assert python.is_file(), 'Recreate the recorded environment before resuming.'
    environment = dict(os.environ)
    if not environment.get('WANDB_API_KEY'):
        environment['WANDB_API_KEY'] = Path('/root/.config/laser/wandb-api-key').read_text().strip()
    environment.update(LASER_PROJECT_ROOT=str(runtime), CUDA_VISIBLE_DEVICES='0,1,2,3,4,5',
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
        LASER_EVAL_WORKERS='2', NCCL_NVLS_ENABLE='0', PYTHONUNBUFFERED='1',
        TORCH_HOME='/workspace/tmp/official-rqvae-eval-cache')
    for key in ('WANDB_SERVICE', '_WANDB_SERVICE', 'CUDA_LAUNCH_BLOCKING'):
        environment.pop(key, None)
    command = [str(python), '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=6',
        str(runtime / 'scripts/tools/resume_imagenet_rq8.py'), '--resume', str(checkpoint),
        '--tokenizer', str(OUTPUT / 'checkpoints/tokenizer.pt'),
        '--codebook', str(OUTPUT / 'checkpoints/codebook.pt'),
        '--manifests', str(ROOT / 'outputs/imagenet-scaled-rq-stage2-20260913'),
        '--reference', str(ROOT / 'third_party/rq-vae-transformer/assets/fid_stats/imagenet_256_train.npz'),
        '--data', '/workspace/Projects/data/imagenet2012', '--output', str(OUTPUT),
        '--token-cache', str(cache),
        '--run-id', RUN_ID, '--batch-size', str(verification['microbatch']),
        '--workers', str(verification['workers']), '--save-every', '100',
        '--target-chunk-size', str(verification['target_chunk_size']),
        '--ce-chunk-size', str(verification['ce_chunk_size'])]
    with (OUTPUT / 'training.log').open('ab') as log:
        process = subprocess.Popen(command, cwd=runtime, env=environment, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = dict(pid=process.pid, command=command, run_id=RUN_ID, launched_unix=time.time(),
                   verification=str(RECOVERY / 'verification.json'))
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
