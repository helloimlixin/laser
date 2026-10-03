"""Preflight and supervise both matched two-H100 Church experiments."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = Path('/mnt/laser-church/normalized-noise-comparison')
children = {}


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def status(phase, **extra):
    row = dict(phase=phase, supervisor_pid=os.getpid(),
               children={name: dict(pid=p.pid, exit_code=p.poll()) for name, p in children.items()},
               updated_unix=time.time(), **extra)
    atomic(ROOT/'status.json', row)
    print(json.dumps(row), flush=True)


def environment(gpus):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='4', NCCL_NVLS_ENABLE='0', PYTHONUNBUFFERED='1',
        TORCH_HOME='/mnt/laser-church/torch-cache', WANDB_MODE='online', WANDB_DISABLE_GIT='true',
        WANDB_ENTITY='helloimlixin-rutgers', WANDB_PROJECT='laser',
        WANDB_API_KEY=Path('/root/.config/laser/wandb-api-key').read_text().strip(),
        WANDB_CACHE_DIR='/mnt/laser-church/wandb-cache', WANDB_DATA_DIR='/mnt/laser-church/wandb-data',
        WANDB_DIR='/mnt/laser-church/wandb', LASER_CHECKPOINT_STAGING_DIR=str(ROOT/'serialization'),
        LASER_ACCUMULATION='4')
    for key in ['WANDB_SERVICE', '_WANDB_SERVICE', 'WANDB_RUN_ID', 'WANDB_NAME', 'WANDB_RESUME', 'WANDB_RESUME_MODE', 'SMOKE_TEST']:
        env.pop(key, None)
    return env


def launch(variant, script, arguments, log):
    base = Path(variant['base'])
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2',
               str(base/script), *arguments]
    with (base/log).open('a') as stream:
        process = subprocess.Popen(command, cwd=base, env=environment(variant['gpus']),
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    children[variant['run_id']] = process


def wait_all(phase):
    while True:
        status(phase)
        if all(p.poll() is not None for p in children.values()):
            break
        time.sleep(30)
    failed = {name: p.returncode for name, p in children.items() if p.returncode}
    if failed:
        raise RuntimeError(f'{phase} failed: {failed}')
    children.clear()


def main():
    lock = Path('/mnt/laser-church/supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    variants = read(ROOT/'prepared.json')['variants']
    assert read(ROOT/'validation.json')['passed']
    assert read(ROOT/'audit-numerical.json')['passed']
    assert read(ROOT/'heldout-probe.json')['passed']
    for variant in variants:
        base = Path(variant['base'])
        if not (base/'benchmark-a4/complete.json').exists():
            launch(variant, 'train.py', ['--mode', 'benchmark-a4', '--steps', '12'], 'benchmark-a4.log')
    wait_all('benchmark-training')
    initializations = []
    for variant in variants:
        base = Path(variant['base'])
        assert read(base/'benchmark-a4/complete.json')['passed']
        rows = [read(base/f'benchmark-a4/performance-rank{rank}.json') for rank in range(2)]
        assert max(row['peak_allocated_gib'] for row in rows) < 76
        initializations.append(read(base/'benchmark-a4/initialization.json')['full_state_sha256'])
        atomic(base/'benchmark-selection.json', dict(passed=True, accumulation_steps=4, microbatch_per_gpu=256,
            world_size=2, total_batch=2048, rows=rows, note='Identical fixed batch layout for both temperatures'))
    assert len(set(initializations)) == 1, 'Fresh model weights differ between experimental arms'
    atomic(ROOT/'initialization-comparison.json', dict(passed=True, full_state_sha256=initializations[0],
                                                      run_ids=[v['run_id'] for v in variants]))
    for variant in variants:
        if not (Path(variant['base'])/'fid-benchmark.json').exists():
            launch(variant, 'benchmark_fid.py', [], 'fid-benchmark.log')
    wait_all('benchmark-fid')
    for variant in variants:
        base = Path(variant['base'])
        assert read(base/'fid-benchmark.json')['passed']
        if not (base/'train/provenance-upload.json').exists():
            files = {p.name: p for p in base.glob('*.py')}
            for prefix, directory in [('runtime', ROOT/'runtime'), ('official-metrics', base/'official-metrics')]:
                files.update({prefix+'/'+str(p.relative_to(directory)): p for p in directory.rglob('*.py')
                              if '__pycache__' not in p.parts})
            atomic(base/'source-manifest.json', {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                                 for name, path in sorted(files.items())})
            with tarfile.open(base/'runtime-source.tar.gz', 'w:gz') as archive:
                for name, path in sorted(files.items()):
                    archive.add(path, arcname=name, recursive=False)
                for name in ['validation.json', 'audit-numerical.json', 'plan.json', 'heldout-probe.json']:
                    archive.add(base/name, arcname=name, recursive=False)
        if not (base/'train/request.json').exists():
            assert not (base/'train/checkpoints/last.pt').exists(), 'Production must start from scratch'
        launch(variant, 'train.py', ['--mode', 'train'], 'train.log')
    wait_all('training')
    status('completed', results={v['run_id']: read(Path(v['base'])/'train/complete.json') for v in variants})


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        status('failed', error=repr(error))
        raise
