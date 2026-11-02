"""Benchmark, validate, stage data, and supervise a configured CC3M joint run."""
import argparse
import concurrent.futures
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

import yaml


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def stage(cfg):
    import numpy as np
    from src.data.cc3m_joint import shard_index
    manifest = json.loads(Path(cfg['data_manifest']).read_text())
    order = np.random.default_rng(np.random.SeedSequence([cfg['seed'], 0])).permutation(len(manifest['train']))
    specs = manifest['validation']+[manifest['train'][int(i)] for i in order]

    def copy(spec):
        shard_index(spec, cfg['data_source'], cfg['data_local'])
        return spec['name']

    base = Path(cfg['local_checkpoints']).parent
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for count, name in enumerate(pool.map(copy, specs), 1):
            write(base/'staging.json', dict(completed=count, total=len(specs), last=name, time=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--stage-only', action='store_true')
    args = parser.parse_args()
    path = args.config.resolve()
    cfg = yaml.safe_load(path.read_text())
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    if args.stage_only:
        stage(cfg)
        return
    base, out = Path(cfg['local_checkpoints']).parent, Path(cfg['output'])
    base.mkdir(parents=True, exist_ok=True)
    lock = (base/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = base/'source'
    repro = base/'live/reproducibility'
    repro.mkdir(parents=True, exist_ok=True)
    # Freeze the executable before the first launch; resumes use that snapshot.
    if not (source/'train.py').exists():
        source.mkdir(exist_ok=True)
        for name in ('src', 'third_party/rq-vae-transformer'):
            shutil.copytree(root/name, source/name,
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', 'notebooks'))
        shutil.copyfile(root/'train.py', source/'train.py')
    for key in ('stage1', 'codebook', 'reconstruction_codebook'):
        with Path(cfg[key]).open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if digest != cfg[key+'_sha256']:
            raise ValueError(f'Asset checksum mismatch: {key}')
    credential = Path('/root/.config/laser/wandb.key').read_text().strip()
    env = dict(os.environ, WANDB_API_KEY=credential, WANDB_DISABLE_GIT='true',
        WANDB_CACHE_DIR=str(base/'wandb-cache'), WANDB_DATA_DIR=str(base/'wandb-data'),
        WANDB_ARTIFACT_DIR=str(base/'wandb-artifacts'), TORCH_HOME=str(base/'torch-cache'),
        PYTHONPATH=str(source), RQ_UPSTREAM_ROOT=str(source/'third_party/rq-vae-transformer'),
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
        PYTHONWARNINGS='ignore::FutureWarning', LASER_CC3M_CHECKPOINT_MIRROR='0')
    logs = base/'logs'
    logs.mkdir(exist_ok=True)
    children = []

    def stop(*_):
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def execute(command, phase, background=False):
        with (logs/(phase+'.log')).open('ab', buffering=0) as log:
            child = subprocess.Popen(command, cwd=source, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        children.append(child)
        if background:
            return child
        write(base/'status.json', dict(phase=phase, pid=os.getpid(), child_pid=child.pid,
            command=command, run_id=cfg['run_id'], time=time.time()))
        code = child.wait()
        if code:
            write(base/'status.json', dict(phase='failed', failed_phase=phase, exit_code=code,
                run_id=cfg['run_id'], time=time.time()))
            raise RuntimeError(f'{phase} exited with {code}; inspect {logs/(phase+".log")}')

    from src.training.cc3m_storage import WorkspaceMirror
    mirror = WorkspaceMirror([(logs, out/'logs'), (base/'live', out)]).start()
    stager = execute([sys.executable, str(Path(__file__).resolve()), '--config', str(path),
                      '--stage-only'], 'staging', background=True)
    try:
        command = [sys.executable, '-m', 'src.training.cc3m_joint', '--config', str(path)]
        if not (base/'live/benchmark.json').exists():
            execute([*command, '--benchmark'], 'benchmark')
        cfg = yaml.safe_load(path.read_text())
        (root/'configs/stage2'/f'{cfg["run_id"]}.yaml').write_text(
            '# @package _global_\n'+path.read_text())
        shutil.copyfile(path, repro/'recipe.yaml')
        manifest = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in source.rglob('*') if p.is_file() and p.suffix in ('.py', '.json', '.yaml', '.txt', '.gz')}
        write(repro/'code-manifest.json', manifest)
        with zipfile.ZipFile(repro/'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in manifest:
                archive.write(source/name, name)
        smoke = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                 f'--nproc-per-node={cfg["world_size"]}', '-m', 'src.training.cc3m_joint',
                 '--config', str(path), '--smoke']
        if not (base/'live/preflight/passed.json').exists():
            execute(smoke, 'gpu_preflight')
        assert json.loads((base/'live/preflight/passed.json').read_text())['passed']
        write(base/'live/launch.json', dict(run_id=cfg['run_id'], world_size=cfg['world_size'],
            global_batch=cfg['global_batch'], microbatch=cfg['microbatch'], accumulation=cfg['accumulation'],
            epochs=cfg['epochs'], preflight_passed=True, time=time.time()))
        production = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                      f'--nproc-per-node={cfg["world_size"]}', 'train.py', '--config', str(path)]
        execute(production, 'training')
        write(base/'status.json', dict(phase='completed' if (base/'live/completed.json').exists()
              else 'stopped', run_id=cfg['run_id'], time=time.time()))
    finally:
        if stager.poll() is None:
            os.killpg(stager.pid, signal.SIGTERM)
        mirror.close()


if __name__ == '__main__':
    main()
