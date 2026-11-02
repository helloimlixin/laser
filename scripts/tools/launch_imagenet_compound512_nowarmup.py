"""Benchmark the archived compound512 runtime, then launch a fresh RQ-cosine run.

Uses both locally available B300s, with the original effective batch of 2048.
Credentials are read from an owner-only file and never serialized into artifacts.
"""
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from omegaconf import OmegaConf

ROOT = Path('/workspace/Projects/laser')
RUN = 'imagenet-compound512-nowarmup-2b300-20261028'
BASE = Path('/tmp/laser-' + RUN)
SOURCE = BASE / 'source'
OUT = ROOT / 'outputs' / RUN


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def status(phase, **values):
    record = dict(phase=phase, supervisor_pid=os.getpid(), updated_unix=time.time(), **values)
    write(BASE / 'status.json', record)
    write(OUT / 'status.json', record)
    print(json.dumps(record), flush=True)


def environment(batch, compiled, encoder=16):
    assert 2048 % (2 * batch) == 0
    env = dict(os.environ, PYTHONPATH=':'.join(map(str, [SOURCE, SOURCE/'runtime', SOURCE/'scripts/tools', BASE])),
        PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
        LASER_ACCUMULATION=str(2048 // (2 * batch)), LASER_ENCODER_BATCH=str(encoder),
        LASER_DATA_WORKERS='12', LASER_COMPILE_BLOCKS=str(int(compiled)),
        TORCHINDUCTOR_COMPILE_THREADS='4', TORCHINDUCTOR_CACHE_DIR=str(BASE/'inductor-cache'),
        TORCHINDUCTOR_FX_GRAPH_CACHE='1', TORCH_HOME=str(BASE/'torch-cache'),
        CUDA_CACHE_MAXSIZE='4294967296',
        WANDB_DIR=str(BASE/'wandb'), WANDB_CACHE_DIR=str(BASE/'wandb-cache'),
        WANDB_DATA_DIR=str(BASE/'wandb-data'), PYTORCH_ALLOC_CONF='expandable_segments:True',
        NCCL_NVLS_ENABLE='0', MPLBACKEND='Agg')
    env['WANDB_API_KEY'] = Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
    return env


def execute(label, cfg, batch, compiled, *, synthetic=False, encoder=16):
    path = BASE / (label + '.yaml')
    OmegaConf.save(cfg, path)
    entry = BASE/'benchmark_entry.py' if synthetic else SOURCE/'train.py'
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2',
               str(entry), '--config', str(path)]
    with (BASE / (label + '.log')).open('a') as log:
        child = subprocess.Popen(command, cwd=SOURCE, env=environment(batch, compiled, encoder),
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        while True:
            result = child.poll()
            values = dict(label=label, child_pid=child.pid, batch_size=batch,
                accumulation=2048//(2*batch), compiled=compiled, encoder_batch=encoder,
                log=str(BASE/(label+'.log')))
            metrics = Path(cfg.options.output)/'metrics.jsonl'
            if metrics.exists():
                rows = metrics.read_text().splitlines()
                if rows:
                    row = json.loads(rows[-1])
                    values['latest_metrics'] = {k: row[k] for k in
                        ['train/global_step', 'train/loss', 'train/grad_norm', 'train/lr', 'train/images_per_second']}
            status('training' if label == 'production' else 'benchmark', **values)
            if result is not None:
                return result
            time.sleep(10)


def trial_config(label):
    cfg = OmegaConf.load(BASE/'production.yaml')
    folder = BASE/'trials'/label
    cfg.options.update(output=str(folder), checkpoint_dir=str(folder/'checkpoints'),
        wandb_mode='disabled', upload_checkpoints=False, fid_every=0, fid_early_epochs=0,
        sample_grid_every=0, save_step_freq=0, max_optimizer_steps=22)
    return cfg


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (BASE/'train/metrics.jsonl').exists():
        raise RuntimeError('Fresh production already has progress; use full-state recovery explicitly')
    # Synthetic image batches include the unchanged FP32 teacher and the full
    # optimizer workload. An additional real-JPEG canary verifies the complete
    # data path before a separate production initialization.
    (BASE/'benchmark_entry.py').write_text('''import argparse
import torch
from torch.utils.data import Dataset
from omegaconf import OmegaConf
from src.training import rqtransformer as training
from src.training.imagenet_compound_runtime import install
from src.training.options import options_to_argv
class Images(Dataset):
    classes = [str(i) for i in range(1000)]
    def __init__(self):
        self.images = torch.rand(32, 3, 256, 256, generator=torch.Generator().manual_seed(73))*2-1
    def __len__(self): return 1281167
    def __getitem__(self, i): return self.images[i % 32], i % 1000
if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--config');a=p.parse_args()
    c=OmegaConf.load(a.config);install(c)
    training.source_image_dataset=lambda *a,**k:Images()
    training.main(options_to_argv(OmegaConf.to_container(c.options,resolve=True)))
''')
    status('compiler_parity')
    sys.path[:0] = [str(SOURCE/'scripts/tools'), str(SOURCE), str(SOURCE/'runtime')]
    import tune_imagenet_compound_execution as tune
    tune.TUNE, tune.SOURCE = BASE, SOURCE
    try:
        compiled = tune.verify_compilation()
    except Exception as error:
        write(BASE/'compilation-parity.json', dict(passed=False, error=repr(error)))
        compiled = False
    results = []
    candidates = [(128, False), (128, True), (256, True), (512, True)] if compiled else [(128, False), (256, False)]
    for batch, use_compile in candidates:
        label = f'{"compiled" if use_compile else "eager"}-b{batch}'
        cfg = trial_config(label)
        cfg.options.batch_size = batch
        code = execute(label, cfg, batch, use_compile, synthetic=True)
        row = dict(label=label, batch_size=batch, compiled=use_compile, returncode=code, passed=False)
        metrics = Path(cfg.options.output)/'metrics.jsonl'
        if code == 0 and metrics.exists():
            values = [json.loads(x) for x in metrics.read_text().splitlines()]
            last = values[-1]
            row.update(passed=math.isfinite(last['train/loss']) and math.isfinite(last['train/grad_norm']),
                       images_per_second=last['train/images_per_second'], loss=last['train/loss'],
                       grad_norm=last['train/grad_norm'], lr=last['train/lr'],
                       measurement='optimizer steps 11-20; synthetic image IO; unchanged online FP32 teacher')
        else:
            row['error_tail'] = (BASE/(label+'.log')).read_text()[-3000:]
        results.append(row)
        write(BASE/'benchmarks.json', results)
        write(OUT/'benchmarks.json', results)
        shutil.rmtree(Path(cfg.options.checkpoint_dir), ignore_errors=True)
        if batch >= 256 and not row['passed']:
            break
    passed = [r for r in results if r['passed']]
    if not passed:
        raise RuntimeError('No throughput candidate passed; see benchmark logs')
    best = max(passed, key=lambda r: r['images_per_second'])
    write(BASE/'selection.json', best)
    status('waiting_for_verified_data', selected=best)
    while not (BASE/'data/ready.json').exists():
        time.sleep(10)
    canary = trial_config('real-data-canary')
    canary.options.batch_size = best['batch_size']
    code = execute('real-data-canary', canary, best['batch_size'], best['compiled'])
    if code:
        raise RuntimeError('Real-data canary failed; see log')
    import torch
    receipt = json.loads((Path(canary.options.checkpoint_dir)/'latest.json').read_text())
    snapshot = torch.load(receipt['path'], map_location='cpu', weights_only=False, mmap=True)
    assert snapshot['global_step'] == 22 and receipt['rng_ranks'] == 2
    assert snapshot['scheduler']['last_epoch'] == 22
    assert snapshot['scheduler']['T_max'] == 62600
    assert {int(v['step']) for v in snapshot['optimizer']['state'].values()} == {22}
    expected_lr = .0005 * (1 + math.cos(math.pi*22/62600))/2
    assert math.isclose(snapshot['optimizer']['param_groups'][0]['lr'], expected_lr, rel_tol=1e-9)
    write(BASE/'records/canary.json',dict(passed=True,global_step=22,rng_ranks=2,
        optimizer_age=22,initial_lr=.0005,warmup_steps=0,total_schedule_steps=62600,
        real_data=True,production_weights_reused=False,selected=best))
    del snapshot
    shutil.rmtree(Path(canary.options.checkpoint_dir))
    cfg = OmegaConf.load(BASE/'production.yaml')
    cfg.options.batch_size = best['batch_size']
    OmegaConf.save(cfg, BASE/'production.yaml')
    OmegaConf.save(cfg, BASE/'records/recipe.yaml')
    for name in ['benchmarks.json','selection.json','compilation-parity.json']:
        shutil.copyfile(BASE/name,BASE/'records'/name)
    for path in (BASE/'records').iterdir():
        if path.is_file(): shutil.copyfile(path,OUT/path.name)
    shutil.copyfile(BASE/'production.yaml',OUT/'production.yaml')
    code = execute('production',cfg,best['batch_size'],best['compiled'])
    if code:
        raise RuntimeError('Production exited with code '+str(code))
    status('completed')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        status('failed', error=repr(error))
        raise
