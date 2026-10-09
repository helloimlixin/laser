"""Continue the repaired ImageNet run with official metrics and full recovery saves."""
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

RUN_ID = 'imagenet-rfid421-pairfix-repair-scratch-8gpu-20261004b'
RUN_PATH = 'helloimlixin-rutgers/laser/' + RUN_ID


def record(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('Frozen source differs at: ' + before[:100])
    return text.replace(before, after)


def prepare(base, evidence):
    import torch
    import yaml
    sys.path.insert(0, str(base / 'source/runtime'))
    from src.training.full_resume_upload import recovery_metadata
    local = base / 'inputs/local-last.pt'
    payload = torch.load(local, map_location='cpu', weights_only=False, mmap=True)
    metadata = recovery_metadata(payload)
    assert metadata['world_size'] == 8 and metadata['adam_step'] == metadata['global_step']
    assert metadata['learning_rate_schedule']['kind'] == 'cosine'
    expected = 3e-7 + .5 * (.0006 - 3e-7) * (1 + math.cos(math.pi * metadata['global_step'] / 62600))
    assert math.isclose(metadata['saved_learning_rates'][0], expected, rel_tol=1e-10)
    assert metadata['learning_rate_schedule']['state']['last_epoch'] == metadata['global_step']
    record(evidence / 'continuation-20261005/recovery-verification.json', metadata)
    checkpoint_dir = evidence / 'train/checkpoints'
    latest = checkpoint_dir / 'last.pt'
    # The persistent mount restored symlink aliases as empty files. Preserve
    # that stub and reconstruct the alias from the verified immutable payload.
    source = checkpoint_dir / '.checkpoint-data/last-138110-1791184742541247019.pt'
    assert source.is_file() and source.stat().st_size == local.stat().st_size
    if latest.exists() and not latest.is_symlink():
        assert latest.stat().st_size == 0, 'Refuse to replace a real newer checkpoint'
        latest.rename(checkpoint_dir / 'last.pre-continuation-empty-alias')
    if not latest.exists():
        latest.symlink_to(source.relative_to(checkpoint_dir))
    shutil.copytree(base / 'source/support', base / 'support', dirs_exist_ok=True)
    entry = (base / 'source/entry.py').read_text()
    entry = replace_once(entry, "assert ARGS.metric_backend=='torchmetrics'", "assert ARGS.metric_backend=='original-rqvae'")
    entry = replace_once(entry, "kwargs['resume']='allow'", "kwargs['resume']='must'")
    entry = replace_once(entry,
        "  saved=payload.get('config',{})\n",
        "  saved=payload.get('config',{})\n"
        "  if saved.get('metric_backend')!='original-rqvae':\n"
        "   payload['best_fid']=[];payload['best_inception']=[]\n")
    entry = replace_once(entry,
        "  evaluation_protocol='Matched reference run: torchmetrics, ImageNet validation50k, generated50k',",
        "  evaluation_protocol='Official RQ-Transformer Inception, FID and 10-split IS; ImageNet validation50k, generated50k',\n"
        "  evaluation_backend_changed_at_step=31000,previous_metric_backend='torchmetrics',")
    entry = replace_once(entry, " step=int(payload['global_step']);epoch=int(payload['epoch'])",
        " step=int(payload['global_step']);epoch=int(payload['epoch'])\n"
        " if OFFICIAL_METRICS is not None and OFFICIAL_METRICS['global_step']==step:\n"
        "  payload=dict(payload,original_rqtransformer_metrics=OFFICIAL_METRICS)")
    injected = '''
from official_metrics import install as install_official_metrics
install_official_metrics(ROOT)
OFFICIAL_METRICS=None
original_evaluate=training.evaluate_generation_metrics
def evaluate(*args,**kwargs):
 global OFFICIAL_METRICS
 device=next(args[0].parameters()).device
 # Metrics must not consume the saved dropout/coefficient-target RNG streams.
 with torch.random.fork_rng(devices=[device.index]):
  torch.random.default_generator.manual_seed(261001+dist.get_rank())
  torch.cuda.manual_seed(261001+dist.get_rank())
  result=original_evaluate(*args,**kwargs)
 step=(INITIAL_ADAM_STEP if INITIAL_ADAM_STEP is not None else INITIAL_RECOVERY['global_step'])+UPDATES
 OFFICIAL_METRICS=dict(global_step=step,fid=result[0],inception_score=result[1],
  inception_score_std=result[2],metric_backend='original_rqtransformer',real_images=50000,
  generated_images=50000,real_split='val',inception_splits=10,seed=261001)
 if dist.get_rank()==0:
  record(EVIDENCE/'continuation-20261005'/f'official-metrics-step{step}.json',OFFICIAL_METRICS)
  if WB is not None:
   WB.log({'eval/fid_original_rqtransformer':result[0],
    'eval/inception_score_original_rqtransformer':result[1],
    'eval/inception_score_std_original_rqtransformer':result[2],
    'train/global_step':step})
 return result
training.evaluate_generation_metrics=evaluate
'''
    entry = replace_once(entry, 'from src.training.cli import main\n', injected + '\nfrom src.training.cli import main\n')
    (base / 'entry.py').write_text(entry)
    (base / 'support/official_metrics.py').write_text('''"""Execute upstream metric functions without importing its unused CLIP backend."""
import importlib.util
import sys
import types

def install(root):
 from src import rqvae_metrics as adapter
 folder=root/'third_party/rq-vae-transformer/rqvae/metrics'
 package=types.ModuleType('laser_official_rq_metrics')
 package.__path__=[str(folder)]
 sys.modules[package.__name__]=package
 modules={}
 for name in ['inception','fid','IS']:
  qualified=package.__name__+'.'+name
  spec=importlib.util.spec_from_file_location(qualified,folder/(name+'.py'))
  module=importlib.util.module_from_spec(spec)
  sys.modules[qualified]=module
  spec.loader.exec_module(module)
  modules[name]=module
 adapter.frechet_distance=modules['fid'].frechet_distance
 def score(probabilities,splits=10):
  mean,std=modules['IS'].calculate_kl_div(probabilities,splits)
  return float(mean),float(std)
 adapter.inception_score=score
''')
    recipe = yaml.safe_load((base / 'inputs/resume-active-config.yaml').read_text())
    recipe['options'].update(checkpoint=str(base / 'inputs/resume-stage1-tokenizer.pt'),
        token_cache=str(base / 'inputs/resume-imagenet-k4-cache.pt'),
        data='/tmp/laser-imagenet-stage2/imagenet', output=str(base / 'production/train'),
        checkpoint_dir=str(checkpoint_dir), resume=True, wandb_mode='online',
        metric_backend='original-rqvae', fid_reference_stats=None)
    recipe['options'].pop('resume_checkpoint', None)
    (base / 'recipe.yaml').write_text(yaml.safe_dump(recipe, sort_keys=False))
    audit = evidence / 'continuation-20261005'
    audit.mkdir(exist_ok=True)
    shutil.copyfile(base / 'entry.py', audit / 'entry.py')
    shutil.copyfile(base / 'recipe.yaml', audit / 'recipe.yaml')
    shutil.copyfile(base / 'support/official_metrics.py', audit / 'official_metrics.py')
    weights = base / 'torch-cache/hub/checkpoints'
    weights.mkdir(parents=True, exist_ok=True)
    for name in ['weights-inception-2015-12-05-6726825d.pth', 'pt_inception-2015-12-05-6726825d.pth']:
        shutil.copyfile(base / ('inputs/resume-' + name), weights / name)
    cache = base / 'checkpoint-upload-cache/objects'
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(str(source.resolve()).encode()).hexdigest()
    os.link(local, cache / (key + '.pt'))
    identity = {k: getattr(source.stat(), k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')}
    record(cache / (key + '.json'), dict(source=str(source.resolve()), identity=identity))
    print(json.dumps(dict(resume_step=metadata['global_step'], lr=expected,
        adam_restored=True, scheduler_restored=True, official_metrics=True)), flush=True)


def supervise(base, evidence, key_file):
    lock = (base / 'continuation.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stopping = False
    child = None
    phase = None
    def stop(_sig, _frame):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            for path in (evidence / 'verification' / phase).glob('process-rank*.json'):
                try:
                    os.kill(json.loads(path.read_text())['pid'], signal.SIGTERM)
                except ProcessLookupError:
                    pass
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    env = os.environ.copy()
    env.update(WANDB_API_KEY=key_file.read_text().strip(), LASER_RUN_BASE=str(base),
        LASER_PERSISTENT_BASE=str(evidence), LASER_ACCUMULATION='3',
        LASER_CHECKPOINT_STAGING_DIR=str(base / 'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(base / 'checkpoint-upload-cache'),
        LASER_CHECKPOINT_IMMUTABLE_FILES='1', LASER_COMPILE_BLOCKS='1',
        LASER_PREVIEW_KEEP_OPTIMIZER='1', CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
        WANDB_DIR=str(base / 'wandb'), WANDB_CACHE_DIR=str(base / 'wandb-cache'),
        WANDB_DATA_DIR=str(base / 'wandb-data'), WANDB_CONFIG_DIR=str(base / 'wandb-config'),
        TORCHINDUCTOR_CACHE_DIR=str(base / 'inductor-cache'), TORCHINDUCTOR_COMPILE_THREADS='2',
        TORCH_HOME=str(base / 'torch-cache'), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', NCCL_NVLS_ENABLE='0',
        TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
    for name in ['wandb','wandb-cache','wandb-data','wandb-config']:
        (base / name).mkdir(exist_ok=True)
    for attempt in range(3):
        if stopping:
            break
        phase = os.environ.get('LASER_CONTINUATION_TAG', 'continuation_20261005') + '_' + str(attempt)
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
            '--nproc-per-node=8', str(base / 'entry.py'), '--config', str(base / 'recipe.yaml')]
        log = evidence / 'continuation-20261005' / (phase + '.log')
        with log.open('a') as stream:
            child = subprocess.Popen(command, cwd=base, env=dict(env, LASER_PHASE=phase),
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            while child.poll() is None:
                record(evidence / 'continuation-20261005/status.json', dict(state='running',
                    supervisor_pid=os.getpid(), torchrun_pid=child.pid, attempt=attempt, phase=phase,
                    heartbeat_unix=time.time(), log=str(log), run_path=RUN_PATH,
                    target_epoch=100, world_size=8, metric_backend='original-rqvae'))
                time.sleep(5)
        if child.returncode == 0 or stopping:
            break
        tail = log.read_text()[-24000:]
        if any(word in tail for word in ['non-finite','out of memory','AssertionError','ModuleNotFoundError']):
            break
        time.sleep(10)
    record(evidence / 'continuation-20261005/status.json', dict(state='stopped_resumable' if stopping
        else 'completed' if child.returncode == 0 else 'failed', exit_code=child.returncode,
        supervisor_pid=os.getpid(), torchrun_pid=child.pid, attempt=attempt, phase=phase,
        heartbeat_unix=time.time(), log=str(log), run_path=RUN_PATH))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--key-file', type=Path, required=True)
    parser.add_argument('--supervise', action='store_true')
    args = parser.parse_args()
    if args.supervise:
        supervise(args.base, args.evidence, args.key_file)
    else:
        prepare(args.base, args.evidence)


if __name__ == '__main__':
    main()
