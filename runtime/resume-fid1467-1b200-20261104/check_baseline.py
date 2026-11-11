"""Remeasure unchanged source weights with the continuation's exact FID50k protocol."""
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time

BASE = Path('/tmp/laser-fid1467-b200/frozen')
OUT = Path('/workspace/Projects/laser/outputs/imnet-j16384-a4096-fid1467-resume100-1b200-20261104')
DIAG = BASE / 'diagnostics/source-baseline'
TRAINER_PID = 4882


def record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.pending')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def evaluate():
    runpy.run_path(str(BASE / 'train.py'), run_name='diagnostic_import')
    import torch
    from src import rqvae_metrics
    from src.training import rq_imagenet_shared as shared
    from src.training.laser_joint_recipe import JointLaserRecipe

    torch.set_num_threads(4)
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    shared.BASE, shared.OUT = BASE, DIAG
    shared.PROFILE = JointLaserRecipe(BASE)
    shared.PROFILE.generation_batch = 64
    rqvae_metrics.INCEPTION_PATH = BASE / 'upstream/rqvae/metrics/inception.py'
    aux, _ = shared.PROFILE.tokenizer(device)
    model, optimizer, config = shared.fresh_model(device)
    source = torch.load(BASE / 'inputs/resume-checkpoint.pt', map_location='cpu',
                        weights_only=False, mmap=True)
    assert source['global_step'] == 45698 and source['epoch'] == 73
    assert source['native_sampling'] == shared.PROFILE.sampling_settings()
    model.load_state_dict(source['state_dict'], strict=True)
    original = source['evaluation']
    del source, optimizer, config
    start = time.monotonic()
    result = shared.generation(model, aux, device, 0, 1, 73, None, batch=64, seed=261001)
    report = dict(passed=True, source_step=45698, source_epoch=73,
                  source_artifact='helloimlixin-rutgers/laser/imnet-j16384-a4096-fid1481-resume100-4h100-20261104-checkpoint:v26',
                  source_weights_unchanged=True, world_size=1, generation_batch=64,
                  seed=261001, generated_images=50000, real_images=1281167,
                  real_split='train', metric_backend='original_rqtransformer',
                  inherited_four_gpu_result=original, measured_one_gpu_result=result,
                  elapsed_seconds=time.monotonic() - start, time=time.time())
    record(DIAG / 'result.json', report)
    record(OUT / 'diagnostics/source-baseline.json', report)
    print(json.dumps(report), flush=True)


def supervise():
    cmdline = Path(f'/proc/{TRAINER_PID}/cmdline').read_bytes()
    assert str(BASE / 'train.py').encode() in cmdline.split(b'\0')
    child = None

    def terminate(signum, _frame):
        if child is not None and child.poll() is None:
            child.terminate()
        raise SystemExit(128 + signum)

    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, terminate)
    DIAG.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1',
               LASER_FUSED_ADAMW='1', LASER_TRAIN_SDPA='0',
               LASER_COMPILE_NATIVE_LOSS='0',
               PYTORCH_ALLOC_CONF='expandable_segments:True')
    os.kill(TRAINER_PID, signal.SIGSTOP)
    try:
        with (DIAG / 'evaluation.log').open('ab', buffering=0) as log:
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--evaluate'],
                                     cwd=BASE, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                path = DIAG / 'generation-progress.json'
                progress = json.loads(path.read_text()) if path.exists() else None
                record(OUT / 'diagnostics/status.json', dict(
                    phase='remeasuring_source_baseline', time=time.time(),
                    trainer_temporarily_held=True, trainer_pid=TRAINER_PID,
                    diagnostic_pid=child.pid, progress=progress))
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    pass
        if child.returncode:
            raise RuntimeError(f'Baseline evaluation failed: {DIAG / "evaluation.log"}')
    finally:
        os.kill(TRAINER_PID, signal.SIGCONT)
        record(OUT / 'diagnostics/status.json', dict(
            phase='completed' if child is not None and child.returncode == 0 else 'failed',
            trainer_resumed=True, trainer_pid=TRAINER_PID, time=time.time()))


if __name__ == '__main__':
    if '--evaluate' in sys.argv:
        evaluate()
    else:
        supervise()
