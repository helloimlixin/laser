"""Queue the eight-GPU sampler after its parent training and uploads complete."""
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
child = None


def read(path):
    return json.loads(Path(path).read_text())


def write(path, record):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(record, indent=2, default=str) + '\n')
    temporary.replace(path)


def status(phase, **extra):
    write(BASE / 'status.json', dict(phase=phase, supervisor_pid=os.getpid(),
          worker_pid=None if child is None else child.pid, updated_unix=time.time(), **extra))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def interrupted(signum, frame):
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signum)
    status('interrupted', signal=signum)
    raise SystemExit(128 + signum)


def main():
    global child
    lock = (BASE / '.supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    request = read(BASE / 'request.json')
    parent = Path(request['training_directory'])
    assert not (BASE / 'complete.json').exists(), 'Sweep already complete'
    for relative, digest in read(BASE / 'source-manifest.json').items():
        assert sha(BASE / relative) == digest, relative
    while True:
        current = read(parent / 'status.json')
        if current['phase'] == 'complete':
            break
        if current['phase'] in ['failed', 'interrupted']:
            raise RuntimeError(f'Parent training {current["phase"]}: {current}')
        os.kill(current['supervisor_pid'], 0)
        status('waiting_for_training', parent_status=current)
        time.sleep(30)
    complete = read(parent / 'train/complete.json')
    uploads = read(parent / 'train/checkpoint-upload.json')
    assert complete['epoch'] == 90 and complete['global_step'] == 7380
    assert uploads['verified_online'] and uploads['global_step'] == 7380
    status('freezing_checkpoint', parent_complete=complete)
    for relative, expected in read(parent / 'source-manifest.json').items():
        assert sha(parent / relative) == expected, relative
    config = read(parent / 'train/request.json')['config']
    assert sha(config['checkpoint']) == request['stage1_checkpoint_sha256']
    assert sha(config['fid_reference_stats']) == request['reference_sha256']
    checkpoint_directory = parent / 'train/checkpoints'
    candidates = sorted(checkpoint_directory.glob('best_fid_*.pt'))
    assert len(candidates) == 1, candidates
    snapshot = BASE / 'frozen-best.pt'
    if not snapshot.exists():
        os.link(candidates[0], snapshot)
    with snapshot.open('rb') as stream:
        digest = base64.b64encode(hashlib.file_digest(stream, 'md5').digest()).decode()
    remote_file = next(item for item in uploads['files'] if item['name'] == 'best-fid-01.pt')
    assert digest == remote_file['md5'] and snapshot.stat().st_size == remote_file['bytes']
    import torch
    payload = torch.load(snapshot, map_location='cpu', weights_only=False, mmap=True)
    frozen = dict(path=str(snapshot), original_path=str(candidates[0]), sha256=sha(snapshot), md5=digest,
                  bytes=snapshot.stat().st_size, epoch=payload['epoch'], global_step=payload['global_step'],
                  training_fid_50000=payload['fid'], artifact=uploads['artifact'],
                  artifact_file='best-fid-01.pt', stage1_checkpoint_sha256=request['stage1_checkpoint_sha256'],
                  runtime_manifest_sha256=sha(parent / 'source-manifest.json'))
    write(BASE / 'frozen-checkpoint.json', frozen)
    del payload
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', OMP_NUM_THREADS='8',
               OPENBLAS_NUM_THREADS='8', MKL_NUM_THREADS='8', PYTHONUNBUFFERED='1', NCCL_NVLS_ENABLE='0',
               WANDB_API_KEY=Path('/root/.config/laser/wandb-api-key').read_text().strip(),
               WANDB_ENTITY='helloimlixin-rutgers', WANDB_PROJECT='laser', WANDB_MODE='online',
               TORCH_HOME='/workspace/tmp/official-rqvae-eval-cache')
    for key in ['WANDB_SERVICE', '_WANDB_SERVICE', 'WANDB_RUN_ID', 'WANDB_NAME', 'WANDB_RESUME',
                'WANDB_RESUME_MODE', 'SMOKE_TEST']:
        env.pop(key, None)
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
               str(BASE / 'evaluate.py'), '--directory', str(BASE)]
    with (BASE / 'evaluate.log').open('a') as stream:
        child = subprocess.Popen(command, cwd=BASE, env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        while True:
            status('evaluating', command=command, checkpoint=frozen)
            try:
                code = child.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                continue
    if code:
        raise RuntimeError(f'Evaluator exited {code}; inspect {BASE / "evaluate.log"}')
    result = read(BASE / 'complete.json')
    assert result['passed'] and result['upload']['verified_online']
    status('complete', selected_setting=result['selected_setting'],
           selected_fid_50000=result['selected_fid_50000'], upload=result['upload'])


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        status('failed', error=str(error))
        raise
