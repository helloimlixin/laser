"""Persist complete checkpoints independently of slow W&B model uploads."""
import concurrent.futures
import json
import os
from pathlib import Path
import time

import torch

POOL = concurrent.futures.ThreadPoolExecutor(max_workers=1)
JOBS = []
ORIGINAL = None


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def publish(run, snapshot, out, result, metadata, aliases):
    """Own a separate immutable NVMe link until W&B acknowledges the upload."""
    import wandb
    try:
        artifact = wandb.Artifact(run.id+'-checkpoint', type='model', metadata=metadata)
        artifact.add_file(str(snapshot), name='last.pt', policy='immutable', skip_cache=True)
        artifact = run.log_artifact(artifact, aliases=aliases)
        artifact.wait()
        assert str(artifact.state) == 'COMMITTED'
        receipt = dict(result, state='COMMITTED', artifact=artifact.qualified_name)
        write(out/'cloud-checkpoints'/f"step-{result['global_step']:07d}.json", receipt)
        write(out/'cloud-checkpoint.json', receipt)
        snapshot.unlink()
        print(json.dumps(dict(phase='cloud-checkpoint-committed', **receipt)), flush=True)
        return receipt
    except Exception as error:
        # The complete persistent checkpoint and upload link remain recoverable.
        receipt = dict(result, state='UPLOAD_FAILED', error=str(error), upload_snapshot=str(snapshot))
        write(out/'cloud-checkpoint-errors'/f"step-{result['global_step']:07d}.json", receipt)
        print(json.dumps(dict(phase='cloud-checkpoint-upload-failed', **receipt)), flush=True)
        return receipt


def save_checkpoint(state, out, phase, run, better, best_fid):
    if phase != 'train' or run is None:
        return ORIGINAL(state, out, phase, run, better, best_fid)
    from src.training import k4_checkpoint_io as io
    out = Path(out)
    folder = out/'persistent-train/checkpoints'; folder.mkdir(parents=True, exist_ok=True)
    path = folder/f"step-{state['global_step']:07d}.pt"
    started = time.monotonic()
    io.atomic_torch_save(state, path)
    result = dict(path=str(path), global_step=int(state['global_step']), bytes=path.stat().st_size,
                  best_fid=best_fid, state='PERSISTED', persistent_checkpoint_complete=True)
    evaluation = state.get('evaluation')
    snapshot = None
    if evaluation is not None:
        cache, receipt = io._local_checkpoint_paths(path)
        identity = json.loads(receipt.read_text())
        # This writer just committed these exact serialized bytes. Object-store
        # timestamp finalization must not turn the upload into a remote reread.
        assert identity['source'] == str(path.resolve()) and cache.stat().st_size == result['bytes']
        queue = Path(os.environ['LASER_CHECKPOINT_UPLOAD_CACHE_DIR'])/'upload-queue'
        queue.mkdir(exist_ok=True)
        snapshot = queue/f"step-{state['global_step']:07d}-{time.time_ns()}.pt"
        os.link(cache, snapshot)
        result['cloud_state'] = 'QUEUED'
    else:
        result['cloud_state'] = 'NOT_REQUESTED_FOR_UNEVALUATED_CHECKPOINT'
    write(out/'checkpoint.json', result); write(folder/'latest.json', result)
    if better:
        write(folder/'best.json', result)
        run.summary.update({'selection/best_fid_with_is_preserved':best_fid,
            'selection/inception_score_at_best_fid':evaluation['inception_score'],
            'selection/is_preservation_floor':state.get('preservation_floor')})
    keep = {path}
    best = folder/'best.json'
    if best.exists(): keep.add(Path(json.loads(best.read_text())['path']))
    io.prune_local_checkpoint_cache(keep)
    for old in folder.glob('step-*.pt'):
        if old not in keep: old.unlink()
    if snapshot is not None:
        aliases = ['latest', f"step-{state['global_step']}"]
        if better: aliases.extend(['best','best-fid-is-preserved'])
        metadata = dict(global_step=state['global_step'], epoch=state['epoch'], evaluation=evaluation,
                        is_preservation_floor=state.get('preservation_floor'))
        JOBS.append(POOL.submit(publish,run,snapshot,out,result,metadata,aliases))
    print(json.dumps(dict(phase='checkpoint-persisted', seconds=time.monotonic()-started, **result)),flush=True)
    return result


def finish_uploads():
    for job in JOBS: job.result()
    POOL.shutdown(wait=True)


def install():
    global ORIGINAL
    from src.training import joint_runtime_support
    ORIGINAL = joint_runtime_support.save_checkpoint
    joint_runtime_support.save_checkpoint = save_checkpoint
