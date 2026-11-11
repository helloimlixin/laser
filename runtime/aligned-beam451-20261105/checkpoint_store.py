"""Save locally first; cloud publication is an independent retryable operation."""
import json
from pathlib import Path
import shutil
import time
import torch


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.pending');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def save_checkpoint(state,out,phase,run,better,best_fid):
    out=Path(out)
    folder=out/'persistent-train/checkpoints' if phase=='train' else out/'smoke/checkpoints'
    folder.mkdir(parents=True,exist_ok=True)
    if phase=='train':
        # Leave room for the next full optimizer state and its upload parts.
        while shutil.disk_usage(out).free < 128*1024**3:
            write(out/'checkpoint-storage-wait.json',dict(state='WAITING',time=time.time(),
                reason='Waiting for committed cloud backups to release disk space',
                global_step=int(state['global_step']),free_bytes=shutil.disk_usage(out).free))
            time.sleep(30)
        if (out/'checkpoint-storage-wait.json').exists():
            write(out/'checkpoint-storage-wait.json',dict(state='READY',time=time.time()))
    path=folder/f"step-{state['global_step']:07d}.pt";temp=path.with_suffix('.partial')
    torch.save(state,temp);temp.replace(path)
    result=dict(path=str(path),global_step=int(state['global_step']),bytes=path.stat().st_size,
        best_fid=float(best_fid),saved_at=time.time(),local_state='COMPLETE',cloud_state='PENDING' if run else 'NOT_REQUESTED')
    write(out/'checkpoint.json' if phase=='train' else out/'smoke/checkpoint.json',result)
    write(folder/'latest.json',result)
    if better:write(folder/'best.json',result)
    if run is not None:
        write(out/'cloud-queue'/f"step-{state['global_step']:07d}.json",
              dict(result,run_id=state['run_id'],epoch=int(state['epoch']),better=bool(better)))
    print(json.dumps(dict(phase='checkpoint-local-complete',**result)),flush=True)
