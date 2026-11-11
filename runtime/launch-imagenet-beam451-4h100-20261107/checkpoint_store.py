"""Persist complete training state without blocking updates on cloud uploads."""
import json
import os
from pathlib import Path
import shutil
import time
import torch


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.pending')
    tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(path)


def save_checkpoint(state, out, phase, run, better, best_fid):
    out = Path(out)
    base = Path(os.environ['LASER_BEAM_BASE'])
    folder = base / ('checkpoints' if phase == 'train' else 'smoke/checkpoints')
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"step-{state['global_step']:07d}.pt"
    tmp = path.with_suffix('.partial')
    with tmp.open('wb') as handle:
        torch.save(state, handle); handle.flush(); os.fsync(handle.fileno())
    tmp.replace(path)
    receipt = dict(path=str(path), global_step=int(state['global_step']), bytes=path.stat().st_size,
        epoch=int(state['epoch']), best_fid=float(best_fid), full_state=True,
        local_state='COMPLETE', saved_at=time.time())
    if phase == 'train':
        durable = out / 'persistent-train/checkpoints' / path.name
        durable.parent.mkdir(parents=True, exist_ok=True)
        temporary = durable.with_suffix('.partial')
        shutil.copyfile(path, temporary)
        assert temporary.stat().st_size == receipt['bytes']
        temporary.replace(durable)
        receipt.update(persistent_path=str(durable), persistent_state='COMPLETE')
        write(out / 'checkpoint.json', receipt)
        write(folder / 'latest.json', receipt)
        if better: write(folder / 'best.json', receipt)
        keep = {p.name for p in sorted(folder.glob('step-*.pt'))[-2:]}
        if (folder / 'best.json').exists():
            keep.add(Path(json.loads((folder / 'best.json').read_text())['path']).name)
        for root in (folder, durable.parent):
            for old in root.glob('step-*.pt'):
                if old.name not in keep: old.unlink()
        if run:
            run.summary['checkpoint/latest_step'] = receipt['global_step']
            run.summary['checkpoint/persistent_path'] = str(durable)
    else:
        write(base / 'smoke/checkpoint.json', receipt)
    print(json.dumps(dict(phase='checkpoint-complete', **receipt)), flush=True)
