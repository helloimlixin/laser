"""Keep every evaluated epoch checkpoint and verify its complete durable copy."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
import torch
from runtime.persistent_copy import copy_file


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix('.pending');pending.write_text(json.dumps(value,indent=2)+'\n');pending.replace(path)


def verify_copy(source,target):
    size=source.stat().st_size;block=32*1024**2
    def compare(first):
        count=min(block,size-first)
        with source.open('rb') as left,target.open('rb') as right:
            left.seek(first);right.seek(first);a=left.read(count);b=right.read(count)
        assert len(a)==len(b)==count and hashlib.sha256(a).digest()==hashlib.sha256(b).digest()
        return count
    with ThreadPoolExecutor(max_workers=8) as pool:verified=sum(pool.map(compare,range(0,size,block)))
    assert verified==size and target.stat().st_size==size
    with source.open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
    return digest,verified


def save_checkpoint(state,out,phase,run,better,best_fid):
    out=Path(out);base=Path(os.environ['LASER_BEAM_BASE'])
    folder=base/('checkpoints' if phase=='train' else 'smoke/checkpoints');folder.mkdir(parents=True,exist_ok=True)
    path=folder/f"step-{state['global_step']:07d}.pt";pending=path.with_suffix('.partial')
    with pending.open('wb') as stream:
        torch.save(state,stream);stream.flush();os.fsync(stream.fileno())
    pending.replace(path)
    receipt=dict(path=str(path),global_step=int(state['global_step']),bytes=path.stat().st_size,
        epoch=int(state['epoch']),best_fid=float(best_fid),full_state=True,local_state='COMPLETE',saved_at=time.time())
    if phase=='train':
        durable=out/'persistent-train/checkpoints'/path.name;durable.parent.mkdir(parents=True,exist_ok=True)
        temporary=durable.with_suffix('.partial');copy_file(path,temporary)
        assert temporary.stat().st_size==receipt['bytes']
        canonical=int(state['next_microbatch'])==0
        if canonical:
            digest,verified=verify_copy(path,temporary)
            receipt.update(sha256=digest,verified_bytes=verified,retain_evaluated_epoch=True)
        temporary.replace(durable)
        receipt.update(persistent_path=str(durable),persistent_state='COMPLETE')
        write(out/'checkpoint.json',receipt);write(folder/'latest.json',receipt)
        if better:write(folder/'best.json',receipt)
        index_path=out/'epoch-checkpoints.json'
        index=json.loads(index_path.read_text()) if index_path.exists() else {}
        if canonical:
            index[str(state['global_step'])]=dict(receipt,lr=state['optimizer']['param_groups'][0]['lr'])
            write(index_path,index)
        keep={p.name for p in sorted(folder.glob('step-*.pt'))[-2:]}
        if (folder/'best.json').exists():keep.add(Path(json.loads((folder/'best.json').read_text())['path']).name)
        keep.add('step-0042568.pt')
        if (out/'tuned-best.json').exists():
            best=json.loads((out/'tuned-best.json').read_text());keep.add(f"step-{best['global_step']:07d}.pt")
        for old in folder.glob('step-*.pt'):
            if old.name not in keep:old.unlink()
        durable_keep=keep|{Path(r['persistent_path']).name for r in index.values()}
        for old in durable.parent.glob('step-*.pt'):
            if old.name not in durable_keep:old.unlink()
        if run:
            run.summary['checkpoint/latest_step']=receipt['global_step']
            run.summary['checkpoint/persistent_path']=str(durable)
            run.summary['checkpoint/retained_evaluated_epochs']=len(index)
    else:write(base/'smoke/checkpoint.json',receipt)
    print(json.dumps(dict(phase='checkpoint-complete',**receipt)),flush=True)
