"""Prune redundant local copies only after verifying their committed backup."""
import hashlib
import json
from pathlib import Path
import shutil
import time


def cleanup(out):
    out=Path(out);folder=out/'persistent-train/checkpoints'
    checkpoints=sorted(folder.glob('step-*.pt'),reverse=True)
    keep=set(checkpoints[:2])
    for name in ['latest.json','best.json']:
        p=folder/name
        if p.exists():keep.add(Path(json.loads(p.read_text())['path']))
    removed=[]
    for path in checkpoints:
        receipt=out/'cloud-receipts'/f'{path.stem}.json'
        if not receipt.exists() or path in keep:continue
        record=json.loads(receipt.read_text())
        if record.get('state')!='COMMITTED' or not record.get('artifact'):continue
        if record.get('path')!=str(path) or record.get('bytes')!=path.stat().st_size:continue
        with path.open('rb') as handle:actual=hashlib.file_digest(handle,'sha256').hexdigest()
        if actual!=record.get('sha256'):continue
        path.unlink();removed.append(str(path))
    # Uploaded binary parts are redundant even when the complete file is kept.
    for receipt in (out/'cloud-receipts').glob('step-*.json'):
        record=json.loads(receipt.read_text())
        parts=out/'cloud-parts'/receipt.stem
        manifest=parts/'checkpoint-manifest.json'
        if record.get('state')!='COMMITTED' or not record.get('artifact') or not manifest.exists():continue
        spec=json.loads(manifest.read_text())
        if spec.get('sha256')!=record.get('sha256') or spec.get('bytes')!=record.get('bytes'):continue
        shutil.rmtree(parts);removed.append(str(parts))
    if removed:
        with (out/'retention.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(time=time.time(),removed=removed,
                kept_latest=2,kept_best=True,verified_cloud_backup_required=True))+'\n')
    return removed
