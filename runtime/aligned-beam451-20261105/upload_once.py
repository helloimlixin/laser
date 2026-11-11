"""Publish a local checkpoint as verified 1GiB binary parts, then commit a receipt."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def write(path,data):
    temp=path.with_suffix('.pending');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def main(job_path):
    out=job_path.parent.parent
    job=json.loads(job_path.read_text());source=Path(job['path'])
    folder=out/'cloud-parts'/source.stem;folder.mkdir(parents=True,exist_ok=True)
    manifest_path=folder/'checkpoint-manifest.json'
    if not manifest_path.exists():
        parts=[];total=hashlib.sha256()
        with source.open('rb') as handle:
            index=0
            while True:
                first=handle.read(16*1024**2)
                if not first:break
                path=folder/f'last.pt.part-{index:03d}';temp=path.with_suffix('.partial')
                digest=hashlib.sha256();size=0;data=first
                with temp.open('wb') as part:
                    while data:
                        part.write(data);digest.update(data);total.update(data);size+=len(data)
                        if size>=1024**3:break
                        data=handle.read(min(16*1024**2,1024**3-size))
                temp.replace(path)
                parts.append(dict(name=path.name,bytes=size,sha256=digest.hexdigest()));index+=1
        manifest=dict(format='binary-concatenation-v1',file='last.pt',bytes=source.stat().st_size,
            sha256=total.hexdigest(),global_step=job['global_step'],run_id=job['run_id'],parts=parts)
        assert sum(p['bytes'] for p in parts)==manifest['bytes']
        write(manifest_path,manifest)
    manifest=json.loads(manifest_path.read_text())
    os.environ.update(WANDB_API_KEY=Path('/root/.config/laser/wandb.key').read_text().strip(),
        WANDB_BASE_URL='https://api.forge.coreweave.com',WANDB_CONSOLE='off',WANDB_DISABLE_GIT='true',
        WANDB_CACHE_DIR=str(out/'upload-cache'),CUDA_VISIBLE_DEVICES='')
    import wandb
    run=wandb.init(entity='helloimlixin-rutgers',project='laser',id=job['run_id']+'-checkpoint-sync',
        name='Checkpoint sync for '+job['run_id'],job_type='checkpoint-upload',resume='allow',
        dir=str(out),config=dict(primary_training_run=job['run_id'],format=manifest['format']),
        settings=wandb.Settings(init_timeout=120))
    artifact=wandb.Artifact(job['run_id']+'-checkpoint',type='model',metadata=dict(
        global_step=job['global_step'],epoch=job['epoch'],primary_training_run=job['run_id'],
        format=manifest['format'],checkpoint_sha256=manifest['sha256'],checkpoint_bytes=manifest['bytes']))
    for part in manifest['parts']:
        artifact.add_file(str(folder/part['name']),name=part['name'],policy='immutable',skip_cache=True)
    artifact.add_file(str(manifest_path),name=manifest_path.name,policy='immutable',skip_cache=True)
    aliases=['latest',f"step-{job['global_step']}"]
    if job['better']:aliases.append('best')
    artifact=run.log_artifact(artifact,aliases=aliases);artifact.wait(timeout=1800)
    assert str(artifact.state)=='COMMITTED'
    files={file.name:file.size for file in artifact.files()}
    assert all(files[part['name']]==part['bytes'] for part in manifest['parts'])
    receipt=dict(state='COMMITTED',artifact=artifact.qualified_name,global_step=job['global_step'],
        path=str(source),bytes=manifest['bytes'],sha256=manifest['sha256'],format=manifest['format'],
        parts=len(manifest['parts']),time=time.time())
    receipts=out/'cloud-receipts';receipts.mkdir(exist_ok=True)
    write(receipts/job_path.name,receipt)
    write(out/'cloud-sync-status.json',receipt)
    run.summary['checkpoint/last_committed_step']=job['global_step'];run.finish()
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('job',type=Path);args=parser.parse_args()
    main(args.job)
