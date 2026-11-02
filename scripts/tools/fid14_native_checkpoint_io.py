"""Immutable payloads and in-process alias tracking for the resumed RQ run.

Do not read a just-replaced public alias to decide what to snapshot/upload:
the workspace filesystem can return its previous small-file contents. Keep
the committed immutable payload identity until all dependent actions finish.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid


def install(g):
    io=g['checkpoint_io'];training=g['training'];base=g['BASE']
    cache=base/'native-checkpoint-cache';cache.mkdir(exist_ok=True)
    committed={};by_payload={}
    fallback=io._checkpoint_upload_source

    def key(path):return str(Path(path).absolute())

    def alias(payload,target):
        temporary=target.with_name(target.name+'.'+uuid.uuid4().hex+'.link')
        temporary.symlink_to(os.path.relpath(payload,target.parent))
        temporary.replace(target)

    def persist(serialized,target):
        serialized,target=Path(serialized),Path(target)
        target.parent.mkdir(parents=True,exist_ok=True)
        folder=target.parent/'.checkpoint-data';folder.mkdir(exist_ok=True)
        identity=uuid.uuid4().hex
        payload=folder/(target.stem+'-'+identity+'.pt')
        local=cache/(identity+'.pt')
        size=serialized.stat().st_size
        with serialized.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
        for attempt in range(3):
            try:
                shutil.copyfile(serialized,payload)
                assert payload.stat().st_size==size
                with serialized.open('rb') as source,payload.open('rb') as durable:
                    assert source.read(64)==durable.read(64)
                    source.seek(-64,2);durable.seek(-64,2)
                    assert source.read(64)==durable.read(64)
                break
            except (OSError,AssertionError):
                payload.unlink(missing_ok=True)
                if attempt==2:raise
                time.sleep(2*(attempt+1))
        os.link(serialized,local)
        record=dict(local=str(local),persistent=str(payload),bytes=size,sha256=sha,
            public_alias=str(target),time=time.time())
        import torch
        metadata=torch.load(serialized,map_location='cpu',mmap=True,weights_only=False)
        record.update(global_step=int(metadata['global_step']),epoch=int(metadata['epoch']))
        del metadata
        # Publish only after the immutable payload and local upload inode exist.
        receipt=folder/(identity+'.json')
        receipt.write_text(json.dumps(record,indent=2)+'\n')
        committed[key(target)]=record;by_payload[key(payload)]=record
        alias(payload,target)
        serialized.unlink()

    def upload_source(path):
        record=committed.get(key(path))
        if record is None:
            record=by_payload.get(key(Path(path).resolve()))
        if record is not None:
            local=Path(record['local'])
            assert local.stat().st_size==record['bytes']
            return local
        return fallback(Path(path))

    def snapshot(source,target):
        def commit():
            record=committed.get(key(source))
            if record is None:
                resolved=Path(source).resolve(strict=True)
                record=by_payload.get(key(resolved),dict(persistent=str(resolved),local=str(upload_source(source)),
                    bytes=resolved.stat().st_size))
            payload=Path(record['persistent'])
            assert payload.is_file()
            committed[key(target)]=dict(record)
            alias(payload,Path(target))
            g['submit_upload']()
        g['WRITER'].append(commit)

    imported=g['EVIDENCE']/'native-source-import.json'
    if imported.exists():
        record=json.loads(imported.read_text())
        assert record['passed']
        by_payload[key(record['persistent'])]=record
    # A restarted worker must recover local upload inodes for every retained
    # winner, rather than trying to hard-link a workspace file onto NVMe.
    for name in ('native-train','native-smoke'):
        folder=g['EVIDENCE']/name/'checkpoints/.checkpoint-data'
        for receipt in folder.glob('*.json'):
            record=json.loads(receipt.read_text())
            local,payload=Path(record['local']),Path(record['persistent'])
            if not (local.is_file() and payload.is_file()
                    and local.stat().st_size==payload.stat().st_size==record['bytes']):
                continue
            by_payload[key(payload)]=record
            public=key(record['public_alias'])
            previous=committed.get(public)
            if previous is None or record['global_step']>previous['global_step']:
                committed[public]=record
    io._persist_serialized_checkpoint=persist
    io._checkpoint_upload_source=upload_source
    training.snapshot_checkpoint=snapshot
    # Retain immutable payloads for this bounded 22-epoch continuation. Alias
    # deletion must never invalidate a checkpoint or an upload still in flight.
    training.remove_checkpoint=lambda path:Path(path).unlink(missing_ok=True)
    return dict(committed=committed,by_payload=by_payload)
