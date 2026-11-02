"""Immutable RQ checkpoint identities independent of mutable FUSE aliases."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid


def install(io, base, out):
    base, out = Path(base), Path(out)
    records, aliases = {}, {}
    fallback = io._checkpoint_upload_source

    def key(path):
        return str(Path(path).absolute())

    def register(record):
        records[record['persistent']] = record
        previous = aliases.get(record['public_alias'])
        if previous is None or record['global_step'] > previous['global_step']:
            aliases[record['public_alias']] = record

    for directory in ('persistent-train', 'smoke'):
        for receipt in (out/directory/'checkpoints/.checkpoint-data').glob('*.json'):
            record = json.loads(receipt.read_text())
            if Path(record['persistent']).is_file():
                register(record)
    evaluated = [r for r in records.values() if r.get('fid') is not None]
    if evaluated:
        best = min(evaluated, key=lambda r: r['fid'])
        aliases[key(out/'persistent-train/checkpoints/best.pt')] = best

    def link(source, target):
        record = aliases.get(key(source)) or records.get(key(source))
        assert record is not None
        target = Path(target)
        temporary = target.with_name(target.name+'.'+uuid.uuid4().hex+'.link')
        temporary.symlink_to(os.path.relpath(record['persistent'], target.parent))
        temporary.replace(target)
        aliases[key(target)] = record

    def persist(serialized, target):
        import torch
        serialized, target = Path(serialized), Path(target)
        folder = target.parent/'.checkpoint-data'
        folder.mkdir(parents=True, exist_ok=True)
        identity = uuid.uuid4().hex
        payload = folder/f'{target.stem}-{identity}.pt'
        size = serialized.stat().st_size
        with serialized.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        for attempt in range(3):
            try:
                shutil.copyfile(serialized, payload)
                assert payload.stat().st_size == size
                with serialized.open('rb') as original, payload.open('rb') as durable:
                    assert original.read(64) == durable.read(64)
                    original.seek(-64, 2)
                    durable.seek(-64, 2)
                    assert original.read(64) == durable.read(64)
                break
            except (OSError, AssertionError):
                payload.unlink(missing_ok=True)
                if attempt == 2:
                    raise
                time.sleep(2*(attempt+1))
        local, local_receipt = io._local_checkpoint_paths(payload)
        local.parent.mkdir(parents=True, exist_ok=True)
        os.link(serialized, local)
        metadata = torch.load(serialized, mmap=True, map_location='cpu', weights_only=False)
        evaluation = metadata.get('evaluation') or {}
        record = dict(persistent=key(payload), local=key(local), bytes=size, sha256=digest,
            public_alias=key(target), global_step=int(metadata['global_step']),
            epoch=int(metadata['epoch']), fid=evaluation.get('fid'), time=time.time())
        local_receipt.write_text(json.dumps(dict(source=key(payload),
            identity=io._checkpoint_file_identity(payload)))+'\n')
        (folder/f'{identity}.json').write_text(json.dumps(record, indent=2)+'\n')
        register(record)
        link(payload, target)
        serialized.unlink()

    def source(path):
        record = aliases.get(key(path)) or records.get(key(path))
        if record and Path(record['local']).exists():
            return Path(record['local'])
        return fallback(Path(path))

    def prune():
        milestones = {626*e for e in (1, 2, 4, 6, 10, 20, 40, 60, 80, 100)}
        keep = {r['persistent'] for r in aliases.values()}
        keep.update(r['persistent'] for r in records.values() if r['global_step'] in milestones)
        return io.prune_local_checkpoint_cache(keep)

    io._persist_serialized_checkpoint = persist
    io._checkpoint_upload_source = source
    io.link_committed_checkpoint = link
    io.prune_committed_checkpoint_cache = prune

