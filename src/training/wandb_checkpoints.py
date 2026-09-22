"""Publish immutable checkpoint snapshots and record confirmed online versions."""
from pathlib import Path
import time

from src.original_rq_training import atomic_json


def publish_checkpoint_bundle(run, name, paths, epoch, *, metadata, extras,
                              receipt_path, best_name, best_alias):
    import wandb
    artifact = wandb.Artifact(name, type='model', metadata=dict(epoch=epoch, **metadata))
    names = set()
    for path in paths:
        path = Path(path)
        if path.exists():
            artifact.add_file(str(path), name=path.name, policy='immutable')
            names.add(path.name)
    for path in extras:
        path = Path(path)
        if path.exists():
            artifact.add_file(str(path), name=path.name)
    aliases = ['latest', 'last', f'epoch-{epoch:03d}']
    if best_name in names:
        aliases.append(best_alias)
    run.log_artifact(artifact, aliases=aliases).wait()
    atomic_json(receipt_path, dict(artifact=artifact.qualified_name, epoch=epoch,
        state=artifact.state, uploaded_unix=time.time(),
        files={name:dict(bytes=entry.size, digest=entry.digest)
               for name, entry in artifact.manifest.entries.items()}))
