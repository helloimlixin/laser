from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.tools.fid14_native_checkpoint_io import install


def test_stale_public_alias_cannot_corrupt_best_snapshot_or_upload_source(tmp_path):
    io=SimpleNamespace(_checkpoint_upload_source=lambda path:path)
    training=SimpleNamespace()
    g=dict(checkpoint_io=io,training=training,BASE=tmp_path,EVIDENCE=tmp_path,
        WRITER=SimpleNamespace(append=lambda fn:fn()),submit_upload=lambda:None)
    registry=install(g)
    target=tmp_path/'native-train/checkpoints/last.pt'
    def persist(step):
        file=tmp_path/f'local{step}.pt'
        torch.save(dict(global_step=step,epoch=0,state_dict=dict(x=torch.arange(8))),file)
        io._persist_serialized_checkpoint(file,target)
        return Path(registry['committed'][str(target)]['persistent'])
    first=persist(1)
    second=persist(2)
    # Reproduce the mount returning the previous alias after a replacement.
    target.unlink();target.symlink_to(first)
    assert torch.load(target,weights_only=True)['global_step']==1
    assert torch.load(io._checkpoint_upload_source(target),weights_only=True)['global_step']==2
    best=target.parent/'best.pt'
    training.snapshot_checkpoint(target,best)
    assert best.resolve()==second and torch.load(best,weights_only=True)['global_step']==2
    assert torch.load(io._checkpoint_upload_source(best),weights_only=True)['global_step']==2
    training.remove_checkpoint(best)
    assert first.exists() and second.exists()
    restarted=SimpleNamespace(_checkpoint_upload_source=lambda path:path)
    install(dict(g,checkpoint_io=restarted,training=SimpleNamespace()))
    assert torch.load(restarted._checkpoint_upload_source(target),weights_only=True)['global_step']==2
