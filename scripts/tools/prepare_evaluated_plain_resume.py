"""Persist an evaluated, fully resumable plain checkpoint for production."""
import json
import os
from pathlib import Path
import sys


def main():
    base=Path('/tmp/laser-imagenet-pair-memory-investigation-20261007')
    out=Path('/workspace/Projects/laser/outputs/imagenet-pair-memory-investigation-20261007')
    production=Path('/workspace/Projects/laser/outputs/imagenet-rfid421-classcond-8h100-20261007')
    sys.path[:0]=[str(base),str(base/'source'),str(base/'source/runtime')]
    import torch
    from src.training import k4_checkpoint_io as io
    from restore_evaluated_plain import restore_plain_best
    torch.set_num_threads(4)
    os.environ.update(LASER_BRANCH='baseline',LASER_PHASE='train',LASER_WANDB_RESUME='must',RANK='prepare',
        LASER_CHECKPOINT_STAGING_DIR=str(base/'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR='/tmp/laser-imagenet-pair-memory-20261007/checkpoint-cache',
        LASER_CHECKPOINT_IMMUTABLE_FILES='1')
    metadata=json.loads((out/'production-best-metadata.json').read_text())
    proof=json.loads((out/'baseline-endpoint-verification.json').read_text())
    original=torch.load(proof['local_serialization'],map_location='cpu',mmap=True,weights_only=False)
    payload=restore_plain_best(original,Path(proof['checkpoint']),out,lambda *args:None)
    assert payload is not original
    ready=out/'baseline/continuation-ready.pt'
    assert not ready.exists(), 'Evaluated production checkpoint is immutable and single-use'
    io.atomic_torch_save(payload,ready)
    local=io._checkpoint_upload_source(ready.resolve())
    assert local!=ready.resolve()
    recovered=torch.load(local,map_location='cpu',mmap=True,weights_only=False)
    assert recovered['global_step']==recovered['scheduler']['last_epoch']==3798
    assert recovered['fid']==metadata['fid'] and recovered['best_fid']==payload['best_fid']
    assert len(recovered['optimizer']['state'])==798
    assert {int(v['step']) for v in recovered['optimizer']['state'].values()}=={3798}
    for name,value in original['state_dict'].items():
        assert torch.equal(value,recovered['state_dict'][name]), name
    for index,state in original['optimizer']['state'].items():
        for name,value in state.items():
            other=recovered['optimizer']['state'][index][name]
            assert torch.equal(value,other) if torch.is_tensor(value) else value==other
    source=ready.resolve()
    for path in (Path(metadata['best_fid_checkpoint']),Path(metadata['best_inception_checkpoint']),
                 production/'train/checkpoints/last.pt'):
        temporary=path.with_name(path.name+'.evaluated-link.tmp')
        temporary.unlink(missing_ok=True);temporary.symlink_to(source);temporary.replace(path)
        assert path.resolve()==source
    metadata.update(continuation_checkpoint=str(source),continuation_local_serialization=str(local))
    (out/'production-best-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    report=dict(passed=True,global_step=3798,fid=metadata['fid'],inception_score=metadata['inception_score'],
        every_model_tensor_bitwise_equal=True,every_optimizer_tensor_bitwise_equal=True,
        optimizer_states=798,scheduler_age=3798,rng_ranks=len(recovered['rng_state_by_rank']),
        full_checkpoint=str(source),local_serialization=str(local),production_last_points_to_evaluated_state=True,
        best_FID_and_IS_references_retained=True)
    (out/'evaluated-production-checkpoint-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
