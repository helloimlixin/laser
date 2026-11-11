"""One-time migration from correlated training RNG to independent rank streams."""
from pathlib import Path
import difflib
import shutil

ROOT=Path('/workspace/Projects/laser')
BASE=Path('/tmp/laser-beam451-stage2-4h100-20261107')
AUDIT=ROOT/'outputs/imagenet-beam451-aligned-joint16k-scratch100-4h100-20261107/audit-stage1-preservation-20261108'


def main():
    path=BASE/'source/src/training/rq_imagenet_shared.py'
    old=path.read_text()
    assert 'rng_migration_needed' not in old,'Migration already installed'
    code=old
    def replace(a,b):
        nonlocal code
        assert code.count(a)==1,(a,code.count(a))
        code=code.replace(a,b)
    replace('    completed_epoch=start_micro=global_step=0',
        '    rng_migration_needed=False\n    rng_seed_base=261001\n    rng_transition_step=None\n    completed_epoch=start_micro=global_step=0')
    replace("        restored_evaluation=payload.get('evaluation')", """        restored_evaluation=payload.get('evaluation')
        from src.training.rq_rng import POLICY
        rng_migration_needed=payload['runtime'].get('training_rng_policy')!=POLICY
        rng_seed_base=(261001+1000003*global_step if rng_migration_needed
                       else int(payload['runtime']['training_rng_seed_base']))
        rng_transition_step=(global_step if rng_migration_needed
                             else payload['runtime'].get('training_rng_transition_step'))""")
    replace("    if not args.resume:assert not optimizer.state and optimizer.param_groups[0]['lr']==.0005", """    from src.training.rq_rng import configure_training_rng,audit_training_rng
    rng_metadata=configure_training_rng(rng_seed_base,rank,
        resuming=bool(args.resume) and not rng_migration_needed)
    rng_metadata['training_rng_transition_step']=rng_transition_step
    rng_proof=audit_training_rng(device)
    rng_proofs=[None]*world
    dist.all_gather_object(rng_proofs,rng_proof)
    assert len({value['rng_state_sha256'] for value in rng_proofs})==world
    assert len({value['dropout_probe_sha256'] for value in rng_proofs})==world
    write_json(folder/f'rng-startup-rank{rank}.json',dict(rank=rank,**rng_metadata,**rng_proof,
        independent_streams_verified_across_ranks=True))
    if not args.resume:assert not optimizer.state and optimizer.param_groups[0]['lr']==.0005""")
    replace('topology_change_is_not_bitwise_replay=True,**PROFILE.metadata())',
        'topology_change_is_not_bitwise_replay=True,**PROFILE.metadata(),**rng_metadata)')
    replace("        run.define_metric('train/global_step');run.define_metric('train/epoch')", """        run.config.update(rng_metadata,allow_val_change=True)
        run.summary.update({'training/rng_policy':rng_metadata['training_rng_policy'],
            'training/rng_transition_step':rng_transition_step,
            'training/independent_rank_rng_verified':True})
        run.define_metric('train/global_step');run.define_metric('train/epoch')""")
    compile(code,str(path),'exec')
    AUDIT.mkdir(parents=True,exist_ok=True)
    (AUDIT/'rng-runtime.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),code.splitlines(True),
        fromfile='original/rq_imagenet_shared.py',tofile='independent_rng/rq_imagenet_shared.py')))
    shutil.copyfile(ROOT/'src/training/rq_rng.py',BASE/'source/src/training/rq_rng.py')
    path.write_text(code)
    shutil.copyfile(Path(__file__),BASE/'runtime/patch_rng.py')
    print('Installed migration; current processes retain original code until the checkpoint restart.')


if __name__=='__main__':main()
