import importlib.util
from pathlib import Path

import pytest


spec=importlib.util.spec_from_file_location('site_pooling_trial',
    Path(__file__).parents[1]/'scripts/tools/run_imagenet_site_pooling_trial.py')
trial=importlib.util.module_from_spec(spec)
spec.loader.exec_module(trial)


def test_anchor_requires_completed_epoch12_and_all_best_snapshots(tmp_path):
    best=tmp_path/'best.pt'
    payload=dict(epoch=12,global_step=7512,best_fid=[(40,str(best))],best_inception=[])
    assert not trial.ready_for_anchor(payload,'Epoch 12: FID=40.0; saved last.pt')
    best.write_bytes(b'complete')
    assert trial.ready_for_anchor(payload,'Epoch 12: FID=40.0; saved last.pt')
    assert not trial.ready_for_anchor(payload,'FID generation still in progress')
    assert not trial.ready_for_anchor(dict(payload,epoch=11),'Epoch 12: FID=40.0; saved last.pt')
    assert not trial.ready_for_anchor(dict(payload,global_step=7500),'Epoch 12: FID=40.0; saved last.pt')


@pytest.mark.parametrize('better',[False,True])
def test_evaluated_plain_preserves_training_objects_and_selects_best(better,tmp_path):
    payload=dict(global_step=9560,state_dict={},optimizer={},scheduler={},rng_state_by_rank=[],config={},
        best_fid=[(40.,'/original/best-fid.pt')],best_inception=[(25.,'/original/best-is.pt')])
    evaluation=dict(global_step=9560,fid_seed=261001,fid=35. if better else 45.,
                    inception_score=27. if better else 23.,inception_score_std=.3)
    result,links=trial.evaluated_plain_payload(payload,evaluation,tmp_path/'ready.pt')
    assert all(result[k] is payload[k] for k in ('state_dict','optimizer','scheduler','rng_state_by_rank','config'))
    assert result['fid']==evaluation['fid']
    if better:
        assert result['best_fid'][0][0]==35. and result['best_inception'][0][0]==27.
        assert len(links)==2
    else:
        assert result['best_fid']==payload['best_fid'] and result['best_inception']==payload['best_inception']
        assert not links
    with pytest.raises(AssertionError):
        trial.evaluated_plain_payload(payload,dict(evaluation,fid_seed=271001),tmp_path/'ready.pt')


def test_checkpoint_cleanup_preserves_shared_anchor_and_best_aliases(tmp_path):
    import ast
    import os
    import tempfile
    from types import SimpleNamespace
    entry=Path(__file__).parents[1]/'scripts/tools/imagenet_site_pooling_entry.py'
    tree=ast.parse(entry.read_text())
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='snapshot_checkpoint')
    out=tmp_path/'trial';folder=tmp_path/'production/checkpoints/.checkpoint-data'
    folder.mkdir(parents=True);out.mkdir()
    staging=tmp_path/'staging';staging.mkdir()
    source=folder/'last-payload.pt';source.write_bytes(b'checkpoint')
    local=tmp_path/'serialized.pt';local.write_bytes(b'checkpoint')
    (folder.parent/'last.pt').symlink_to(source)
    anchor=folder/'anchor-payload.pt';anchor.write_bytes(b'anchor')
    (out/'anchor.pt').symlink_to(anchor)
    previous_best=folder/'previous-best.pt';previous_best.write_bytes(b'best')
    aliases=out/'sum/train/checkpoints';aliases.mkdir(parents=True)
    (aliases/'previous-best.pt').symlink_to(previous_best)
    orphan=folder/'orphan.pt';orphan.write_bytes(b'orphan')
    cache=tmp_path/'cache';cache.mkdir()
    def cache_paths(path):return [cache/(path.name+'.cache'),cache/(path.name+'.json')]
    for p in cache_paths(orphan):p.write_bytes(b'cold cache')
    def persist(serialized,destination):
        payload=folder/'new-best-payload.pt';payload.write_bytes(serialized.read_bytes())
        destination.symlink_to(payload);serialized.unlink()
    io=SimpleNamespace(_checkpoint_upload_source=lambda p:local,
        _persist_serialized_checkpoint=persist,_local_checkpoint_paths=cache_paths)
    namespace=dict(Path=Path,OUT=out,PLAN={'fid_seeds':[261001,271001]},
        os=os,tempfile=tempfile,checkpoint_io=io)
    original=os.environ.get('LASER_CHECKPOINT_STAGING_DIR')
    os.environ['LASER_CHECKPOINT_STAGING_DIR']=str(staging)
    try:
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(entry),'exec'),namespace)
        namespace['snapshot_checkpoint'](source,folder.parent/'best_fid_new.pt')
    finally:
        if original is None:os.environ.pop('LASER_CHECKPOINT_STAGING_DIR')
        else:os.environ['LASER_CHECKPOINT_STAGING_DIR']=original
    assert all(p.is_file() for p in [source,anchor,previous_best,folder/'new-best-payload.pt'])
    assert not orphan.exists() and all(not p.exists() for p in cache_paths(orphan))
