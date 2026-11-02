import copy
import ast
from pathlib import Path

import pytest
import torch

from src.training.exact_global_batch import ExactGlobalBatchSampler
from src.training.portable_imagenet_recovery import remap_payload, restore_rng, validate_layout


def payload(tmp_path, *, batch=68):
    winner = tmp_path / 'best.pt'
    winner.touch()
    return dict(epoch=14, batch_idx=batch, global_step=14 * 626 + batch // 4,
                state_dict={'weight': torch.ones(2)}, optimizer={'state': {'moments': torch.ones(2)}},
                config=dict(exact_global_batch=True, total_batch_size=2048,
                            training_images=1281167, world_size=4, accumulation_steps=4,
                            fid_reference_stats='/old/reference.npz'),
                best_fid=[(38., '/old/best.pt')], best_inception=[(33., '/old/best.pt')],
                rng_state_by_rank=[dict(torch_cpu=torch.Generator().manual_seed(r).get_state())
                                   for r in range(4)])


@pytest.mark.parametrize('world,accumulation', [(4, 4), (8, 2), (16, 1), (32, 1)])
def test_changed_layout_preserves_remaining_global_batches_and_training_state(tmp_path, world, accumulation):
    old = payload(tmp_path)
    new = remap_payload(old, world=world, accumulation=accumulation,
                        checkpoint_dir=tmp_path, reference=tmp_path/'reference.npz', reference_sha256='same')
    assert old['batch_idx'] == 68
    assert new['batch_idx'] == 17 * accumulation
    assert new['global_step'] == old['global_step']
    assert new['optimizer'] is old['optimizer'] and new['state_dict'] is old['state_dict']
    assert new['best_inception'][0][1] == str(tmp_path/'best.pt')
    # Each layout consumes the same shuffled global batch, with no image skipped/repeated.
    def indices(w, a):
        result=[]
        for rank in range(w):
            sampler=ExactGlobalBatchSampler(range(42000),2048,w,rank,a,seed=261001)
            sampler.set_epoch(14)
            sampler.set_start_batch(17*a)
            iterator=iter(sampler)
            result.extend(i for _ in range(a) for i in next(iterator))
        return sorted(result)
    assert indices(4,4) == indices(world,accumulation)


def test_epoch_end_cursor_is_normalized_without_replaying_the_last_batch(tmp_path):
    new=remap_payload(payload(tmp_path,batch=626*4),world=8,accumulation=2,
                      checkpoint_dir=tmp_path,reference='new',reference_sha256='same')
    assert (new['epoch'],new['batch_idx'],new['global_step'])==(15,0,15*626)


def test_inconsistent_step_and_partial_optimizer_boundary_are_rejected(tmp_path):
    old=payload(tmp_path)
    old['global_step']+=1
    with pytest.raises(ValueError,match='cursor disagree'):
        remap_payload(old,world=8,accumulation=2,checkpoint_dir=tmp_path,reference='new',reference_sha256='same')
    old=payload(tmp_path,batch=69)
    with pytest.raises(ValueError,match='optimizer boundary'):
        remap_payload(old,world=8,accumulation=2,checkpoint_dir=tmp_path,reference='new',reference_sha256='same')


def test_gpu_expansion_retains_old_rng_and_derives_distinct_reproducible_new_streams(tmp_path,monkeypatch):
    old=payload(tmp_path)
    monkeypatch.setenv('WORLD_SIZE','8')
    draws={}
    for rank in [0,3,4,5,7]:
        monkeypatch.setenv('RANK',str(rank))
        assert restore_rng(old,torch.device('cpu'),261001)=='resized'
        if rank<4: assert torch.equal(torch.get_rng_state(),old['rng_state_by_rank'][rank]['torch_cpu'])
        draws[rank]=torch.rand(16)
        restore_rng(old,torch.device('cpu'),261001)
        assert torch.equal(torch.rand(16),draws[rank])
    assert all(not torch.equal(draws[a],draws[b]) for a in draws for b in draws if a!=b)


def test_global_batch_change_and_unmapped_layout_change_are_rejected(tmp_path):
    old=payload(tmp_path)['config']
    with pytest.raises(ValueError,match='original exact global batch'):
        validate_layout(old,8,2,4096,0)
    with pytest.raises(ValueError,match='remapped recovery cursor'):
        validate_layout(old,8,2,2048,0)


def test_portable_storage_guard_accepts_explicit_new_machine_root_only(tmp_path):
    from scripts.tools.imagenet_portable_resume import make_entry
    original="""assert world == 4 and backend == 'ddp'
native_checkpoint_load = torch.load
def load():
    if True:
        payload = original_logging.rebase_checkpoint_fids(payload, policy['previous_evaluations'])
"""
    tree=ast.parse(make_entry(original))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='portable_checkpoint_directory')
    namespace={'OUT':tmp_path,'Path':Path}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'guard','exec'),namespace)
    expected=tmp_path/'train/checkpoints'
    assert namespace['portable_checkpoint_directory'](tmp_path/'train',expected)==expected
    with pytest.raises(ValueError,match='explicit recovery output root'):
        namespace['portable_checkpoint_directory'](tmp_path/'train',tmp_path/'other')
