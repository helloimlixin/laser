import copy

import pytest
import torch
from omegaconf import OmegaConf

from src.training.exact_global_batch import ExactGlobalBatchSampler
from src.training.microbatch_migration import remap_microbatch_cursor
from src.training.imagenet_compound_runtime import validate_resume_recipe


def saved(accumulation=8):
    return dict(exact_global_batch=True, world_size=8, accumulation_steps=accumulation,
                total_batch_size=2048, batch_size=256//accumulation)


@pytest.mark.parametrize('new_accumulation', [2, 4, 8])
def test_remaining_global_images_and_final_partial_batch_are_identical(new_accumulation):
    dataset = range(2048*2+1167)
    results = []
    for accumulation in [8, new_accumulation]:
        batches = []
        cursor = remap_microbatch_cursor(saved(), world=8, accumulation=accumulation,
                                          total_batch=2048, batch_idx=8)
        for rank in range(8):
            sampler = ExactGlobalBatchSampler(dataset, 2048, 8, rank, accumulation, 37)
            sampler.set_epoch(3); sampler.set_start_batch(cursor)
            batches.append(list(sampler))
        global_batches = []
        for step in range(2):
            global_batches.append(sorted(i for rank_batches in batches
                for micro in rank_batches[step*accumulation:(step+1)*accumulation]
                for i in micro))
        results.append(global_batches)
    assert results[0] == results[1]
    assert list(map(len, results[1])) == [2048, 1167]


@pytest.mark.parametrize('change', [dict(batch_idx=7), dict(world=4), dict(total_batch=1024),
                                   dict(accumulation=0)])
def test_unsafe_migrations_are_rejected(change):
    kwargs=dict(world=8, accumulation=4, total_batch=2048, batch_idx=4000)
    kwargs.update(change)
    with pytest.raises(ValueError): remap_microbatch_cursor(saved(), **kwargs)


def test_layout_guard_is_explicit_and_leaves_original_layout_unmodified(monkeypatch):
    from runtime.layout_migration import validate_resume_layout
    s = saved(); original = copy.deepcopy(s)
    monkeypatch.delenv('LASER_ALLOW_ACCUMULATION_CHANGE', raising=False)
    with pytest.raises(ValueError): validate_resume_layout(s,8,4,2048,4000)
    monkeypatch.setenv('LASER_ALLOW_ACCUMULATION_CHANGE','1')
    assert validate_resume_layout(s,8,4,2048,4000) is True
    assert s == original
    assert remap_microbatch_cursor(s,world=8,accumulation=4,total_batch=2048,batch_idx=4000) == 2000


def test_recipe_guard_allows_execution_changes_and_rejects_objective_changes():
    cfg=OmegaConf.load('configs/stage2/imagenet-compound512-audited-8a100-20261026.yaml').options
    prior=OmegaConf.to_container(cfg)
    cfg.batch_size=128; cfg.resume=True
    validate_resume_recipe(prior,cfg)
    cfg.coeff_target_mode='soft'
    with pytest.raises(ValueError,match='coeff_target_mode'): validate_resume_recipe(prior,cfg)
