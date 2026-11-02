import random

import numpy as np
import pytest
import torch

from src.training.rq_layout_rng import expand_four_to_eight
from src.training.exact_global_batch import ExactGlobalBatchSampler


def states():
    return [dict(python=random.Random(i).getstate(), numpy=np.random.RandomState(i).get_state(),
        cpu=torch.Generator().manual_seed(i).get_state(),
        cuda=torch.Generator().manual_seed(100+i).get_state()) for i in range(4)]


def cuda_factory(seed):
    return torch.Generator().manual_seed(seed).get_state()


def test_added_streams_are_repeatable_and_original_streams_untouched():
    saved = states()
    before = torch.get_rng_state().clone()
    expanded, seeds = expand_four_to_eight(saved, 'a'*64, 5634, cuda_factory)
    repeated, repeated_seeds = expand_four_to_eight(saved, 'a'*64, 5634, cuda_factory)
    assert len(expanded) == 8 and len(set(seeds.values())) == 4
    assert seeds == repeated_seeds and torch.equal(before, torch.get_rng_state())
    for index, value in enumerate(expanded):
        reference = saved[index] if index < 4 else repeated[index]
        assert value['python'] == reference['python']
        np.testing.assert_array_equal(value['numpy'][1], reference['numpy'][1])
        assert torch.equal(value['cpu'], reference['cpu'])
        assert torch.equal(value['cuda'], reference['cuda'])
    assert len({value['cuda'].numpy().tobytes() for value in expanded}) == 8
    expanded[0]['cpu'][0] ^= 1
    assert not torch.equal(expanded[0]['cpu'], saved[0]['cpu'])


@pytest.mark.parametrize('count,sha,step', [(3,'a'*64,1),(4,'bad',1),(4,'a'*64,-1)])
def test_invalid_source_is_rejected(count, sha, step):
    with pytest.raises(ValueError):
        expand_four_to_eight(states()[:count], sha, step, cuda_factory)


def test_four_and_eight_gpus_use_identical_global_batches_and_weighted_gradients():
    data = torch.linspace(-2, 3, 2048*2+1167, dtype=torch.float64)
    results = []
    for world, accumulation in [(4, 8), (8, 4)]:
        samplers = [ExactGlobalBatchSampler(data, 2048, world, r, accumulation, 261001)
            for r in range(world)]
        for sampler in samplers:
            sampler.set_epoch(9)
        batches = [list(s) for s in samplers]
        layout = []
        for step in range(3):
            gradients, all_indices = [], []
            for sampler, rank_batches in zip(samplers, batches):
                parameter = torch.tensor(.7, dtype=torch.float64, requires_grad=True)
                for micro in range(accumulation):
                    cursor = step*accumulation+micro
                    indices = rank_batches[cursor]
                    all_indices.extend(indices)
                    # This is the actual shared driver's DDP loss scaling.
                    loss = ((parameter*data[indices]-1)**2).mean()
                    (loss*world*len(indices)/sampler.global_count(cursor)).backward()
                gradients.append(parameter.grad)
            layout.append((sorted(all_indices), torch.stack(gradients).mean()))
        results.append(layout)
    for four, eight in zip(*results):
        assert four[0] == eight[0]
        torch.testing.assert_close(four[1], eight[1], rtol=1.e-14, atol=1.e-14)
