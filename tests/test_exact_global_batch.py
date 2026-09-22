import pytest
import torch

from src.training.exact_global_batch import ExactGlobalBatchSampler


@pytest.mark.parametrize('accumulation', [1, 2, 4])
def test_full_church_coverage_and_exact_batch(accumulation):
    samplers = [ExactGlobalBatchSampler(range(126227), 2048, 5, rank, accumulation) for rank in range(5)]
    batches = [list(s) for s in samplers]
    assert all(len(b) == 62 * accumulation for b in batches)
    visited = []
    for step in range(62):
        indices = [i for b in batches for micro in range(accumulation)
                   for i in b[step * accumulation + micro]]
        assert len(indices) == (2048 if step < 61 else 1299)
        visited.extend(indices)
    assert sorted(visited) == list(range(126227))


@pytest.mark.parametrize('accumulation', [1, 2, 4])
def test_unequal_rank_gradients_equal_global_mean_including_partial_batch(accumulation):
    data = torch.linspace(-2, 3, 133, dtype=torch.float64)
    samplers = [ExactGlobalBatchSampler(data, 80, 5, r, accumulation) for r in range(5)]
    batches = [list(s) for s in samplers]
    for step in range(2):
        gradients, all_indices = [], []
        for sampler, rank_batches in zip(samplers, batches):
            parameter = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
            for micro in range(accumulation):
                cursor = step * accumulation + micro
                indices = rank_batches[cursor]
                all_indices.extend(indices)
                loss = ((parameter * data[indices] - 1) ** 2).mean() / accumulation
                (loss * sampler.backward_scale(len(indices), cursor)).backward()
            gradients.append(parameter.grad)
        reference = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
        ((reference * data[all_indices] - 1) ** 2).mean().backward()
        torch.testing.assert_close(torch.stack(gradients).mean(), reference.grad, rtol=1e-14, atol=1e-14)


def test_resume_preserves_order_and_does_not_consume_global_rng():
    sampler = ExactGlobalBatchSampler(range(126227), 2048, 5, 3, 2, seed=17)
    sampler.set_epoch(19)
    state = torch.get_rng_state().clone()
    batches = list(sampler)
    sampler.set_start_batch(24)
    assert list(sampler) == batches[24:]
    assert len(sampler) == 100
    assert torch.equal(state, torch.get_rng_state())
    sampler.set_epoch(20)
    assert list(sampler) != batches[24:]
    with pytest.raises(ValueError, match='optimizer boundary'):
        sampler.set_start_batch(3)
