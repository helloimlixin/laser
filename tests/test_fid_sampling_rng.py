import pytest
import torch

from src.training.fid_reference import fixed_evaluation_rng, with_fixed_evaluation_rng


def test_seeded_cpu_evaluation_is_repeatable_and_preserves_training_rng():
    torch.manual_seed(77)
    before = torch.get_rng_state().clone()
    with fixed_evaluation_rng(261001, 'cpu', rank=3):
        first = torch.rand(16)
    assert torch.equal(before, torch.get_rng_state())
    torch.rand(7)
    with fixed_evaluation_rng(261001, 'cpu', rank=3):
        assert torch.equal(first, torch.rand(16))


def test_rank_seeds_are_distinct_and_match_the_declared_seed_rule():
    with fixed_evaluation_rng(91, 'cpu', rank=0):
        first = torch.rand(16)
    with fixed_evaluation_rng(91, 'cpu', rank=1):
        second = torch.rand(16)
    expected = torch.rand(16, generator=torch.Generator().manual_seed(92))
    assert torch.equal(second, expected)
    assert not torch.equal(first, second)


def test_unspecified_seed_retains_native_rng_consumption():
    torch.manual_seed(123)
    with fixed_evaluation_rng(None, 'cpu'):
        first = torch.rand(5)
    second = torch.rand(5)
    torch.manual_seed(123)
    assert torch.equal(torch.cat((first, second)), torch.rand(10))


def test_evaluator_wrapper_applies_seed_and_restores_model_mode_and_rng():
    model = torch.nn.Linear(2, 2).train()
    @with_fixed_evaluation_rng
    def evaluate(model, *, fid_seed=None):
        model.eval()
        return torch.rand(16)
    before = torch.get_rng_state().clone()
    first = evaluate(model, fid_seed=91)
    assert model.training and torch.equal(before, torch.get_rng_state())
    torch.rand(7)
    assert torch.equal(first, evaluate(model, fid_seed=91))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_evaluation_is_repeatable_and_restores_cpu_and_gpu_rng():
    device = torch.device('cuda:0')
    torch.manual_seed(77)
    before_cpu = torch.get_rng_state().clone()
    before_cuda = torch.cuda.get_rng_state(device).clone()
    with fixed_evaluation_rng(261001, device, rank=3):
        first = torch.rand(16, device=device)
        torch.rand(7)
    assert torch.equal(before_cpu, torch.get_rng_state())
    assert torch.equal(before_cuda, torch.cuda.get_rng_state(device))
    torch.rand(11, device=device)
    with fixed_evaluation_rng(261001, device, rank=3):
        assert torch.equal(first, torch.rand(16, device=device))
