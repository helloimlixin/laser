"""Regressions for numerical solves, fixed-code windows, and training schedules."""
import numpy as np
import pytest
import torch

from src.models.dictionary_learner import DictionaryLearning
from src.adaptive_scaled_atom_rq import AdaptiveScaledAtomRQ, fit_atom_levels


@pytest.mark.parametrize('perturbation', [1e-3, 1e-4, 0.])
def test_omp_dependent_supports_match_stable_least_squares(perturbation):
    torch.manual_seed(93)
    dictionary = torch.nn.functional.normalize(torch.randn(6, 1) + perturbation * torch.randn(6, 8), dim=0)
    signals = torch.randn(6, 32)
    learner = DictionaryLearning(num_embeddings=8, embedding_dim=6, sparsity_level=4)
    support, values, prefixes = learner.batch_omp_with_support_and_prefixes(signals, dictionary)
    design = dictionary.T[support].transpose(1, 2).double()
    reference = torch.linalg.lstsq(design, signals.T.double()[..., None], driver='gelsd').solution
    reference_sse = (signals.T.double() - (design @ reference).squeeze(-1)).square().sum()
    errors = torch.stack([(signals.T.double() - (dictionary.T[support[:, :k]] * v[..., None]).sum(1).double()).square().sum(-1)
                          for k, v in enumerate(prefixes, 1)])
    assert torch.isfinite(values).all()
    assert values.abs().max() < 1e6
    assert errors[-1].sum() <= reference_sse * 1.001 + 1e-4
    assert (errors[1:] <= errors[:-1] + 1e-4).all()


def test_omp_ridge_fallback_uses_the_same_regularized_objective():
    torch.manual_seed(93)
    dictionary = torch.nn.functional.normalize(torch.randn(6, 1) + 1e-4 * torch.randn(6, 8), dim=0)
    signals = torch.randn(6, 8)
    learner = DictionaryLearning(num_embeddings=8, embedding_dim=6, sparsity_level=4, omp_ridge=.05)
    support, values = learner.batch_omp_with_support(signals, dictionary)
    design = dictionary.T[support].transpose(1, 2).double()
    expected = torch.linalg.solve(design.transpose(1, 2) @ design + .05 * torch.eye(4),
                                  design.transpose(1, 2) @ signals.T.double()[..., None]).squeeze(-1)
    torch.testing.assert_close(values.double(), expected, atol=2e-5, rtol=2e-4)


def test_float32_omp_is_not_downcast_by_encoder_autocast():
    torch.manual_seed(27)
    learner = DictionaryLearning(num_embeddings=8, embedding_dim=6, sparsity_level=3)
    dictionary = torch.nn.functional.normalize(torch.randn(6, 8), dim=0)
    signals = torch.randn(6, 16)
    expected = learner.batch_omp_with_support(signals, dictionary)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = learner.batch_omp_with_support(signals, dictionary)
    assert torch.equal(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def make_dictionary(**kwargs):
    learner = DictionaryLearning(num_embeddings=2, embedding_dim=2, sparsity_level=1,
        dictionary_update_mode='alternating_residual', dictionary_update_min_usage=1,
        dictionary_update_relaxation=1., **kwargs)
    with torch.no_grad():
        learner.dictionary.copy_(torch.eye(2))
    return learner


@pytest.mark.parametrize('steps', [1, 2])
def test_dictionary_update_keeps_every_microbatch(steps):
    separate = make_dictionary(dictionary_update_accumulation_steps=steps)
    combined = make_dictionary(dictionary_update_accumulation_steps=steps)
    first = torch.tensor([[[[1.]], [[.2]]]])
    second = torch.tensor([[[[.2]], [[1.]]]]).repeat(3, 1, 1, 1)
    for _ in range(steps):
        separate(first)
        separate(second)
        separate.alternating_dictionary_update_after_step_()
        combined(torch.cat([first, second]))
        combined.alternating_dictionary_update_after_step_()
    torch.testing.assert_close(separate.dictionary, combined.dictionary, atol=0, rtol=0)
    assert int(separate._last_dictionary_updated_atom_count) == 2
    assert not separate._dictionary_update_microbatches


def test_amp_skip_discards_pending_codes_without_advancing_schedule():
    learner = make_dictionary(dictionary_update_accumulation_steps=2)
    learner(torch.tensor([[[[1.]], [[.2]]]]))
    learner.alternating_dictionary_update_after_step_()
    learner(torch.tensor([[[[.2]], [[1.]]]]))
    learner.alternating_dictionary_update_after_step_(optimizer_updated=False)
    assert int(learner._dictionary_update_step) == 1
    assert learner._last_dictionary_update_batch is None
    assert not learner._dictionary_update_accumulator
    assert not learner._dictionary_update_microbatches
    torch.testing.assert_close(learner.dictionary, torch.eye(2))


def test_zero_coefficient_slots_are_not_counted_as_dictionary_usage():
    learner = make_dictionary(dead_atom_revival=True)
    learner(torch.zeros(1, 2, 1, 3))
    assert int(learner._atom_usage_window.sum()) == 0
    assert learner.alternating_dictionary_update_after_step_() == 0


def test_gradient_mode_advances_initialization_and_quantization_schedules():
    torch.manual_seed(9)
    learner = DictionaryLearning(num_embeddings=2, embedding_dim=2, sparsity_level=1,
        dictionary_update_mode='gradient', data_init_from_first_batch=True, data_init_start_step=2,
        coefficient_quantization_bits=4, coefficient_quantization_max=2.,
        coefficient_quantization_start_step=2, coefficient_quantization_warmup_steps=2)
    optimizer = torch.optim.SGD(learner.parameters(), lr=.01)
    for _ in range(6):
        optimizer.zero_grad(set_to_none=True)
        learner(torch.tensor([[[[1., .2]], [[.2, 1.]]]], requires_grad=True))
        learner._last_bottleneck_objective_for_backward.backward()
        learner.project_dictionary_gradient_()
        optimizer.step()
        learner.alternating_dictionary_update_after_step_()
        learner.normalize_dictionary_()
    assert int(learner._dictionary_update_step) == 6
    assert bool(learner._data_initialized)
    assert learner._coefficient_quantization_fraction() == 1.


def test_shared_coefficient_fit_preserves_exact_repeated_token_reconstruction():
    quantizer = AdaptiveScaledAtomRQ(torch.ones(1, 1), torch.tensor([[-1., 1.]]), depth=4)
    trace = fit_atom_levels(quantizer, np.array([[4.]], dtype=np.float32), passes=2, batch_size=1)
    torch.testing.assert_close(quantizer.levels, torch.tensor([[-1., 1.]]), rtol=0, atol=0)
    assert all(row['latent_mse_after_update'] == 0 for row in trace)


def test_shared_coefficient_fit_measures_reassigned_codes_and_never_regresses():
    torch.manual_seed(7)
    quantizer = AdaptiveScaledAtomRQ(torch.randn(4, 6), torch.tensor([-.2, .2]).expand(6, -1), depth=4)
    latents = torch.randn(19, 4)
    initial = (latents - quantizer.quantize(latents)['quantized']).double().square().mean().item()
    trace = fit_atom_levels(quantizer, latents, passes=3, batch_size=5)
    final = (latents.double() - quantizer.quantize(latents)['quantized'].double()).square().mean().item()
    assert final < initial
    assert final == pytest.approx(trace[-1]['latent_mse_after_update'], rel=1e-12)
    for row in trace:
        assert row['latent_mse_after_update'] <= row['latent_mse_before_update']
        assert row['regularized_objective_after_update'] <= row['regularized_objective_before_update']


def test_shared_coefficient_fit_restores_book_after_rejected_proposal(monkeypatch):
    import src.compact_rq_fitting as fitting
    monkeypatch.setattr(fitting, 'solve_regularized_levels',
                        lambda normal, rhs, current, prior, ridge: current * 100.)
    quantizer = AdaptiveScaledAtomRQ(torch.ones(1, 1), torch.tensor([[-1., 1.]]), depth=4)
    trace = fit_atom_levels(quantizer, torch.tensor([[4.]]), passes=1)
    assert not trace[0]['accepted']
    torch.testing.assert_close(quantizer.levels, torch.tensor([[-1., 1.]]), rtol=0, atol=0)
