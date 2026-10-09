from types import MethodType, SimpleNamespace

import pytest
import torch

from src.training.rqtransformer import LaserAux
from src.training.stochastic_image_pairs import (stochastic_components, stochastic_image_components,
    stochastic_soft_image_targets)


def fixture():
    rng = torch.Generator().manual_seed(218)
    dictionary = torch.nn.functional.normalize(torch.randn(12, 25, generator=rng), dim=0)
    signals = torch.randn(2, 3, 4, 12, generator=rng)
    scales = torch.tensor([3., 2., 1., .5])
    return signals, dictionary, scales


def test_stochastic_supports_keep_matching_full_refit_coefficients():
    signals, dictionary, scales = fixture()
    out = stochastic_components(signals, dictionary, scales, temperature=.4, site_chunk_size=7)
    active = dictionary.T[out['atoms']].transpose(-1, -2)
    expected = torch.linalg.lstsq(active, signals[..., None]).solution.squeeze(-1)
    torch.testing.assert_close(out['coefficients'] * scales, expected, atol=2e-5, rtol=2e-5)
    assert (out['atoms'].sort(-1).values.diff(dim=-1) > 0).all()
    assert not out['coefficients'].requires_grad


def test_both_draws_vary_on_identical_images_and_replay_global_training_rng():
    signals, dictionary, scales = fixture()
    aux = SimpleNamespace(encoder=lambda images: images, quant_conv=lambda z: z,
        dictionary=dictionary, coeff_scales=scales, sparsity_level=4,
        coeff_bins=torch.linspace(-3, 3, 127), coeff_vocab_size=127,
        num_atoms=25, vocab_size=152, clamp_coeffs=False)
    images = signals.permute(0, 3, 1, 2)

    def draw():
        a, c = stochastic_image_components(aux, images, temperature=.4, site_chunk_size=7)
        tokens, targets = LaserAux.sparse_targets(aux, a, c, temp=.01125, compact=True)
        return a, c, tokens, targets[1]

    torch.manual_seed(861)
    state = torch.get_rng_state()
    first, second = draw(), draw()
    assert not torch.equal(first[0], second[0])
    assert not torch.equal(first[2][..., 1::2], second[2][..., 1::2])
    # Each coefficient distribution belongs to the sampled support's refit.
    expected = (-(first[1][..., None] - aux.coeff_bins).square() / .01125).softmax(-1)
    torch.testing.assert_close(first[3], expected)
    torch.set_rng_state(state)
    for original, replay in zip(first, draw()):
        torch.testing.assert_close(original, replay, rtol=0, atol=0)
    assert aux._stochastic_pair_gram.shape == (25, 25)


def test_sampling_is_fp32_under_autocast_and_restores_tf32_setting():
    signals, dictionary, scales = fixture()
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        with torch.autocast('cpu', dtype=torch.bfloat16):
            out = stochastic_components(signals, dictionary, scales, temperature=.4)
        assert out['coefficients'].dtype == torch.float32
        assert torch.backends.cuda.matmul.allow_tf32
        assert torch.isfinite(out['coefficients']).all()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize('temperature', [0., -1., float('nan'), float('inf')])
def test_invalid_temperature_rejected(temperature):
    with pytest.raises(ValueError, match='temperature'):
        stochastic_components(*fixture(), temperature=temperature)


def test_clipping_cannot_silently_break_matching_omp_coefficients():
    aux = SimpleNamespace(clamp_coeffs=True)
    with pytest.raises(ValueError, match='unclipped'):
        stochastic_image_components(aux, torch.empty(0), temperature=.1)


def test_online_soft_pairs_resample_both_components_and_replay_complete_joint_targets():
    signals, dictionary, scales = fixture()
    aux = SimpleNamespace(encoder=lambda images: images, quant_conv=lambda z: z,
        dictionary=dictionary, coeff_scales=scales, sparsity_level=4,
        coeff_bins=torch.linspace(-3, 3, 127), coeff_vocab_size=127,
        num_atoms=25, vocab_size=152, clamp_coeffs=False, soft_target_physical=False)
    aux.sparse_targets = MethodType(LaserAux.sparse_targets, aux)
    images = signals.permute(0, 3, 1, 2)
    torch.manual_seed(918)
    state = torch.get_rng_state()
    def draw():
        return stochastic_soft_image_targets(aux, images, atom_temperature=.4,
            coefficient_temperature=.01125, variants=16, site_chunk_size=7)
    first, second = draw(), draw()
    assert not torch.equal(first['atoms'], second['atoms'])
    assert not torch.equal(first['tokens'][..., 1::2], second['tokens'][..., 1::2])
    torch.testing.assert_close(first['atom_weights'].sum(-1), torch.ones_like(first['atoms'], dtype=torch.float32))
    torch.testing.assert_close(first['coefficient_probabilities'].sum(-1), torch.ones_like(first['coefficients']))
    assert (first['atom_weights'][..., 0, :] > 0).all()
    # Full posterior labels can differ from the selected trajectory's kernel.
    selected_kernel = (-(first['coefficients'][..., None] - aux.coeff_bins).square()/.01125).softmax(-1)
    assert not torch.allclose(first['coefficient_probabilities'], selected_kernel)
    torch.set_rng_state(state)
    replay = draw()
    for key in first:
        torch.testing.assert_close(first[key], replay[key], rtol=0, atol=0)


def test_soft_pair_objective_matches_dense_conditional_cross_entropy_and_gradients():
    from src.training.physical_pair_crps import physical_pair_objective_components
    a = torch.tensor([[[.3, -.4, .2], [.2, -.8, .9]]], requires_grad=True)
    masked = a.masked_fill(torch.tensor([[[False, False, False], [True, False, False]]]), -torch.inf)
    c = torch.randn(1, 2, 5, requires_grad=True)
    atoms = torch.tensor([[0, 2]])
    ids = torch.tensor([[[0, 0, 2], [0, 1, 2]]])
    weights = torch.tensor([[[.2, .3, .5], [0., .4, .6]]])
    coefficient_q = torch.randn_like(c).softmax(-1)
    total, classification, _ = physical_pair_objective_components(masked, c, atoms,
        coefficient_q, torch.linspace(-1, 1, 5), torch.tensor(0.), 4,
        target_atom_ids=ids, target_atom_weights=weights)
    dense = torch.zeros_like(a).scatter_add(-1, ids, weights)
    safe = torch.where(dense > 0, masked.log_softmax(-1), 0.)
    expected = (-(dense * safe).sum(-1).mean()
        -(coefficient_q * c.log_softmax(-1)).sum(-1).mean())/2
    torch.testing.assert_close(classification, expected)
    torch.testing.assert_close(total, expected/4)
    actual_gradient = torch.autograd.grad(total, (a, c), retain_graph=True)
    expected_gradient = torch.autograd.grad(expected/4, (a, c))
    for actual, reference in zip(actual_gradient, expected_gradient):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, reference)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA graph equivalence requires CUDA')
@pytest.mark.parametrize('chunk',[7,24])
def test_cuda_graph_teacher_preserves_eager_draws_joint_labels_and_rng(chunk):
    signals,dictionary,scales = [v.cuda() for v in fixture()]
    aux = SimpleNamespace(encoder=lambda images: images,quant_conv=lambda z: z,
        dictionary=dictionary,coeff_scales=scales,sparsity_level=4,
        coeff_bins=torch.linspace(-3,3,127,device='cuda'),coeff_vocab_size=127,
        num_atoms=25,vocab_size=152,clamp_coeffs=False,soft_target_physical=False)
    aux.sparse_targets = MethodType(LaserAux.sparse_targets,aux)
    images = signals.permute(0,3,1,2)
    torch.cuda.manual_seed(918)
    start = torch.cuda.get_rng_state()
    def draw(backend):
        return stochastic_soft_image_targets(aux,images,atom_temperature=.4,
            coefficient_temperature=.01125,variants=16,site_chunk_size=chunk,backend=backend)
    reference = draw('eager')
    reference_rng = torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(start)
    actual = draw('cudagraph')
    assert torch.equal(reference_rng,torch.cuda.get_rng_state())
    for key in reference:
        torch.testing.assert_close(actual[key],reference[key],rtol=0,atol=0)
    torch.cuda.set_rng_state(start)
    replay = draw('cudagraph')
    for key in actual:
        torch.testing.assert_close(actual[key],replay[key],rtol=0,atol=0)
