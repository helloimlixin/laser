from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch.nn import functional as F

from src.continuous_sparse_codec import ContinuousSparseCodes, continuous_image_codes, continuous_latents
from src.models.continuous_coefficient_mixture import ContinuousCoefficientMixture
from src.models.continuous_compound_rq import build_continuous_compound_model, continuous_pair_loss
from src.training.laser_hard_ar import sparse_encoding


def case():
    torch.manual_seed(261107)
    arch = OmegaConf.create(dict(type='rq-transformer', block_size=[2, 2, 4], embed_dim=12,
        input_embed_dim=8, shared_tok_emb=True, shared_cls_emb=True, input_emb_vqvae=True,
        head_emb_vqvae=True, cumsum_depth_ctx=True, vocab_size=11, vocab_size_cond=1000,
        block_size_cond=1, body=dict(n_layer=2, block=dict(n_head=3, resid_pdrop=0.)),
        head=dict(n_layer=2, block=dict(n_head=3, resid_pdrop=0.))))
    model = build_continuous_compound_model(arch, num_atoms=11,
                                            coefficient_scales=[3., 2., 1., .5]).eval()
    aux = SimpleNamespace(dictionary=F.normalize(torch.randn(8, 11), dim=0), sparsity_level=4,
                          encoder=lambda x: x, quant_conv=lambda x: x)
    codes = continuous_image_codes(aux, torch.randn(2, 8, 2, 2) * 4, encoder_batch_size=1)
    return model, aux, codes, torch.tensor([0, 999])


def test_targets_are_the_original_full_refit_coefficients_without_clipping_or_rng():
    _, aux, _, _ = case()
    images = torch.randn(2, 8, 2, 2) * 10
    rng = torch.get_rng_state().clone()
    codes = continuous_image_codes(aux, images, encoder_batch_size=1)
    a, c = sparse_encoding(images.permute(0, 2, 3, 1).float(), aux.dictionary, 4)
    assert torch.equal(codes.atoms, a) and torch.equal(codes.coefficients, c)
    assert c.abs().max() > 3
    assert torch.equal(rng, torch.get_rng_state())
    expected = (aux.dictionary.T[a] * c[..., None]).sum(-2)
    torch.testing.assert_close(continuous_latents(codes, aux.dictionary), expected, rtol=0, atol=0)
    assert codes.split(1)[0].coefficients.dtype == torch.float32
    assert not codes.coefficients.requires_grad


@torch.no_grad()
def test_current_coefficient_and_future_pairs_never_enter_own_predictions():
    model, aux, codes, labels = case()
    expected = model(codes, aux, labels)
    coefficients = codes.coefficients.clone()
    coefficients[:, 0, 0, 2] += 9
    changed = model(ContinuousSparseCodes(codes.atoms, coefficients), aux, labels)
    torch.testing.assert_close(changed['atom_logits'][:, 0, 0, :3],
                               expected['atom_logits'][:, 0, 0, :3], rtol=0, atol=0)
    for actual, original in zip(changed['coefficient_parameters'], expected['coefficient_parameters']):
        torch.testing.assert_close(actual[:, 0, 0, :3], original[:, 0, 0, :3], rtol=0, atol=0)
    assert not torch.equal(changed['atom_logits'][:, 0, 0, 3], expected['atom_logits'][:, 0, 0, 3])
    # Change the current atom to another valid support, keeping previous pairs.
    atoms = codes.atoms.clone()
    for row in range(len(atoms)):
        unused = next(a for a in range(11) if a not in atoms[row, 0, 0].tolist())
        atoms[row, 0, 0, 2] = unused
    changed = model(ContinuousSparseCodes(atoms, codes.coefficients), aux, labels)
    torch.testing.assert_close(changed['atom_logits'][:, 0, 0, :3],
                               expected['atom_logits'][:, 0, 0, :3], rtol=0, atol=0)
    assert not torch.equal(changed['coefficient_parameters'][0][:, 0, 0, 2],
                           expected['coefficient_parameters'][0][:, 0, 0, 2])
    assert model._codes is model._native_hidden is None


@torch.no_grad()
def test_native_cached_atoms_and_continuous_density_match_teacher_forcing():
    model, aux, truth, labels = case()
    expected = model(truth, aux, labels)
    prefix = ContinuousSparseCodes(torch.zeros_like(truth.atoms), torch.zeros_like(truth.coefficients))
    model.init_cache()
    for h in range(2):
        for w in range(2):
            for d in range(4):
                out = model.cached_forward(prefix, aux, labels, sample_loc=(h, w, d))
                torch.testing.assert_close(out['atom_logits'], expected['atom_logits'][:, h, w, d],
                                           atol=3e-6, rtol=2e-5)
                prefix.atoms[:, h, w, d] = truth.atoms[:, h, w, d]
                vectors = aux.dictionary.T[prefix.atoms[:, h, w, d]][:, None]
                parameters = model.coefficient_parameters(out['hidden'], vectors, d)
                for actual, original in zip(parameters, expected['coefficient_parameters']):
                    torch.testing.assert_close(actual[:, 0], original[:, h, w, d], atol=3e-6, rtol=2e-5)
                prefix.coefficients[:, h, w, d] = truth.coefficients[:, h, w, d]
    model.init_cache()


def test_physical_likelihood_and_all_parameter_gradients_are_finite():
    model, aux, codes, labels = case()
    output = model(codes, aux, labels)
    loss, metrics = continuous_pair_loss(output, codes, model.coefficient_scales)
    atom = -output['atom_logits'].log_softmax(-1).gather(-1, codes.atoms[..., None]).mean()
    coefficient = -ContinuousCoefficientMixture.log_prob(output['coefficient_parameters'],
        codes.coefficients[..., None], model.coefficient_scales[:, None]).mean()
    torch.testing.assert_close(loss, (1.5 * atom + coefficient) / 2.5)
    torch.testing.assert_close(metrics['joint_nll'], atom + coefficient)
    loss.backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())


def test_sampling_is_reproducible_and_contains_real_coefficients():
    model, aux, _, labels = case()
    a = model.sample_sparse(2, aux, labels, generator=torch.Generator().manual_seed(17))
    b = model.sample_sparse(2, aux, labels, generator=torch.Generator().manual_seed(17))
    a.validate(11, 4)
    assert torch.equal(a.atoms, b.atoms) and torch.equal(a.coefficients, b.coefficients)
    assert a.coefficients.unique().numel() > 16
    assert model._codes is model._native_hidden is None
    assert model._cache['spatial_ctx_hw'] is None


@pytest.mark.parametrize('autocast', [False, True])
def test_optimizer_can_fit_a_fixed_batch_and_reload_exactly(autocast):
    model, aux, codes, labels = case()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    losses = []
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=autocast):
            output = model(codes, aux, labels)
            loss, _ = continuous_pair_loss(output, codes, model.coefficient_scales)
        assert all(p.dtype == torch.float32 for p in output['coefficient_parameters'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] - .3
    restored, _, _, _ = case()
    restored.load_state_dict(model.state_dict(), strict=True)
    expected, actual = model(codes, aux, labels), restored(codes, aux, labels)
    torch.testing.assert_close(actual['atom_logits'], expected['atom_logits'], atol=0, rtol=0)


def test_density_is_normalized_in_physical_units_and_sampled_in_the_same_units():
    weights = torch.tensor([[0., 1.]])
    means = torch.tensor([[-.3, .7]])
    std = torch.tensor([[.4, .5]])
    values = torch.linspace(-100, 100, 100001)[None]
    density = ContinuousCoefficientMixture.log_prob((weights, means, std), values, .8).exp()[0]
    torch.testing.assert_close(torch.trapezoid(density, values[0]), torch.tensor(1.), atol=2e-5, rtol=0)


def test_invalid_supports_and_scales_are_rejected():
    _, _, codes, _ = case()
    repeated = codes.atoms.clone()
    repeated[..., 1] = repeated[..., 0]
    with pytest.raises(ValueError, match='distinct'):
        ContinuousSparseCodes(repeated, codes.coefficients).validate(11, 4)


@torch.no_grad()
def test_spatial_and_depth_execution_match_the_unmodified_upstream_backbone():
    from src.rq_upstream import RQTransformer
    model, aux, codes, labels = case()
    model.input_mlp.bias.normal_(std=.2)
    model.head_mlp.bias.normal_(std=.2)
    reference = RQTransformer(model.config).eval()
    reference.load_state_dict({name: model.state_dict()[name] for name in reference.state_dict()}, strict=True)
    model._codes = codes
    reference.embed_with_model_aux = model.embed_with_model_aux
    expected = reference(codes.atoms, aux, labels)
    for d in range(1, 4):
        expected[..., d, :].scatter_(-1, codes.atoms[..., :d], -torch.inf)
    actual = model(codes, aux, labels)
    torch.testing.assert_close(actual['atom_logits'], expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA AMP check')
def test_native_fp16_amp_keeps_coefficient_likelihood_and_samples_finite():
    model, aux, codes, labels = case()
    device = torch.device('cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0')
    model = model.to(device)
    aux.dictionary = aux.dictionary.to(device)
    codes = ContinuousSparseCodes(codes.atoms.to(device), codes.coefficients.to(device))
    labels = labels.to(device)
    scaler = torch.amp.GradScaler('cuda', init_scale=128.)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    loss, _ = continuous_pair_loss(model(codes, aux, labels, amp=True), codes, model.coefficient_scales)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())
    scaler.step(optimizer); scaler.update()
    samples = model.sample_sparse(2, aux, labels, amp=True)
    samples.validate(11, 4)
