from types import SimpleNamespace

import pytest
import torch

from src.training.full_vocab_omp import attach_supports_first_omp, online_omp_targets
from src.training.rqtransformer import CompoundLaserRQTransformer, compound_objective
from tests.test_compound_pair_autoregressive import tiny_aux, tiny_config


def teacher_fixture():
    g = torch.Generator().manual_seed(22)
    dictionary = torch.nn.functional.normalize(torch.randn(8, 31, generator=g), dim=0)
    aux = SimpleNamespace(dictionary=dictionary, coeff_scales=torch.tensor([2., 1., .7, .5]),
                          coeff_bins=torch.linspace(-3, 3, 29))
    return torch.randn(9, 8, generator=g), aux


def test_full_teacher_matches_each_actual_omp_prefix_and_final_joint_refit():
    x, aux = teacher_fixture()
    out = online_omp_targets(x, aux, temperature=.7, coefficient_temperature=.03125,
                             site_chunk_size=4, generator=torch.Generator().manual_seed(9))
    for row in range(len(x)):
        residual = x[row].double()
        dictionary = aux.dictionary.double()
        for depth in range(4):
            scores = (residual @ dictionary).square() / .7
            scores[out['atoms'][row, :depth]] = -torch.inf
            torch.testing.assert_close(out['atom_probabilities'][row, depth].double(),
                                       scores.softmax(-1), atol=2e-6, rtol=2e-5)
            support = dictionary[:, out['atoms'][row, :depth+1]]
            coefficients = torch.linalg.lstsq(support, x[row].double()).solution
            residual = x[row].double() - support @ coefficients
        torch.testing.assert_close(out['coefficients'][row].double() * aux.coeff_scales,
                                   coefficients, atol=3e-6, rtol=3e-5)
    centers = out['coefficients']
    expected = (-(centers[..., None]-aux.coeff_bins).square()/.03125).softmax(-1)
    torch.testing.assert_close(out['coefficient_probabilities'], expected)
    # More than 16 atoms have positive probability at every step in this fixture.
    assert (out['atom_probabilities'].gt(0).sum(-1) > 16).all()


def test_rng_replay_and_fresh_draws_without_finite_bank():
    x, aux = teacher_fixture()
    g = torch.Generator().manual_seed(3)
    state = g.get_state()
    a = online_omp_targets(x, aux, temperature=.7, coefficient_temperature=.1, generator=g)
    b = online_omp_targets(x, aux, temperature=.7, coefficient_temperature=.1, generator=g)
    g.set_state(state)
    replay = online_omp_targets(x, aux, temperature=.7, coefficient_temperature=.1, generator=g)
    for key in a:
        torch.testing.assert_close(a[key], replay[key], atol=0, rtol=0)
    assert not torch.equal(a['atoms'], b['atoms'])


def model_fixture():
    torch.manual_seed(77)
    config = tiny_config(4)
    config.block_size = [1, 2, 4]
    model = attach_supports_first_omp(CompoundLaserRQTransformer(config, 7, 5,
        pair_autoregressive=True, micro_transformer_layers=1,
        depth_specific_coeff_heads=True)).eval()
    atoms = torch.tensor([[[[0, 2, 4, 6], [1, 3, 5, 0]]]])
    coeff = torch.arange(8).reshape_as(atoms) % 5
    return model, tiny_aux(4), atoms*5 + coeff


@torch.no_grad()
def test_all_supports_precede_coefficients_in_causal_forward():
    model, aux, packed = model_fixture()
    expected = model(packed, model_aux=aux)
    # Changing any coefficient leaves ALL support predictions at this site fixed.
    for depth in range(4):
        changed = packed.clone()
        changed[0, 0, 0, depth] = changed[0, 0, 0, depth] // 5 * 5 + (changed[0, 0, 0, depth] % 5 + 1) % 5
        out = model(changed, model_aux=aux)
        torch.testing.assert_close(out['atom_logits'][:, :, :1], expected['atom_logits'][:, :, :1], atol=0, rtol=0)
        torch.testing.assert_close(out['coeff_logits'][:, :, :1, :depth+1], expected['coeff_logits'][:, :, :1, :depth+1], atol=0, rtol=0)
        assert not torch.equal(out['atom_logits'][:, :, 1:], expected['atom_logits'][:, :, 1:])
    # Last support is available to the very first coefficient prediction.
    changed = packed.clone()
    changed[0, 0, 0, -1] = 5*5 + changed[0, 0, 0, -1] % 5
    out = model(changed, model_aux=aux)
    torch.testing.assert_close(out['atom_logits'][:, :, :1], expected['atom_logits'][:, :, :1], atol=0, rtol=0)
    assert not torch.equal(out['coeff_logits'][:, :, :1, 0], expected['coeff_logits'][:, :, :1, 0])


@torch.no_grad()
def test_all_eight_cached_events_match_dense_with_unknown_future_placeholders():
    model, aux, packed = model_fixture()
    expected = model(packed, model_aux=aux)
    generated = torch.full_like(packed, 2)
    model.init_cache()
    for site in range(2):
        for event in range(8):
            hidden = model.cached_head_output(generated, aux, None, (0, site, event), amp=False)
            if event < 4:
                actual = model.classifier(hidden)
                if event:
                    actual.scatter_(1, generated[:, 0, site, :event] // 5, -torch.inf)
                torch.testing.assert_close(actual, expected['atom_logits'][:, 0, site, event], atol=2e-6, rtol=1e-5)
                generated[:, 0, site, event] = packed[:, 0, site, event] // 5 * 5 + generated[:, 0, site, event] % 5
            else:
                depth = event-4
                atom = generated[:, 0, site, depth] // 5
                actual = model.coefficient_logits(hidden, aux.dictionary.T[atom], depth_index=depth)
                torch.testing.assert_close(actual, expected['coeff_logits'][:, 0, site, depth], atol=2e-6, rtol=1e-5)
                generated[:, 0, site, depth] = packed[:, 0, site, depth]


def test_dense_soft_loss_ignores_zero_mass_masked_atoms_and_all_parameters_learn():
    model, aux, packed = model_fixture()
    out = model.train()(packed, model_aux=aux)
    atoms = packed // 5
    qa = torch.randn_like(out['atom_logits']).masked_fill(~out['atom_logits'].isfinite(), -torch.inf).softmax(-1)
    qc = torch.randn_like(out['coeff_logits']).softmax(-1)
    loss, _ = compound_objective(out['atom_logits'], out['coeff_logits'], None, atoms, qc, None,
        atom_weight=1.5, geometry_weight=0, accumulation=1, target_atom_probabilities=qa)
    loga = out['atom_logits'].float().log_softmax(-1)
    logc = out['coeff_logits'].float().log_softmax(-1)
    expected = -(1.5 * (qa * loga.masked_fill(qa == 0, 0)).sum(-1) + (qc * logc).sum(-1)).mean()/2.5
    torch.testing.assert_close(loss, expected)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


@torch.no_grad()
def test_generation_preserves_distinct_support_and_decodable_coefficients():
    model, aux, _ = model_fixture()
    atoms, coefficients = model.sample_compound(2, aux, amp=False, atom_top_k=7)
    assert atoms.shape == coefficients.shape == (2, 1, 2, 4)
    assert (atoms.sort(-1).values.diff(dim=-1) > 0).all()
    assert coefficients.min() >= 0 and coefficients.max() < 5
    assert torch.isfinite(aux.compound_embeddings(atoms, coefficients).sum(-2)).all()
