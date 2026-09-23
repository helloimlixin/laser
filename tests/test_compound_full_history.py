from types import SimpleNamespace
import copy

import pytest
import torch

from src.models.compound_full_history import FullHistoryCompoundTransformer, causal_pair_attention


def setup(block_size=(2, 2, 3), normalize=False, dictionary=False):
    torch.manual_seed(91)
    model = FullHistoryCompoundTransformer(num_atoms=11, coeff_vocab_size=7,
        block_size=block_size, width=24, heads=3, atom_layers=2,
        coefficient_layers=2, dropout=0., normalize_coefficient_inputs=normalize, dictionary_atom_conditioning=dictionary).eval()
    aux = SimpleNamespace(dictionary=torch.nn.functional.normalize(torch.randn(256, 11), dim=0),
                          coeff_bins=torch.tensor([-50., -9., -1., 0., 2., 11., 60.]))
    events = model.events
    atoms = (torch.arange(events) % 11).reshape(1, *block_size)
    coefficients = (torch.arange(events) % 7).reshape_as(atoms)
    return model, aux, atoms, coefficients


@pytest.mark.parametrize(('normalize', 'dictionary'), [(False, False), (True, False), (True, True)])
def test_future_and_current_values_never_leak_into_predictions(normalize, dictionary):
    model, aux, atoms, coefficients = setup(normalize=normalize, dictionary=dictionary)
    packed = atoms * 7 + coefficients
    with torch.no_grad():
        expected = model(packed, model_aux=aux)
        for event in range(model.events):
            changed = packed.clone().reshape(-1)
            changed[event] = (changed[event] // 7) * 7 + (changed[event] % 7 + 2) % 7
            changed[event+1:] = torch.randint(0, 77, changed[event+1:].shape)
            actual = model(changed.reshape_as(packed), model_aux=aux)
            for key in expected:
                torch.testing.assert_close(actual[key].reshape(model.events, -1)[:event+1],
                    expected[key].reshape(model.events, -1)[:event+1], rtol=0, atol=0)
        changed = packed.clone(); changed.reshape(-1)[0] = 10 * 7
        actual = model(changed, model_aux=aux)
        assert torch.equal(actual['atom_logits'].reshape(model.events, -1)[0], expected['atom_logits'].reshape(model.events, -1)[0])
        assert not torch.equal(actual['coeff_logits'].reshape(model.events, -1)[0], expected['coeff_logits'].reshape(model.events, -1)[0])


@pytest.mark.parametrize('block_size', [(2, 2, 3), (8, 8, 4)])
@pytest.mark.parametrize(('normalize', 'dictionary'), [(False, False), (True, False), (True, True)])
def test_both_full_history_caches_match_dense_across_all_events(block_size, normalize, dictionary):
    model, aux, atoms, coefficients = setup(block_size, normalize, dictionary)
    with torch.no_grad():
        expected = model(atoms * 7 + coefficients, model_aux=aux)
        model.init_cache()
        for event in range(model.events):
            a = model.cached_atom_logits(atoms, coefficients, aux, event)
            c = model.cached_coefficient_logits(atoms, coefficients, atoms.reshape(1, -1)[:, event], aux, event)
            torch.testing.assert_close(a, expected['atom_logits'].reshape(1, model.events, -1)[:, event], atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(c, expected['coeff_logits'].reshape(1, model.events, -1)[:, event], atol=2e-6, rtol=2e-5)
        assert all(block._length == model.events for block in [*model.atom_decoder, *model.coefficient_decoder])
        model.init_cache()
        assert all(block._keys is None and block._length == 0 for block in [*model.atom_decoder, *model.coefficient_decoder])


@pytest.mark.parametrize(('normalize', 'dictionary'), [(False, False), (True, False), (True, True)])
def test_both_decoders_preserve_individual_event_history_across_sites(normalize, dictionary):
    model, aux, atoms, coefficients = setup(normalize=normalize, dictionary=dictionary)
    packed = atoms * 7 + coefficients
    with torch.no_grad():
        expected = model(packed, model_aux=aux)
        # Same sum of contributions at the first site, different pair order.
        changed = packed.clone()
        changed.reshape(-1)[:2] = packed.reshape(-1)[:2].flip(0)
        actual = model(changed, model_aux=aux)
        for key in expected:
            assert not torch.equal(expected[key].reshape(model.events, -1)[-1],
                                   actual[key].reshape(model.events, -1)[-1])
        for prior_event in [0, 2, 3, 7]:
            changed = packed.clone().reshape(-1)
            changed[prior_event] = changed[prior_event] // 7 * 7 + (changed[prior_event] % 7 + 1) % 7
            actual = model(changed.reshape_as(packed), model_aux=aux)
            for key in expected:
                assert not torch.equal(expected[key].reshape(model.events, -1)[-1],
                                       actual[key].reshape(model.events, -1)[-1])


@pytest.mark.parametrize(('normalize', 'dictionary'), [(False, False), (True, False), (True, True)])
def test_both_decoders_and_every_coefficient_classifier_receive_gradients(normalize, dictionary):
    model, aux, atoms, coefficients = setup(normalize=normalize, dictionary=dictionary)
    model.train()
    output = model(atoms * 7 + coefficients, model_aux=aux)
    atom_loss = -output['atom_logits'].log_softmax(-1).gather(-1, atoms[..., None]).mean()
    coefficient_loss = -output['coeff_logits'].log_softmax(-1).gather(-1, coefficients[..., None]).mean()
    (atom_loss + coefficient_loss).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_cache_enforces_atom_then_coefficient_order():
    model, aux, atoms, coefficients = setup()
    with pytest.raises(ValueError, match='current atom'):
        model.cached_coefficient_logits(atoms, coefficients, atoms.reshape(1, -1)[:, 0], aux, 0)
    model.cached_atom_logits(atoms, coefficients, aux, 0)
    with pytest.raises(ValueError, match='completed pairs'):
        model.cached_atom_logits(atoms, coefficients, aux, 1)


def test_pair_embedding_preserves_large_signed_physical_values():
    model, aux, atoms, coefficients = setup()
    with torch.no_grad():
        model.atom_embedding.weight.zero_()
        model.coefficient_embedding.weight.zero_()
        actual = model.pair_embedding(atoms, coefficients, aux)
        expected = model.physical_projection(aux.dictionary.T[atoms] * aux.coeff_bins[coefficients][..., None])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_current_atom_signal_survives_large_history_activations():
    model, _, _, _ = setup(normalize=True)
    history = torch.randn(2, 1, model.width)
    previous = torch.randn_like(history)
    prefix = torch.randn(2, 1, 256)
    atoms = torch.tensor([[1], [2]])
    with torch.no_grad():
        normal = model.coefficient_inputs(history, atoms, previous, prefix)
        large = model.coefficient_inputs(history * 50000, atoms, previous, prefix)
        changed_atom = model.coefficient_inputs(history * 50000, atoms.flip(0), previous, prefix)
    torch.testing.assert_close(normal, large, rtol=1e-4, atol=1e-4)
    assert float((large - changed_atom).square().mean().sqrt()) > .25


def test_dictionary_vector_conditions_current_coefficient_independently_of_atom_id():
    model, aux, atoms, coefficients = setup(normalize=True, dictionary=True)
    changed_aux = SimpleNamespace(dictionary=aux.dictionary.clone(), coeff_bins=aux.coeff_bins)
    changed_aux.dictionary[:, atoms.reshape(-1)[0]] *= -1
    with torch.no_grad():
        expected = model(atoms * 7 + coefficients, model_aux=aux)
        actual = model(atoms * 7 + coefficients, model_aux=changed_aux)
    # At event zero no dictionary vector has yet entered completed history.
    torch.testing.assert_close(actual['atom_logits'].reshape(model.events, -1)[0],
        expected['atom_logits'].reshape(model.events, -1)[0], rtol=0, atol=0)
    assert not torch.equal(actual['coeff_logits'].reshape(model.events, -1)[0],
        expected['coeff_logits'].reshape(model.events, -1)[0])


def test_dictionary_fusion_preserves_atom_signal_with_large_history_and_separate_events():
    model, aux, _, _ = setup(normalize=True, dictionary=True)
    history = torch.randn(2, 3, model.width)
    previous = torch.randn_like(history)
    prefix = torch.randn(2, 3, 256)
    atoms = torch.tensor([[1, 2, 3], [4, 5, 6]])
    with torch.no_grad():
        normal = model.coefficient_inputs(history, atoms, previous, prefix, aux)
        large = model.coefficient_inputs(history * 50000, atoms, previous, prefix, aux)
        changed_atom = model.coefficient_inputs(history * 50000, atoms.flip(0), previous, prefix, aux)
        individual = torch.cat([
            model.coefficient_inputs(history[:, i:i+1], atoms[:, i:i+1],
                                     previous[:, i:i+1], prefix[:, i:i+1], aux)
            for i in range(3)], dim=1)
    torch.testing.assert_close(normal, large, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(normal, individual, rtol=1e-5, atol=1e-6)
    assert float((large - changed_atom).square().mean().sqrt()) > .1


def test_dictionary_conditioning_requires_balanced_fields():
    with pytest.raises(ValueError, match='normalized conditioning'):
        FullHistoryCompoundTransformer(dictionary_atom_conditioning=True)


def test_dictionary_model_optimizer_and_rng_resume_exactly():
    model, aux, atoms, coefficients = setup(normalize=True, dictionary=True)
    model.train()
    # Exercise stochastic residual paths as well as the new fusion parameters.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = .1
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005)

    def update(current, opt):
        opt.zero_grad(set_to_none=True)
        output = current(atoms * 7 + coefficients, model_aux=aux)
        loss = (-output['atom_logits'].log_softmax(-1).gather(-1, atoms[..., None]).mean()
                -output['coeff_logits'].log_softmax(-1).gather(-1, coefficients[..., None]).mean())
        loss.backward()
        opt.step()

    update(model, optimizer)
    saved = copy.deepcopy((model.state_dict(), optimizer.state_dict(), torch.get_rng_state()))
    update(model, optimizer)
    expected = copy.deepcopy(model.state_dict())
    model.load_state_dict(saved[0])
    optimizer.load_state_dict(saved[1])
    torch.set_rng_state(saved[2])
    update(model, optimizer)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


def test_pair_attention_matches_dense_softmax_outputs_and_gradients():
    torch.manual_seed(402)
    inputs = [torch.randn(5, 3, 2, 8, requires_grad=True) for _ in range(3)]
    reference_inputs = [value.detach().clone().requires_grad_() for value in inputs]
    actual = causal_pair_attention(*inputs)
    expected = torch.nn.functional.scaled_dot_product_attention(*reference_inputs, is_causal=True)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    for value, reference in zip(inputs, reference_inputs):
        torch.testing.assert_close(value.grad, reference.grad, rtol=3e-5, atol=2e-6)
