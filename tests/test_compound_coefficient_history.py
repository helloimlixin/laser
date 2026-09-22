import copy

import pytest
import torch

from src.models.compound_coefficient_decoder import CompoundCoefficientHistoryDecoder
from src.training.compound_history import attach_coefficient_history_decoder
from src.training.rqtransformer import CompoundLaserRQTransformer, build_model
from tests.test_compound_pair_autoregressive import tiny_aux, tiny_config


def setup_model(*, activate=False):
    torch.manual_seed(31)
    config = tiny_config(depth=3)
    config.block_size = [2, 2, 3]
    base = CompoundLaserRQTransformer(config, 7, 5, pair_autoregressive=True,
                                     micro_transformer_layers=1,
                                     depth_specific_coeff_heads=True).eval()
    model = attach_coefficient_history_decoder(copy.deepcopy(base), width=12, heads=3,
                                              layers=2, dropout=0.)
    if activate:
        torch.nn.init.normal_(model.coefficient_history_decoder.output.weight, std=.1)
    atoms = torch.tensor([[[[0, 2, 4], [1, 3, 5]], [[2, 4, 6], [0, 3, 6]]]])
    coeffs = torch.arange(12).reshape(1, 2, 2, 3) % 5
    return base, model, tiny_aux(3), atoms, coeffs


def test_zero_residual_preserves_predictions_and_parameter_order():
    base, model, aux, atoms, coeffs = setup_model()
    names = list(dict(base.named_parameters()))
    assert list(dict(model.named_parameters()))[:len(names)] == names
    with torch.no_grad():
        expected = base(atoms * 5 + coeffs, model_aux=aux)
        actual = model(atoms * 5 + coeffs, model_aux=aux)
    for key in expected:
        assert torch.equal(expected[key], actual[key])


def test_active_decoder_uses_current_atom_without_current_or_future_coefficients():
    _, model, aux, atoms, coeffs = setup_model(activate=True)
    tokens = atoms * 5 + coeffs
    with torch.no_grad():
        expected = model(tokens, model_aux=aux)
        for event in [0, 2, 3, 7, 11]:
            changed = tokens.clone().reshape(-1)
            changed[event] = (changed[event] // 5) * 5 + (changed[event] % 5 + 2) % 5
            changed[event + 1:] = torch.randint(0, 35, changed[event + 1:].shape)
            actual = model(changed.reshape_as(tokens), model_aux=aux)
            for key in expected:
                assert torch.equal(expected[key].reshape(12, -1)[:event + 1],
                                   actual[key].reshape(12, -1)[:event + 1])
        changed = tokens.clone()
        changed[0, 0, 0, 1] = 3 * 5 + coeffs[0, 0, 0, 1]
        actual = model(changed, model_aux=aux)
        assert torch.equal(expected['atom_logits'][0, 0, 0, 1], actual['atom_logits'][0, 0, 0, 1])
        assert not torch.equal(expected['coeff_logits'][0, 0, 0, 1], actual['coeff_logits'][0, 0, 0, 1])


def test_active_cached_decoder_matches_dense_across_spatial_boundaries():
    _, model, aux, atoms, coeffs = setup_model(activate=True)
    tokens = atoms * 5 + coeffs
    with torch.no_grad():
        expected = model(tokens, model_aux=aux)['coeff_logits'].reshape(12, -1)
        model.init_cache()
        results = []
        for event in range(12):
            site, depth = divmod(event, 3)
            h, w = divmod(site, 2)
            hidden = model.cached_head_output(tokens, aux, None, (h, w, depth), amp=False)
            results.append(model.coefficient_logits(hidden, aux.dictionary.T[atoms[:, h, w, depth]],
                                                    depth_index=depth)[0])
        torch.testing.assert_close(torch.stack(results), expected, atol=2e-6, rtol=1e-5)
        model.init_cache()
        assert model.coefficient_history_decoder._next_event == 0
        assert all(block._kv is None for block in model.coefficient_history_decoder.blocks)


def test_physical_prefix_is_local_and_strictly_shifted_with_signed_raw_values():
    _, model, aux, atoms, coeffs = setup_model()
    aux.coeff_bins = aux.coeff_bins * 50
    # Closure in tiny_aux references the original bins, so use an explicit method.
    aux.compound_embeddings = lambda a, c: aux.dictionary.T[a] * (aux.coeff_bins[c] * aux.coeff_scales)[..., None]
    previous, prefix = model.coefficient_history_inputs(atoms, coeffs, aux)
    physical = aux.compound_embeddings(atoms, coeffs).reshape(1, 4, 3, 4)
    torch.testing.assert_close(prefix.reshape(1, 4, 3, 4)[:, :, 0], torch.zeros(1, 4, 4))
    torch.testing.assert_close(prefix.reshape(1, 4, 3, 4)[:, :, 1], physical[:, :, 0])
    torch.testing.assert_close(prefix.reshape(1, 4, 3, 4)[:, :, 2], physical[:, :, :2].sum(2))
    pairs = model.compound_pair_embeddings(aux, atoms, coeffs).reshape(1, 12, 4)
    torch.testing.assert_close(previous[:, 1:], pairs[:, :-1])


def test_decoder_directly_uses_earlier_pair_memory_and_prefix():
    torch.manual_seed(32)
    decoder = CompoundCoefficientHistoryDecoder(12, 4, width=12, heads=3, layers=2,
                                                dropout=0., max_events=6).eval()
    torch.nn.init.normal_(decoder.output.weight, std=.1)
    hidden, atom, pair, prefix = torch.randn(1, 6, 12), *[torch.randn(1, 6, 4) for _ in range(3)]
    with torch.no_grad():
        expected = decoder(hidden, atom, pair, prefix)
        changed = pair.clone(); changed[:, 0] += 10
        assert not torch.equal(expected[:, -1], decoder(hidden, atom, changed, prefix)[:, -1])
        changed = prefix.clone(); changed[:, -1] += 10
        assert not torch.equal(expected[:, -1], decoder(hidden, atom, pair, changed)[:, -1])


def test_new_parameters_learn_and_cache_rejects_out_of_order_calls():
    _, model, aux, atoms, coeffs = setup_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    for _ in range(2):
        optimizer.zero_grad()
        model(atoms * 5 + coeffs, model_aux=aux)['coeff_logits'].square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.coefficient_history_decoder.parameters())
        optimizer.step()
    assert model.coefficient_history_decoder.blocks[0].qkv.weight.grad.abs().sum() > 0
    decoder = model.coefficient_history_decoder
    with torch.no_grad(), pytest.raises(ValueError, match='expected cached event'):
        decoder(torch.zeros(1, 1, 12), *[torch.zeros(1, 1, 4) for _ in range(3)], event=3)


def test_history_decoder_rejects_support_only_model():
    model = CompoundLaserRQTransformer(tiny_config(), 7, 5)
    with pytest.raises(ValueError, match='full-pair'):
        attach_coefficient_history_decoder(model)


def test_model_factory_rejects_incompatible_history_settings():
    with pytest.raises(ValueError, match='plain full-pair'):
        build_model(12, 7, coefficient_history_layers=2)
    with pytest.raises(ValueError, match='nonnegative'):
        build_model(12, 7, coefficient_history_layers=-1)
