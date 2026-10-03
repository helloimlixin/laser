import copy

import torch

from src.training.compound_history import attach_coefficient_history_decoder
from src.training.rqtransformer import OrthogonalCompoundLaserRQTransformer
from tests.test_orthogonal_compound_tokens import tiny_aux, tiny_config


def setup_model():
    torch.manual_seed(61)
    config = tiny_config()
    config.block_size = [2, 2, 3]
    base = OrthogonalCompoundLaserRQTransformer(
        config, 7, 5, micro_transformer_layers=1,
        depth_specific_coeff_heads=True,
    ).eval()
    model = attach_coefficient_history_decoder(
        copy.deepcopy(base), width=12, heads=3, layers=2, dropout=0.,
    )
    torch.nn.init.normal_(model.coefficient_history_decoder.output.weight, std=.1)
    atoms = torch.tensor([[[[0, 2, 4], [1, 3, 5]], [[2, 4, 6], [0, 3, 6]]]])
    coeffs = torch.arange(12).reshape_as(atoms) % 5
    aux = tiny_aux()
    aux.coeff_scales.copy_(torch.tensor([2., 1.3, .7]))
    return model, aux, atoms, coeffs


def test_orthogonal_history_is_causal_in_both_atoms_and_coefficients():
    model, aux, atoms, coeffs = setup_model()
    tokens = atoms * 5 + coeffs
    with torch.no_grad():
        expected = model(tokens, model_aux=aux)
        for event in [0, 2, 3, 7, 11]:
            changed = tokens.clone().reshape(-1)
            changed[event] = changed[event] // 5 * 5 + (changed[event] % 5 + 2) % 5
            changed[event + 1:] = torch.randint(0, 35, changed[event + 1:].shape)
            actual = model(changed.reshape_as(tokens), model_aux=aux)
            for key in expected:
                torch.testing.assert_close(expected[key].reshape(12, -1)[:event + 1],
                                           actual[key].reshape(12, -1)[:event + 1], rtol=0, atol=0)
        changed = tokens.clone()
        changed[0, 0, 0, 1] = 3 * 5 + coeffs[0, 0, 0, 1]
        actual = model(changed, model_aux=aux)
        assert torch.equal(expected['atom_logits'][0, 0, 0, 1], actual['atom_logits'][0, 0, 0, 1])
        assert not torch.equal(expected['coeff_logits'][0, 0, 0, 1], actual['coeff_logits'][0, 0, 0, 1])


def test_orthogonal_history_cached_logits_match_dense_across_sites():
    model, aux, atoms, coeffs = setup_model()
    tokens = atoms * 5 + coeffs
    with torch.no_grad():
        expected = model(tokens, model_aux=aux)['coeff_logits'].reshape(12, -1)
        model.init_cache()
        # Use an incomplete generation buffer: future ground truth is unavailable.
        packed = torch.full_like(tokens, 2)
        results = []
        for event in range(12):
            site, depth = divmod(event, 3)
            h, w = divmod(site, 2)
            hidden = model.cached_head_output(packed, aux, None, (h, w, depth), amp=False)
            basis, _ = aux.orthogonal_basis(atoms[:, h, w, :depth + 1])
            results.append(model.coefficient_logits(hidden, basis[:, -1], depth_index=depth)[0])
            packed[:, h, w, depth] = tokens[:, h, w, depth]
        torch.testing.assert_close(torch.stack(results), expected, atol=3e-6, rtol=1e-5)


def test_orthogonal_history_prefix_matches_decoder_and_partial_supports():
    model, aux, atoms, coeffs = setup_model()
    previous, prefix = model.coefficient_history_inputs(atoms, coeffs, aux)
    contributions = aux.orthogonal_embeddings(atoms, coeffs)
    prefix = prefix.reshape_as(contributions)
    assert torch.equal(prefix[..., 0, :], torch.zeros_like(prefix[..., 0, :]))
    torch.testing.assert_close(prefix[..., 1:, :], contributions.cumsum(-2)[..., :-1, :])
    for event in range(1, 12):
        torch.testing.assert_close(model._cached_previous_pair(aux, atoms, coeffs, event), previous[:, event])
    for depth in [1, 2]:
        torch.testing.assert_close(model._cached_prefix(aux, atoms[:, 0, 0, :depth], coeffs[:, 0, 0, :depth]), prefix[:, 0, 0, depth])


def test_orthogonal_history_sampling_resets_cache_and_keeps_distinct_atoms():
    model, aux, _, _ = setup_model()
    atoms, coeffs = model.sample_compound(2, aux, amp=False, atom_top_k=1, coeff_top_k=1)
    assert torch.isfinite(aux.orthogonal_embeddings(atoms, coeffs)).all()
    assert (atoms.sort(-1).values.diff(dim=-1) != 0).all()
    assert model.coefficient_history_decoder._next_event == 0
