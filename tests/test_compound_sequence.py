import torch

from src.training.compound_sequence import attach_pair_sequence_decoders
from src.training.rqtransformer import CompoundLaserRQTransformer
from tests.test_compound_pair_autoregressive import tiny_aux, tiny_config


def case():
    torch.manual_seed(71)
    config = tiny_config(depth=4)
    config.block_size = [2, 2, 4]
    model = attach_pair_sequence_decoders(CompoundLaserRQTransformer(
        config, 7, 5, micro_transformer_layers=1,
        depth_specific_coeff_heads=True, pair_autoregressive=True),
        width=12, layers=2, heads=3, dropout=0).eval()
    atoms = torch.arange(16).reshape(1, 2, 2, 4) % 7
    coefficients = torch.arange(16).reshape_as(atoms) % 5
    return model, tiny_aux(4), atoms * 5 + coefficients


@torch.no_grad()
def test_pair_sequence_causality_and_complete_history():
    model, aux, tokens = case()
    baseline = model(tokens, model_aux=aux)
    for event in range(16):
        for component in ('atom', 'coefficient'):
            changed = tokens.clone()
            atom, coefficient = divmod(int(changed.flatten()[event]), 5)
            if component == 'atom':
                atom = (atom + 1) % 7
            else:
                coefficient = (coefficient + 2) % 5
            changed.flatten()[event] = atom * 5 + coefficient
            actual = model(changed, model_aux=aux)
            for key in baseline:
                first_changed = event + (not (component == 'atom' and key == 'coeff_logits'))
                original, modified = baseline[key].flatten(0, -2), actual[key].flatten(0, -2)
                torch.testing.assert_close(original[:first_changed], modified[:first_changed], atol=0, rtol=0)
                for later in range(first_changed, 16):
                    assert not torch.equal(original[later], modified[later])


@torch.no_grad()
def test_both_sequence_caches_match_dense_with_future_placeholders():
    model, aux, tokens = case()
    expected = model(tokens, model_aux=aux)
    generated = torch.full_like(tokens, 2)
    for _ in range(2):
        model.init_cache()
        for event in range(16):
            site, depth = divmod(event, 4)
            h, w = divmod(site, 2)
            hidden = model.cached_head_output(generated, aux, None, (h, w, depth), amp=False)
            support = model.classifier(hidden)
            if depth:
                support.scatter_(1, generated[:, h, w, :depth] // 5, -torch.inf)
            atom = tokens[:, h, w, depth] // 5
            values = model.coefficient_logits(hidden, aux.dictionary.T[atom], depth_index=depth)
            for key, actual in [('atom_logits', support), ('coeff_logits', values)]:
                torch.testing.assert_close(actual, expected[key][:, h, w, depth], atol=2e-6, rtol=1e-5)
            generated[:, h, w, depth] = tokens[:, h, w, depth]
        assert model.support_sequence_decoder._next_event == 16
        assert model.coefficient_sequence_decoder._next_event == 16
    model.init_cache()
    assert all(b._kv is None for d in [model.support_sequence_decoder, model.coefficient_sequence_decoder]
               for b in d.blocks)


def test_both_stacks_learn_on_first_step_and_support_has_no_atom_projection():
    model, aux, tokens = case()
    output = model.train()(tokens, model_aux=aux)
    support = output['atom_logits'].log_softmax(-1).gather(-1, (tokens // 5)[..., None])
    coefficients = output['coeff_logits'].log_softmax(-1).gather(-1, (tokens % 5)[..., None])
    loss = -(support + coefficients).mean()
    loss.backward()
    assert model.support_sequence_decoder.atom_projection is None
    for decoder in (model.support_sequence_decoder, model.coefficient_sequence_decoder):
        for name, parameter in decoder.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name


@torch.no_grad()
def test_sequence_sampling_distinct_support_and_cache_reset():
    model, aux, _ = case()
    atoms, coefficients = model.sample_compound(2, aux, amp=False, atom_top_k=7)
    assert atoms.shape == coefficients.shape == (2, 2, 2, 4)
    assert (atoms.sort(-1).values.diff(dim=-1) > 0).all()
    assert model.support_sequence_decoder._next_event == 0
    assert model.coefficient_sequence_decoder._next_event == 0
