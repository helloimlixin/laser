import pytest
import torch

from test_compound_pair_autoregressive import tiny_aux, tiny_config
from src.church_looped_pair import LoopedPairRQTransformer


def model(variant):
    torch.manual_seed(791)
    config = tiny_config(4)
    config.block_size = [2, 2, 4]
    config.head.n_layer = 6
    return LoopedPairRQTransformer(config, 7, 5, variant).eval()


def inputs():
    atoms = torch.tensor([[[[0, 2, 4, 6], [1, 3, 5, 6]],
                           [[0, 1, 2, 3], [2, 3, 4, 5]]]])
    ids = torch.arange(16).reshape_as(atoms) % 5
    return atoms, atoms * 5 + ids


def test_equal_initial_function_and_summed_shared_gradients():
    recurrent, control = model('looped'), model('unrolled')
    _, tokens = inputs()
    outputs = [m(tokens, model_aux=tiny_aux(4)) for m in (recurrent, control)]
    for name in outputs[0]:
        assert torch.equal(outputs[0][name], outputs[1][name])
    for out in outputs:
        (out['coeff_logits'].square().mean() + out['atom_logits'][torch.isfinite(out['atom_logits'])].square().mean()).backward()
    for i, shared in enumerate(recurrent.head_transformer.blocks):
        for name, parameter in shared.named_parameters():
            gradients = [dict(control.head_transformer.blocks[i + 2 * j].named_parameters())[name].grad for j in range(3)]
            torch.testing.assert_close(parameter.grad, sum(gradients), atol=1e-7, rtol=2e-5)


@pytest.mark.parametrize('variant', ['looped', 'unrolled'])
def test_full_grid_cached_logits_and_separate_loop_caches(variant):
    m, aux = model(variant), tiny_aux(4)
    atoms, tokens = inputs()
    with torch.no_grad():
        teacher = m(tokens, model_aux=aux)
        m.init_cache()
        for h in range(2):
            for w in range(2):
                for d in range(4):
                    hidden = m.cached_head_output(tokens, aux, None, (h, w, d), amp=False)
                    logits = m.classifier(hidden)
                    if d:
                        logits.scatter_(1, atoms[:, h, w, :d], -float('inf'))
                    coefficients = m.coefficient_logits(hidden, aux.dictionary.t()[atoms[:, h, w, d]], d)
                    torch.testing.assert_close(logits, teacher['atom_logits'][:, h, w, d], atol=2e-6, rtol=2e-5)
                    torch.testing.assert_close(coefficients, teacher['coeff_logits'][:, h, w, d], atol=2e-6, rtol=2e-5)
        if variant == 'looped':
            caches = [c for loop in m.head_transformer._pass_caches for c in loop]
            assert len({id(c) for c in caches}) == 6
            assert all(c['past_kv'] is not None for c in caches)
        m.init_cache()
        if variant == 'looped':
            assert all(c['past_kv'] is None for row in m.head_transformer._pass_caches for c in row)


@pytest.mark.parametrize('variant', ['looped', 'unrolled'])
def test_past_pair_conditions_future_and_no_target_leakage(variant):
    m, aux = model(variant), tiny_aux(4)
    _, tokens = inputs()
    changed = tokens.clone()
    changed[:, 0, 0, 0] += 1
    with torch.no_grad():
        before, after = [m(t, model_aux=aux) for t in (tokens, changed)]
    for field in ('atom_logits', 'coeff_logits'):
        assert torch.equal(before[field][:, 0, 0, 0], after[field][:, 0, 0, 0])
        assert not torch.equal(before[field][:, 0, 0, 1], after[field][:, 0, 0, 1])
        assert not torch.equal(before[field][:, 0, 1, 0], after[field][:, 0, 1, 0])
    changed = tokens.clone()
    changed[:, 0, 0, 1] += 5  # Current atom affects its coefficient, not its atom logit.
    with torch.no_grad():
        current = m(changed, model_aux=aux)
    assert torch.equal(before['atom_logits'][:, 0, 0, 1], current['atom_logits'][:, 0, 0, 1])
    assert not torch.equal(before['coeff_logits'][:, 0, 0, 1], current['coeff_logits'][:, 0, 0, 1])


def test_sampling_valid_support_and_coefficient_bins():
    m = model('looped')
    atoms, ids = m.sample_compound(2, tiny_aux(4), atom_top_k=7, coeff_top_p=.5, amp=False)
    assert atoms.shape == ids.shape == (2, 2, 2, 4)
    assert ((ids >= 0) & (ids < 5)).all()
    assert (atoms.sort(-1).values.diff(dim=-1) > 0).all()


def test_invalid_loop_configuration_rejected():
    with pytest.raises(ValueError, match='Head depth'):
        LoopedPairRQTransformer(tiny_config(4), 7, 5)
