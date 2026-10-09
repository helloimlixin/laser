import copy

import pytest
import torch

from src.training.coefficient_cross_attention import extend_optimizer_for_appended_parameters
from src.training.pair_memory_cross_attention import attach_pair_memory_queries
from src.training.rqtransformer import CompoundLaserRQTransformer
from tests.test_compound_pair_autoregressive import tiny_aux, tiny_config


def case(mode='cross', active=False):
    torch.manual_seed(882)
    config = tiny_config(depth=3)
    config.block_size = [2, 2, 3]
    config.vocab_size_cond = 3
    base = CompoundLaserRQTransformer(config, 7, 5, pair_autoregressive=True).eval()
    model = attach_pair_memory_queries(copy.deepcopy(base), width=12, heads=3, mode=mode)
    if active:
        torch.nn.init.normal_(model.pair_memory_queries.atom_output.weight, std=.1)
        torch.nn.init.normal_(model.pair_memory_queries.coefficient_output.weight, std=.1)
    atoms = torch.tensor([[[[0, 2, 4], [1, 3, 5]], [[2, 4, 6], [0, 3, 6]]]])
    coefficients = torch.arange(12).reshape_as(atoms) % 5
    return base, model, tiny_aux(3), atoms * 5 + coefficients


@pytest.mark.parametrize('mode', ['cross', 'mlp'])
def test_neutral_outputs_preserve_checkpoint_and_parameter_order(mode):
    base, model, aux, tokens = case(mode)
    old_names = list(dict(base.named_parameters()))
    assert list(dict(model.named_parameters()))[:len(old_names)] == old_names
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in base.state_dict().items())
    with torch.no_grad():
        before, after = base(tokens, model_aux=aux), model(tokens, model_aux=aux)
    for key in before:
        assert torch.equal(before[key], after[key])


@pytest.mark.parametrize('mode', ['cross', 'mlp'])
def test_both_queries_are_strictly_causal_and_coefficient_sees_current_atom(mode):
    _, model, aux, tokens = case(mode, active=True)
    with torch.no_grad():
        expected = model(tokens, model_aux=aux)
        for event in range(12):
            changed = tokens.clone().reshape(-1)
            changed[event:] = (changed[event:] + 11) % 35
            actual = model(changed.reshape_as(tokens), model_aux=aux)
            torch.testing.assert_close(expected['atom_logits'].reshape(12, -1)[:event+1],
                                       actual['atom_logits'].reshape(12, -1)[:event+1], atol=0, rtol=0)
            torch.testing.assert_close(expected['coeff_logits'].reshape(12, -1)[:event],
                                       actual['coeff_logits'].reshape(12, -1)[:event], atol=0, rtol=0)
            changed = tokens.clone().reshape(-1)
            changed[event] = changed[event] // 5 * 5 + (changed[event] % 5 + 2) % 5
            changed[event+1:] = (changed[event+1:] + 11) % 35
            actual = model(changed.reshape_as(tokens), model_aux=aux)
            for key in expected:
                torch.testing.assert_close(expected[key].reshape(12, -1)[:event+1],
                                           actual[key].reshape(12, -1)[:event+1], atol=0, rtol=0)
        changed = tokens.clone()
        changed.flatten()[4] = 6 * 5 + changed.flatten()[4] % 5
        actual = model(changed, model_aux=aux)
        assert torch.equal(expected['atom_logits'].reshape(12, -1)[4], actual['atom_logits'].reshape(12, -1)[4])
        assert not torch.equal(expected['coeff_logits'].reshape(12, -1)[4], actual['coeff_logits'].reshape(12, -1)[4])
        other_class = model(tokens, model_aux=aux, cond=torch.tensor([2]))
        assert not torch.equal(expected['atom_logits'], other_class['atom_logits'])


@pytest.mark.parametrize('mode', ['cross', 'mlp'])
def test_dense_and_cached_both_heads_match_across_sites(mode):
    _, model, aux, targets = case(mode, active=True)
    with torch.no_grad():
        expected = model(targets, model_aux=aux)
        generated = torch.full_like(targets, 2)
        model.init_cache()
        for event in range(12):
            site, depth = divmod(event, 3)
            h, w = divmod(site, 2)
            hidden = model.cached_head_output(generated, aux, None, (h, w, depth), amp=False)
            atom_logits = model.classifier(hidden)
            if depth:
                atom_logits.scatter_(-1, (targets[:, h, w, :depth] // 5), -torch.inf)
            torch.testing.assert_close(atom_logits, expected['atom_logits'][:, h, w, depth], atol=3e-6, rtol=2e-5)
            atom = targets[:, h, w, depth] // 5
            coefficients = model.coefficient_logits(hidden, aux.dictionary.T[atom], depth_index=depth)
            torch.testing.assert_close(coefficients, expected['coeff_logits'][:, h, w, depth], atol=3e-6, rtol=2e-5)
            generated[:, h, w, depth] = targets[:, h, w, depth]
        assert model.pair_memory_queries._next_event == 12
        model.init_cache()
        assert model.pair_memory_queries._pending is None
        assert all(block._length == 0 for block in model.pair_memory_queries.memory_blocks)


def test_memory_retains_coefficient_even_when_physical_contribution_is_zero():
    _, model, aux, tokens = case(active=True)
    # Zero dictionary vectors remove all physical contributions, but previous
    # coefficient identities remain distinct memory fields.
    aux.dictionary.zero_()
    atoms, coefficients = model.unpack(tokens)
    before = model.memory_fields(aux, atoms, coefficients)
    changed = coefficients.clone()
    changed.flatten()[0] = 4
    after = model.memory_fields(aux, atoms, changed)
    assert torch.equal(before[2], after[2])
    assert not torch.equal(before[1], after[1])
    with torch.no_grad():
        expected = model.pair_memory_queries.encode_memory(*(field.reshape(1, 12, -1) for field in before))
        actual = model.pair_memory_queries.encode_memory(*(field.reshape(1, 12, -1) for field in after))
    assert torch.equal(expected[:, :1], actual[:, :1])
    assert not torch.equal(expected[:, 1:], actual[:, 1:])


@pytest.mark.parametrize('mode', ['cross', 'mlp'])
def test_new_parameters_receive_gradients_and_optimizer_ages_are_preserved(mode):
    base, model, aux, tokens = case(mode)
    old_optimizer = torch.optim.AdamW(base.parameters(), lr=.001)
    outputs = base(tokens, model_aux=aux)
    (outputs['atom_logits'][torch.isfinite(outputs['atom_logits'])].square().mean()
     + outputs['coeff_logits'].square().mean()).backward()
    old_optimizer.step()
    saved = old_optimizer.state_dict()
    for state in saved['state'].values():
        state['step'].fill_(1500)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    optimizer.load_state_dict(extend_optimizer_for_appended_parameters(
        saved, dict(base.named_parameters()), dict(model.named_parameters())))
    old_count = len(list(base.parameters()))
    parameters = list(model.parameters())
    assert all(parameter not in optimizer.state for parameter in parameters[old_count:])
    model.train()
    for update in range(2):
        optimizer.zero_grad()
        outputs = model(tokens, model_aux=aux)
        loss = outputs['atom_logits'][torch.isfinite(outputs['atom_logits'])].square().mean() + outputs['coeff_logits'].square().mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.pair_memory_queries.parameters())
        optimizer.step()
        assert all(optimizer.state[p]['step'] == 1501 + update for p in parameters[:old_count])
        assert all(optimizer.state[p]['step'] == 1 + update for p in parameters[old_count:])
    assert model.pair_memory_queries.memory_scalar.weight.grad.abs().sum() > 0
    assert model.pair_memory_queries.atom_blocks[0].ffn[0].weight.grad.abs().sum() > 0
    assert model.pair_memory_queries.coefficient_blocks[0].ffn[0].weight.grad.abs().sum() > 0


def test_cache_rejects_missing_coefficient_or_wrong_event():
    _, model, aux, tokens = case(active=True)
    with torch.no_grad():
        model.init_cache()
        model.cached_head_output(tokens, aux, None, (0, 0, 0), amp=False)
        with pytest.raises(ValueError, match='completed coefficient'):
            model.cached_head_output(tokens, aux, None, (0, 0, 1), amp=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA BF16')
@pytest.mark.parametrize('mode', ['cross', 'mlp'])
def test_bf16_compiled_training_and_cached_sampling(mode):
    _, model, aux, targets = case(mode, active=True)
    model = model.cuda()
    targets = targets.cuda()
    for name in ['dictionary', 'coeff_bins', 'coeff_scales']:
        setattr(aux, name, getattr(aux, name).cuda())
    aux.compound_embeddings = lambda a, c: aux.dictionary.T[a] * (aux.coeff_bins[c] * aux.coeff_scales)[..., None]
    eager = model.pair_memory_queries.forward
    model.pair_memory_queries.forward = torch.compile(eager, fullgraph=True, dynamic=True)
    model.train()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        output = model(targets, model_aux=aux)
        loss = output['atom_logits'][torch.isfinite(output['atom_logits'])].float().square().mean() + output['coeff_logits'].float().square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.pair_memory_queries.parameters())
    model.pair_memory_queries.forward = eager
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        expected = model(targets, model_aux=aux)
        model.init_cache()
        generated = torch.full_like(targets, 2)
        for event in range(12):
            site, depth = divmod(event, 3)
            h, w = divmod(site, 2)
            hidden = model.cached_head_output(generated, aux, None, (h, w, depth), amp=True)
            atom_logits = model.classifier(hidden)
            if depth:
                atom_logits.scatter_(-1, targets[:, h, w, :depth] // 5, -torch.inf)
            coefficients = model.coefficient_logits(hidden, aux.dictionary.T[targets[:, h, w, depth] // 5], depth_index=depth)
            torch.testing.assert_close(atom_logits, expected['atom_logits'][:, h, w, depth], atol=.04, rtol=.04)
            torch.testing.assert_close(coefficients, expected['coeff_logits'][:, h, w, depth], atol=.04, rtol=.04)
            generated[:, h, w, depth] = targets[:, h, w, depth]
