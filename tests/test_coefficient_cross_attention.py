import copy

import torch

from src.models.coefficient_cross_attention import CoefficientPairCrossAttention
from src.training.coefficient_cross_attention import (
    attach_coefficient_cross_attention, extend_optimizer_for_appended_parameters,
)
from src.training.rqtransformer import CompoundLaserRQTransformer
from tests.test_compound_pair_autoregressive import tiny_aux, tiny_config


def case(active=False):
    torch.manual_seed(773)
    config = tiny_config(depth=3)
    config.block_size = [2, 2, 3]
    base = CompoundLaserRQTransformer(config, 7, 5, pair_autoregressive=True).eval()
    model = attach_coefficient_cross_attention(copy.deepcopy(base), width=12, layers=2, heads=3)
    if active:
        torch.nn.init.normal_(model.coefficient_history_decoder.output.weight, std=.1)
    atoms = torch.tensor([[[[0,2,4],[1,3,5]],[[2,4,6],[0,3,6]]]])
    coefficients = torch.arange(12).reshape_as(atoms) % 5
    return base, model, tiny_aux(3), atoms * 5 + coefficients


def test_neutral_adapter_preserves_existing_weights_predictions_and_parameter_order():
    base, model, aux, tokens = case()
    old = list(dict(base.named_parameters()))
    assert list(dict(model.named_parameters()))[:len(old)] == old
    for name, value in base.state_dict().items():
        assert torch.equal(value, model.state_dict()[name])
    with torch.no_grad():
        expected, actual = base(tokens, model_aux=aux), model(tokens, model_aux=aux)
    for key in expected:
        assert torch.equal(expected[key], actual[key])


def test_current_coefficients_and_future_pairs_cannot_leak_but_current_atom_is_visible():
    _, model, aux, tokens = case(True)
    with torch.no_grad():
        expected = model(tokens, model_aux=aux)
        for event in range(12):
            changed = tokens.clone().reshape(-1)
            changed[event] = changed[event] // 5 * 5 + (changed[event] % 5 + 2) % 5
            changed[event+1:] = (changed[event+1:] + 11) % 35
            actual = model(changed.reshape_as(tokens), model_aux=aux)
            for key in expected:
                torch.testing.assert_close(expected[key].reshape(12,-1)[:event+1],
                                           actual[key].reshape(12,-1)[:event+1], atol=0, rtol=0)
        changed = tokens.clone()
        changed.flatten()[4] = 6 * 5 + changed.flatten()[4] % 5
        actual = model(changed, model_aux=aux)
        assert torch.equal(expected['atom_logits'].reshape(12,-1)[4], actual['atom_logits'].reshape(12,-1)[4])
        assert not torch.equal(expected['coeff_logits'].reshape(12,-1)[4], actual['coeff_logits'].reshape(12,-1)[4])


def test_cross_attention_reads_each_individual_past_pair_and_masks_future_memory():
    torch.manual_seed(774)
    decoder = CoefficientPairCrossAttention(12,4,width=12,layers=2,heads=3,max_events=6).eval()
    torch.nn.init.normal_(decoder.output.weight,std=.1)
    hidden = torch.randn(1,6,12)
    atom, memory = torch.randn(1,6,4), torch.randn(1,6,4)
    with torch.no_grad():
        expected = decoder(hidden,atom,memory)
        for position in range(6):
            changed = memory.clone()
            changed[:,position] += torch.tensor([3.,-1.,2.,4.])
            actual = decoder(hidden,atom,changed)
            assert torch.equal(expected[:,:position],actual[:,:position])
            assert not torch.equal(expected[:,position:],actual[:,position:])


def test_cached_generation_matches_teacher_forcing_with_future_placeholders():
    _, model, aux, targets = case(True)
    with torch.no_grad():
        expected = model(targets,model_aux=aux)
        generated = torch.full_like(targets,2)
        model.init_cache()
        for event in range(12):
            site, depth = divmod(event,3)
            h,w = divmod(site,2)
            hidden = model.cached_head_output(generated,aux,None,(h,w,depth),amp=False)
            atom = targets[:,h,w,depth] // 5
            actual = model.coefficient_logits(hidden,aux.dictionary.T[atom],depth_index=depth)
            torch.testing.assert_close(actual,expected['coeff_logits'][:,h,w,depth],atol=2e-6,rtol=1e-5)
            generated[:,h,w,depth] = targets[:,h,w,depth]
        model.init_cache()
        assert all(block._kv is None for block in model.coefficient_history_decoder.blocks)


def test_new_attention_weights_receive_finite_nonzero_gradients_after_neutral_start():
    _,model,aux,tokens = case()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001)
    for _ in range(2):
        optimizer.zero_grad()
        model(tokens,model_aux=aux)['coeff_logits'].square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.coefficient_history_decoder.parameters())
        optimizer.step()
    assert model.coefficient_history_decoder.blocks[0].key_value.weight.grad.abs().sum() > 0


def test_optimizer_migration_retains_moments_and_starts_only_new_parameters_at_zero():
    base, model, aux, tokens = case()
    old_optimizer = torch.optim.AdamW(base.parameters(), lr=.001)
    base(tokens, model_aux=aux)['coeff_logits'].square().mean().backward()
    old_optimizer.step()
    saved = old_optimizer.state_dict()
    for state in saved['state'].values():
        state['step'].fill_(1250)
    migrated = extend_optimizer_for_appended_parameters(
        saved, dict(base.named_parameters()), dict(model.named_parameters()),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    optimizer.load_state_dict(migrated)
    old_parameters = len(list(base.parameters()))
    parameters = list(model.parameters())
    for index, state in saved['state'].items():
        for key, value in state.items():
            assert torch.equal(value, optimizer.state[parameters[index]][key])
    assert all(p not in optimizer.state for p in parameters[old_parameters:])
    model(tokens, model_aux=aux)['coeff_logits'].square().mean().backward()
    optimizer.step()
    assert all(optimizer.state[parameters[index]]['step'] == 1251 for index in saved['state'])
    assert all(optimizer.state[p]['step'] == 1 for p in parameters[old_parameters:])
