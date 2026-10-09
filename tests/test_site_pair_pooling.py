import copy

import pytest
import torch

from src.models.site_pair_pooling import SitePairPooling
from src.training.coefficient_cross_attention import extend_optimizer_for_appended_parameters
from src.training.rqtransformer import CompoundLaserRQTransformer
from src.training.site_pair_pooling import attach_site_pair_pooling
from tests.test_compound_pair_autoregressive import tiny_aux, tiny_config


def case(mode='attention', active=False):
    torch.manual_seed(882)
    config = tiny_config(depth=4)
    config.block_size = [2, 2, 4]
    config.vocab_size_cond = 3
    base = CompoundLaserRQTransformer(config, 7, 5, pair_autoregressive=True).eval()
    model = attach_site_pair_pooling(copy.deepcopy(base), width=15, heads=3, mode=mode)
    if active:
        torch.nn.init.normal_(model.site_pooling.output.weight, std=.1)
    atoms = torch.tensor([[[[0, 2, 4, 6], [1, 3, 5, 6]],
                           [[2, 4, 5, 6], [0, 1, 3, 6]]]])
    coefficients = torch.arange(16).reshape_as(atoms) % 5
    return base, model, tiny_aux(4), atoms * 5 + coefficients


def test_attention_and_mlp_capacity_match_exactly():
    modules = [SitePairPooling(1536, 4, mode=m) for m in ('attention', 'mlp')]
    counts = [sum(p.numel() for p in model.parameters()) for model in modules]
    assert counts[0] == counts[1]
    assert counts[0] < 5_000_000
    shared = set(modules[0].state_dict()) & set(modules[1].state_dict())
    assert 'depth_embedding' in shared and 'output.weight' in shared


@pytest.mark.parametrize('mode', ['attention', 'mlp'])
def test_initial_predictions_weights_parameter_order_and_rng_are_preserved(mode):
    base, model, aux, tokens = case(mode)
    old = list(dict(base.named_parameters()))
    assert list(dict(model.named_parameters()))[:len(old)] == old
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in base.state_dict().items())
    with torch.no_grad():
        rng = torch.get_rng_state().clone()
        before = base(tokens, model_aux=aux)
        after = model(tokens, model_aux=aux)
        assert torch.equal(rng, torch.get_rng_state())
    for key in before:
        assert torch.equal(before[key], after[key])


@pytest.mark.parametrize('mode', ['attention', 'mlp'])
def test_future_sites_and_pairs_cannot_leak_and_current_atom_conditions_coefficient(mode):
    _, model, aux, tokens = case(mode, active=True)
    with torch.no_grad():
        expected = model(tokens, model_aux=aux, cond=torch.tensor([1]))
        for event in range(16):
            changed = tokens.clone().reshape(-1)
            changed[event:] = (changed[event:] + 11) % 35
            actual = model(changed.reshape_as(tokens), model_aux=aux, cond=torch.tensor([1]))
            torch.testing.assert_close(expected['atom_logits'].reshape(16, -1)[:event+1],
                                       actual['atom_logits'].reshape(16, -1)[:event+1], atol=0, rtol=0)
            torch.testing.assert_close(expected['coeff_logits'].reshape(16, -1)[:event],
                                       actual['coeff_logits'].reshape(16, -1)[:event], atol=0, rtol=0)
            changed = tokens.clone().reshape(-1)
            changed[event] = changed[event] // 5 * 5 + (changed[event] % 5 + 2) % 5
            changed[event+1:] = (changed[event+1:] + 11) % 35
            actual = model(changed.reshape_as(tokens), model_aux=aux, cond=torch.tensor([1]))
            for key in expected:
                torch.testing.assert_close(expected[key].reshape(16, -1)[:event+1],
                                           actual[key].reshape(16, -1)[:event+1], atol=0, rtol=0)
        changed = tokens.clone()
        changed.flatten()[5] = 4 * 5 + changed.flatten()[5] % 5
        actual = model(changed, model_aux=aux, cond=torch.tensor([1]))
        assert torch.equal(expected['atom_logits'].reshape(16, -1)[5], actual['atom_logits'].reshape(16, -1)[5])
        assert not torch.equal(expected['coeff_logits'].reshape(16, -1)[5], actual['coeff_logits'].reshape(16, -1)[5])
        other_class = model(tokens, model_aux=aux, cond=torch.tensor([2]))
        assert not torch.equal(expected['atom_logits'], other_class['atom_logits'])


@pytest.mark.parametrize('mode', ['attention', 'mlp'])
def test_dense_and_cached_generation_match_with_partial_current_site(mode):
    _, model, aux, targets = case(mode, active=True)
    with torch.no_grad():
        expected = model(targets, model_aux=aux)
        generated = torch.full_like(targets, 2)
        model.init_cache()
        for event in range(16):
            site, depth = divmod(event, 4)
            h, w = divmod(site, 2)
            hidden = model.cached_head_output(generated, aux, None, (h, w, depth), amp=False)
            logits = model.classifier(hidden)
            if depth:
                logits.scatter_(-1, targets[:, h, w, :depth] // 5, -torch.inf)
            coefficients = model.coefficient_logits(hidden, aux.dictionary.T[targets[:, h, w, depth] // 5], depth_index=depth)
            torch.testing.assert_close(logits, expected['atom_logits'][:, h, w, depth], atol=4e-6, rtol=2e-5)
            torch.testing.assert_close(coefficients, expected['coeff_logits'][:, h, w, depth], atol=4e-6, rtol=2e-5)
            generated[:, h, w, depth] = targets[:, h, w, depth]


@pytest.mark.parametrize('mode', ['attention', 'mlp'])
def test_depth_tags_distinguish_equal_sum_pair_permutations(mode):
    torch.manual_seed(71)
    module = SitePairPooling(12, 4, width=15, heads=3, mode=mode).eval()
    torch.nn.init.normal_(module.output.weight, std=.2)
    pairs = torch.randn(2, 3, 4, 12)
    permutation = pairs.flip(-2)
    torch.testing.assert_close(pairs.sum(-2), permutation.sum(-2))
    assert not torch.allclose(module(pairs), module(permutation))


@pytest.mark.parametrize('mode', ['attention', 'mlp'])
def test_optimizer_migration_gradients_and_inference_ablation(mode):
    base, model, aux, tokens = case(mode)
    optimizer = torch.optim.AdamW(base.parameters(), lr=.001)
    outputs = base(tokens, model_aux=aux)
    loss = outputs['atom_logits'][torch.isfinite(outputs['atom_logits'])].square().mean() + outputs['coeff_logits'].square().mean()
    loss.backward();optimizer.step()
    saved = optimizer.state_dict()
    for state in saved['state'].values():
        state['step'].fill_(7512)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    optimizer.load_state_dict(extend_optimizer_for_appended_parameters(
        saved, dict(base.named_parameters()), dict(model.named_parameters())))
    parameters = list(model.parameters())
    old_count = len(list(base.parameters()))
    assert all(p not in optimizer.state for p in parameters[old_count:])
    model.train()
    for update in range(2):
        optimizer.zero_grad()
        outputs = model(tokens, model_aux=aux)
        loss = outputs['atom_logits'][torch.isfinite(outputs['atom_logits'])].square().mean() + outputs['coeff_logits'].square().mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.site_pooling.parameters())
        optimizer.step()
        assert all(optimizer.state[p]['step'] == 7513 + update for p in parameters[:old_count])
        assert all(optimizer.state[p]['step'] == 1 + update for p in parameters[old_count:])
    assert model.site_pooling.input_proj.weight.grad.abs().sum() > 0
    model.eval()
    model.site_pooling.enabled = False
    with torch.no_grad():
        pairs = torch.randn(2, 4, 12)
        assert torch.equal(model.site_pooling(pairs), pairs.sum(-2))
    with pytest.raises(RuntimeError, match='inference only'):
        model.site_pooling(pairs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA BF16')
@pytest.mark.parametrize('mode', ['attention', 'mlp'])
def test_actual_width_cuda_compilation_and_neutral_initial_sum(mode):
    module=SitePairPooling(1536,4,width=480,heads=8,mode=mode).cuda()
    inputs=torch.randn(2,64,4,1536,device='cuda',dtype=torch.bfloat16)
    module.forward=torch.compile(module.forward,fullgraph=True,dynamic=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        result=module(inputs)
        assert torch.equal(result,inputs.sum(-2))
        result.float().square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
    assert module.output.weight.grad.abs().sum()>0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA BF16')
@pytest.mark.parametrize('mode', ['attention', 'mlp'])
def test_cuda_bf16_compiled_training_and_cached_generation(mode):
    _, model, aux, targets = case(mode, active=True)
    model = model.cuda();targets = targets.cuda()
    for name in ('dictionary', 'coeff_bins', 'coeff_scales'):
        setattr(aux, name, getattr(aux, name).cuda())
    aux.compound_embeddings = lambda a, c: aux.dictionary.T[a] * (aux.coeff_bins[c] * aux.coeff_scales)[..., None]
    eager = model.site_pooling.forward
    model.site_pooling.forward = torch.compile(eager, fullgraph=True, dynamic=True)
    model.train()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        output = model(targets, model_aux=aux)
        loss = output['atom_logits'][torch.isfinite(output['atom_logits'])].float().square().mean() + output['coeff_logits'].float().square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.site_pooling.parameters())
    model.site_pooling.forward = eager;model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        expected = model(targets, model_aux=aux)
        generated = torch.full_like(targets, 2);model.init_cache()
        for event in range(16):
            site, depth = divmod(event, 4);h, w = divmod(site, 2)
            hidden = model.cached_head_output(generated, aux, None, (h, w, depth), amp=True)
            logits = model.classifier(hidden)
            if depth:logits.scatter_(-1, targets[:, h, w, :depth] // 5, -torch.inf)
            coefficients = model.coefficient_logits(hidden, aux.dictionary.T[targets[:, h, w, depth] // 5], depth_index=depth)
            torch.testing.assert_close(logits, expected['atom_logits'][:, h, w, depth], atol=.05, rtol=.05)
            torch.testing.assert_close(coefficients, expected['coeff_logits'][:, h, w, depth], atol=.05, rtol=.05)
            generated[:, h, w, depth] = targets[:, h, w, depth]
