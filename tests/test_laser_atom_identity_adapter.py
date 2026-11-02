import pytest
import torch

from test_laser_native_compound_adapter import case
from src.models.laser_atom_identity_adapter import build_atom_identity_model
from src.training.laser_hard_ar import token_cross_entropy


def test_zero_identity_preserves_logits_base_gradients_and_receives_gradient():
    arch, base, aux, tokens, labels = case()
    model = build_atom_identity_model(arch, num_atoms=11, coefficient_bins=33).eval()
    model.load_compound_state(base.state_dict())
    expected, actual = base(tokens, aux, labels), model(tokens, aux, labels)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    token_cross_entropy(expected, tokens).backward()
    token_cross_entropy(actual, tokens).backward()
    for name, parameter in base.named_parameters():
        torch.testing.assert_close(dict(model.named_parameters())[name].grad, parameter.grad, rtol=0, atol=0)
    gradient = model.atom_identity_embedding.weight.grad
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert torch.count_nonzero(gradient[4:]) == 0
    assert set(model.state_dict()) - set(base.state_dict()) == {'atom_identity_embedding.weight'}


@torch.no_grad()
def test_nonzero_identity_keeps_native_cache_and_atom_coefficient_causality():
    arch, base, aux, truth, labels = case()
    model = build_atom_identity_model(arch, num_atoms=11, coefficient_bins=33).eval()
    model.load_compound_state(base.state_dict())
    model.atom_identity_embedding.weight.normal_(std=.2)
    expected = model(truth, aux, labels)
    prefix = torch.zeros_like(truth); prefix[..., 1::2] = 11+16
    model.init_cache()
    for h in range(2):
        for w in range(2):
            for event in range(8):
                value = model.cached_forward(prefix[:, :h+1], aux, labels, sample_loc=(h,w,event))
                key = 'atom_logits' if event%2 == 0 else 'coeff_logits'
                value = value[:, :11] if event%2 == 0 else value[:, 11:]
                torch.testing.assert_close(value, expected[key][:,h,w,event//2], rtol=1.e-5, atol=3.e-6)
                prefix[:,h,w,event] = truth[:,h,w,event]
    changed = truth.clone(); changed[:,0,0,4] = 9
    actual = model(changed, aux, labels)
    torch.testing.assert_close(actual['atom_logits'][:,0,0,:3], expected['atom_logits'][:,0,0,:3], rtol=0, atol=0)
    torch.testing.assert_close(actual['coeff_logits'][:,0,0,:2], expected['coeff_logits'][:,0,0,:2], rtol=0, atol=0)
    assert not torch.allclose(actual['coeff_logits'][:,0,0,2], expected['coeff_logits'][:,0,0,2])


def test_meta_reload_materializes_only_new_table_and_rejects_missing_base_tensor():
    arch, base, aux, tokens, labels = case()
    with torch.device('meta'):
        model = build_atom_identity_model(arch, num_atoms=11, coefficient_bins=33)
    model.load_compound_state(base.state_dict(), assign=True)
    assert all(p.device.type == 'cpu' for p in model.parameters())
    assert torch.count_nonzero(model.atom_identity_embedding.weight) == 0
    model.eval()
    actual, expected = model(tokens, aux, labels), base(tokens, aux, labels)
    torch.testing.assert_close(actual['atom_logits'], expected['atom_logits'], rtol=0, atol=0)
    incomplete = dict(base.state_dict()); incomplete.pop('input_mlp.bias')
    with pytest.raises(ValueError): model.load_compound_state(incomplete)


@torch.no_grad()
def test_existing_adam_moments_survive_and_identity_table_starts_without_moments():
    from src.training.laser_pair_ablation import restore_named_adam
    arch, base, aux, tokens, labels = case()
    optimizer = torch.optim.AdamW(base.parameters(), lr=.0005)
    with torch.enable_grad(): token_cross_entropy(base(tokens, aux, labels), tokens).backward()
    optimizer.step()
    states = {name:optimizer.state[p] for name,p in base.named_parameters()}
    model = build_atom_identity_model(arch, num_atoms=11, coefficient_bins=33)
    model.load_compound_state(base.state_dict())
    continued = torch.optim.AdamW(model.parameters(), lr=.0005)
    restore_named_adam(continued, model, states)
    assert model.atom_identity_embedding.weight not in continued.state
    for name, parameter in model.named_parameters():
        if name not in states: continue
        for key in ['step','exp_avg','exp_avg_sq']:
            torch.testing.assert_close(continued.state[parameter][key], states[name][key], rtol=0, atol=0)
