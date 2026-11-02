"""Causality, cache and gradient checks for the maintained compound recipe."""
import copy

import pytest
import torch

from src.training.compound_history import attach_coefficient_history_decoder
from src.training.compound_balance import shared_hidden_gradient_metrics
from src.training import rqtransformer as training
from tests.test_compound_coefficient_history import setup_model
from tests.test_compound_pair_autoregressive import tiny_config


def normalized_case():
    base, _, aux, atoms, coefficients = setup_model()
    model = attach_coefficient_history_decoder(copy.deepcopy(base), width=12, heads=3,
                                               layers=2, dropout=0., normalize_fields=True)
    torch.nn.init.normal_(model.coefficient_history_decoder.output.weight, std=.1)
    return model, aux, atoms, coefficients


@torch.no_grad()
def test_normalized_history_is_causal_and_uses_current_atom():
    model, aux, atoms, coefficients = normalized_case()
    tokens = atoms * 5 + coefficients
    expected = model(tokens, model_aux=aux)
    for event in (0, 2, 3, 7, 11):
        changed = tokens.clone().reshape(-1)
        changed[event] = changed[event] // 5 * 5 + (changed[event] % 5 + 2) % 5
        changed[event + 1:] = torch.randint(0, 35, changed[event + 1:].shape)
        actual = model(changed.reshape_as(tokens), model_aux=aux)
        for key in expected:
            torch.testing.assert_close(actual[key].reshape(12, -1)[:event + 1],
                                       expected[key].reshape(12, -1)[:event + 1], rtol=0, atol=0)
    changed = tokens.clone()
    changed[0, 0, 0, 1] = 3 * 5 + coefficients[0, 0, 0, 1]
    actual = model(changed, model_aux=aux)
    torch.testing.assert_close(actual['atom_logits'][0, 0, 0, 1],
                               expected['atom_logits'][0, 0, 0, 1], rtol=0, atol=0)
    assert not torch.equal(actual['coeff_logits'][0, 0, 0, 1], expected['coeff_logits'][0, 0, 0, 1])


@torch.no_grad()
def test_normalized_history_cache_reads_only_generated_prefix():
    model, aux, atoms, coefficients = normalized_case()
    truth = atoms * 5 + coefficients
    generated = torch.full_like(truth, 2)
    expected = model(truth, model_aux=aux)
    model.init_cache()
    for event in range(12):
        site, depth = divmod(event, 3)
        h, w = divmod(site, 2)
        hidden = model.cached_head_output(generated, aux, None, (h, w, depth), amp=False)
        atom_logits = model.classifier(hidden)
        if depth:
            atom_logits.scatter_(-1, atoms[:, h, w, :depth], -torch.inf)
        torch.testing.assert_close(atom_logits[0], expected['atom_logits'].reshape(12, -1)[event],
                                   atol=2e-6, rtol=1e-5)
        actual = model.coefficient_logits(hidden, aux.dictionary.T[atoms[:, h, w, depth]],
                                          depth_index=depth)
        torch.testing.assert_close(actual[0], expected['coeff_logits'].reshape(12, -1)[event],
                                   atol=2e-6, rtol=1e-5)
        generated.reshape(-1)[event] = truth.reshape(-1)[event]
    model.init_cache()
    assert model.coefficient_history_decoder._next_event == 0


def test_factory_builds_normalized_history_and_cli_exposes_options(monkeypatch):
    original = training.CompoundLaserRQTransformer
    def tiny_model(config, **kwargs):
        return original(tiny_config(depth=3), **kwargs)
    monkeypatch.setattr(training, 'CompoundLaserRQTransformer', tiny_model)
    model = training.build_model(12, 7, compound=True, compound_pair_autoregressive=True,
                                 coeff_vocab_size=5, coefficient_history_layers=2,
                                 coefficient_history_width=12, coefficient_history_heads=3,
                                 coefficient_history_normalize_fields=True)
    assert len(model.coefficient_history_decoder.blocks) == 2
    assert model.coefficient_history_decoder.field_norms is not None
    options = {action.dest for action in training.build_parser()._actions}
    assert {'coefficient_history_layers', 'coefficient_history_normalize_fields',
            'compound_gradient_diagnostics_every'} <= options


def test_gradient_probe_preserves_parameter_gradients_and_detects_both_branches():
    _, model, aux, atoms, coefficients = setup_model(activate=True)
    reference = copy.deepcopy(model)
    model.capture_shared_hidden = True
    output = model(atoms * 5 + coefficients, model_aux=aux)
    a = -output['atom_logits'].log_softmax(-1).gather(-1, atoms[..., None]).squeeze(-1)
    c = -output['coeff_logits'].log_softmax(-1).gather(-1, coefficients[..., None]).squeeze(-1)
    metrics = shared_hidden_gradient_metrics(a, c, output['shared_hidden'], atom_weight=.5)
    assert all(parameter.grad is None for parameter in model.parameters())
    an = metrics['train/shared_atom_gradient_norm_rank0']
    cn = metrics['train/shared_coefficient_gradient_norm_rank0']
    assert an > 0 and cn > 0
    assert metrics['train/weighted_shared_gradient_ratio_rank0'] == pytest.approx(.5 * an / cn)
    assert metrics['train/equal_norm_atom_weight_rank0'] == pytest.approx(cn / an)
    (.5 * a.mean() + c.mean()).backward()
    expected = reference(atoms * 5 + coefficients, model_aux=aux)
    ar = -expected['atom_logits'].log_softmax(-1).gather(-1, atoms[..., None]).mean()
    cr = -expected['coeff_logits'].log_softmax(-1).gather(-1, coefficients[..., None]).mean()
    (.5 * ar + cr).backward()
    for left, right in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
