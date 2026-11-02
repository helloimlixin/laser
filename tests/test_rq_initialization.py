import copy

import pytest
import torch

from src.training import rqtransformer as training
from tests.test_offset_pair_prior import case


def tiny_factory(monkeypatch):
    _, model, aux, tokens, cond = case()
    monkeypatch.setattr(training, '_build_model_uninitialized', lambda *a, **kw: copy.deepcopy(model))
    return model, aux, tokens, cond


def test_fresh_factory_applies_upstream_scale_and_retains_zero_residual(monkeypatch):
    original, _, _, _ = tiny_factory(monkeypatch)
    assert original.cond_emb.weight.std() > .9
    fresh = training.build_model(12, 7)
    assert .019 < fresh.cond_emb.weight.std() < .021
    for module in fresh.modules():
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            assert torch.count_nonzero(module.bias) == 0
    assert torch.count_nonzero(fresh.coefficient_history_decoder.output.weight) == 0
    assert {k: tuple(v.shape) for k, v in original.state_dict().items()} == {
        k: tuple(v.shape) for k, v in fresh.state_dict().items()}


@torch.no_grad()
def test_legacy_reproduction_and_strict_checkpoint_restore(monkeypatch):
    original, aux, tokens, cond = tiny_factory(monkeypatch)
    legacy = training.build_model(12, 7, rq_initialization='legacy')
    for key, value in original.state_dict().items():
        torch.testing.assert_close(legacy.state_dict()[key], value, rtol=0, atol=0)
    fresh = training.build_model(12, 7)
    fresh.load_state_dict(original.state_dict(), strict=True)
    before, after = original(tokens, aux, cond), fresh(tokens, aux, cond)
    for field in before:
        torch.testing.assert_close(before[field], after[field], rtol=0, atol=0)


def test_efficient_recursive_builder_initializes_only_after_all_modules_attached(monkeypatch):
    from src.training import rq_initialization
    calls = []
    def initialize(model):
        calls.append(model)
        assert hasattr(model, 'coefficient_history_decoder')
        assert model.__class__.__name__ == 'OffsetPairRQTransformer'
        return model
    monkeypatch.setattr(rq_initialization, 'initialize_rq_prior', initialize)
    with torch.device('meta'):
        model = training.build_model(18432, 16384, coeff_vocab_size=2048, sparsity_level=4,
            physical_pair_context=True, efficient_pair_backbone=True,
            compound_micro_transformer_layers=2, compound_depth_specific_coeff_heads=True,
            coefficient_history_layers=2, coefficient_history_normalize_fields=True)
    assert calls == [model]


def test_invalid_policy_and_cli_default(monkeypatch):
    tiny_factory(monkeypatch)
    with pytest.raises(ValueError, match='initialization policy'):
        training.build_model(12, 7, rq_initialization='unknown')
    args = training.build_parser().parse_args(['--checkpoint','teacher.pt','--data','data','--output','out'])
    assert args.rq_initialization == 'upstream'
