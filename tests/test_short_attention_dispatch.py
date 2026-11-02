import torch
import pytest

from src.models.rqtransformer.attentions import causal_short_attention, MultiSelfAttention
from src.models.rqtransformer.configs import AttentionBlockConfig


@pytest.mark.parametrize("length", [2, 4, 8])
def test_short_attention_outputs_and_gradients(length):
    torch.manual_seed(74)
    inputs = [torch.randn(3, 2, length, 16, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    reference = torch.nn.functional.scaled_dot_product_attention(*inputs, is_causal=True)
    actual = causal_short_attention(*inputs)
    torch.testing.assert_close(actual, reference, rtol=1e-11, atol=1e-11)
    cotangent = torch.randn_like(actual)
    expected = torch.autograd.grad(reference, inputs, cotangent, retain_graph=True)
    gradients = torch.autograd.grad(actual, inputs, cotangent)
    for got, wanted in zip(gradients, expected):
        torch.testing.assert_close(got, wanted, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("length", [2, 4])
def test_dispatch_uses_matching_training_sequence_and_preserves_sampling(length, monkeypatch):
    from src.models.rqtransformer import attentions

    calls = []
    original = attentions.causal_short_attention

    def observed(*args):
        calls.append(args[0].shape[-2])
        return original(*args)

    monkeypatch.setattr(attentions, "causal_short_attention", observed)
    layer = MultiSelfAttention(AttentionBlockConfig(embed_dim=32, n_head=2, attn_pdrop=0))
    layer.short_attention_backend = "compiled"
    layer(torch.randn(2, length, 32))
    assert calls == [length]
    with torch.no_grad():
        layer(torch.randn(2, length, 32))
    layer.eval()(torch.randn(2, length, 32))
    layer.train()
    layer.attn_drop.p = .1
    layer(torch.randn(2, length, 32))
    assert calls == [length]
