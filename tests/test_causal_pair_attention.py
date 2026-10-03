import copy

import pytest
import torch
from torch.nn import functional as F

from src.models.rqtransformer.attentions import causal_pair_attention
from src.training.rqtransformer import CompoundLaserRQTransformer, compound_objective
from tests.test_compound_pair_autoregressive import tiny_config, tiny_aux


def test_pair_attention_matches_sdpa_outputs_and_gradients():
    torch.manual_seed(83)
    tensors = [torch.randn(3, 2, 2, 8, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    expected = F.scaled_dot_product_attention(*tensors, is_causal=True)
    actual = causal_pair_attention(*tensors)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    probe = torch.randn_like(actual)
    left = torch.autograd.grad((expected * probe).sum(), tensors, retain_graph=True)
    right = torch.autograd.grad((actual * probe).sum(), tensors)
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-12)
    assert torch.count_nonzero(right[0][..., 0, :]) == 0


@pytest.mark.parametrize('residual_dropout', [0., .1])
def test_corrected_geometry_and_all_gradients_preserve_independent_dropout(residual_dropout):
    torch.manual_seed(31)
    config = tiny_config(depth=4)
    config.head.block.resid_pdrop = residual_dropout
    original = CompoundLaserRQTransformer(config, 7, 5, micro_transformer_layers=2,
        depth_specific_coeff_heads=True, pair_autoregressive=True,
        mask_seen_atoms_training=False).double().train()
    optimized = copy.deepcopy(original)
    for block in optimized.coeff_micro_transformer.blocks:
        block.attn.pair_attention_backend = "eager"
    aux = tiny_aux(depth=4)
    aux.dictionary = aux.dictionary.double()
    # tiny_aux's embedding closure captures the original dictionary; replace it.
    aux.compound_embeddings = lambda a, c: aux.dictionary.t()[a] * (aux.coeff_bins[c] * aux.coeff_scales)[..., None]
    packed = torch.arange(8).reshape(2, 1, 1, 4) + 4
    targets = F.one_hot(packed % 5, 5).double()
    outputs = []
    for model in (original, optimized):
        torch.manual_seed(791)
        out = model(packed, model_aux=aux, conditional_geometry_top_k=4)
        loss, _ = compound_objective(out['atom_logits'], out['coeff_logits'], None,
            packed // 5, targets, aux.compound_embeddings(packed // 5, packed % 5),
            atom_weight=1.5, geometry_weight=.05, accumulation=1,
            distribution_geometry=True, geometry_prediction=out['geometry_prediction'])
        loss.backward()
        outputs.append((out, loss, torch.get_rng_state()))
    for key in outputs[0][0]:
        torch.testing.assert_close(outputs[0][0][key], outputs[1][0][key], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(outputs[0][1], outputs[1][1], rtol=1e-7, atol=1e-7)
    assert torch.equal(outputs[0][2], outputs[1][2])
    assert original.state_dict().keys() == optimized.state_dict().keys()
    for (name, p), (_, q) in zip(original.named_parameters(), optimized.named_parameters()):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=2e-5, atol=2e-7, msg=name)


def test_sampling_and_attention_dropout_use_original_kernel(monkeypatch):
    from src.models.rqtransformer import attentions
    from src.models.rqtransformer.configs import AttentionBlockConfig
    config = AttentionBlockConfig(embed_dim=12, n_head=3, attn_pdrop=.1)
    attention = attentions.MultiSelfAttention(config)
    attention.pair_attention_backend = "compiled"
    def forbidden(*args, **kwargs):
        raise AssertionError('Pair shortcut must not run with attention dropout or in eval')
    monkeypatch.setattr(attentions, 'causal_pair_attention', forbidden)
    attention.train()(torch.randn(2, 2, 12))
    attention.attn_drop.p = 0.
    attention.eval()(torch.randn(2, 2, 12))
    attention.train()
    with torch.no_grad():
        attention(torch.randn(2, 2, 12))
