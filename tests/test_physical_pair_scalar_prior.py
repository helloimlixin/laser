from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from src.models.physical_pair_scalar_prior import PhysicalPairScalarRQTransformer
from src.models.rqtransformer.configs import RQTransformerConfig
from src.training.rqtransformer import build_model


def case():
    torch.manual_seed(211)
    config = RQTransformerConfig.create(OmegaConf.create({
        "type": "rq-transformer", "block_size": [2, 2, 8], "embed_dim": 12,
        "input_embed_dim": 4, "shared_tok_emb": True, "shared_cls_emb": True,
        "input_emb_vqvae": True, "head_emb_vqvae": True, "cumsum_depth_ctx": True,
        "vocab_size": 12, "vocab_size_cond": 1000, "block_size_cond": 1,
        "body": {"n_layer": 1, "block": {"n_head": 3, "resid_pdrop": 0.}},
        "head": {"n_layer": 1, "block": {"n_head": 3, "resid_pdrop": 0.}},
    }))
    model = PhysicalPairScalarRQTransformer(config, 7).eval()
    aux = SimpleNamespace(dictionary=torch.randn(4, 7), coeff_bins=torch.linspace(-1, 1, 5),
                          coeff_scales=torch.tensor([1., 2., 3., 4.]), num_atoms=7,
                          coeff_vocab_size=5, sparsity_level=4)
    tokens = torch.empty(1, 2, 2, 8, dtype=torch.long)
    tokens[..., 0::2] = torch.tensor([0, 1, 2, 3])
    tokens[..., 1::2] = 7 + torch.tensor([0, 1, 3, 4])
    return model, aux, tokens, torch.tensor([999])


@torch.no_grad()
def test_cached_predictions_match_full_forward_without_future_targets():
    model, aux, target, cond = case()
    output = model(target, aux, cond)
    generated = torch.zeros_like(target)
    generated[..., 1::2] = 9
    model.init_cache()
    for h in range(2):
        for w in range(2):
            for event in range(8):
                logits = model.cached_forward(generated[:, :h + 1], aux, cond,
                                               sample_loc=(h, w, event))
                expected = output["atom_logits" if event % 2 == 0 else "coeff_logits"][:, h, w, event // 2]
                actual = logits[:, :7] if event % 2 == 0 else logits[:, 7:]
                torch.testing.assert_close(actual, expected, rtol=3e-6, atol=8e-7)
                generated[:, h, w, event] = target[:, h, w, event]


@torch.no_grad()
def test_atom_conditioning_and_coefficient_causality():
    model, aux, tokens, cond = case()
    baseline = model(tokens, aux, cond)
    changed = tokens.clone()
    changed[:, 0, 0, 0] = 6
    atom_change = model(changed, aux, cond)
    torch.testing.assert_close(baseline["atom_logits"][:, 0, 0, 0],
                               atom_change["atom_logits"][:, 0, 0, 0], rtol=0, atol=0)
    assert not torch.equal(baseline["coeff_logits"][:, 0, 0, 0], atom_change["coeff_logits"][:, 0, 0, 0])
    changed = tokens.clone()
    changed[:, 0, 0, 1] = 11
    coefficient_change = model(changed, aux, cond)
    for key in baseline:
        torch.testing.assert_close(baseline[key][:, 0, 0, 0], coefficient_change[key][:, 0, 0, 0], rtol=0, atol=0)
        assert not torch.equal(baseline[key][:, 0, 0, 1], coefficient_change[key][:, 0, 0, 1])
    changed = tokens.clone()
    changed[:, -1, -1, -1] = 8
    future_change = model(changed, aux, cond)
    for key in baseline:
        torch.testing.assert_close(baseline[key], future_change[key], rtol=0, atol=0)


def test_unique_atom_support_and_finite_soft_target_gradients():
    model, aux, tokens, cond = case()
    output = model(tokens, aux, cond)
    atoms = tokens[..., 0::2]
    for depth in range(1, 4):
        assert torch.isneginf(output["atom_logits"][..., depth, :].gather(-1, atoms[..., :depth])).all()
    atom_nll = -output["atom_logits"].log_softmax(-1).gather(-1, atoms[..., None]).mean()
    probabilities = torch.randn_like(output["coeff_logits"]).softmax(-1)
    coefficient_nll = -(probabilities * output["coeff_logits"].log_softmax(-1)).sum(-1).mean()
    (atom_nll + coefficient_nll).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    torch.manual_seed(13)
    sampled = model.sample_sparse(2, aux, cond=torch.tensor([151, 19]), amp=False)
    support = sampled[..., 0::2].sort(-1).values
    assert (support.diff(dim=-1) > 0).all()
    assert ((sampled[..., 1::2] >= 7) & (sampled[..., 1::2] < 12)).all()


@torch.no_grad()
def test_physical_scaling_is_applied_once_and_checkpoint_restores():
    model, aux, tokens, cond = case()
    physical = model._physical_pairs(tokens, aux)[2]
    torch.testing.assert_close(physical, aux.coeff_bins[tokens[..., 1::2] - 7] * aux.coeff_scales)
    restored, _, _, _ = case()
    restored.load_state_dict(model.state_dict(), strict=True)
    for key, value in model(tokens, aux, cond).items():
        torch.testing.assert_close(value, restored(tokens, aux, cond)[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="scalar-token"):
        build_model(12, 7, physical_pair_context=True, compound=True)
