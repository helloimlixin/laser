from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from src.models.laser_historical_rq_adapter import build_historical_pair_model
from src.training.laser_ar_factors import configure
from src.training.laser_hard_ar import token_cross_entropy
from src.orthogonal_sparse_codec import ordered_support_basis


def setup_case(orthogonal):
    torch.manual_seed(261015)
    arch = OmegaConf.create(dict(type='rq-transformer', block_size=[2, 2, 4],
        embed_dim=12, input_embed_dim=8, shared_tok_emb=True, shared_cls_emb=True,
        input_emb_vqvae=True, head_emb_vqvae=True, cumsum_depth_ctx=True,
        vocab_size=11, vocab_size_cond=1000, block_size_cond=1,
        body=dict(n_layer=2, block=dict(n_head=3, resid_pdrop=0.)),
        head=dict(n_layer=2, block=dict(n_head=3, resid_pdrop=0.))))
    model = build_historical_pair_model(arch, num_atoms=11, coefficient_bins=33,
        history_width=12, history_heads=3, history_dropout=0.).eval()
    with torch.no_grad():
        model.input_mlp.bias.normal_(std=.2)
        model.coefficient_history_decoder.output.weight.normal_(std=.02)
    aux = SimpleNamespace(dictionary=torch.nn.functional.normalize(torch.randn(8, 11), dim=0),
        num_atoms=11, coeff_vocab_size=33, sparsity_level=4,
        coeff_bins=torch.linspace(-3, 3, 33), coeff_scales=torch.tensor([3., 2., 1., .5]))
    if orthogonal:
        configure(SimpleNamespace(recipe=None), model, aux, coordinates='orthogonal', bins=2048)
    aux.compound_embeddings = lambda atoms, ids: aux.dictionary.T[atoms] * (
        aux.coeff_bins[ids] * aux.coeff_scales)[..., None]
    aux.orthogonal_basis = lambda atoms: ordered_support_basis(aux.dictionary.T[atoms])
    aux.orthogonal_embeddings = lambda atoms, ids: aux.orthogonal_basis(atoms)[0] * (
        aux.coeff_bins[ids] * aux.coeff_scales[:atoms.shape[-1]])[..., None]
    tokens = torch.empty(2, 2, 2, 8, dtype=torch.long)
    tokens[..., 0::2] = torch.tensor([0, 1, 2, 3])
    tokens[..., 1::2] = 11 + torch.tensor([10, 15, 20, 23])
    return arch, model, aux, tokens, torch.tensor([0, 999])


@pytest.mark.parametrize('orthogonal', [False, True])
@torch.no_grad()
def test_full_historical_model_matches_with_trained_projection_bias_and_history(orthogonal):
    from src.training.rqtransformer import CompoundLaserRQTransformer, OrthogonalCompoundLaserRQTransformer
    from src.training.compound_history import attach_coefficient_history_decoder
    from src.models.rqtransformer.configs import RQTransformerConfig
    arch, model, aux, tokens, labels = setup_case(orthogonal)
    cls = OrthogonalCompoundLaserRQTransformer if orthogonal else CompoundLaserRQTransformer
    options = {} if orthogonal else dict(pair_autoregressive=True)
    old = cls(RQTransformerConfig.create(arch), 11, 33,
        micro_transformer_layers=2, depth_specific_coeff_heads=True, **options)
    attach_coefficient_history_decoder(old, width=12, layers=2, heads=3, dropout=0.)
    old.eval().load_state_dict(model.state_dict(), strict=True)
    expected, actual = old(model.pack(tokens), aux, labels), model(tokens, aux, labels)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=1.e-5, atol=3.e-6)


@pytest.mark.parametrize('orthogonal', [False, True])
@torch.no_grad()
def test_historical_full_history_cache_matches_teacher_forcing_and_is_causal(orthogonal):
    _, model, aux, truth, labels = setup_case(orthogonal)
    expected = model(truth, aux, labels)
    generated = torch.zeros_like(truth)
    generated[..., 1::2] = 11 + 16
    model.init_cache()
    for h in range(2):
        for w in range(2):
            for event in range(8):
                output = model.cached_forward(generated[:, :h + 1], aux, labels,
                    sample_loc=(h, w, event))
                key = 'atom_logits' if event % 2 == 0 else 'coeff_logits'
                actual = output[:, :11] if event % 2 == 0 else output[:, 11:]
                torch.testing.assert_close(actual, expected[key][:, h, w, event // 2],
                                           rtol=1.e-5, atol=3.e-6)
                generated[:, h, w, event] = truth[:, h, w, event]
    changed = truth.clone()
    changed[:, 0, 0, 5] += 3
    output = model(changed, aux, labels)
    for key in expected:
        torch.testing.assert_close(output[key][:, 0, 0, :3],
                                   expected[key][:, 0, 0, :3], rtol=0, atol=0)


def test_historical_adapter_trains_all_parameters_using_one_categorical_objective():
    _, model, aux, tokens, labels = setup_case(True)
    model.train()
    token_cross_entropy(model(tokens, aux, labels), tokens).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
