import torch
from omegaconf import OmegaConf

from src.masked_coefficient_head import MaskedCoefficientHead, ids_to_bits, bits_to_ids
from scripts.train_church_bar_coefficients import CoefficientExperimentPrior
from src.models.rqtransformer.configs import RQTransformerConfig
from types import SimpleNamespace


def test_all_2048_coefficient_ids_roundtrip_without_changing_values():
    ids = torch.arange(2048)
    assert torch.equal(bits_to_ids(ids_to_bits(ids)), ids)


def test_unknown_bit_values_cannot_influence_predictions():
    torch.manual_seed(0)
    head = MaskedCoefficientHead(12, width=16, bits=4, depth=2).eval()
    context = torch.randn(8, 12)
    values = torch.randint(2, (8, 4)).float()
    known = torch.tensor([True, False, True, False]).expand_as(values)
    changed = torch.where(known, values, 1 - values)
    assert torch.equal(head(context, values, known, torch.zeros(8, dtype=torch.long)),
                       head(context, changed, known, torch.zeros(8, dtype=torch.long)))


def test_sampling_starts_unknown_and_never_overwrites_revealed_bits():
    torch.manual_seed(1)
    head = MaskedCoefficientHead(12, width=16, bits=4, depth=2).eval()
    ids, trace = head.sample(torch.randn(32, 12), torch.zeros(32, dtype=torch.long),
                             schedule=(1, 1, 2), return_trace=True)
    assert not trace[0][1].any()
    assert [int(mask[0].sum()) for _, mask in trace] == [0, 1, 2]
    final = ids_to_bits(ids, 4)
    for values, known in trace:
        assert torch.equal(values[known], final[known])
    assert ((ids >= 0) & (ids < 16)).all()


def small_prior(mode):
    cfg = RQTransformerConfig.create(OmegaConf.create({
        "type": "rq-transformer", "block_size": [1, 2, 2], "embed_dim": 12,
        "input_embed_dim": 4, "shared_tok_emb": True, "shared_cls_emb": True,
        "input_emb_vqvae": True, "head_emb_vqvae": True, "cumsum_depth_ctx": True,
        "vocab_size": 7, "vocab_size_cond": 1, "block_size_cond": 1,
        "embd_pdrop": 0,
        "body": {"n_layer": 1, "block": {"n_head": 3, "resid_pdrop": 0, "attn_pdrop": 0}},
        "head": {"n_layer": 1, "block": {"n_head": 3, "resid_pdrop": 0, "attn_pdrop": 0}},
    }))
    model = CoefficientExperimentPrior(cfg, 7, 8, mode=mode, width=16,
                                       micro_transformer_layers=1,
                                       depth_specific_coeff_heads=True,
                                       pair_autoregressive=True).eval()
    dictionary = torch.randn(4, 7)
    bins = torch.linspace(-1, 1, 8)
    aux = SimpleNamespace(dictionary=dictionary, coeff_bins=bins, coeff_scales=torch.ones(2))
    aux.compound_embeddings = lambda a, c: dictionary.t()[a] * bins[c].unsqueeze(-1)
    return model, aux


def test_current_and_future_coefficients_excluded_from_bar_context():
    torch.manual_seed(2)
    model, aux = small_prior("bar")
    packed = torch.tensor([[[[8, 19], [35, 44]]]])
    with torch.no_grad():
        before = model(packed, model_aux=aux)
        altered = packed.clone()
        altered[0, 0, 0, 1] = 23  # Same current atom, different current coefficient.
        altered[0, 0, 1] = torch.tensor([48, 9])  # Future support and coefficients.
        after = model(altered, model_aux=aux)
    assert torch.equal(before["coefficient_context"][0, 0, 0], after["coefficient_context"][0, 0, 0])
    assert torch.equal(before["atom_logits"][0, 0, 0], after["atom_logits"][0, 0, 0])
    assert not torch.equal(before["coefficient_context"][0, 0, 1], after["coefficient_context"][0, 0, 1])


def test_cached_coefficient_context_matches_parallel_context():
    torch.manual_seed(3)
    model, aux = small_prior("bar")
    packed = torch.tensor([[[[8, 19], [35, 44]]]])
    with torch.no_grad():
        expected = model(packed, model_aux=aux)["coefficient_context"]
        model.init_cache()
        for w in range(2):
            for d in range(2):
                hidden = model.cached_head_output(packed, aux, None, (0, w, d), amp=False)
                atom = packed[:, 0, w, d] // 8
                actual = model.refine_coefficient_hidden(hidden, aux.dictionary.t()[atom])
                torch.testing.assert_close(actual, expected[:, 0, w, d], atol=2e-6, rtol=1e-5)


def test_masked_head_can_learn_conditioned_coefficients():
    torch.manual_seed(4)
    head = MaskedCoefficientHead(8, width=24, layers=2, bits=3, depth=1)
    context = torch.eye(8)
    ids = torch.arange(8)
    depth = torch.zeros(8, dtype=torch.long)
    optimizer = torch.optim.Adam(head.parameters(), lr=.02)
    for _ in range(180):
        optimizer.zero_grad()
        loss, _ = head.loss(context, ids, depth)
        loss.backward()
        optimizer.step()
    predicted = head.sample(context, depth, schedule=(1, 1, 1), greedy=True)
    assert torch.equal(predicted, ids)
