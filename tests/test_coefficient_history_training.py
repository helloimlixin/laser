import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from types import SimpleNamespace

from scripts.train_church_bar_coefficients import CoefficientExperimentPrior
from src.models.rqtransformer.configs import RQTransformerConfig
from src.coefficient_history_training import (
    EpochStream, categorical_draw, recovery_mask, sample_coefficient_span,
    scheduled_lr, recovery_strength, sign_nll,
)


def tiny_model():
    cfg = RQTransformerConfig.create(OmegaConf.create({
        "type": "rq-transformer", "block_size": [1, 2, 2], "embed_dim": 12,
        "input_embed_dim": 4, "shared_tok_emb": True, "shared_cls_emb": True,
        "input_emb_vqvae": True, "head_emb_vqvae": True, "cumsum_depth_ctx": True,
        "vocab_size": 7, "vocab_size_cond": 1, "block_size_cond": 1,
        "embd_pdrop": 0,
        "body": {"n_layer": 1, "block": {"n_head": 3, "resid_pdrop": 0, "attn_pdrop": 0}},
        "head": {"n_layer": 1, "block": {"n_head": 3, "resid_pdrop": 0, "attn_pdrop": 0}},
    }))
    model = CoefficientExperimentPrior(cfg, 7, 8, mode="categorical", micro_transformer_layers=1,
                                       depth_specific_coeff_heads=True, pair_autoregressive=True)
    dictionary, bins = torch.randn(4, 7), torch.linspace(-1, 1, 8)
    aux = SimpleNamespace(dictionary=dictionary, coeff_bins=bins, coeff_scales=torch.ones(2))
    aux.compound_embeddings = lambda a, c: dictionary.t()[a] * bins[c].unsqueeze(-1)
    return model, aux


def test_generated_span_does_not_read_any_coefficient_targets_inside_it():
    torch.manual_seed(4)
    model, aux = tiny_model()
    truth = torch.tensor([[[[8, 19], [35, 44]]]])
    altered = (truth // 8) * 8 + 7 - truth % 8
    uniforms = torch.tensor([[.23, .44, .61, .77]])
    first = sample_coefficient_span(model, aux, truth, 0, 2, uniforms)
    second = sample_coefficient_span(model, aux, altered, 0, 2, uniforms)
    assert torch.equal(first, second)
    assert torch.equal(first // 8, truth // 8)
    assert model.training
    assert all(parameter.grad is None for parameter in model.parameters())


def test_parallel_cache_prefill_matches_replaying_the_real_prefix():
    torch.manual_seed(5)
    model, aux = tiny_model()
    truth = torch.tensor([[[[8, 19], [35, 44]]]])
    uniforms = torch.tensor([[.23, .77]])
    jumped = sample_coefficient_span(model, aux, truth, 1, 1, uniforms)
    sequential = truth.clone()
    model.eval()
    model.init_cache()
    with torch.no_grad():
        for w in range(2):
            for d in range(2):
                hidden = model.cached_head_output(sequential, aux, None, (0, w, d), amp=False)
                if w == 1:
                    atom = sequential[:, 0, w, d] // 8
                    refined = model.refine_coefficient_hidden(hidden, aux.dictionary.t()[atom])
                    coefficient = categorical_draw(model.classify_coefficients(refined, d), uniforms[:, d])
                    sequential[:, 0, w, d] = atom * 8 + coefficient
    model.init_cache()
    assert torch.equal(jumped, sequential)
    assert torch.equal(jumped[:, :, :1], truth[:, :, :1])


def test_sign_loss_is_exact_marginal_probability_and_has_gradients():
    torch.manual_seed(6)
    logits = torch.randn(2, 3, 8, requires_grad=True)
    targets = torch.tensor([[0, 3, 4], [7, 2, 5]])
    probability = logits.softmax(-1)
    expected = -torch.where(targets >= 4, probability[..., 4:].sum(-1), probability[..., :4].sum(-1)).log()
    actual = sign_nll(logits, targets)
    torch.testing.assert_close(actual, expected)
    actual.mean().backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_epoch_stream_resume_preserves_data_order_and_partial_batches():
    stream = EpochStream(11, seed=9)
    batches = [stream.next(4) for _ in range(3)]
    assert sorted(torch.cat([b[0] for b in batches]).tolist()) == list(range(11))
    assert [len(b[0]) for b in batches] == [4, 4, 3]
    assert [b[2] for b in batches] == [False, False, True]
    resumed = EpochStream(11, seed=100)
    resumed.load_state_dict(stream.state_dict())
    for _ in range(7):
        left, right = stream.next(4), resumed.next(4)
        assert torch.equal(left[0], right[0]) and left[1:] == right[1:]


def test_recovery_targets_follow_generated_history_and_schedule_keeps_warmup_clean():
    mask = recovery_mask((2, 3, 4), start_site=1, span_sites=2).flatten()
    assert torch.equal(torch.where(mask)[0], torch.arange(5, 16))
    assert recovery_strength(5) == 0
    assert recovery_strength(10) == .5
    assert recovery_strength(15) == 1
    assert scheduled_lr(5, 100) == 2e-4
    assert abs(scheduled_lr(100, 100) - 1e-5) < 1e-12
