import copy
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from src.models.offset_pair_prior import OffsetPairRQTransformer
from src.models.rqtransformer.configs import RQTransformerConfig
from src.training.compound_history import attach_coefficient_history_decoder
from src.training.rqtransformer import CompoundLaserRQTransformer, build_model


def case(coefficient_bins=5):
    torch.manual_seed(211)
    config = RQTransformerConfig.create(OmegaConf.create({
        'type': 'rq-transformer', 'block_size': [2, 2, 4], 'embed_dim': 12,
        'input_embed_dim': 4, 'shared_tok_emb': True, 'shared_cls_emb': True,
        'input_emb_vqvae': True, 'head_emb_vqvae': True, 'cumsum_depth_ctx': True,
        'vocab_size': 7, 'vocab_size_cond': 1000, 'block_size_cond': 1,
        'body': {'n_layer': 1, 'block': {'n_head': 3, 'resid_pdrop': 0.}},
        'head': {'n_layer': 1, 'block': {'n_head': 3, 'resid_pdrop': 0.}},
    }))
    native = CompoundLaserRQTransformer(config, 7, coefficient_bins, micro_transformer_layers=2,
        depth_specific_coeff_heads=True, pair_autoregressive=True)
    native = attach_coefficient_history_decoder(native, width=8, layers=1,
        heads=2, dropout=0., normalize_fields=True).eval()
    model = OffsetPairRQTransformer.from_pair_backbone(copy.deepcopy(native))
    aux = SimpleNamespace(dictionary=torch.randn(4, 7), coeff_bins=torch.linspace(-1, 1, coefficient_bins),
                          coeff_scales=torch.tensor([1., 2., 3., 4.]), num_atoms=7,
                          coeff_vocab_size=coefficient_bins, sparsity_level=4)
    aux.compound_embeddings = lambda a, c: aux.dictionary.T[a] * (
        aux.coeff_bins[c] * aux.coeff_scales)[..., None]
    tokens = torch.empty(1, 2, 2, 8, dtype=torch.long)
    tokens[..., 0::2] = torch.tensor([0, 1, 2, 3])
    tokens[..., 1::2] = 7 + torch.tensor([0, 1, coefficient_bins-2, coefficient_bins-1])
    return native, model, aux, tokens, torch.tensor([999])


def test_logits_loss_gradients_and_checkpoint_exactly_match_original_pair_backbone():
    native, model, aux, tokens, cond = case()
    atoms, coefficients = model.fields(tokens)
    # Product encoding appears only in this legacy oracle, never in the new path.
    old = native(atoms * 5 + coefficients, aux, cond)
    new = model(tokens, aux, cond)
    for field in ('atom_logits', 'coeff_logits'):
        torch.testing.assert_close(old[field], new[field], rtol=0, atol=0)
    def loss(out):
        return (-out['atom_logits'].log_softmax(-1).gather(-1, atoms[..., None]).mean()
                -out['coeff_logits'].log_softmax(-1).gather(-1, coefficients[..., None]).mean())
    loss(old).backward()
    loss(new).backward()
    for (a, pa), (b, pb) in zip(native.named_parameters(), model.named_parameters(), strict=True):
        assert a == b
        assert pa.grad is not None, a
        torch.testing.assert_close(pa.grad, pb.grad, rtol=0, atol=0)
    model.load_state_dict(native.state_dict(), strict=True)


@pytest.mark.parametrize('learned_history', [False, True])
@torch.no_grad()
def test_cached_predictions_match_teacher_forced_predictions_and_only_four_native_steps(learned_history):
    _, model, aux, target, cond = case()
    if learned_history:
        model.coefficient_history_decoder.output.weight.normal_(std=.08)
        # Residual quantization permits reusing an atom with a new coefficient.
        model.mask_seen_atoms_training = False
        target[..., 4] = target[..., 0]
    output = model(target, aux, cond)
    generated = torch.zeros_like(target)
    generated[..., 1::2] = 9
    calls = []
    hook = model.head_transformer.register_forward_pre_hook(lambda *args: calls.append(1))
    model.init_cache()
    # cached_forward invokes the stack's cached method, so count it directly.
    original = model.head_transformer.cached_forward
    def counted(value):
        calls.append(1)
        return original(value)
    model.head_transformer.cached_forward = counted
    for h in range(2):
        for w in range(2):
            for event in range(8):
                logits = model.cached_forward(generated, aux, cond, sample_loc=(h, w, event))
                expected = output['coeff_logits' if event % 2 else 'atom_logits'][:, h, w, event // 2]
                actual = logits[:, 7:] if event % 2 else logits[:, :7]
                torch.testing.assert_close(actual, expected, rtol=5e-6, atol=1e-6)
                invalid = logits[:, :7] if event % 2 else logits[:, 7:]
                assert torch.isneginf(invalid).all()
                generated[:, h, w, event] = target[:, h, w, event]
    hook.remove()
    assert len(calls) == 2 * 2 * 4


@torch.no_grad()
def test_joint_candidates_share_history_and_commit_only_the_selected_pair():
    _, model, aux, tokens, cond = case()
    model.coefficient_history_decoder.output.weight.normal_(std=.08)
    full = model(tokens, aux, cond)
    model.init_cache()
    for h in range(2):
        for w in range(2):
            for depth in range(4):
                model.cached_forward(tokens, aux, cond, sample_loc=(h, w, 2*depth))
                before = model.coefficient_history_decoder._next_event
                scorer = model.joint_coefficient_scorer(aux)
                atoms = tokens[:, h, w, 2*depth, None]
                branches = torch.cat((atoms, (atoms+1)%7), dim=-1)
                candidates = scorer(torch.tensor([0]), branches)
                assert model.coefficient_history_decoder._next_event == before
                committed = model.cached_forward(tokens, aux, cond, sample_loc=(h, w, 2*depth+1))[:, 7:]
                torch.testing.assert_close(candidates[:, 0], committed, atol=2e-6, rtol=1e-5)
                torch.testing.assert_close(committed, full['coeff_logits'][:, h, w, depth], atol=2e-6, rtol=1e-5)
                assert model.coefficient_history_decoder._next_event == before+1


@torch.no_grad()
def test_atom_conditioning_past_coefficient_signal_and_future_causality():
    _, model, aux, tokens, cond = case()
    baseline = model(tokens, aux, cond)
    changed = tokens.clone()
    changed[:, 0, 0, 0] = 6
    current_atom = model(changed, aux, cond)
    torch.testing.assert_close(baseline['atom_logits'][:, 0, 0, 0],
                               current_atom['atom_logits'][:, 0, 0, 0], rtol=0, atol=0)
    assert not torch.equal(baseline['coeff_logits'][:, 0, 0, 0],
                           current_atom['coeff_logits'][:, 0, 0, 0])
    changed = tokens.clone()
    changed[:, 0, 0, 1] = 11
    coefficient = model(changed, aux, cond)
    for field in baseline:
        torch.testing.assert_close(baseline[field][:, 0, 0, 0], coefficient[field][:, 0, 0, 0], rtol=0, atol=0)
        assert not torch.equal(baseline[field][:, 0, 0, 1], coefficient[field][:, 0, 0, 1])
    changed = tokens.clone()
    changed[:, -1, -1, -1] = 8
    future = model(changed, aux, cond)
    for field in baseline:
        torch.testing.assert_close(baseline[field], future[field], rtol=0, atol=0)


@torch.no_grad()
def test_sampling_uses_offset_ids_and_distinct_atoms():
    _, model, aux, tokens, cond = case()
    sampled = model.sample_sparse(2, aux, torch.tensor([151, 19]), amp=False)
    assert sampled.shape == (2, 2, 2, 8)
    atoms, coefficients = model.fields(sampled)
    assert (atoms.sort(-1).values.diff(dim=-1) > 0).all()
    assert ((coefficients >= 0) & (coefficients < 5)).all()
    assert ((sampled[..., 1::2] >= 7) & (sampled[..., 1::2] < 12)).all()
    model.init_cache()
    with pytest.raises(ValueError, match='current atom'):
        model.cached_forward(tokens, aux, cond, sample_loc=(0, 0, 1))


def test_production_builder_keeps_original_parameter_geometry():
    with torch.device('meta'):
        model = build_model(18432, 16384, coeff_vocab_size=2048, sparsity_level=4,
            physical_pair_context=True, efficient_pair_backbone=True,
            compound_micro_transformer_layers=2, compound_depth_specific_coeff_heads=True,
            coefficient_history_layers=2, coefficient_history_normalize_fields=True)
    assert tuple(model.pos_emb_d.shape) == (1, 4, 1536)
    assert tuple(model.classifier.linear.weight.shape) == (16384, 1536)
    assert len(model.coeff_classifier) == 4
    assert model.token_vocabulary_size == 18432
    assert model.token_block_size == (8, 8, 8)


@pytest.mark.parametrize('bins', [16, 32, 64, 128, 256, 512])
def test_lower_bit_coefficients_keep_offset_transport_and_cached_predictions(bins):
    _, model, aux, tokens, cond = case(bins)
    full = model(tokens, aux, cond)
    assert full['coeff_logits'].shape[-1] == bins
    model.init_cache()
    for event in range(8):
        logits = model.cached_forward(tokens, aux, cond, amp=False, sample_loc=(0, 0, event))
        expected = full['coeff_logits' if event % 2 else 'atom_logits'][:, 0, 0, event//2]
        actual = logits[:, 7:] if event % 2 else logits[:, :7]
        torch.testing.assert_close(actual, expected, rtol=5e-6, atol=1e-6)
    loss = full['coeff_logits'].float().square().mean() + full['atom_logits'].float().square().mean()
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    model.eval()
    sampled = model.sample_sparse(1, aux, cond, atom_top_p=1., coeff_top_p=1., amp=False)
    assert ((sampled[..., 1::2] >= 7) & (sampled[..., 1::2] < 7+bins)).all()
