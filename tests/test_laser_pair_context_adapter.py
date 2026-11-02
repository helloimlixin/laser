from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from src.models.laser_pair_context_adapter import build_pair_context_model
from src.training.laser_ar_factors import configure
from src.training.laser_hard_ar import token_cross_entropy
from src.training.laser_pair_ablation import restore_named_adam
from src.training.laser_rq_minimal import build_model


def case(orthogonal):
    torch.manual_seed(261015)
    arch = OmegaConf.create(dict(type='rq-transformer', block_size=[2, 2, 4],
        embed_dim=12, input_embed_dim=8, shared_tok_emb=True, shared_cls_emb=True,
        input_emb_vqvae=True, head_emb_vqvae=True, cumsum_depth_ctx=True,
        vocab_size=11, vocab_size_cond=1000, block_size_cond=1,
        body=dict(n_layer=2, block=dict(n_head=3, resid_pdrop=0.)),
        head=dict(n_layer=2, block=dict(n_head=3, resid_pdrop=0.))))
    original = build_model(arch, num_atoms=11, coefficient_bins=33, depth=4).eval()
    candidate = build_pair_context_model(arch, num_atoms=11, coefficient_bins=33).eval()
    candidate.load_scalar_state(original.state_dict())
    aux = SimpleNamespace(dictionary=torch.nn.functional.normalize(torch.randn(8, 11), dim=0),
        num_atoms=11, coeff_vocab_size=33, sparsity_level=4,
        coeff_bins=torch.linspace(-3, 3, 33), coeff_scales=torch.tensor([3., 2., 1., .5]))
    if orthogonal:
        for model in (original, candidate):
            configure(SimpleNamespace(recipe=None), model, aux, coordinates='orthogonal')
    tokens = torch.empty(2, 2, 2, 8, dtype=torch.long)
    tokens[..., 0::2] = torch.tensor([0, 1, 2, 3])
    tokens[..., 1::2] = 11 + torch.tensor([10, 15, 20, 23])
    return original, candidate, aux, tokens, torch.tensor([0, 999])


@pytest.mark.parametrize('orthogonal', [False, True])
def test_zero_residual_preserves_all_logits_and_existing_gradients_exactly(orthogonal):
    original, candidate, aux, tokens, labels = case(orthogonal)
    outputs = [model(tokens, aux, labels) for model in (original, candidate)]
    for key in outputs[0]:
        torch.testing.assert_close(outputs[1][key], outputs[0][key], rtol=0, atol=0)
    for output in outputs:
        token_cross_entropy(output, tokens).backward()
    for name, p in original.named_parameters():
        q = dict(candidate.named_parameters())[name]
        torch.testing.assert_close(q.grad, p.grad, rtol=0, atol=0)
    for name, p in candidate.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert candidate.pair_context_residual[-1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize('orthogonal', [False, True])
@torch.no_grad()
def test_learned_residual_cache_is_causal_and_never_changes_physical_decoding(orthogonal):
    original, candidate, aux, truth, labels = case(orthogonal)
    candidate.pair_context_residual[-1].weight.normal_(std=.02)
    expected = candidate(truth, aux, labels)
    assert not torch.allclose(expected['coeff_logits'], original(truth, aux, labels)['coeff_logits'])
    for left, right in zip(candidate._physical_pairs(truth, aux), original._physical_pairs(truth, aux)):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    generated = torch.zeros_like(truth)
    generated[..., 1::2] = 11 + 16
    candidate.init_cache()
    for h in range(2):
        for w in range(2):
            for event in range(8):
                logits = candidate.cached_forward(generated[:, :h + 1], aux, labels,
                    sample_loc=(h, w, event))
                actual = logits[:, :11] if event % 2 == 0 else logits[:, 11:]
                key = 'atom_logits' if event % 2 == 0 else 'coeff_logits'
                torch.testing.assert_close(actual, expected[key][:, h, w, event // 2],
                    rtol=1.e-5, atol=3.e-6)
                generated[:, h, w, event] = truth[:, h, w, event]
    changed = truth.clone()
    changed[:, 0, 0, 5] += 2
    prediction = candidate(changed, aux, labels)
    for key in expected:
        torch.testing.assert_close(prediction[key][:, 0, 0, :3], expected[key][:, 0, 0, :3],
            rtol=0, atol=0)


def test_old_adam_moments_are_preserved_and_only_residual_state_starts_fresh():
    original, candidate, aux, tokens, labels = case(True)
    optimizer = torch.optim.AdamW(original.parameters(), lr=.0005)
    token_cross_entropy(original(tokens, aux, labels), tokens).backward()
    optimizer.step()
    candidate.load_scalar_state(original.state_dict())
    restored = torch.optim.AdamW(candidate.parameters(), lr=.0005)
    states = {name: optimizer.state[p] for name, p in original.named_parameters()}
    restore_named_adam(restored, candidate, states)
    for name, p in candidate.named_parameters():
        if name not in states:
            assert p not in restored.state
        else:
            for key, value in states[name].items():
                torch.testing.assert_close(restored.state[p][key], value, rtol=0, atol=0)
    malformed = dict(original.state_dict())
    malformed.pop('pos_emb_d')
    with pytest.raises(ValueError, match='complete scalar state'):
        candidate.load_scalar_state(malformed)


def test_recipe_explicitly_selects_pair_context_and_freezes_its_source(tmp_path):
    from runtime.rq_imagenet_entry import freeze_shared_runtime
    from src.training.rq_imagenet_recipes import LaserRecipe
    from pathlib import Path
    _, candidate, _, _, _ = case(False)
    recipe = dict(run_id='context-test', auxiliary_loss_terms=0, microbatch=2,
        epochs=20, lr_schedule_epochs=100, evaluation=dict(batch_size=2),
        tokenizer=dict(num_atoms=11, coefficient_bins=33, depth=4),
        target_mode='hard_sparse_tokens', objective='token_cross_entropy',
        targets=dict(encoder_batch_size=1), decoder='scalar_pair_context')
    (tmp_path/'inputs').mkdir()
    OmegaConf.save(OmegaConf.create(recipe), tmp_path/'inputs/recipe.yaml')
    profile = LaserRecipe(tmp_path)
    built = profile.model(candidate.config)
    assert type(built) is type(candidate)
    assert 'pair context' in profile.metadata()['architecture']
    root = Path(__file__).resolve().parents[1]
    freeze_shared_runtime(tmp_path/'runtime-test', root, include_stochastic_teacher=False)
    name = 'src/models/laser_pair_context_adapter.py'
    assert (tmp_path/'runtime-test/source'/name).read_bytes() == (root/name).read_bytes()


def test_meta_loading_materializes_only_new_parameters_and_assigns_scalar_checkpoint():
    original, _, aux, tokens, labels = case(False)
    with torch.device('meta'):
        candidate = build_pair_context_model(original.config, num_atoms=11, coefficient_bins=33)
    for module in (candidate.coefficient_identity, candidate.pair_context_residual):
        module.to_empty(device='cpu').apply(candidate._init_weights)
    candidate.load_scalar_state(original.state_dict(), assign=True)
    candidate.eval()
    assert all(not p.is_meta for p in candidate.parameters())
    for key, expected in original(tokens, aux, labels).items():
        torch.testing.assert_close(candidate(tokens, aux, labels)[key], expected, rtol=0, atol=0)
