from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from src.models.laser_compound_rq_adapter import build_compound_model
from src.training.laser_pair_ablation import (
    configure_coefficient_policy, restore_named_adam, transplant_state)
from src.training.laser_rq_minimal import build_model
from src.training.laser_ar_factors import configure
from src.training.laser_hard_ar import token_cross_entropy
from rqvae.models.rqtransformer.attentions import AttentionStack


def case():
    torch.manual_seed(2015)
    arch = OmegaConf.create(dict(type='rq-transformer', block_size=[2, 2, 4],
        embed_dim=12, input_embed_dim=8, shared_tok_emb=True, shared_cls_emb=True,
        input_emb_vqvae=True, head_emb_vqvae=True, cumsum_depth_ctx=True,
        vocab_size=11, vocab_size_cond=1000, block_size_cond=1,
        body=dict(n_layer=2, block=dict(n_head=3, resid_pdrop=0.)),
        head=dict(n_layer=2, block=dict(n_head=3, resid_pdrop=0.))))
    model = build_compound_model(arch, num_atoms=11, coefficient_bins=33).eval()
    aux = SimpleNamespace(dictionary=torch.nn.functional.normalize(torch.randn(8, 11), dim=0),
        num_atoms=11, coeff_vocab_size=33, sparsity_level=4,
        coeff_bins=torch.linspace(-3, 3, 33), coeff_scales=torch.tensor([3., 2., 1., .5]))
    tokens = torch.empty(2, 2, 2, 8, dtype=torch.long)
    tokens[..., 0::2] = torch.tensor([0, 1, 2, 3])
    tokens[..., 1::2] = 11 + torch.tensor([10, 15, 20, 23])
    return arch, model, aux, tokens, torch.tensor([0, 999])


@pytest.mark.parametrize('orthogonal', [False, True])
@torch.no_grad()
def test_compound_sampler_cache_matches_full_teacher_forcing_across_sites(orthogonal):
    _, model, aux, truth, labels = case()
    if orthogonal:
        # Coordinate adapter does not depend on vocabulary size in this test.
        profile = SimpleNamespace(recipe=None)
        configure(profile, model, aux, coordinates='orthogonal', bins=2048)
    expected = model(truth, aux, labels)
    generated = torch.zeros_like(truth)
    generated[..., 1::2] = 11 + 16
    model.init_cache()
    for h in range(2):
        for w in range(2):
            for event in range(8):
                actual = model.cached_forward(generated[:, :h + 1], aux, labels,
                                              sample_loc=(h, w, event))
                key = 'atom_logits' if event % 2 == 0 else 'coeff_logits'
                actual = actual[:, :11] if event % 2 == 0 else actual[:, 11:]
                torch.testing.assert_close(actual, expected[key][:, h, w, event // 2],
                                           rtol=1.e-5, atol=2.e-6)
                generated[:, h, w, event] = truth[:, h, w, event]
    sampled = model.sample_sparse(2, aux, labels, amp=False, atom_top_k=11)
    assert (sampled[..., 0::2].sort(-1).values.diff(dim=-1) > 0).all()


def test_pair_predictions_are_causal_and_selected_atom_conditions_coefficient():
    _, model, aux, tokens, labels = case()
    original = model(tokens, aux, labels)
    changed = tokens.clone()
    changed[:, 0, 0, 4] = 9
    output = model(changed, aux, labels)
    torch.testing.assert_close(output['atom_logits'][:, 0, 0, :3],
                               original['atom_logits'][:, 0, 0, :3], rtol=0, atol=0)
    torch.testing.assert_close(output['coeff_logits'][:, 0, 0, :2],
                               original['coeff_logits'][:, 0, 0, :2], rtol=0, atol=0)
    assert not torch.allclose(output['coeff_logits'][:, 0, 0, 2],
                              original['coeff_logits'][:, 0, 0, 2])
    changed = tokens.clone()
    changed[:, 0, 0, 5] += 2
    output = model(changed, aux, labels)
    for key in original:
        torch.testing.assert_close(output[key][:, 0, 0, :3], original[key][:, 0, 0, :3], rtol=0, atol=0)


def test_all_pair_parameters_have_finite_gradients_on_the_single_ce_objective():
    _, model, aux, tokens, labels = case()
    model.train()
    loss = token_cross_entropy(model(tokens, aux, labels), tokens)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
    assert type(model.body_transformer) is type(model.head_transformer) is AttentionStack
    assert type(model.coeff_micro_transformer) is AttentionStack


def test_explicit_transfer_keeps_backbone_weights_and_adam_and_slices_classifiers():
    arch, pair, aux, tokens, labels = case()
    scalar = build_model(arch, num_atoms=11, coefficient_bins=33, depth=4).eval()
    optimizer = torch.optim.AdamW(scalar.parameters(), lr=.0005)
    token_cross_entropy(scalar(tokens, aux, labels), tokens).backward()
    optimizer.step()
    source = scalar.state_dict()
    states, mapping = transplant_state(pair, source, list(dict(scalar.named_parameters())), optimizer.state_dict())
    restored = torch.optim.AdamW(pair.parameters(), lr=.0005)
    restore_named_adam(restored, pair, states)
    for name, p in pair.named_parameters():
        if name.startswith(('body_transformer.', 'head_transformer.')):
            torch.testing.assert_close(p, source[name], rtol=0, atol=0)
            reference = optimizer.state[dict(scalar.named_parameters())[name]]
            torch.testing.assert_close(restored.state[p]['exp_avg'], reference['exp_avg'], rtol=0, atol=0)
        if mapping[name] is None:
            assert p not in restored.state
    torch.testing.assert_close(pair.pos_emb_d, scalar.pos_emb_d[:, ::2], rtol=0, atol=0)
    torch.testing.assert_close(pair.classifier.linear.weight, scalar.classifier.linear.weight[:11], rtol=0, atol=0)
    torch.testing.assert_close(pair.coeff_classifier[3][1].weight, scalar.classifier.linear.weight[11:], rtol=0, atol=0)
    torch.testing.assert_close(pair.pair_embeddings(tokens, aux), pair._physical_pairs(tokens, aux)[3], rtol=0, atol=0)


def test_fixed_support_soft_coefficients_do_not_consume_global_rng_and_eval_is_clean():
    _, model, aux, tokens, labels = case()
    profile = SimpleNamespace(targets=lambda tokenizer, images: (tokens, tokens))
    configure_coefficient_policy(profile, model, aux, policy='soft')
    model.train()
    before = torch.get_rng_state()
    targets, contexts = profile.targets(aux, None)
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    torch.testing.assert_close(contexts[..., 0::2], tokens[..., 0::2], rtol=0, atol=0)
    probabilities = targets['coefficient_probabilities']
    torch.testing.assert_close(probabilities.sum(-1), torch.ones_like(probabilities[..., 0]))
    logits = model(contexts, aux, labels)
    assert torch.isfinite(profile.loss(model, logits, targets))
    model.eval()
    clean, contexts = profile.targets(aux, None)
    torch.testing.assert_close(clean, tokens, rtol=0, atol=0)
    torch.testing.assert_close(contexts, tokens, rtol=0, atol=0)
    torch.testing.assert_close(profile.loss(model, logits, clean), token_cross_entropy(logits, clean), rtol=0, atol=0)


@pytest.mark.parametrize('orthogonal', [False, True])
def test_restored_pair_decoder_matches_the_historical_pair_factorization(orthogonal):
    from src.training.rqtransformer import CompoundLaserRQTransformer, OrthogonalCompoundLaserRQTransformer
    from src.models.rqtransformer.configs import RQTransformerConfig as HistoricalConfig
    from src.orthogonal_sparse_codec import ordered_support_basis
    arch, model, aux, tokens, labels = case()
    if orthogonal:
        historical = OrthogonalCompoundLaserRQTransformer(HistoricalConfig.create(arch), 11, 33,
            micro_transformer_layers=2, depth_specific_coeff_heads=True).eval()
        configure(SimpleNamespace(recipe=None), model, aux, coordinates='orthogonal', bins=2048)
        aux.orthogonal_basis = lambda atoms: ordered_support_basis(aux.dictionary.T[atoms])
        aux.orthogonal_embeddings = lambda atoms, ids: aux.orthogonal_basis(atoms)[0] * (
            aux.coeff_bins[ids] * aux.coeff_scales)[..., None]
    else:
        historical = CompoundLaserRQTransformer(HistoricalConfig.create(arch), 11, 33,
            micro_transformer_layers=2, depth_specific_coeff_heads=True, pair_autoregressive=True).eval()
    historical.load_state_dict(model.state_dict(), strict=True)
    aux.compound_embeddings = lambda atoms, ids: aux.dictionary.T[atoms] * (
        aux.coeff_bins[ids] * aux.coeff_scales)[..., None]
    packed = tokens[..., 0::2] * 33 + tokens[..., 1::2] - 11
    reference, actual = historical(packed, aux, labels), model(tokens, aux, labels)
    for key in reference:
        torch.testing.assert_close(actual[key], reference[key], rtol=1.e-5, atol=2.e-6)


def test_shared_recipe_runs_fixed_support_soft_coefficients_without_legacy_teacher(tmp_path):
    from src.training.rq_imagenet_recipes import LaserRecipe
    arch, _, aux, _, labels = case()
    recipe = dict(run_id='pair-policy-test', auxiliary_loss_terms=0, microbatch=2,
        epochs=20, lr_schedule_epochs=100, evaluation=dict(batch_size=2),
        tokenizer=dict(num_atoms=11, coefficient_bins=2048, depth=4),
        targets=dict(encoder_batch_size=1), target_mode='hard_sparse_tokens',
        objective='fixed_support_coefficient_cross_entropy', decoder='compound_pair',
        coefficient_policy='soft', ar_factors=dict(coordinates='orthogonal', coefficient_bins=2048))
    (tmp_path/'inputs').mkdir()
    OmegaConf.save(OmegaConf.create(recipe), tmp_path/'inputs/recipe.yaml')
    profile = LaserRecipe(tmp_path)
    profile.shape = (2, 2, 8)
    aux.coeff_bins, aux.coeff_vocab_size = torch.linspace(-3, 3, 2048), 2048
    aux.encoder = aux.quant_conv = lambda x: x
    model = profile.model(arch).train()
    profile.adapt(model, aux)
    images = torch.randn(2, 8, 2, 2)
    targets, codes = profile.targets(aux, images)
    verification = profile.verify_targets(aux, targets, codes)
    assert verification['soft_targets'] and not verification['stochastic_codes']
    assert verification['sampled_trajectories'] == 0
    assert profile.metadata()['soft_labels'] and profile.metadata()['teacher_bank_size'] == 0
    profile.loss(model, model(codes, aux, labels), targets).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_continuation_launcher_respects_microbatch_to_preserve_resume_data_cursor():
    import math
    from scripts.tools.launch_laser_hard_ar_imagenet import training_microbatch
    microbatch = training_microbatch([81559] * 4, requested=32)
    accumulation = math.ceil(2048 / (4 * microbatch))
    saved_microbatch_cursor = 128 * 16
    assert microbatch == 32
    assert saved_microbatch_cursor // accumulation == 128
    assert training_microbatch([81559]*4, requested=64) == 64
    assert training_microbatch([48000]*4, requested=64) == 32


def test_frozen_entry_uses_its_own_runtime_helper_even_if_workspace_package_is_loaded(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys
    from runtime.rq_imagenet_entry import freeze_shared_runtime
    root = Path(__file__).resolve().parents[1]
    freeze_shared_runtime(tmp_path, root, include_stochastic_teacher=False)
    script = '''import sys,types
from pathlib import Path
from runtime.rq_imagenet_entry import launch
stub=types.ModuleType('src.training.rq_imagenet_shared')
def main(kind,base):
 import runtime.amp_replay as replay
 assert Path(replay.__file__).resolve()==Path(base)/'runtime/amp_replay.py',replay.__file__
stub.main=main
sys.modules[stub.__name__]=stub
launch('laser')
'''
    subprocess.run([sys.executable, '-c', script], cwd=root, check=True,
                   env=dict(os.environ, LASER_RQ_BASE=str(tmp_path), CUDA_VISIBLE_DEVICES=''))
