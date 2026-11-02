import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from src.models.dictionary_learner import DictionaryLearning
from src.training import laser_hard_ar as hard
from src.training.rq_imagenet_recipes import LaserRecipe
from src.training.rq_imagenet_shared import training_backward
from src.training.laser_rq_minimal import build_model


def example():
    torch.manual_seed(317)
    arch = OmegaConf.create(dict(type='rq-transformer', block_size=[2, 2, 4],
        embed_dim=12, input_embed_dim=4, shared_tok_emb=True, shared_cls_emb=True,
        input_emb_vqvae=True, head_emb_vqvae=True, cumsum_depth_ctx=True,
        vocab_size=7, vocab_size_cond=1000, block_size_cond=1,
        body=dict(n_layer=1, block=dict(n_head=3, resid_pdrop=0.)),
        head=dict(n_layer=1, block=dict(n_head=3, resid_pdrop=0.))))
    model = build_model(arch, num_atoms=7, coefficient_bins=5, depth=4)
    aux = SimpleNamespace(encoder=lambda x: x, quant_conv=lambda x: x,
        dictionary=torch.nn.functional.normalize(torch.randn(4, 7), dim=0),
        coeff_bins=torch.linspace(-3, 3, 5), coeff_scales=torch.tensor([3., 2., 1., .5]),
        num_atoms=7, coeff_vocab_size=5, sparsity_level=4)
    return model, aux, torch.randn(2, 4, 2, 2), torch.tensor([0, 999]), OmegaConf.create(dict(encoder_batch_size=1))


@pytest.mark.parametrize('depth', [1, 4])
def test_encoding_matches_stage1_support_and_final_coefficients(depth):
    torch.manual_seed(391)
    dictionary = torch.nn.functional.normalize(torch.randn(16, 31), dim=0)
    signals = torch.randn(3, 2, 16)
    reference = DictionaryLearning(num_embeddings=31, embedding_dim=16, sparsity_level=depth)
    atoms, values = hard.sparse_encoding(signals, dictionary, depth)
    expected_atoms, expected_values = reference.batch_omp_with_support(signals.reshape(-1, 16).T, dictionary)
    torch.testing.assert_close(atoms.reshape(-1, depth), expected_atoms, rtol=0, atol=0)
    torch.testing.assert_close(values.reshape(-1, depth), expected_values, rtol=1e-5, atol=2e-6)


def test_nearest_coefficient_bins_match_exhaustive_search_including_clipping_and_ties():
    bins = torch.tensor([-3., -1., 1., 3.])
    values = torch.cat((torch.linspace(-10, 10, 137), torch.tensor([-2., 0., 2.])))
    expected = (values[:, None] - bins).abs().argmin(-1)
    torch.testing.assert_close(hard.nearest_coefficient_ids(values, bins), expected, rtol=0, atol=0)


def test_hard_image_encoding_consumes_no_rng_and_has_no_soft_target_tensors():
    _, aux, images, _, options = example()
    rng = torch.get_rng_state().clone()
    tokens = hard.image_tokens(aux, images, options)
    hard.verify_tokens(aux, tokens)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    torch.manual_seed(51)
    torch.testing.assert_close(hard.image_tokens(aux, images, options), tokens, rtol=0, atol=0)
    assert isinstance(tokens, torch.Tensor) and tokens.dtype == torch.long


def test_hard_ce_is_exact_next_token_likelihood_and_all_parameters_receive_gradients():
    model, aux, images, labels, options = example()
    tokens = hard.image_tokens(aux, images, options)
    logits = model(tokens, aux, labels)
    actual = hard.token_cross_entropy(logits, tokens)
    atom = -logits['atom_logits'].log_softmax(-1).gather(-1, tokens[..., 0::2, None]).squeeze(-1)
    coefficient = -logits['coeff_logits'].log_softmax(-1).gather(
        -1, (tokens[..., 1::2] - aux.num_atoms)[..., None]).squeeze(-1)
    torch.testing.assert_close(actual, torch.stack((atom, coefficient), -1).mean())
    actual.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_hard_recipe_uses_shared_training_and_cannot_resume_the_soft_recipe(tmp_path):
    model, aux, images, labels, options = example()
    folder = tmp_path/'inputs'
    folder.mkdir()
    config = dict(run_id='hard-test', name='hard test', auxiliary_loss_terms=0,
        microbatch=2, epochs=20, lr_schedule_epochs=100, evaluation=dict(batch_size=2),
        target_mode='hard_sparse_tokens', objective='hard_next_token_cross_entropy',
        targets=dict(encoder_batch_size=1))
    OmegaConf.save(OmegaConf.create(config), folder/'recipe.yaml')
    profile = LaserRecipe(tmp_path)
    profile.shape = (2, 2, 8)
    targets, tokens = profile.targets(aux, images)
    assert targets is tokens
    assert profile.verify_targets(aux, targets, tokens)['sampled_trajectories'] == 0
    metadata = profile.metadata()
    assert not metadata['stochastic_codes'] and not metadata['soft_labels']
    assert metadata['teacher_bank_size'] == 0 and metadata['objective'] == 'hard_next_token_cross_entropy'
    loss = training_backward(profile, model, aux, images, labels,
        torch.amp.GradScaler('cuda', enabled=False), .5, amp=False)
    assert torch.isfinite(loss)
    with pytest.raises(AssertionError):
        profile.verify_resume(dict(recipe=dict(config, target_mode='conditional_soft')))


def test_frozen_hard_runtime_excludes_and_never_imports_trajectory_teacher(tmp_path):
    from runtime.rq_imagenet_entry import freeze_shared_runtime
    root = Path(__file__).resolve().parents[1]
    freeze_shared_runtime(tmp_path, root, include_stochastic_teacher=False)
    for name in ('stochastic_compound.py', 'training/stochastic_image_pairs.py',
                 'training/stochastic_omp_graph.py', 'training/omp_joint_targets.py'):
        assert not (tmp_path/'source/src'/name).exists()
    script = f"""import sys
sys.path.insert(0, {str(tmp_path/'source')!r})
import src.training.rq_imagenet_shared
import src.training.laser_hard_ar
from src.training.laser_rq_minimal import build_model
for name in ('src.stochastic_compound','src.training.stochastic_image_pairs',
             'src.training.stochastic_omp_graph','src.training.omp_joint_targets'):
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable, '-c', script], check=True, cwd=tmp_path,
        env=dict(os.environ, RQ_UPSTREAM_ROOT=str(root/'third_party/rq-vae-transformer'),
                 CUDA_VISIBLE_DEVICES=''))
