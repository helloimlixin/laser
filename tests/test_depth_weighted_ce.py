import pytest
import torch
from omegaconf import OmegaConf

from src.scaled_atom_training import soft_cross_entropy
from src.training.depth_weighted_ce import depth_weighted_soft_cross_entropy, normalize_depth_weights
from src.training.laser_joint_recipe import JointLaserRecipe


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
@pytest.mark.parametrize('chunk', [1, 7, 128])
def test_loss_gradients_and_unweighted_diagnostics_match_dense(dtype, chunk):
    torch.manual_seed(821)
    # Noncontiguous inputs and chunk boundaries crossing depth groups.
    logits = torch.randn(2, 3, 5, 43, dtype=dtype).transpose(0, 1).requires_grad_()
    dense = logits.detach().clone().requires_grad_()
    targets = torch.randn(logits.shape).softmax(-1) * .9999
    weights = [1.5, 1.25, 1., .75, .5]
    actual, metrics = depth_weighted_soft_cross_entropy(logits, targets, weights, chunk)
    token_ce = -(targets * dense.float().log_softmax(-1)).sum(-1)
    expected = (token_ce * torch.tensor(weights)).mean()
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(logits.grad, dense.grad, rtol=2.e-3, atol=1.e-7)
    torch.testing.assert_close(metrics['unweighted_ce'], token_ce.mean())
    entropy = -(targets * targets.clamp_min(1.e-30).log()).sum(-1)
    for d in range(5):
        torch.testing.assert_close(metrics[f'ce_depth{d + 1}'], token_ce[..., d].mean())
        torch.testing.assert_close(metrics[f'kl_depth{d + 1}'], (token_ce - entropy)[..., d].mean())
    assert all(not value.requires_grad for value in metrics.values())


def test_actual_vocabulary_fp16_gradients_with_live_amp_scale():
    torch.manual_seed(822)
    logits = (torch.randn(2, 3, 5, 20481) * 3).half().requires_grad_()
    dense = logits.detach().clone().requires_grad_()
    targets = torch.zeros(logits.shape).scatter_(-1, torch.randint(20481, (2, 3, 5, 1)), .99)
    targets += .01 / 20481
    weights = [1.5, 1.25, 1., .75, .5]
    actual, _ = depth_weighted_soft_cross_entropy(logits, targets, weights, 7)
    expected = (-(targets * dense.float().log_softmax(-1)).sum(-1) * torch.tensor(weights)).mean()
    # Current scaler, microbatch weight, and full B*H*W*D reduction size.
    scale = 1048576. * .25 * 30 / 20480
    (actual * scale).backward()
    (expected * scale).backward()
    relative = (logits.grad.float() - dense.grad.float()).norm() / dense.grad.float().norm()
    assert relative < 1.e-3
    assert torch.isfinite(logits.grad).all()


def test_default_recipe_retains_exact_original_loss_and_gradients():
    profile = JointLaserRecipe.__new__(JointLaserRecipe)
    profile.recipe = OmegaConf.create({'targets': {'chunk_size': 7}})
    profile.shape = (2, 2, 5)
    logits = torch.randn(2, 2, 2, 5, 43, requires_grad=True)
    native = logits.detach().clone().requires_grad_()
    targets = torch.randn(logits.shape).softmax(-1)
    actual = profile.loss(None, logits, targets)
    expected = soft_cross_entropy(native, targets, 7)
    actual.backward()
    expected.backward()
    assert torch.equal(actual, expected)
    assert torch.equal(logits.grad, native.grad)
    assert not hasattr(profile, 'last_loss_metrics')


def test_recipe_opt_in_changes_only_loss_and_exposes_unweighted_metrics():
    profile = JointLaserRecipe.__new__(JointLaserRecipe)
    profile.recipe = OmegaConf.create({'targets': {'chunk_size': 7},
        'loss': {'depth_weights': [1.5, 1.25, 1., .75, .5]}})
    profile.shape = (2, 2, 5)
    logits = torch.randn(2, 2, 2, 5, 43, requires_grad=True)
    targets = torch.randn(logits.shape).softmax(-1)
    loss = profile.loss(None, logits, targets)
    dense = -(targets * logits.float().log_softmax(-1)).sum(-1)
    expected = (dense * torch.tensor(profile.depth_loss_weights())).mean()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(profile.last_loss_metrics['unweighted_ce'], dense.mean())


def test_uniform_weights_reproduce_native_and_common_rescaling_has_no_effect():
    torch.manual_seed(823)
    logits = torch.randn(2, 5, 43, requires_grad=True)
    targets = torch.randn(logits.shape).softmax(-1)
    native = soft_cross_entropy(logits, targets, 7)
    uniform, _ = depth_weighted_soft_cross_entropy(logits, targets, [1.] * 5, 7)
    rescaled, _ = depth_weighted_soft_cross_entropy(logits, targets, [3.] * 5, 7)
    torch.testing.assert_close(native, uniform)
    assert torch.equal(uniform, rescaled)
    expected_gradient, = torch.autograd.grad(native, logits)
    actual_gradient, = torch.autograd.grad(uniform, logits)
    torch.testing.assert_close(expected_gradient, actual_gradient)


def test_weighted_global_mean_matches_four_and_eight_rank_accumulation():
    from src.training.exact_global_batch import ExactGlobalBatchSampler
    from src.training.rq_imagenet_shared import training_backward

    class Model(torch.nn.Module):
        def __init__(self, initial):
            super().__init__()
            self.weight = torch.nn.Parameter(initial.clone())

        def forward(self, codes, **kwargs):
            return (codes @ self.weight).reshape(-1, 5, 7)

    class Scaler:
        def scale(self, loss):
            return loss * 1048576.

    torch.manual_seed(824)
    features = torch.randn(5263, 2)
    targets = torch.randn(5263, 5, 7).softmax(-1)
    data = torch.cat((features, targets.flatten(1)), 1)
    initial = torch.randn(2, 35)
    profile = JointLaserRecipe.__new__(JointLaserRecipe)
    profile.recipe = OmegaConf.create({'targets': {'chunk_size': 7},
        'loss': {'depth_weights': [1.5, 1.25, 1., .75, .5]}})
    profile.shape = (1, 1, 5)
    profile.targets = lambda aux, images: (images[:, 2:].reshape(-1, 5, 7), images[:, :2])
    layouts = []
    for world, accumulation in ((4, 8), (8, 4)):
        samplers = [ExactGlobalBatchSampler(data, 2048, world, rank, accumulation, 261001)
                    for rank in range(world)]
        batches = [list(sampler) for sampler in samplers]
        layout = []
        for step in range(3):
            gradients, all_indices = [], []
            for sampler, rank_batches in zip(samplers, batches):
                model = Model(initial)
                for micro in range(accumulation):
                    absolute = step * accumulation + micro
                    indices = rank_batches[absolute]
                    all_indices.extend(indices)
                    training_backward(profile, model, None, data[indices], None, Scaler(),
                        world * len(indices) / sampler.global_count(absolute), amp=False)
                gradients.append(model.weight.grad / 1048576.)
            actual = torch.stack(gradients).mean(0)
            reference = Model(initial)
            logits = reference(features[all_indices])
            ce = -(targets[all_indices] * logits.log_softmax(-1)).sum(-1)
            (ce * torch.tensor(profile.depth_loss_weights())).mean().backward()
            torch.testing.assert_close(actual, reference.weight.grad, rtol=3.e-5, atol=2.e-8)
            layout.append(actual)
        layouts.append(layout)
    for four, eight in zip(*layouts):
        torch.testing.assert_close(four, eight, rtol=3.e-5, atol=2.e-8)


@pytest.mark.parametrize('weights', [[], [1.] * 4, [1., 1., 1., 1., 0.],
    [1., 1., 1., 1., -1.], [1., 1., 1., 1., float('nan')],
    [1., 1., 1., 1., float('inf')], [1.e-300, 1.e300, 1., 1., 1.]])
def test_invalid_or_degenerate_weights_are_rejected(weights):
    with pytest.raises(ValueError):
        normalize_depth_weights(weights, 5)
