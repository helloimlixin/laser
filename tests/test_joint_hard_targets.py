import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.models.laser_joint_tokenizer import JointAtomCoefficientRQ
from src.training.laser_joint_recipe import JointLaserRecipe


def example(policy):
    torch.manual_seed(123)
    quantizer = JointAtomCoefficientRQ(torch.eye(3),
        [[0, 0, 0, 1, 1, 2, 2]] * 2,
        [[0., -1., 1., -1., 1., -1., 1.],
         [0., -.5, .5, -.5, .5, -.5, .5]])
    profile = JointLaserRecipe.__new__(JointLaserRecipe)
    profile.recipe = OmegaConf.create({'targets': {'policy': policy,
        'temperature': [.2, .1], 'chunk_size': 3}})
    profile.shape = (2, 2, 2)
    class Aux:
        def encode(self, images):
            return images
    aux = Aux()
    aux.quantizer = quantizer
    images = torch.randn(3, 2, 2, 3)
    return profile, aux, images


def test_hard_targets_are_exact_deterministic_native_quantizer_codes():
    profile, aux, images = example('hard_deterministic')
    state = torch.random.get_rng_state().clone()
    targets, codes = profile.targets(aux, images)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert torch.equal(codes, aux.quantizer.quantize(images)['codes'])
    assert torch.equal(targets, F.one_hot(codes, aux.quantizer.vocab_size).float())
    assert targets.dtype == torch.float32 and not targets.requires_grad
    proof = profile.verify_targets(aux, targets, codes)
    assert not proof['soft_targets'] and not proof['stochastic_prefixes']
    assert proof['entropy'] == 0. and proof['temperature'] is None


def test_hard_target_loss_and_gradient_match_integer_cross_entropy():
    profile, aux, images = example('hard_deterministic')
    targets, codes = profile.targets(aux, images)
    actual_logits = torch.randn_like(targets, requires_grad=True)
    expected_logits = actual_logits.detach().clone().requires_grad_()
    actual = profile.loss(None, actual_logits, targets)
    expected = F.cross_entropy(expected_logits.flatten(0, -2), codes.flatten())
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_logits.grad, expected_logits.grad)


def test_default_recipe_retains_exact_soft_targets_and_rng_consumption():
    profile, aux, images = example('soft_stochastic')
    del profile.recipe.targets.policy
    torch.manual_seed(456)
    actual = profile.targets(aux, images)
    state = torch.random.get_rng_state().clone()
    torch.manual_seed(456)
    expected = aux.quantizer.get_soft_codes(images, [.2, .1], stochastic=True, chunk_size=3)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert torch.equal(torch.random.get_rng_state(), state)


def test_encoder_batching_preserves_targets_precision_and_random_stream():
    profile, aux, _ = example('soft_stochastic')
    images = torch.randn(37, 2, 2, 3)
    torch.manual_seed(456)
    expected = profile.targets(aux, images)
    expected_rng = torch.random.get_rng_state().clone()
    profile.encoder_batch_size = 64
    torch.manual_seed(456)
    actual = profile.targets(aux, images)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert actual[0].dtype == torch.float32
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
