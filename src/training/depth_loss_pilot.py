"""First-batch numerical and pairing receipts for the depth-loss experiment."""
import hashlib

import torch

from src.scaled_atom_training import soft_cross_entropy
from src.training.depth_weighted_ce import depth_weighted_soft_cross_entropy


def fingerprint(tensor):
    tensor = tensor.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def verify_first_batch(targets, codes, images):
    depth, vocabulary = targets.shape[-2:]
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(targets.device).clone() if targets.is_cuda else None
    count = min(125, targets.numel() // vocabulary)
    count -= count % depth
    q = targets.reshape(-1, vocabulary)[:count].reshape(-1, depth, vocabulary).detach()
    values = torch.sin(torch.arange(q.numel(), device=q.device, dtype=torch.float32) * .017)
    x = (values.reshape_as(q) * 3).half().requires_grad_()
    native = x.detach().clone().requires_grad_()
    actual, _ = depth_weighted_soft_cross_entropy(x, q, [1.] * depth, 128)
    expected = soft_cross_entropy(native, q, 128)
    scale = 1048576. * .25 * count / (64 * 64 * depth)
    (actual * scale).backward()
    (expected * scale).backward()
    error = float((x.grad.float() - native.grad.float()).norm() /
                  native.grad.float().norm().clamp_min(1.e-30))
    torch.testing.assert_close(actual, expected, rtol=2.e-6, atol=2.e-5)
    assert error < 1.e-3 and torch.isfinite(x.grad).all()
    assert torch.equal(cpu_rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state(targets.device))
    return dict(passed=True, native_uniform_loss_gradient_relative_error=error,
        image_fingerprint=fingerprint(images[:2]),
        code_fingerprint=fingerprint(codes.reshape(-1)[:1024]),
        target_fingerprint=fingerprint(q), target_rows=count,
        training_rng_unchanged=True, cuda=targets.is_cuda)
