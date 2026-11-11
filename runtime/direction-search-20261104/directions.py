"""Measured optimization directions using the same underlying Adam state."""
import math
import torch


@torch.no_grad()
def step(optimizer, mode):
    parameters = [p for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
    if mode not in {'adam', 'no_momentum', 'gradient', 'reverse_adam'}:
        raise ValueError(mode)
    before = [p.detach().clone() for p in parameters]
    gradients = [p.grad for p in parameters]
    gradient_norm = torch.stack(torch._foreach_norm(gradients)).square().sum().sqrt()
    optimizer.step()
    delta = torch._foreach_sub(parameters, before)
    adam_norm = torch.stack(torch._foreach_norm(delta)).square().sum().sqrt()
    del delta
    if mode == 'reverse_adam':
        # Reflect the completed AdamW update through its starting point.
        torch._foreach_mul_(parameters, -1)
        torch._foreach_add_(parameters, before, alpha=2)
    elif mode == 'gradient':
        scale = float(adam_norm / gradient_norm.clamp_min(1e-30))
        for parameter, previous, gradient in zip(parameters, before, gradients):
            parameter.copy_(previous).add_(gradient, alpha=-scale)
    delta = torch._foreach_sub(parameters, before)
    update_norm = torch.stack(torch._foreach_norm(delta)).square().sum().sqrt()
    dot = torch.stack([(d * g).sum() for d, g in zip(delta, gradients)]).sum()
    cosine = dot / (update_norm * gradient_norm).clamp_min(1e-30)
    result = dict(direction=mode, reference_adam_update_l2=float(adam_norm),
                  actual_update_l2=float(update_norm), gradient_l2=float(gradient_norm),
                  update_gradient_cosine=float(cosine))
    if not all(math.isfinite(x) for x in result.values() if isinstance(x, float)):
        raise FloatingPointError('Nonfinite direction/update diagnostics')
    return result


def configure(optimizer, mode):
    if mode == 'no_momentum':
        for group in optimizer.param_groups:
            group['betas'] = (0.0, group['betas'][1])
    # Current-gradient and reverse-Adam modes retain and advance the actual
    # Adam statistics. Their checkpoints also store the required direction,
    # so resuming them must use this driver rather than ordinary Adam updates.


def self_test():
    initial = torch.tensor([1.0, -2.0, 0.25])
    gradient = torch.tensor([0.3, -0.4, 0.2])
    results = {}
    for mode in ['adam', 'no_momentum', 'gradient', 'reverse_adam']:
        parameter = torch.nn.Parameter(initial.clone())
        optimizer = torch.optim.AdamW([parameter], lr=0.01, weight_decay=0.0001,
                                      betas=(0.9, 0.95))
        configure(optimizer, mode)
        parameter.grad = gradient.clone()
        proof = step(optimizer, mode)
        results[mode] = (parameter.detach().clone(), proof)
        assert int(optimizer.state[parameter]['step']) == 1
    torch.testing.assert_close(results['reverse_adam'][0] - initial,
                               -(results['adam'][0] - initial), rtol=1e-5, atol=3e-7)
    assert results['gradient'][1]['update_gradient_cosine'] < -0.9999
    assert abs(results['gradient'][1]['actual_update_l2'] /
               results['gradient'][1]['reference_adam_update_l2'] - 1) < 1e-5
    return {mode: proof for mode, (_, proof) in results.items()}
