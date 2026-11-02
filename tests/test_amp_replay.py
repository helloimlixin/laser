import copy

import pytest
import torch

from runtime.amp_replay import ReplayWindow, recoverable_amp_step


def test_overflow_replays_identical_stochastic_accumulation_without_losing_adam_update():
    torch.manual_seed(11)
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Dropout(.4))
    initial = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=62600)
    scaler = torch.amp.GradScaler('cpu', init_scale=16.)
    batches = [(torch.randn(2, 4), torch.randn(2, 3), index) for index in range(2)]
    window = ReplayWindow('cpu');window.begin()
    observed = []

    def backward(images, labels, index):
        jitter = torch.randn_like(images) * .1
        loss = (model(images + jitter) - labels).square().mean() / 2
        observed.append((index, jitter.clone(), loss.detach().clone()))
        scaler.scale(loss).backward()

    for images, labels, index in batches:
        window.append(images, labels, index);backward(images, labels, index)
    next(model.parameters()).grad.flatten()[0] = float('inf')
    events = []
    _, retries = recoverable_amp_step(model.parameters(), optimizer, scaler, max_norm=1.,
        replay_backward=lambda: window.replay(backward), on_overflow=events.append)
    scheduler.step()
    assert retries == 1 and events == [dict(retry=1, previous_scale=16., scale=8.)]
    assert len(observed) == 4
    for first, repeated in zip(observed[:2], observed[2:]):
        assert first[0] == repeated[0]
        torch.testing.assert_close(first[1], repeated[1], rtol=0, atol=0)
        torch.testing.assert_close(first[2], repeated[2], rtol=0, atol=0)
    assert all(int(state['step']) == 1 for state in optimizer.state.values())
    assert scheduler.last_epoch == 1

    reference = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Dropout(.4))
    reference.load_state_dict(initial)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=.0005)
    reference_scaler = torch.amp.GradScaler('cpu', init_scale=8.)
    torch.set_rng_state(window.rng['cpu'])
    for images, labels, _ in batches:
        loss = (reference(images + torch.randn_like(images) * .1) - labels).square().mean() / 2
        reference_scaler.scale(loss).backward()
    reference_scaler.unscale_(reference_optimizer)
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1., error_if_nonfinite=True)
    reference_scaler.step(reference_optimizer);reference_scaler.update()
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_persistent_overflow_is_bounded_and_never_updates_parameters():
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([parameter], lr=.0005)
    scaler = torch.amp.GradScaler('cpu', init_scale=16.)

    def bad_backward():
        scaler.scale(parameter.sum()).backward()
        parameter.grad[0] = float('inf')

    bad_backward()
    with pytest.raises(FloatingPointError, match='persisted'):
        recoverable_amp_step([parameter], optimizer, scaler, max_norm=1.,
                              replay_backward=bad_backward, max_retries=2)
    assert not optimizer.state
    torch.testing.assert_close(parameter, torch.ones(2), rtol=0, atol=0)


@pytest.mark.parametrize('enabled', [False, True])
def test_non_amp_failures_do_not_update_optimizer(enabled):
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([parameter])
    scaler = torch.amp.GradScaler('cpu', init_scale=1., enabled=enabled)
    scaler.scale(parameter.sum()).backward()
    # Finite FP32 gradients whose FP32 L2 norm overflows are not an AMP event.
    parameter.grad.fill_(1e30)
    with pytest.raises(RuntimeError, match='non-finite'):
        recoverable_amp_step([parameter], optimizer, scaler, max_norm=1.,
                              replay_backward=lambda: pytest.fail('must not replay'))
    assert not optimizer.state
    torch.testing.assert_close(parameter, torch.ones(2), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_fp16_cuda_overflow_recovers_with_identical_online_draws():
    torch.manual_seed(41);torch.cuda.manual_seed_all(41)
    model = torch.nn.Linear(4, 3).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005)
    scaler = torch.amp.GradScaler('cuda', init_scale=16.)
    images = torch.randn(2, 4);labels = torch.randn(2, 3)
    window = ReplayWindow('cuda:0');window.begin();window.append(images, labels, 0)
    observed = []

    def backward(images, labels, _):
        with torch.autocast('cuda', dtype=torch.float16):
            noisy = images.cuda() + torch.randn(2, 4, device='cuda') * .1
            prediction = model(noisy)
        loss = (prediction.float() - labels.cuda()).square().mean()
        observed.append((noisy.clone(), loss.detach().clone()))
        scaler.scale(loss).backward()

    backward(images, labels, 0)
    next(model.parameters()).grad.flatten()[0] = float('inf')
    _, retries = recoverable_amp_step(model.parameters(), optimizer, scaler,
        max_norm=1., replay_backward=lambda: window.replay(backward))
    assert retries == 1 and scaler.get_scale() == 8.
    for first, repeated in zip(observed[0], observed[1]):
        torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    assert all(int(state['step']) == 1 for state in optimizer.state.values())
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
def test_replay_restores_private_coefficient_generator_alongside_global_rng():
    import torch
    from runtime.amp_replay import ReplayWindow
    generator = torch.Generator().manual_seed(721)
    window = ReplayWindow('cpu', generators=[generator])
    window.begin()
    expected = []
    for index in range(3):
        images, labels = torch.zeros(1), torch.zeros(1, dtype=torch.long)
        window.append(images, labels, index)
        expected.append((torch.rand(5), torch.rand(7, generator=generator)))
    global_after, private_after = torch.get_rng_state(), generator.get_state()
    def backward(images, labels, index):
        torch.testing.assert_close(torch.rand(5), expected[index][0], rtol=0, atol=0)
        torch.testing.assert_close(torch.rand(7, generator=generator), expected[index][1], rtol=0, atol=0)
    window.replay(backward)
    torch.testing.assert_close(torch.get_rng_state(), global_after, rtol=0, atol=0)
    torch.testing.assert_close(generator.get_state(), private_after, rtol=0, atol=0)

