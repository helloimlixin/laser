import copy

import pytest
import torch

from src.training.continuation_lr import RampHoldCosineSchedule


POLICY = dict(anchor_step=5, anchor_lr=.004, peak_lr=.008, ramp_steps=2,
              hold_until_step=10, total_steps=20, min_lr=.0001)


@pytest.mark.parametrize("downward", [False, True])
def test_continuation_has_no_lr_jump_and_reaches_ramp_hold_decay_endpoints(downward):
    parameter = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.AdamW([parameter], lr=.004)
    parameter.grad = torch.ones_like(parameter)
    opt.step()
    moments = copy.deepcopy(opt.state_dict()["state"])
    policy = dict(POLICY, peak_lr=.002) if downward else POLICY
    schedule = RampHoldCosineSchedule(opt, policy=policy, completed_steps=5)
    assert opt.param_groups[0]["lr"] == .004
    for key, value in moments[0].items():
        torch.testing.assert_close(opt.state_dict()["state"][0][key], value, rtol=0, atol=0)
    expected = ({6: .003, 7: .002, 10: .002, 15: .00105, 20: .0001}
                if downward else {6: .006, 7: .008, 10: .008, 15: .00405, 20: .0001})
    for step in range(6, 21):
        schedule.step()
        if step in expected:
            assert opt.param_groups[0]["lr"] == pytest.approx(expected[step])
    with pytest.raises(ValueError, match="exhausted"):
        schedule.step()


@pytest.mark.parametrize("resume_step", [5, 6, 7, 10, 15, 20])
@pytest.mark.parametrize("downward", [False, True])
def test_resume_reproduces_optimizer_and_schedule_trajectory(resume_step, downward):
    parameter = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.AdamW([parameter], lr=.004)
    policy = dict(POLICY, peak_lr=.002) if downward else POLICY
    schedule = RampHoldCosineSchedule(opt, policy=policy, completed_steps=5)
    for _ in range(5, resume_step):
        parameter.grad = parameter.square().detach()
        opt.step()
        schedule.step()
    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored_opt = torch.optim.AdamW([restored_parameter], lr=.004)
    restored_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    restored = RampHoldCosineSchedule(restored_opt, policy=policy,
        completed_steps=resume_step, state_dict=schedule.state_dict())
    for _ in range(resume_step, 20):
        assert restored.get_last_lr() == schedule.get_last_lr()
        parameter.grad = parameter.square().detach()
        restored_parameter.grad = restored_parameter.square().detach()
        opt.step(); schedule.step()
        restored_opt.step(); restored.step()
        torch.testing.assert_close(parameter, restored_parameter, rtol=0, atol=0)
    with pytest.raises(ValueError, match="policy changed"):
        RampHoldCosineSchedule(restored_opt, policy=dict(POLICY, peak_lr=.009),
            completed_steps=20, state_dict=restored.state_dict())
    with pytest.raises(ValueError, match="step mismatch"):
        RampHoldCosineSchedule(restored_opt, policy=policy,
            completed_steps=19, state_dict=restored.state_dict())


def test_explicit_policy_revision_preserves_lr_moments_and_subsequent_resume():
    parameter = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.AdamW([parameter], lr=.004)
    old = RampHoldCosineSchedule(opt, policy=POLICY, completed_steps=5)
    for _ in range(5, 8):
        parameter.grad = parameter.square().detach()
        opt.step(); old.step()
    moments = copy.deepcopy(opt.state_dict()["state"])
    revision = dict(POLICY, anchor_step=8, anchor_lr=.008, peak_lr=.004,
                    ramp_steps=2, hold_until_step=10)
    schedule = RampHoldCosineSchedule(opt, policy=revision, completed_steps=8,
        state_dict=old.state_dict(), revision_from=POLICY)
    assert opt.param_groups[0]["lr"] == .008
    for key, value in moments[0].items():
        torch.testing.assert_close(opt.state_dict()["state"][0][key], value, rtol=0, atol=0)
    schedule.step()
    assert schedule.get_last_lr()[0] == pytest.approx(.006)
    restored = RampHoldCosineSchedule(opt, policy=revision, completed_steps=9,
                                     state_dict=schedule.state_dict(), revision_from=POLICY)
    restored.step()
    assert restored.get_last_lr()[0] == .004


@pytest.mark.parametrize("failure", ["missing_source", "wrong_source", "wrong_anchor", "wrong_lr"])
def test_policy_revision_rejects_unverified_migration(failure):
    parameter = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.AdamW([parameter], lr=.008)
    old = RampHoldCosineSchedule(opt, policy=POLICY, completed_steps=8)
    revision = dict(POLICY, anchor_step=8, anchor_lr=.008, peak_lr=.004,
                    ramp_steps=2, hold_until_step=10)
    source = POLICY
    if failure == "missing_source": source = None
    if failure == "wrong_source": source = dict(POLICY, peak_lr=.009)
    if failure == "wrong_anchor": revision["anchor_step"] = 7
    if failure == "wrong_lr": opt.param_groups[0]["lr"] = .007
    with pytest.raises(ValueError):
        RampHoldCosineSchedule(opt, policy=revision, completed_steps=8,
                              state_dict=old.state_dict(), revision_from=source)
