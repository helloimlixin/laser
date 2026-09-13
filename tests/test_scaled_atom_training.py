import torch
from src.scaled_atom_training import (TrainingScaledAtomRQ,soft_cross_entropy,
    FidLearningRateControl,advance_cosine_scheduler)
from src.scaled_atom_rq import ScaledAtomRQ


def test_full_vocabulary_chunked_targets_match_dense_rq_at_every_depth():
    torch.manual_seed(826)
    dictionary=torch.randn(7,11)
    levels=torch.tensor([-2.,-.6,.6,2.])
    z=torch.randn(3,2,2,7)
    reference=ScaledAtomRQ(dictionary,levels)
    model=TrainingScaledAtomRQ(dictionary,levels)
    expected,codes=reference.soft_codes(z,.5,stochastic=False)
    actual,chosen=model.get_soft_codes(z,.5,stochastic=False,chunk_size=5)
    torch.testing.assert_close(actual,expected,atol=3e-6,rtol=3e-5)
    assert torch.equal(chosen,codes)
    # With the same chunk size as the dense reference, RNG trajectories match too.
    torch.manual_seed(722)
    expected,codes=reference.soft_codes(z,.5,stochastic=True)
    torch.manual_seed(722)
    actual,chosen=model.get_soft_codes(z,.5,stochastic=True,chunk_size=128)
    torch.testing.assert_close(actual,expected,atol=3e-6,rtol=3e-5)
    assert torch.equal(chosen,codes)


def test_chunked_ce_matches_dense_loss_and_gradient_with_nonunit_target_sums():
    torch.manual_seed(278)
    for dtype in (torch.float32,torch.float16):
        logits=torch.randn(2,3,4,43,dtype=dtype,requires_grad=True)
        reference=logits.detach().clone().requires_grad_()
        targets=torch.rand_like(logits,dtype=torch.float32).softmax(-1)*.9999
        expected=-(targets*reference.float().log_softmax(-1)).sum(-1).mean()
        actual=soft_cross_entropy(logits,targets,chunk_size=7)
        expected.backward()
        actual.backward()
        torch.testing.assert_close(actual,expected)
        torch.testing.assert_close(logits.grad,reference.grad,atol=3e-7,rtol=2e-3)


def test_lr_control_ignores_small_fid_jitter_and_reduces_persistent_regression():
    control=FidLearningRateControl()
    assert control.observe(1,30.) is None
    assert control.observe(5,20.) is None
    assert control.observe(10,20.1) is None
    assert control.observe(15,20.5) is None
    assert control.observe(20,21.)=='two_regressions'
    assert control.multiplier==.5
    assert control.observe(25,21.1) is None
    assert control.observe(30,21.2)=='two_regressions'
    assert control.multiplier==.25


def test_lr_reduction_is_applied_once_instead_of_compounding_each_update():
    import math
    parameter=torch.nn.Parameter(torch.ones(()))
    optimizer=torch.optim.SGD([parameter],lr=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100)
    multiplier=1.
    for step in range(1,40):
        if step in (5,20):
            multiplier*=.5
            optimizer.param_groups[0]['lr']*=.5
        optimizer.step()
        advance_cosine_scheduler(scheduler,optimizer,multiplier)
        expected=.01*(1+math.cos(math.pi*step/100))/2*multiplier
        assert abs(optimizer.param_groups[0]['lr']-expected)<1e-12
