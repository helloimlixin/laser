import pytest
import torch

from src.models.laser_joint_tokenizer import JointAtomCoefficientRQ
from src.training.laser_compact_joint_vocabulary import reconstruct_fixed_codes, refit_compact_book
from src.training.laser_joint_vocabulary import allocate_shared_coefficients, pack_codebook


def test_adaptive_layout_preserves_integer_semantics_and_reduces_final_error():
    torch.manual_seed(26101720)
    dictionary = torch.nn.functional.normalize(torch.randn(9, 3), dim=0)
    allocations = [2, 4, 8]
    books = {n: torch.cat((-torch.linspace(2., .3, n//2), torch.linspace(.3, 2., n//2))).expand(3, -1)
             for n in [2, 4, 6, 8]}
    spec = pack_codebook(dictionary, [books]*4, [allocations]*4)
    q = JointAtomCoefficientRQ(spec['dictionary'], spec['atom_ids'], spec['coefficients'])
    codes = torch.randint(q.vocab_size, (120, 4))
    signals = reconstruct_fixed_codes(q, codes)+torch.randn(120, 9)*.08
    old_layout = q.atom_ids.clone()
    old_sign = q.coefficients.sign()
    report = refit_compact_book(q, signals, codes)
    assert report['fixed_assignment_mse_after'] < report['fixed_assignment_mse_before']
    assert torch.equal(q.atom_ids, old_layout)
    assert torch.equal(q.coefficients.sign(), old_sign)
    for atom in range(3):
        selected = (q.atom_ids[0] == atom) & (torch.arange(q.vocab_size) > 0)
        assert (q.coefficients[:, selected].diff(dim=-1) >= 0).all()
    torch.testing.assert_close(q.dictionary.norm(dim=0), torch.ones(3))
    actual = (signals-reconstruct_fixed_codes(q, codes)).square().mean()
    torch.testing.assert_close(actual, torch.tensor(report['fixed_assignment_mse_after']))


def test_adaptive_budget_keeps_more_directions_than_uniform_product():
    # At the same 12-entry budget, six directions all retain both signs,
    # while a uniform four-level product can retain only three directions.
    errors = {2:torch.tensor([5., 4., 3., 2., 1., .5]),
              4:torch.tensor([1., 1., 1., 1., .5, .1])}
    assert allocate_shared_coefficients([errors]*4, 12) == [2]*6
    allocation = allocate_shared_coefficients([errors]*4, 16)
    assert allocation == [4, 4, 2, 2, 2, 2]
    assert sum(allocation) == 16


def test_refit_rejects_depth_dependent_pair_layout():
    dictionary = torch.eye(2)
    layout = torch.tensor([[0,0,0,1,1], [0,1,1,0,0]])
    q = JointAtomCoefficientRQ(dictionary, layout, torch.tensor([[0.,-1.,1.,-1.,1.]]*2))
    with pytest.raises(ValueError, match='shared'):
        refit_compact_book(q, torch.zeros(3,2), torch.zeros(3,2,dtype=torch.long))
