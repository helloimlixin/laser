import torch

from src.models.laser_joint_tokenizer import JointAtomCoefficientRQ
from src.training.laser_compact_joint_vocabulary import reconstruct_fixed_codes, refit_compact_book


def test_dictionary_and_scalar_updates_reduce_exact_final_reconstruction_error():
    torch.manual_seed(752)
    dictionary = torch.nn.functional.normalize(torch.randn(7, 3), dim=0)
    atoms = torch.cat((torch.zeros(1, dtype=torch.long), torch.arange(3).repeat_interleave(4))).expand(4, -1)
    values = torch.cat((torch.zeros(4, 1), torch.tensor([-2., -.7, .7, 2.]).repeat(4, 3)), -1)
    q = JointAtomCoefficientRQ(dictionary, atoms, values)
    # Includes repeated atoms across depths, which require the cross terms in
    # the dictionary normal equations for a correct final-error update.
    codes = torch.randint(1, 13, (80, 4))
    truth = dictionary+torch.randn_like(dictionary)*.05
    x = sum(truth.T[atoms[d, codes[:, d]]]*values[d, codes[:, d], None] for d in range(4))
    result = refit_compact_book(q, x, codes)
    actual = float((x-reconstruct_fixed_codes(q, codes)).square().mean())
    assert result['fixed_assignment_mse_after'] < result['fixed_assignment_mse_before']
    torch.testing.assert_close(torch.tensor(actual), torch.tensor(result['fixed_assignment_mse_after']))
    torch.testing.assert_close(q.dictionary.norm(dim=0), torch.ones(3))
    levels = q.coefficients[:, 1:].view(4, 3, 4)
    assert (levels[..., :2] < 0).all() and (levels[..., 2:] > 0).all()
    assert (levels[..., 1:] >= levels[..., :-1]).all()
    assert torch.equal(q.atom_ids, atoms)
    assert (q.coefficients[:, 0] == 0).all()
    result = q.quantize(x)
    torch.testing.assert_close(result['quantized'], q.embed(result['codes']).sum(-2), rtol=1.e-5, atol=1.e-6)


def test_unused_atoms_and_zero_signal_do_not_collapse():
    torch.manual_seed(753)
    dictionary = torch.nn.functional.normalize(torch.randn(7, 3), dim=0)
    atoms = torch.cat((torch.zeros(1, dtype=torch.long), torch.arange(3).repeat_interleave(4))).expand(4, -1)
    values = torch.cat((torch.zeros(4, 1), torch.tensor([-2., -.7, .7, 2.]).repeat(4, 3)), -1)
    q = JointAtomCoefficientRQ(dictionary, atoms, values)
    refit_compact_book(q, torch.zeros(20, 7), torch.zeros(20, 4, dtype=torch.long))
    torch.testing.assert_close(q.dictionary, dictionary)
    torch.testing.assert_close(q.coefficients, values)
    assert (q.quantize(torch.zeros(20, 7))['codes'] == 0).all()
    codes = torch.randint(1, 9, (40, 4))
    signals = reconstruct_fixed_codes(q, codes)+torch.randn(40, 7)*.02
    old_direction = q.dictionary[:, 2].clone()
    old_levels = q.coefficients[:, 9:].clone()
    refit_compact_book(q, signals, codes)
    torch.testing.assert_close(q.dictionary[:, 2], old_direction)
    torch.testing.assert_close(q.coefficients[:, 9:], old_levels)
