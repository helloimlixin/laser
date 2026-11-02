from dataclasses import replace

import pytest
import torch

from src.models.typed_pair_rq import TypedPairCodebook
from src.training.typed_pair_targets import (
    TypedPairTargets, best_joint_pairs, typed_pair_cross_entropy, typed_pair_targets,
)


@pytest.mark.parametrize('top_k', [1, 3, 8, 20, 100])
def test_nearest_complete_pairs_match_brute_cartesian_vocabulary(top_k):
    torch.manual_seed(31)
    dictionary = torch.randn(3, 5) * torch.tensor([.2, .5, 1., 2., 5.])
    bins = torch.tensor([-7., -.7, -.02, .4, 3., 11.])
    residual = torch.cat((torch.randn(13, 3), torch.tensor([[100., -100., 100.]])))
    a, b, scores = best_joint_pairs(residual, dictionary, bins, top_k)
    brute = -(residual[:, None, None] - dictionary.T[None, :, None]*bins[None, None, :, None]).square().sum(-1)
    expected = brute.flatten(1).topk(min(top_k, brute[0].numel()), -1)
    # The common residual norm was dropped from the efficient score.
    torch.testing.assert_close(scores-residual.square().sum(-1, keepdim=True), expected.values,
                               atol=.006, rtol=3e-5)
    assert torch.equal(a*len(bins)+b, expected.indices)


def test_pair_embedding_preserves_association_and_four_rq_positions():
    codebook = TypedPairCodebook(torch.eye(2), torch.tensor([-1., 0., .5, 2.]), depth=2)
    a, b = torch.tensor([[0, 1]]), torch.tensor([[2, 3]])
    tokens = codebook.pack(a, b)
    assert tokens.tolist() == [[0, 4, 1, 5]]
    assert codebook.compound_embeddings(a, b).shape == (1, 2, 2)
    torch.testing.assert_close(codebook.reconstruct(tokens), torch.tensor([[.5, 2.]]))
    swapped = codebook.pack(a, b.flip(-1))
    assert not torch.equal(codebook.reconstruct(tokens), codebook.reconstruct(swapped))
    for bad in (tokens.float(), torch.tensor([[0, 1, 1, 5]]), torch.tensor([[2, 4, 1, 5]])):
        with pytest.raises(ValueError):
            codebook.fields(bad)


def test_each_stochastic_pair_changes_the_next_residual_teacher_without_refit():
    dictionary = torch.tensor([[1., .8], [0., .6]])
    bins = torch.tensor([-.8, -.1, .4, .9, 1.4])
    codebook = TypedPairCodebook(dictionary, bins, depth=3)
    latent = torch.tensor([[1., .5], [-.7, .6], [.1, -.8]])
    args = dict(top_k=7, relative_temperature=.5, chunk_size=2)
    target = typed_pair_targets(codebook, latent, generator=torch.Generator().manual_seed(17), **args)
    replay = typed_pair_targets(codebook, latent, generator=torch.Generator().manual_seed(17), **args)
    assert torch.equal(target.tokens, replay.tokens)
    atoms, coefficients = codebook.fields(target.tokens)
    residual = latent.clone()
    for depth in range(3):
        a, b, scores = best_joint_pairs(residual, dictionary, bins, 7)
        assert torch.equal(a, target.candidate_atoms[:, depth])
        assert torch.equal(b, target.candidate_bins[:, depth])
        weights = (scores/(.5*residual.square().sum(-1, keepdim=True))).softmax(-1)
        torch.testing.assert_close(weights, target.pair_weights[:, depth])
        conditional = weights*(a == atoms[:, depth, None])
        conditional /= conditional.sum(-1, keepdim=True)
        torch.testing.assert_close(conditional, target.coefficient_weights_given_atom[:, depth])
        residual -= dictionary.T[atoms[:, depth]]*bins[coefficients[:, depth, None]]
        torch.testing.assert_close(target.residual_energies[:, depth], residual.square().sum(-1))
    torch.testing.assert_close(latent-codebook.reconstruct(target.tokens), residual)


def test_factorized_pair_ce_and_gradient_equal_cartesian_joint_ce_in_expectation():
    torch.manual_seed(32)
    atom_logits = torch.randn(1, 3, dtype=torch.float64, requires_grad=True)
    conditional_logits = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    q = torch.rand(3, 4, dtype=torch.float64); q /= q.sum()
    a, b = torch.meshgrid(torch.arange(3), torch.arange(4), indexing='ij')
    target = TypedPairTargets(torch.zeros(1, 2, dtype=torch.long), a.flatten()[None, None],
        b.flatten()[None, None], q.flatten()[None, None], torch.zeros(1, 1, 12), torch.zeros(1, 1))
    expected_loss = atom_logits.sum()*0
    for atom in range(3):
        conditional = q*(a == atom); conditional /= conditional.sum()
        current = replace(target, coefficient_weights_given_atom=conditional.flatten()[None, None])
        loss, _, _ = typed_pair_cross_entropy(dict(atom_logits=atom_logits[:, None],
            coeff_logits=conditional_logits[atom][None, None]), current)
        expected_loss = expected_loss + q[atom].sum()*loss
    joint_log_probability = atom_logits.log_softmax(-1).T+conditional_logits.log_softmax(-1)
    brute = -(q*joint_log_probability).sum()
    torch.testing.assert_close(expected_loss, brute, atol=1e-12, rtol=1e-12)
    for actual, reference in zip(torch.autograd.grad(expected_loss, (atom_logits, conditional_logits), retain_graph=True),
                                 torch.autograd.grad(brute, (atom_logits, conditional_logits))):
        torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)


def test_hard_teacher_loss_is_one_joint_token_nll_and_zero_mass_is_safe():
    codebook = TypedPairCodebook(torch.eye(2), torch.tensor([-.5, 0., 1.]), depth=2)
    targets = typed_pair_targets(codebook, torch.tensor([[2., -.5]]), top_k=6, stochastic=False)
    a, b = codebook.fields(targets.tokens)
    atom_logits = torch.randn(1, 2, 2, requires_grad=True)
    coeff_logits = torch.randn(1, 2, 3, requires_grad=True)
    # One impossible zero-mass pair candidate must not create NaN.
    assert (a == 0).all() and (b == 2).all()
    with torch.no_grad():
        atom_logits[..., 1] = -torch.inf
        coeff_logits[..., 0] = -torch.inf
    loss, _, _ = typed_pair_cross_entropy(dict(atom_logits=atom_logits, coeff_logits=coeff_logits), targets)
    brute = (-atom_logits.log_softmax(-1).gather(-1, a[..., None])
             -coeff_logits.log_softmax(-1).gather(-1, b[..., None])).mean()
    torch.testing.assert_close(loss, brute)
    loss.backward()
    assert torch.isfinite(atom_logits.grad).all() and torch.isfinite(coeff_logits.grad).all()
