import pytest
import torch

from src.compound_joint_sampling import (
    CompoundJointPolicy, atom_proposals, candidate_coefficient_logits,
    joint_probabilities, sample_compound_joint,
)
from tests.test_compound_ancestral_sampling import setup, assert_caches_clear


def test_enumeration_recovers_exact_joint_and_atom_marginal():
    weights = torch.tensor([[.8, .2]])
    logits = torch.tensor([[[1., -2., 0.], [-1., 2., 0.]]])
    actual = joint_probabilities(weights, logits)
    expected = weights[..., None] * logits.softmax(-1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.sum(-1), weights)


def test_prior_draw_duplicates_are_combined_without_squaring_prior(monkeypatch):
    monkeypatch.setattr(torch, 'multinomial', lambda *a, **kw: torch.tensor([[0, 0, 0, 1]]))
    atoms, weights = atom_proposals(torch.tensor([[.8, .2]]), 4)
    assert torch.equal(atoms, torch.tensor([[0, 0, 0, 1]]))
    torch.testing.assert_close(weights, torch.tensor([[.75, 0., 0., .25]]))
    joint = joint_probabilities(weights, torch.zeros(1, 4, 3))
    torch.testing.assert_close(joint.sum(-1), weights)


def test_candidate_head_matches_independent_atom_branches_without_cache_mutation():
    model, aux = setup()
    packed = torch.zeros(2, 2, 2, 3, dtype=torch.long)
    model.init_cache()
    hidden = model.cached_head_output(packed, aux, None, (0, 0, 0), amp=False)
    cache = model._cache['spatial_ctx_hw']
    head_cache = [block._cache['past_kv'] for block in model.head_transformer.blocks]
    atoms = torch.tensor([[0, 2, 4], [1, 3, 5]])
    actual = candidate_coefficient_logits(model, aux, hidden, atoms, depth=0)
    expected = torch.stack([model.coefficient_logits(hidden, aux.dictionary.T[atoms[:, k]],
                                                    depth_index=0) for k in range(3)], dim=1)
    torch.testing.assert_close(actual, expected)
    assert model._cache['spatial_ctx_hw'] is cache
    assert all(block._cache['past_kv'] is old for block, old in zip(model.head_transformer.blocks, head_cache))


@pytest.mark.parametrize('proposal', ['prior', 'top'])
def test_actual_model_joint_sampling_preserves_support_caches_and_weights(proposal):
    model, aux = setup()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    policy = CompoundJointPolicy(candidate_atoms=7, atom_proposal=proposal, atom_top_k=0,
                                atom_top_p=None, coefficient_temperatures=(1., 1., 1.), joint_top_p=.9)
    first = sample_compound_joint(model, 4, aux, amp=False, policy=policy,
                                  generator=torch.Generator().manual_seed(81))
    second = sample_compound_joint(model, 4, aux, amp=False, policy=policy,
                                   generator=torch.Generator().manual_seed(81))
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    atoms, coefficients = first
    for depth in range(1, 3): assert not (atoms[..., depth, None] == atoms[..., :depth]).any()
    assert coefficients.min() >= 0 and coefficients.max() < 5
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    assert_caches_clear(model)


def test_pair_nucleus_keeps_high_joint_mass_across_atom_boundaries():
    weights = torch.tensor([[.55, .45]])
    conditional = torch.tensor([[[.5, .5], [.99, .01]]])
    probabilities = joint_probabilities(weights, conditional.log(), top_p=.4)
    assert probabilities[0, 1, 0] == 1
    assert probabilities.count_nonzero() == 1


def test_joint_mass_one_retains_the_full_conditional_tail():
    weights = torch.ones(1, 1)
    logits = torch.tensor([[[0., -25., -30.]]])
    actual = joint_probabilities(weights, logits, top_p=1.)
    torch.testing.assert_close(actual, joint_probabilities(weights, logits, top_p=None), rtol=0, atol=0)
    assert (actual > 0).all()


def test_sampler_clears_caches_after_conditional_head_failure(monkeypatch):
    model, aux = setup()
    def fail(*args, **kwargs): raise RuntimeError('coefficient failure')
    monkeypatch.setattr(model, 'coefficient_logits', fail)
    with pytest.raises(RuntimeError, match='coefficient failure'):
        sample_compound_joint(model, 2, aux, amp=False,
            policy=CompoundJointPolicy(coefficient_temperatures=(1., 1., 1.)))
    assert_caches_clear(model)
