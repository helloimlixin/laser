import random

import numpy as np
import pytest
import torch

from src.training.rq_rng import POLICY, audit_training_rng, configure_training_rng


def test_isolated_rng_restores_all_streams_and_replays():
    from src.training.rq_rng import isolated_rng
    configure_training_rng(981, 0)
    expected = draw()
    configure_training_rng(981, 0)
    with isolated_rng(123, 'cpu'):
        first = draw()
    with isolated_rng(123, 'cpu'):
        repeated = draw()
    actual = draw()
    assert first[:2] == repeated[:2]
    assert torch.equal(first[2], repeated[2])
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])


def draw():
    return (random.random(), float(np.random.random()), torch.rand(8))


def test_rank_streams_are_distinct_repeatable_and_leave_model_weights_unchanged():
    torch.manual_seed(261001)
    model = torch.nn.Linear(16, 16)
    initial = {k:v.clone() for k,v in model.state_dict().items()}
    fingerprints = []
    for rank in range(8):
        metadata = configure_training_rng(261001, rank)
        first = draw()
        proof = audit_training_rng('cpu')
        configure_training_rng(261001, rank)
        repeated = draw()
        assert first[:2] == repeated[:2]
        torch.testing.assert_close(first[2], repeated[2], rtol=0, atol=0)
        assert metadata['training_rng_policy'] == POLICY
        assert metadata['training_rng_seed'] == 261001 + rank
        assert proof['training_rng_unchanged_by_audit']
        fingerprints.append(proof['dropout_probe_sha256'])
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, initial[key], rtol=0, atol=0)
    assert len(set(fingerprints)) == 8


def test_resume_does_not_reseed_or_advance_saved_random_streams():
    configure_training_rng(261001, 5)
    draw()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    metadata = configure_training_rng(261001, 5, resuming=True)
    actual = draw()
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    expected = draw()
    assert metadata['training_rng_restored_from_checkpoint']
    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)


def test_dropout_audit_preserves_random_state_and_is_repeatable():
    configure_training_rng(261001, 2)
    before = torch.get_rng_state().clone()
    first = audit_training_rng('cpu')
    second = audit_training_rng('cpu')
    assert torch.equal(before, torch.get_rng_state())
    assert first == second


@pytest.mark.parametrize('base_seed,rank', [(-1, 0), (1, -1), (2**63-1, 1)])
def test_invalid_training_seeds_are_rejected(base_seed, rank):
    with pytest.raises(ValueError, match='nonnegative'):
        configure_training_rng(base_seed, rank)
