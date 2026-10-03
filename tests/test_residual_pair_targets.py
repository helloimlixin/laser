import pytest
import torch

from src.training.residual_pair_targets import (
    ResidualPairTargets, residual_pair_cross_entropy, residual_pair_targets,
)


def fixture():
    g = torch.Generator().manual_seed(19)
    return (torch.randn(7, 3, generator=g, dtype=torch.float64),
            torch.randn(3, 5, generator=g, dtype=torch.float64),
            torch.tensor([[-1.2, -.3, .4, 1.1], [-.9, -.1, .2, .8],
                          [-.7, -.2, .3, .9]], dtype=torch.float64))


def test_pair_teacher_matches_explicit_vector_distances_on_actual_sampled_prefix():
    x, dictionary, values = fixture()
    out = residual_pair_targets(x, dictionary, values, temperature=.5,
                               site_chunk_size=3, atom_chunk_size=2,
                               generator=torch.Generator().manual_seed(31))
    r = x.clone()
    for d, grid in enumerate(values):
        book = dictionary.T[:, None, :] * grid[None, :, None]
        distances = (r[:, None, None, :] - book).square().sum(-1)
        joint = (-distances.flatten(1) / .5).softmax(-1).reshape(7, 5, 4)
        marginal = joint.sum(-1)
        torch.testing.assert_close(out.atom_probabilities[:, d], marginal)
        rows = torch.arange(len(x))
        conditional = joint[rows, out.atoms[:, d]] / marginal[rows, out.atoms[:, d], None]
        torch.testing.assert_close(out.coefficient_probabilities[:, d], conditional)
        r -= book[out.atoms[:, d], out.coefficient_ids[:, d]]
    torch.testing.assert_close(out.reconstruction, x - r)
    assert not out.reconstruction.requires_grad


def test_replay_and_fresh_visits():
    x, dictionary, values = fixture()
    g = torch.Generator().manual_seed(53)
    state = g.get_state()
    first = residual_pair_targets(x, dictionary, values, generator=g)
    second = residual_pair_targets(x, dictionary, values, generator=g)
    g.set_state(state)
    replay = residual_pair_targets(x, dictionary, values, generator=g)
    assert torch.equal(first.atoms, replay.atoms)
    assert torch.equal(first.coefficient_ids, replay.coefficient_ids)
    assert not (torch.equal(first.atoms, second.atoms)
                and torch.equal(first.coefficient_ids, second.coefficient_ids))
    torch.testing.assert_close(first.atom_probabilities[:, 0], second.atom_probabilities[:, 0])
    assert not torch.equal(first.atom_probabilities[:, 1:], second.atom_probabilities[:, 1:])


def test_depth_extension_keeps_existing_pairs_and_allows_repeated_atoms():
    x, dictionary, values = fixture()
    shallow = residual_pair_targets(x, dictionary, values[:2],
                                    generator=torch.Generator().manual_seed(9))
    deep = residual_pair_targets(x, dictionary, values,
                                 generator=torch.Generator().manual_seed(9))
    assert torch.equal(shallow.atoms, deep.atoms[:, :2])
    assert torch.equal(shallow.coefficient_ids, deep.coefficient_ids[:, :2])
    repeated = residual_pair_targets(x, dictionary[:, :1], values,
                                     generator=torch.Generator().manual_seed(9))
    assert torch.equal(repeated.atoms, torch.zeros_like(repeated.atoms))


def test_expected_factorized_loss_and_gradients_equal_full_joint_cross_entropy():
    qa = torch.tensor([.2, .3, .5], dtype=torch.float64)
    qc = torch.tensor([[.1, .9], [.6, .4], [.7, .3]], dtype=torch.float64)
    atom_logits = torch.tensor([.1, -.4, .7], dtype=torch.float64, requires_grad=True)
    coeff_logits = torch.tensor([[.2, -.5], [-.4, .1], [.6, .8]],
                                dtype=torch.float64, requires_grad=True)
    losses = []
    for a in range(3):
        targets = ResidualPairTargets(torch.tensor(a), torch.tensor(0), qa, qc[a], torch.empty(0))
        losses.append(residual_pair_cross_entropy(atom_logits, coeff_logits[a], targets))
    expected = (qa * torch.stack(losses)).sum()
    log_joint = atom_logits.log_softmax(-1)[:, None] + coeff_logits.log_softmax(-1)
    direct = -(qa[:, None] * qc * log_joint).sum()
    torch.testing.assert_close(expected, direct)
    got = torch.autograd.grad(expected, (atom_logits, coeff_logits), retain_graph=True)
    want = torch.autograd.grad(direct, (atom_logits, coeff_logits))
    for actual, reference in zip(got, want):
        torch.testing.assert_close(actual, reference)


def test_fp32_spatial_shape_and_chunked_marginal():
    x, dictionary, values = fixture()
    x = x[:6].float().reshape(2, 3, 3)
    out = residual_pair_targets(x, dictionary.float(), values.float(), atom_chunk_size=2)
    assert out.atoms.shape == (2, 3, 3)
    assert out.atom_probabilities.shape == (2, 3, 3, 5)
    torch.testing.assert_close(out.atom_probabilities.sum(-1), torch.ones(2, 3, 3))
    torch.testing.assert_close(out.coefficient_probabilities.sum(-1), torch.ones(2, 3, 3))
    assert torch.isfinite(out.reconstruction).all()


@pytest.mark.parametrize('temperature', [0., -1., float('nan'), float('inf')])
def test_bad_temperature_rejected(temperature):
    with pytest.raises(ValueError):
        residual_pair_targets(*fixture(), temperature=temperature)
