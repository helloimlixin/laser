import torch

from src.training.rqtransformer import LaserAux, build_parser


def aux():
    instance = object.__new__(LaserAux)
    torch.nn.Module.__init__(instance)
    instance.num_atoms = 11
    instance.coeff_vocab_size = 9
    instance.vocab_size = 20
    instance.sparsity_level = 4
    instance.coeff_bins = torch.linspace(-3, 3, 9)
    instance.coeff_scales = torch.tensor([4., 2., 1.5, 1.])
    instance.soft_target_physical = False
    return instance


def test_soft_loss_retains_exact_hard_prefix_tokens_and_rng():
    teacher = aux()
    atoms = torch.arange(4).expand(2, 1, 1, 4)
    # Bin centers, half-bin ties, clipped endpoints and general values.
    values = torch.tensor([-3., -.375, .19, 4., -4., 0., .375, 3.]).reshape(2, 1, 1, 4)
    before = torch.get_rng_state().clone()
    hard, (_, hard_probs) = teacher.sparse_targets(
        atoms, values, stochastic=False, hard=True, compact=True)
    soft, (_, soft_probs) = teacher.sparse_targets(
        atoms, values, temp=.5, stochastic=False, hard=False, compact=True)
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(hard, soft)
    assert torch.equal(soft[..., 0::2], atoms)
    assert soft[..., 1::2].min() >= teacher.num_atoms
    assert soft[..., 1::2].max() < teacher.vocab_size
    assert not torch.equal(hard_probs, soft_probs)
    torch.testing.assert_close(soft_probs.sum(-1), torch.ones_like(values))
    assert (soft_probs.max(-1).values < 1).all()


def test_stochastic_soft_inputs_remain_available_and_change_rng():
    teacher = aux()
    atoms = torch.arange(4).expand(32, 1, 1, 4)
    values = torch.zeros_like(atoms, dtype=torch.float)
    before = torch.get_rng_state().clone()
    sampled, _ = teacher.sparse_targets(
        atoms, values, stochastic=True, hard=False, compact=True)
    assert not torch.equal(before, torch.get_rng_state())
    nearest, _ = teacher.sparse_targets(
        atoms, values, stochastic=False, hard=False, compact=True)
    assert not torch.equal(sampled, nearest)
    assert torch.equal(sampled[..., 0::2], atoms)


def test_input_sampling_cli_is_independent_and_backward_compatible():
    parser = build_parser()
    required = ['--checkpoint', 's1.pt', '--data', 'data', '--output', 'out']
    assert parser.parse_args(required).stochastic_coefficient_inputs is None
    args = parser.parse_args(required + ['--coeff-target-mode', 'soft',
                                         '--no-stochastic-coefficient-inputs'])
    assert args.coeff_target_mode == 'soft'
    assert args.stochastic_coefficient_inputs is False
    assert parser.parse_args(required + ['--stochastic-coefficient-inputs']).stochastic_coefficient_inputs is True
