import torch
from torch import nn

from src.training.heldout_monitor import evaluate_heldout


class Aux:
    coeff_vocab_size = 2

    def compound_coeff_ids(self, coefficients, **kwargs):
        # Deliberately consume RNG to ensure monitoring isolates even a noisy helper.
        torch.rand(3)
        return coefficients.long(), torch.nn.functional.one_hot(coefficients.long(), 2).float()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(2))
        self.dropout = nn.Dropout(.5)

    def forward(self, tokens, **kwargs):
        assert not self.training
        logits = self.bias.expand(*tokens.shape, 2)
        return {'atom_logits': logits, 'coeff_logits': logits}


def test_fixed_monitor_preserves_rng_modes_parameters_and_reports_per_image_losses():
    model = Model().train()
    model.dropout.eval()  # Preserve heterogeneous mode flags, too.
    data = {'atoms': torch.zeros(5, 2, 2, 4, dtype=torch.long),
            'coefficients': torch.zeros(5, 2, 2, 4),
            'input_coefficient_ids': torch.zeros(5, 2, 2, 4, dtype=torch.long)}
    rng = torch.get_rng_state().clone()
    before = model.bias.detach().clone()
    result = evaluate_heldout(model, Aux(), {'train_fresh': data, 'validation_fresh': data}, batch_size=2)
    assert torch.equal(torch.get_rng_state(), rng)
    assert model.training and not model.dropout.training
    assert torch.equal(model.bias, before) and model.bias.grad is None
    assert result['validation_fresh']['images'] == 5
    assert abs(result['validation_fresh']['atom_nll']['mean'] - torch.log(torch.tensor(2.)).item()) < 1e-7
    assert result['validation_fresh']['atom_nll']['se'] < 1e-7
    assert result['validation_minus_train']['atom_nll']['mean'] == 0


def test_monitor_restores_training_state_after_failure():
    model = Model().train()
    data = {'atoms': torch.zeros(1, 2, 2, 4), 'coefficients': torch.zeros(1),
            'input_coefficient_ids': torch.zeros(1)}
    try:
        evaluate_heldout(model, Aux(), {'invalid': data})
    except ValueError:
        pass
    else:
        raise AssertionError('invalid split accepted')
    assert model.training
