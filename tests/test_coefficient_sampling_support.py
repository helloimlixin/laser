import types

import pytest
import torch

from src.training.coefficient_support import sampling_coefficient_support
from src.training.imagenet_ffhq_adapter import ImageNetFFHQCompound
from tests.test_imagenet_ffhq_adaptation import tiny


def model_and_aux():
    config, aux = tiny()
    model = ImageNetFFHQCompound(config, 7, 5,
        micro_transformer_layers=2, depth_specific_coeff_heads=True).eval()
    with torch.no_grad():
        for classifier in model.coeff_classifier:
            classifier[-1].weight.zero_()
            classifier[-1].bias.copy_(torch.tensor([100., 0., -1., 0., 100.]))
    return model, aux


def test_restricted_tokens_are_used_by_later_predictions_and_training_is_unchanged():
    model, aux = model_and_aux()
    original = model.cached_head_output
    histories = []
    def tracked(self, packed, auxiliary, cond, sampling_idx, amp=True):
        histories.append(packed.clone())
        return original(packed, auxiliary, cond, sampling_idx, amp=amp)
    model.cached_head_output = types.MethodType(tracked, model)
    model.coefficient_sampling_limits = [.1, 1.51, .1, 1.51]
    atoms, ids = model.sample_compound(2, aux, cond=torch.tensor([5, 999]),
        atom_top_k=7, coeff_top_k=5, amp=False)
    assert bool((aux.coeff_bins[ids].abs() <= torch.tensor(model.coefficient_sampling_limits)).all())
    assert (ids[..., 0] == 2).all() and (ids[..., 2] == 2).all()
    assert torch.equal(histories[1][:, 0, 0, 0] % 5, ids[:, 0, 0, 0])
    assert all(not classifier._forward_hooks for classifier in model.coeff_classifier)
    assert torch.isfinite(model.coeff_classifier[0](torch.zeros(2, 12))).all()
    assert bool((atoms.sort(-1).values.diff(dim=-1) > 0).all())


def test_hooks_are_removed_when_sampling_fails():
    model, aux = model_and_aux()
    with pytest.raises(RuntimeError, match='probe failure'):
        with sampling_coefficient_support(model, aux.coeff_bins, [1.51] * 4):
            assert model.coeff_classifier[0]._forward_hooks
            raise RuntimeError('probe failure')
    assert all(not classifier._forward_hooks for classifier in model.coeff_classifier)


@pytest.mark.parametrize('limits', [[1.], [float('nan')] * 4, [-1.] * 4])
def test_invalid_support_is_rejected(limits):
    model, aux = model_and_aux()
    with pytest.raises(ValueError):
        with sampling_coefficient_support(model, aux.coeff_bins, limits):
            pass
