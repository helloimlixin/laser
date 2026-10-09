import pytest
import torch

from src.training.ema_recovery import ema_recovery_metadata


def payload():
    return dict(global_step=102, state_dict={'weight': torch.ones(3)},
        parameter_ema=dict(decay=.999, updates=2, origin_global_step=100,
                           values={'weight': torch.ones(3) * .5}),
        ema_original_rqtransformer_metrics=dict(global_step=102, fid=15.,
            metric_backend='original_rqtransformer', weight_state='ema'))


def test_metadata_preserves_raw_optimizer_and_distinguishes_scored_ema():
    raw = payload()
    result = ema_recovery_metadata(raw)
    assert result['training_weights'] == result['optimizer_weights'] == 'state_dict'
    assert result['inference_weights'] == 'parameter_ema.values'
    assert result['updates'] == 2 and result['parameter_elements'] == 3
    assert torch.equal(raw['state_dict']['weight'], torch.ones(3))


@pytest.mark.parametrize('field,value', [('updates', 1), ('updates', -1),
    ('decay', float('nan')), ('origin_global_step', -1)])
def test_wrong_ema_clock_or_policy_is_rejected(field, value):
    raw = payload();raw['parameter_ema'][field] = value
    with pytest.raises(ValueError):ema_recovery_metadata(raw)


def test_wrong_scored_weights_or_shape_is_rejected():
    for mutate in (lambda p: p['ema_original_rqtransformer_metrics'].update(global_step=101),
                   lambda p: p['ema_original_rqtransformer_metrics'].update(weight_state='raw'),
                   lambda p: p['parameter_ema']['values'].update(weight=torch.ones(4))):
        raw = payload();mutate(raw)
        with pytest.raises(ValueError):ema_recovery_metadata(raw)


def test_ema_must_cover_every_adam_parameter():
    raw = payload();raw['optimizer'] = dict(param_groups=[dict(params=[0, 1])])
    with pytest.raises(ValueError, match='missing training parameters'):
        ema_recovery_metadata(raw)
