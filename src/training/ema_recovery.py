"""Describe averaged inference weights without changing the Adam training model."""
import math


def ema_recovery_metadata(payload):
    state = payload.get('parameter_ema')
    if state is None:
        return None
    decay, updates = float(state['decay']), int(state['updates'])
    origin = int(state['origin_global_step'])
    if not (math.isfinite(decay) and 0 <= decay < 1 and updates >= 0
            and origin >= 0 and origin + updates == int(payload['global_step'])):
        raise ValueError('EMA policy or update clock disagrees with the training state')
    values = state['values']
    raw = payload['state_dict']
    if 'optimizer' in payload and len(values) != sum(
            len(g['params']) for g in payload['optimizer']['param_groups']):
        raise ValueError('EMA checkpoint is missing training parameters')
    if not values or any(k not in raw or v.shape != raw[k].shape
                         or v.dtype != raw[k].dtype for k, v in values.items()):
        raise ValueError('EMA parameters do not match the raw training model')
    metrics = payload.get('ema_original_rqtransformer_metrics')
    if metrics is not None and (
            int(metrics['global_step']) != int(payload['global_step'])
            or metrics['metric_backend'] != 'original_rqtransformer'
            or metrics['weight_state'] != 'ema'):
        raise ValueError('EMA evaluation does not belong to the saved EMA weights')
    return dict(decay=decay, updates=updates, origin_global_step=origin,
                parameter_tensors=len(values), parameter_elements=sum(v.numel() for v in values.values()),
                training_weights='state_dict', optimizer_weights='state_dict',
                inference_weights='parameter_ema.values',
                resume_policy='Restore raw model, trained Adam, and EMA separately; continue updates on the raw model.',
                original_rqtransformer_metrics=metrics)
