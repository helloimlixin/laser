"""Attach an initially neutral coefficient cross-attention experiment."""
import copy
import math

from src.models.coefficient_cross_attention import CoefficientPairCrossAttention
from src.training.compound_history import HistoryConditionedCompoundTransformer
from src.training.rqtransformer import CompoundLaserRQTransformer


def attach_coefficient_cross_attention(model, *, width=512, layers=2, heads=8):
    if type(model) is not CompoundLaserRQTransformer or not model.pair_autoregressive:
        raise ValueError('coefficient cross-attention requires the plain full-pair compound model')
    if model.causal_prefix_state or model.contribution_head is not None:
        raise ValueError('coefficient cross-attention requires the standard compound objective')
    parameter = next(model.parameters())
    model.__class__ = HistoryConditionedCompoundTransformer
    model.coefficient_history_decoder = CoefficientPairCrossAttention(
        model.config.embed_dim, model.config.input_embed_dim, width=width,
        layers=layers, heads=heads, max_events=math.prod(model.block_size),
    ).to(device=parameter.device, dtype=parameter.dtype)
    model.coefficient_history_decoder.train(model.training)
    model._coefficient_sample_context = None
    return model


def extend_optimizer_for_appended_parameters(state, old_names, new_names):
    """Retain Adam moments/ages and leave appended parameters uninitialized."""
    old_names, new_names = list(old_names), list(new_names)
    if new_names[:len(old_names)] != old_names:
        raise ValueError('existing optimizer parameter order must remain unchanged')
    groups = state['param_groups']
    if len(groups) != 1 or groups[0]['params'] != list(range(len(old_names))):
        raise ValueError('expected one optimizer group in original parameter order')
    if set(state['state']) - set(groups[0]['params']):
        raise ValueError('optimizer state contains unknown parameter IDs')
    updated = dict(state, state=dict(state['state']), param_groups=copy.deepcopy(groups))
    updated['param_groups'][0]['params'] = list(range(len(new_names)))
    return updated
