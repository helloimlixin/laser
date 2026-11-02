"""Extend a trained compound GPT without resetting its historical Adam state."""
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict, replace

import torch

from src.models.compound_gpt import CompoundGPT, CompoundGPTConfig, CausalCoefficientConditioner


def upgrade_checkpoint(snapshot, *, layers=2, width=512, heads=8, seed=261024):
    """Return an upgraded CPU checkpoint and a migration receipt.

    Existing tensors are immutable references while the new checkpoint is
    serialized; no old weight, moment, counter or RNG tensor is changed.
    """
    old_config = CompoundGPTConfig(**snapshot['config']['backbone_config'])
    if old_config.coefficient_conditioner_layers:
        raise ValueError('checkpoint already has a coefficient conditioner')
    new_config = replace(old_config, attention_dropout=0., embedding_dropout=0.,
        ffn_intermediate_dropout=0., coefficient_conditioner_layers=layers,
        coefficient_conditioner_width=width, coefficient_conditioner_heads=heads)
    with torch.device('meta'):
        old_model, new_model = CompoundGPT(old_config), CompoundGPT(new_config)
    old_names = [name for name, _ in old_model.named_parameters()]
    new_names = [name for name, _ in new_model.named_parameters()]
    if old_names != new_names[:len(old_names)]:
        raise ValueError('historical parameter order changed')
    old_optimizer = snapshot['optimizer']
    if len(old_optimizer['param_groups']) != 1:
        raise ValueError('migration requires the historical single Adam parameter group')
    old_ids = old_optimizer['param_groups'][0]['params']
    if len(old_ids) != len(old_names) or not all(isinstance(i, int) for i in old_ids):
        raise ValueError('historical optimizer parameter layout mismatch')
    if set(old_ids) != set(old_optimizer['state']):
        raise ValueError('historical optimizer state is incomplete')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        conditioner = CausalCoefficientConditioner(new_config.width, width, heads, layers,
            new_config.block_size[-1], new_config.dropout, 0., 0.)
        conditioner.set_output_bins(new_config.coeff_vocab_size)
    added = OrderedDict(('coefficient_conditioner.' + key, value)
                        for key, value in conditioner.state_dict().items())
    state = OrderedDict(snapshot['state_dict'])
    state.update(added)
    new_model.load_state_dict(state, strict=True, assign=True)
    # Adam ids follow optimizer parameter order, which remains an exact prefix.
    new_ids = list(range(max(old_ids) + 1, max(old_ids) + 1 + len(new_names) - len(old_names)))
    groups = [dict(old_optimizer['param_groups'][0], params=[*old_ids, *new_ids])]
    optimizer_state = {key: dict(value) for key, value in old_optimizer['state'].items()}
    template_step = next(iter(optimizer_state.values()))['step']
    for identifier, name in zip(new_ids, new_names[len(old_names):]):
        parameter = state[name]
        entry = dict(step=torch.zeros_like(template_step), exp_avg=torch.zeros_like(parameter),
                     exp_avg_sq=torch.zeros_like(parameter))
        if groups[0].get('amsgrad'):
            entry['max_exp_avg_sq'] = torch.zeros_like(parameter)
        optimizer_state[identifier] = entry
    config = deepcopy(snapshot['config'])
    config.update(backbone_config=asdict(new_config), gpt_attention_dropout=0.,
        gpt_embedding_dropout=0., gpt_ffn_intermediate_dropout=0.,
        gpt_coefficient_conditioner_layers=layers, gpt_coefficient_conditioner_width=width,
        gpt_coefficient_conditioner_heads=heads, conditioner_added_at_step=int(snapshot['global_step']))
    updated = dict(snapshot, state_dict=state,
        optimizer=dict(old_optimizer, state=optimizer_state, param_groups=groups), config=config)
    receipt = dict(source_global_step=int(snapshot['global_step']),
        old_parameters=sum(p.numel() for p in old_model.parameters()),
        parameters=sum(p.numel() for p in new_model.parameters()),
        added_parameters=sum(p.numel() for p in conditioner.parameters()),
        existing_parameter_tensors=len(old_names), added_parameter_tensors=len(new_ids),
        existing_optimizer_states_preserved=len(old_ids), new_optimizer_age=0,
        conditioner_layers=layers, conditioner_width=width, conditioner_heads=heads,
        new_parameter_seed=seed, zero_initial_output=True,
        attention_dropout=0., embedding_dropout=0., ffn_intermediate_dropout=0.,
        residual_dropout=new_config.dropout)
    return updated, receipt
