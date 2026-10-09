"""Full-state migration with separate Adam ages for added history parameters."""
import copy

import torch

ARCHITECTURE = 'gated_full_history_physical_compound_v2'


def migrate_checkpoint(payload, model, original_parameter_names, *, accumulation=4):
    if payload.get('compound_transfer') is not None:
        raise ValueError('Checkpoint already has compound history')
    names = list(dict(model.named_parameters()))
    new_names = [name for name in names if name not in set(original_parameter_names)]
    old_optimizer = payload['optimizer']
    if len(old_optimizer['param_groups']) != 1:
        raise ValueError('Expected the source single AdamW parameter group')
    old_ids = old_optimizer['param_groups'][0]['params']
    if len(old_ids) != len(original_parameter_names) or set(old_ids) != set(old_optimizer['state']):
        raise ValueError('Source Adam state is incomplete')
    ages = {int(state['step']) for state in old_optimizer['state'].values()}
    if len(ages) != 1:
        raise ValueError('Source Adam ages disagree')
    old_age = ages.pop()
    old_by_name = dict(zip(original_parameter_names, old_ids))
    state = {}
    for index, (name, parameter) in enumerate(model.named_parameters()):
        if name in old_by_name:
            value = old_optimizer['state'][old_by_name[name]]
            if value['exp_avg'].shape != parameter.shape or value['exp_avg_sq'].shape != parameter.shape:
                raise ValueError(f'Source Adam parameter mapping mismatch: {name}')
            state[index] = value
        else:
            # Fresh parameters must not inherit the trained Adam bias counter.
            state[index] = dict(step=torch.tensor(0.), exp_avg=torch.zeros_like(parameter),
                                exp_avg_sq=torch.zeros_like(parameter))
    source_accumulation = int(payload['config']['accumulation_steps'])
    cursor = int(payload.get('batch_idx', 0))
    if cursor % source_accumulation:
        raise ValueError('Migration requires an optimizer boundary')
    config = dict(payload['config'], architecture=ARCHITECTURE,
        compound_event_history=True, compound_event_order='raster_then_depth_atom_coefficient_pair',
        accumulation_steps=accumulation,
        batch_size=(payload['config']['total_batch_size']+8*accumulation-1)//(8*accumulation))
    transfer = dict(version=ARCHITECTURE, source_global_step=int(payload['global_step']),
        source_common_adam_step=old_age, parameter_names=names, new_parameter_names=new_names,
        source_metrics=copy.deepcopy(payload.get('original_rqtransformer_metrics')),
        all_common_weights_and_moments_preserved=True, new_parameters_start_adam_at_zero=True)
    result = dict(payload, state_dict=model.state_dict(), optimizer=dict(old_optimizer,
        state=state, param_groups=[dict(old_optimizer['param_groups'][0], params=list(range(len(names))))]),
        config=config, compound_transfer=transfer,
        batch_idx=cursor//source_accumulation*accumulation,
        best_fid=[], best_inception=[], original_rqtransformer_metrics=None,
        fid=None, inception_score=None, inception_score_std=None)
    for key in ('train_loss_tracking', 'train_epoch_loss_tracking'):
        result.pop(key, None)
    validate_optimizer_ages(result)
    return result


def validate_optimizer_ages(payload):
    transfer = payload['compound_transfer']
    if transfer['version'] != ARCHITECTURE:
        raise ValueError('Unknown compound-history checkpoint architecture')
    names = transfer['parameter_names']
    new = set(transfer['new_parameter_names'])
    if len(set(names)) != len(names) or not new or not new < set(names):
        raise ValueError('Invalid compound-history parameter inventory')
    updates = int(payload['global_step'])-int(transfer['source_global_step'])
    if updates < 0:
        raise ValueError('Compound-history optimizer cursor precedes transfer')
    old_age = int(transfer['source_common_adam_step'])+updates
    optimizer = payload['optimizer']
    ids = [index for group in optimizer['param_groups'] for index in group['params']]
    if ids != list(range(len(names))) or set(ids) != set(optimizer['state']):
        raise ValueError('Compound-history Adam parameter order/state is incomplete')
    for index, name in enumerate(names):
        values = optimizer['state'][index]
        expected = updates if name in new else old_age
        if int(values['step']) != expected:
            raise ValueError(f'Incorrect Adam age for {name}: expected {expected}')
        if values['exp_avg'].shape != payload['state_dict'][name].shape or values['exp_avg_sq'].shape != payload['state_dict'][name].shape:
            raise ValueError(f'Incorrect Adam moment shape for {name}')
    return dict(common_adam_step=old_age, new_adam_step=updates,
                common_parameter_tensors=len(names)-len(new), new_parameter_tensors=len(new))


def verify_live_optimizer(optimizer, model, transfer, global_step):
    names = list(dict(model.named_parameters()))
    if names != transfer['parameter_names']:
        raise ValueError('Live compound-history parameter order changed')
    updates = global_step-transfer['source_global_step']
    new = set(transfer['new_parameter_names'])
    common_age = transfer['source_common_adam_step']+updates
    for name, parameter in model.named_parameters():
        expected = updates if name in new else common_age
        state = optimizer.state[parameter]
        if not {'step', 'exp_avg', 'exp_avg_sq'} <= set(state) or int(state['step']) != expected:
            raise ValueError(f'Live compound-history Adam age/state mismatch: {name}')
    return common_age
