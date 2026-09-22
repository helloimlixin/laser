"""Explicit batch/topology migration without changing the training objective."""
from copy import deepcopy
import math


def prepare_resume(checkpoint, contract, world_size, dataset_size, allow_layout_change=False,
                   allow_epoch_extension=False):
    saved = checkpoint['training_config']
    old_world = checkpoint['world_size']
    old_prior, prior = saved['prior'], contract['prior']
    old_updates = math.ceil(dataset_size / old_world) // old_prior['batch_size'] // old_prior['accumulation']
    reference_updates = checkpoint.get('schedule_reference_updates', old_updates)
    state = deepcopy(checkpoint['progress'])
    state.setdefault('epoch_fraction', state['batch'] / (old_updates * old_prior['accumulation']))
    comparable_saved = deepcopy(saved)
    extension = None
    if old_prior.get('epochs') != prior.get('epochs'):
        if not allow_epoch_extension:
            raise ValueError('Epoch budget changed without explicit extension permission')
        if int(prior['epochs']) <= int(old_prior['epochs']):
            raise ValueError('An epoch extension must increase the budget')
        comparable_saved['prior']['epochs'] = prior['epochs']
        extension = dict(kind='epoch_extension', from_epochs=int(old_prior['epochs']),
                         to_epochs=int(prior['epochs']), epoch=state['epoch'], step=state['step'],
                         schedule_reference_epochs=checkpoint.get('schedule_reference_epochs', old_prior['epochs']),
                         schedule_policy='retain original decay, then hold final learning rate')
    changed = old_world != world_size or comparable_saved != contract
    if not changed:
        return state, reference_updates, extension
    if not allow_layout_change:
        raise ValueError('GPU layout or training configuration changed on resume')
    old_other, new_other = deepcopy(comparable_saved), deepcopy(contract)
    for value in (old_other, new_other):
        for key in ('batch_size', 'accumulation'):
            value['prior'].pop(key)
    if old_other != new_other:
        raise ValueError('Layout migration cannot change the model, data, or training objective')
    consumed = state['batch'] * old_prior['batch_size'] * old_world
    global_batch = prior['batch_size'] * prior['accumulation'] * world_size
    # Round down: at most one new update is replayed, with no omitted examples.
    state['batch'] = consumed // global_batch * prior['accumulation']
    migration = dict(old_world_size=old_world, world_size=world_size,
                     old_global_batch=old_prior['batch_size'] * old_prior['accumulation'] * old_world,
                     global_batch=global_batch, replayed_images=consumed % global_batch,
                     epoch=state['epoch'], step=state['step'],
                     schedule_epoch_fraction=state['epoch_fraction'])
    if extension is not None:
        migration['epoch_extension'] = extension
    return state, reference_updates, migration


def schedule_position(epoch_position, reference_updates, reference_epochs):
    """Keep extensions at the saved schedule's final LR, without a restart."""
    return min(epoch_position * reference_updates, reference_epochs * reference_updates - 1)


def epoch_fraction(initial_fraction, initial_batch, next_batch, total_batches):
    """Preserve the saved schedule position through a partial-epoch migration."""
    if total_batches == initial_batch:
        return 1.
    return initial_fraction + (1. - initial_fraction) * (next_batch - initial_batch) / (total_batches - initial_batch)
