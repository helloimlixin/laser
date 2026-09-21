"""Recovery checks for the frozen released-tokenizer Church control."""


def validate_control_resume(payload, *, config, cache, protocol, run_id, loader_batches):
    required = {'state_dict', 'optimizer', 'scheduler', 'scaler', 'rng_states',
                'epoch', 'batch_in_epoch', 'step', 'attempts', 'skipped_amp_updates',
                'tokenizer', 'config', 'initial_weights_sha256', 'fid_protocol',
                'run_id', 'ranked_fid_checkpoints'}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f'Incomplete control checkpoint: {sorted(missing)}')
    if payload['config'] != config or payload['run_id'] != run_id:
        raise ValueError('Control configuration or run identity changed')
    if payload['tokenizer'] != cache or payload['fid_protocol'] != protocol:
        raise ValueError('Released tokenizer, cache, or FID protocol changed')
    if len(payload['rng_states']) != 2:
        raise ValueError('Control requires both saved rank RNG states')
    for state in payload['rng_states']:
        if not {'torch', 'cuda', 'numpy', 'python'} <= state.keys():
            raise ValueError('Incomplete RNG state')
    for name in ['epoch', 'batch_in_epoch', 'step', 'attempts', 'skipped_amp_updates']:
        if not isinstance(payload[name], int) or payload[name] < 0:
            raise ValueError(f'Invalid counter: {name}')
    if payload['attempts'] != payload['step'] + payload['skipped_amp_updates']:
        raise ValueError('Inconsistent optimizer counters')
    batch = payload['batch_in_epoch']
    accumulation = config['experiment']['accumulation_steps']
    if batch > loader_batches or (batch != loader_batches and batch % accumulation):
        raise ValueError('Checkpoint cursor is not an optimizer boundary')
    pending = payload.get('pending_evaluation_epoch')
    if pending is not None and (pending != payload['epoch'] or batch != 0):
        raise ValueError('Invalid pending evaluation boundary')
    return payload['epoch'], batch
