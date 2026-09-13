"""Learning-rate schedules for the audio prior, including bounded continuation."""
import math


def learning_rate_at(step, config, total_steps):
    continuation = config.get('continuation')
    if continuation is not None:
        start = continuation['start_step']
        if step < start or total_steps <= start:
            raise ValueError('Continuation requires a restored step and a later end step')
        warmup = continuation.get('warmup_steps', 0)
        peak = continuation.get('peak_lr', continuation['start_lr'])
        if warmup < 0 or warmup >= total_steps - start:
            raise ValueError('Continuation warmup must fit inside the remaining budget')
        if warmup and step < start + warmup:
            return continuation['start_lr'] + (peak - continuation['start_lr']) * (step - start) / warmup
        progress = min(1., (step - start - warmup) / (total_steps - start - warmup))
        return continuation['end_lr'] + .5 * (peak - continuation['end_lr']) * (
            1 + math.cos(math.pi * progress))
    warmup = config['warmup_steps']
    if step < warmup:
        return config['learning_rate'] * (step + 1) / warmup
    progress = min(1., max(0., (step - warmup) / max(1, total_steps - warmup)))
    ratio = config['min_lr_ratio']
    return config['learning_rate'] * (ratio + (1 - ratio) * .5 * (1 + math.cos(math.pi * progress)))
