"""Checkpointable FID plateau control for the corrected Church experiment."""
import copy
import math


DEFAULT_POLICY = {'factor': .5, 'patience': 2, 'min_delta': .25,
                  'cooldown': 2, 'min_lr': 1e-6, 'samples': 4096}


def metric_protocol(result):
    return {key: result.get(key, default) for key, default in {
        'samples': None, 'seed': None, 'atom_top_k': None,
        'coefficient_sampling': None, 'temperature': None, 'precision': None,
        'world_size': 1, 'generation_batch_per_gpu': 32,
        'seed_rule': 'single continuous RNG stream',
    }.items()}


class FidLearningRate:
    def __init__(self, config=None):
        self.config = dict(DEFAULT_POLICY if config is None else config)
        p = self.config
        if not (0 < p['factor'] < 1 and p['patience'] >= 1 and p['cooldown'] >= 0
                and p['min_delta'] >= 0 and p['min_lr'] > 0 and p['samples'] >= 2):
            raise ValueError('Invalid FID learning-rate policy')
        self.multiplier = 1.
        self.best = None
        self.bad_checks = 0
        self.cooldown_remaining = 0
        self.last_step = -1
        self.last_fid = None
        self.protocol = None
        self.reductions = 0

    def state_dict(self):
        return copy.deepcopy(vars(self))

    @classmethod
    def from_state(cls, state, config=None):
        instance = cls(state['config'] if config is None else config)
        if instance.config != state['config']:
            raise ValueError('Cannot change the saved FID learning-rate policy')
        if vars(instance).keys() != state.keys():
            raise ValueError('Invalid FID learning-rate state')
        instance.__dict__.update(copy.deepcopy(state))
        if not (0 < instance.multiplier <= 1 and instance.bad_checks >= 0
                and instance.cooldown_remaining >= 0 and instance.reductions >= 0):
            raise ValueError('Invalid FID learning-rate counters')
        return instance

    def learning_rate(self, base_lr):
        if not math.isfinite(base_lr) or base_lr <= 0:
            raise ValueError('Expected a finite positive base learning rate')
        return max(self.config['min_lr'], base_lr * self.multiplier)

    def observe(self, step, result, base_lr):
        fid = float(result['fid'])
        protocol = metric_protocol(result)
        if not math.isfinite(fid) or fid < 0:
            raise ValueError('Invalid FID score')
        if protocol['samples'] != self.config['samples']:
            raise ValueError('FID sample count differs from the monitored series')
        if result.get('optimizer_step', step) != step:
            raise ValueError('FID belongs to a different checkpoint step')
        if step <= self.last_step:
            if step == self.last_step and (fid != self.last_fid or protocol != self.protocol):
                raise ValueError('An already observed evaluation changed')
            return None
        before = self.learning_rate(base_lr)
        event = {'evaluation_step': step, 'fid': fid, 'lr_before': before}
        if protocol != self.protocol:
            # Never compare scores across sample-count, RNG or sampler changes.
            self.protocol = protocol
            self.best = fid
            self.bad_checks = self.cooldown_remaining = 0
            decision = 'baseline'
        else:
            improved = fid < self.best and fid <= self.best - self.config['min_delta']
            if improved:
                self.best = fid
            if self.cooldown_remaining:
                self.cooldown_remaining -= 1
                self.bad_checks = 0
                decision = 'cooldown_improved' if improved else 'cooldown'
            elif improved:
                self.bad_checks = 0
                decision = 'improved'
            else:
                self.bad_checks += 1
                decision = 'watch'
                if self.bad_checks >= self.config['patience']:
                    self.bad_checks = 0
                    if before > self.config['min_lr']:
                        self.multiplier *= self.config['factor']
                        self.reductions += 1
                        self.cooldown_remaining = self.config['cooldown']
                        decision = 'reduced'
                    else:
                        decision = 'at_floor'
        self.last_step, self.last_fid = int(step), fid
        event.update(decision=decision, lr_after=self.learning_rate(base_lr),
                     multiplier=self.multiplier, best_fid=self.best,
                     bad_checks=self.bad_checks, cooldown_remaining=self.cooldown_remaining,
                     reductions=self.reductions)
        return event


def synchronize_observation(controller, step, result, base_lr):
    """Rank zero decides once; every optimizer uses the same saved multiplier."""
    import torch.distributed as dist
    distributed = dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    payload = [None, None]
    if rank == 0:
        payload = [controller.state_dict(), controller.observe(step, result, base_lr)]
        payload[0] = controller.state_dict()
    if distributed:
        dist.broadcast_object_list(payload, src=0)
        if rank != 0:
            controller.__dict__.update(FidLearningRate.from_state(payload[0], controller.config).__dict__)
    return payload[1]
