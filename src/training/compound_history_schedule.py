"""Separate rates for pretrained and newly added compound-history parameters."""
import copy
import math

KIND = 'compound-history-discriminative-cosine-v1'


def parameter_segments(names, new_names):
    """Contiguous groups retain checkpoint parameter IDs and tensor order."""
    new = set(new_names)
    if not new or not new < set(names) or len(names) != len(set(names)):
        raise ValueError('Invalid compound parameter inventory')
    segments = []
    for index, name in enumerate(names):
        kind = 'history' if name in new else 'pretrained'
        if not segments or segments[-1][0] != kind:
            segments.append((kind, []))
        segments[-1][1].append(index)
    return segments


def repartition_optimizer_state(state, names, new_names):
    """Change groups only; reuse every original Adam tensor and parameter ID."""
    segments = parameter_segments(names, new_names)
    groups = state['param_groups']
    expected = list(range(len(names)))
    if [i for group in groups for i in group['params']] != expected:
        raise ValueError('Adam parameter order differs from the compound inventory')
    if len(groups) != 1:
        actual = [(g.get('compound_group'), g['params']) for g in groups]
        if actual != segments:
            raise ValueError('Saved compound optimizer groups changed')
        return state
    return dict(state, param_groups=[dict(groups[0], params=ids, compound_group=kind)
                                    for kind, ids in segments])


class CompoundHistorySchedule:
    def __init__(self, optimizer, state):
        if state.get('kind') != KIND:
            raise ValueError('Expected a compound-history schedule')
        self.optimizer = optimizer
        self.policy = copy.deepcopy(state['policy'])
        self.source_scheduler = copy.deepcopy(state['source_scheduler'])
        self.last_epoch = int(state['last_epoch'])
        self.last_observation_step = int(state['last_observation_step'])
        self.last_fid = state['last_fid']
        self.best = float(state['best'])
        p = self.policy
        if (not 0 <= p['start_step'] < p['start_step'] + p['warmup_steps'] < p['total_steps']
                or p['min_lr'] != 0 or not p['pretrained_peak_lr'] > 0
                or not p['history_peak_lr'] > 0 or not p['history_start_lr'] > 0
                or not p['start_step'] <= self.last_epoch <= p['total_steps']):
            raise ValueError('Invalid compound schedule policy or cursor')
        for group in optimizer.param_groups:
            expected = self.rate_at(group['compound_group'], self.last_epoch)
            if not math.isclose(group['lr'], expected, rel_tol=1e-12, abs_tol=1e-20):
                raise ValueError('Adam LR does not match its compound schedule cursor')
        self._apply()

    def rate_at(self, group, step):
        p = self.policy
        if not p['start_step'] <= step <= p['total_steps']:
            raise ValueError('Schedule step outside the remaining training interval')
        if group == 'pretrained':
            return p['pretrained_peak_lr'] * .5 * (1 + math.cos(math.pi * step / p['total_steps']))
        if group != 'history':
            raise ValueError('Unknown compound parameter group')
        warm_end = p['start_step'] + p['warmup_steps']
        if step < warm_end:
            progress = (step - p['start_step']) / p['warmup_steps']
            return p['history_start_lr'] + (p['history_peak_lr']-p['history_start_lr']) * progress
        progress = (step - warm_end) / (p['total_steps'] - warm_end)
        return p['history_peak_lr'] * .5 * (1 + math.cos(math.pi * progress))

    def _apply(self):
        for group in self.optimizer.param_groups:
            group['lr'] = self.rate_at(group['compound_group'], self.last_epoch)
            group['initial_lr'] = self.policy[group['compound_group'] + '_peak_lr']

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

    def step(self):
        if self.last_epoch >= self.policy['total_steps']:
            raise ValueError('Compound schedule exhausted')
        self.last_epoch += 1
        self._apply()

    def observe(self, fid):
        fid = float(fid)
        if not math.isfinite(fid) or fid < 0:
            raise ValueError('Invalid official FID')
        if self.last_epoch == self.last_observation_step and fid == self.last_fid:
            return None
        if self.last_epoch <= self.last_observation_step:
            raise ValueError('Official observation precedes optimizer progress')
        improved = fid < self.best
        self.best = min(self.best, fid)
        self.last_fid = fid
        self.last_observation_step = self.last_epoch
        lr = self.rate_at('pretrained', self.last_epoch)
        return dict(fid=fid, schedule_step=self.last_epoch, decision='improved' if improved else 'monitor',
                    lr_before=lr, lr_after=lr, history_lr=self.rate_at('history',self.last_epoch),
                    multiplier=1., reductions=0, controller_best=self.best)

    def state_dict(self):
        return dict(kind=KIND, policy=copy.deepcopy(self.policy), last_epoch=self.last_epoch,
                    last_observation_step=self.last_observation_step, last_fid=self.last_fid,
                    best=self.best, source_scheduler=copy.deepcopy(self.source_scheduler))


def migrate_schedule(optimizer, source_state, *, completed_steps, initial_lr,
                     total_steps, history_peak_lr=1e-5, warmup_steps=200):
    if source_state['kind'] != 'original-rqtransformer-global-cosine-v1':
        raise ValueError('Migration requires the saved absolute global cosine')
    source_policy = source_state['policy']
    if (source_state['last_epoch'] != completed_steps or source_policy['total_steps'] != total_steps
            or source_policy['initial_lr'] != initial_lr or source_policy['min_lr'] != 0
            or source_policy.get('adaptive_reductions') is not False):
        raise ValueError('Source cosine configuration or cursor changed')
    current_lr = initial_lr * .5 * (1 + math.cos(math.pi * completed_steps / total_steps))
    if any(not math.isclose(g['lr'],current_lr,rel_tol=1e-12,abs_tol=1e-20) for g in optimizer.param_groups):
        raise ValueError('Source Adam LR is inconsistent with the saved clock')
    state = dict(kind=KIND, policy=dict(start_step=completed_steps, total_steps=total_steps,
        pretrained_peak_lr=initial_lr, history_peak_lr=history_peak_lr,
        history_start_lr=current_lr, warmup_steps=warmup_steps, min_lr=0.),
        last_epoch=completed_steps, last_observation_step=source_state['last_observation_step'],
        last_fid=source_state['last_fid'], best=source_state['best'], source_scheduler=copy.deepcopy(source_state))
    return CompoundHistorySchedule(optimizer, state)
