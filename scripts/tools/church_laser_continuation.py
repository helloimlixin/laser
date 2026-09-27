"""Restore the saved cosine/FID control, including its unscaled cosine LR."""
import json
import math

from church_fid_lr import FidLearningRate, synchronize_observation
from church_control_resume_support import ranked_candidates
from church_control_resume_support import upload_checkpoints as _upload


class ContinuationLearningRate:
    def __init__(self, optimizer, scheduler, scale=.5, state=None):
        self.optimizer, self.scheduler = optimizer, scheduler
        if state is not None:
            self.controller = FidLearningRate.from_state(state['controller'])
            self.nominal_lrs = list(state['nominal_lrs'])
            for group, nominal in zip(optimizer.param_groups, self.nominal_lrs):
                assert math.isclose(group['lr'], self.controller.learning_rate(nominal), rel_tol=1e-12)
        else:
            self.controller = FidLearningRate(dict(factor=.5, patience=2, min_delta=.1,
                cooldown=1, min_lr=1e-6, samples=50000))
            self.controller.multiplier = scale
            self.nominal_lrs = [g['lr'] for g in optimizer.param_groups]
        self._apply()

    def _apply(self):
        for group, nominal in zip(self.optimizer.param_groups, self.nominal_lrs):
            group['lr'] = self.controller.learning_rate(max(nominal, 1e-30))

    def state_dict(self):
        return dict(controller=self.controller.state_dict(), nominal_lrs=list(self.nominal_lrs))

    def step(self):
        for group, nominal in zip(self.optimizer.param_groups, self.nominal_lrs):
            group['lr'] = nominal
        self.scheduler.step()
        self.nominal_lrs = [g['lr'] for g in self.optimizer.param_groups]
        self._apply()

    def observe(self, step, fid, protocol):
        result = dict(fid=fid, samples=50000, optimizer_step=step, seed=71000,
            atom_top_k=protocol['sampling']['top_k'], temperature=protocol['sampling']['temperature'],
            coefficient_sampling=json.dumps(protocol, sort_keys=True),
            generation_batch_per_gpu=100, world_size=1,
            seed_rule='independent fixed global batches; hardware-independent rank assignment')
        event = synchronize_observation(self.controller, step, result, max(self.nominal_lrs[0], 1e-30))
        self._apply()
        return event


def upload_checkpoints(run, out, ranked, **kwargs):
    if run is not None:
        _upload(run, out, ranked, **kwargs)
