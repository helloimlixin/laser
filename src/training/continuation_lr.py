"""Continue an optimizer with a gradual LR change, hold, and cosine decay."""
import math


class RampHoldCosineSchedule:
    kind = "continuation-ramp-hold-cosine-v1"

    def __init__(self, optimizer, *, policy, completed_steps, state_dict=None,
                 revision_from=None):
        self.optimizer = optimizer
        self.policy = dict(policy)
        p = self.policy
        if not (0 <= p["anchor_step"] < p["total_steps"]
                and p["ramp_steps"] > 0
                and p["anchor_step"] + p["ramp_steps"] <= p["hold_until_step"] < p["total_steps"]
                and 0 <= p["min_lr"] <= min(p["anchor_lr"], p["peak_lr"])
                and all(math.isfinite(p[key]) for key in ("anchor_lr", "peak_lr", "min_lr"))
                and p["peak_lr"] > 0):
            raise ValueError("Invalid continuation LR policy")
        self.last_epoch = int(completed_steps)
        if not p["anchor_step"] <= self.last_epoch <= p["total_steps"]:
            raise ValueError("Continuation scheduler/checkpoint step outside policy")
        if state_dict is not None:
            if state_dict.get("kind") != self.kind:
                raise ValueError("Continuation LR policy changed on resume")
            if state_dict["last_epoch"] != self.last_epoch:
                raise ValueError("Continuation scheduler/checkpoint step mismatch")
            if state_dict.get("policy") != p:
                if revision_from is None or state_dict.get("policy") != dict(revision_from):
                    raise ValueError("Continuation LR policy changed on resume")
                if self.last_epoch != p["anchor_step"]:
                    raise ValueError("LR policy revision requires its exact anchor step")
                if any(abs(group["lr"] - p["anchor_lr"]) > 1e-12
                       for group in optimizer.param_groups):
                    raise ValueError("LR policy revision requires the checkpoint anchor LR")
        self.base_lrs = [p["peak_lr"] for _ in optimizer.param_groups]
        self._apply()

    def _apply(self):
        p, step = self.policy, self.last_epoch
        ramp_end = p["anchor_step"] + p["ramp_steps"]
        if step < ramp_end:
            fraction = (step - p["anchor_step"]) / p["ramp_steps"]
            lr = p["anchor_lr"] + (p["peak_lr"] - p["anchor_lr"]) * fraction
        elif step <= p["hold_until_step"]:
            lr = p["peak_lr"]
        else:
            fraction = (step - p["hold_until_step"]) / (p["total_steps"] - p["hold_until_step"])
            lr = p["min_lr"] + .5 * (p["peak_lr"] - p["min_lr"]) * (1 + math.cos(math.pi * fraction))
        for group in self.optimizer.param_groups:
            group["lr"] = lr
            group["initial_lr"] = p["peak_lr"]
        self._last_lr = [lr for _ in self.optimizer.param_groups]

    def step(self):
        if self.last_epoch >= self.policy["total_steps"]:
            raise ValueError("Continuation LR schedule exhausted")
        self.last_epoch += 1
        self._apply()

    def get_last_lr(self):
        return list(self._last_lr)

    def state_dict(self):
        return dict(kind=self.kind, policy=dict(self.policy), last_epoch=self.last_epoch,
                    base_lrs=list(self.base_lrs), _last_lr=list(self._last_lr))
