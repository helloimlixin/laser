"""Learning-rate continuation and durable ranked checkpoints for Church stage 2."""
import copy
import json
import math
import os
from pathlib import Path
import tempfile

from src.church_fid_lr import FidLearningRate


class ContinuationLearningRate:
    """Apply a saved FID multiplier without compounding the cosine recurrence."""

    def __init__(self, optimizer, scheduler, *, scale=.5, state=None):
        self.optimizer, self.scheduler = optimizer, scheduler
        policy = dict(factor=.5, patience=2, min_delta=.1, cooldown=1,
                      min_lr=1e-6, samples=50000)
        self.controller = FidLearningRate(policy)
        self.nominal_lrs = list(scheduler.get_last_lr())
        if state is None:
            if not math.isfinite(scale) or not 0 < scale <= 1:
                raise ValueError('Resume LR scale must be in (0, 1]')
            self.controller.multiplier = scale
        else:
            self.controller = FidLearningRate.from_state(state['controller'], policy)
            self.nominal_lrs = list(state['nominal_lrs'])
        if len(self.nominal_lrs) != len(optimizer.param_groups):
            raise ValueError('Learning-rate parameter groups changed')
        self.apply()

    def apply(self):
        for group, nominal in zip(self.optimizer.param_groups, self.nominal_lrs):
            group['lr'] = max(self.controller.config['min_lr'],
                              nominal * self.controller.multiplier)

    def step(self):
        # CosineAnnealingLR is recursive: feed it its unscaled previous LR.
        for group, nominal in zip(self.optimizer.param_groups, self.nominal_lrs):
            group['lr'] = nominal
        self.scheduler.step()
        self.nominal_lrs = list(self.scheduler.get_last_lr())
        self.apply()

    def observe(self, step, fid, protocol):
        result = dict(fid=fid, samples=50000, optimizer_step=step,
                      seed=71000, temperature=protocol['sampling']['temperature'],
                      atom_top_k=protocol['sampling']['top_k'], world_size=2,
                      generation_batch_per_gpu=100,
                      coefficient_sampling=json.dumps(protocol, sort_keys=True))
        event = self.controller.observe(step, result,
                                        max(self.nominal_lrs[0], 1e-30))
        self.apply()
        return event

    def state_dict(self):
        return dict(controller=self.controller.state_dict(),
                    nominal_lrs=list(self.nominal_lrs))


def ranked_candidates(previous, *, fid, epoch, step):
    """Rank only evaluated checkpoints, with deterministic tie handling."""
    if not math.isfinite(fid) or fid < 0:
        raise ValueError('Expected a finite nonnegative FID')
    candidate = dict(fid=float(fid), epoch=int(epoch), step=int(step),
                     path=f'fid-epoch{epoch:03d}-step{step:07d}.pt')
    rows = [copy.deepcopy(row) for row in previous if row['step'] != step]
    rows.append(candidate)
    return sorted(rows, key=lambda row: (row['fid'], row['step']))[:3]


def upload_checkpoints(run, output, ranked, *, epoch, step, protocol):
    """Commit an immutable latest+top-three artifact before allowing pruning.

    Files are hardlinked from atomically saved checkpoints and remain pinned
    until W&B confirms the upload. Failures leave all training states intact.
    """
    if run is None:
        return None
    import wandb

    output = Path(output)
    metadata = dict(run_id=run.id, epoch=epoch, optimizer_step=step,
                    ranked_fid_checkpoints=ranked, fid_protocol=protocol,
                    checkpoint_policy='latest full state plus best three FID states')
    with tempfile.TemporaryDirectory(prefix='.upload-', dir=output) as temporary:
        staging = Path(temporary)
        sources = [('last.pt', output / 'last.pt')]
        sources += [(f'best-fid-{i:02d}.pt', output / row['path'])
                    for i, row in enumerate(ranked, 1)]
        artifact = wandb.Artifact(f'model-{run.id}-selected-checkpoints',
                                  type='model', metadata=metadata)
        for name, source in sources:
            os.link(source, staging / name)
            artifact.add_file(str(staging / name), name=name)
        manifest = staging / 'selection.json'
        manifest.write_text(json.dumps(metadata, indent=2) + '\n')
        artifact.add_file(str(manifest), name='selection.json')
        committed = run.log_artifact(artifact, aliases=['latest'])
        committed.wait()
        receipt = dict(artifact=committed.qualified_name, **metadata)
        temp_receipt = output / 'checkpoint-upload.tmp'
        temp_receipt.write_text(json.dumps(receipt, indent=2) + '\n')
        temp_receipt.replace(output / 'checkpoint-upload.json')
        run.summary['checkpoint_upload/last_step'] = step
        run.summary['checkpoint_upload/artifact'] = committed.qualified_name
        run.summary['checkpoint_upload/best_fid_count'] = len(ranked)
        return receipt
