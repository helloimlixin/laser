"""Audit a fresh Stage-2 trajectory and recovery of only that trajectory."""
import json
import math
import os
from pathlib import Path
import sys

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B / 'runtime'))
import torch
import wandb
import yaml

plan = json.loads((B / 'queue-plan.json').read_text())
policy = json.loads((B / 'target-policy.json').read_text())
options = yaml.safe_load((B / 'active.yaml').read_text())['options']
fresh = not options['resume']
assert options.get('init_stage2_checkpoint') is None
assert options['wandb_id'] == plan['run_id']
if fresh:
    assert options.get('resume_checkpoint') is None
else:
    assert Path(options['resume_checkpoint']).resolve().is_relative_to(Path(plan['persistent']) / 'train/checkpoints')

original_init = wandb.init


def audited_init(*args, **kwargs):
    c = kwargs['config']
    assert c['world_size'] == 8 and c['batch_size'] == 252 and c['accumulation_steps'] == 1
    assert c['total_batch_size'] == 2016 and c['optimizer_steps_per_epoch'] == 635
    assert c['total_optimizer_updates'] == 63500 and c['epochs'] == 100
    assert c['training_data_mode'] == 'online-fresh-images' and c['stochastic_atom_supports']
    assert c['combination_target_policy'] == policy
    assert c['fid_every'] == 5 and c['sample_grid_every'] == 200 and c['fid_num_samples'] == 50000
    assert c['lr_schedule'] == 'warmup-cosine' and c['warmup_epochs'] == 2
    if fresh:
        assert c['stage2_initialization'] == 'scratch'
        assert c['initial_global_step'] == 0 and c['initial_optimizer_state_count'] == 0
    else:
        assert c['stage2_initialization'] == 'resume'
        assert c['initial_global_step'] > 0 and c['initial_optimizer_state_count'] == 870
    c.update(trajectory_initialization='scratch', inherited_parent_weights=False,
             inherited_parent_optimizer=False, preceding_run=plan['parent_run'],
             support_teacher_from_first_update=True, frozen_stage1=True,
             target_distance='squared distance of complete jointly fitted sparse combination',
             teacher_entropy_definition='total conditional atom-plus-coefficient entropy per 4-pair latent site')
    kwargs['allow_val_change'] = True
    run = original_init(*args, **kwargs)
    run.config.update(c, allow_val_change=True)
    for p in (B / 'evidence').iterdir():
        if p.is_file():
            run.save(str(p), base_path=str(B), policy='now')
    return run


wandb.init = audited_init
original_step = torch.optim.AdamW.step
count = 0
initial = None


def audited_step(self, *args, **kwargs):
    global count, initial
    if count == 0:
        initial = dict(initial_optimizer_states=len(self.state), initial_lr=self.param_groups[0]['lr'], fresh=fresh)
        if fresh:
            assert len(self.state) == 0
            assert math.isclose(initial['initial_lr'], options['lr'] * options['warmup_start_ratio'], abs_tol=1e-12)
        else:
            assert len(self.state) == 870
            assert len({int(v['step']) for v in self.state.values()}) == 1
        phase = 'fresh' if fresh else 'resume'
        startup = dict(rank=int(os.environ['RANK']), run_id=plan['run_id'], **initial)
        (B / 'verification' / f'{phase}-startup-rank{startup["rank"]}.json').write_text(json.dumps(startup, indent=2))
    result = original_step(self, *args, **kwargs)
    count += 1
    if count == 20:
        tensors = [p for group in self.param_groups for p in group['params']]
        tensors += [s[k] for s in self.state.values() for k in ('exp_avg', 'exp_avg_sq')]
        assert len(self.state) == 870
        assert bool(torch.stack([torch.isfinite(t).all() for t in tensors]).all())
        r = dict(rank=int(os.environ['RANK']), retained_updates=count,
                 all_weights_and_moments_finite=True, optimizer_states=870, **initial)
        (B / 'verification' / f'rank{r["rank"]}.json').write_text(json.dumps(r, indent=2))
        if fresh:
            (B / 'verification' / f'fresh-rank{r["rank"]}.json').write_text(json.dumps(r, indent=2))
    return result


torch.optim.AdamW.step = audited_step
from src.training.cli import main
raise SystemExit(main())
