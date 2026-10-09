"""Try scalar coefficient CRPS from a full protected best, retaining LR/Adam."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import continue_imagenet_fid16379 as recovery

RUN_ID = 'imagenet-rfid421-epoch64-crps005-floor0-8h100-20261006'
MAX_WEIGHT = .05
RAMP_STEPS = 626
record = recovery.record


def install_crps(base):
    """Wire the actual frozen physical-pair branch, including its compiled loss."""
    repository = Path(__file__).resolve().parents[2]
    runtime = base / 'source/runtime/src/training'
    shutil.copyfile(repository / 'src/training/physical_pair_crps.py', runtime / 'physical_pair_crps.py')
    trainer = runtime / 'rqtransformer.py'
    code = trainer.read_text()
    def replace(old, new):
        nonlocal code
        assert code.count(old) == 1, old
        code = code.replace(old, new)
    replace('from src.training.exact_global_batch import ExactGlobalBatchSampler\n',
            'from src.training.exact_global_batch import ExactGlobalBatchSampler\n'
            'from src.training.physical_pair_crps import physical_pair_objective, crps_weight_at_step\n')
    replace('    p.add_argument("--geometry-loss-weight", type=float, default=0.0)',
            '    p.add_argument("--coeff-crps-ramp-steps", type=int, default=0)\n'
            '    p.add_argument("--coeff-crps-start-step", type=int, default=0)\n'
            '    p.add_argument("--geometry-loss-weight", type=float, default=0.0)')
    replace('    if args.coeff_crps_weight < 0:\n        raise ValueError("--coeff-crps-weight cannot be negative")',
            '    if not math.isfinite(args.coeff_crps_weight) or args.coeff_crps_weight < 0:\n'
            '        raise ValueError("CRPS weight must be finite and nonnegative")\n'
            '    if args.coeff_crps_ramp_steps < 0 or args.coeff_crps_start_step < 0:\n'
            '        raise ValueError("CRPS schedule steps cannot be negative")\n'
            '    if (args.coeff_crps_ramp_steps or args.coeff_crps_start_step) and not args.physical_pair_context:\n'
            '        raise ValueError("CRPS step ramp requires physical-pair context")')
    replace('''                        loss = compiled_sparse_objective(
                            atom_logits, coeff_logits, *compact_targets, accumulation
                        )''',
            '''                        crps_weight = coeff_logits.new_tensor(
                            crps_weight_at_step(args.coeff_crps_weight, global_step,
                                                args.coeff_crps_start_step, args.coeff_crps_ramp_steps),
                            dtype=torch.float32,
                        )
                        loss = physical_pair_objective(
                            atom_logits, coeff_logits, *compact_targets,
                            aux.coeff_bins, crps_weight, accumulation,
                        )''')
    trainer.write_text(code)
    entry = base / 'entry.py'
    code = entry.read_text()
    start = code.index('def sparse_objective(')
    end = code.index('\ndef submit_upload():', start)
    code = code[:start] + '''from src.training.physical_pair_crps import (
 physical_pair_objective_components, crps_weight_at_step, validate_coefficient_bins)
compiled_crps_objective=torch.compile(physical_pair_objective_components,fullgraph=True,dynamic=True) if COMPILE else physical_pair_objective_components

def physical_pair_objective(atom_logits,coeff_logits,atoms,probabilities,bins,weight,accumulation):
 total,classification,crps=compiled_crps_objective(atom_logits,coeff_logits,atoms,probabilities,bins,weight,accumulation)
 audit=VERIFY/('crps-step20-rank'+os.environ['RANK']+'.json')
 if UPDATES==19 and not audit.exists():
  # Probe the compiled CRPS gradient on real training logits in a separate
  # graph. Native training keeps its single backward and donated buffers.
  slices=(slice(0,1),)*(atoms.ndim-1)+(slice(None),)
  probe_coeff=coeff_logits[slices].detach().clone().requires_grad_(True)
  probe=compiled_crps_objective(atom_logits[slices].detach(),probe_coeff,atoms[slices],probabilities[slices].detach(),bins,weight,accumulation)
  gradient=torch.autograd.grad(weight*probe[2]/accumulation,probe_coeff)[0]
  gradient_norm=float(gradient.float().norm());del gradient
  scalar_weight=float(weight)
  assert scalar_weight>0 and bool(torch.isfinite(total)) and bool(torch.isfinite(crps))
  assert gradient_norm>0 and bool(torch.isfinite(torch.tensor(gradient_norm)))
  record(audit,dict(global_step=INITIAL_RECOVERY['global_step']+UPDATES,
   weight=scalar_weight,max_weight=ARGS.coeff_crps_weight,start_step=ARGS.coeff_crps_start_step,
   ramp_steps=ARGS.coeff_crps_ramp_steps,classification=float(classification.detach()),
   range_normalized_crps=float(crps.detach()),loss=float(total.detach()),
   weighted_crps_contribution=scalar_weight*float(crps.detach())/accumulation,
   compiled_auxiliary_gradient_norm_on_training_logits=gradient_norm,accumulation=accumulation,
   bin_count=bins.numel(),bin_min=float(bins[0]),bin_max=float(bins[-1]),
   normalization='CDF squared integral divided by coefficient bin range',
   finite=True,training_loss_only=True))
 return total
training.physical_pair_objective=physical_pair_objective
''' + code[end:]
    def replace_entry(old, new):
        nonlocal code
        assert code.count(old) == 1, old
        code = code.replace(old, new)
    replace_entry(" kwargs['clamp_coeffs']=False;return original_aux(self,*args,**kwargs)",
                  " kwargs['clamp_coeffs']=False\n result=original_aux(self,*args,**kwargs)\n"
                  " from src.training.physical_pair_crps import validate_coefficient_bins\n"
                  " validate_coefficient_bins(self.coeff_bins,self.coeff_bins.numel())\n return result")
    replace_entry(" step=int(payload['global_step']);epoch=int(payload['epoch'])\n",
                  " step=int(payload['global_step']);epoch=int(payload['epoch'])\n"
                  " payload=dict(payload,objective_revision=dict(version='physical-pair-range-normalized-crps-v1',\n"
                  "  max_weight=ARGS.coeff_crps_weight,start_step=ARGS.coeff_crps_start_step,ramp_steps=ARGS.coeff_crps_ramp_steps,\n"
                  "  completed_steps=max(0,step-ARGS.coeff_crps_start_step),\n"
                  "  next_weight=crps_weight_at_step(ARGS.coeff_crps_weight,step,ARGS.coeff_crps_start_step,ARGS.coeff_crps_ramp_steps)))\n")
    replace_entry("  INITIAL_RECOVERY=recovery_metadata(payload)\n",
                  "  objective=payload['objective_revision']\n"
                  "  assert objective['version']=='physical-pair-range-normalized-crps-v1'\n"
                  "  assert (objective['max_weight'],objective['start_step'],objective['ramp_steps'])==(ARGS.coeff_crps_weight,ARGS.coeff_crps_start_step,ARGS.coeff_crps_ramp_steps)\n"
                  "  assert objective['completed_steps']==max(0,payload['global_step']-ARGS.coeff_crps_start_step)\n"
                  "  assert objective['next_weight']==crps_weight_at_step(ARGS.coeff_crps_weight,payload['global_step'],ARGS.coeff_crps_start_step,ARGS.coeff_crps_ramp_steps)\n"
                  "  INITIAL_RECOVERY=recovery_metadata(payload)\n")
    # Keep this before the source block: automatic forks replace that block.
    replace_entry(' config.update(resume_source_epoch=',
                  " config.update(objective_revision='physical-pair-range-normalized-crps-v1',\n"
                  "  objective='existing equal atom/coeff CE plus range-normalized coefficient CRPS',\n"
                  "  coeff_crps_normalization='coefficient bin range',\n"
                  "  coeff_crps_max_weight=ARGS.coeff_crps_weight,coeff_crps_start_step=ARGS.coeff_crps_start_step,\n"
                  "  coeff_crps_ramp_steps=ARGS.coeff_crps_ramp_steps,objective_clock='saved global step')\n"
                  ' config.update(resume_source_epoch=')
    entry.write_text(code)


def prepare(base, evidence, parent_base, parent_evidence):
    import torch
    import yaml
    torch.set_num_threads(4)
    audit = evidence / 'continuation-20261005'
    checkpoints = audit / 'train/checkpoints'
    checkpoints.mkdir(parents=True, exist_ok=True)
    latest = checkpoints / 'last.pt'
    if latest.exists():
        raise RuntimeError('Refuse to replace an existing continuation')
    for directory in ('source/runtime', 'support'):
        shutil.copytree(parent_base / directory, base / directory, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('imagenet_fid_rewind_guard.py', 'imagenet_fid_rewind_supervisor.py',
                 'continue_imagenet_coeff_crps.py'):
        shutil.copyfile(Path(__file__).with_name(name), base / 'support' / name)
    (base / 'inputs').mkdir(exist_ok=True)
    source = base / 'inputs/source-epoch064.pt'
    pinned = parent_base / 'final-cpu-upload/best-fid-resume.pt'
    if not source.exists():
        os.link(pinned, source)
    sys.path[:0] = [str(base / 'source/runtime'), str(base / 'support')]
    from src.training import k4_checkpoint_io as checkpoint_io
    from src.training.full_resume_upload import recovery_metadata
    from src.training.fid_adaptive_schedule import FidAdaptiveSchedule
    raw = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    original = recovery_metadata(raw)
    assert (original['epoch'], original['global_step'], original['adam_step'], original['next_microbatch']) == (64, 40064, 5634, 0)
    assert original['rng_ranks'] == 8 and original['adam_parameters'] == 782
    metrics = original['original_rqtransformer_metrics']
    assert metrics['metric_backend'] == 'original_rqtransformer' and metrics['fid'] == 15.554077729782705
    state = raw['scheduler']
    initial_lr = original['saved_learning_rates'][0]
    assert state['policy']['min_lr'] == 0. and state['last_epoch'] == state['last_observation_step'] == 5008
    assert initial_lr == FidAdaptiveSchedule.lr_at_step(state['policy'], state['last_epoch'], state['multiplier'])
    parent_recipe = yaml.safe_load((parent_base / 'recipe.yaml').read_text())
    parent_id = parent_recipe['options']['wandb_id']
    recipe = parent_recipe
    recipe['options'].update(checkpoint=str(base / 'inputs/resume-stage1-tokenizer.pt'),
        output=str(base / 'production/train'), checkpoint_dir=str(checkpoints),
        wandb_id=RUN_ID, lr_schedule_restart_id=RUN_ID,
        coeff_crps_weight=MAX_WEIGHT, coeff_crps_ramp_steps=RAMP_STEPS,
        coeff_crps_start_step=original['global_step'],
        wandb_name=f'ImageNet rFID4.21 K4 | epoch64 FID15.554 | CE+CRPS0.05 ramp626 | LR{initial_lr:.3e} floor0 | 8 H100')
    recipe['options'].pop('resume_checkpoint', None)
    for name in ('resume-stage1-tokenizer.pt', 'resume-weights-inception-2015-12-05-6726825d.pth'):
        os.link(parent_base / 'inputs' / name, base / 'inputs' / name)
    for name in ('torch-cache', 'inductor-cache'):
        (base / name).symlink_to((parent_base / name).resolve(), target_is_directory=True)
    record(base / 'official-baseline.json', metrics)
    shutil.copyfile(parent_base / 'entry.py', base / 'entry.py')
    install_crps(base)
    entry = (base / 'entry.py').read_text()
    start = entry.index(' config.update(resume_source_epoch=')
    end = entry.index(' WB=original_init(*args,**kwargs)', start)
    entry = entry[:start] + (
        ' config.update(resume_source_epoch=64,resume_source_global_step=40064,\n'
        f'  source_run={"helloimlixin-rutgers/laser/"+parent_id!r},\n'
        '  source_checkpoint_epoch=64,source_checkpoint_step=40064,source_checkpoint_fid=15.554077729782705,\n'
        '  source_inception_score=73.79450225830078,source_optimizer_restored=True,trained_source_optimizer_available=True,\n'
        "  optimizer_initialization='restored all trained AdamW moments and 782 counters at5634',\n"
        "  rng_initialization='restored all 8 epoch64 RNG streams',\n"
        "  evaluation_protocol='Official RQ-Transformer FID/10-split IS only; validation50k/generated50k',\n"
        "  baseline_evaluation_provenance='inherited official score from unchanged source model',\n"
        "  learning_rate_schedule='saved zero-floor cosine through epoch100; strict FID regression rewind',\n"
        "  lr_continuation='same current best checkpoint LR and complete scheduler history; trained Adam retained',\n"
        '  source_scheduler_step=5008,continuation_scheduler_origin_adam_step=626,continuation_scheduler_origin_global_step=35056,\n'
        f'  restart_initial_lr={initial_lr!r},lr_floor_before=0.,lr_floor_after=0.,automatic_fid_rewind=True,cosine_decay_end_epoch=100)\n'
    ) + entry[end:]
    (base / 'entry.py').write_text(entry)
    (base / 'recipe.yaml').write_text(yaml.safe_dump(recipe, sort_keys=False))
    from imagenet_fid_rewind_guard import install_guard
    install_guard(base)
    fid_alias = checkpoints / 'best_fid_15.5541_epoch_064.pt'
    is_alias = checkpoints / 'best_is_73.7945_epoch_064.pt'
    objective = dict(version='physical-pair-range-normalized-crps-v1',
        max_weight=MAX_WEIGHT, start_step=40064, ramp_steps=RAMP_STEPS, completed_steps=0, next_weight=0.)
    payload = dict(raw, config=dict(raw['config'], **recipe['options']), objective_revision=objective,
        best_fid=[(metrics['fid'], str(fid_alias))], best_inception=[(metrics['inception_score'], str(is_alias))])
    # Objective change only: scheduler clock, amplitude, groups, moments and RNG all survive unchanged.
    assert payload['optimizer'] is raw['optimizer'] and payload['scheduler'] is raw['scheduler']
    os.environ.update(LASER_CHECKPOINT_STAGING_DIR=str(base / 'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(base / 'checkpoint-upload-cache'), LASER_CHECKPOINT_IMMUTABLE_FILES='1')
    checkpoint_io.atomic_torch_save(payload, latest)
    for alias in (fid_alias, is_alias, checkpoints / 'resume-anchor.pt'):
        alias.symlink_to(latest.resolve().relative_to(checkpoints))
    restored = torch.load(checkpoint_io._checkpoint_upload_source(latest), map_location='cpu', weights_only=False, mmap=True)
    compared = 0
    for key, tensor in raw['state_dict'].items():
        assert torch.equal(tensor, restored['state_dict'][key]), key
        compared += tensor.numel()
    for key, fields in raw['optimizer']['state'].items():
        for field, tensor in fields.items():
            if torch.is_tensor(tensor):
                assert torch.equal(tensor, restored['optimizer']['state'][key][field]), (key, field)
                compared += tensor.numel()
    for rank, streams in enumerate(raw['rng_state_by_rank']):
        for name, tensor in streams.items():
            assert torch.equal(tensor, restored['rng_state_by_rank'][rank][name])
    assert restored['scheduler'] == raw['scheduler'] and restored['optimizer']['param_groups'] == raw['optimizer']['param_groups']
    assert restored['objective_revision'] == objective
    metadata = recovery_metadata(restored)
    with source.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'md5').hexdigest()
    record(audit / 'source-parent.json', dict(base=str(parent_base), run_id=parent_id))
    record(audit / 'source-state-verification.json', original)
    record(audit / 'restart-state-verification.json', dict(metadata=metadata,
        source_md5=digest, tensor_values_compared=compared, model_and_adam_tensors_identical=True,
        rank_rng_identical=True, schedule_clock_preserved=True, optimizer_groups_identical=True,
        scheduler_identical=True, objective_revision=objective))
    record(audit / 'source-checkpoint-selection.json', dict(file=str(source), epoch=64,
        global_step=40064, adam_step=5634, fid=metrics['fid'], md5=digest,
        bytes=source.stat().st_size, full_optimizer_state=True))
    for name in ('entry.py', 'recipe.yaml', 'official-baseline.json'):
        shutil.copyfile(base / name, audit / name)
    (evidence / 'status.json').symlink_to('continuation-20261005/status.json')
    print(json.dumps(dict(prepared=True,run=RUN_ID,lr=initial_lr,objective=objective,
                         tensor_values_compared=compared)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'upload-recovery', 'supervise', 'train-once'])
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--key-file', type=Path, required=True)
    p.add_argument('--parent-base', type=Path)
    p.add_argument('--parent-evidence', type=Path)
    args = p.parse_args()
    if args.action == 'prepare':
        assert args.parent_base is not None and args.parent_evidence is not None
        prepare(args.base, args.evidence, args.parent_base, args.parent_evidence)
        return
    import yaml
    run_id = yaml.safe_load((args.base / 'recipe.yaml').read_text())['options']['wandb_id']
    recovery.RUN_ID = recovery.runner.RUN_ID = run_id
    recovery.RUN_PATH = recovery.runner.RUN_PATH = 'helloimlixin-rutgers/laser/' + run_id
    parent = json.loads((args.evidence / 'continuation-20261005/source-parent.json').read_text())
    recovery.PARENT, recovery.SOURCE_RUN_ID = Path(parent['base']), parent['run_id']
    if args.action == 'upload-recovery':
        recovery.upload_recovery(args.base, args.evidence, args.key_file)
    elif args.action == 'train-once':
        recovery.runner.supervise(args.base, args.evidence, args.key_file)
    else:
        from imagenet_fid_rewind_supervisor import supervise_with_rewinds
        supervise_with_rewinds(args.base, args.evidence, args.key_file, driver=Path(__file__).resolve())


if __name__ == '__main__':
    main()
