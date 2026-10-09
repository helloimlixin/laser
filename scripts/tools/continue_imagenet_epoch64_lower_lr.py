"""Rewind epoch64 with trained Adam intact and an explicit lower LR at zero floor."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import continue_imagenet_fid16379 as recovery

SOURCE_RUN_ID = 'imagenet-rfid421-epoch64-floor0-8h100-20261005'
RUN_ID = 'imagenet-rfid421-epoch64-floor0-lowlr-8h100-20261005b'
RUN_PATH = 'helloimlixin-rutgers/laser/' + RUN_ID
PARENT = Path('/tmp/laser-imagenet-epoch64-floor0-20261005-resume')
recovery.RUN_ID, recovery.RUN_PATH = RUN_ID, RUN_PATH
recovery.SOURCE_RUN_ID, recovery.PARENT = SOURCE_RUN_ID, PARENT
recovery.runner.RUN_ID, recovery.runner.RUN_PATH = RUN_ID, RUN_PATH
record, replace_once = recovery.record, recovery.replace_once


def prepare(base, evidence):
    import torch
    import yaml
    torch.set_num_threads(4)
    audit = evidence / 'continuation-20261005'
    checkpoints = audit / 'train/checkpoints'
    checkpoints.mkdir(parents=True, exist_ok=True)
    latest = checkpoints / 'last.pt'
    if latest.exists():
        raise RuntimeError('Refuse to prepare over an existing continuation')
    for directory in ('source/runtime', 'support'):
        shutil.copytree(PARENT / directory, base / directory, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__'))
    for module in ('imagenet_lower_floor_lr.py', 'imagenet_zero_floor_lr.py', 'imagenet_epoch64_lower_lr.py'):
        shutil.copyfile(Path(__file__).with_name(module), base / 'support' / module)
    shutil.copyfile(Path(__file__).resolve().parents[2] / 'src/training/fid_adaptive_schedule.py',
                    base / 'source/runtime/src/training/fid_adaptive_schedule.py')
    sys.path[:0] = [str(base / 'source/runtime'), str(base / 'support')]
    from src.training import k4_checkpoint_io as checkpoint_io
    from src.training.full_resume_upload import recovery_metadata
    from imagenet_zero_floor_lr import create_scheduler
    from imagenet_epoch64_lower_lr import lower_rewind_lr, INITIAL_LR
    source = base / 'inputs/source-epoch064.pt'
    raw = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    original = recovery_metadata(raw)
    assert (original['epoch'], original['global_step'], original['adam_step'],
            original['next_microbatch']) == (64, 40064, 5634, 0)
    assert original['rng_ranks'] == 8 and original['adam_parameters'] == 782
    metrics = original['original_rqtransformer_metrics']
    assert metrics['fid'] == 15.554077729782705 and metrics['metric_backend'] == 'original_rqtransformer'
    assert raw['scheduler']['last_epoch'] == raw['scheduler']['last_observation_step'] == 5008
    assert raw['scheduler']['last_fid'] == metrics['fid']
    # Preserve the earlier accepted controller best alongside the raw winner.
    assert raw['scheduler']['best'] == 15.556776106764232
    assert raw['scheduler']['policy']['min_lr'] == 0.
    assert raw['scheduler']['policy']['decay_start_step'] == 5008 and raw['scheduler']['policy']['decay_steps'] == 22536
    optimizer, state = lower_rewind_lr(raw['optimizer'], raw['scheduler'], initial_lr=INITIAL_LR)
    initial_lr = optimizer['param_groups'][0]['lr']
    fake = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=initial_lr)
    scheduler = create_scheduler(fake, initial_lr=1e-4, min_lr=0., total_steps=27544,
                                 completed_steps=5008, state_dict=state)
    state = scheduler.state_dict()
    for key in ('last_epoch', 'best', 'last_observation_step', 'last_fid', 'reductions', 'bad_epochs', 'cooldown_remaining'):
        assert state[key] == raw['scheduler'][key]
    assert state['policy'] == raw['scheduler']['policy']
    assert state['multiplier'] < raw['scheduler']['multiplier']
    assert state['policy']['patience'] == 1 and state['policy']['cooldown'] == 0
    assert state['policy']['min_delta'] == .02 and state['policy']['min_lr'] == 0.
    assert state['policy']['decay_start_step'] == 5008 and state['policy']['decay_steps'] == 22536
    record(audit / 'source-state-verification.json', original)
    recipe = yaml.safe_load((PARENT / 'recipe.yaml').read_text())
    recipe['options'].update(checkpoint=str(base / 'inputs/resume-stage1-tokenizer.pt'),
        output=str(base / 'production/train'), checkpoint_dir=str(checkpoints),
        lr_schedule_restart_id=RUN_ID, wandb_id=RUN_ID, fid_every=1, min_lr=0.,
        wandb_name='ImageNet rFID4.21 K4 | epoch64 FID15.554 | startLR1.767e-7, floor0, official every epoch | 8 H100')
    recipe['options'].pop('resume_checkpoint', None)
    for name in ('resume-stage1-tokenizer.pt', 'resume-weights-inception-2015-12-05-6726825d.pth'):
        target = base / 'inputs' / name
        if not target.exists():
            os.link(PARENT / 'inputs' / name, target)
    for name in ('torch-cache', 'inductor-cache'):
        if not (base / name).exists():
            (base / name).symlink_to(PARENT / name, target_is_directory=True)
    (base / 'official-baseline.json').write_text(json.dumps(metrics, indent=2) + '\n')
    entry = (PARENT / 'entry.py').read_text()
    assert 'from imagenet_zero_floor_lr import create_scheduler, observe_official_fid' in entry
    start = entry.index(' config.update(resume_source_epoch=64,')
    end = entry.index(' WB=original_init(*args,**kwargs)', start)
    description = ('Saved clock5008/27544 and trained Adam retained; startLR1.7674647817084015e-7; floor0; '
                   'continuation cosine5008..27544 reaches zero at epoch100; '
                   'halve LR after each official evaluation without FID improvement of0.02; '
                   'no cooldown; report at_floor only when the actual LR cannot decrease')
    entry = entry[:start] + (
        " config.update(resume_source_epoch=64,resume_source_global_step=40064,\n"
        f"  source_run='helloimlixin-rutgers/laser/{SOURCE_RUN_ID}',\n"
        "  source_checkpoint_epoch=64,source_checkpoint_step=40064,source_checkpoint_fid=15.554077729782705,\n"
        "  source_inception_score=73.79450225830078,source_optimizer_restored=True,trained_source_optimizer_available=True,\n"
        "  optimizer_initialization='restored trained epoch64 AdamW moments and all 782 counters at5634',\n"
        "  rng_initialization='restored all 8 epoch64 checkpoint RNG streams',\n"
        "  evaluation_protocol='Official RQ-Transformer Inception/FID/10-split IS only; validation50k/generated50k',\n"
        "  baseline_evaluation_provenance='same source model and fixed official evaluation seed; weights verified unchanged',\n"
        f"  learning_rate_schedule={description!r},\n"
        "  lr_continuation='saved clock and FID controller history preserved; LR amplitude reduced once at rewind; zero floor and endpoint100 preserved; trained Adam preserved',\n"
        "  source_scheduler_step=5008,continuation_scheduler_origin_adam_step=626,\n"
        "  continuation_scheduler_origin_global_step=35056,\n"
        f"  restart_initial_lr={initial_lr!r},lr_floor_before=0.,lr_floor_after=0.,manual_lower_lr_rewind=True,previous_run_epoch67_lr=3.534929563416803e-7,manual_lr_fraction_of_previous_run=.5,cosine_decay_epochs=36,cosine_decay_end_epoch=100)\n"
    ) + entry[end:]
    (base / 'entry.py').write_text(entry)
    (base / 'recipe.yaml').write_text(yaml.safe_dump(recipe, sort_keys=False))
    fid_alias = checkpoints / 'best_fid_15.5541_epoch_064.pt'
    is_alias = checkpoints / 'best_is_73.7945_epoch_064.pt'
    revision = dict(source_run=SOURCE_RUN_ID, source_epoch=64, source_global_step=40064,
        source_adam_step=5634, original_lr=original['saved_learning_rates'][0], initial_lr=initial_lr,
        old_scheduler=raw['scheduler'], new_scheduler=state, optimizer_moments_reset=False,
        schedule_clock_preserved=True, schedule_origin_global_step=35056,
        schedule_origin_adam_step=626, remaining_schedule_steps=22536, target_epoch=100,
        decay_epochs=36, decay_end_epoch=100, decay_steps=22536, old_floor=0., new_floor=0.,
        manual_lr_revision=True, previous_run_epoch=67, previous_run_lr=3.534929563416803e-7,
        fraction_of_previous_run_lr=.5, source_multiplier=raw['scheduler']['multiplier'],
        revised_multiplier=state['multiplier'])
    payload = dict(raw, optimizer=optimizer, scheduler=state,
        config=dict(raw['config'], **recipe['options']), lr_revision=revision,
        best_fid=[(metrics['fid'], str(fid_alias))],
        best_inception=[(metrics['inception_score'], str(is_alias))])
    os.environ.update(LASER_CHECKPOINT_STAGING_DIR=str(base / 'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(base / 'checkpoint-upload-cache'),
        LASER_CHECKPOINT_IMMUTABLE_FILES='1')
    checkpoint_io.atomic_torch_save(payload, latest)
    for alias in (fid_alias, is_alias, checkpoints / 'resume-anchor.pt'):
        alias.symlink_to(latest.resolve().relative_to(checkpoints))
    restored = torch.load(checkpoint_io._checkpoint_upload_source(latest),
                          map_location='cpu', weights_only=False, mmap=True)
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
    assert restored['scheduler'] == state
    metadata = recovery_metadata(restored)
    assert metadata['adam_step'] == 5634 and metadata['saved_learning_rates'] == [initial_lr]
    with source.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'md5').hexdigest()
    record(audit / 'restart-state-verification.json', dict(metadata=metadata, revision=revision,
        source_md5=digest, tensor_values_compared=compared, model_and_adam_tensors_identical=True,
        rank_rng_identical=True, schedule_clock_preserved=True))
    record(audit / 'source-checkpoint-selection.json', dict(file=str(source), epoch=64,
        global_step=40064, adam_step=5634, fid=metrics['fid'], md5=digest,
        bytes=source.stat().st_size, full_optimizer_state=True))
    for name in ('entry.py', 'recipe.yaml', 'official-baseline.json'):
        shutil.copyfile(base / name, audit / name)
    if not (evidence / 'status.json').exists():
        (evidence / 'status.json').symlink_to('continuation-20261005/status.json')
    print(json.dumps(dict(prepared=True, epoch=64, global_step=40064, adam_step=5634,
        scheduler_step=5008, lr=initial_lr, tensor_values_compared=compared)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'upload-recovery', 'supervise', 'train-once'])
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--key-file', type=Path, required=True)
    args = p.parse_args()
    if args.action == 'prepare':
        prepare(args.base, args.evidence)
    else:
        import yaml
        options = yaml.safe_load((args.base / 'recipe.yaml').read_text())['options']
        run_id = options['wandb_id']
        recovery.RUN_ID = recovery.runner.RUN_ID = run_id
        recovery.RUN_PATH = recovery.runner.RUN_PATH = 'helloimlixin-rutgers/laser/' + run_id
        parent_file = args.evidence / 'continuation-20261005/source-parent.json'
        if parent_file.exists():
            parent = json.loads(parent_file.read_text())
            recovery.PARENT = Path(parent['base'])
            recovery.SOURCE_RUN_ID = parent['run_id']
        if args.action == 'upload-recovery':
            recovery.upload_recovery(args.base, args.evidence, args.key_file)
        elif args.action == 'train-once':
            recovery.runner.supervise(args.base, args.evidence, args.key_file)
        else:
            from imagenet_fid_rewind_supervisor import supervise_with_rewinds
            supervise_with_rewinds(args.base, args.evidence, args.key_file,
                                   driver=Path(__file__).resolve())


if __name__ == '__main__':
    main()
