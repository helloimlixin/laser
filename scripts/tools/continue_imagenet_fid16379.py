"""Fork the complete epoch56 official FID16.379 checkpoint with lower LR."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile

import continue_imagenet_pairfix_repair as runner

SOURCE_RUN_ID = 'imagenet-rfid421-fid17459-epoch55-warmcosine-8h100-20261005'
RUN_ID = 'imagenet-rfid421-fid16379-epoch56-lowcosine-8h100-20261005'
RUN_PATH = 'helloimlixin-rutgers/laser/' + RUN_ID
PARENT = Path('/tmp/laser-imagenet-fid17-20261005-resume')
runner.RUN_ID, runner.RUN_PATH = RUN_ID, RUN_PATH
record, replace_once = runner.record, runner.replace_once


def prepare(base, evidence):
    import torch
    import yaml
    torch.set_num_threads(4)
    audit = evidence / 'continuation-20261005'
    audit.mkdir(parents=True, exist_ok=True)
    checkpoints = audit / 'train/checkpoints'
    checkpoints.mkdir(parents=True, exist_ok=True)
    if (checkpoints / 'last.pt').exists():
        raise RuntimeError('Refuse to prepare over an existing continuation')
    shutil.copytree(PARENT / 'source/runtime', base / 'source/runtime', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(PARENT / 'support', base / 'support', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__'))
    # Freeze the same tested scheduler implementation used by the local tests.
    repository = Path(__file__).resolve().parents[2]
    shutil.copyfile(repository / 'src/training/fid_adaptive_schedule.py',
                    base / 'source/runtime/src/training/fid_adaptive_schedule.py')
    shutil.copyfile(Path(__file__).with_name('imagenet_fid16379_lr.py'),
                    base / 'support/imagenet_fid16379_lr.py')
    sys.path[:0] = [str(base / 'source/runtime'), str(base / 'support')]
    from src.training import k4_checkpoint_io as checkpoint_io
    from src.training.full_resume_upload import recovery_metadata
    from imagenet_fid16379_lr import create_scheduler, BASELINE_FID
    source = base / 'inputs/source-epoch056.pt'
    raw = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    original = recovery_metadata(raw)
    assert (original['epoch'], original['global_step'], original['adam_step'],
            original['next_microbatch']) == (56, 35056, 626, 0)
    assert original['adam_parameters'] == 782 and original['rng_ranks'] == 8
    metrics = original['original_rqtransformer_metrics']
    assert metrics['fid'] == BASELINE_FID and metrics['metric_backend'] == 'original_rqtransformer'
    record(audit / 'source-state-verification.json', original)
    recipe = yaml.safe_load((PARENT / 'recipe.yaml').read_text())
    recipe['options'].update(checkpoint=str(base / 'inputs/resume-stage1-tokenizer.pt'),
        output=str(base / 'production/train'), checkpoint_dir=str(checkpoints),
        lr=1e-4, min_lr=3e-7, lr_schedule_epochs=44, warmup_epochs=0.,
        warmup_start_ratio=1., lr_schedule_restart_id=RUN_ID,
        wandb_id=RUN_ID, wandb_name='ImageNet rFID4.21 K4 | epoch56 FID16.379 | lower LR and official FID plateau decay | 8 H100')
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
    entry = replace_once(entry, 'ARGS.lr==0.0003', 'ARGS.lr==0.0001')
    entry = replace_once(entry, 'ARGS.lr_schedule_epochs==45', 'ARGS.lr_schedule_epochs==44')
    entry = replace_once(entry, 'CHECKPOINT_EPOCH=55', 'CHECKPOINT_EPOCH=56')
    entry = replace_once(entry, "'train/epoch':55,'train/global_step':34430,",
                         "'train/epoch':56,'train/global_step':35056,")
    start = entry.index(" config.update(source_run=")
    end = entry.index(" config.update(physical_accumulation_steps=", start)
    entry = entry[:start] + " config.update(training_augmentation='Resize256 RandomCrop256 RandomHorizontalFlip(p=0.5), fresh each epoch',\n  online_stage1_encoding=True,coefficient_clipping=False,stage1_rfid=4.210914134979248,\n  architecture='physical-pair-scalar-rqtransformer-imagenet-1400m',\n  full_optimizer_checkpoint_uploads=True,checkpoint_slots=['last.pt','best-fid-resume.pt','best-is-resume.pt'])\n" + entry[end:]
    start = entry.index(' config.update(resume_source_epoch=55,')
    end = entry.index(' WB=original_init(*args,**kwargs)', start)
    entry = entry[:start] + " config.update(resume_source_epoch=56,resume_source_global_step=35056,\n  source_run='helloimlixin-rutgers/laser/" + SOURCE_RUN_ID + "',\n  source_checkpoint_epoch=56,source_checkpoint_step=35056,source_checkpoint_fid=16.378750087355286,\n  source_inception_score=71.47351837158203,source_optimizer_restored=True,trained_source_optimizer_available=True,\n  optimizer_initialization='restored trained epoch56 AdamW moments and all 782 counters at626',\n  rng_initialization='restored all 8 epoch56 checkpoint RNG streams',\n  evaluation_protocol='Official RQ-Transformer Inception/FID/10-split IS only; validation50k/generated50k',\n  baseline_evaluation_provenance='same source model and fixed official evaluation seed; weights verified unchanged',\n  learning_rate_schedule='44-epoch cosine 1e-4 to3e-7; halve after3 official evaluations without FID improvement of0.05; one-evaluation cooldown',\n  lr_continuation='intentional scheduler migration only; trained Adam and RNG preserved',\n  source_scheduler_step=626,continuation_scheduler_origin_adam_step=626,\n  continuation_scheduler_origin_global_step=35056)\n" + entry[end:]
    entry = replace_once(entry, 'from imagenet_fid17_lr import create_scheduler',
                         'from imagenet_fid16379_lr import create_scheduler, observe_official_fid')
    entry = replace_once(entry,
        ' # This run restarted Adam at epoch55: global step and Adam step differ by 34430.',
        ' # Preserve the checkpoint global cursor; this continuation has a new LR clock.')
    entry = replace_once(entry,
        "  generated_images=50000,real_split='val',inception_splits=10,seed=261001)\n if dist.get_rank()==0:",
        "  generated_images=50000,real_split='val',inception_splits=10,seed=261001)\n"
        " decision=observe_official_fid(result[0])\n"
        " record(VERIFY/('lr-controller-rank'+os.environ['RANK']+'.json'),decision)\n"
        " if dist.get_rank()==0:\n"
        "  record(EVIDENCE/'continuation-20261005'/f'lr-decision-step{step}.json',decision)\n"
        "  if WB is not None and decision is not None:\n"
        "   WB.log({'train/global_step':step,'train/lr':decision['lr_after'],\n"
        "    'lr_controller/decision':decision['decision'],'lr_controller/reductions':decision['reductions']})")
    (base / 'entry.py').write_text(entry)
    (base / 'recipe.yaml').write_text(yaml.safe_dump(recipe, sort_keys=False))
    dummy = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-4)
    schedule = create_scheduler(dummy, initial_lr=1e-4, min_lr=3e-7, total_steps=27544)
    fid_alias = checkpoints / 'best_fid_16.3788_epoch_056.pt'
    is_alias = checkpoints / 'best_is_71.4735_epoch_056.pt'
    optimizer = dict(raw['optimizer'], param_groups=[dict(g, lr=1e-4, initial_lr=1e-4)
                                                       for g in raw['optimizer']['param_groups']])
    migration = dict(source_run='helloimlixin-rutgers/laser/' + SOURCE_RUN_ID,
        source_global_step=35056, source_epoch=56, source_adam_step=626,
        original_lr=original['saved_learning_rates'][0], initial_lr=1e-4,
        original_scheduler=raw['scheduler'], optimizer_moments_reset=False,
        scheduler_clock_restarted=True, schedule_steps=27544, target_epoch=100)
    payload = dict(raw, optimizer=optimizer, scheduler=schedule.state_dict(),
        config=dict(raw['config'], **recipe['options']), lr_migration=migration,
        best_fid=[(metrics['fid'], str(fid_alias))],
        best_inception=[(metrics['inception_score'], str(is_alias))])
    os.environ.update(LASER_CHECKPOINT_STAGING_DIR=str(base / 'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(base / 'checkpoint-upload-cache'),
        LASER_CHECKPOINT_IMMUTABLE_FILES='1')
    latest = checkpoints / 'last.pt'
    assert not latest.exists(), 'Refuse to overwrite a running continuation'
    checkpoint_io.atomic_torch_save(payload, latest)
    for alias in (fid_alias, is_alias, checkpoints / 'resume-anchor.pt'):
        alias.symlink_to(latest.resolve().relative_to(checkpoints))
    restored = torch.load(checkpoint_io._checkpoint_upload_source(latest),
                          map_location='cpu', weights_only=False, mmap=True)
    compared = 0
    for key, tensor in raw['state_dict'].items():
        assert torch.equal(tensor, restored['state_dict'][key]), key
        compared += tensor.numel()
    for key, state in raw['optimizer']['state'].items():
        for field, tensor in state.items():
            if torch.is_tensor(tensor):
                assert torch.equal(tensor, restored['optimizer']['state'][key][field]), (key, field)
                compared += tensor.numel()
    for rank, streams in enumerate(raw['rng_state_by_rank']):
        for key, tensor in streams.items():
            assert torch.equal(tensor, restored['rng_state_by_rank'][rank][key])
    metadata = recovery_metadata(restored)
    assert metadata['adam_step'] == 626 and restored['scheduler']['last_epoch'] == 0
    with source.open('rb') as stream:
        md5 = hashlib.file_digest(stream, 'md5').hexdigest()
    record(audit / 'restart-state-verification.json', dict(metadata=metadata,
        migration=migration, source_md5=md5, tensor_values_compared=compared,
        model_and_adam_tensors_identical=True, rank_rng_identical=True))
    for name in ('entry.py', 'recipe.yaml', 'official-baseline.json'):
        shutil.copyfile(base / name, audit / name)
    record(audit / 'source-checkpoint-selection.json', dict(file=str(source),
        epoch=56, global_step=35056, fid=metrics['fid'], bytes=source.stat().st_size,
        md5=md5, full_optimizer_state=True))
    status = evidence / 'status.json'
    if not status.exists():
        status.symlink_to('continuation-20261005/status.json')
    print(json.dumps(dict(prepared=True, epoch=56, global_step=35056, adam_step=626,
        lr=1e-4, scheduler_step=0, tensor_values_compared=compared)), flush=True)


def upload_recovery(base, evidence, key_file):
    import wandb
    import yaml
    os.environ['WANDB_API_KEY'] = key_file.read_text().strip()
    audit = evidence / 'continuation-20261005'
    options = yaml.safe_load((base / 'recipe.yaml').read_text())['options']
    verification = json.loads((audit / 'restart-state-verification.json').read_text())
    recovery = verification['metadata']
    wb = wandb.init(entity='helloimlixin-rutgers', project='laser', id=RUN_ID,
                    name=options['wandb_name'], resume='allow', mode='online',
                    config=options, allow_val_change=True, dir=str(base))
    wb.summary['execution/state'] = 'prepared'
    wb.summary['evaluation/backend'] = 'official RQ-Transformer only'
    wb.finish()
    record(audit / 'online-run-created.json', dict(run=RUN_PATH, online=True))
    directory = base / 'recovery-upload'
    directory.mkdir(exist_ok=True)
    shutil.copyfile(base / 'recipe.yaml', directory / 'resume-active-config.yaml')
    with tarfile.open(directory / 'resume-frozen-code.tar.gz', 'w:gz', compresslevel=4) as archive:
        for name in ('entry.py', 'support', 'official-baseline.json'):
            archive.add(base / name, arcname=name, filter=lambda item:None if '__pycache__' in item.name else item)
        if (base / 'ema-trial.json').exists():
            archive.add(base / 'ema-trial.json', arcname='ema-trial.json')
        if (base / 'global-cosine-trial.json').exists():
            archive.add(base / 'global-cosine-trial.json', arcname='global-cosine-trial.json')
        archive.add(base / 'source/runtime', arcname='runtime',
                    filter=lambda item:None if '__pycache__' in item.name else item)
    for name in ('resume-environment.json', 'resume-dataset-identity.json'):
        source = PARENT / 'recovery-upload' / name
        if source.resolve() != (directory / name).resolve():
            shutil.copyfile(source, directory / name)
    for name in ('source-checkpoint-selection.json', 'restart-state-verification.json'):
        shutil.copyfile(audit / name, directory / name)
    for name in ('resume-stage1-tokenizer.pt', 'resume-weights-inception-2015-12-05-6726825d.pth'):
        target = directory / name
        if not target.exists():
            os.link(base / 'inputs' / name, target)
    dependencies = []
    for file in sorted(directory.iterdir()):
        with file.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        dependencies.append(dict(file=file.name, bytes=file.stat().st_size, sha256=digest))
    record(directory / 'resume-bundle-manifest.json', dict(schema='laser-full-recovery-v1',
        run=RUN_ID, dependencies=dependencies, source_run=SOURCE_RUN_ID,
        source_epoch=recovery['epoch'], source_step=recovery['global_step'],
        source_adam_step=recovery['adam_step'], trained_optimizer_restored=True,
        initial_lr=recovery['saved_learning_rates'][0], min_lr=options['min_lr'],
        total_schedule_steps=recovery['learning_rate_schedule']['state']['policy']['total_steps'],
        official_evaluations_only=True,
        slots=(dict(latest='last.pt',raw_fid='best-raw-fid-resume.pt',raw_is='best-raw-is-resume.pt',
                    ema_fid='best-ema-fid-resume.pt',ema_is='best-ema-is-resume.pt')
               if (base / 'ema-trial.json').exists()
               else dict(latest='last.pt',fid='best-fid-resume.pt',is_score='best-is-resume.pt'))))
    sys.path.insert(0, str(base / 'support'))
    from verified_wandb_checkpoint_upload import VerifiedCloudUpload
    VerifiedCloudUpload(RUN_PATH, audit / 'cloud-recovery-bundle-receipt.json')(
        sorted(directory.iterdir()), recovery['epoch'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'upload-recovery', 'supervise'])
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--key-file', type=Path, required=True)
    args = p.parse_args()
    if args.action == 'prepare':
        prepare(args.base, args.evidence)
    elif args.action == 'upload-recovery':
        upload_recovery(args.base, args.evidence, args.key_file)
    else:
        runner.supervise(args.base, args.evidence, args.key_file)


if __name__ == '__main__':
    main()
