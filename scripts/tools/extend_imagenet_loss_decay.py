"""Extend the stopped loss-decay run in place, preserving the full saved state."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

from continue_imagenet_loss_decay import RUN_ID, record


def extend(base, evidence):
    import torch
    import yaml
    torch.set_num_threads(4)
    audit = evidence / 'continuation-20261005'
    marker = base / 'continuation-extension.json'
    assert not marker.exists(), 'Continuation is already configured'
    previous_status = json.loads((audit / 'status.json').read_text())
    assert previous_status['state'] == 'stopped_resumable'
    cloud = json.loads((evidence / 'final-cloud-checkpoint-receipt.json').read_text())
    assert cloud['complete'], 'Wait for the stopped run checkpoint uploads to finish'
    archive = audit / 'completed-epoch068-trial'
    archive.mkdir(exist_ok=False)
    for path in audit.iterdir():
        if path.is_file():shutil.copyfile(path,archive / path.name)
    for name in ('recipe.yaml','entry.py','official-baseline.json'):
        shutil.copyfile(base / name,archive / name)
    for name in ('trial-final-state-verification.json','trial-checkpoint-ready.json','final-cloud-checkpoint-receipt.json','loss-decay-trial-status.json'):
        shutil.copyfile(evidence / name,archive / name)
    os.environ.update(CUDA_VISIBLE_DEVICES='',LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(base / 'checkpoint-upload-cache'),
                      LASER_CHECKPOINT_IMMUTABLE_FILES='1')
    sys.path[:0] = [str(base / 'source/runtime'),str(base / 'support')]
    from src.training.k4_checkpoint_io import _checkpoint_upload_source
    from src.training.full_resume_upload import recovery_metadata
    from imagenet_loss_decay_lr import create_scheduler,ContinuationWarmupSchedule
    checkpoints = audit / 'train/checkpoints'
    native = (checkpoints / 'last.pt').resolve(strict=True)
    local = Path(_checkpoint_upload_source(native))
    assert os.path.samefile(local,base / 'final-cpu-upload/last.pt')
    assert os.path.samefile(local,base / 'final-cpu-upload/best-fid-resume.pt')
    source = base / 'inputs/source-epoch068-full.pt'
    os.link(local,source)
    anchor = checkpoints / 'resume-epoch068-anchor.pt'
    anchor.symlink_to(native.relative_to(checkpoints))
    raw = torch.load(source,map_location='cpu',mmap=True,weights_only=False)
    metadata = recovery_metadata(raw)
    assert (metadata['epoch'],metadata['global_step'],metadata['adam_step'],metadata['next_microbatch']) == (68,42568,8138,0)
    assert metadata['rng_ranks'] == metadata['world_size'] == 8 and metadata['adam_parameters'] == 782
    state = raw['scheduler']
    assert state['last_epoch'] == 7512 and state['policy']['total_steps'] == 27544
    initial_lr = metadata['saved_learning_rates'][0]
    assert initial_lr == ContinuationWarmupSchedule.lr_at_step(state['policy'],state['last_epoch'])
    dummy = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))],lr=initial_lr)
    assert create_scheduler(dummy,initial_lr=1e-5,min_lr=0.,total_steps=27544,
        completed_steps=7512,state_dict=state).state_dict() == state
    with source.open('rb') as f:digest = hashlib.file_digest(f,'md5').digest()
    online = next(x for x in cloud['files'] if x['name']=='last.pt')
    assert base64.b64encode(digest).decode() == online['md5'] and source.stat().st_size == online['bytes']
    recipe = yaml.safe_load((base / 'recipe.yaml').read_text())
    recipe['options'].update(epochs=100,max_optimizer_steps=0,
        wandb_name='ImageNet K4 | continue epoch68 FID15.5148 | saved Adam and zero-floor cosine to100 | 8H100')
    assert recipe['options']['wandb_id'] == recipe['options']['lr_schedule_restart_id'] == RUN_ID
    code = (base / 'entry.py').read_text()
    old = (" config.update(automatic_fid_rewind=False,fid_rewind_policy='bounded three-epoch sustained trial; temporary regressions retained; protected full best kept',\n"
           "  trial_target_epoch=68,trial_epochs=3,trial_lr_warmup_updates=200,trial_peak_lr=1e-5)\n")
    new = (" config.update(automatic_fid_rewind=False,fid_rewind_policy='sustained continuation; protected full best retained; official FID monitored',\n"
           "  bounded_trial=False,trial_target_epoch=None,trial_epochs=0,continuation_target_epoch=100,\n"
           "  warmup_restarted=False,continuation_peak_lr=1e-5)\n")
    assert code.count(old)==1
    code = code.replace(old,new)
    start = code.index(' config.update(resume_source_epoch=')
    end = code.index(' WB=original_init(*args,**kwargs)',start)
    metrics = metadata['original_rqtransformer_metrics']
    assert metrics['fid'] == 15.514826508438034 and metrics['global_step'] == 42568
    code = code[:start] + (
        " config.update(resume_source_epoch=68,resume_source_global_step=42568,\n"
        f"  source_run={'helloimlixin-rutgers/laser/'+RUN_ID!r},source_checkpoint_epoch=68,\n"
        "  source_checkpoint_step=42568,source_checkpoint_fid=15.514826508438034,\n"
        "  source_optimizer_restored=True,trained_source_optimizer_available=True,\n"
        "  optimizer_initialization='trained AdamW moments and all782 counters preserved at8138',\n"
        "  rng_initialization='all8 saved CPU/CUDA RNG streams restored',\n"
        "  evaluation_protocol='Official RQ-Transformer FID/10-split IS only; validation50k/generated50k',\n"
        "  learning_rate_schedule='saved cosine to zero at epoch100; completed warmup retained; official FID monitored',\n"
        "  lr_continuation='exact saved scheduler and optimizer; extension of training endpoint only',\n"
        f"  source_scheduler_step=7512,restart_initial_lr={initial_lr!r},lr_floor_after=0.,cosine_decay_end_epoch=100)\n"
    ) + code[end:]
    code,count = re.subn(r'(?m)^CHECKPOINT_EPOCH=65$', 'CHECKPOINT_EPOCH=68',code);assert count==1
    old = "WB.log({'train/epoch':65,'train/global_step':40690,"
    assert code.count(old)==1
    code = code.replace(old,"WB.log({'train/epoch':68,'train/global_step':42568,")
    old = "record(EVIDENCE/'trial-checkpoint-ready.json'"
    assert code.count(old)==1
    code = code.replace(old,"record(EVIDENCE/'continuation-checkpoint-ready.json'")
    compile(code,str(base / 'entry.py'),'exec')
    (base / 'entry.py').write_text(code)
    (base / 'recipe.yaml').write_text(yaml.safe_dump(recipe,sort_keys=False))
    record(base / 'official-baseline.json',metrics)
    revision = dict(source_run=RUN_ID,source_epoch=68,source_global_step=42568,
        original_lr=initial_lr,initial_lr=initial_lr,peak_lr=1e-5,target_epoch=100,
        old_scheduler=state,new_scheduler=state,schedule_clock_preserved=True,
        optimizer_moments_reset=False,warmup_restarted=False,official_evaluations_only=True,
        temporary_fid_regressions_permitted=True)
    record(audit / 'source-parent.json',dict(base=str(base),run_id=RUN_ID))
    record(audit / 'source-state-verification.json',metadata)
    record(audit / 'restart-state-verification.json',dict(metadata=metadata,source_md5=digest.hex(),
        model_and_adam_tensors_identical=True,rank_rng_identical=True,schedule_clock_preserved=True,
        scheduler_identical=True,optimizer_groups_identical=True,immutable_checkpoint_identity_verified=True,
        cloud_checkpoint_checksum_verified=True,source_train_loss_tracking=raw['train_loss_tracking'],revision=revision))
    record(audit / 'source-checkpoint-selection.json',dict(file=str(source),epoch=68,global_step=42568,
        adam_step=8138,fid=metrics['fid'],md5=digest.hex(),bytes=source.stat().st_size,full_optimizer_state=True))
    for name in ('entry.py','recipe.yaml','official-baseline.json'):
        shutil.copyfile(base / name,audit / name)
    repository = Path(__file__).resolve().parents[2]
    for name in ('continue_imagenet_pairfix_repair.py','continue_imagenet_fid16379.py',
                 'continue_imagenet_loss_decay.py','verify_imagenet_loss_decay.py',
                 'drain_imagenet_full_checkpoints.py','extend_imagenet_loss_decay.py'):
        shutil.copyfile(repository / 'scripts/tools' / name,base / 'support' / name)
    shutil.copyfile(repository / 'scripts/tools/verify_imagenet_loss_decay.py',audit / 'verify_imagenet_loss_decay.py')
    (audit / 'online-run-created.json').unlink()
    record(marker,dict(source_epoch=68,source_step=42568,target_epoch=100,initial_lr=initial_lr,
        restart_warmup=False,phase_prefix='continuation_20261006_epoch68',prepared_at=time.time()))
    print(json.dumps(dict(prepared=True,same_run=RUN_ID,source_epoch=68,source_fid=metrics['fid'],
        source_adam_step=8138,source_scheduler_step=7512,initial_lr=initial_lr,target_epoch=100,
        full_optimizer_and_scheduler_preserved=True,warmup_restarted=False)),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--evidence',type=Path,required=True)
    args = p.parse_args()
    extend(args.base,args.evidence)


if __name__=='__main__':main()
