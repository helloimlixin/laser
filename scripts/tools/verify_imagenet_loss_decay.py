"""Verify the first durable full-state warmup continuation and its online proof."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base','evidence','key-file'):p.add_argument('--'+name,type=Path,required=True)
    args = p.parse_args()
    audit = args.evidence / 'continuation-20261005'
    proof = json.loads((audit / 'restart-state-verification.json').read_text())
    source = proof['metadata']
    step, adam = source['global_step'],source['adam_step']
    initial_lr = source['saved_learning_rates'][0]
    saved_loss_tracking = proof.get('source_train_loss_tracking') or dict(updates=0,samples=0)
    deadline = time.monotonic()+1200
    receipt = args.evidence / 'last-local-save.json'
    while not receipt.exists() or json.loads(receipt.read_text())['step'] < step+20:
        status_path = audit / 'status.json'
        if status_path.exists() and json.loads(status_path.read_text())['state']=='failed':
            raise RuntimeError('Training failed before the durable warmup verification')
        if time.monotonic()>deadline:raise RuntimeError('Twenty-update full checkpoint did not commit')
        time.sleep(10)
    status = json.loads((audit / 'status.json').read_text())
    phase = args.evidence / 'verification' / status.get('phase','continuation_20261005_'+str(status['attempt']))
    os.environ.update(CUDA_VISIBLE_DEVICES='',LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(args.base / 'checkpoint-upload-cache'),
        LASER_CHECKPOINT_IMMUTABLE_FILES='1')
    sys.path[:0] = [str(args.base / 'source/runtime'),str(args.base / 'support')]
    import torch
    torch.set_num_threads(4)
    from src.training.k4_checkpoint_io import _checkpoint_upload_source
    from src.training.full_resume_upload import recovery_metadata,atomic_json
    from imagenet_loss_decay_lr import ContinuationWarmupSchedule,create_scheduler
    groups = {}
    for prefix in ('startup','resume-rng','step1','step20','crps-step20','loss-step1','loss-step20'):
        paths = sorted(phase.glob(prefix+'-rank*.json'));assert len(paths)==8,prefix
        groups[prefix] = [json.loads(x.read_text()) for x in paths]
    for item in groups['startup']:
        assert item['adam_step']==adam and item['optimizer_parameters']==782 and item['world']==8
        assert item['lr']==initial_lr
    for item in groups['resume-rng']:
        assert item['restored_exactly'] and item['global_step']==step and item['next_microbatch']==0
    for name,updates in (('step1',1),('step20',20)):
        expected_lr = ContinuationWarmupSchedule.lr_at_step(source['learning_rate_schedule']['state']['policy'],
            source['learning_rate_schedule']['state']['last_epoch']+updates-1)
        for item in groups[name]:
            assert item['finite'] and item['adam_step']==adam+updates
            assert math.isclose(item['lr'],expected_lr,rel_tol=1e-12,abs_tol=1e-20)
    for item in groups['crps-step20']:
        assert item['finite'] and item['compiled_auxiliary_gradient_norm_on_training_logits']>0
    for name,updates in (('loss-step1',1),('loss-step20',20)):
        assert all(item==groups[name][0] for item in groups[name])
        for item in groups[name]:
            assert item['metrics']['train/loss_global_batch_samples']==2048
            assert item['state']['last_step']==step+updates
            assert item['state']['updates']==saved_loss_tracking['updates']+updates
            assert item['state']['samples']==saved_loss_tracking['samples']+2048*updates
            assert 'train/epoch_cross_entropy_mean' in item['metrics']
    native = (audit / 'train/checkpoints/last.pt').resolve(strict=True)
    local = Path(_checkpoint_upload_source(native))
    raw = torch.load(local,map_location='cpu',mmap=True,weights_only=False)
    metadata = recovery_metadata(raw)
    state = raw['scheduler']
    assert metadata['global_step']>=step+20 and metadata['adam_step']==metadata['global_step']-34430
    assert state['last_epoch']==metadata['global_step']-35056
    assert metadata['rng_ranks']==metadata['world_size']==8 and metadata['adam_parameters']==782
    assert state['policy']['min_lr']==0. and state['policy']['adaptive_reductions'] is False
    if proof.get('scheduler_identical'):
        assert state['policy']==source['learning_rate_schedule']['state']['policy']
        assert state['source_controller']==source['learning_rate_schedule']['state']['source_controller']
    else:
        assert state['source_controller']==proof['revision']['old_scheduler']
    assert state['policy']['warmup_steps']==200 and state['policy']['initial_lr']==1e-5
    assert math.isclose(metadata['saved_learning_rates'][0],ContinuationWarmupSchedule.lr_at_step(
        state['policy'],state['last_epoch']),rel_tol=1e-12,abs_tol=1e-20)
    assert raw['train_loss_tracking']['last_step']==raw['train_epoch_loss_tracking']['state']['last_step']==metadata['global_step']
    assert all(s['exp_avg'].dtype==s['exp_avg_sq'].dtype==torch.float32 for s in raw['optimizer']['state'].values())
    # A reload must use the saved warmup position rather than its initial rate.
    dummy = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))],lr=metadata['saved_learning_rates'][0])
    assert create_scheduler(dummy,initial_lr=1e-5,min_lr=0.,total_steps=state['policy']['total_steps'],
        completed_steps=state['last_epoch'],state_dict=state).state_dict()==state
    selection = json.loads((audit / 'source-checkpoint-selection.json').read_text())
    old = torch.load(selection['file'],map_location='cpu',mmap=True,weights_only=False)
    selected = sorted([(k,t) for k,t in raw['state_dict'].items() if t.is_floating_point() and t.numel()>1000],
        key=lambda x:x[1].numel(),reverse=True)[:8]
    changes = {}
    for key,tensor in selected:
        delta = tensor-old['state_dict'][key]
        changes[key] = dict(elements=tensor.numel(),changed=int(torch.count_nonzero(delta)),
                           maximum_absolute_update=float(delta.abs().max()))
    assert sum(x['changed'] for x in changes.values())>0
    report = dict(verified_at=time.time(),metadata=metadata,source_step=step,source_adam_step=adam,
        protected_source_fid=old['original_rqtransformer_metrics']['fid'],
        initial_lr=initial_lr,peak_lr=1e-5,warmup_steps=200,warmup_restarted=not proof.get('scheduler_identical',False),
        target_epoch=proof['revision']['target_epoch'],temporary_fid_regressions_permitted=True,
        full_checkpoint=str(native),checkpoint_bytes=local.stat().st_size,
        all8_finite_after20=True,all8_exact_rng_resume=True,trained_adam_fp32_preserved=True,
        original_schedule_clock_preserved=True,saved_warmup_position_reload_verified=True,
        parameter_updates_nonzero=True,sampled_weight_changes=changes,
        global_batch_loss_verified_all8=True,epoch_loss_tracking_checkpointed=True,
        loss_after1=groups['loss-step1'][0],loss_after20=groups['loss-step20'][0],official_evaluations_only=True)
    atomic_json(audit / 'production-state-verification.json',report)
    os.environ['WANDB_API_KEY']=args.key_file.read_text().strip()
    import wandb
    from verified_wandb_checkpoint_upload import VerifiedCloudUpload
    from continue_imagenet_loss_decay import RUN_PATH
    run=wandb.Api(timeout=120).run(RUN_PATH)
    assert run.config['automatic_fid_rewind'] is False and run.config['lr']==1e-5
    files=[audit/'production-state-verification.json',audit/'restart-state-verification.json',
        audit/'resume-tests.txt',audit/'verify_imagenet_loss_decay.py']
    VerifiedCloudUpload(RUN_PATH,audit/'production-verification-cloud-receipt.json')(files,metadata['epoch'])
    run.summary.update({'verification/all8_finite_after20':True,'verification/trained_adam_preserved':True,
        'verification/full_checkpoint_committed':True,'verification/warmup_resume_position':True,
        'verification/global_and_epoch_loss_means':True,'verification/parameter_updates_nonzero':True})
    print(json.dumps(dict(verified=True,source_fid=report['protected_source_fid'],persisted_step=metadata['global_step'],
        persisted_adam=metadata['adam_step'],initial_lr=initial_lr,peak_lr=1e-5,all8_finite=True,
        loss_tracking_verified=True,cloud_proof_verified=True)),flush=True)


if __name__=='__main__':main()
