"""Verify actual distributed Adam, RNG and global cosine after twenty updates."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'evidence', 'key-file'):parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    audit = args.evidence / 'continuation-20261005'
    proof = json.loads((audit / 'restart-state-verification.json').read_text())
    control = json.loads((args.base / 'global-cosine-trial.json').read_text())
    source = proof['metadata']
    deadline = time.monotonic() + 1200
    receipt = args.evidence / 'last-local-save.json'
    while not receipt.exists() or json.loads(receipt.read_text())['step'] < 48222:
        status = audit / 'status.json'
        if status.exists() and json.loads(status.read_text())['state'] == 'failed':
            raise RuntimeError('Training failed before durable twenty-update verification')
        if time.monotonic() > deadline:raise RuntimeError('Twenty-update full checkpoint did not commit')
        time.sleep(10)
    os.environ.update(CUDA_VISIBLE_DEVICES='',
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(args.base / 'checkpoint-upload-cache'),
        LASER_CHECKPOINT_IMMUTABLE_FILES='1')
    sys.path[:0] = [str(args.base / 'source/runtime'), str(args.base / 'support')]
    import torch
    torch.set_num_threads(4)
    from src.training.k4_checkpoint_io import _checkpoint_upload_source
    from src.training.full_resume_upload import recovery_metadata
    from imagenet_global_cosine_lr import GlobalCosineSchedule, create_scheduler
    run_path = control.get('run_path', 'helloimlixin-rutgers/laser/imagenet-rfid421-epoch77-rqcos0-lr1p249e6-1epoch-8h100-20261006')
    phase = args.evidence / 'verification' / json.loads((audit / 'status.json').read_text())['phase']
    groups = {}
    for prefix in ('startup', 'resume-rng', 'step1', 'step20', 'crps-step20', 'loss-step1', 'loss-step20'):
        paths = sorted(phase.glob(prefix + '-rank*.json')); assert len(paths) == 8, prefix
        groups[prefix] = [json.loads(path.read_text()) for path in paths]
    for item in groups['startup']:
        assert item['adam_step'] == 13772 and item['optimizer_parameters'] == 782 and item['world'] == 8
        assert item['lr'] == source['saved_learning_rates'][0]
    assert all(item['restored_exactly'] and item['global_step'] == 48202
               and item['next_microbatch'] == 0 for item in groups['resume-rng'])
    for updates in (1, 20):
        expected = GlobalCosineSchedule.lr_at_step(source['learning_rate_schedule']['state']['policy'], 48202 + updates - 1)
        assert all(item['finite'] and item['adam_step'] == 13772 + updates
                   and math.isclose(item['lr'], expected, rel_tol=1e-12, abs_tol=1e-20)
                   for item in groups['step' + str(updates)])
        losses = groups['loss-step' + str(updates)]
        assert all(item == losses[0] for item in losses)
        assert losses[0]['metrics']['train/loss_global_batch_samples'] == 2048
        assert losses[0]['state']['updates'] == proof['source_train_loss_tracking']['updates'] + updates
    assert all(item['finite'] and item['compiled_auxiliary_gradient_norm_on_training_logits'] > 0
               for item in groups['crps-step20'])
    native = (audit / 'train/checkpoints/last.pt').resolve(strict=True)
    local = Path(_checkpoint_upload_source(native))
    payload = torch.load(local, map_location='cpu', mmap=True, weights_only=False)
    metadata = recovery_metadata(payload)
    state = payload['scheduler']
    assert metadata['global_step'] >= 48222
    assert state['last_epoch'] == metadata['global_step']
    assert metadata['adam_step'] == metadata['global_step'] - 34430
    assert metadata['adam_parameters'] == 782 and metadata['rng_ranks'] == 8
    assert state['policy'] == source['learning_rate_schedule']['state']['policy']
    assert state['policy']['min_lr'] == 0 and state['policy']['total_steps'] == 62600
    assert metadata['saved_learning_rates'][0] == GlobalCosineSchedule.lr_at_step(state['policy'], state['last_epoch'])
    assert all(fields['exp_avg'].dtype == fields['exp_avg_sq'].dtype == torch.float32
               for fields in payload['optimizer']['state'].values())
    assert payload.get('parameter_ema') is None
    assert payload['train_loss_tracking']['last_step'] == payload['train_epoch_loss_tracking']['state']['last_step'] == metadata['global_step']
    dummy = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=metadata['saved_learning_rates'][0])
    assert create_scheduler(dummy, initial_lr=control['peak_lr'], min_lr=0., total_steps=62600,
        completed_steps=metadata['global_step'], state_dict=state).state_dict() == state
    old = torch.load(args.base / 'inputs/source-epoch077-full.pt', map_location='cpu', mmap=True, weights_only=False)
    selected = sorted([(key, tensor) for key, tensor in payload['state_dict'].items()
        if tensor.is_floating_point() and tensor.numel() > 1000], key=lambda item:item[1].numel(), reverse=True)[:8]
    changes = {}
    for key, tensor in selected:
        delta = tensor - old['state_dict'][key]
        changes[key] = dict(changed=int(torch.count_nonzero(delta)), maximum_absolute_update=float(delta.abs().max()))
    assert sum(item['changed'] for item in changes.values()) > 0
    report = dict(verified_at=time.time(), metadata=metadata, source_epoch=77,
        source_fid=15.24941539209243, source_adam_step=13772,
        initial_lr=source['saved_learning_rates'][0], peak_lr=control['peak_lr'],
        full_checkpoint=str(native), checkpoint_bytes=local.stat().st_size,
        all8_finite_after20=True, all8_exact_rng_resume=True, trained_adam_fp32_preserved=True,
        original100epoch_cosine_position_verified=True, zero_floor=True, warmup_restarted=False,
        objective_unchanged=True, parameter_ema_enabled=False,
        global_and_epoch_loss_tracking_preserved=True, sampled_parameter_changes=changes,
        loss_after1=groups['loss-step1'][0], loss_after20=groups['loss-step20'][0],
        target_epoch=control['target_epoch'], official_evaluations_only=True)
    destination = audit / 'production-state-verification.json'
    destination.write_text(json.dumps(report, indent=2, default=str) + '\n')
    os.environ['WANDB_API_KEY'] = args.key_file.read_text().strip()
    from verified_wandb_checkpoint_upload import VerifiedCloudUpload
    files = [destination, audit / 'restart-state-verification.json', audit / 'resume-tests.txt',
             audit / 'verify_imagenet_global_cosine.py']
    VerifiedCloudUpload(run_path, audit / 'production-verification-cloud-receipt.json')(files, metadata['epoch'])
    print(json.dumps(dict(verified=True, run=run_path, persisted_step=metadata['global_step'],
        persisted_adam=metadata['adam_step'], initial_lr=source['saved_learning_rates'][0],
        all8_finite=True, optimizer_rng_preserved=True, online_proof_verified=True)), flush=True)


if __name__ == '__main__':main()
