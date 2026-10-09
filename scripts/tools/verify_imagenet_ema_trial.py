"""Verify durable raw/Adam/EMA recovery after twenty real distributed updates."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'evidence', 'key-file'):p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    audit = args.evidence / 'continuation-20261005'
    proof = json.loads((audit / 'restart-state-verification.json').read_text())
    source = proof['metadata']
    source_step = source['global_step']
    deadline = time.monotonic() + 1200
    receipt = args.evidence / 'last-local-save.json'
    while not receipt.exists() or json.loads(receipt.read_text())['step'] < source_step + 20:
        status_path = audit / 'status.json'
        if status_path.exists() and json.loads(status_path.read_text())['state'] == 'failed':
            raise RuntimeError('EMA training failed before the twenty-update checkpoint')
        if time.monotonic() > deadline:raise RuntimeError('Durable twenty-update EMA checkpoint did not commit')
        time.sleep(10)
    os.environ.update(CUDA_VISIBLE_DEVICES='', LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(args.base / 'checkpoint-upload-cache'),
                      LASER_CHECKPOINT_IMMUTABLE_FILES='1')
    sys.path[:0] = [str(args.base / 'source/runtime'), str(args.base / 'support')]
    import torch
    torch.set_num_threads(4)
    from src.training.k4_checkpoint_io import _checkpoint_upload_source
    from src.training.full_resume_upload import recovery_metadata
    from imagenet_loss_decay_lr import ContinuationWarmupSchedule, create_scheduler
    from continue_imagenet_ema_trial import RUN_PATH
    phase = args.evidence / 'verification' / json.loads((audit / 'status.json').read_text())['phase']
    groups = {}
    for prefix in ('startup', 'resume-rng', 'step1', 'step20', 'ema-startup', 'ema-step1', 'ema-step20',
                   'crps-step20', 'loss-step1', 'loss-step20'):
        paths = sorted(phase.glob(prefix + '-rank*.json')); assert len(paths) == 8, prefix
        groups[prefix] = [json.loads(x.read_text()) for x in paths]
    for item in groups['startup']:
        assert item['adam_step'] == 13772 and item['optimizer_parameters'] == 782
        assert item['lr'] == source['saved_learning_rates'][0] and item['world'] == 8
    assert all(x['restored_exactly'] and x['global_step'] == 48202 for x in groups['resume-rng'])
    for item in groups['ema-startup']:
        assert item['initialized_from_raw_exactly'] and item['updates'] == 0 and item['decay'] == .999
        assert item['parameter_tensors'] == 782 and item['parameter_elements'] == 1391864832
    for updates in (1, 20):
        expected_lr = ContinuationWarmupSchedule.lr_at_step(source['learning_rate_schedule']['state']['policy'],
            source['learning_rate_schedule']['state']['last_epoch'] + updates - 1)
        for item in groups['step' + str(updates)]:
            assert item['finite'] and item['adam_step'] == 13772 + updates
            assert math.isclose(item['lr'], expected_lr, rel_tol=1e-12, abs_tol=1e-20)
        for item in groups['ema-step' + str(updates)]:
            assert item['finite'] and item['updates'] == updates and item['updated_after_adam']
        losses = groups['loss-step' + str(updates)]
        assert all(item == losses[0] for item in losses)
        assert losses[0]['metrics']['train/loss_global_batch_samples'] == 2048
        assert losses[0]['state']['updates'] == proof['source_train_loss_tracking']['updates'] + updates
    assert all(x['finite'] and x['compiled_auxiliary_gradient_norm_on_training_logits'] > 0
               for x in groups['crps-step20'])
    native = (audit / 'train/checkpoints/last.pt').resolve(strict=True)
    local = Path(_checkpoint_upload_source(native))
    payload = torch.load(local, map_location='cpu', mmap=True, weights_only=False)
    metadata = recovery_metadata(payload)
    assert metadata['global_step'] >= 48222 and metadata['adam_step'] == metadata['global_step'] - 34430
    assert payload['scheduler']['last_epoch'] == metadata['global_step'] - 35056
    assert payload['scheduler']['policy'] == source['learning_rate_schedule']['state']['policy']
    assert payload['parameter_ema']['updates'] == metadata['global_step'] - 48202
    assert payload['parameter_ema']['decay'] == .999 and metadata['rng_ranks'] == 8
    assert payload['training_state_weights'] == payload['optimizer_state_weights'] == 'raw'
    assert all(s['exp_avg'].dtype == s['exp_avg_sq'].dtype == torch.float32
               for s in payload['optimizer']['state'].values())
    dummy = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=metadata['saved_learning_rates'][0])
    assert create_scheduler(dummy, initial_lr=1e-4, min_lr=0., total_steps=27544,
        completed_steps=payload['scheduler']['last_epoch'], state_dict=payload['scheduler']).state_dict() == payload['scheduler']
    old = torch.load(args.base / 'inputs/source-epoch077-full.pt', map_location='cpu', mmap=True, weights_only=False)
    selected = sorted(payload['parameter_ema']['values'],
                      key=lambda k: payload['parameter_ema']['values'][k].numel(), reverse=True)[:8]
    changes = {}
    for key in selected:
        raw, ema, initial = payload['state_dict'][key], payload['parameter_ema']['values'][key], old['state_dict'][key]
        changes[key] = dict(raw_changed=int(torch.count_nonzero(raw - initial)),
                            ema_changed=int(torch.count_nonzero(ema - initial)),
                            raw_differs_from_ema=int(torch.count_nonzero(raw - ema)))
    assert all(any(c[field] > 0 for c in changes.values())
               for field in ('raw_changed', 'ema_changed', 'raw_differs_from_ema'))
    report = dict(verified_at=time.time(), source_epoch=77, source_fid=15.24941539209243,
        source_adam_step=13772, initial_lr=source['saved_learning_rates'][0], learning_rate_factor=10.,
        metadata=metadata, checkpoint_bytes=local.stat().st_size,
        all8_exact_rng_resume=True, all8_raw_and_ema_finite_after20=True,
        trained_adam_fp32_preserved=True, saved_cosine_clock_preserved=True,
        ema_initialized_from_source_exactly=True, ema_updated_after_real_adam=True,
        raw_and_ema_saved_separately=True, saved_ema_update_clock_verified=True,
        global_and_epoch_loss_tracking_preserved=True, sampled_parameter_changes=changes,
        target_epoch=82, official_evaluations_only=True)
    destination = audit / 'production-state-verification.json'
    destination.write_text(json.dumps(report, indent=2, default=str) + '\n')
    os.environ['WANDB_API_KEY'] = args.key_file.read_text().strip()
    import wandb
    from verified_wandb_checkpoint_upload import VerifiedCloudUpload
    files = [destination, audit / 'restart-state-verification.json', audit / 'resume-tests.txt',
             audit / 'verify_imagenet_ema_trial.py']
    VerifiedCloudUpload(RUN_PATH, audit / 'production-verification-cloud-receipt.json')(files, metadata['epoch'])
    run = wandb.Api(timeout=120).run(RUN_PATH)
    run.summary.update({'verification/raw_and_ema_full_checkpoint_committed':True,
        'verification/all8_raw_and_ema_finite_after20':True,
        'verification/trained_adam_and_rng_preserved':True,
        'verification/tenfold_lr_without_clock_reset':True})
    print(json.dumps(dict(verified=True, run=RUN_PATH, persisted_step=metadata['global_step'],
        initial_lr=source['saved_learning_rates'][0], checkpoint_bytes=local.stat().st_size,
        ema_updates=payload['parameter_ema']['updates'], online_proof_verified=True)), flush=True)


if __name__ == '__main__':main()
