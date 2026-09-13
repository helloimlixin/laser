#!/usr/bin/env python3
"""Fork optimizer-preserving matched codec continuations from the finished pair."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import torch
from archive.scripts.train_mdctcodec_matched import train
from src.mdctcodec_matched import tensor_hash


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root, source, target):
    if (root / 'prepared.json').exists():
        assert json.loads((root/'protocol.json').read_text())['generator_updates'] == target
        return
    root.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((source / 'protocol.json').read_text())
    parent_budget = protocol['generator_updates']
    assert target > parent_budget
    protocol.update(name='mdctcodec-matched-6kbps-long', group='mdctcodec-6kbps-long-20260912',
                    generator_updates=target, discriminator_updates=target)
    lineage = {}
    for arm in ['laser', 'rvq']:
        parent = source / arm / 'checkpoints' / 'last.ckpt'
        state = torch.load(parent, map_location='cpu', weights_only=False)
        assert int(state['state_dict']['_manual_train_step']) == parent_budget
        model_hash = tensor_hash(state['state_dict'])
        rates = [g['lr'] for opt in state['optimizer_states'] for g in opt['param_groups']]
        assert len(set(rates)) == 1
        start_lr = rates[0]
        updates = dict(lr_schedule='cosine', lr_schedule_start_step=parent_budget,
                       lr_schedule_total_steps=target-parent_budget, warmup_steps=0,
                       min_lr_ratio=2e-5/start_lr)
        for key in ['learning_rate', 'encoder_learning_rate', 'decoder_learning_rate',
                    'adapter_learning_rate', 'disc_learning_rate']:
            updates[key] = start_lr
        if 'continuation_hparams' in protocol:
            assert protocol['continuation_hparams'] == updates
        protocol['continuation_hparams'] = updates
        state['hyper_parameters'].update(updates)
        # The parent closed its partial terminal epoch, but Lightning retained
        # stale within-epoch counters. Start the saved next epoch at batch zero.
        loop = state['loops']['fit_loop']
        for key in ['epoch_loop.batch_progress', 'epoch_loop.val_loop.batch_progress']:
            loop[key]['current'] = {name: 0 for name in loop[key]['current']}
            loop[key]['is_last_batch'] = False
        loop['epoch_loop.manual_optimization.optim_step_progress']['current'] = {'ready': 0, 'completed': 0}
        state['callbacks']['PairedAudit'] = {'records': [], 'batches': 0}
        output = root / arm
        checkpoints = output / 'checkpoints'
        checkpoints.mkdir(parents=True, exist_ok=True)
        # Preserve the previous best three even if the continuation regresses.
        for key, callback in state['callbacks'].items():
            if not key.startswith('ModelCheckpoint'):
                continue
            for old_path in callback['best_k_models']:
                shutil.copy2(old_path, checkpoints / Path(old_path).name)
            callback['best_k_models'] = {str((checkpoints / Path(p).name).resolve()): v
                                        for p, v in callback['best_k_models'].items()}
            for field in ['best_model_path', 'kth_best_model_path', 'last_model_path']:
                if callback[field]:
                    callback[field] = str((checkpoints / Path(callback[field]).name).resolve())
            callback['dirpath'] = str(checkpoints.resolve())
        resume = output / 'resume_200k.ckpt'
        torch.save(state, resume)
        restored = torch.load(resume, map_location='cpu', weights_only=False)
        assert tensor_hash(restored['state_dict']) == model_hash
        # Serialization of optimizer tensors and counters is lossless.
        for before, after in zip(state['optimizer_states'], restored['optimizer_states']):
            assert before['param_groups'] == after['param_groups']
            for pid, values in before['state'].items():
                for name, value in values.items():
                    assert torch.equal(value, after['state'][pid][name]) if torch.is_tensor(value) else value == after['state'][pid][name]
        shutil.copy2(source / arm / 'data_order.jsonl', output / 'data_order.jsonl')
        lineage[arm] = {'parent_checkpoint': str(parent.resolve()), 'parent_sha256': sha(parent),
            'parent_run': json.loads((source / arm / 'run.json').read_text()),
            'resume_checkpoint': str(resume.resolve()), 'resume_sha256': sha(resume),
            'state_dict_sha256': model_hash, 'optimizer_preserved': True,
            'resumed_epoch': state['epoch'], 'resumed_generator_updates': parent_budget}
    for name in ['manifest.json', 'calibration.json']:
        shutil.copy2(source / name, root / name)
    assert sha(root / 'manifest.json') == protocol['manifest_sha256']
    protocol['continuation'] = {'source': str(source.resolve()), 'lineage': lineage,
        'additional_generator_updates_per_arm': target-parent_budget,
        'schedule': 'Identical continuous cosine LR for G and D, current LR to 2e-5',
        'join': 'Both start saved epoch235 at batch0 after the same partial terminal epoch; stale within-epoch counters reset, global counters preserved',
        'frozen_tts_codec': 'Stage 2 continues its original immutable dictionary/cache; no silent dictionary substitution'}
    (root / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    (root / 'prepared.json').write_text(json.dumps({'protocol_sha256': sha(root/'protocol.json'),
        'lineage': lineage}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('outputs/mdctcodec_matched_6kbps_long'))
    p.add_argument('--source', type=Path, default=Path('outputs/mdctcodec_matched_6kbps_rangefix'))
    p.add_argument('--target', type=int, default=400000)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--arm', choices=['laser', 'rvq'])
    p.add_argument('--resume', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--mode', choices=['online', 'disabled'], default='online')
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    if args.prepare:
        prepare(args.root, args.source, args.target)
        return
    if not args.arm:
        p.error('--arm required')
    args.root = args.root.resolve()
    args.resume = args.resume or args.root / args.arm / 'resume_200k.ckpt'
    args.output = args.output or args.root / (f'preflight_{args.arm}' if args.smoke else args.arm)
    args.workers, args.metric_workers = (2, 2) if args.smoke else (8, 8)
    restored_step = int(torch.load(args.resume, map_location='cpu', weights_only=False)['state_dict']['_manual_train_step'])
    args.updates = restored_step + 2 if args.smoke else 0
    args.validation_limit = 2 if args.smoke else 0
    args.train_batches = 2 if args.smoke else 0
    train(args)


if __name__ == '__main__':
    main()
