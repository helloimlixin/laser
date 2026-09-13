#!/usr/bin/env python3
"""Train a text-conditioned prior over frozen LASER audio codes."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader
import wandb
import yaml

from src.models.laser_tts import LaserTTS, TTSConfig
from src.tts_data import TTSDataset, FrameBatchSampler, collate_tts
from src.tts_runtime import ASREvaluator, CodecDecoder, word_error
from src.audio_research_media import render_tts_preview
from src.tts_schedule import learning_rate_at
from src.tts_generation_validation import GenerationValidation
from src.tts_pairing import load_paired_initialization, batch_chain


def atomic_save(value, path):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary); temporary.replace(path)


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/research/mdctcodec_tts_rangefix.yaml'))
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--new-run', action='store_true', help='Fork an explicitly resumed checkpoint into a new W&B run')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--codec-checkpoint', type=Path, help='Override codec path after restoring W&B inputs')
    parser.add_argument('--smoke-steps', type=int, default=0)
    parser.add_argument('--smoke-previews', action='store_true', help='Exercise one 30-frame preview during a smoke run')
    parser.add_argument('--no-previews', action='store_true')
    parser.add_argument('--mode', choices=['online', 'disabled'], default='online')
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text()); train_cfg = cfg['train']
    if cfg.get('paired'):
        frozen = json.loads(Path(cfg['paired']['protocol']).read_text())
        assert cfg['train'] == frozen['train'] and cfg['seed'] == frozen['seed'], 'Paired training recipe changed'
        assert cfg['model'] == {k:v for k,v in frozen['arms'][cfg['model']['codec']]['model_config'].items()
                                if k not in ('phone_vocab', 'speakers')}
    if args.smoke_previews:
        if not args.smoke_steps:parser.error('--smoke-previews requires --smoke-steps')
        train_cfg['preview_count']=1;train_cfg['preview_max_frames']=30
    if args.smoke_steps and train_cfg.get('generation_validation'):
        train_cfg['generation_validation']['items'] = 2
        train_cfg['generation_validation']['max_frames'] = 30
    output = args.output or Path(cfg['output']); output.mkdir(parents=True, exist_ok=True)
    if (output/'run.json').exists() and not args.resume:
        raise RuntimeError('Existing TTS run: resume explicitly or choose a new output directory')
    checkpoints = output / 'checkpoints'; checkpoints.mkdir(exist_ok=True)
    seed = cfg['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    device = torch.device(cfg['device'])
    cache_path = args.cache or Path(cfg['cache'])
    cache = torch.load(cache_path, map_location='cpu', weights_only=True)
    metadata = {k: v for k, v in cache.items() if k != 'records'}
    codec_path = args.codec_checkpoint or Path(cache['codec_checkpoint'])
    assert hashlib.sha256(codec_path.read_bytes()).hexdigest() == cache['codec_sha256']
    metadata['codec_checkpoint'] = str(codec_path.resolve())
    train_data = TTSDataset(cache, 'train', limit=128 if args.smoke_steps else 0)
    val_data = TTSDataset(cache, 'validation', limit=16 if args.smoke_steps else train_cfg['validation_items'])
    if not train_data or not val_data:
        raise ValueError('Training and validation must both contain complete utterances')
    # Select a stable, speaker-balanced validation subset rather than taking
    # the first entries from the speaker-sorted inventory.
    if not args.smoke_steps:
        all_val = TTSDataset(cache, 'validation')
        by_speaker = {}
        for record in all_val.records:
            by_speaker.setdefault(record['speaker'], []).append(record)
        rng = random.Random(seed)
        for records in by_speaker.values(): rng.shuffle(records)
        val_data.records = [records[i] for i in range(max(map(len, by_speaker.values())))
                            for speaker, records in sorted(by_speaker.items()) if i < len(records)][:train_cfg['validation_items']]
        val_data.lengths = [len(r['codes']) + 1 for r in val_data.records]
    validation_manifest = [{k: v for k, v in r.items() if k != 'codes'} for r in val_data.records]
    (output / 'validation_manifest.json').write_text(json.dumps(validation_manifest, indent=2, ensure_ascii=False))
    cfg['cache_metadata'] = metadata
    (output / 'resolved_config.json').write_text(json.dumps(cfg, indent=2))
    sampler = FrameBatchSampler(train_data.lengths, train_cfg['frame_budget'], train_cfg['max_batch'], seed=seed)
    val_sampler = FrameBatchSampler(val_data.lengths, train_cfg['frame_budget'], train_cfg['max_batch'], shuffle=False)
    loader_options = dict(collate_fn=collate_tts, num_workers=train_cfg['num_workers'], pin_memory=True,
                          persistent_workers=train_cfg['num_workers'] > 0,
                          generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(train_data, batch_sampler=sampler, **loader_options)
    val_loader = DataLoader(val_data, batch_sampler=val_sampler, **loader_options)
    model_cfg = TTSConfig(phone_vocab=max(cache['phone_to_id'].values()) + 1,
                          speakers=len(cache['speaker_to_id']), **cfg['model'])
    model = LaserTTS(model_cfg).to(device)
    if cfg.get('paired'):
        load_paired_initialization(model, cache, cache_path, cfg['paired'])
        # Vocabulary-specific initialization consumes different RNG draws. Reset
        # training RNG after loading the common weights, identically in both arms.
        torch.manual_seed(seed + 1); random.seed(seed + 1); np.random.seed(seed + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg['learning_rate'], betas=(.9, .95),
                                  weight_decay=train_cfg['weight_decay'])
    budget_sampler = sampler
    if args.smoke_steps:
        # A small smoke dataset must not shorten the restored LR schedule.
        budget_sampler = FrameBatchSampler(TTSDataset(cache, 'train').lengths,
            train_cfg['frame_budget'], train_cfg['max_batch'], seed=seed)
    total_steps = min(train_cfg['max_steps'], math.ceil(len(budget_sampler) / train_cfg['accumulate']) * train_cfg['epochs'])
    if cfg.get('paired'):
        # Bucket packing varies slightly by epoch. Use the exact shared schedule.
        scheduled = 0
        for schedule_epoch in range(train_cfg['epochs']):
            budget_sampler.epoch = schedule_epoch
            scheduled += math.ceil(len(budget_sampler) / train_cfg['accumulate'])
        budget_sampler.epoch = 0
        total_steps = min(train_cfg['max_steps'], scheduled)
    start_epoch, step, completed, skip_batches, best = 0, 0, 0, 0, []
    previous_run_id = None
    best_generation = []
    data_chain, seen_frames, seen_batches = '00' * 32, 0, 0
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        assert state['metadata']['codec_sha256'] == metadata['codec_sha256']
        assert state['metadata']['phone_to_id'] == metadata['phone_to_id']
        assert state['metadata']['speaker_to_id'] == metadata['speaker_to_id']
        assert asdict(TTSConfig(**state['model_config'])) == asdict(model_cfg)
        model.load_state_dict(state['model'], strict=True); optimizer.load_state_dict(state['optimizer'])
        step, completed = state['step'], state['completed_epochs']
        start_epoch, skip_batches = state['epoch'], state['batch_in_epoch']
        best = state['best']; previous_run_id = state.get('wandb_id')
        best_generation = state.get('best_generation', [])
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state(state['cuda_rng'], device)
        random.setstate(state['random_rng'])
        data_chain = state.get('data_chain', data_chain)
        seen_frames = state.get('seen_frames', 0); seen_batches = state.get('seen_batches', 0)
    audit_path = output / 'data_order.jsonl'
    if cfg.get('paired') and audit_path.exists():
        old = [json.loads(line) for line in audit_path.read_text().splitlines()]
        old = [row for row in old if row['step'] <= step]
        assert not old or (old[-1]['step'] == step and old[-1]['chain'] == data_chain)
        audit_path.write_text(''.join(json.dumps(row) + '\n' for row in old))
    if train_cfg.get('continuation') and (not args.resume or step < train_cfg['continuation']['start_step']):
        raise ValueError('Continuation config requires the corresponding optimizer checkpoint')
    parent_run_id = previous_run_id
    if args.new_run:
        if not args.resume or (output/'run.json').exists():
            raise ValueError('--new-run requires a resume checkpoint and a fresh output directory')
        previous_run_id = None
        cfg['parent'] = {'run_id': parent_run_id, 'checkpoint': str(args.resume.resolve()),
                        'sha256': hashlib.sha256(args.resume.read_bytes()).hexdigest(), 'step': step}
    generation_validator = (GenerationValidation(cache, train_cfg['generation_validation'], output, seed)
                            if train_cfg.get('generation_validation') else None)
    (output / 'resolved_config.json').write_text(json.dumps(cfg, indent=2))
    run = wandb.init(entity=cfg['wandb']['entity'], project=cfg['wandb']['project'], mode=args.mode,
                     name=cfg['wandb']['name'], group=cfg['wandb']['group'], job_type='stage2-tts',
                     dir=str(output), config={**cfg, 'model_resolved': asdict(model_cfg),
                         'parameters': sum(p.numel() for p in model.parameters()), 'total_steps': total_steps,
                         'train_items': len(train_data), 'validation_items': len(val_data)},
                     id=previous_run_id if args.mode == 'online' else None,
                     resume='must' if previous_run_id and args.mode == 'online' else None)
    (output / 'run.json').write_text(json.dumps({'id': run.id, 'url': run.url, 'pid': os.getpid()}))
    if args.resume and (output / 'completion.json').exists():
        (output / 'completion.json').replace(output / f'completion_before_resume_step{step}.json')
    run.summary.update({'status': 'running', 'resumed_from_step': step})
    if args.new_run and args.mode == 'online':
        run.use_artifact(f'{cfg["wandb"]["entity"]}/{cfg["wandb"]["project"]}/mdctcodec-tts-inputs-{metadata["codec_sha256"][:12]}:v0')
    if args.mode == 'online':
        # W&B keeps the old config when resuming; explicitly record the new budget.
        run.config.update({**cfg, 'total_steps': total_steps}, allow_val_change=True)
        source = wandb.Artifact(f'mdctcodec-tts-source-{run.id}', type='code')
        source_paths = ['src/tts_data.py', 'src/tts_runtime.py', 'src/models/laser_tts.py',
                     'src/training/mdctcodec_tts.py', 'src/audio_research_media.py',
                     'scripts/tools/cache_mdctcodec_tts.py', 'scripts/tools/train_mdctcodec_tts.py',
                     'archive/scripts/generate_mdctcodec_tts.py', str(args.config),
                     'tests/test_laser_tts.py', 'requirements-audio.txt',
                     'requirements.txt', 'archive/scripts/benchmark_mdctcodec_vctk.py',
                     'docs/mdctcodec-stage2-rangefix-2026-09-12.md']
        source_paths += [str(p) for p in Path('src').rglob('*.py')]
        for path in sorted(set(source_paths)):
            if Path(path).is_file(): source.add_file(path, name=path)
        source.add_file(str(output / 'resolved_config.json'), name='resolved_config.json')
        source.add_file(str(output / 'validation_manifest.json'), name='validation_manifest.json')
        if generation_validator:
            source.add_file(str(output/'generation_validation_manifest.json'), name='generation_validation_manifest.json')
            source.add_file('scripts/tools/score_tts_generation_validation.py', name='scripts/tools/score_tts_generation_validation.py')
        run.log_artifact(source, aliases=['latest', f'from-step-{step}']).wait()
    if args.mode == 'online' and not args.resume:
        inputs = wandb.Artifact(f'mdctcodec-tts-inputs-{metadata["codec_sha256"][:12]}', type='dataset',
                               metadata=metadata)
        inputs.add_file(str(cache_path), name='tokens.pt')
        inputs.add_file(str(codec_path), name='frozen_codec.ckpt')
        run.log_artifact(inputs, aliases=['latest']).wait()
        run.use_artifact(inputs)
    run.define_metric('val/token_nll', summary='min')
    run.define_metric('val_generation/wer', summary='min')
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGINT, request_stop)
    started = time.monotonic(); initial_step = step
    decoder, recognizer = None, None

    def save_last(epoch, batch_index):
        atomic_save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'model_config': asdict(model_cfg), 'train_config': cfg, 'metadata': metadata,
            'step': step, 'epoch': epoch, 'completed_epochs': completed, 'batch_in_epoch': batch_index,
            'best': best, 'best_generation': best_generation,
            'wandb_id': run.id if args.mode == 'online' else None,
            'data_chain': data_chain, 'seen_frames': seen_frames, 'seen_batches': seen_batches,
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(device),
            'random_rng': random.getstate()}, checkpoints / 'last.pt')

    def upload(reason):
        if args.mode != 'online': return
        artifact = wandb.Artifact(f'model-stage2-tts-{run.id}-selected-checkpoints', type='model',
            metadata={'completed_epochs': completed, 'step': step, 'monitor': 'val/token_nll',
                      'mode': 'min', 'reason': reason, 'best': best, 'best_generation': best_generation,
                      'generation_monitor': 'val_generation/wer', 'codec_sha256': metadata['codec_sha256']})
        artifact.add_file(str(checkpoints / 'last.pt'), name='last.pt')
        for item in best: artifact.add_file(item['path'], name=Path(item['path']).name)
        for item in best_generation: artifact.add_file(item['path'], name=Path(item['path']).name)
        if audit_path.exists(): artifact.add_file(str(audit_path), name='data_order.jsonl')
        if cfg.get('paired'): artifact.add_file(cfg['paired']['protocol'], name='paired_protocol.json')
        run.log_artifact(artifact, aliases=['latest', 'best-plus-last', f'epoch-{completed:03d}']).wait()

    @torch.inference_mode()
    def validate():
        model.eval(); totals = {}; count = 0
        for batch in val_loader:
            batch = to_device(batch, device)
            with torch.autocast('cuda', dtype=torch.bfloat16): metrics = model(batch)
            n = int(batch['lengths'].sum())
            for key in [k for k in metrics if k in ('nll', 'atom_nll', 'coefficient_nll', 'token_accuracy') or k.startswith('field_')]:
                totals[key] = totals.get(key, 0.) + float(metrics[key]) * n
            count += n
        return {f'val/{k if k != "nll" else "token_nll"}': v / count for k, v in totals.items()}

    @torch.inference_mode()
    def previews(epoch):
        nonlocal decoder, recognizer
        if args.no_previews: return
        # Reload presentation settings so color changes need no training restart.
        preview_style = yaml.safe_load(args.config.read_text()).get('media', {})
        if decoder is None: decoder = CodecDecoder(codec_path, cfg['evaluation_device'])
        if recognizer is None: recognizer = ASREvaluator(cfg['evaluation_device'])
        preview_dir = output / 'samples' / f'epoch_{epoch:03d}'; preview_dir.mkdir(parents=True, exist_ok=True)
        chosen, seen = [], set()
        for i, r in enumerate(val_data.records):
            if r['speaker'] not in seen and 15 <= len(r['text_key']) <= 100 and r['seconds'] <= 6:
                chosen.append(i); seen.add(r['speaker'])
                if len(chosen) == train_cfg['preview_count']: break
        if not chosen:
            raise RuntimeError('Validation subset contains no suitable TTS preview utterances')
        results, errors, words, media = [], 0, 0, {}
        id_to_phone={value:key for key,value in metadata['phone_to_id'].items()}
        with torch.random.fork_rng(devices=[device.index or 0]):
            for index in chosen:
                # Hold sampling randomness fixed across checkpoints for each prompt.
                torch.manual_seed(seed + 10000 + index)
                item = val_data[index]; r = item['record']
                phones=item['phones'][None].to(device)
                speaker=torch.tensor([item['speaker']],device=device)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    tokens, info = model.generate(phones,speaker,max_frames=train_cfg['preview_max_frames'])
                    memory,text_mask=model.text_memory(phones,speaker)
                    _,attention=model.temporal(tokens[None],memory,text_mask,speaker,alignment=True)
                audio, payload = decoder.decode(tokens)
                path = preview_dir / (Path(r['path']).stem + '.wav')
                sf.write(path, audio, 48000,subtype='FLOAT'); path.with_suffix('.bin').write_bytes(payload)
                hypothesis = recognizer.transcribe(audio)
                err, num_words = word_error(r['text'], hypothesis); errors += err; words += num_words
                reference, _ = sf.read(r['path'], dtype='float32')
                ref_path=preview_dir/(Path(r['path']).stem+'_reference.wav')
                sf.write(ref_path,reference,48000,subtype='FLOAT')
                ref_hypothesis = recognizer.transcribe(reference)
                ref_err, ref_words = word_error(r['text'], ref_hypothesis)
                row = {'text': r['text'], 'speaker': r['speaker'], 'asr_text': hypothesis,
                       'wer': err / max(1, num_words), 'reference_asr_wer': ref_err / max(1, ref_words), **info}
                results.append(row)
                figures=render_tts_preview(reference,audio,preview_dir/Path(r['path']).stem,
                    title=f'{r["speaker"]} | Epoch {epoch} | {r["text"]}',
                    attention=attention[0].float().cpu().numpy(),
                    phone_labels=[id_to_phone.get(int(v),'EOS' if int(v)==2 else 'UNK') for v in item['phones']],
                    attention_cmap=preview_style.get('attention_cmap', 'Blues'))
                media[f'generated/{index}/audio']=wandb.Audio(str(path),caption=r['text'])
                media[f'generated/{index}/reference']=wandb.Audio(str(ref_path),caption=r['text'])
                for kind,image_path in figures.items():media[f'generated/{index}/{kind}']=wandb.Image(image_path)
        (preview_dir / 'results.json').write_text(json.dumps(results, indent=2))
        run.log({**media,'generation/asr_wer': errors / max(1, words),
                 'generation/eos_fraction': sum(r['eos_reached'] for r in results) / max(1, len(results)),
                 'generation/transcripts': wandb.Table(columns=list(results[0]), data=[list(r.values()) for r in results])}, step=step)
        if args.mode=='online':
            artifact=wandb.Artifact(f'mdctcodec-tts-previews-{run.id}',type='audio-evaluation',
                metadata={'epoch':epoch,'step':step,'codec_sha256':metadata['codec_sha256'],
                          'scope':'Free-running known-speaker validation prompts; original levels; independent durations'})
            artifact.add_dir(str(preview_dir),name=f'epoch_{epoch:03d}')
            run.log_artifact(artifact,aliases=['latest',f'epoch-{epoch:03d}']).wait()

    def validate_generation():
        nonlocal decoder
        if decoder is None: decoder = CodecDecoder(codec_path, cfg['evaluation_device'])
        result, target = generation_validator.evaluate(model, decoder, completed, step, device)
        run.log({f'val_generation/{key}': result[key] for key in
                 ['wer', 'cer', 'items', 'reference_wer', 'eos_fraction']}, step=step)
        score = result['wer']
        if len(best_generation) < 3 or score < best_generation[-1]['score']:
            path = checkpoints / f'tts-wer-epoch{completed:03d}-step{step:07d}.pt'
            atomic_save({'model': model.state_dict(), 'model_config': asdict(model_cfg), 'metadata': metadata,
                'step': step, 'epoch': completed-1, 'validation_wer': score,
                'generation_manifest_sha256': generation_validator.manifest_sha}, path)
            best_generation.append({'score': score, 'path': str(path.resolve()), 'epoch': completed, 'step': step})
            best_generation.sort(key=lambda x: x['score'])
            while len(best_generation) > 3:
                obsolete = Path(best_generation.pop()['path'])
                if obsolete.parent.resolve() == checkpoints.resolve(): obsolete.unlink(missing_ok=True)
        if args.mode == 'online':
            artifact = wandb.Artifact(f'mdctcodec-tts-generation-validation-{run.id}', type='audio-evaluation',
                metadata={k: v for k, v in result.items() if k != 'rows'})
            artifact.add_dir(str(target))
            artifact.add_file(str(output/'generation_validation_manifest.json'))
            run.log_artifact(artifact, aliases=['latest', f'epoch-{completed:03d}']).wait()

    # Anchor checkpoint selection on the same validation prompts before updating weights.
    if generation_validator and not best_generation and (args.resume or train_cfg.get('evaluate_untrained', True)):
        validate_generation()
        save_last(start_epoch, skip_batches)
        upload('continuation_baseline')
    print(json.dumps({'parameters': sum(p.numel() for p in model.parameters()), 'train_items': len(train_data),
                      'validation_items': len(val_data), 'steps_per_epoch': math.ceil(len(sampler)/train_cfg['accumulate']),
                      'resume_step': step, 'url': run.url}), flush=True)
    stop_reason = 'epochs_complete'
    for epoch in range(start_epoch, train_cfg['epochs']):
        sampler.epoch = epoch; model.train(); optimizer.zero_grad(set_to_none=True)
        epoch_batches = len(loader)
        rolling = {}; micro_count = 0; epoch_complete = True; last_batch = 0
        for batch_index, batch in enumerate(loader):
            if epoch == start_epoch and batch_index < skip_batches: continue
            if cfg.get('paired'):
                data_chain = batch_chain(data_chain, batch)
                seen_frames += int(batch['lengths'].sum()); seen_batches += 1
            batch = to_device(batch, device)
            group_start = (batch_index // train_cfg['accumulate']) * train_cfg['accumulate']
            group_size = min(train_cfg['accumulate'], epoch_batches - group_start)
            if cfg.get('paired') and batch_index == group_start:
                # Codec/ASR construction and generation must not perturb the
                # common dropout stream in the following training update.
                torch.manual_seed(seed + 1 + step)
            guide = train_cfg['guided_attention_weight'] * max(0., 1 - step / train_cfg['guided_attention_steps'])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                metrics = model(batch, guide_weight=guide)
            if not torch.isfinite(metrics['loss']): raise RuntimeError('Nonfinite TTS training loss')
            (metrics['loss'] / group_size).backward()
            micro_count += 1
            for key, value in metrics.items(): rolling[key] = rolling.get(key, 0.) + float(value.detach())
            last_batch = batch_index + 1
            if last_batch % train_cfg['accumulate'] and last_batch != epoch_batches: continue
            for group in optimizer.param_groups:
                group['lr'] = learning_rate_at(step, train_cfg, total_steps)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg['gradient_clip'], error_if_nonfinite=True)
            optimizer.step(); optimizer.zero_grad(set_to_none=True); step += 1
            if cfg.get('paired'):
                with audit_path.open('a') as stream:
                    stream.write(json.dumps({'step': step, 'epoch': epoch, 'batch_in_epoch': last_batch,
                        'chain': data_chain, 'frames': seen_frames, 'batches': seen_batches,
                        'learning_rate': optimizer.param_groups[0]['lr']}) + '\n')
            if step % 10 == 0 or step <= 3:
                values = {f'train/{k}': v / micro_count for k, v in rolling.items()}
                values.update(epoch=epoch, learning_rate=optimizer.param_groups[0]['lr'], gradient_norm=float(norm),
                              elapsed_seconds=time.monotonic() - started,
                              optimizer_steps_per_second=(step - initial_step) / max(1., time.monotonic() - started))
                run.log(values, step=step)
                print('STEP', step, json.dumps(values), flush=True)
                rolling, micro_count = {}, 0
            if step % train_cfg['save_every_steps'] == 0: save_last(epoch, last_batch)
            if stop_requested or time.monotonic() - started >= train_cfg['max_hours'] * 3600 or step >= total_steps or (args.smoke_steps and step - initial_step >= args.smoke_steps):
                epoch_complete = last_batch == epoch_batches
                stop_reason = 'signal' if stop_requested else 'smoke_complete' if args.smoke_steps else 'step_or_time_budget'
                if epoch_complete and epoch + 1 == train_cfg['epochs'] and not args.smoke_steps:
                    stop_reason = 'epochs_complete'
                break
        values = validate(); score = values['val/token_nll']
        completed = epoch + 1 if epoch_complete else epoch
        run.log({**values, 'completed_epochs': completed}, step=step)
        print('VALIDATION', step, completed, json.dumps(values), flush=True)
        if len(best) < 3 or score < best[-1]['score']:
            path = checkpoints / f'tts-epoch{epoch+1:03d}-step{step:07d}.pt'
            atomic_save({'model': model.state_dict(), 'model_config': asdict(model_cfg), 'metadata': metadata,
                         'step': step, 'epoch': epoch, 'validation_nll': score}, path)
            best.append({'score': score, 'path': str(path.resolve())}); best.sort(key=lambda x: x['score'])
            while len(best) > 3:
                obsolete = Path(best.pop()['path'])
                if obsolete.parent.resolve() == checkpoints.resolve(): obsolete.unlink(missing_ok=True)
        if generation_validator and epoch_complete and completed % train_cfg['generation_validation']['every_epochs'] == 0:
            validate_generation()
        save_last(epoch + 1 if epoch_complete else epoch, 0 if epoch_complete else last_batch)
        if epoch_complete and completed % train_cfg['upload_every_epochs'] == 0: upload('completed_epoch')
        if not args.smoke_steps and ((completed == 1 and train_cfg.get('preview_first_epoch', True)) or (epoch_complete and completed % train_cfg['preview_every_epochs'] == 0)):
            previews(completed)
        if args.smoke_previews:previews(completed)
        if stop_reason != 'epochs_complete': break
    upload('train_end')
    result = {'status': stop_reason, 'step': step, 'completed_epochs': completed, 'best': best,
              'best_generation': best_generation, 'run_url': run.url,
              'data_chain': data_chain, 'seen_frames': seen_frames, 'seen_batches': seen_batches}
    (output / 'completion.json').write_text(json.dumps(result, indent=2))
    run.summary.update(result); run.finish(); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
