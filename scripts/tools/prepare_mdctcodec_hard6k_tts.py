#!/usr/bin/env python3
"""Freeze the best ViSQOL codecs from a completed hard6k pair for fresh TTS."""
import argparse
import json
import math
from pathlib import Path
import shutil
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
import torch
from src.models.laser_tts import LaserTTS,TTSConfig
from src.tts_pairing import common_state,file_sha,record_signature,state_sha
from src.tts_data import FrameBatchSampler
from scripts.tools.prepare_mdctcodec_tts_pair import write,finalize


def select_best(stage1,arm):
    protocol=json.loads((stage1/'protocol.json').read_text())
    target=protocol['generator_updates']
    done=json.loads((stage1/arm/'completion.json').read_text())
    if done['status']!='complete' or done['generator_updates']!=target:
        raise ValueError(f'{arm}: stage1 must finish its full {target}-update allocation')
    source=Path(done['best_checkpoint']).resolve()
    scores={str(Path(p).resolve()):v for p,v in done['best_three'].items()}
    if str(source) not in scores or scores[str(source)]!=max(scores.values()):
        raise ValueError(f'{arm}: selected codec is not the highest validation ViSQOL checkpoint')
    if abs(scores[str(source)]-done['best_validation_visqol'])>1e-6:
        raise ValueError('ViSQOL selection metadata disagree')
    if not source.is_file():raise FileNotFoundError(source)
    state=torch.load(source,map_location='cpu',weights_only=False)
    hp=state['hyper_parameters']
    if hp.get('hard_rate_cap_bps')!=6000:raise ValueError('Expected the hard 6kbps codec')
    if arm=='laser':assert hp['num_embeddings']==4096 and hp['sparsity_level']==4
    else:assert hp['num_embeddings']==1024 and hp['rq_code_depth']==4
    selected_updates=int(state['state_dict']['_manual_train_step'])
    assert 0<selected_updates<=target
    return source,{'stage1_training_budget':target,'stage1_selected_updates':selected_updates,
        'stage1_validation_visqol':done['best_validation_visqol'],
        'stage1_checkpoint_selection':'Highest validation ViSQOL among checkpoints from the completed respective stage1 run',
        'stage1_run_url':done['url'],'stage1_completion_sha256':file_sha(stage1/arm/'completion.json'),
        'source_codec_checkpoint':str(source),'codec_sha256':file_sha(source)}


def prepare(root,stage1):
    root=root.resolve();stage1=stage1.resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/'plan.json').exists():
        plan=json.loads((root/'plan.json').read_text())
        assert plan['stage1_protocol_sha256']==file_sha(stage1/'protocol.json')
        for item in plan['arms'].values():assert file_sha(item['codec_checkpoint'])==item['codec_sha256']
        return
    selections={arm:select_best(stage1,arm) for arm in ('laser','rvq')}
    inventory_path=REPO/'outputs/mdctcodec_tts_rangefix/cache/inventory.json'
    inventory=json.loads(inventory_path.read_text())
    # Preserve the already reserved, unused 100-text paired test set; no test selection by codec scores.
    benchmark_source=REPO/'outputs/mdctcodec_tts_paired/benchmark/manifest.json'
    manifest=json.loads(benchmark_source.read_text())
    manifest['generation'].update(max_seconds=15,min_seconds=.2)
    manifest['limitations'] += [
        'Hard6k LASER has four atom/bin pairs per coded frame; RVQ has four codebook IDs. Coded frame rates and vocabulary-specific parameter counts differ.',
        'Batch order and update budgets match using original audio duration at the native 150Hz grid; actual coded token counts differ.']
    write(root/'benchmark/manifest.json',manifest)
    seed=20260914;arms={};shared=None
    torch.set_num_threads(4)
    (root/'frozen_codecs').mkdir(exist_ok=True)
    for arm,(source,selection) in selections.items():
        destination=root/'frozen_codecs'/f'{arm}.ckpt'
        if destination.exists():assert file_sha(destination)==selection['codec_sha256']
        else:
            temporary=destination.with_suffix('.tmp');shutil.copyfile(source,temporary)
            assert file_sha(temporary)==selection['codec_sha256'];temporary.replace(destination)
        cfg=TTSConfig(phone_vocab=max(inventory['phone_to_id'].values())+1,
            speakers=len(inventory['speaker_to_id']),width=512,heads=8,text_layers=3,audio_layers=8,
            depth_layers=2,dropout=.1,codec=arm,laser_atoms=4096,laser_sparsity=4,
            coefficient_levels=9,depth_positions=8,hard_rate_cap_bps=6000)
        torch.manual_seed(seed);model=LaserTTS(cfg)
        if shared is None:shared={k:v.clone() for k,v in common_state(model).items()}
        else:
            missing=model.load_state_dict(shared,strict=False)
            assert not missing.unexpected_keys and all(k.startswith(('fields.','heads.')) for k in missing.missing_keys)
        assert state_sha(common_state(model))==state_sha(shared)
        initial=root/f'{arm}_initial.pt'
        if initial.exists():assert state_sha(torch.load(initial,weights_only=True))==state_sha(model.state_dict())
        else:torch.save(model.state_dict(),initial)
        arms[arm]={**selection,'codec_checkpoint':str(destination),'cache':str(root/f'{arm}_cache/tokens.pt'),
            'initialization':str(initial),'initialization_sha256':file_sha(initial),'model_config':vars(cfg),
            'parameters':sum(p.numel() for p in model.parameters()),'common_parameters':sum(v.numel() for v in shared.values()),
            'vocab_sizes_without_eos':[4096,9]*4 if arm=='laser' else [1024]*4}
        del model
    train=dict(epochs=160,max_steps=120000,max_hours=24,frame_budget=8192,max_batch=16,
        accumulate=4,learning_rate=.0003,warmup_steps=1000,min_lr_ratio=1/30,weight_decay=.01,
        gradient_clip=1.,guided_attention_weight=.2,guided_attention_steps=8000,num_workers=4,
        save_every_steps=250,upload_every_epochs=5,preview_every_epochs=5,preview_count=16,
        preview_max_frames=1500,preview_max_seconds=10,validation_items=256,
        evaluate_untrained=False,preview_first_epoch=False,
        generation_validation=dict(items=64,every_epochs=10,max_frames=2250,max_seconds=15,
            python='/workspace/tts-benchmark-env/bin/python',
            asr_model_record='outputs/mdctcodec_tts_benchmark/models/Systran--faster-whisper-large-v3.json'))
    sampler=FrameBatchSampler([(r['samples']+359)//320+1 for r in inventory['records'] if r['split']=='train'],
        train['frame_budget'],train['max_batch'],seed=seed)
    scheduled_updates=0
    for epoch in range(train['epochs']):
        sampler.epoch=epoch;scheduled_updates+=math.ceil(len(sampler)/train['accumulate'])
    assert scheduled_updates<=train['max_steps'],'Step ceiling would interrupt the matched epoch budget'
    plan={'version':2,'name':'mdctcodec-hard6k-k4-paired-rqtransformer-20260914','seed':seed,
        'arms':arms,'common_initialization_sha256':state_sha(shared),
        'record_signature':record_signature(inventory,include_frames=False),'record_signature_basis':'utterance_metadata',
        'batch_length_basis':'native_150hz','stage1_protocol_sha256':file_sha(stage1/'protocol.json'),
        'stage1_root':str(stage1),'source_inventory':str(inventory_path),'source_inventory_sha256':file_sha(inventory_path),
        'benchmark_manifest_sha256':file_sha(root/'benchmark/manifest.json'),
        'reserved_benchmark_source_sha256':file_sha(benchmark_source),'train':train,'hard_rate_cap_bps':6000,
        'scheduled_updates_per_arm':scheduled_updates,
        'cache_counts':inventory['counts'],'train_audio_hours':sum(r['seconds'] for r in inventory['records'] if r['split']=='train')/3600,
        'cache_encoding':'Fresh full-utterance CUDA FP32 encoding for both frozen codecs; matmul TF32 off, cuDNN TF32 on. Exact packet equality and decoded original duration checked per worker.',
        'prior':'Common 512-wide text/temporal backbone and two-layer causal depth transformer; identical shared initial weights. LASER eight fields and RVQ four.',
        'laser_representation':'Four atom ID + coefficient bin pairs in OMP selection order; 4096 atoms, nine coefficient symbols, all four atoms distinct. Payload packing sorts paired values losslessly.',
        'generation_duration':'Per-codec frame budgets for the same 10s preview / 15s evaluation ceilings; serialized packets include duration and CRC and obey 6000 bit/s.',
        'selection':'Stage1: best respective validation ViSQOL after completion. Stage2: lowest validation64 Whisper WER at every10 completed epochs, earliest tie; fixed epoch160 endpoint also evaluated.',
        'checkpoint_upload':'Online every5 completed epochs: latest, best3 validation WER, best3 within-arm NLL; codec inputs immutable by SHA256.',
        'media':'16 fixed validation audio samples with waveforms, log-mel spectrograms and blue attention maps every5 completed epochs.',
        'compute_budget_gpu_hours':48,'budget_scope':'Assigned GPU wall time across cache encoding, preflight, paired TTS training and evaluation; waiting consumes none. Each training arm has a 24h ceiling.',
        'budget_incomplete_policy':'No matched final comparison unless both finish160 epochs with identical audited batch and optimizer-step sequences.',
        'limitations':manifest['limitations']}
    write(root/'plan.json',plan)
    print(json.dumps({'frozen_stage1':arms,'protocol':str(root/'plan.json')},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--stage1',type=Path)
    p.add_argument('--finalize',action='store_true');args=p.parse_args()
    if args.finalize:finalize(args.root)
    else:prepare(args.root,args.stage1)
