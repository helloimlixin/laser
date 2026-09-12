#!/usr/bin/env python3
"""Verify, report and archive the corrected codec comparison before stage 2."""
import hashlib
import html
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import soundfile as sf
from scipy.signal import butter,sosfiltfilt
import wandb

ROOT=Path('outputs/mdctcodec_fair_comparison')
SYSTEMS=['laser_original','laser_low_lr','mdctcodec_released','mdctcodec_trained_rvq',
         'dac_official','encodec_24k','encodec_48k','flowdec_6kbps']
METRICS=['visqol_audio48k','visqol_speech16k','pesq_wb16k','stoi16k']
LABELS={'laser_original':'LASER original (selected for stage 2)', 'laser_low_lr':'LASER lower LR',
        'mdctcodec_released':'Released MDCTCodec','mdctcodec_trained_rvq':'Recovered RVQ control',
        'dac_official':'DAC 44.1k, official full-utterance, seven codebooks',
        'encodec_24k':'EnCodec 24k mono, official ECDC',
        'encodec_48k':'EnCodec 48k stereo on duplicated mono, official ECDC',
        'flowdec_6kbps':'FlowDec-75m, eight codebooks, EMA, NFE=6'}


def listening(manifest):
    root=ROOT/'blinded_listening';root.mkdir(exist_ok=True)
    selected=[r for r in manifest['items'] if r['split']=='long9s' and r['id'].endswith('-00')]
    rng=random.Random(2026091203);key=[];sections=[]
    for index,item in enumerate(selected,1):
        trial=root/f'trial_{index:02d}';trial.mkdir(exist_ok=True)
        reference,sr=sf.read(item['reference'],dtype='float32')
        sf.write(trial/'reference.wav',reference,sr)
        versions=SYSTEMS+['hidden_reference','lowpass_anchor'];rng.shuffle(versions)
        controls=[]
        for j,system in enumerate(versions):
            letter=chr(65+j)
            if system=='hidden_reference':audio=reference
            elif system=='lowpass_anchor':audio=sosfiltfilt(butter(8,3500,fs=48000,output='sos'),reference).astype(np.float32)
            else:audio,_=sf.read(ROOT/'systems'/system/(item['id']+'.wav'),dtype='float32')
            sf.write(trial/(letter+'.wav'),audio,48000)
            key.append({'trial':index,'label':letter,'system':system,'id':item['id']})
            controls.append(f'<div class="sample"><b>{letter}</b><audio controls preload="none" src="trial_{index:02d}/{letter}.wav"></audio><label>Quality 0–100 <input type="number" min="0" max="100" data-trial="{index}" data-label="{letter}"></label></div>')
        sections.append(f'<section><h2>Trial {index}</h2><p>Reference</p><audio controls preload="none" src="trial_{index:02d}/reference.wav"></audio>'+''.join(controls)+'</section>')
    (ROOT/'blinded_listening_key.json').write_text(json.dumps(key,indent=2))
    page='''<!doctype html><meta charset="utf-8"><title>Codec listening comparison</title>
<style>body{font:16px system-ui;max-width:950px;margin:35px auto;background:#f8f9fb;color:#20242c}section{padding:20px;background:white;margin:20px 0;border-radius:12px}.sample{display:flex;gap:20px;align-items:center;margin:15px 0}audio{width:350px}input{width:70px}button{padding:12px 20px}p{line-height:1.5}</style>
<h1>Blinded codec listening comparison</h1><p>Use the same headphones and volume throughout. Compare each anonymous sample with the reference, then rate its overall audio quality from 0 (bad) to 100 (excellent). Leave a score empty if you have not listened. Each trial includes a hidden reference and a low-pass anchor. The order changes between trials.</p>
<p>This is a listening worksheet. No human ratings or MOS results have been collected yet. Your ratings stay in this browser until you export them.</p>
<button onclick="save()">Export ratings</button>'''+''.join(sections)+'''
<script>function save(){const ratings=[...document.querySelectorAll('input')].filter(x=>x.value!=='').map(x=>({trial:+x.dataset.trial,label:x.dataset.label,score:+x.value}));if(ratings.some(x=>x.score<0||x.score>100)){alert('Scores must be between 0 and 100');return}const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({protocol:'mdctcodec-fair-20260912',ratings},null,2)],{type:'application/json'}));a.download='codec-listening-ratings.json';a.click();URL.revokeObjectURL(a.href)}</script>'''
    (root/'index.html').write_text(page)


def main():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    mh=hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest()
    expected={r['id'] for r in manifest['items']};rows=[];summaries=[];models={}
    for system in SYSTEMS:
        directory=ROOT/'systems'/system
        state=json.loads((directory/'complete.json').read_text())
        assert state['records']==432 and state['manifest_sha256']==mh
        data=[json.loads((directory/(key+'.json')).read_text()) for key in sorted(expected)]
        assert {r['id'] for r in data}==expected and all(r['serialization_roundtrip'] for r in data)
        assert all(np.isfinite([r[k] for k in METRICS]).all() for r in data)
        rows.extend(data);summaries.extend(json.loads((directory/'summary.json').read_text()))
        models[system]=json.loads((directory/'model.json').read_text())
    paired=[];rng=np.random.default_rng(2026091204);draws=rng.integers(0,8,size=(5000,8))
    for split in ['fresh200','locked200','long9s']:
        subset=[r for r in rows if r['split']==split]
        speakers=sorted({r['speaker'] for r in subset});assert len(speakers)==8
        by_system={s:{r['id']:r for r in subset if r['system']==s} for s in SYSTEMS}
        for reference in ['laser_original','laser_low_lr']:
            for system in SYSTEMS:
                if system==reference:continue
                for metric in METRICS:
                    groups=[[r[metric]-by_system[system][key][metric] for key,r in by_system[reference].items() if r['speaker']==speaker] for speaker in speakers]
                    totals=np.array([sum(v) for v in groups]);counts=np.array([len(v) for v in groups])
                    boot=totals[draws].sum(1)/counts[draws].sum(1)
                    paired.append({'split':split,'reference':reference,'baseline':system,'metric':metric,
                                   'mean_delta':float(totals.sum()/counts.sum()),
                                   'ci95_low':float(np.quantile(boot,.025)),'ci95_high':float(np.quantile(boot,.975))})
    for name,value in [('results.json',summaries),('per_recording.json',rows),('paired_deltas.json',paired),('models.json',models)]:
        (ROOT/name).write_text(json.dumps(value,indent=2))
    fresh={r['system']:r for r in summaries if r['split']=='fresh200'}
    native=max(fresh,key=lambda s:fresh[s]['visqol_audio48k'])
    speech=max(fresh,key=lambda s:fresh[s]['visqol_speech16k'])
    lines=['# Corrected 6 kbps VCTK codec comparison','',
           'This report compares specific available checkpoints with their documented inference paths. '
           'It is not an equal-training-budget LASER/RVQ ablation, and it does not claim comprehensive current SOTA.','',
           f"On the new confirmation set, LASER original scores {fresh['laser_original']['visqol_audio48k']:.4f} in full-band ViSQOL. "
           f"The highest full-band ViSQOL among the evaluated checkpoints is {LABELS[native]} ({fresh[native]['visqol_audio48k']:.4f}); "
           f"the highest common-bandwidth speech ViSQOL is {LABELS[speech]} ({fresh[speech]['visqol_speech16k']:.4f}). "
           'These scores measure different aspects of reconstruction quality.','',
           'The primary confirmation set contains 200 newly selected recordings, balanced over eight speakers excluded from LASER optimization. '
           'It excludes 556 filenames recovered from previous test and validation manifests. '
           'Models, inference settings and the new manifest were frozen before scoring. '
           'The original 200-file test is repeated for correction, and 32 fixed nine-second segments provide a duration sensitivity check.','',
           'All decoders consume serialized discrete codes and declared metadata. DAC uses official full-utterance compression with -16 LUFS normalization and original-loudness restoration. '
           'EnCodec uses official ECDC compression/decompression, with a verified numerical comparison against direct upstream decoding. '
           'Its reader has a documented exact-integer frame-count fix: upstream floating arithmetic requests 386 frames for some 385-frame files and fails. '
           'The fix changes neither encoded codes nor synthesis; its complete source is archived under compatibility/. '
           'FlowDec uses the official 6 kbps eight-codebook setting with EMA weights and the documented midpoint solver (three steps / six function evaluations). '
           'Its underlying NDAC decoder is structurally eight samples shorter than its nominal frame expansion; eight zero guard samples are included before encoding, '
           'all resulting code frames are counted, and output is trimmed to original length.','',
           'ViSQOL audio mode uses the original 48 kHz bandwidth. Both signals are identically resampled to 16 kHz for speech-mode ViSQOL, wideband PESQ and STOI. '
           'These are separate quality questions; no aggregate winning score is invented.','']
    for split,title in [('fresh200','Fresh 200-recording confirmation set'),('locked200','Corrected original 200-recording set'),('long9s','Nine-second duration sensitivity check')]:
        lines += ['## '+title,'','| System | Payload kbps | ViSQOL audio48k | ViSQOL speech16k | PESQ-WB | STOI16k |','|---|---:|---:|---:|---:|---:|']
        for system in SYSTEMS:
            r=next(r for r in summaries if r['system']==system and r['split']==split)
            lines.append(f"| {LABELS[system]} | {r['payload_kbps']:.4f} | {r['visqol_audio48k']:.4f} | {r['visqol_speech16k']:.4f} | {r['pesq_wb16k']:.4f} | {r['stoi16k']:.4f} |")
        lines.append('')
    lines += ['## Rates and uncertainty','',
        'All headline rates are actual fixed-width packed token payload sizes divided by original audio duration, including frame padding. '
        'Auxiliary coefficient bounds/DAC loudness values/EnCodec scales are reported separately as decoder-value side information in results.json. '
        'Complete serialized file rates are also reported; LAC1 is an experimental JSON-header container, whereas EnCodec uses ECDC. '
        'Header formats are not equally compact, so file sizes are not ranked as codec quality. No hidden per-frame continuous latent bypasses the payload.','',
        'paired_deltas.json contains 5,000-draw paired speaker-bootstrap 95% intervals for every LASER/baseline metric comparison. '
        'Eight speaker clusters are a limited sample; intervals are descriptive, unadjusted for multiple comparisons, '
        'and do not include training-seed variability, checkpoint tuning or subjective-metric bias. '
        'The pretrained models have different corpora and potentially different public-data exposure. '
        'The common 16 kHz metrics are bandwidth-matched diagnostics of native codec outputs, not retraining each architecture at 16 kHz.','',
        'The blinded_listening/index.html worksheet contains eight anonymous nine-second trials, hidden references and low-pass anchors. '
        'Its assignment key is separate. No human listening scores have been collected; no MOS or formal MUSHRA result is claimed.','',
        '## Stage 2 decision','',
        'The original LASER checkpoint at epoch 287 was fixed for stage 2 before this evaluation because its original validation ViSQOL '
        '(4.150097) exceeds the lower-LR checkpoint\'s validation score (4.144818). Test scores do not change this selection. '
        'The existing TTS run was paused at optimizer step 3756 after five completed epochs, with optimizer/RNG state and its best three checkpoints uploaded. '
        'After this report is committed online, resume that run from its saved state using the same frozen 6 kbps tokenizer.','',
        '## Reproducibility and references','',
        'Manifest hash: `'+mh+'`. Source files, per-recording scores, model hashes, dependency versions and the listening worksheet are archived with this report. '
        'APCodec was inspected, but its cloned official repository did not include a downloadable checkpoint; no guessed or reimplemented score is substituted.','',
        '[MDCTCodec](https://github.com/PB20000090/MDCTCodec), '
        '[DAC](https://github.com/descriptinc/descript-audio-codec), '
        '[EnCodec](https://github.com/facebookresearch/encodec), '
        '[FlowDec](https://github.com/facebookresearch/FlowDec), '
        '[ViSQOL](https://github.com/google/visqol).']
    (ROOT/'comparison.md').write_text('\n'.join(lines)+'\n')
    listening(manifest)
    (ROOT/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    provenance={name:subprocess.check_output(['git','-C',path,'rev-parse','HEAD'],text=True).strip() for name,path in [('laser','.'),('mdctcodec','outputs/mdctcodec_reference'),('flowdec','outputs/flowdec_reference')]}
    (ROOT/'source_commits.json').write_text(json.dumps(provenance,indent=2))
    run=wandb.init(entity='helloimlixin-rutgers',project='laser',name='mdctcodec-laser-fair-6kbps-vctk-confirmation',
                   group='mdctcodec-fair-20260912',job_type='benchmark-report',dir=str(ROOT),
                   config={'manifest_sha256':mh,'systems':SYSTEMS,'protocol':manifest['protocol'],'stage2_checkpoint':manifest['codec_candidates']['original']})
    columns=list(summaries[0]);run.log({'comparison/results':wandb.Table(columns=columns,data=[[r[k] for k in columns] for r in summaries])})
    artifact=wandb.Artifact(f'mdctcodec-fair-comparison-{run.id}',type='benchmark',metadata={'manifest_sha256':mh,'systems':SYSTEMS})
    for path in ROOT.glob('*.json'):artifact.add_file(str(path),name=path.name)
    for name in ['comparison.md','environment.txt']:artifact.add_file(str(ROOT/name),name=name)
    artifact.add_dir(str(ROOT/'blinded_listening'),name='blinded_listening')
    artifact.add_dir(str(ROOT/'compatibility'),name='compatibility')
    source_paths=['src/fair_audio_codecs.py','src/mdctcodec_bitstream.py','src/audio_logging.py',
                  'scripts/prepare_mdctcodec_fair_manifest.py','scripts/benchmark_mdctcodec_fair.py',
                  'scripts/report_mdctcodec_fair.py','scripts/benchmark_mdctcodec_vctk.py',
                  'scripts/benchmark_mdctcodec_trained_rvq.py','scripts/train_mdctcodec_tts.py',
                  'scripts/setup_mdctcodec_audio.sh','requirements-audio.txt','requirements.txt']
    source_paths += [str(p) for p in Path('src').rglob('*.py')]
    for path in sorted(set(source_paths)):artifact.add_file(path,name='source/'+path)
    for subtree in ['flowdec','config','data']:
        for path in (Path('outputs/flowdec_reference')/subtree).rglob('*'):
            if path.is_file() and path.suffix in ['.py','.yaml','.cpp','.cu','.h','.npy']:
                artifact.add_file(str(path),name='source/flowdec/'+str(path.relative_to('outputs/flowdec_reference')))
    run.log_artifact(artifact,aliases=['latest','fair-6kbps']).wait()
    for r in summaries:
        if r['split']=='fresh200':
            for metric in METRICS:run.summary[f'fresh200/{r["system"]}/{metric}']=r[metric]
    run.summary['report_scope']='fair checkpoint comparison; no equal-budget architectural or comprehensive SOTA claim'
    result={'run_id':run.id,'url':run.url,'artifact':artifact.qualified_name,'state':artifact.state,
            'manifest_sha256':mh,'records':len(rows),'stage2_codec':manifest['codec_candidates']['original']}
    (ROOT/'report_complete.json').write_text(json.dumps(result,indent=2))
    run.finish();print(json.dumps(result,indent=2))


if __name__=='__main__':main()
