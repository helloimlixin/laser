#!/usr/bin/env python3
"""Publish matched checkpoint audio/figures while existing MDCTCodec jobs run.

This CPU-only observer has its own W&B run in the training group. It reads only
committed checkpoint artifacts and fixed validation files, never the locked test.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
os.environ.setdefault('VISQOL_BINARY',str(REPO/'outputs/visqol/bin/visqol'))
import numpy as np
import soundfile as sf
import torch
import wandb

from archive.scripts.benchmark_mdctcodec_vctk import align_mdct, load_reference
from archive.scripts.benchmark_mdctcodec_trained_rvq import payload_roundtrip
from archive.scripts.benchmark_mdctcodec_fair import measure_record
from src.audio_research_media import (LABELS, SAMPLE_RATE, N_FFT, HOP, MEL_BINS,
    select_validation_examples, decode_payload_tokens, render_audio_comparison,
    render_token_diagnostics)
from src.mdctcodec_matched import reconstruct_serialized
from src.models.laser import LASER


def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()


def available_snapshots(api, root, parents, protocol):
    arms={}
    for arm,parent in parents.items():
        found={}
        for artifact in api.run(parent['path']).logged_artifacts():
            if artifact.type!='model' or artifact.state!='COMMITTED':continue
            meta=artifact.metadata
            epoch,step=meta.get('epoch'),meta.get('global_step')
            if isinstance(epoch,int) and epoch>0 and epoch%5==0 and step==2*epoch*protocol['batches_per_epoch']:
                found[int(step)//2]={'artifact':f'{parent["entity"]}/{parent["project"]}/{artifact.name}',
                                    'completed_epochs':epoch,'generator_updates':int(step)//2}
        complete=root/arm/'completion.json'
        if complete.exists():
            record=read(complete)
            if record['status']=='complete' and record.get('artifact'):
                step=record['generator_updates']
                found[step]={'artifact':record['artifact'],'completed_epochs':record['completed_full_epochs'],
                             'generator_updates':step,'final':True}
        arms[arm]=found
    return {step:{arm:arms[arm][step] for arm in arms} for step in sorted(set(arms['laser']) & set(arms['rvq']))}


def reference_cache(output, paths, reference_root):
    cache=output/'reference';cache.mkdir(exist_ok=True)
    sources={name:sha(reference_root/name) for name in
             ['encoder_00200000_onlyvctk','decoder_00200000_onlyvctk','quantize.py']}
    marker=cache/'complete.json'
    identity={'sources':sources,'paths':paths,'sample_rate':SAMPLE_RATE}
    if marker.exists():
        assert read(marker)==identity
        return cache
    encoder,quantizer,decoder=load_reference(reference_root,'cpu')
    with torch.inference_mode():
        for path in paths:
            reference,sr=sf.read(path,dtype='float32');assert sr==SAMPLE_RATE
            stem=Path(path).stem
            sf.write(cache/f'{stem}_reference.wav',reference,sr,subtype='FLOAT')
            x,length=align_mdct(torch.from_numpy(reference)[None,None])
            ids=quantizer(encoder(x),n_quantizers=4)[1]
            payload,parsed=payload_roundtrip(ids.cpu().numpy())
            y=decoder(quantizer.from_codes(torch.from_numpy(parsed))[0])[0,0,:length].clamp(-1,1).numpy()
            assert np.isfinite(y).all() and y.shape==reference.shape
            target=cache/f'{stem}_released_mdctcodec.wav'
            sf.write(target,y,sr,subtype='FLOAT');target.with_suffix('.bin').write_bytes(payload)
            metrics=measure_record(path,str(target),{'id':stem,'speaker':Path(path).parent.name,
                                   'system':'released_mdctcodec','split':'fixed_validation_preview'})
            write(target.with_suffix('.json'),metrics)
    write(marker,identity)
    return cache


def publish_snapshot(run, root, output, paths, snapshot, step, cache):
    target=output/'snapshots'/f'update-{step:07d}';target.mkdir(parents=True,exist_ok=True)
    provenance={};decoded={arm:{} for arm in ['laser','rvq']};tokens={arm:[] for arm in decoded}
    measured={arm:{} for arm in decoded};pending=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for arm,source in snapshot.items():
            artifact=run.use_artifact(source['artifact'])
            artifact.wait()
            checkpoint_dir=output/'checkpoint_cache'/f'{arm}-{step}'
            checkpoint=Path(artifact.get_entry('last.ckpt').download(root=str(checkpoint_dir)))
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert int(saved['state_dict']['_manual_train_step'])==step
            model=LASER.load_from_checkpoint(checkpoint,map_location='cpu',strict=True).eval()
            bound=float(model.bottleneck.coefficient_quantization_max) if arm=='laser' else None
            provenance[arm]={**source,'sha256':sha(checkpoint),'bound':bound,
                             'checkpoint_epoch_index':saved['epoch']}
            if arm=='laser':
                observer=saved['callbacks']['TrainingCoefficientRange']
                provenance[arm]['training_clipping_window_mean']=float(np.mean(observer['clipping']))
            with torch.inference_mode():
                for path in paths:
                    stem=Path(path).stem;reference,sr=sf.read(path,dtype='float32');assert sr==SAMPLE_RATE
                    y,payload=reconstruct_serialized(model,torch.from_numpy(reference)[None,None])
                    y=y[0,0].numpy();decoded[arm][stem]=y
                    ids,coefficients=decode_payload_tokens(payload,arm)
                    item={'ids':ids,'coefficients':coefficients,'utterance':stem}
                    if arm=='laser':
                        assert model.bottleneck.coefficient_quantization_max==bound
                        item['raw_coefficient_clipping_fraction']=float(model.bottleneck._last_coefficient_saturation_fraction)
                        item['raw_coefficient_abs_p999']=float(model.bottleneck._last_coefficient_abs_p999)
                    tokens[arm].append(item)
                    wav=target/f'{stem}_{arm}.wav';sf.write(wav,y,sr,subtype='FLOAT');wav.with_suffix('.bin').write_bytes(payload)
                    meta={'id':stem,'speaker':Path(path).parent.name,'system':arm,
                          'split':'fixed_validation_preview','generator_updates':step,
                          'payload_kbps':len(payload)*8/(len(reference)/SAMPLE_RATE)/1000}
                    pending.append((arm,stem,pool.submit(measure_record,path,str(wav),meta)))
            del model,saved
        for arm,stem,future in pending:measured[arm][stem]=future.result()
    # Inputs are saved in their original scale. W&B Audio(path) avoids per-clip gain.
    media={'media/generator_updates':step,'media/completed_epochs':snapshot['laser']['completed_epochs']}
    table=wandb.Table(columns=['speaker','utterance','seconds','reference','LASER','RVQ','released_MDCTCodec',
        'LASER_ViSQOL_audio48k','RVQ_ViSQOL_audio48k','released_ViSQOL_audio48k','log_mel','waveform_MDCT'])
    rows=[];galleries={};audios={name:[] for name in LABELS}
    for path in paths:
        stem=Path(path).stem
        reference,sr=sf.read(path,dtype='float32')
        released,_=sf.read(cache/f'{stem}_released_mdctcodec.wav',dtype='float32')
        waves={'reference':reference,'laser':decoded['laser'][stem],'rvq':decoded['rvq'][stem],
               'released_mdctcodec':released}
        title=f'{stem} | {step:,} generator updates | nominal 6 kbps'
        figures,distortion=render_audio_comparison(waves,target/stem,title)
        for kind,image_path in figures.items():galleries.setdefault(kind,[]).append(wandb.Image(image_path,caption=title))
        for name,wave in waves.items():
            wav=target/f'{stem}_{name}.wav'
            if not wav.exists():sf.write(wav,wave,SAMPLE_RATE,subtype='FLOAT')
            if name=='released_mdctcodec':
                wav.with_suffix('.bin').write_bytes((cache/f'{stem}_released_mdctcodec.bin').read_bytes())
            audios[name].append(wandb.Audio(str(wav),caption=f'{stem} | {LABELS[name]} | original level'))
        clip={'utterance':stem,'speaker':Path(path).parent.name,'seconds':len(reference)/SAMPLE_RATE,'models':{}}
        for arm in ['laser','rvq','released_mdctcodec']:
            values=read(cache/f'{stem}_released_mdctcodec.json') if arm=='released_mdctcodec' else measured[arm][stem]
            clip['models'][arm]={**values,**distortion[arm]}
        rows.append(clip)
        table.add_data(clip['speaker'],stem,clip['seconds'],*[audios[n][-1] for n in LABELS],
                       *[clip['models'][n]['visqol_audio48k'] for n in ['laser','rvq','released_mdctcodec']],
                       wandb.Image(figures['log_mel']),wandb.Image(figures['waveform_mdct']))
        print('AUDIO_MEDIA_CLIP',json.dumps({'step':step,'utterance':stem}),flush=True)
    token_image,token_metrics=render_token_diagnostics(tokens,target,f'{step:,} generator updates')
    media['figures/token_diagnostics']=wandb.Image(token_image)
    for arm in ['laser','rvq']:
        if arm=='laser':
            token_metrics[arm]['raw_coefficient_clipping_fraction']=float(np.mean([t['raw_coefficient_clipping_fraction'] for t in tokens[arm]]))
            token_metrics[arm]['saved_bound']=provenance[arm]['bound']
            token_metrics[arm]['training_clipping_window_mean']=provenance[arm]['training_clipping_window_mean']
            for k,v in token_metrics[arm].items():media[f'preview_tokens/{arm}/{k}']=v
        else:
            for level,values in token_metrics[arm].items():
                for k,v in values.items():media[f'preview_tokens/{arm}/{level}/{k}']=v
        full_val=root/arm/'validation'/f'step-{step:07d}.json'
        if full_val.exists():media[f'validation128/{arm}/visqol_audio48k']=read(full_val)['visqol']
    for name in ['laser','rvq','released_mdctcodec']:
        for metric in ['visqol_audio48k','visqol_speech16k','pesq_wb16k','stoi16k','preview_lsd_db',
                       'log_mel_mae_db','rms_gain_db','snr_db','waveform_mae','waveform_endpoint_fraction']:
            media[f'preview/{name}/{metric}']=float(np.mean([row['models'][name][metric] for row in rows]))
    media.update({f'figures/{k}':v for k,v in galleries.items()})
    media.update({f'audio/{k}':v for k,v in audios.items()})
    media['audio/examples']=table
    report={'generator_updates':step,'snapshot':provenance,'examples':rows,'token_metrics':token_metrics,
            'released_reference':read(cache/'complete.json'),
            'scope':'Eight fixed validation utterances; diagnostic previews, not the locked test or a SOTA evaluation.'}
    write(target/'report.json',report)
    run.log(media)
    artifact=wandb.Artifact(f'mdctcodec-audio-media-{run.id}',type='audio-evaluation',
                            metadata={'generator_updates':step,'snapshot':provenance,'preview_files':len(paths)})
    artifact.add_dir(str(target),name='media')
    for file in ['manifest.json','protocol.json']:
        artifact.add_file(str(output/file),name=file)
    artifact.add_file(str(cache/'complete.json'),name='released_reference_provenance.json')
    logged=run.log_artifact(artifact,aliases=['latest',f'update-{step:07d}']).wait()
    report['artifact']=logged.qualified_name
    write(target/'published.json',{'artifact':logged.qualified_name,'generator_updates':step,'published_utc':now()})
    run.summary.update({'latest_media_generator_updates':step,'latest_media_artifact':logged.qualified_name,
                        'latest_media_completed_epochs':snapshot['laser']['completed_epochs']})
    print('AUDIO_MEDIA_PUBLISHED',json.dumps({'step':step,'artifact':logged.qualified_name,'url':run.url}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=REPO/'outputs/mdctcodec_matched_6kbps_rangefix')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--poll-seconds',type=int,default=60)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--start-epoch',type=int)
    args=parser.parse_args();os.chdir(REPO)
    torch.set_num_threads(4)
    root=args.root.resolve();output=(args.output or root/'audio_media').resolve();output.mkdir(parents=True,exist_ok=True)
    # Only one observer may write this dashboard/history, including after restart.
    import fcntl
    lock=(output/'observer.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=read(root/'protocol.json');manifest=read(root/'manifest.json')
    assert sha(root/'manifest.json')==protocol['manifest_sha256']
    paths=select_validation_examples(manifest)
    for path in paths:assert sha(path)==manifest['evaluation_file_sha256'][path]
    parents={}
    for arm in ['laser','rvq']:
        record=read(root/arm/'run.json')
        parents[arm]={**record,'entity':'helloimlixin-rutgers','project':'laser',
                      'path':'helloimlixin-rutgers/laser/'+record['id']}
    media_protocol={'parent_runs':parents,'selection':'Lexically first validation recording per heldout speaker',
        'cadence':'Every five completed epochs, from newest common snapshot at observer startup; final checkpoint too',
        'sample_rate':SAMPLE_RATE,'n_fft':N_FFT,'hop_length':HOP,'mel_bins':MEL_BINS,
        'spectra':'Hann STFT; |STFT/sum(window)|^2 then triangular HTK mel filtering; 10log10 power, floor 1e-12',
        'color_limits_db':[-100,0],'mel_error_limits_db':[-30,30],
        'mdct':'Signed orthonormal sine-window MDCT, 80-sample window / 40-sample hop',
        'mdct_color_limits':'Symmetric reference p99.5 absolute coefficient per fixed clip; shared across models/epochs',
        'audio_levels':'Original levels; no peak/RMS normalization, alignment, or per-recording coefficient calibration',
        'preview_lsd':'Mean over frames of RMS difference of STFT log-power bins, including DC/Nyquist; floor -120 dB',
        'scope':'Validation diagnostics only. Released checkpoint is a separate pretrained reference.',
        'paper':'https://arxiv.org/html/2411.00464v1#S4.F4',
        'audio_demos':'https://pb20000090.github.io/MDCTCodecSLT2024/',
        'source_sha256':{str(p.relative_to(REPO)):sha(p) for p in [Path(__file__),REPO/'src/audio_research_media.py']}}
    write(output/'manifest.json',{'validation_paths':paths,'sha256':{p:sha(p) for p in paths},
                                'parent_manifest_sha256':protocol['manifest_sha256']})
    write(output/'protocol.json',media_protocol)
    previous=read(output/'run.json') if (output/'run.json').exists() else {}
    run=wandb.init(entity='helloimlixin-rutgers',project='laser',mode='online',
        id=previous.get('id'),resume='must' if previous else None,dir=str(output),
        name=protocol['name']+'-audio-visualizations',group=protocol['group'],job_type='audio-media',
        config=media_protocol,tags=['mdctcodec','6kbps','audio','spectrograms','matched','validation'])
    run.define_metric('media/generator_updates')
    for key in ['preview/*','preview_tokens/*','validation128/*','figures/*','audio/*']:
        run.define_metric(key,step_metric='media/generator_updates')
    write(output/'run.json',{'id':run.id,'url':run.url,'pid':os.getpid(),'started_utc':now(),'parents':parents})
    api=wandb.Api()
    for parent in parents.values():
        remote=api.run(parent['path'])
        # The live training SDK owns summary and may replace external additions.
        # Config and notes keep the link without competing with metric logging.
        remote.config['audio_media_run_url']=run.url
        remote.config['audio_media_policy']='Fixed validation previews every five matched epochs'
        if run.url not in (remote.notes or ''):
            remote.notes=(remote.notes or '')+'\nAudio previews and research figures: '+run.url
        remote.update()
    source=wandb.Artifact(f'mdctcodec-audio-media-protocol-{run.id}',type='experiment')
    for path in [output/'protocol.json',output/'manifest.json',Path(__file__),REPO/'src/audio_research_media.py']:
        source.add_file(str(path),name=path.name)
    run.log_artifact(source).wait()
    cache=reference_cache(output,paths,REPO/'outputs/mdctcodec_reference/MDCTCodec')
    state=read(output/'state.json') if (output/'state.json').exists() else {'start_step':None,'published':[]}
    while True:
        try:
            api=wandb.Api()
            available=available_snapshots(api,root,parents,protocol)
            if state['start_step'] is None and available:
                state['start_step']=(args.start_epoch*protocol['batches_per_epoch'] if args.start_epoch is not None else max(available))
                write(output/'state.json',state)
            pending=[step for step in available if step >= (state['start_step'] or 0) and step not in state['published']]
            for step in pending:
                publish_snapshot(run,root,output,paths,available[step],step,cache)
                state['published'].append(step);write(output/'state.json',state)
            write(output/'heartbeat.json',{'time_utc':now(),'published':state['published'],
                                          'available_common_steps':list(available),'status':'watching'})
            run.summary['status']='watching_checkpoints'
            if (output/'last_error.json').exists():
                (output/'last_error.json').rename(output/'recovered_error.json')
                run.summary['last_media_error']=None
            if args.once and state['published']:break
            if protocol['generator_updates'] in state['published']:break
            if (root/'pair_halted.json').exists():
                raise RuntimeError('Parent training pair halted; media observer stopping')
        except Exception as error:
            write(output/'last_error.json',{'time_utc':now(),'error':str(error),'traceback':traceback.format_exc()})
            print(traceback.format_exc(),flush=True)
            run.summary['last_media_error']=str(error)
            if args.once or (root/'pair_halted.json').exists():raise
        time.sleep(max(10,args.poll_seconds))
    run.summary['status']='complete'
    run.finish()


if __name__=='__main__':main()
