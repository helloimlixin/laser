#!/usr/bin/env python3
"""Calibrate compact LASER audio vocabularies and screen on validation only."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault('VISQOL_BINARY', str(REPO/'outputs/visqol/bin/visqol'))
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F
import wandb

from src.audio_scaled_quantizer import (fit_residual_centers, nearest_centers, pack_integer_frames,
    unpack_integer_frames, rate_spec, sparse_to_joint)
from src.scaled_atom_rq import ScaledAtomRQ, continuous_matching_pursuit, fit_signed_levels
from src.models.laser import LASER
from src.mdctcodec_bitstream import pack_frames, unpack_frames
from src.mdctcodec_matched import reconstruct_serialized
from src.tts_pairing import file_sha, state_sha
from src.audio_research_media import render_audio_comparison, LABELS, COLORS
from archive.scripts.benchmark_mdctcodec_vctk import align_mdct
from archive.scripts.benchmark_mdctcodec_fair import measure_record
from archive.scripts.compare_mdctcodec_recovered_recipe import paired_comparison


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)); temp.replace(path)


def parameter_sha(model):
    return state_sha({name: value.detach().cpu() for name, value in model.named_parameters()})


def prepare(root, calibration_count, smoke):
    original = REPO/'outputs/mdctcodec_matched_6kbps_rangefix'
    source = json.loads((original/'manifest.json').read_text())
    tts = json.loads((REPO/'outputs/mdctcodec_tts_rangefix/cache/inventory.json').read_text())
    by_speaker = defaultdict(list)
    train_paths = set(source['train'])
    for r in tts['records']:
        if r['split'] == 'train' and r['path'] in train_paths: by_speaker[r['speaker']].append(r)
    for records in by_speaker.values():
        records.sort(key=lambda r: hashlib.sha256(f'20260913:audio-levels:{r["path"]}'.encode()).hexdigest())
    calibration = [records[i]['path'] for i in range(max(map(len, by_speaker.values())))
                   for speaker, records in sorted(by_speaker.items()) if i < len(records)][:calibration_count]
    assert len(calibration) == calibration_count
    validation = source['validation']
    if smoke:
        validation = [min(p for p in validation if Path(p).parent.name == s)
                      for s in sorted(source['heldout_speakers'])][:2]
    assert not set(calibration) & (set(source['validation']) | set(source['test']))
    selected = {a: json.loads((original/a/'completion.json').read_text()) for a in ('laser', 'rvq')}
    codecs = {a: {'checkpoint': s['best_checkpoint'], 'sha256': file_sha(s['best_checkpoint']),
                 'training_budget': s['generator_updates'], 'validation_visqol': s['best_validation_visqol']}
              for a, s in selected.items()}
    manifest = {'version': 1, 'seed': 20260913, 'smoke': smoke, 'codecs': codecs,
        'calibration': calibration, 'validation': validation,
        'source_manifest_sha256': file_sha(original/'manifest.json'),
        'file_sha256': {p: file_sha(p) for p in calibration + validation},
        'levels': [8, 16, 32], 'levels_definition': 'Nonzero signed levels, plus one canonical joint zero token',
        'fit': 'Symmetric scalar Lloyd on continuous MP2 coefficients from full training utterances; common levels for OMP scalar and joint residual searches',
        'refinement': 'Training-only residual vector codebook with a fixed zero entry; 64/16/4 entries after two 17/18/19-bit joint codes, respectively. This is a hybrid sparse-plus-small-RVQ candidate.',
        'precision': 'CUDA FP32; matmul TF32 disabled, cuDNN TF32 enabled, same for every arm',
        'rate': '150 frames/s; raw packed integers, at most7 final byte-padding bits per utterance. Codebooks and original sample count are metadata; no per-frame padding used to claim6 kbps.',
        'controls': ['original_omp127_6k', 'continuous_omp2', 'continuous_mp2', 'matched_rvq_6k'],
        'screen': 'Fixed stage1 validation128,16 utterances per8 held-out speakers; final codec and TTS test audio is untouched',
        'gate': {'max_mean_visqol_loss': .05, 'max_ci95_visqol_loss': .10,
                 'max_mean_pesq_loss': .05, 'max_mean_stoi_loss': .005,
                 'rule': 'Among nominal6k hybrid candidates satisfying all gates, prefer the smallest joint vocabulary. Otherwise retain original codec. Exploratory validation screen, not a formal noninferiority or SOTA result.'},
        'limitations': ['Frozen codec post-training quantizer changes; no encoder/decoder adaptation.',
            'One codec checkpoint,8 validation speakers; calibration and variant selection are not an independent test.',
            'Adding a small residual vector codebook changes the bottleneck; it is labeled separately from pure scaled-atom quantization.',
            'Continuous-coefficient controls have no finite transmitted rate. Smaller raw vocabularies are labeled at their actual lower rates.'],
        'queued_stage2_recipe_changed': False}
    path = root/'manifest.json'
    if path.exists(): assert json.loads(path.read_text()) == manifest
    else: write(path, manifest)
    return manifest


@torch.inference_mode()
def encoded(model, path, device):
    wave, rate = sf.read(path, dtype='float32'); assert rate == 48000 and wave.ndim == 1
    x, length = align_mdct(torch.from_numpy(wave).to(device)[None, None])
    latent = model._to_bottleneck_input(model.pre_bottleneck(model.encoder(x)))
    z = latent[0, :, 0].T.contiguous()
    return wave, x, z, length


@torch.inference_mode()
def chunk_quantize(quantizer, z):
    parts = [quantizer.quantize(chunk) for chunk in z.split(2048)]
    return {key: torch.cat([p[key] for p in parts]) for key in ('codes', 'quantized')}


@torch.inference_mode()
def calibrate(model, dictionary, manifest, root, device):
    path = root/'codebooks.pt'
    if path.exists():
        saved = torch.load(path, map_location=device, weights_only=True)
        assert saved['manifest_sha256'] == file_sha(root/'manifest.json')
        torch.testing.assert_close(saved['dictionary'], dictionary, rtol=0, atol=0)
        return saved
    signals, coefficients = [], []
    for i, path in enumerate(manifest['calibration']):
        assert file_sha(path) == manifest['file_sha256'][path]
        _, _, z, _ = encoded(model, path, device)
        mp = continuous_matching_pursuit(z, dictionary, depth=2)
        signals.append(z); coefficients.append(mp['coefficients'])
        if (i+1) % 32 == 0 or i+1 == len(manifest['calibration']): print('CALIBRATED_AUDIO', i+1, flush=True)
    signals = torch.cat(signals); coefficients = torch.cat(coefficients)
    levels, refinements, diagnostics = {}, {}, {}
    for count in manifest['levels']:
        fitted = fit_signed_levels(coefficients, count)
        quantizer = ScaledAtomRQ(dictionary, fitted, depth=2)
        result = chunk_quantize(quantizer, signals)
        residual = signals - result['quantized']
        rate = rate_spec(count, refine=True)
        centers = fit_residual_centers(residual, rate['refinement_entries'])
        ids = nearest_centers(residual, centers)
        levels[str(count)] = fitted.cpu(); refinements[str(count)] = centers.cpu()
        diagnostics[str(count)] = {'levels': fitted.cpu().tolist(), **rate,
            'training_latent_mse_base': float(residual.square().mean()),
            'training_latent_mse_refined': float((residual-centers[ids]).square().mean())}
        print('FIT_LEVELS', count, json.dumps(diagnostics[str(count)]), flush=True)
    spec = {'manifest_sha256': file_sha(root/'manifest.json'), 'dictionary': dictionary.cpu(),
            'levels': levels, 'refinements': refinements, 'diagnostics': diagnostics,
            'coefficient_quantiles': torch.quantile(coefficients.abs().flatten(), coefficients.new_tensor([0.,.5,.9,.99,.999,1.])).cpu().tolist()}
    torch.save(spec, root/'codebooks.pt')
    write(root/'calibration.json', {k:v for k,v in spec.items() if k not in ('dictionary','levels','refinements')})
    fig, ax = plt.subplots(figsize=(10,4), layout='constrained')
    ax.hist(coefficients.cpu().numpy().ravel(), bins=160, density=True, color='#0072B2', alpha=.65)
    for i,(count, values) in enumerate(levels.items()):
        ax.scatter(values, np.full(len(values), .005+i*.005), label=f'{count} nonzero levels', marker='|', s=180)
    ax.set(xlabel='Coefficient in physical latent units', ylabel='Training density', title='Training-only coefficient calibration')
    ax.legend(); fig.savefig(root/'coefficient_levels.png', dpi=140); plt.close(fig)
    return {**spec, 'dictionary': dictionary, 'levels': {k:v.to(device) for k,v in levels.items()},
            'refinements': {k:v.to(device) for k,v in refinements.items()}}


def metric_job(source, target, metadata):
    result_path = target.with_suffix('.json')
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        assert saved['manifest_sha256'] == metadata['manifest_sha256']
        assert saved['decoded_sha256'] == file_sha(target)
        return saved
    result = measure_record(source, str(target), metadata)
    write(result_path, result)
    return result


@torch.inference_mode()
def evaluate(model, dictionary, spec, manifest, root, device, workers, run):
    bound = model.bottleneck.coefficient_quantization_max
    quantizers = {n: ScaledAtomRQ(dictionary, spec['levels'][str(n)].to(device), depth=2) for n in manifest['levels']}
    refinements = {n: spec['refinements'][str(n)].to(device) for n in quantizers}
    rvq = LASER.load_from_checkpoint(manifest['codecs']['rvq']['checkpoint'], map_location='cpu').to(device).eval()
    rvq.requires_grad_(False); rvq_hash = parameter_sha(rvq)
    rows, pending = defaultdict(list), []
    manifest_sha = file_sha(root/'manifest.json')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, source in enumerate(manifest['validation']):
            assert file_sha(source) == manifest['file_sha256'][source]
            reference, x, z, length = encoded(model, source, device)
            atoms, coefficients = model.bottleneck.batch_omp_with_support(z.T, dictionary)
            integers = coefficients.clamp(-bound, bound).div(bound/63).round().long()
            blob = pack_frames(atoms.cpu().numpy(), integers.cpu().numpy())
            parsed_atoms, parsed_int = unpack_frames(blob)
            original = (dictionary.T[torch.from_numpy(parsed_atoms).to(device)] *
                        (torch.from_numpy(parsed_int).to(device).float()*(bound/63))[..., None]).sum(-2)
            if i == 0:
                direct, _, source_codes = model.encode(x)
                torch.testing.assert_close(source_codes.support[0,0], atoms, rtol=0, atol=0)
                torch.testing.assert_close(direct[0,:,0].T, original, rtol=1e-5, atol=1e-5)
                write(root/'original_codec_parity.json', {'serialized_latent_matches_source_encode': True,
                    'checkpoint_sha256': manifest['codecs']['laser']['sha256'], 'path': source})
            candidates = {'original_omp127_6k': (original, blob, {'bits_per_frame':40,'nominal_kbps':6.,'kind':'original'})}
            candidates['continuous_omp2'] = ((dictionary.T[atoms]*coefficients[...,None]).sum(-2), None,
                                            {'bits_per_frame':None,'nominal_kbps':None,'kind':'continuous diagnostic'})
            candidates['continuous_mp2'] = (continuous_matching_pursuit(z,dictionary,depth=2)['quantized'], None,
                                            {'bits_per_frame':None,'nominal_kbps':None,'kind':'continuous diagnostic'})
            for n, quantizer in quantizers.items():
                pure = rate_spec(n); refined = rate_spec(n, refine=True)
                scalar = sparse_to_joint(atoms, coefficients, quantizer.levels)
                joint = quantizer.quantize(z)
                for name, codes in [(f'omp_levels{n}',scalar),(f'rq_levels{n}',joint['codes'])]:
                    payload = pack_integer_frames(codes.cpu().numpy(),pure['vocab_sizes'])
                    parsed = torch.from_numpy(unpack_integer_frames(payload,pure['vocab_sizes'],len(z))).to(device)
                    torch.testing.assert_close(parsed,codes,rtol=0,atol=0)
                    decoded = quantizer.embed(parsed).sum(-2)
                    candidates[name] = (decoded,payload,{**pure,'kind':'OMP scalar' if name.startswith('omp') else 'joint scaled RQ'})
                residual = z-joint['quantized']
                correction_ids = nearest_centers(residual,refinements[n])
                codes = torch.cat((joint['codes'],correction_ids[:,None]),1)
                payload = pack_integer_frames(codes.cpu().numpy(),refined['vocab_sizes'])
                parsed = torch.from_numpy(unpack_integer_frames(payload,refined['vocab_sizes'],len(z))).to(device)
                torch.testing.assert_close(parsed,codes,rtol=0,atol=0)
                decoded = quantizer.embed(parsed[:,:2]).sum(-2)+refinements[n][parsed[:,2]]
                candidates[f'rq_levels{n}_refined6k'] = (decoded,payload,{**refined,'kind':'hybrid scaled RQ plus residual vector'})
            for name, (latent,payload,rate) in candidates.items():
                output = root/'audio'/name/(Path(source).stem+'.wav'); output.parent.mkdir(parents=True,exist_ok=True)
                waveform = model.decode(latent.T[None,:,None])[0,0,:length].float().cpu().numpy().clip(-1,1)
                assert waveform.shape == reference.shape and np.isfinite(waveform).all()
                if not output.with_suffix('.json').exists():
                    sf.write(output,waveform,48000,subtype='FLOAT')
                    if payload is not None: output.with_suffix('.bin').write_bytes(payload)
                meta={'id':Path(source).stem,'utterance':Path(source).name,'speaker':Path(source).parent.name,
                    'system':name,'manifest_sha256':manifest_sha,'samples':length,'frames':len(z),
                    'payload_bits':len(payload)*8 if payload is not None else None,
                    'latent_mse':float((z-latent).square().mean()),'waveform_snr_db':float(10*np.log10(
                        (np.square(reference.astype(np.float64)).sum()+1e-12)/
                        (np.square((reference-waveform).astype(np.float64)).sum()+1e-12))),**rate}
                pending.append(pool.submit(metric_job,source,output,meta))
            waveform, payload = reconstruct_serialized(rvq,torch.from_numpy(reference).to(device)[None,None])
            output = root/'audio/matched_rvq_6k'/(Path(source).stem+'.wav'); output.parent.mkdir(parents=True,exist_ok=True)
            if not output.with_suffix('.json').exists():
                sf.write(output,waveform[0,0].cpu().numpy(),48000,subtype='FLOAT'); output.with_suffix('.bin').write_bytes(payload)
            meta={'id':Path(source).stem,'utterance':Path(source).name,'speaker':Path(source).parent.name,
                'system':'matched_rvq_6k','manifest_sha256':manifest_sha,'samples':length,'frames':len(z),
                'payload_bits':len(payload)*8,'bits_per_frame':40,'nominal_kbps':6.,'kind':'matched stage1 RVQ control'}
            pending.append(pool.submit(metric_job,source,output,meta))
            while len(pending)>workers*2:
                result=pending.pop(0).result();rows[result['system']].append(result)
            if (i+1)%8==0 or i+1==len(manifest['validation']):
                print('VALIDATION_AUDIO',i+1,'/',len(manifest['validation']),flush=True)
                run.log({'progress/validation_utterances':i+1})
        for job in pending:
            result=job.result();rows[result['system']].append(result)
    assert parameter_sha(rvq)==rvq_hash
    return dict(rows)


def report(rows, manifest, root, run):
    metrics=['visqol_audio48k','visqol_speech16k','pesq_wb16k','stoi16k']
    summaries,differences={},{}
    for name, records in rows.items():
        assert len(records)==len(manifest['validation'])
        assert {r['utterance'] for r in records}=={Path(p).name for p in manifest['validation']}
        first=records[0]
        summaries[name]={m:float(np.mean([r[m] for r in records])) for m in metrics}
        summaries[name].update(items=len(records),nominal_kbps=first['nominal_kbps'],kind=first['kind'],
            payload_kbps=(sum(r['payload_bits'] for r in records)/(sum(r['samples'] for r in records)/48000)/1000
                          if first['payload_bits'] is not None else None),
            joint_vocab=first.get('vocab_sizes',[None])[0])
        if 'latent_mse' in first:
            summaries[name]['latent_mse']=sum(r['latent_mse']*r['frames'] for r in records)/sum(r['frames'] for r in records)
        differences[name]={m:paired_comparison([{**r,'visqol_audio48k':r[m]} for r in records],
            [{**r,'visqol_audio48k':r[m]} for r in rows['original_omp127_6k']]) for m in metrics}
    gate=manifest['gate'];passes=[]
    for n in manifest['levels']:
        name=f'rq_levels{n}_refined6k';d=differences[name]
        passed=(d['visqol_audio48k']['mean']>=-gate['max_mean_visqol_loss'] and
            d['visqol_audio48k']['speaker_bootstrap_ci95'][0]>=-gate['max_ci95_visqol_loss'] and
            d['pesq_wb16k']['mean']>=-gate['max_mean_pesq_loss'] and d['stoi16k']['mean']>=-gate['max_mean_stoi_loss'])
        summaries[name]['passes_prespecified_screen']=passed
        if passed:passes.append(name)
    recommendation=passes[0] if passes else 'retain_original_omp127_6k'
    if manifest['smoke']:recommendation='smoke_only_no_selection'
    result={'status':'complete','manifest_sha256':file_sha(root/'manifest.json'),'summaries':summaries,
        'paired_deltas_vs_original':differences,'recommendation':recommendation,'gate':gate,
        'limitations':manifest['limitations'],'queued_stage2_recipe_changed':False}
    write(root/'results.json',result)
    lines=['# Compact LASER audio coefficient validation screen','',
        'All encoder, dictionary and decoder parameters were frozen. Levels and residual correction codebooks were fitted only on training audio. Every reported reconstruction used the same validation recordings.','',
        '| Quantizer | Nominal kbps | Actual payload kbps | ViSQOL audio | ViSQOL speech | PESQ-WB | STOI |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,v in summaries.items():
        rate='continuous' if v['nominal_kbps'] is None else f'{v["nominal_kbps"]:.1f}'
        actual='—' if v['payload_kbps'] is None else f'{v["payload_kbps"]:.4f}'
        lines.append(f'| {name} | {rate} | {actual} | '+ ' | '.join(f'{v[m]:.4f}' for m in metrics)+' |')
    lines+=['',f'Prespecified screening recommendation: **{recommendation}**.','',
        'The 6 kbps refined variants add a small residual vector codebook; they are hybrid candidates. Raw8/16/32-level variants retain their lower bitrate labels.','',*manifest['limitations']]
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    for prefix,label,color in [('omp_levels','OMP with calibrated scalar levels','#009E73'),
                               ('rq_levels','Joint scaled-atom RQ','#0072B2')]:
        names=[f'{prefix}{n}' for n in manifest['levels']]
        ax.plot([summaries[n]['payload_kbps'] for n in names],[summaries[n]['visqol_audio48k'] for n in names],
                'o-',label=label,color=color)
    for n in manifest['levels']:
        value=summaries[f'rq_levels{n}_refined6k']
        ax.scatter(value['payload_kbps'],value['visqol_audio48k'],marker='D',color='#D55E00')
        ax.annotate(f'{n}+refine',(value['payload_kbps'],value['visqol_audio48k']),
                    xytext=(-8,-8),textcoords='offset points',ha='right',va='top')
    baseline=summaries['original_omp127_6k']
    ax.axhline(baseline['visqol_audio48k'],color='#263238',ls='--',label='Original127-level codec')
    ax.set(xlabel='Serialized payload (kbps)',ylabel='Mean ViSQOL audio48k',title='Frozen audio codec: coefficient-rate screen')
    ax.legend();fig.savefig(root/'rate_quality.png',dpi=140);plt.close(fig)
    run.summary.update({'recommendation':recommendation,'items':len(manifest['validation']),
                        'status':'complete','queued_stage2_recipe_changed':False})
    for name,value in summaries.items():run.log({f'comparison/{name}/{k}':v for k,v in value.items() if isinstance(v,(float,int))})
    run.log({'comparison/rate_quality':wandb.Image(str(root/'rate_quality.png')),
             'calibration/coefficient_levels':wandb.Image(str(root/'coefficient_levels.png'))})
    # One predetermined validation utterance per speaker, independent of scores.
    preview_paths=[min(p for p in manifest['validation'] if Path(p).parent.name==s)
                   for s in sorted({Path(p).parent.name for p in manifest['validation']})]
    display=['original_omp127_6k','rq_levels8','rq_levels8_refined6k','rq_levels16','rq_levels32']
    LABELS.update({name:name.replace('_',' ') for name in display})
    COLORS.update(dict(zip(display,['#263238','#0072B2','#D55E00','#009E73','#CC79A7'])))
    table=wandb.Table(columns=['speaker','system','text_source','audio','reference'])
    for source in preview_paths:
        stem=Path(source).stem; waves={'reference':sf.read(source,dtype='float32')[0]}
        for name in rows:
            path=root/'audio'/name/(stem+'.wav')
            table.add_data(Path(source).parent.name,name,source,wandb.Audio(str(path)),wandb.Audio(source))
        for name in display:waves[name]=sf.read(root/'audio'/name/(stem+'.wav'),dtype='float32')[0]
        figures,_=render_audio_comparison(waves,root/'figures'/stem,stem)
        run.log({f'audio/{stem}/{key}':wandb.Image(path) for key,path in figures.items()})
    run.log({'audio/listening_comparison':table})
    if not run.disabled:
        artifact=wandb.Artifact(f'mdctcodec-audio-levels-{run.id}',type='audio-quantizer-study',metadata=result)
        for name in ('manifest.json','calibration.json','codebooks.pt','results.json','comparison.md',
                     'original_codec_parity.json','frozen_parameters.json','coefficient_levels.png','rate_quality.png'):
            artifact.add_file(str(root/name),name=name)
        for name in ('audio','figures'):artifact.add_dir(str(root/name),name=name)
        for path in [Path(__file__),REPO/'src/audio_scaled_quantizer.py',REPO/'src/scaled_atom_rq.py']:
            artifact.add_file(str(path),name='source/'+path.name)
        saved=run.log_artifact(artifact,aliases=['latest']).wait()
        result.update(url=run.url,artifact=saved.qualified_name)
    write(root/'complete.json',{k:v for k,v in result.items() if k not in ('summaries','paired_deltas_vs_original')})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('outputs/mdctcodec_audio_levels_20260913'))
    p.add_argument('--device',default='cuda:0');p.add_argument('--workers',type=int,default=8)
    p.add_argument('--calibration-items',type=int,default=256)
    p.add_argument('--smoke',action='store_true');p.add_argument('--mode',choices=['online','disabled'],default='online')
    args=p.parse_args();root=args.root;root.mkdir(parents=True,exist_ok=True)
    if (root/'complete.json').exists():print('Already complete');return
    if (root/'failure.json').exists():
        (root/'failure.json').rename(root/f'failure_before_retry_{time.time_ns()}.json')
    torch.set_num_threads(4);torch.manual_seed(20260913)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    manifest=prepare(root,args.calibration_items,args.smoke)
    previous=json.loads((root/'run.json').read_text()) if (root/'run.json').exists() else None
    run=wandb.init(entity='helloimlixin-rutgers',project='laser',name='mdctcodec-audio-8-16-32-level-validation',
        group='mdctcodec-audio-compact-vocabulary-20260913',job_type='quantizer-validation',dir=str(root),
        mode=args.mode,config=manifest,id=previous['id'] if previous and args.mode=='online' else None,
        resume='must' if previous and args.mode=='online' else None)
    write(root/'run.json',{'id':run.id,'url':run.url,'pid':os.getpid()})
    started=time.monotonic()
    try:
        source=manifest['codecs']['laser']
        assert file_sha(source['checkpoint'])==source['sha256']
        model=LASER.load_from_checkpoint(source['checkpoint'],map_location='cpu').to(args.device).eval()
        model.requires_grad_(False);before=parameter_sha(model)
        dictionary=F.normalize(model.bottleneck.effective_dictionary().detach().float(),dim=0,eps=1e-8)
        assert dictionary.shape==(32,8192) and model.bottleneck.sparsity_level==2
        spec=calibrate(model,dictionary,manifest,root,args.device)
        rows=evaluate(model,dictionary,spec,manifest,root,args.device,args.workers,run)
        after=parameter_sha(model);assert before==after
        write(root/'frozen_parameters.json',{'before_sha256':before,'after_sha256':after,'identical':True})
        report(rows,manifest,root,run)
        run.summary.update({'elapsed_seconds':time.monotonic()-started})
    except BaseException as error:
        write(root/'failure.json',{'error':str(error),'elapsed_seconds':time.monotonic()-started})
        run.summary.update({'status':'failed','error':str(error)});raise
    finally:run.finish()


if __name__=='__main__':main()
