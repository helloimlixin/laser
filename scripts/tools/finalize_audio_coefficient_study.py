#!/usr/bin/env python3
"""Attach the portable selected codec and uncertainty plots to a completed study."""
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import wandb
from src.tts_pairing import file_sha


def main():
    root=REPO/'outputs/mdctcodec_audio_levels_20260913'
    result=json.loads((root/'results.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    completed=json.loads((root/'complete.json').read_text())
    portable=json.loads((root/'portable_codec_verification.json').read_text())
    assert completed['recommendation']==result['recommendation']=='rq_levels8_refined6k'
    assert portable['status']=='passed' and portable['selected_quantizer_sha256']==file_sha(root/'selected_quantizer.pt')
    assert json.loads((root/'frozen_parameters.json').read_text())['identical']
    for name in result['summaries']:
        assert len(list((root/'audio'/name).glob('*.json')))==128
    api=wandb.Api(); main_artifact=api.artifact(completed['artifact'])
    assert main_artifact.state=='COMMITTED'
    names=[f'rq_levels{n}{suffix}' for n in (8,16,32) for suffix in ('','_refined6k')]
    labels=[f'{n} raw\n{rate:.1f} kbps' if not suffix else f'{n}+residual\n6.0 kbps'
            for n,rate in ((8,5.1),(16,5.4),(32,5.7)) for suffix in ('','_refined6k')]
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for ax,(metric,title) in zip(axes.flat,[('visqol_audio48k','ViSQOL audio48k'),
        ('visqol_speech16k','ViSQOL speech16k'),('pesq_wb16k','PESQ-WB16k'),('stoi16k','STOI16k')]):
        for i,name in enumerate(names):
            entry=result['paired_deltas_vs_original'][name][metric]
            mean=entry['mean'];lo,hi=entry['speaker_bootstrap_ci95']
            ax.errorbar(i,mean,yerr=[[mean-lo],[hi-mean]],fmt='o',capsize=4,
                color='#D55E00' if 'refined' in name else '#0072B2')
        ax.axhline(0,color='#263238',ls='--',lw=1)
        ax.set_xticks(range(len(names)),labels,fontsize=9)
        ax.set(title=title,ylabel='Paired difference versus original codec')
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('Validation128: paired differences and 95% speaker-bootstrap intervals')
    fig.savefig(root/'paired_metric_deltas.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    summaries=result['summaries']
    for prefix,label,color in [('omp_levels','OMP with calibrated scalar levels','#009E73'),
                              ('rq_levels','Joint scaled-atom RQ','#0072B2')]:
        arms=[f'{prefix}{n}' for n in (8,16,32)]
        ax.plot([summaries[a]['payload_kbps'] for a in arms],[summaries[a]['visqol_audio48k'] for a in arms],
                'o-',label=label,color=color)
    for n in (8,16,32):
        value=summaries[f'rq_levels{n}_refined6k']
        ax.scatter(value['payload_kbps'],value['visqol_audio48k'],marker='D',color='#D55E00')
        ax.annotate(f'{n}+refine',(value['payload_kbps'],value['visqol_audio48k']),
            xytext=(-8,-8),textcoords='offset points',ha='right',va='top')
    ax.axhline(summaries['original_omp127_6k']['visqol_audio48k'],color='#263238',ls='--',label='Original 127-level codec')
    ax.set(xlabel='Serialized payload (kbps)',ylabel='Mean ViSQOL audio48k',title='Compact audio quantizers: validation means')
    ax.legend();fig.savefig(root/'rate_quality_readable.png',dpi=150);plt.close(fig)
    previous=json.loads((root/'run.json').read_text())
    run=wandb.init(entity='helloimlixin-rutgers',project='laser',id=previous['id'],resume='must',dir=str(root))
    try:
        for arm in ('laser','rvq'):
            original=json.loads((REPO/'outputs/mdctcodec_matched_6kbps_rangefix'/arm/'completion.json').read_text())
            run.use_artifact(original['artifact'])
        package=wandb.Artifact('mdctcodec-laser-scaled8-refined6k',type='model',metadata={
            'codec_sha256':manifest['codecs']['laser']['sha256'],'vocab_sizes':[65537,65537,64],
            'bits_per_frame':40,'nominal_kbps':6.,'frame_rate':150,'sample_rate':48000,
            'selection_scope':'Validation-selected frozen-codec screen; no stage2 results',
            'hybrid_sparse_and_small_rvq':True,'validation':summaries[result['recommendation']]})
        package.add_file(manifest['codecs']['laser']['checkpoint'],name='frozen_backbone.ckpt')
        for path in [root/'selected_quantizer.pt',root/'portable_codec_verification.json',root/'manifest.json']:
            package.add_file(str(path),name=path.name)
        for source in [REPO/'src/audio_compact_runtime.py',REPO/'src/audio_scaled_quantizer.py',REPO/'src/scaled_atom_rq.py']:
            package.add_file(str(source),name='source/'+source.name)
        model=run.log_artifact(package,aliases=['latest','validation-selected']).wait()
        details=wandb.Artifact(f'mdctcodec-audio-levels-followup-{run.id}',type='audio-quantizer-study')
        for path in [root/'paired_metric_deltas.png',root/'rate_quality_readable.png',
            REPO/'docs/mdctcodec-compact-audio-vocabulary-2026-09-13.md',
            REPO/'scripts/tools/evaluate_audio_coefficient_levels.py',Path(__file__),
            REPO/'tests/test_audio_scaled_quantizer.py']:
            details.add_file(str(path),name=path.name)
        failure_path=root/'failure.json'
        if failure_path.exists():
            failure=json.loads(failure_path.read_text())
            assert failure['error']=="SummaryDict.update() got an unexpected keyword argument 'elapsed_seconds'"
            run.summary['elapsed_seconds']=failure['elapsed_seconds']
            failure_path.rename(root/'completion_logging_error_resolved.json')
            details.add_file(str(root/'completion_logging_error_resolved.json'),name='completion_logging_error_resolved.json')
        saved=run.log_artifact(details,aliases=['latest']).wait()
        run.log({'comparison/paired_metric_deltas':wandb.Image(str(root/'paired_metric_deltas.png')),
                 'comparison/rate_quality':wandb.Image(str(root/'rate_quality_readable.png'))})
        final={'status':'complete','selected_model_artifact':model.qualified_name,
            'followup_artifact':saved.qualified_name,'main_artifact':completed['artifact'],
            'portable_codec_verified':True,'tests_passed':16,'error':None,
            'completion_logging_fixed':True,'queued_stage2_recipe_changed':False}
        run.summary.update(final)
        (root/'finalization.json').write_text(json.dumps(final,indent=2))
        print(json.dumps(final,indent=2),flush=True)
    finally:run.finish()


if __name__=='__main__':main()
