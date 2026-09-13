#!/usr/bin/env python3
"""Evaluate the two priors only after their complete training audits agree."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np
import torch
from src.tts_pairing import file_sha
from scripts.tools.benchmark_mdctcodec_tts import aggregate, write_json


def verify_pair(root):
    protocol=json.loads((root/'protocol.json').read_text())
    completed={a:json.loads((root/a/'completion.json').read_text()) for a in ('laser','rvq')}
    audits={a:[json.loads(line) for line in (root/a/'data_order.jsonl').read_text().splitlines()]
            for a in completed}
    for arm, state in completed.items():
        assert state['completed_epochs']==protocol['train']['epochs'], 'Both arms must finish the frozen epoch budget'
        assert state['best_generation'], 'No validation-selected checkpoint'
        assert len(audits[arm])==state['step']
        assert [r['step'] for r in audits[arm]]==list(range(1,state['step']+1))
        assert audits[arm][-1]['chain']==state['data_chain']
        config=json.loads((root/arm/'resolved_config.json').read_text())
        assert config['paired']['protocol_sha256']==file_sha(root/'protocol.json')
        assert config['train']==protocol['train'] and config['seed']==protocol['seed']
    assert audits['laser']==audits['rvq'], 'Examples, ordering, accumulation, updates or LR diverged'
    return protocol, completed


def select(root):
    protocol, completed=verify_pair(root)
    manifest=json.loads((root/'benchmark/manifest.json').read_text())
    assert file_sha(root/'benchmark/manifest.json')==protocol['benchmark_manifest_sha256']
    assert len(manifest['items'])==len({r['target']['text_key'] for r in manifest['items']})==100
    selection_path=root/'benchmark/selection.json'
    selection=json.loads(selection_path.read_text()) if selection_path.exists() else {}
    for arm, completion in completed.items():
        best=min(completion['best_generation'], key=lambda x:(x['score'],x['epoch']))
        for kind, source in [('best_wer',Path(best['path'])),('endpoint',root/arm/'checkpoints/last.pt')]:
            name=f'{arm}_{kind}'; target=root/'benchmark/selected'/f'{name}.pt'
            if target.exists():
                if name not in selection:
                    raise RuntimeError(f'Incomplete checkpoint freeze at {target}; inspect before recovery')
                frozen=selection[name]
                assert frozen['sha256']==file_sha(target)
                selection[name]=frozen; continue
            state=torch.load(source,map_location='cpu',weights_only=False)
            assert state['metadata']['codec_sha256']==protocol['arms'][arm]['codec_sha256']
            slim={k:state[k] for k in ('model','model_config','metadata','step','epoch')}
            for key in ('validation_wer','validation_nll'):
                if key in state: slim[key]=state[key]
            target.parent.mkdir(parents=True,exist_ok=True); torch.save(slim,target)
            selection[name]={'checkpoint':str(target.resolve()),'sha256':file_sha(target),
                'source_sha256':file_sha(source),'step':state['step'],
                'selected_epoch':best['epoch'] if kind=='best_wer' else completion['completed_epochs'],
                'validation_wer':best['score'] if kind=='best_wer' else None}
            write_json(selection_path,selection)
    write_json(root/'benchmark/selection.json',selection)
    return protocol,selection


def report(root):
    import wandb
    protocol,completed=verify_pair(root)
    bench=root/'benchmark'; manifest=json.loads((bench/'manifest.json').read_text())
    selection=json.loads((bench/'selection.json').read_text())
    names=['reference','codec_laser','codec_rvq','laser_best_wer','rvq_best_wer','laser_endpoint','rvq_endpoint']
    rows={arm:[json.loads((bench/'scores'/arm/(r['id']+'.json')).read_text()) for r in manifest['items']] for arm in names}
    for arm, items in rows.items():
        assert len(items)==100 and len({r['id'] for r in items})==100
        assert all(r['manifest_sha256']==file_sha(bench/'manifest.json') for r in items)
    summaries={arm:aggregate(items) for arm,items in rows.items()}
    for arm,items in rows.items():
        if arm!='reference':
            summaries[arm]['payload_kbps']=sum(r['payload_bytes'] for r in items)*8/sum(r['seconds'] for r in items)/1000
    indices=np.random.default_rng(20260913).integers(0,100,size=(5000,100))
    samples={}
    for arm,items in rows.items():
        errors=np.array([r['word_errors'] for r in items]); words=np.array([r['words'] for r in items])
        samples[arm]={'wer':errors[indices].sum(1)/words[indices].sum(1)}
        for key in ('utmos','speaker_similarity_ecapa'):
            samples[arm][key]=np.array([r[key] for r in items])[indices].mean(1)
        summaries[arm]['ci95']={k:np.percentile(v,[2.5,97.5]).tolist() for k,v in samples[arm].items()}
    differences={}
    for kind in ('best_wer','endpoint'):
        a,b=f'laser_{kind}',f'rvq_{kind}'
        differences[kind]={k:{'delta':summaries[a][k]-summaries[b][k],
            'ci95':np.percentile(samples[a][k]-samples[b][k],[2.5,97.5]).tolist()} for k in samples[a]}
    result={'matched_training_audit_passed':True,'protocol_sha256':file_sha(root/'protocol.json'),
        'manifest_sha256':file_sha(bench/'manifest.json'),'items':100,'distinct_texts':100,
        'completed_epochs':completed['laser']['completed_epochs'],'updates_per_arm':completed['laser']['step'],
        'parameters':{a:protocol['arms'][a]['parameters'] for a in ('laser','rvq')},
        'summaries':summaries,'laser_minus_rvq':differences,'selection':selection,'limitations':protocol['limitations']}
    write_json(bench/'results.json',result)
    lines=['# Controlled VCTK LASER vs RVQ TTS at 6 kbps','',
        'Both fresh priors completed the same 160 epochs, with identical audited batches, update counts, learning rates, and shared-backbone initialization. Primary checkpoints were selected by validation-only Whisper WER.','',
        '| System | WER % | CER % | UTMOS | ECAPA | RTF |','|---|---:|---:|---:|---:|---:|']
    for arm,v in summaries.items():
        lines.append(f'| {arm} | {100*v["wer"]:.2f} | {100*v["cer"]:.2f} | {v["utmos"]:.3f} | {v["speaker_similarity_ecapa"]:.3f} | {v["rtf"]:.3f} |')
    lines+=['',*protocol['limitations'],'','Paired speaker-bootstrap intervals and all per-file results accompany the report.']
    (bench/'comparison.md').write_text('\n'.join(lines)+'\n')
    run=wandb.init(entity='helloimlixin-rutgers',project='laser',name='mdctcodec-6kbps-paired-rqtransformer-test100',
        group=protocol['name'],job_type='paired-tts-benchmark',dir=str(bench),config=protocol)
    run.summary.update({k:v for k,v in result.items() if k not in ('selection','summaries')})
    for arm,v in summaries.items(): run.log({f'comparison/{arm}/{k}':x for k,x in v.items() if k!='ci95'})
    table=wandb.Table(columns=['system','speaker','text','hypothesis','WER','UTMOS','audio','reference'])
    for arm,items in rows.items():
        for i,r in enumerate(items):
            table.add_data(arm,r['speaker'],r['text'],r['asr_text'],r['wer'],r['utmos'],
                wandb.Audio(r['audio_path']),wandb.Audio(manifest['items'][i]['target']['path']))
    run.log({'comparison/listening_samples':table})
    artifact=wandb.Artifact(f'mdctcodec-paired-tts-benchmark-{run.id}',type='benchmark',metadata=result)
    for name in ('results.json','comparison.md','manifest.json','selection.json','metric_provenance.json'):
        artifact.add_file(str(bench/name),name=name)
    for name in ('generated','scores','models'):
        artifact.add_dir(str(bench/name),name=name)
    artifact.add_file(str(root/'protocol.json'),name='protocol.json')
    for arm in ('laser','rvq'):
        artifact.add_file(str(root/arm/'data_order.jsonl'),name=f'{arm}_data_order.jsonl')
    for source in [Path(__file__),REPO/'scripts/tools/benchmark_mdctcodec_tts.py']:
        artifact.add_file(str(source),name='source/'+source.name)
    saved=run.log_artifact(artifact,aliases=['latest']).wait()
    write_json(bench/'complete.json',{'url':run.url,'artifact':saved.qualified_name,'matched_training_audit_passed':True})
    run.finish()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('outputs/mdctcodec_tts_paired'))
    p.add_argument('--device',default='cuda:0'); args=p.parse_args()
    root=args.root; bench=root/'benchmark'
    if (bench/'complete.json').exists():
        print('Benchmark already complete'); return
    protocol,selection=select(root)
    (bench/'models').mkdir(exist_ok=True)
    for source in Path('outputs/mdctcodec_tts_benchmark/models').glob('*.json'):
        target=bench/'models'/source.name
        if not target.exists(): shutil.copy2(source,target)
    benchmark='scripts/tools/benchmark_mdctcodec_tts.py'
    names=['reference','codec_laser','codec_rvq',*selection]
    for arm in names:
        codec='rvq' if 'rvq' in arm else 'laser'
        command=[sys.executable,'-u',benchmark,'--phase','generate','--root',str(bench),
            '--arm',arm,'--cache',protocol['arms'][codec]['cache'],'--device',args.device]
        if arm in selection: command+=['--checkpoint',selection[arm]['checkpoint']]
        subprocess.run(command,check=True)
    subprocess.run(['/workspace/tts-benchmark-env/bin/python','-u',benchmark,'--phase','score',
        '--root',str(bench),'--arm',','.join(names),'--device',args.device],check=True)
    report(root)


if __name__=='__main__': main()
