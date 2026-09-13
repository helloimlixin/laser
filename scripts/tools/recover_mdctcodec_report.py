#!/usr/bin/env python3
"""Verify and publish a completed matched evaluation after an upload-only failure."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
import numpy as np
import soundfile as sf
import wandb


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args();root=args.root.resolve();output=root/'comparison'
    protocol,manifest,result=read(root/'protocol.json'),read(root/'manifest.json'),read(output/'results.json')
    assert sha(root/'protocol.json')==result['protocol_sha256']
    assert sha(root/'manifest.json')==result['manifest_sha256']==protocol['manifest_sha256']
    streams={arm:[json.loads(line) for line in (root/arm/'data_order.jsonl').read_text().splitlines()] for arm in ['laser','rvq']}
    assert streams['laser']==streams['rvq'] and sum(r['batches'] for r in streams['laser'])==200000
    selected={}
    for arm in ['laser','rvq']:
        complete=read(root/arm/'completion.json')
        assert complete==result['selected_models'][arm]
        assert complete['generator_updates']==200000 and complete['status']=='complete'
        selected[arm]={'checkpoint':complete['best_checkpoint'],'sha256':sha(complete['best_checkpoint']),
                       'artifact':complete['artifact']}
    expected={Path(p).stem:p for p in manifest['test']}
    for path in manifest['test']:assert sha(path)==manifest['evaluation_file_sha256'][path]
    for system in ['laser','rvq','released_mdctcodec']:
        records=read(output/system/'results.json')
        assert len(records)==200 and {r['id'] for r in records}==set(expected)
        for r in records:
            wav=output/system/(r['id']+'.wav');payload=wav.with_suffix('.bin')
            assert sha(wav)==r['decoded_sha256']
            info=sf.info(wav)
            assert info.samplerate==48000 and info.channels==1 and info.frames==r['samples']
            assert payload.stat().st_size*8==r['payload_bits']
        for metric in ['visqol_audio48k','visqol_speech16k','pesq_wb16k','stoi16k']:
            values=[r[metric] for r in records]
            assert np.isfinite(values).all()
            assert abs(float(np.mean(values))-result['results'][system][metric])<1e-10
        rate=sum(r['payload_bits'] for r in records)/(sum(r['samples'] for r in records)/48000)/1000
        assert abs(rate-result['results'][system]['payload_kbps'])<1e-10
    recovery=root/'report_recovery';recovery.mkdir(exist_ok=True)
    with tarfile.open(root/'source.tar.gz') as archive:
        source=archive.extractfile('scripts/report_mdctcodec_matched.py').read()
    original=recovery/'original_report_mdctcodec_matched.py';original.write_bytes(source)
    verification={'kind':'Upload-only recovery; all existing evaluation outputs verified, no checkpoint reselection',
        'cause':'Source moved to archive while the original reporter process still held its old __file__ path',
        'selected_models':selected,'verified_systems':3,'verified_recordings_per_system':200,
        'data_order_exact_match':True,'original_reporter_sha256':sha(original),'source_archive_sha256':sha(root/'source.tar.gz')}
    (recovery/'verification.json').write_text(json.dumps(verification,indent=2))
    run=wandb.init(entity='helloimlixin-rutgers',project='laser',id=args.run_id,resume='must',mode='online',dir=str(recovery))
    for item in selected.values():run.use_artifact(item['artifact'])
    artifact=wandb.Artifact(f'mdctcodec-matched-comparison-{run.id}',type='benchmark',metadata=verification)
    for system in ['laser','rvq','released_mdctcodec']:
        artifact.add_dir(str(output/system),name='comparison/'+system)
    for name in ['results.json','comparison.md']:artifact.add_file(str(output/name),name='comparison/'+name)
    for path in [root/'manifest.json',root/'protocol.json',root/'laser/data_order.jsonl',root/'rvq/data_order.jsonl']:
        artifact.add_file(str(path),name=str(path.relative_to(root)))
    for path in [original,recovery/'verification.json',Path(__file__)]:artifact.add_file(str(path),name='recovery/'+path.name)
    logged=run.log_artifact(artifact,aliases=['latest','complete']).wait()
    complete={'status':'complete','url':run.url,'artifact':logged.qualified_name,'recovered_upload':True}
    (root/'comparison_complete.json').write_text(json.dumps(complete,indent=2))
    run.summary.update(complete)
    run.summary['recovery_reason']=verification['cause']
    run.finish();print(json.dumps(complete,indent=2))


if __name__=='__main__':main()
