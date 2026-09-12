#!/usr/bin/env python3
"""After both budgets complete, compare validation-selected models on locked test."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import signal
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
os.environ.setdefault('VISQOL_BINARY',str(Path('outputs/visqol/bin/visqol').resolve()))
import numpy as np
import soundfile as sf
import torch
import wandb
from archive.scripts.benchmark_mdctcodec_vctk import align_mdct,load_reference
from archive.scripts.benchmark_mdctcodec_fair import measure_record
from archive.scripts.benchmark_mdctcodec_trained_rvq import payload_roundtrip
from archive.scripts.compare_mdctcodec_recovered_recipe import paired_comparison
from src.mdctcodec_matched import reconstruct_serialized
from src.models.laser import LASER


def running(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1][0]!='Z'
    except FileNotFoundError:return False


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('outputs/mdctcodec_matched_6kbps'))
    p.add_argument('--pids',type=int,nargs='*',default=[])
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--workers',type=int,default=8)
    args=p.parse_args()
    root=args.root
    protocol=json.loads((root/'protocol.json').read_text())
    while any(running(pid) for pid in args.pids):
        failures=[str(root/arm/'failure.json') for arm in ['laser','rvq'] if (root/arm/'failure.json').exists()]
        if failures and protocol.get('stop_peer_on_failure'):
            for pid in args.pids:
                cmdline=Path(f'/proc/{pid}/cmdline')
                if running(pid) and cmdline.exists() and b'train_mdctcodec_matched.py' in cmdline.read_bytes():
                    os.kill(pid,signal.SIGINT)
            (root/'pair_halted.json').write_text(json.dumps({'failures':failures,'reason':'Stop peer after arm failure'},indent=2))
            raise RuntimeError('Paired training halted: '+str(failures))
        time.sleep(30)
    completed={arm:json.loads((root/arm/'completion.json').read_text()) for arm in ['laser','rvq']}
    protocol=json.loads((root/'protocol.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    manifest_hash=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()
    for record in completed.values():
        assert record['status']=='complete' and record['generator_updates']==protocol['generator_updates']
        assert record['manifest_sha256']==manifest_hash
    streams={arm:[json.loads(line) for line in (root/arm/'data_order.jsonl').read_text().splitlines()]
             for arm in completed}
    if streams['laser']!=streams['rvq']:
        raise RuntimeError('Data-order audit differs between arms: do not claim matched examples')
    assert sum(r['batches'] for r in streams['laser'])==protocol['generator_updates']
    output=root/'comparison';output.mkdir(exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    run=wandb.init(entity='helloimlixin-rutgers',project='laser',mode='online',
        name=protocol['name']+'-200k-final',job_type='benchmark',
        group=protocol.get('group','mdctcodec-matched-scratch-6kbps-20260912'),dir=str(output),
        config={'protocol':protocol,'selected_models':completed,'manifest_sha256':manifest_hash})
    summaries,records={},{}
    for name in ['rvq','laser','released_mdctcodec']:
        system_dir=output/name;system_dir.mkdir(exist_ok=True)
        if name=='released_mdctcodec':
            encoder,quantizer,decoder=load_reference(Path('outputs/mdctcodec_reference/MDCTCodec'),args.device)
        else:
            model=LASER.load_from_checkpoint(completed[name]['best_checkpoint'],map_location='cpu',strict=True).to(args.device).eval()
        rows,jobs=[],[]
        with torch.inference_mode(),ThreadPoolExecutor(max_workers=args.workers) as pool:
            for path in manifest['test']:
                assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==manifest['evaluation_file_sha256'][path]
                reference,rate=sf.read(path,dtype='float32');assert rate==48000
                x=torch.from_numpy(reference).to(args.device)[None,None]
                if name=='released_mdctcodec':
                    x,_=align_mdct(x);ids=quantizer(encoder(x),n_quantizers=4)[1]
                    payload,parsed=payload_roundtrip(ids.cpu().numpy())
                    y=decoder(quantizer.from_codes(torch.from_numpy(parsed).to(args.device))[0])[...,:len(reference)].clamp(-1,1)
                else:y,payload=reconstruct_serialized(model,x)
                decoded=y[0,0].float().cpu().numpy()
                assert decoded.shape==reference.shape and np.isfinite(decoded).all()
                target=system_dir/(Path(path).stem+'.wav')
                sf.write(target,decoded,48000,subtype='FLOAT')
                target.with_suffix('.bin').write_bytes(payload)
                meta={'id':Path(path).stem,'utterance':Path(path).name,'speaker':Path(path).parent.name,
                      'system':name,'payload_bits':len(payload)*8,'samples':len(reference)}
                jobs.append(pool.submit(measure_record,path,str(target),meta))
            rows=[job.result() for job in jobs]
        records[name]=rows
        summaries[name]={k:float(np.mean([r[k] for r in rows])) for k in
            ['visqol_audio48k','visqol_speech16k','pesq_wb16k','stoi16k']}
        summaries[name]['payload_kbps']=sum(r['payload_bits'] for r in rows)/(sum(r['samples'] for r in rows)/48000)/1000
        (system_dir/'results.json').write_text(json.dumps(rows,indent=2))
        run.log({f'comparison/{name}/{k}':v for k,v in summaries[name].items()})
        if name!='released_mdctcodec':del model
        torch.cuda.empty_cache()
    paired={name:{metric:paired_comparison(
        [{**r,'visqol_audio48k':r[metric]} for r in records['laser']],
        [{**r,'visqol_audio48k':r[metric]} for r in records[name]])
        for metric in ['visqol_audio48k','visqol_speech16k','pesq_wb16k','stoi16k']}
        for name in ['rvq','released_mdctcodec']}
    result={'results':summaries,'laser_minus_reference':paired,'data_order_exact_match':True,
            'protocol_sha256':hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest(),
            'selected_models':completed,'manifest_sha256':manifest_hash,
            'limitations':'One paired seed; bottleneck-specific initialization, auxiliary losses and parameter counts differ. '
                          'Released weights have their own training history. No comprehensive SOTA claim.'}
    (output/'results.json').write_text(json.dumps(result,indent=2))
    lines=['# Matched MDCTCodec RVQ–LASER experiment at 6 kbps','',
           'Both fresh arms completed 200,000 generator updates with identical common initialization and audited batches/crops. '
           'Checkpoints were selected only by validation ViSQOL. Test recordings were frozen before training.','',
           '| Model | Payload kbps | ViSQOL audio48 | ViSQOL speech16 | PESQ-WB | STOI16 |',
           '|---|---:|---:|---:|---:|---:|']
    for name,r in summaries.items():
        lines.append('| '+name+' | '+' | '.join(f'{r[k]:.4f}' for k in
            ['payload_kbps','visqol_audio48k','visqol_speech16k','pesq_wb16k','stoi16k'])+' |')
    lines.extend(['',result['limitations'],'','Paired speaker-bootstrap intervals are in results.json.'])
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')
    artifact=wandb.Artifact(f'mdctcodec-matched-comparison-{run.id}',type='benchmark')
    for name in ['laser','rvq','released_mdctcodec']:
        artifact.add_dir(str(output/name),name='comparison/'+name)
    for name in ['results.json','comparison.md']:
        artifact.add_file(str(output/name),name='comparison/'+name)
    for path in [root/'manifest.json',root/'protocol.json',root/'laser/data_order.jsonl',root/'rvq/data_order.jsonl',Path(__file__)]:
        artifact.add_file(str(path),name=str(path.relative_to(root)) if root in path.parents else path.name)
    artifact=run.log_artifact(artifact).wait()
    (root/'comparison_complete.json').write_text(json.dumps({'url':run.url,'artifact':artifact.qualified_name},indent=2))
    run.finish();print(json.dumps(result,indent=2))


if __name__=='__main__':main()
