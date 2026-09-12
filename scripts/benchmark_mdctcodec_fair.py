#!/usr/bin/env python3
"""Evaluate serialized codecs on a frozen manifest with shared metric inputs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from pesq import pesq
from pystoi import stoi

from src.audio_logging import _measure_visqol,is_visqol_available


def initialize_metrics():torch.set_num_threads(1)


def measure_record(reference_path,decoded_path,metadata):
    reference,sr=sf.read(reference_path,dtype='float32')
    decoded,rate=sf.read(decoded_path,dtype='float32')
    assert sr==rate==48000 and reference.shape==decoded.shape
    r16=resample_poly(reference,1,3).astype(np.float32)
    d16=resample_poly(decoded,1,3).astype(np.float32)
    a=_measure_visqol(torch.from_numpy(reference),torch.from_numpy(decoded),sample_rate=48000,mode='audio')
    s=_measure_visqol(torch.from_numpy(r16),torch.from_numpy(d16),sample_rate=16000,mode='speech')
    if a is None or s is None or not np.isfinite([a,s]).all():
        raise RuntimeError('Official ViSQOL failed for '+metadata['id'])
    p=float(pesq(16000,r16,d16,'wb'))
    t=float(stoi(r16,d16,16000))
    return {**metadata,'visqol_audio48k':float(a),'visqol_speech16k':float(s),
            'pesq_wb16k':p,'stoi16k':t,
            'decoded_sha256':hashlib.sha256(Path(decoded_path).read_bytes()).hexdigest()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('outputs/mdctcodec_fair_comparison'))
    parser.add_argument('--systems',nargs='+',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--limit',type=int,default=0,help='Preflight only; uses validation audio, never partial test')
    args=parser.parse_args()
    os.environ.setdefault('VISQOL_BINARY',str(Path('outputs/visqol/bin/visqol').resolve()))
    assert is_visqol_available()
    torch.set_num_threads(4)
    torch.cuda.set_device(torch.device(args.device))
    # Match the prior benchmark's default FP32 arithmetic.
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True
    from src.fair_audio_codecs import CodecAdapter
    manifest=json.loads((args.root/'manifest.json').read_text())
    manifest_hash=hashlib.sha256((args.root/'manifest.json').read_bytes()).hexdigest()
    items=manifest['items']
    if args.limit:
        old=json.loads(Path('outputs/mdctcodec_benchmark_vctk200/manifest.json').read_text())
        items=[{'id':'preflight-'+Path(p).stem,'reference':p,'split':'validation_preflight',
                'speaker':Path(p).parent.name,'samples':sf.info(p).frames} for p in old['validation'][:args.limit]]
    for name in args.systems:
        output=args.root/('preflight' if args.limit else 'systems')/name
        output.mkdir(parents=True,exist_ok=True)
        adapter=CodecAdapter(name,manifest,args.device)
        (output/'model.json').write_text(json.dumps(adapter.info,indent=2))
        started=time.monotonic();jobs=[];rows=[]
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize_metrics) as pool:
            for i,item in enumerate(items):
                record_path=output/(item['id']+'.json')
                if record_path.exists():
                    row=json.loads(record_path.read_text())
                    assert row['manifest_sha256']==manifest_hash
                    rows.append(row);continue
                reference,sr=sf.read(item['reference'],dtype='float32');assert sr==48000
                decoded,blob,rates=adapter.reconstruct(reference,item['id'])
                path=output/(item['id']+'.wav');sf.write(path,decoded,48000,subtype='FLOAT')
                path.with_suffix('.codec').write_bytes(blob)
                metadata={**item,**rates,'system':name,'manifest_sha256':manifest_hash,
                          'seconds':len(reference)/48000,'decoder_input':'serialized codes and declared metadata only'}
                jobs.append((record_path,pool.submit(measure_record,item['reference'],str(path),metadata)))
                if len(jobs)>=args.workers*2:
                    p,future=jobs.pop(0);row=future.result();p.write_text(json.dumps(row,indent=2));rows.append(row)
                if (i+1)%25==0:print(name,'reconstructed',i+1,'/',len(items),'seconds',round(time.monotonic()-started,1),flush=True)
            for p,future in jobs:
                row=future.result();p.write_text(json.dumps(row,indent=2));rows.append(row)
        summaries=[]
        for split in sorted({r['split'] for r in rows}):
            data=[r for r in rows if r['split']==split];duration=sum(r['seconds'] for r in data)
            summary={'system':name,'split':split,'n':len(data),
                     **{k:float(np.mean([r[k] for r in data])) for k in ['visqol_audio48k','visqol_speech16k','pesq_wb16k','stoi16k']},
                     **{k.replace('_bits','_kbps'):sum(r[k] for r in data)/duration/1000 for k in ['payload_bits','decoder_value_side_bits','serialized_file_bits']}}
            summaries.append(summary)
        (output/'summary.json').write_text(json.dumps(summaries,indent=2))
        (output/'complete.json').write_text(json.dumps({'records':len(rows),'manifest_sha256':manifest_hash,'seconds':time.monotonic()-started}))
        print('COMPLETE',json.dumps(summaries),flush=True)
        del adapter;torch.cuda.empty_cache()


if __name__=='__main__':main()
