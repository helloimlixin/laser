#!/usr/bin/env python3
"""Audit DAC inference paths on validation audio; never select on test audio."""
import json
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import soundfile as sf
from torchaudio.functional import resample
from audiotools import AudioSignal
import dac

from archive.scripts.benchmark_mdctcodec_vctk import measure


def main():
    torch.set_num_threads(4)
    manifest = json.loads(Path('outputs/mdctcodec_benchmark_vctk200/manifest.json').read_text())
    groups = {}
    for path in manifest['validation']:
        groups.setdefault(Path(path).parent.name, []).append(path)
    selected = [paths[i] for i in range(2) for _, paths in sorted(groups.items())]
    assert not set(selected).intersection(manifest['test'])
    root = Path('outputs/mdctcodec_fairness_audit'); root.mkdir(exist_ok=True)
    (root/'validation_manifest.json').write_text(json.dumps({'validation':selected},indent=2))
    model = dac.DAC.load(dac.utils.download(model_type='44khz',model_bitrate='8kbps')).to('cuda:1').eval()
    jobs = []
    with ThreadPoolExecutor(max_workers=4) as pool, torch.inference_mode():
        for path in selected:
            ref, sr = sf.read(path,dtype='float32'); assert sr==48000
            x = torch.from_numpy(ref)[None,None].to('cuda:1')
            for mode in ['raw_encode_decode','official_full_utterance','official_default_chunks']:
                if mode == 'raw_encode_decode':
                    x44=resample(x,48000,44100)
                    encoded=model.encode(model.preprocess(x44,44100),n_quantizers=7)
                    reconstructed=model.decode(encoded[0])[...,:x44.shape[-1]]
                    y=resample(reconstructed,44100,48000)[...,:len(ref)]
                    bits=encoded[1].numel()*10
                else:
                    signal=AudioSignal(x,48000)
                    encoded=model.compress(signal,n_quantizers=7,
                                           win_duration=None if mode=='official_full_utterance' else 1.)
                    y=model.decompress(encoded).audio_data
                    bits=encoded.codes.numel()*10
                out=y[0,0].cpu().numpy().clip(-1,1)
                assert out.shape==ref.shape and np.isfinite(out).all()
                future=pool.submit(measure,(Path(path).name,ref,out))
                jobs.append((mode,bits,len(ref)/48000,future))
            print('AUDITED',Path(path).name,flush=True)
        rows=[]
        for mode,bits,seconds,future in jobs:
            rows.append({'mode':mode,'payload_bits':bits,'seconds':seconds,**future.result()})
    summary=[]
    for mode in sorted({r['mode'] for r in rows}):
        data=[r for r in rows if r['mode']==mode]
        summary.append({'mode':mode,'utterances':len(data),
                        'visqol':float(np.mean([r['visqol_audio48k'] for r in data])),
                        'stoi':float(np.mean([r['stoi'] for r in data])),
                        'payload_kbps':sum(r['payload_bits'] for r in data)/sum(r['seconds'] for r in data)/1000})
    result={'scope':'16 validation utterances / eight speakers; diagnostic only',
            'checkpoint':'DAC 44.1 kHz standard 8 kbps release, seven codebooks',
            'bitrate_excludes':'headers and per-utterance loudness metadata',
            'summary':summary,'rows':rows}
    (root/'dac_inference_paths.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
