#!/usr/bin/env python3
"""Freeze old-test, fresh-test and longer-segment protocols before evaluation."""
import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np
import soundfile as sf

ROOT = Path('outputs/mdctcodec_fair_comparison')
SPEAKERS = ['p360','p361','p362','p363','p364','p374','p376','s5']


def main():
    ROOT.mkdir(exist_ok=True)
    target = ROOT/'manifest.json'
    if target.exists():
        print('Using frozen manifest',target); return
    old = json.loads(Path('outputs/mdctcodec_benchmark_vctk200/manifest.json').read_text())
    forbidden, exclusion_sources = set(), []
    # Paths moved between clusters; compare canonical recording filenames.
    for path in sorted(Path('outputs').rglob('*.json')):
        if not any(k in str(path) for k in ['mdctcodec','vctk_mdct']) or 'tts' in str(path): continue
        if 'manifest' not in path.name and not path.name.startswith('validation'): continue
        if ROOT in path.parents: continue
        content = path.read_text()
        found = re.findall(r'(?:p\d{3}|s5)_\d+_mic2\.(?:flac|wav)',content)
        if found:
            forbidden.update(p.rsplit('.',1)[0] for p in found)
            exclusion_sources.append({'path':str(path),'sha256':hashlib.sha256(content.encode()).hexdigest()})
    audio_root=Path('/workspace/Projects/data/vctk/wav48_silence_trimmed')
    # Explicitly reserve historical loss-validation files as well.
    all_heldout=sorted(p for s in SPEAKERS for p in (audio_root/s).glob('*_mic2.flac'))
    forbidden.update(p.stem for p in all_heldout[:64])
    rng=random.Random(2026091202)
    items=[]
    refs=ROOT/'references';refs.mkdir(exist_ok=True)
    def add(path,split,speaker,item_id=None,sources=None):
        data,rate=sf.read(path,dtype='float32');assert rate==48000 and data.ndim==1
        item_id=item_id or f'{split}-{path.stem}'
        output=refs/(item_id+'.wav');sf.write(output,data,48000,subtype='FLOAT')
        items.append({'id':item_id,'split':split,'speaker':speaker,'reference':str(output.resolve()),
                      'source_paths':sources or [str(path.resolve())],'samples':len(data),
                      'reference_sha256':hashlib.sha256(output.read_bytes()).hexdigest()})
    for path in old['test']:
        p=Path(path);add(p,'locked200',p.parent.name)
    selected_fresh=set()
    for speaker in SPEAKERS:
        pool=[p for p in sorted((audio_root/speaker).glob('*_mic2.flac')) if p.stem not in forbidden and sf.info(p).duration>=2]
        rng.shuffle(pool)
        assert len(pool)>=45,(speaker,len(pool))
        for path in pool[:25]:
            add(path,'fresh200',speaker);selected_fresh.add(path.stem)
        remaining=iter(pool[25:])
        for i in range(4):
            chunks=[];sources=[];n=0
            while n<8*48000:
                path=next(remaining);data,rate=sf.read(path,dtype='float32');assert rate==48000
                chunks.extend([data,np.zeros(4800,dtype=np.float32)])
                n+=len(data)+4800;sources.append(str(path.resolve()))
            speech=np.concatenate(chunks)[:8*48000]
            data=np.pad(speech,(24000,24000))
            path=refs/f'long9s-{speaker}-{i:02d}.wav';sf.write(path,data,48000,subtype='FLOAT')
            add(path,'long9s',speaker,item_id=f'long9s-{speaker}-{i:02d}',sources=sources)
    assert not selected_fresh&forbidden
    long_sources={Path(p).stem for r in items if r['split']=='long9s' for p in r['source_paths']}
    assert not long_sources&(selected_fresh|forbidden)
    validation_checkpoints={}
    for name,path in [('original','outputs/vctk_mdctcodec_stage1_6kbps/final_evaluation.json'),
                      ('low_lr','outputs/vctk_mdctcodec_stage1_6kbps_low_lr/final_evaluation.json')]:
        state=json.loads(Path(path).read_text())
        validation_checkpoints[name]={k:state[k] for k in ['checkpoint','validation_visqol']}
        validation_checkpoints[name]['sha256']=hashlib.sha256(Path(state['checkpoint']).read_bytes()).hexdigest()
    manifest={'seed':2026091202,'sample_rate':48000,'items':items,
              'fresh_excluded_recordings':len(forbidden),'excluded_stems':sorted(forbidden),
              'exclusion_sources':exclusion_sources,'codec_candidates':validation_checkpoints,
              'stage2_codec_selection':'original: higher original validation ViSQOL; fixed before this test',
              'protocol':{'primary':'fresh200: 25 full utterances per speaker; excludes recovered scored/validation files',
                          'replication':'locked200: original 200-file comparison',
                          'duration_sensitivity':'long9s: 4 segments per speaker; concatenate unused full recordings with 0.1s gaps, take first 8s, add 0.5s silence each end',
                          'metrics':['ViSQOL audio48k','ViSQOL speech16k','PESQ wideband16k','STOI16k'],
                          'selection':'all candidates and settings frozen before test scoring',
                          'comparison_scope':'released/checkpoint systems, not equal-budget architecture ablation'}}
    target.write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'counts':{s:sum(r['split']==s for r in items) for s in ['locked200','fresh200','long9s']},
                      'excluded':len(forbidden),'manifest_sha256':hashlib.sha256(target.read_bytes()).hexdigest()},indent=2))


if __name__=='__main__':main()
