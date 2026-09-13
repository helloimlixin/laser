#!/usr/bin/env python3
"""Frozen-checkpoint validation probes; float/dense modes are not 6 kbps codecs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO))
os.environ.setdefault('VISQOL_BINARY',str(REPO/'outputs/visqol/bin/visqol'))
import numpy as np
import torch
import wandb
from archive.scripts.compare_mdctcodec_recovered_recipe import evaluate,paired_comparison


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=REPO/'outputs/mdctcodec_matched_6kbps_rangefix')
    p.add_argument('--device',default='cuda:0');args=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    root=args.root;output=root/'gap_diagnostics';output.mkdir(exist_ok=True)
    manifest=json.loads((root/'manifest.json').read_text())
    complete=json.loads((root/'laser/completion.json').read_text());checkpoint=complete['best_checkpoint']
    paths=manifest['validation'];assert len(paths)==128 and not set(paths)&set(manifest['test'])
    for path in paths:assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==manifest['evaluation_file_sha256'][path]
    rows={};summary={}
    for mode in ['current_q7','current_float_diagnostic','current_dense_diagnostic']:
        rows[mode],rate=evaluate(checkpoint,paths,torch.device(args.device),mode,6)
        summary[mode]={'visqol_audio48k':float(np.mean([r['visqol_audio48k'] for r in rows[mode]])),
                       'stoi48k':float(np.mean([r['stoi'] for r in rows[mode]])),'payload_kbps':rate}
        (output/f'{mode}.json').write_text(json.dumps(rows[mode],indent=2))
        print('PROBE_RESULT',mode,json.dumps(summary[mode]),flush=True)
    result={'checkpoint':checkpoint,'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        'validation_paths':paths,'results':summary,'minus_q7':{mode:paired_comparison(rows[mode],rows['current_q7']) for mode in rows if mode!='current_q7'},
        'interpretation_limits':['All weights frozen; 128 validation files, no test adaptation.',
            'Float removes coefficient quantization at inference; it is not a 6 kbps codec.',
            'Dense bypass changes the decoder input distribution and is not an achievable quality upper bound.',
            'These probes cannot establish which alternative training method would perform best.']}
    (output/'results.json').write_text(json.dumps(result,indent=2))
    run=wandb.init(entity='helloimlixin-rutgers',project='laser',mode='online',dir=str(output),
        name='mdctcodec-matched-rangefix-frozen-gap-diagnostics',group='mdctcodec-matched-scratch-6kbps-rangefix-20260912',
        job_type='validation-diagnostics',config={'checkpoint':checkpoint,'checkpoint_sha256':result['checkpoint_sha256']})
    artifact=wandb.Artifact(f'mdctcodec-gap-diagnostics-{run.id}',type='benchmark',metadata={'scope':'Frozen validation probes'})
    for path in output.glob('*.json'):artifact.add_file(str(path),name=path.name)
    artifact.add_file(str(Path(__file__)),name=Path(__file__).name)
    published=run.log_artifact(artifact).wait()
    for mode,values in summary.items():run.summary[mode+'/visqol_audio48k']=values['visqol_audio48k']
    (output/'run.json').write_text(json.dumps({'url':run.url,'artifact':published.qualified_name},indent=2))
    run.finish();print(json.dumps(result,indent=2),flush=True)
