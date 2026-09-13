#!/usr/bin/env python3
"""Check real two-GPU LR reductions and resume using isolated synthetic FID events."""
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
ROOT=Path(__file__).resolve().parents[2]


def main():
    base=ROOT/'outputs/church-joint-geometry-20260912';reports=base/'fid-monitor'
    source=reports/'pre-monitor-last.pt'
    saved=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
    start=saved['step'];initial_config=saved['config'];del saved
    output=Path('/tmp/church-fid-monitor-verification')
    if output.exists():raise FileExistsError(output)
    result=json.loads((base/'joint/evaluations/step-007395/screen/metrics.json').read_text())
    for step in (8000,9000):
        directory=output/f'evaluations/step-{step:06d}/screen';directory.mkdir(parents=True)
        (directory/'metrics.json').write_text(json.dumps({**result,'fid':23.6,'optimizer_step':step})+'\n')
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1','OMP_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'8',
        'MKL_NUM_THREADS':'8','TORCH_HOME':'/workspace/tmp/official-rqvae-eval-cache',
        'LASER_VGG16_WEIGHTS':'/workspace/tmp/laser-vgg/vgg16-397923af.pth'}
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
        str(ROOT/'scripts/tools/train_church_joint_distributed.py'),'--output',str(output),
        '--microbatch','32','--generation-batch','512','--verification-skip-evaluation']
    for label,extra in [('initial',['--source-checkpoint',str(source),'--fid-lr-policy',str(reports/'policy.json'),
                                    '--stop-after-step',str(start+2)]),
                        ('resume',['--stop-after-step',str(start+4)])]:
        with (reports/f'verify-{label}.log').open('w') as log:
            subprocess.run(command+extra,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        checkpoint=torch.load(output/'last.pt',map_location='cpu',weights_only=False,mmap=True)
        controller=checkpoint['fid_lr_state']
        assert controller['reductions']==1 and controller['multiplier']==.5 and controller['last_step']==9000
        assert controller['cooldown_remaining']==2 and checkpoint['config']==initial_config
        assert {int(v['step']) for v in checkpoint['optimizer']['state'].values()}=={checkpoint['step']}
        print(json.dumps({'phase':label,'step':checkpoint['step'],'multiplier':controller['multiplier'],
            'reductions':controller['reductions'],'lr':checkpoint['optimizer']['param_groups'][0]['lr']}),flush=True)
        del checkpoint
    rows=[json.loads(line) for line in (output/'history.jsonl').read_text().splitlines()]
    updates=[r for r in rows if r.get('phase')=='train']
    assert [r['optimizer_step'] for r in updates]==list(range(start+1,start+5))
    assert all(r['train/lr']==r['train/base_lr']*.5 for r in updates)
    assert len([r for r in rows if r.get('lr_monitor/decision')=='reduced'])==1
    production=torch.load(base/'joint/last.pt',map_location='cpu',weights_only=False,mmap=True)
    assert production['step']==start and production.get('fid_lr_state') is None
    receipt={'passed':True,'source_step':start,'verification_final_step':start+4,'world_size':2,
        'synthetic_metric_events_only_in':str(output),'production_checkpoint_unchanged':True,
        'halved_lr_applied_to_four_actual_updates':True,'saved_reduction_not_repeated_on_resume':True,
        'optimizer_progress_preserved':True,'recipe_config_unchanged':True}
    (reports/'full-model-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
