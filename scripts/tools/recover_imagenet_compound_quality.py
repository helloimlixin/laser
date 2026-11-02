"""Measure sampler changes on fixed weights, then resume retained production state."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from omegaconf import OmegaConf

BASE=Path('/tmp/laser-imagenet-compound512-audit-20261026')
Q=BASE/'quality'
OUT=Path('/workspace/Projects/laser/outputs/imagenet-compound512-audited-8a100-20261026/quality')
SOURCE=BASE/'speedup/source'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,default=str)+'\n');tmp.replace(path)


def status(phase,**value):
    value=dict(phase=phase,supervisor_pid=os.getpid(),time=time.time(),**value)
    write(Q/'status.json',value);write(OUT/'status.json',value)
    print(json.dumps(value),flush=True)


def env():
    sys.path.insert(0,str(SOURCE/'scripts/tools'))
    from tune_imagenet_compound_execution import environment
    return environment(4,True)


def execute(policy,samples):
    folder=Q/f'{policy}-{samples}'
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8',
        str(Q/'diagnose.py'),'evaluate','--checkpoint',str(Q/'evaluation-checkpoint.pt'),
        '--output',str(folder),'--samples',str(samples),'--policy',policy]
    log=Q/f'{policy}-{samples}.log'
    with log.open('a') as f:
        child=subprocess.Popen(command,cwd=SOURCE,env=env(),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        while child.poll() is None:
            status('quality_evaluation',policy=policy,samples=samples,child_pid=child.pid,log=str(log))
            time.sleep(10)
    if child.returncode:raise RuntimeError(f'Quality evaluation failed: {log}')
    return json.loads((folder/'result.json').read_text())


def resume(selection):
    cfg=OmegaConf.load(BASE/'speedup/production.yaml')
    cfg.options.resume=True
    if selection:
        cfg.options.update(selection['sampling'])
    path=Q/'production.yaml';OmegaConf.save(cfg,path)
    shutil.copyfile(path,OUT/'production.yaml')
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8',
             str(SOURCE/'train.py'),'--config',str(path)]
    with (Q/'production.log').open('a') as log:
        child=subprocess.Popen(command,cwd=SOURCE,env=env(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        while child.poll() is None:
            row=json.loads((BASE/'train/metrics.jsonl').read_text().splitlines()[-1])
            status('training',child_pid=child.pid,sampler=None if selection is None else selection['sampling'],
                latest_metrics={k:row[k] for k in ['train/global_step','train/loss','train/images_per_second']})
            time.sleep(10)
        if child.returncode:raise RuntimeError(f'Production exited {child.returncode}; see {Q}/production.log')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    status('waiting_for_quality_diagnostics')
    while not (Q/'handoff.json').exists() or not (Q/'trained-diagnostics.json').exists():
        if (Q/'diagnostics-failed.json').exists():break
        time.sleep(5)
    selection=None
    try:
        diagnostics=json.loads((Q/'trained-diagnostics.json').read_text())
        shutil.copyfile(Q/'trained-diagnostics.json',OUT/'trained-diagnostics.json')
        if not diagnostics['cache_parity_passed']:raise RuntimeError('Trained-cache parity failed; do not adopt sampler changes')
        screens=[execute(policy,5000) for policy in ['current','prior_validated','lower_atom_temperature']]
        write(Q/'screening.json',screens);write(OUT/'screening.json',screens)
        candidate=min(screens[1:],key=lambda r:r['fid'])
        if candidate['fid']<screens[0]['fid']-2:
            confirmed=execute(candidate['policy'],50000)
            selection=confirmed if confirmed['fid']<confirmed['stored_baseline_fid50k']-2 else None
            comparison=dict(checkpoint_step=confirmed['checkpoint_step'],
                baseline_fid50k=confirmed['stored_baseline_fid50k'],
                candidate_fid50k=confirmed['fid'],candidate_inception=confirmed['inception_score'],
                candidate_policy=confirmed['policy'],sampling=confirmed['sampling'],adopted=selection is not None,
                same_weights=True,same_reference=True,same_seed=261027,generated_per_policy=50000,
                baseline='Recorded epoch-10 full-FID result from the exact retained checkpoint',
                production_resume_step=json.loads((Q/'handoff.json').read_text())['receipt']['global_step'])
            write(Q/'comparison.json',comparison);write(OUT/'comparison.json',comparison)
        else:
            write(Q/'comparison.json',dict(adopted=False,reason='No sampler passed the screening improvement threshold'))
        write(Q/'decision-ready.json',dict(adopted=selection is not None))
    except Exception as error:
        write(Q/'diagnosis-failed.json',dict(error=repr(error)))
        status('diagnostic_failed_retaining_original_recipe',error=repr(error))
    finally:
        resume(selection)


if __name__=='__main__':main()
