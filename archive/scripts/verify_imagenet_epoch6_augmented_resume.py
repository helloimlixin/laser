"""Verify real-image updates and exact recovery of the epoch-six warm start."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import yaml

BASE = Path('/tmp/laser-imagenet-epoch6-aug-stage2')
EVIDENCE = Path('/workspace/Projects/laser/outputs/imagenet-rfid421-epoch6-augcosine-5h200-20261002')


def main():
    ready = json.loads(Path('/tmp/laser-imagenet-stage2/imagenet/training-ready.json').read_text())
    audit = json.loads((BASE/'fresh-augmentation-audit.json').read_text())
    assert ready['training_images'] == 1281167 and audit['acceptable']
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4', LASER_RUNTIME_ROOT=str(BASE/'runtime'),
        LASER_RUN_BASE=str(BASE), LASER_PERSISTENT_BASE=str(EVIDENCE), LASER_LOCAL_PREFLIGHT='1',
        LASER_ACCUMULATION='2',LASER_COMPILE_BLOCKS='1',LASER_COMPILE_OBJECTIVE='1',
        OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',TORCHINDUCTOR_COMPILE_THREADS='2',
        TORCHINDUCTOR_CACHE_DIR=str(BASE/'inductor-cache'),NCCL_NVLS_ENABLE='0',
        TORCH_HOME='/tmp/laser-imagenet-stage2/torch-cache',PYTHONUNBUFFERED='1')
    recipe = yaml.safe_load((BASE/'recipe.yaml').read_text())
    options = recipe['options']
    options.update(output=str(BASE/'preflight/train'),checkpoint_dir=str(BASE/'preflight/train/checkpoints'),
        wandb_mode='disabled',sample_grid_every=0,fid_every=0,save_step_freq=0,upload_checkpoints=False,
        resume=True,smoke_test=True)
    records=[]
    for phase,steps,source in [('preflight',20,BASE/'epoch6-fresh-adam.pt'),
                               ('preflight_resume',2,BASE/'preflight/train/checkpoints/last.pt')]:
        options.update(max_optimizer_steps=steps,resume_checkpoint=str(source))
        config=BASE/(phase+'-recipe.yaml')
        config.write_text(yaml.safe_dump(recipe,sort_keys=False))
        env['LASER_PHASE']=phase
        command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=5',
                 str(BASE/'entry-production.py'),'--config',str(config)]
        started=time.monotonic()
        print(json.dumps(dict(phase=phase,status='starting')),flush=True)
        with (BASE/(phase+'.log')).open('w') as log:
            completed=subprocess.run(command,env=env,cwd=BASE/'runtime',stdout=log,stderr=subprocess.STDOUT)
        record=dict(phase=phase,exit_code=completed.returncode,seconds=time.monotonic()-started)
        records.append(record)
        (BASE/'preflight-progress.json').write_text(json.dumps(records,indent=2))
        print(json.dumps(record),flush=True)
        assert completed.returncode == 0, 'Inspect isolated preflight log before production'
    payload=torch.load(BASE/'preflight/train/checkpoints/last.pt',map_location='cpu',mmap=True,weights_only=False)
    assert payload['global_step'] == 3778 and payload['epoch'] == 6 and payload['batch_idx'] == 44
    assert len(payload['optimizer']['state']) == 782
    assert {int(s['step']) for s in payload['optimizer']['state'].values()} == {22}
    assert len(payload['rng_state_by_rank']) == 5
    assert payload['scheduler']['last_epoch'] == 3778 and payload['scheduler']['T_max'] == 62600
    assert payload['config']['token_cache'] is None and not payload['config']['coefficient_clipping']
    expected_lr=3e-7+.5*(.00066-3e-7)*(1+math.cos(math.pi*3778/62600))
    assert abs(payload['optimizer']['param_groups'][0]['lr']-expected_lr)<1e-12
    fresh=[json.loads((BASE/'preflight/verification'/f'startup-rank{rank}.json').read_text()) for rank in range(5)]
    finite=[json.loads((BASE/'preflight/verification'/f'step20-rank{rank}.json').read_text()) for rank in range(5)]
    resumed=[json.loads((BASE/'preflight_resume/verification'/f'startup-rank{rank}.json').read_text()) for rank in range(5)]
    assert all(r['fresh'] and r['optimizer_states']==0 and r['optimizer_step_before']==0 for r in fresh)
    assert all(r['finite'] and r['optimizer_step']==20 for r in finite)
    assert all(not r['fresh'] and r['optimizer_step_before']==20 for r in resumed)
    assert 'restored exact per-rank checkpoint streams' in (BASE/'preflight_resume.log').read_text()
    proof=dict(passed=True,phases=records,fresh=fresh,finite_step20=finite,resumed=resumed,
        source_epoch=6,source_global_step=3756,source_optimizer_reset=True,checkpoint_step=3778,
        new_adam_steps=22,scheduler_step=3778,current_lr=expected_lr,total_scheduler_steps=62600,
        fresh_online_training_images=True,token_cache_used=False,world_size=5,global_batch=2048,
        production_loads_preflight_weights=False)
    for path in (BASE/'preflight-summary.json',EVIDENCE/'preflight-summary.json'):
        path.write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(dict(passed=True,checkpoint_step=3778,new_adam_steps=22,current_lr=expected_lr)),flush=True)


if __name__ == '__main__':
    main()
