#!/usr/bin/env python3
"""Run verified Church stage1 -> fresh cache/book -> preflight -> fresh stage2."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

from relaunch_original_church_rq import current_conversation_credential

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/church-laser-three-epoch-20260914'
PYTHON='/tmp/laser-sign-venv/bin/python'
STAGE1_RUN='church-laser-ft3ep-official-20260914'
STAGE2_RUN='church-laser-ft3ep-rq32k-scratch-20260914'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def main():
    assert not (BASE/'pipeline-receipt.json').exists()
    proof=json.loads((BASE/'verification.json').read_text())
    assert proof['stage1_full_batch_verified'] and proof['preparation_smoke_passed']
    assert proof['strict_checkpoint_reload']
    manifest=json.loads((BASE/'source-manifest.json').read_text())
    for name,expected in manifest.items():assert digest(ROOT/name)==expected,name
    assert not Path('/proc/107567').exists() and not Path('/proc/107568').exists()
    credential=current_conversation_credential()
    environment={**os.environ,'WANDB_API_KEY':credential,'CUDA_VISIBLE_DEVICES':'0,1',
        'OMP_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'8','MKL_NUM_THREADS':'8',
        'TORCH_HOME':'/workspace/tmp/official-rqvae-eval-cache',
        'WANDB_ENTITY':'helloimlixin-rutgers','WANDB_PROJECT':'laser',
        'WANDB_CHECKPOINT_UPLOAD':'0','WANDB_RESUME_MODE':'never'}
    for name in ('WANDB_SERVICE','_WANDB_SERVICE','SMOKE_TEST'):environment.pop(name,None)
    stages=[];started=time.time()
    write(BASE/'pipeline-receipt.json',dict(pid=os.getpid(),started_unix=started,
        stage1_run=STAGE1_RUN,stage2_run=STAGE2_RUN,verified_sources=len(manifest),
        stage1_epochs=3,stage2_from_scratch=True,preflight_weights_used_for_production=False))
    def status(phase,**values):
        record=dict(phase=phase,pid=os.getpid(),updated_unix=time.time(),
            elapsed_seconds=time.time()-started,completed_stages=stages,**values)
        write(BASE/'pipeline-status.json',record);print(json.dumps(record),flush=True)
    def run_phase(name,script,arguments,run_id,offline=False):
        for file,expected in manifest.items():assert digest(ROOT/file)==expected,file
        command=[PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
            str(ROOT/'scripts/tools'/script),*map(str,arguments)]
        env={**environment,'WANDB_MODE':'disabled' if offline else 'online','WANDB_RUN_ID':run_id,
             'WANDB_NAME':run_id,'WANDB_RUN_GROUP':'church-laser-three-epoch-20260914'}
        with (BASE/f'{name}.log').open('ab') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT)
            write(BASE/f'{name}-launch.json',dict(pid=child.pid,command=command,started_unix=time.time()))
            status(name,torchrun_pid=child.pid,command=command)
            while child.poll() is None:
                time.sleep(15)
                write(BASE/'pipeline-heartbeat.json',dict(phase=name,torchrun_pid=child.pid,
                    parent_pid=os.getpid(),updated_unix=time.time()))
            if child.returncode:
                raise RuntimeError(f'{name} exited with code {child.returncode}; see {name}.log')
        stages.append(name)
    try:
        run_phase('stage1','finetune_church_laser_three_epochs.py',
            ['--output',BASE/'stage1'],STAGE1_RUN)
        stage1=json.loads((BASE/'stage1/complete.json').read_text())
        assert stage1['epoch']==3 and not stage1['smoke_only'] and stage1['optimizer_steps']==2961
        assert digest(stage1['checkpoint'])==stage1['checkpoint_sha256']
        run_phase('preparation','prepare_church_laser_three_epoch_stage2.py',
            ['--stage1-directory',BASE/'stage1','--output',BASE/'preparation'],STAGE2_RUN,offline=True)
        prepared=BASE/'preparation'
        cache=json.loads((prepared/'cache/complete.json').read_text())
        assert cache['stage1_epochs']==3 and cache['images']==126227 and not cache['cache_reused']
        assert cache['checkpoint_sha256']==stage1['checkpoint_sha256']
        arguments=['--cache',prepared/'cache','--calibration',prepared/'temperature-calibration.json',
            '--run-id',STAGE2_RUN,'--batch-size','512']
        run_phase('stage2-preflight','train_church_laser_three_epoch_stage2.py',
            [*arguments,'--output',BASE/'stage2-preflight','--max-updates','2','--offline'],STAGE2_RUN,offline=True)
        status('verifying_stage2_preflight')
        module_spec=importlib.util.spec_from_file_location('new_church_prior',ROOT/'scripts/tools/train_church_laser_three_epoch_stage2.py')
        module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
        import torch
        preflight=BASE/'stage2-preflight'
        state=torch.load(preflight/'last.pt',map_location='cpu',weights_only=False,mmap=True)
        assert state['step']==2 and state['attempts']==2
        assert state['tokenizer']['checkpoint_sha256']==stage1['checkpoint_sha256']
        for rank in range(2):assert json.loads((preflight/f'sampler-smoke-rank{rank}.json').read_text())['passed']
        cfg=module.OmegaConf.create(state['config'])
        model,optimizer=module.fresh_transformer(cfg)
        model.load_state_dict(state['state_dict'],strict=True);optimizer.load_state_dict(state['optimizer'])
        assert state['scheduler']['after']['T_max']==18600
        for value in model.state_dict().values():assert torch.isfinite(value).all()
        write(BASE/'stage2-verification.json',dict(passed=True,strict_checkpoint_reload=True,
            stage1_epochs=3,new_tokenizer_sha256=stage1['checkpoint_sha256'],global_batch=2048,
            successful_updates=2,skipped_updates=0,new_cache_verified=True,new_codebook_verified=True,
            calibrated_temperature=state['config']['loss']['temp'],preflight_weights_used_for_production=False))
        del model,optimizer,state
        run_phase('stage2','train_church_laser_three_epoch_stage2.py',
            [*arguments,'--output',BASE/'stage2'],STAGE2_RUN)
        status('complete')
    except BaseException as error:
        status('failed',error_type=type(error).__name__,error=str(error))
        raise


if __name__=='__main__':main()
