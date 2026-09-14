#!/usr/bin/env python3
"""Adopt the running Church fine-tune, then use released file-based FID."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
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


def process_identity(pid):
    path=Path(f'/proc/{pid}')
    try:
        fields=(path/'stat').read_text().rsplit(') ',1)[1].split()
        if fields[0]=='Z':return None
        return dict(pid=pid,start_ticks=int(fields[19]),
            command_sha256=hashlib.sha256((path/'cmdline').read_bytes()).hexdigest())
    except FileNotFoundError:
        return None


def main():
    assert not (BASE/'official-fid-supervisor.json').exists()
    proof=json.loads((BASE/'verification.json').read_text())
    assert proof['stage1_full_batch_verified'] and proof['preparation_smoke_passed']
    assert proof['strict_checkpoint_reload']
    fid_proof=json.loads((BASE/'official-fid-verification/verification.json').read_text())
    assert fid_proof['passed'] and fid_proof['cpu_rng_restored'] and fid_proof['cuda_rng_restored']
    assert fid_proof['stale_cache_rejected'] and fid_proof['upstream_function_used']
    prepared_smoke=json.loads((BASE/'preparation-official-fid-smoke/cache/complete.json').read_text())
    assert prepared_smoke['smoke_only'] and prepared_smoke['frozen_state_unchanged']
    handoff=json.loads((BASE/'official-fid-handoff-request.json').read_text())
    manifest=json.loads((BASE/'official-fid-source-manifest.json').read_text())
    for name,expected in manifest.items():assert digest(ROOT/name)==expected,name
    assert not Path('/proc/107567').exists() and not Path('/proc/107568').exists()
    if '--verify-only' in sys.argv:
        print(json.dumps(dict(passed=True,verified_sources=len(manifest),fid_verification=fid_proof['fid'])),flush=True)
        return
    assert process_identity(handoff['previous_supervisor_pid']) is None
    identity=process_identity(handoff['stage1_process']['pid'])
    assert identity==handoff['stage1_process'] or (identity is None and (BASE/'stage1/complete.json').exists())
    credential=current_conversation_credential()
    environment={**os.environ,'WANDB_API_KEY':credential,'CUDA_VISIBLE_DEVICES':'0,1',
        'OMP_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'8','MKL_NUM_THREADS':'8',
        'TORCH_HOME':'/workspace/tmp/official-rqvae-eval-cache',
        'WANDB_ENTITY':'helloimlixin-rutgers','WANDB_PROJECT':'laser',
        'WANDB_CHECKPOINT_UPLOAD':'0','WANDB_RESUME_MODE':'never'}
    for name in ('WANDB_SERVICE','_WANDB_SERVICE','SMOKE_TEST'):environment.pop(name,None)
    stages=[];started=time.time()
    write(BASE/'official-fid-supervisor.json',dict(pid=os.getpid(),started_unix=started,
        stage1_run=STAGE1_RUN,stage2_run=STAGE2_RUN,verified_sources=len(manifest),
        stage1_epochs=3,stage2_from_scratch=True,preflight_weights_used_for_production=False,
        adopted_stage1=handoff['stage1_process'],previous_supervisor_pid=handoff['previous_supervisor_pid'],
        generation_fid='released rqvae.metrics.fid.compute_fid'))
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
        stage1_pid=handoff['stage1_process']['pid']
        status('stage1',torchrun_pid=stage1_pid,adopted_running_finetune=True,
            next_stage_evaluator='released rqvae.metrics.fid')
        while True:
            identity=process_identity(stage1_pid)
            if identity is None:break
            assert identity==handoff['stage1_process'],'Stage-1 PID identity changed'
            write(BASE/'pipeline-heartbeat.json',dict(phase='stage1',torchrun_pid=stage1_pid,
                parent_pid=os.getpid(),updated_unix=time.time(),adopted=True))
            time.sleep(15)
        stages.append('stage1')
        stage1=json.loads((BASE/'stage1/complete.json').read_text())
        assert stage1['epoch']==3 and not stage1['smoke_only'] and stage1['optimizer_steps']==2961
        assert digest(stage1['checkpoint'])==stage1['checkpoint_sha256']
        run_phase('preparation','prepare_church_laser_three_epoch_official_fid.py',
            ['--stage1-directory',BASE/'stage1','--output',BASE/'preparation'],STAGE2_RUN,offline=True)
        prepared=BASE/'preparation'
        cache=json.loads((prepared/'cache/complete.json').read_text())
        assert cache['stage1_epochs']==3 and cache['images']==126227 and not cache['cache_reused']
        assert cache['checkpoint_sha256']==stage1['checkpoint_sha256']
        arguments=['--cache',prepared/'cache','--calibration',prepared/'temperature-calibration.json',
            '--run-id',STAGE2_RUN,'--batch-size','512']
        run_phase('stage2-preflight','train_church_laser_three_epoch_official_fid.py',
            [*arguments,'--output',BASE/'stage2-preflight','--max-updates','2','--offline'],STAGE2_RUN,offline=True)
        status('verifying_stage2_preflight')
        module_spec=importlib.util.spec_from_file_location('new_church_prior',ROOT/'scripts/tools/train_church_laser_three_epoch_official_fid.py')
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
        run_phase('stage2','train_church_laser_three_epoch_official_fid.py',
            [*arguments,'--output',BASE/'stage2'],STAGE2_RUN)
        status('complete')
    except BaseException as error:
        status('failed',error_type=type(error).__name__,error=str(error))
        raise


if __name__=='__main__':main()
