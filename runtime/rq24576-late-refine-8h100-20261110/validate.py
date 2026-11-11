"""Run two 8-GPU updates, then restore raw/EMA/Adam/RNG for two more."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch

HERE=Path(__file__).resolve().parent
PLAN=json.loads((HERE/'plan.json').read_text());BASE=Path(PLAN['base']);OUT=Path(PLAN['output'])


def read(path):return json.loads(Path(path).read_text())


def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')


def environment():
    spec=importlib.util.spec_from_file_location('parent_environment',HERE.parent/'rq24576-native-8h100-20261108/environment.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return dict(module.environment(),LASER_STAGE2_OUTPUT=str(OUT),LASER_BEAM_BASE=str(BASE),
        LASER_TEST_FINAL_TEACHER='1',WANDB_CACHE_DIR=str(BASE/'wandb-cache'))


def main():
    source=Path(PLAN['source_checkpoint'])
    with source.open('rb') as handle:actual=hashlib.file_digest(handle,'sha256').hexdigest()
    assert actual==PLAN['source_checkpoint_sha256']
    write(OUT/'source-checkpoint-verification.json',dict(passed=True,sha256=actual,bytes=source.stat().st_size,
        source_epoch=PLAN['source_epoch'],source_step=PLAN['source_step']))
    for phase in range(2):
        folder=BASE/f'smoke/phase{phase+1}';folder.mkdir(parents=True,exist_ok=True)
        env=dict(environment(),LASER_PRIVATE_SMOKE_DIR=str(folder))
        expected_start=PLAN['source_step']+phase*2;expected_end=expected_start+2
        command=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc-per-node=8',str(BASE/'train.py'),
            '--epochs',str(PLAN['target_epoch']),'--microbatch','64','--resume',str(source),'--smoke-updates',str(expected_end)]
        with (OUT/f'resume-smoke-{phase+1}.log').open('ab',buffering=0) as log:
            child=subprocess.Popen(command,env=env,cwd=BASE,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(OUT/'smoke-launch.json',dict(pid=child.pid,phase=phase+1,command=command,time=time.time()))
        # The validator itself is detached; the interactive agent remains free to report progress.
        assert child.wait()==0,f'Smoke phase {phase+1} failed'
        proof=read(folder/'verification.json')
        assert proof['passed'] and proof['global_step']==expected_end
        assert proof['optimizer_state_count']==780 and proof['all_adam_steps']==expected_end
        assert proof['ema_updates']==(phase+1)*2 and proof['frozen_tokenizer_unchanged']
        parent=torch.load(source,map_location='cpu',mmap=True,weights_only=False)
        ranks=[]
        for rank in range(8):
            startup=read(folder/f'startup-rank{rank}.json')
            rng=read(folder/f'rng-startup-rank{rank}.json');targets=read(folder/f'targets-rank{rank}.json')
            interface=read(folder/f'generation-interface-rank{rank}.json')
            ema_interface=read(folder/f'ema-generation-interface-rank{rank}.json')
            expected_rng=hashlib.sha256(parent['rng_state_by_rank'][rank]['cuda'].numpy().tobytes()).hexdigest()
            assert startup['global_step']==expected_start and startup['restored_optimizer_age']==expected_start
            assert startup['optimizer_state_count_at_start']==780 and startup['ema_updates_at_start']==phase*2
            assert startup['world_size']==8 and startup['global_batch']==2048 and startup['microbatch']==64
            assert startup['source_CNN_unchanged'] and not startup['fresh_stage2'] and not startup['fresh_optimizer']
            assert targets['temperature']==.25 and targets['native_RQBottleneck'] and targets['stochastic_prefixes']
            assert rng['rng_state_sha256']==expected_rng and rng['training_rng_restored_from_checkpoint']
            assert interface['passed'] and interface['fixed_codeword_decoding'] and interface['vocabulary']==24576
            assert ema_interface['passed'] and ema_interface['fixed_codeword_decoding']
            ranks.append(dict(rank=rank,restored_adam_age=expected_start,restored_ema_updates=phase*2,
                cuda_rng_restored_exactly=True,generation_verified=True,final_teacher_temperature_verified=True))
        del parent
        checkpoint=read(BASE/'smoke/checkpoint.json');source=Path(checkpoint['path'])
        state=torch.load(source,map_location='cpu',mmap=True,weights_only=False)
        assert state['state_dict_weights']=='ema' and state['optimizer_weights_key']=='state_dict_raw'
        assert state['ema_updates']==(phase+1)*2
        assert {int(s['step']) for s in state['optimizer']['state'].values()}=={expected_end}
        assert set(state['state_dict'])==set(state['state_dict_raw'])
        assert any(not torch.equal(state['state_dict'][k],state['state_dict_raw'][k]) for k in state['state_dict'])
        del state
        write(OUT/f'resume-phase{phase+1}-verification.json',dict(passed=True,ranks=ranks,updates=2,
            checkpoint=str(source),completed_step=expected_end,ema_and_raw_weights_differ=True))
    write(OUT/'resume-preflight-verification.json',dict(passed=True,source_epoch=PLAN['source_epoch'],
        source_step=PLAN['source_step'],world_size=8,optimizer_states=780,
        parent_resume_and_own_checkpoint_roundtrip=True,raw_and_ema_restored=True,
        all_eight_rng_states_restored=True,tokenizer_unchanged=True,final_teacher_validated=True,
        production_will_restart_from_original_parent=True,time=time.time()))
    print(json.dumps(dict(passed=True,source_epoch=PLAN['source_epoch'],smoke_phases=2)),flush=True)


if __name__=='__main__':main()
