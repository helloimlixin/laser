#!/usr/bin/env python3
"""Pause the protected trainer only after preserving an exact epoch boundary.

The rank-zero checkpoint writer keeps running while other ranks wait. No
optimizer update after the saved boundary may complete. Any pre-stop failure
resumes the waiting ranks, leaving the original experiment running.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time

ROOT=Path(__file__).resolve().parents[2]


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def process_matches(pid,token):
    try:return token in Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    except FileNotFoundError:return False


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--new-run',required=True)
    parser.add_argument('--source-base',type=Path,default=Path('/tmp/laser-compound512-origlr-imagenet-20261016-r3'))
    parser.add_argument('--source-run',default='imagenet-laser-compound512-origlr-scratch100-4h100-20261016-r3')
    parser.add_argument('--source-controller',type=int,default=167733)
    parser.add_argument('--comparison-run',default='imagenet-laser-compound512-epoch2-joint-refined-20261017')
    parser.add_argument('--watch',action='store_true',help='Explicitly arm the checkpoint-boundary pause')
    args=parser.parse_args()
    if not args.watch:parser.error('--watch is required to arm the pause')
    def interrupted(*_):raise KeyboardInterrupt('Handoff interrupted; resume any waiting source ranks')
    signal.signal(signal.SIGTERM,interrupted)
    out=ROOT/'outputs'/args.new_run
    old=ROOT/'outputs'/args.source_run
    token=str(args.source_base/'train.py')
    startup=[json.loads((old/'train'/f'startup-rank{rank}.json').read_text()) for rank in range(4)]
    pids=[r['pid'] for r in startup]
    assert len(set(pids))==4 and all(process_matches(pid,token) for pid in pids)
    assert process_matches(args.source_controller,str(args.source_base))
    progress_paths=[old/'train'/f'progress-rank{rank}.json' for rank in range(4)]
    def steps():return [int(json.loads(p.read_text())['train/global_step']) for p in progress_paths]
    def status(phase,**values):
        write(out/'handoff-status.json',dict(phase=phase,pid=os.getpid(),source_run=args.source_run,
            source_pids=pids,time=time.time(),**values))
    status('waiting_for_prepared_trial_and_finished_sampler_comparison')
    while True:
        prepared=out/'prepared.json'
        comparison=ROOT/'outputs'/args.comparison_run/'status.json'
        if prepared.exists() and comparison.exists():
            if json.loads(comparison.read_text())['phase']=='completed':break
        assert all(process_matches(pid,token) for pid in pids),'Source trainer exited before handoff'
        time.sleep(2)
    plan=json.loads(prepared.read_text())
    assert plan['prepared'] and not plan['training_launched'] and plan['fresh_stage2']
    boundary=(max(steps())//626+1)*626
    stopped=[];source_stopping=False
    try:
        while True:
            status('waiting_for_next_epoch_checkpoint',target_step=boundary,current_steps=steps())
            existing=set((args.source_base/'checkpoint-staging').glob('*.staging'))
            while True:
                current=steps()
                new=set((args.source_base/'checkpoint-staging').glob('*.staging'))-existing
                if new and min(current)==max(current)==boundary:break
                if max(current)>boundary:
                    boundary=(max(current)//626+1)*626
                    status('waiting_for_next_epoch_checkpoint',target_step=boundary,current_steps=current)
                    existing=set((args.source_base/'checkpoint-staging').glob('*.staging'))
                assert all(process_matches(pid,token) for pid in pids)
                time.sleep(.1)
            for pid in pids[1:]:
                assert process_matches(pid,token)
                os.kill(pid,signal.SIGSTOP);stopped.append(pid)
            current=steps()
            if min(current)==max(current)==boundary:break
            for pid in stopped:os.kill(pid,signal.SIGCONT)
            stopped=[]
            boundary=(max(current)//626+1)*626
        status('waiting_for_checkpoint_commit',target_step=boundary,current_steps=steps())
        deadline=time.monotonic()+1200
        receipt=None
        while receipt is None:
            assert min(steps())==max(steps())==boundary,'An optimizer update escaped the boundary'
            for p in (old/'persistent-train/checkpoints/.checkpoint-data').glob('*.json'):
                candidate=json.loads(p.read_text())
                if candidate['global_step']==boundary:receipt=candidate;break
            if time.monotonic()>deadline:raise TimeoutError('Checkpoint writer did not commit within 20 minutes')
            if receipt is None:time.sleep(1)
        status('verifying_boundary_checkpoint',target_step=boundary,receipt=receipt)
        for name in ['local','persistent']:
            path=Path(receipt[name])
            assert path.stat().st_size==receipt['bytes']
            assert digest(path)==receipt['sha256']
        import torch
        payload=torch.load(receipt['local'],map_location='cpu',mmap=True,weights_only=False)
        assert payload['run_id']==args.source_run and payload['global_step']==boundary
        assert payload['next_epoch']==boundary//626 and payload['next_microbatch']==0
        assert len(payload['rng_state_by_rank'])==4
        assert all(int(state['step'])==boundary for state in payload['optimizer']['state'].values())
        assert min(steps())==max(steps())==boundary
        del payload
        proof=dict(passed=True,source_run=args.source_run,target_step=boundary,receipt=receipt,
            persistent_sha256_verified=True,local_sha256_verified=True,
            all_adam_steps=boundary,optimizer_updates_lost=0,all_rank_rng_preserved=True,
            source_recipe_unchanged=True,time=time.time())
        write(out/'preserved-source-checkpoint.json',proof)
        source_stopping=True
        status('stopping_verified_source_trainer',target_step=boundary)
        os.kill(args.source_controller,signal.SIGTERM)
        for pid in stopped:
            if process_matches(pid,token):
                os.kill(pid,signal.SIGINT);os.kill(pid,signal.SIGCONT)
        stopped=[]
        for second in range(45):
            live=[pid for pid in pids if process_matches(pid,token)]
            if not live:break
            if second in [10,30]:
                for pid in live:os.kill(pid,signal.SIGTERM if second==10 else signal.SIGKILL)
            time.sleep(1)
        assert not any(process_matches(pid,token) for pid in pids)
        status('paused_with_verified_checkpoint',target_step=boundary,
            checkpoint=receipt,optimizer_updates_lost=0)
    except BaseException as error:
        status('handoff_failed',error=repr(error),source_stopping=source_stopping)
        raise
    finally:
        for pid in stopped:
            if process_matches(pid,token):os.kill(pid,signal.SIGCONT)


if __name__=='__main__':main()
