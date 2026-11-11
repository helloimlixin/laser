"""Drain a complete checkpoint before a one-time training RNG repair."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path('/workspace/Projects/laser')
BASE=Path('/tmp/laser-beam451-stage2-4h100-20261107')
OUT=ROOT/'outputs/imagenet-beam451-aligned-joint16k-scratch100-4h100-20261107'
AUDIT=OUT/'audit-stage1-preservation-20261108'
SUPERVISOR=ROOT/'runtime/launch-imagenet-beam451-4h100-20261107/supervise.py'


def read(path):return json.loads(path.read_text())


def write(value):
    temp=AUDIT/'rng-transition.pending';temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(AUDIT/'rng-transition.json')


def alive(pid):
    p=Path(f'/proc/{pid}/stat')
    return p.exists() and p.read_text().split()[2]!='Z'


def main():
    progress=read(BASE/'live-training/train/progress-rank0.json')
    target=(int(progress['train/global_step'])//250+1)*250
    supervisor=read(OUT/'launch.json')['supervisor_pid']
    assert str(SUPERVISOR).encode() in Path(f'/proc/{supervisor}/cmdline').read_bytes()
    ranks=[read(BASE/f'live-training/train/startup-rank{rank}.json')['pid'] for rank in range(4)]
    assert all(alive(pid) for pid in ranks)
    write(dict(phase='waiting_for_checkpoint_snapshot',target_step=target,pids=ranks,supervisor_pid=supervisor,time=time.time()))
    partial=BASE/f'checkpoints/step-{target:07d}.partial'
    final=partial.with_suffix('.pt')
    while not partial.exists() and not final.exists():
        assert all(alive(pid) for pid in ranks), 'Training rank exited before checkpoint boundary'
        time.sleep(.02)
    # A checkpoint snapshot precedes serialization. Freeze three training ranks;
    # rank zero's serialization/persistence thread can finish, but no distributed
    # optimizer update can finish while the other three ranks are stopped.
    stopped=[]
    try:
        for pid in ranks[1:]:os.kill(pid,signal.SIGSTOP);stopped.append(pid)
        observed=[read(BASE/f'live-training/train/progress-rank{r}.json')['train/global_step'] for r in range(4)]
        assert observed==[target]*4, f'Unexpected boundary: {observed}'
        write(dict(phase='persisting_checkpoint_before_rng_repair',target_step=target,pids=ranks,observed_steps=observed,time=time.time()))
        while True:
            receipt=read(OUT/'checkpoint.json')
            if receipt['global_step']==target and receipt['persistent_state']=='COMPLETE':break
            assert alive(ranks[0]), 'Rank zero exited before checkpoint persisted'
            time.sleep(2)
        import torch
        state=torch.load(receipt['path'],map_location='cpu',weights_only=False,mmap=True)
        assert state['global_step']==target
        assert len(state['optimizer']['state'])==780
        assert all(int(v['step'])==target for v in state['optimizer']['state'].values())
        assert state['scheduler']['after']['last_epoch']==target
        assert len(state['rng_state_by_rank'])==4
        fingerprint=[hashlib.sha256(v['cuda'].numpy().tobytes()).hexdigest() for v in state['rng_state_by_rank']]
        del state
        observed=[read(BASE/f'live-training/train/progress-rank{r}.json')['train/global_step'] for r in range(4)]
        assert observed==[target]*4
        write(dict(phase='complete_checkpoint_verified_restarting',target_step=target,observed_steps=observed,
            checkpoint=receipt,old_cuda_rng_hashes=fingerprint,weights_optimizer_scheduler_preserved=True,
            completed_updates_lost=0,time=time.time()))
        os.kill(supervisor,signal.SIGTERM)
        # Deliver the supervisor's pending termination signal to stopped ranks.
        time.sleep(1)
        for pid in stopped:
            if alive(pid):os.kill(pid,signal.SIGCONT)
        stopped=[]
        deadline=time.monotonic()+60
        while any(alive(pid) for pid in [supervisor,*ranks]):
            assert time.monotonic()<deadline,'Previous training processes did not terminate'
            time.sleep(.2)
        with (OUT/'supervisor.log').open('ab',buffering=0) as log:
            process=subprocess.Popen([sys.executable,'-u',str(SUPERVISOR)],cwd=ROOT,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(dict(phase='resuming_with_independent_rank_rng',target_step=target,checkpoint=receipt,
            supervisor_pid=process.pid,weights_optimizer_scheduler_preserved=True,completed_updates_lost=0,time=time.time()))
        print(json.dumps(dict(resumed_from_step=target,supervisor_pid=process.pid)),flush=True)
    finally:
        for pid in stopped:
            if alive(pid):os.kill(pid,signal.SIGCONT)


if __name__=='__main__':main()
