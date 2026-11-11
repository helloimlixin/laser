"""Yield all four GPUs at a persisted optimizer boundary without losing updates."""
import json
import argparse
import os
from pathlib import Path
import signal
import time

ROOT=Path('/workspace/Projects/laser')
BASE=Path('/tmp/laser-beam451-stage2-4h100-20261107')
OUT=ROOT/'outputs/imagenet-beam451-aligned-joint16k-scratch100-4h100-20261107'
EXPERIMENT=OUT/'sampler-calibration-20261108'


def read(p):return json.loads(p.read_text())


def write(value):
    tmp=EXPERIMENT/'pause.pending';tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(EXPERIMENT/'pause.json')


def alive(pid):
    p=Path(f'/proc/{pid}/stat')
    return p.exists() and p.read_text().split()[2]!='Z'


def main():
    global EXPERIMENT
    parser=argparse.ArgumentParser();parser.add_argument('--target-step',type=int);parser.add_argument('--experiment',type=Path)
    args=parser.parse_args()
    if args.experiment is not None:EXPERIMENT=args.experiment
    EXPERIMENT.mkdir(exist_ok=True)
    assert not (EXPERIMENT/'pause.json').exists(),'Inspect existing calibration before another pause'
    step=read(BASE/'live-training/train/progress-rank0.json')['train/global_step']
    target=args.target_step if args.target_step is not None else (step//250+1)*250
    assert target>=step,'Requested checkpoint boundary has already passed'
    supervisor=read(OUT/'launch.json')['supervisor_pid']
    assert b'launch-imagenet-beam451-4h100-20261107/supervise.py' in Path(f'/proc/{supervisor}/cmdline').read_bytes()
    ranks=[read(BASE/f'live-training/train/startup-rank{rank}.json')['pid'] for rank in range(4)]
    write(dict(phase='waiting_for_snapshot',target_step=target,pids=ranks,supervisor_pid=supervisor,time=time.time()))
    partial=BASE/f'checkpoints/step-{target:07d}.partial'
    while not partial.exists() and not partial.with_suffix('.pt').exists():
        assert all(alive(p) for p in ranks)
        time.sleep(.02)
    stopped=[]
    try:
        for pid in ranks[1:]:os.kill(pid,signal.SIGSTOP);stopped.append(pid)
        assert [read(BASE/f'live-training/train/progress-rank{r}.json')['train/global_step'] for r in range(4)]==[target]*4
        write(dict(phase='persisting_checkpoint',target_step=target,pids=ranks,time=time.time()))
        while True:
            receipt=read(OUT/'checkpoint.json')
            if receipt['global_step']==target and receipt['persistent_state']=='COMPLETE':break
            assert alive(ranks[0]);time.sleep(2)
        import torch
        state=torch.load(receipt['path'],map_location='cpu',weights_only=False,mmap=True)
        assert state['global_step']==state['scheduler']['after']['last_epoch']==target
        assert len(state['optimizer']['state'])==780
        assert all(int(v['step'])==target for v in state['optimizer']['state'].values())
        assert state['runtime']['training_rng_policy']=='independent_rank_v1'
        del state
        assert [read(BASE/f'live-training/train/progress-rank{r}.json')['train/global_step'] for r in range(4)]==[target]*4
        os.kill(supervisor,signal.SIGTERM);time.sleep(1)
        for pid in stopped:
            if alive(pid):os.kill(pid,signal.SIGCONT)
        stopped=[]
        deadline=time.monotonic()+60
        while any(alive(p) for p in [supervisor,*ranks]):
            assert time.monotonic()<deadline,'Previous worker did not terminate'
            time.sleep(.2)
        write(dict(phase='paused_for_fixed_checkpoint_calibration',target_step=target,checkpoint=receipt,
            completed_updates_lost=0,weights_optimizer_scheduler_rng_preserved=True,time=time.time()))
        print(json.dumps(dict(paused_at_step=target,checkpoint=receipt['path'])),flush=True)
    finally:
        for pid in stopped:
            if alive(pid):os.kill(pid,signal.SIGCONT)


if __name__=='__main__':main()
