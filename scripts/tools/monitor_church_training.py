#!/usr/bin/env python3
"""Persistent local health monitor; LR decisions live in the trainer checkpoint."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from archive.scripts.train_church_ffhq_recipe import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reports',type=Path,required=True)
    args=p.parse_args();args.reports.mkdir(parents=True,exist_ok=True)
    lock=(args.reports/'health-monitor.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=None
    while True:
        now=time.time();status_path=args.output/'status.json'
        status=json.loads(status_path.read_text())
        process=Path('/proc')/str(status['pid'])
        command=(process/'cmdline').read_bytes().split(b'\0') if (process/'cmdline').exists() else []
        alive=(str(ROOT/'scripts/tools/train_church_joint_distributed.py').encode() in command
            and str(args.output.resolve()).encode() in command)
        age=now-status_path.stat().st_mtime
        phase=status['phase']
        health=('complete' if phase=='complete' else 'paused' if phase=='paused' else
            'failed' if phase=='failed' else 'process_missing' if not alive else
            'stalled' if age>300 else 'running')
        lr_path=args.output/'lr-monitor.json'
        lr=json.loads(lr_path.read_text()) if lr_path.exists() else None
        row={'monitor_pid':os.getpid(),'checked_unix':now,'health':health,'trainer_alive':alive,
             'status_age_seconds':age,'training':status,'fid_lr_monitor':lr}
        write_json(args.reports/'health.json',row)
        marker=(health,lr['state']['last_step'] if lr else None,lr['state']['reductions'] if lr else None)
        if marker!=previous:
            print(json.dumps(row),flush=True)
            with (args.reports/'health-events.jsonl').open('a') as handle:handle.write(json.dumps(row)+'\n')
            previous=marker
        if health=='complete':break
        time.sleep(15)


if __name__=='__main__':main()
