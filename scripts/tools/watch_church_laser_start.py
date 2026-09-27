#!/usr/bin/env python3
"""Observe the scheduled handoff until GPU training is verified or the job ends."""
import argparse
import datetime
import fcntl
import json
import math
from pathlib import Path
import subprocess
import time


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    lock = args.output.with_suffix('.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    observations = []
    while True:
        query = subprocess.run(['squeue', '-h', '-j', args.job, '-o', '%T|%R'],
                               text=True, capture_output=True, timeout=30)
        raw = query.stdout.strip()
        state, _, location = raw.partition('|')
        record = dict(job_id=args.job, checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      scheduler_state=state, location_or_reason=location,
                      training_verified=False, observations=observations)
        done = False
        if query.returncode:
            record.update(status='QUERY_ERROR', error=query.stderr.strip())
        elif not raw:
            result = subprocess.run(['sacct', '-n', '-X', '-j', args.job, '-o', 'State,ExitCode', '-P'],
                                    text=True, capture_output=True, timeout=30)
            record.update(status='ALLOCATION_ENDED', accounting=result.stdout.strip())
            done = True
        elif state == 'RUNNING':
            status = read_json(args.base / 'train/status.json')
            gpus = [read_json(args.base / f'gpu-rank{i}.json') for i in range(8)]
            record.update(status='STARTING', trainer=status)
            hardware_ok = all(g and g['world_size'] == 8 and 'L40S' in g['gpu_name'] for g in gpus)
            if hardware_ok:
                hardware_ok = len({(g['hostname'], g['local_rank']) for g in gpus}) == 8
            if hardware_ok:
                record['gpus'] = gpus
            if (status and status.get('phase') == 'training' and
                    status.get('optimizer_step', 0) > 5642 and
                    math.isfinite(status.get('loss', float('nan'))) and
                    time.time() - status['updated_unix'] < 300):
                if not observations or status['optimizer_step'] > observations[-1]['optimizer_step']:
                    observations.append({k: status[k] for k in ('optimizer_step', 'loss', 'updated_unix')})
                record['status'] = 'VERIFYING_TRAINING'
                if (hardware_ok and len(observations) > 1 and
                        observations[-1]['updated_unix'] - observations[0]['updated_unix'] >= 60):
                    record.update(status='HEALTHY', training_verified=True)
                    done = True
        else:
            record['status'] = 'QUEUED'
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(args.output)
        if done:
            print(json.dumps(record), flush=True)
            return
        time.sleep(60)


if __name__ == '__main__':
    main()
