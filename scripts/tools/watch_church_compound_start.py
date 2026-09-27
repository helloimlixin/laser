#!/usr/bin/env python3
"""Read-only startup observation for a queued node-local Church continuation."""
import argparse
import base64
import datetime
import fcntl
import json
import math
import netrc
from pathlib import Path
import subprocess
import time
import urllib.request


def command(argv):
    return subprocess.check_output(argv, text=True, timeout=45).strip()


def online_files():
    query = '''query { project(name:"laser",entityName:"helloimlixin-rutgers") {
      run(name:"church-laser-rfid421-ft3-compound-scratch90-20260918") {
        state files(names:["last.pt","best-fid-01.pt"],first:2) {
          edges { node { name md5 sizeBytes } }
        }
      }
    }}'''
    key = netrc.netrc().authenticators('api.wandb.ai')[2]
    request = urllib.request.Request('https://api.wandb.ai/graphql',
        data=json.dumps({'query': query}).encode(), headers={
            'Content-Type': 'application/json',
            'Authorization': 'Basic ' + base64.b64encode(('api:' + key).encode()).decode()})
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)
    if result.get('errors'):
        raise RuntimeError('W&B query failed')
    return result['data']['project']['run']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True, type=int)
    parser.add_argument('--minimum-step', required=True, type=int)
    parser.add_argument('--initial-md5', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    lock = args.output.with_suffix('.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    observations = []
    while True:
        record = dict(job_id=str(args.job), training_verified=False,
                      checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        done = False
        try:
            row = command(['squeue', '-h', '-j', str(args.job), '-o', '%T|%D|%R'])
            if not row:
                record.update(status='ALLOCATION_ENDED', accounting=command([
                    'sacct', '-n', '-X', '-j', str(args.job), '-o', 'State,ExitCode', '-P']))
                done = True
            else:
                state, nodes, location = row.split('|', 2)
                record.update(scheduler_state=state, location_or_reason=location, status='QUEUED')
                if state == 'RUNNING':
                    code = '''import pathlib,json
b=pathlib.Path('/mnt/scratch/xl598/laser/church-compound/job-JOB/bundle/train')
p=b/'status.json'
print(json.dumps(dict(ranks=[json.loads(x.read_text()) for x in b.glob('rank-*.json')],
    progress=json.loads(p.read_text()) if p.exists() else None)))'''.replace('JOB', str(args.job))
                    raw = command(['srun', '--overlap', '--jobid=' + str(args.job),
                        '--nodes=' + nodes, '--ntasks=' + nodes, '--ntasks-per-node=1',
                        '--cpus-per-task=1', 'python3', '-c', code])
                    reports = [json.loads(line) for line in raw.splitlines()]
                    ranks = [rank for report in reports for rank in report['ranks']]
                    progress = next((r['progress'] for r in reports if r['progress']), None)
                    record.update(status='STARTING', ranks=ranks, trainer=progress)
                    hardware_ok = (len(ranks) == 4 and {r['rank'] for r in ranks} == set(range(4))
                        and all(r['world_size'] == 4 and r['gpu_memory_bytes'] >= 23 * 1024**3
                                for r in ranks))
                    if (progress and math.isfinite(progress['train/loss'])
                            and time.time() - progress['updated_unix'] < 300):
                        step = progress['train/global_step']
                        if not observations or step > observations[-1]['step']:
                            observations.append(dict(step=step, updated_unix=progress['updated_unix']))
                        record['status'] = 'VERIFYING_TRAINING'
                        # Pass a full preview interval; the former RAM failure happened at one.
                        if (hardware_ok and step >= args.minimum_step + 210 and len(observations) > 1
                                and observations[-1]['updated_unix'] - observations[0]['updated_unix'] >= 60):
                            online = online_files()
                            files = {e['node']['name']: e['node'] for e in online['files']['edges']}
                            record['online'] = online
                            if (online['state'] == 'running' and set(files) == {'last.pt', 'best-fid-01.pt'}
                                    and files['last.pt']['md5'] != args.initial_md5
                                    and all(f['md5'] and int(f['sizeBytes']) > 4_000_000_000
                                            for f in files.values())):
                                record.update(status='HEALTHY', training_verified=True)
                                done = True
        except Exception as error:
            record.update(status='OBSERVATION_ERROR', error=str(error))
        record['observations'] = observations[-20:]
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(args.output)
        if done:
            print(json.dumps(record), flush=True)
            return
        time.sleep(60)


if __name__ == '__main__':
    main()
