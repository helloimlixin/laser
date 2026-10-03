"""Launch the authorized scratch experiment only after the preceding run finishes."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import zipfile


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def process_matches(pid, pattern):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        return str(pattern).encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
    except FileNotFoundError:
        return False
    except PermissionError:
        return True


def completion_gate(launch, log, parent, alive=process_matches):
    parent = Path(parent)
    if launch.get('state') != 'finished' or launch.get('returncode') != 0:
        return 'waiting_for_successful_parent_completion'
    for key, script in [('training_pid', 'run_entry.py'), ('supervisor_pid', 'supervise.py')]:
        if alive(launch.get(key), parent / script):
            return 'waiting_for_parent_process_exit'
    match = re.search(r'^Epoch 100: FID=([0-9.eE+-]+);.*saved ', log, re.MULTILINE)
    if match is None or not math.isfinite(float(match.group(1))):
        return 'waiting_for_final_fid'
    return None


def validate_final_payload(x, parent_id, policy):
    c = x['config']
    assert x['epoch'] == 100 and x['global_step'] == 63500
    assert x.get('batch_idx', 0) == 0 and math.isfinite(float(x['fid']))
    assert c['wandb_id'] == parent_id and c['epochs'] == 100
    assert c['combination_target_policy'] == policy
    assert c['total_batch_size'] == 2016 and c['optimizer_steps_per_epoch'] == 635
    assert c['training_data_mode'] == 'online-fresh-images'
    assert x['checkpoint_world_size'] == 8 and len(x['rng_state_by_rank']) == 8
    assert len(x['state_dict']) == 870 and len(x['optimizer']['state']) == 870
    assert x['scheduler']['last_epoch'] == 63500
    assert x['scheduler']['_laser_schedule_config']['total_steps'] == 63500
    assert {int(s['step']) for s in x['optimizer']['state'].values()} == {63500}
    assert x['best_fid'] and x['best_inception']


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_parent(plan):
    import torch
    torch.set_num_threads(4)
    parent = Path(plan['parent_base'])
    persistent = Path(plan['parent_persistent'])
    source = (persistent / 'train/checkpoints/last.pt').resolve(strict=True)
    assert source.stat().st_size > 17_000_000_000 and zipfile.is_zipfile(source)
    key = hashlib.sha256(str(source).encode()).hexdigest()
    local = parent / 'upload-cache/objects' / f'{key}.pt'
    assert local.is_file() and local.stat().st_size == source.stat().st_size
    x = torch.load(local, map_location='cpu', weights_only=False, mmap=True)
    validate_final_payload(x, persistent.name, read_json(parent / 'target-policy.json'))
    for value in list(x['state_dict'].values()) + [
            v[k] for v in x['optimizer']['state'].values() for k in ('exp_avg', 'exp_avg_sq')]:
        assert bool(torch.isfinite(value).all()), 'Nonfinite final parent state'
    best = []
    for metric in ('best_fid', 'best_inception'):
        for score, name in x[metric]:
            path = Path(name).resolve(strict=True)
            assert path.is_relative_to(persistent / 'train/checkpoints')
            assert path.stat().st_size > 5_000_000_000 and zipfile.is_zipfile(path)
            record = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
            assert len(record['state_dict']) == 870 and record['config']['wandb_id'] == persistent.name
            best.append(dict(metric=metric, score=score, path=str(path), bytes=path.stat().st_size))
            del record
    final_fid = float(x['fid'])
    del x
    local_sha = digest(local)
    assert digest(source) == local_sha, 'Final checkpoint copies differ'
    return dict(verified_unix=time.time(), epoch=100, global_step=63500,
                final_fid=final_fid, source=str(source), bytes=source.stat().st_size,
                sha256=local_sha, full_sha256_match=True,
                all_weights_and_moments_finite=True, best_checkpoints=best)


def gpu_memory_idle(rows):
    return len(rows) == 8 and all(0 <= int(used) < 512 for used in rows)


def claim_start(base, record):
    # A crash between claiming and launching requires inspection, not another
    # automatic random initialization. The supervisor separately locks itself.
    path = Path(base) / 'start-request.json'
    with path.open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    base = args.base.resolve()
    plan = read_json(base / 'queue-plan.json')
    persistent = Path(plan['persistent'])
    parent = Path(plan['parent_base'])
    parent_launch = read_json(parent / 'launch.json')
    if args.check_only:
        reason = completion_gate(parent_launch, (parent / 'training.log').read_text(), parent)
        print(json.dumps(dict(ready=reason is None, reason=reason)))
        return 0
    with (base / 'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def status(state, **details):
            record = dict(state=state, watcher_pid=os.getpid(), updated_unix=time.time(),
                          run_id=plan['run_id'], parent_run=plan['parent_run'], **details)
            write_json(base / 'queue-status.json', record)
            try:
                write_json(persistent / 'queue-status.json', record)
            except OSError as exc:
                print(f'Persistent status mirror temporarily unavailable: {type(exc).__name__}', flush=True)

        verified = None
        try:
            while True:
                if (base / 'cancel.request').exists():
                    status('cancelled')
                    return 0
                if (base / 'start-request.json').exists():
                    status('handoff_already_claimed', instruction='Inspect scratch launch.json before restarting this queue.')
                    return 0
                try:
                    launch = read_json(parent / 'launch.json')
                    reason = completion_gate(launch, (parent / 'training.log').read_text(), parent)
                except (OSError, ValueError) as exc:
                    status('waiting_for_parent_metadata', error=type(exc).__name__)
                    time.sleep(30)
                    continue
                if reason:
                    status(reason, parent_state=launch.get('state'))
                    time.sleep(30)
                    continue
                if verified is None:
                    status('verifying_parent_final_checkpoint')
                    verified = verify_parent(plan)
                    for root in (base, persistent, base / 'evidence'):
                        write_json(root / 'parent-completion.json', verified)
                rows = subprocess.check_output([
                    'nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).split()
                if not gpu_memory_idle(rows):
                    status('waiting_for_eight_idle_gpus', memory_used_mib=rows)
                    time.sleep(30)
                    continue
                for relative, expected in plan['control_sha256'].items():
                    assert digest(base / relative) == expected, f'Changed launch file: {relative}'
                for relative, expected in read_json(base / 'evidence/source-manifest.json').items():
                    assert digest(base / 'runtime' / relative) == expected, f'Changed runtime: {relative}'
                for name, expected in plan['asset_sha256'].items():
                    assert digest(name) == expected, f'Changed frozen asset: {name}'
                assert not (base / 'train').exists(), 'Scratch training directory is already in use'
                assert not (persistent / 'train/checkpoints').exists(), 'Scratch checkpoints already exist'
                # Recheck immediately before the one-shot handoff.
                assert completion_gate(read_json(parent / 'launch.json'), (parent / 'training.log').read_text(), parent) is None
                assert not (base / 'cancel.request').exists()
                claim_start(base, dict(requested_unix=time.time(), authorized_run=plan['run_id'],
                                       parent_final_checkpoint_sha256=verified['sha256']))
                with (base / 'supervisor.log').open('ab') as log:
                    child = subprocess.Popen([plan['python'], str(base / 'supervise.py')], cwd=base,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True, env={**os.environ, 'PYTHONUNBUFFERED': '1'})
                status('launched', supervisor_pid=child.pid, parent_final_fid=verified['final_fid'])
                return 0
        except Exception as exc:
            status('blocked_before_launch', error=f'{type(exc).__name__}: {exc}')
            raise


if __name__ == '__main__':
    raise SystemExit(main())
