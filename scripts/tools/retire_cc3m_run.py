"""Free a stopped run's GPUs after preserving full states for CPU-only upload."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch


def publish(base, api_key_file):
    import wandb
    os.environ['WANDB_API_KEY'] = api_key_file.read_text().strip()
    receipt = json.loads((base / 'retired-local-preservation.json').read_text())
    staging = Path(receipt['staging'])
    api = wandb.Api(timeout=60)
    run = api.run(receipt['run_path'])
    for name, expected in receipt['files'].items():
        for attempt in range(3):
            try:
                run.upload_file(str(staging / name), root=str(staging))
                deadline = time.monotonic() + 1800
                while True:
                    remote = api.run(receipt['run_path']).file(name)
                    if remote.size == expected['bytes'] and remote.md5 == expected['md5']:
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError('Cloud checksum acknowledgement: ' + name)
                    time.sleep(10)
                print(json.dumps(dict(slot=name, verified=True, **expected)), flush=True)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(10)
    run.summary.update({'pipeline/phase': 'retired_for_diagnosed_scratch_restart'})
    receipt.update(online_verified=True, timestamp=time.time())
    (base / 'retired-online-preservation.json').write_text(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--local', type=Path)
    parser.add_argument('--stop-record', type=Path)
    parser.add_argument('--api-key-file', type=Path, required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    if args.publish:
        return publish(args.base, args.api_key_file)
    from src.training.cc3m_text import verify_checkpoint_progress
    record = json.loads(args.stop_record.read_text())
    staging = args.local / 'retired-final-uploads'
    staging.mkdir(exist_ok=True)
    files, hashes = {}, {}
    for name in ['last.pt', 'best-fid.pt', 'best-clip.pt']:
        source = args.local / 'checkpoints' / name
        state = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
        options = state['config']
        verify_checkpoint_progress(state, options['train_items'] // options['total_batch_size'],
                                   options['accumulation'], 8)
        assert options['wandb_id'] == args.base.name
        inode = source.stat().st_ino
        if inode not in hashes:
            with source.open('rb') as stream:
                hashes[inode] = base64.b64encode(hashlib.file_digest(stream, 'md5').digest()).decode()
        target = staging / name
        if not target.exists():
            target.hardlink_to(source)
        files[name] = dict(step=state['global_step'], bytes=source.stat().st_size, md5=hashes[inode])
        del state
    children = {}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            parent = int(next(line.split()[1] for line in (proc / 'status').read_text().splitlines()
                              if line.startswith('PPid:')))
            children.setdefault(parent, []).append(int(proc.name))
        except FileNotFoundError:
            pass
    descendants = []
    def walk(pid):
        for child in children.get(pid, []):
            walk(child)
        descendants.append(pid)
    walk(record['torchrun'])
    for pid in descendants + [record['supervisor']]:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    receipt = dict(timestamp=time.time(), files=files, staging=str(staging),
                   run_path=f"{options['wandb_entity']}/{options['wandb_project']}/{options['wandb_id']}",
                   full_states_verified=True)
    (args.base / 'retired-local-preservation.json').write_text(json.dumps(receipt, indent=2))
    with (args.base / 'retired-upload.log').open('a') as log:
        worker = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
            '--base', str(args.base), '--api-key-file', str(args.api_key_file), '--publish'],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps(dict(**receipt, cpu_upload_pid=worker.pid)), flush=True)


if __name__ == '__main__':
    main()
