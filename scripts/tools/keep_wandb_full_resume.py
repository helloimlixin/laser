"""Attach verified full best-FID/IS uploads to a live immutable-checkpoint job."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.training.checkpoint_upload import CheckpointUploader
from src.training.full_resume_upload import FullResumeWinners, atomic_json, cached_payload
from verified_wandb_checkpoint_upload import VerifiedCloudUpload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--checkpoint-dir', required=True, type=Path)
    parser.add_argument('--cache-dir', required=True, type=Path)
    parser.add_argument('--local-dir', required=True, type=Path)
    parser.add_argument('--evidence-dir', required=True, type=Path)
    parser.add_argument('--key-file', type=Path)
    parser.add_argument('--poll-seconds', type=float, default=5)
    parser.add_argument('--supervise', action='store_true',
                        help='Restart the watcher after transient upload failures')
    parser.add_argument('--adopt-pid', type=int,
                        help='Supervise an already running watcher without interrupting its upload')
    args = parser.parse_args()
    if args.key_file:
        os.environ['WANDB_API_KEY'] = args.key_file.read_text().strip()
    os.environ['LASER_CHECKPOINT_IMMUTABLE_FILES'] = '1'
    os.environ['LASER_CHECKPOINT_UPLOAD_CACHE_DIR'] = str(args.cache_dir)
    args.local_dir.mkdir(parents=True, exist_ok=True)
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    if args.supervise:
        lock = (args.local_dir / 'supervisor.lock').open('w')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stopping = False
        child = None
        adopted = args.adopt_pid
        if adopted is not None:
            process = Path(f'/proc/{adopted}')
            if (process.stat().st_uid != os.geteuid() or
                    str(Path(__file__).resolve()).encode() not in
                    (process / 'cmdline').read_bytes().split(b'\0')):
                raise ValueError('Can only adopt this user\'s matching recovery watcher')

        def stop_supervisor(_signum, _frame):
            nonlocal stopping
            stopping = True
            if child is not None and child.poll() is None:
                child.terminate()
            elif adopted is not None and Path(f'/proc/{adopted}').exists():
                os.kill(adopted, signal.SIGTERM)

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, stop_supervisor)
        attempt = 0
        while not stopping:
            if adopted is not None:
                atomic_json(args.evidence_dir / 'supervisor-status.json', dict(
                    pid=os.getpid(), watcher_pid=adopted, attempt=attempt,
                    state='running', adopted=True, started_unix=time.time()))
                while not stopping and Path(f'/proc/{adopted}').exists():
                    time.sleep(5)
                if stopping:
                    break
                adopted = None
            command = [sys.executable, str(Path(__file__).resolve())] + [
                arg for arg in sys.argv[1:] if arg != '--supervise']
            child = subprocess.Popen(command)
            atomic_json(args.evidence_dir / 'supervisor-status.json', dict(
                pid=os.getpid(), watcher_pid=child.pid, attempt=attempt,
                state='running', started_unix=time.time()))
            while child.poll() is None:
                time.sleep(5)
            if stopping or child.returncode == 0:
                break
            print(json.dumps(dict(event='restarting-recovery-watcher',
                                  exit_code=child.returncode, attempt=attempt)), flush=True)
            attempt += 1
            for _ in range(6):
                if stopping:
                    break
                time.sleep(5)
        return
    lock = (args.local_dir / 'watcher.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    winners = FullResumeWinners(args.local_dir / 'winners',
                               archive=args.evidence_dir / 'checkpoints')
    uploader = CheckpointUploader(
        args.local_dir / 'snapshots',
        VerifiedCloudUpload(args.run, args.evidence_dir / 'cloud-full-resume-receipt.json'),
        immutable_sources=True)
    seen = set()
    if winners.state:
        uploader.submit(winners.upload_paths(), max(row['epoch'] for row in winners.state.values()))
    try:
        while not stopping:
            uploader.check()
            for name in ('best-resume.json', 'latest-checkpoint.json'):
                try:
                    source, cached = cached_payload(args.checkpoint_dir / name, args.cache_dir)
                    if str(source) in seen:
                        continue
                    changed = winners.consider(cached, provenance=source)
                    seen.add(str(source))
                    if changed:
                        # Cloud worker pins the inode immediately, independently
                        # of the next trainer save or the durable archive copy.
                        uploader.submit(winners.upload_paths(), max(
                            row['epoch'] for row in winners.state.values()))
                        winners.persist(args.evidence_dir / 'checkpoints')
                        print(json.dumps(dict(event='full-recovery-winners',
                                              updated=changed, source=str(source))), flush=True)
                except FileNotFoundError:
                    # A newer durable save can supersede a receipt/cache pair.
                    continue
            atomic_json(args.evidence_dir / 'watcher-status.json', dict(
                pid=os.getpid(), state='running', heartbeat_unix=time.time(),
                run=args.run, observed_checkpoints=len(seen),
                winners={k: {f: v[f] for f in ('file', 'score', 'global_step', 'epoch')}
                         for k, v in winners.state.items()}))
            time.sleep(args.poll_seconds)
    finally:
        uploader.close()
        atomic_json(args.evidence_dir / 'watcher-status.json', dict(
            pid=os.getpid(), state='stopped', heartbeat_unix=time.time()))


if __name__ == '__main__':
    main()
