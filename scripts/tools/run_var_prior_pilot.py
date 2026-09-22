"""Supervise an isolated prior fine-tune and verify its online checkpoints."""
import argparse
import fcntl
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.tools.run_var341_pipeline import Pipeline
from scripts.tools.run_ffhq_var341_pipeline import FFHQPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--wait-for', type=Path)
    args = parser.parse_args()
    args.base.mkdir(parents=True, exist_ok=True)
    with (args.base / '.pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pipeline = Pipeline(args.base, args.base)
        try:
            if args.wait_for:
                pipeline.status('waiting_for_evaluation', receipt=str(args.wait_for))
                deadline = time.monotonic() + 1800
                while not args.wait_for.exists():
                    if pipeline.stopping or time.monotonic() > deadline:
                        raise RuntimeError('Stopped or timed out waiting for evaluation')
                    time.sleep(5)
            FFHQPipeline.phase(pipeline, 'prior_training',
                pipeline.torchrun(3, 'train.py', ['--config', args.config]),
                args.base / 'train/complete.json')
            online = FFHQPipeline.verify_online_checkpoints(pipeline, 'train',
                ['prior-last.pt', 'prior-best-fid.pt'])
            pipeline.status('complete', artifact=online['artifact'])
        except BaseException as error:
            pipeline.status('failed', error_type=type(error).__name__, error=str(error))
            raise


if __name__ == '__main__':
    main()
