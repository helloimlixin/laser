"""Compare eager/compiled production block shapes during bounded GPU pauses."""
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import statistics
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from src.models.rqtransformer.attentions import AttentionBlock
from src.models.rqtransformer.configs import AttentionBlockConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-pid', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    command = Path(f'/proc/{args.worker_pid}/cmdline').read_bytes()
    assert b'imagenet-rfid421-ffhq-compound-1400m-20260922' in command
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(123)
    rows = []
    for name, batch, length in [('spatial', 128, 64), ('depth', 8192, 4), ('coefficient', 32768, 2)]:
        block = AttentionBlock(AttentionBlockConfig(embed_dim=1536, n_head=24)).cuda()
        x = torch.randn(batch, length, 1536, device='cuda', requires_grad=True)
        eager = block.forward
        compiled = torch.compile(eager, dynamic=False)

        def step(forward):
            block.zero_grad(set_to_none=True)
            x.grad = None
            with torch.autocast('cuda', dtype=torch.bfloat16):
                y = forward(x)
                loss = y.float().square().mean()
            loss.backward()
            return y.detach()

        # Compile before stopping production; stop only for timed comparisons.
        step(compiled)
        torch.cuda.synchronize()
        timer = threading.Timer(40, lambda: os.kill(args.worker_pid, signal.SIGCONT))
        os.kill(args.worker_pid, signal.SIGSTOP)
        timer.start()
        try:
            time.sleep(.5)
            timings = {}
            for mode, forward in [('eager', eager), ('compiled', compiled)]:
                step(forward)
                torch.cuda.synchronize()
                times = []
                for _ in range(5):
                    start = time.perf_counter()
                    step(forward)
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - start)
                timings[mode] = statistics.median(times)
            row = dict(name=name, shape=list(x.shape), seconds=timings,
                       speedup=timings['eager'] / timings['compiled'])
            rows.append(row)
            print(json.dumps(row), flush=True)
        finally:
            os.kill(args.worker_pid, signal.SIGCONT)
            timer.cancel()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2) + '\n')
        del x, block, eager, compiled
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
