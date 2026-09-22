"""Distributed check that cached real-image FID moments preserve the score."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch.distributed as dist
from src.training.cli import load_config
from src.training.var_laser import Experiment, get_inception_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--count', type=int, default=192)
    args = parser.parse_args()
    base = args.base
    out = base/'speed-tuning/reference-cache'
    local = Path('/tmp/laser-var-checkpoints')/base.name
    torch.set_num_threads(4)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl', timeout=timedelta(minutes=5))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = load_config(base/'runtime/configs/experiments/ffhq256-var341-tokenizer.yaml', [
        'smoke_steps=1', 'wandb.mode=disabled', f'output_dir={out}',
        f'execution.checkpoint_dir={local}/speed-tuning/reference-cache',
        f'execution.media_dir=/tmp/laser-var-media/{base.name}/speed-tuning/reference-cache',
        '++execution.cache_reconstruction_reference=true'])
    experiment = Experiment(cfg)
    checkpoint = torch.load(local/'tokenizer/tokenizer-last.pt', map_location='cpu', weights_only=False, mmap=True)
    experiment.vae.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    experiment.inception = get_inception_model().eval().requires_grad_(False).to(experiment.device)
    calls = [0]
    def record_call(*_):
        calls[0] += 1
    hook = experiment.inception.register_forward_pre_hook(record_call)
    values = []
    for epoch in range(2):
        calls[0] = 0
        started = time.monotonic()
        quality = experiment.reconstruction(epoch, args.count)
        values.append(dict(quality=quality, inception_calls_per_rank=calls[0], seconds=time.monotonic()-started))
    hook.remove()
    assert values[0]['inception_calls_per_rank'] == 2 * values[1]['inception_calls_per_rank']
    difference = abs(values[0]['quality']['matched_rfid'] - values[1]['quality']['matched_rfid'])
    assert difference < 1e-6, difference
    if dist.get_rank() == 0:
        record = dict(passed=True, images=args.count, world_size=dist.get_world_size(),
                      absolute_fid_difference=difference, evaluations=values)
        (out/'complete.json').write_text(json.dumps(record, indent=2))
        print(json.dumps(record), flush=True)
    if experiment.run:
        experiment.run.finish()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
