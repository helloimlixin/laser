"""Periodic checkpoint previews and an optional bounded runtime breakdown."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch
from omegaconf import OmegaConf
from torchvision.utils import save_image

from src.continuous_sparse_codec import continuous_image_codes
from src.training.continuous_compound_recipe import ContinuousCompoundRecipe
from src.training.continuous_compound import write_json
from src.training import k4_checkpoint_io


def profile_batch(model, aux, profile, base, output, device):
    """Time real FP32 targets and AMP forward/backward with a short worker pause."""
    from src.models.continuous_compound_rq import continuous_pair_loss
    from src.training.rq_imagenet_runtime import TarImageNet
    from rqvae.img_datasets.transforms import create_transforms
    from rqvae.utils.config import load_config
    config = load_config(base / 'inputs/stage2-training-config.yaml')
    dataset = TarImageNet(base / 'data', 'train', create_transforms(config.dataset, split='train'))
    examples = [dataset[int(i)] for i in torch.linspace(0, len(dataset)-1, 64).long()]
    images = torch.stack([x for x, _ in examples]).to(device)
    labels = torch.tensor([y for _, y in examples], device=device)
    model.train()
    # Warm up the exact kernels before pausing the production workers.
    codes = continuous_image_codes(aux, images, encoder_batch_size=8)
    loss, _ = continuous_pair_loss(model(codes, aux, labels, amp=True), codes, model.coefficient_scales)
    loss.backward(); model.zero_grad(set_to_none=True); torch.cuda.synchronize(device)
    pids = [json.loads((output/'train'/f'progress-rank{r}.json').read_text())['pid'] for r in range(4)]
    watchdog_code = ('import os,signal,time;time.sleep(45);'
                     f'[(os.kill(p,signal.SIGCONT)) for p in {pids!r}]')
    watchdog = subprocess.Popen([sys.executable, '-c', watchdog_code], start_new_session=True)
    paused = []
    measurements = []
    try:
        for pid in pids:
            os.kill(pid, signal.SIGSTOP); paused.append(pid)
        time.sleep(.5)
        for _ in range(3):
            model.zero_grad(set_to_none=True)
            begin, targets_done, backward_done = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            started = time.monotonic(); begin.record()
            codes = continuous_image_codes(aux, images, encoder_batch_size=8)
            targets_done.record()
            loss, _ = continuous_pair_loss(model(codes, aux, labels, amp=True), codes, model.coefficient_scales)
            loss.backward(); backward_done.record(); backward_done.synchronize()
            measurements.append(dict(targets_seconds=begin.elapsed_time(targets_done)/1000,
                prior_forward_backward_seconds=targets_done.elapsed_time(backward_done)/1000,
                wall_seconds=time.monotonic()-started))
    finally:
        for pid in paused:
            try: os.kill(pid, signal.SIGCONT)
            except ProcessLookupError: pass
        watchdog.terminate()
        model.zero_grad(set_to_none=True); model.eval()
    result = dict(microbatch=64, repeats=measurements,
        targets_seconds=sum(r['targets_seconds'] for r in measurements)/len(measurements),
        prior_forward_backward_seconds=sum(r['prior_forward_backward_seconds'] for r in measurements)/len(measurements),
        production_workers_resumed=True, excludes_ddp_and_optimizer=True)
    write_json(output / 'runtime-breakdown.json', result)
    print(json.dumps(dict(runtime_breakdown=result)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--every-updates', type=int, default=100)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    base, output, device = args.base.resolve(), args.output.resolve(), torch.device(args.device)
    torch.cuda.set_device(device)
    profile = ContinuousCompoundRecipe(base)
    model = aux = None
    last_step = -1
    while True:
        checkpoint = output / 'checkpoints/last.pt'
        if not checkpoint.is_file():
            time.sleep(5); continue
        source = k4_checkpoint_io._checkpoint_upload_source(checkpoint.resolve())
        payload = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
        step = int(payload['global_step'])
        evaluation_steps = {int(json.loads(p.read_text())['train/global_step'])
                            for p in (output/'evaluations').glob('epoch*.json')}
        if last_step >= 0 and step < last_step + args.every_updates and not (step in evaluation_steps and step > last_step):
            del payload
            if (output/'completed.json').is_file(): return
            time.sleep(10); continue
        if payload[profile.checkpoint_key] != profile.recipe.tokenizer.sha256:
            raise ValueError('Preview source tokenizer differs from the training job')
        if model is None:
            with torch.device('meta'):
                model = profile.model(OmegaConf.create(payload['config']['arch']))
            model.load_state_dict(payload['state_dict'], strict=True, assign=True)
            model.to(device).eval()
            aux, _ = profile.tokenizer(device)
        else:
            model.load_state_dict(payload['state_dict'], strict=True)
        del payload
        labels = torch.tensor([269,612,265,628,689,301,932,798], device=device).repeat_interleave(4)
        generator = torch.Generator(device=device).manual_seed(261001)
        started = time.monotonic()
        with torch.inference_mode():
            codes = model.sample_sparse(len(labels), aux, labels, amp=True, generator=generator,
                                        **profile.sampling_settings())
            images = torch.cat([profile.decode(aux, c) for c in codes.split(4)])
        path = output/'samples'/f'step{step:06d}-fixed.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        save_image(images.mul(.5).add(.5), path, nrow=4)
        write_json(output/'preview-status.json', dict(checkpoint_step=step, images=32, path=str(path),
            sampling_seconds=time.monotonic()-started, independent_of_training_rng=True, time=time.time()))
        print(json.dumps(dict(preview=str(path), checkpoint_step=step)), flush=True)
        last_step = step
        if args.profile:
            profile_batch(model, aux, profile, base, output, device)
            args.profile = False
        if args.once: return


if __name__ == '__main__':
    main()
