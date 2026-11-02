#!/usr/bin/env python3
"""Make a small, fixed-seed preview without touching the running trainer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path('/tmp/laser-fid14-continue100-20261016')
OUT = Path('/workspace/Projects/laser/outputs/imagenet-fid14-continue100-4h100-20261016')
RUN = 'helloimlixin-rutgers/laser/imagenet-fid14-continue100-4h100-20261016'


def record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def generate(checkpoint, gpu):
    sys.path[:0] = [str(BASE/'frozen/runtime'), str(BASE/'frozen/support'), str(BASE/'support')]
    os.environ['TORCH_HOME'] = str(BASE/'torch-cache')
    import torch
    from omegaconf import OmegaConf
    from src.training import rqtransformer as training
    from src.data.imagenet_labels import class_names_for_dataset
    from native_physical_pair_adapter import LaserPairRQTransformer
    from rqvae.models.rqtransformer.configs import RQTransformerConfig
    from smart_sampling import PairSamplingPolicy, sample_physical_pairs

    torch.set_num_threads(4)
    device = torch.device('cuda', gpu)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(
        13*2**30/torch.cuda.get_device_properties(device).total_memory, device)
    saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    step, epoch = int(saved['global_step']), int(saved['epoch']) + 1
    assert step >= 49454 and len(saved['state_dict']) == 782
    options = saved['config']
    with torch.device('meta'):
        old = training.build_model(18432, 16384, physical_pair_context=True,
            sparsity_level=4, model_preset=options['model_preset'])
        model = LaserPairRQTransformer(RQTransformerConfig.create(OmegaConf.create(
            OmegaConf.to_container(old.config, resolve=True))), 16384)
    del old
    model.load_state_dict(saved['state_dict'], strict=True, assign=True)
    model.requires_grad_(False).to(device).eval()
    aux = training.LaserAux(BASE/'inputs/resume-stage1-tokenizer.pt', 16384, 2048, 3., 6.4,
        coeff_scales=options['coeff_scales'], soft_target_physical=False,
        clamp_coeffs=False, sparsity_level=4).requires_grad_(False).to(device).eval()
    del saved
    classes = torch.tensor([269, 612, 265, 628, 689, 301, 932, 798], device=device)
    labels = classes.repeat_interleave(4)
    policy = PairSamplingPolicy(coefficient_temperatures=(.4, .55, .75, 1.))
    images = []
    started = time.monotonic()
    torch.manual_seed(261001)
    torch.cuda.manual_seed(261001)
    status = OUT/'samples'/'preview-status.json'
    record(status, dict(state='generating', global_step=step, completed_epoch=epoch,
        images=32, generated=0, gpu=gpu, training_restarted=False, time=time.time()))
    with torch.inference_mode():
        for first in range(0, len(labels), 8):
            tokens = sample_physical_pairs(model, 8, aux, labels[first:first+8], policy=policy, amp=True)
            for chunk in tokens.split(4):
                images.append(aux.decode_tokens(chunk).float().mul(.5).add(.5).clamp(0, 1).cpu())
            value = dict(state='generating', global_step=step, completed_epoch=epoch,
                images=32, generated=first+8, elapsed_seconds=time.monotonic()-started,
                gpu=gpu, training_restarted=False, time=time.time())
            record(status, value)
            print(json.dumps(value), flush=True)
    image_path = OUT/'samples'/f'epoch{epoch:03d}-step{step}-joint32.png'
    training.save_class_labeled_grid(torch.cat(images), classes.cpu(),
        class_names_for_dataset('imagenet'), image_path, samples_per_class=4)
    result = dict(state='generated', global_step=step, completed_epoch=epoch, images=32,
        seed=261001, sampler='joint32-depth-strong', checkpoint=str(checkpoint),
        path=str(image_path), sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
        max_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
        elapsed_seconds=time.monotonic()-started, training_restarted=False, time=time.time())
    record(image_path.with_suffix('.json'), result)
    record(status, result)
    print(json.dumps(result), flush=True)
    del model, aux, tokens, chunk, classes, labels
    torch.cuda.empty_cache()
    return image_path


def upload(path):
    import wandb
    from PIL import Image
    os.environ['WANDB_API_KEY'] = Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
    run = wandb.Api(timeout=60).run(RUN)
    prior_receipt = path.with_suffix('.upload.json')
    if prior_receipt.exists() and json.loads(prior_receipt.read_text()).get('uploaded'):
        uploaded = run.file(str(path.resolve().relative_to(OUT)))
        assert uploaded.size == path.stat().st_size
    else:
        uploaded = run.upload_file(str(path), root=str(OUT))
        run.upload_file(str(path.with_suffix('.json')), root=str(OUT))
    metadata = json.loads(path.with_suffix('.json').read_text())
    with Image.open(path) as image:
        width, height = image.size
    media = {'_type': 'image-file', 'path': uploaded.name, 'format': 'png',
        'sha256': metadata['sha256'], 'size': path.stat().st_size,
        'width': width, 'height': height,
        'caption': f"Epoch {metadata['completed_epoch']}, step {metadata['global_step']}; "
            '32 fixed-seed samples, joint32-depth-strong; no rejection.'}
    values = {'samples/continuation': media,
        'samples/completed_epoch': metadata['completed_epoch'],
        'samples/global_step': metadata['global_step']}
    live = run.state == 'running'
    if live:
        # Attach to the trainer's existing local service so its next summary
        # write retains this media. Do not open a second wandb.init writer.
        worker = json.loads((OUT/'verification/native-production/process-rank0.json').read_text())['pid']
        sockets = [p for p in Path('/tmp').glob(f'wandb-{worker}-*/socket') if p.is_socket()]
        assert len(sockets) == 1, 'Expected the live trainer W&B service socket'
        from wandb.sdk.lib.service.service_token import UnixServiceToken
        UnixServiceToken(parent_pid=worker, path=str(sockets[0])).save_to_env()
        os.environ['WANDB_DIR'] = str(BASE/'wandb')
        wandb.setup(settings=wandb.Settings(console='off', silent=True))
        attached = wandb.attach(attach_id=run.id)
        assert attached.id == run.id
        attached.define_metric('samples/global_step')
        attached.define_metric('samples/continuation', step_metric='samples/global_step')
        # The backend allocates its internal step; checkpoint step is separate.
        # The image already exists in run files, so no second copy is needed.
        attached.log(values, commit=True)
        attached._backend.interface.deliver_get_summary().wait_or(timeout=30)
        attached._wl.remove_active_run(attached)
    else:
        # Epoch 100 may finish and release its service before the grid is ready.
        # Once the writer has closed, the public summary is durable.
        assert run.state == 'finished' and metadata['completed_epoch'] == 100
        run.summary.update(values)
    for _ in range(12):
        fresh = wandb.Api(timeout=60).run(RUN)
        if fresh.summary.get('samples/continuation', {}).get('sha256') == metadata['sha256']:
            break
        time.sleep(5)
    else:
        raise RuntimeError('Sample media was not synchronized by the live logger')
    assert fresh.summary['samples/continuation']['sha256'] == metadata['sha256']
    result = dict(run=run.url, filename=uploaded.name, uploaded=True, size=uploaded.size,
        summary_media_verified=True, media_history_logged=live, final_run_closed=not live,
        training_restarted=False, time=time.time())
    record(path.with_suffix('.upload.json'), result)
    print(json.dumps(result), flush=True)


def watch():
    import fcntl
    lock = (BASE/'preview-watch.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    record(OUT/'samples'/'watch-status.json', dict(state='watching', pid=os.getpid(),
        frequency='every completed epoch after its FID evaluation', target_epoch=100,
        training_restarted=False, time=time.time()))
    while True:
        candidates = []
        for receipt in (OUT/'native-train/checkpoints/.checkpoint-data').glob('*.json'):
            data = json.loads(receipt.read_text())
            step = int(data.get('global_step', 0))
            if step < 49454 or step % 626 or not Path(data['local']).exists():
                continue
            if not (OUT/'evaluations'/f'step{step}.json').exists():
                continue
            candidates.append(data)
        if candidates:
            latest = max(candidates, key=lambda item: item['global_step'])
            epoch, step = int(latest['epoch'])+1, int(latest['global_step'])
            path = OUT/'samples'/f'epoch{epoch:03d}-step{step}-joint32.png'
            uploaded = path.with_suffix('.upload.json')
            receipt = json.loads(uploaded.read_text()) if uploaded.exists() else {}
            complete = receipt.get('media_history_logged') or (
                receipt.get('final_run_closed') and receipt.get('summary_media_verified'))
            if not complete:
                usage = subprocess.check_output(['nvidia-smi',
                    '--query-gpu=memory.total,memory.used', '--format=csv,noheader,nounits'], text=True)
                free = [(float(row.split(',')[0])-float(row.split(',')[1]))/1024
                    for row in usage.strip().splitlines()]
                progress = json.loads((OUT/'verification/native-production/progress.json').read_text())
                cursor = int(progress['global_step'])
                # Leave at least a minute before the next distributed FID batch
                # can need its larger caches. The inference allocator is capped.
                until_fid = 626-cursor % 626
                if path.exists() or (max(free) >= 14.5 and (until_fid >= 30 or step == 62600)):
                    command = [sys.executable, str(Path(__file__).resolve())]
                    if path.exists():
                        command += ['--upload', str(path)]
                    else:
                        gpu = max(range(len(free)), key=lambda index: (free[index], index))
                        command += ['--checkpoint', latest['local'], '--gpu', str(gpu)]
                    result = subprocess.run(command, check=False)
                    record(OUT/'samples'/'watch-status.json', dict(state='watching', pid=os.getpid(),
                        last_epoch=epoch, last_step=step, last_exit_code=result.returncode,
                        target_epoch=100, training_restarted=False, time=time.time()))
                    complete = result.returncode == 0
            if complete and step == 62600:
                record(OUT/'samples'/'watch-status.json', dict(state='completed',
                    last_epoch=100, last_step=62600, training_restarted=False, time=time.time()))
                return
        time.sleep(45)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--upload', type=Path)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    if args.watch:
        watch()
    elif args.upload:
        upload(args.upload)
    else:
        assert args.checkpoint is not None
        path = generate(args.checkpoint, args.gpu)
        upload(path)


if __name__ == '__main__':
    main()
