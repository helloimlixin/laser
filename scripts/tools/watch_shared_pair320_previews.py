#!/usr/bin/env python3
"""Log the established class grids from immutable training checkpoints.

Runs independently of the trainer, without touching its weights, optimizer,
RNG streams, or final W&B run state. Media belongs to a linked checkpoint-sampling run.
"""
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist
import wandb

ROOT = Path('/workspace/Projects/laser')
BASE = Path('/tmp/laser-sharedpair24577-d5-4gpu-20261112')
RUN = 'imagenet-sharedpair24577-depth5-scratch100-4gpu-20261112'
OUT = ROOT / 'outputs' / RUN
os.environ['RQ_UPSTREAM_ROOT'] = str(BASE / 'upstream')
sys.path[:0] = [str(BASE / 'source'), str(BASE), str(ROOT)]

from src import rq_upstream
from src.training.laser_joint_recipe import JointLaserRecipe
from src.training.rq_sdpa_backend import sdpa_attention
from src.training.rq_sample_panels import CLASSES, PREVIEW_IMAGES, designated_preview
from src.training.rq_imagenet_runtime import write_json
from rqvae.utils.config import load_config, augment_arch_defaults


def alive(pid):
    try: return Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError: return False


class PreviewWriter:
    """A linked sampling run owns its own image history and run summaries."""
    def __init__(self, config):
        self.run = wandb.init(entity='helloimlixin-rutgers', project='laser',
            id=RUN+'-class-grids', group=RUN, job_type='checkpoint-sampling',
            name='ImageNet shared pair24577 depth5 | established class grids',
            resume='allow', config=dict(config, training_run_id=RUN), dir=str(BASE / 'wandb'))
        self.summary = self.run.summary
        write_json(OUT / 'class-grid-wandb.json', dict(url=self.run.url, id=self.run.id,
            training_run_id=RUN, training_process_uninterrupted=True))

    def publish(self, metadata):
        step = int(metadata['global_step'])
        values = {'samples/global_step': step}
        for key, path in metadata['media'].items():
            values[key] = wandb.Image(path, caption=f'Checkpoint {step}; '+
                ('10 samples per fixed class' if key=='samples/fixed' else '64 fixed-seed samples'))
        self.log(values)

    def log(self, values):
        self.run.log(values)
        self.summary['preview/checkpoint_step'] = int(values['samples/global_step'])

    def finish(self):
        self.run.finish()


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    device = torch.device('cuda:0')  # Worker sees only the selected spare capacity.
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rendezvous = BASE / f'preview-gloo-{os.getpid()}'
    dist.init_process_group('gloo', init_method='file://' + str(rendezvous), rank=0, world_size=1)
    profile = JointLaserRecipe(BASE)
    profile.recipe.preview.designated_classes = True
    profile.execution_options['sampling_cast_cache'] = True
    config = dict(classes=[dict(name=name, imagenet_id=label) for name, label in CLASSES],
        samples_per_class=64, overview_per_class=10, grid_rows=8, grid_columns=8,
        fixed_seed=int(profile.recipe.preview.fixed_seed), source='immutable full training checkpoints',
        independent_from_training_process=True, training_rng_unchanged=True,
        overview_media_key='samples/fixed',
        vocabulary=24577, tokens_per_image=320)
    write_json(OUT / 'sampling-grid-config.json', config)
    run = PreviewWriter(config)
    run.summary.update({'preview/class_grid_configuration': config})
    receipt = OUT / 'class-grid-worker.json'
    completed = -1
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        completed = previous.get('checkpoint_step', -1)
        metadata = OUT / 'samples' / f'step{completed:06d}-panels.json'
        if metadata.exists():
            run.publish(json.loads(metadata.read_text()))
            previous.update(wandb_synced=True, wandb_writer_role='linked sampling run; trainer history and lifecycle unchanged', pid=os.getpid())
            write_json(receipt, previous)
            print(f'Published saved class grids for checkpoint {completed}', flush=True)
    aux, _ = profile.tokenizer(device)
    arch = augment_arch_defaults(load_config(BASE / 'inputs/stage2-training-config.yaml').arch)
    arch.vocab_size = 24577
    with torch.device('meta'): model = profile.model(arch)
    sys.modules[rq_upstream.RQTransformer.__module__].autocast = (
        lambda *args, enabled=True, **kwargs: torch.autocast('cuda', enabled=enabled, dtype=torch.float16))
    supervisor_pid = int((OUT / 'supervisor.pid').read_text())
    try:
        while alive(supervisor_pid):
            records = []
            for path in (OUT / 'persistent-train/checkpoints/.checkpoint-data').glob('*.json'):
                try: records.append(json.loads(path.read_text()))
                except (OSError, json.JSONDecodeError): continue
            latest = max(records, key=lambda row: row['global_step']) if records else None
            if latest and latest['global_step'] > completed:
                path = Path(latest['local'])
                if not path.exists(): path = Path(latest['persistent'])
                payload = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
                assert payload['run_id'] == RUN
                assert payload['joint_codebook_sha256'] == profile.codebook_sha256
                model.load_state_dict(payload['state_dict'], strict=True, assign=True)
                model.to(device).eval().requires_grad_(False)
                step = int(payload['global_step'])
                del payload
                with sdpa_attention(model, batch_limit=512, short_sequence_math=True):
                    designated_preview(profile, model, aux, device, 0, 1, step, run, OUT)
                model.to('meta')
                torch.cuda.empty_cache()
                completed = step
                write_json(receipt, dict(passed=True, checkpoint_step=step, generated_images=PREVIEW_IMAGES,
                    checkpoint_sha256=latest['sha256'], classes=config['classes'],
                    training_process_uninterrupted=True, training_rng_unchanged=True,
                    wandb_synced=True,
                    wandb_writer_role='linked sampling run; trainer history and lifecycle unchanged', pid=os.getpid()))
                print(f'Logged 64 samples for each established class at checkpoint step {step}', flush=True)
            time.sleep(15)
    finally:
        run.finish()
        dist.destroy_process_group()


if __name__ == '__main__': main()
