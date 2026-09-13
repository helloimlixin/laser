#!/usr/bin/env python3
"""Run the released RQ-VAE stage-1 driver; add only observability/I/O adapters."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    root = args.upstream.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root))
    os.chdir(root)
    rank = int(os.environ.get('RANK', '0'))
    if args.smoke:
        os.environ['SMOKE_TEST'] = '1'

    import torch
    import rqvae.img_datasets as datasets
    import rqvae.optimizer as optimizers
    import rqvae.utils.setup as setup_module
    from rqvae.img_datasets.lsun import LSUNClass
    from rqvae.img_datasets.transforms import create_transforms
    from rqvae.utils.writer import Writer
    from rqvae.trainers.trainer_rqvae import Trainer

    # The released LSUN factory opens the same LMDB twice (train and "val").
    # Reuse one handle for the same population; modern lmdb forbids duplicate opens.
    def create_dataset(config, is_eval=False, logger=None):
        transform = create_transforms(config.dataset, split='train', is_eval=is_eval)
        dataset = LSUNClass(config.dataset.root, category_name='church', transform=transform)
        assert len(dataset) == 126227
        if args.smoke:
            dataset = torch.utils.data.Subset(dataset, torch.randperm(len(dataset))[:256])
        if logger:
            logger.info('Official LSUN factory: train and validation both use train LMDB; n=%d', len(dataset))
        return dataset, dataset
    datasets.create_dataset = create_dataset

    class ObservedWriter(Writer):
        def __init__(self, result_path):
            super().__init__(result_path)
            self.metrics = output / 'metrics.jsonl'
            atomic_json(output / 'run-directory.json', {'path': str(result_path)})

        def add_scalar(self, tag, scalar, mode, epoch=0):
            super().add_scalar(tag, scalar, mode, epoch)
            value = float(scalar)
            if not torch.isfinite(torch.tensor(value)):
                raise FloatingPointError(f'Nonfinite stage1 metric {tag}: {value}')
            with self.metrics.open('a') as stream:
                stream.write(json.dumps({'time': time.time(), 'tag': tag, 'value': value,
                                         'mode': mode, 'step_or_epoch': epoch}) + '\n')

        def add_image(self, tag, image, mode, epoch=0):
            super().add_image(tag, image, mode, epoch)
            from torchvision.utils import save_image
            dest = output / 'images' / f'{mode}-{tag.replace("/", "-")}-{epoch}.png'
            dest.parent.mkdir(parents=True, exist_ok=True)
            save_image(image, dest)
    setup_module.Writer = ObservedWriter

    original_optimizer = optimizers.create_optimizer
    started = time.time()
    progress = {'step': 0}
    def create_optimizer(model, config):
        optimizer = original_optimizer(model, config)
        assert not optimizer.state
        if rank == 0:
            atomic_json(output / 'initialization.json', {
                'stage': 1, 'initial_optimizer_entries': 0, 'resume': False,
                'pretrained_rqvae': str(args.checkpoint.resolve()),
                'published_imagenet_rfid': 4.73, 'training_epochs': 1,
                'global_batch': config.experiment.total_batch_size,
                'upstream_commit': '341395e562ac347f5eb62db9f5f08b9f2cc42a60',
            })
        def observed_step(optimizer, positional, keyword):
            progress['step'] += 1
            if rank == 0:
                step = progress['step']
                atomic_json(output / 'status.json', {
                    'phase': 'training', 'optimizer_step': step,
                    'steps_per_epoch': 2 if args.smoke else 987,
                    'epoch_fraction': step / (2 if args.smoke else 987),
                    'lr': optimizer.param_groups[0]['lr'],
                    'elapsed_seconds': time.time() - started,
                    'updated_unix': time.time(), 'pid': os.getpid(),
                    'peak_memory_gb': torch.cuda.max_memory_allocated() / 1e9,
                })
        optimizer.register_step_post_hook(observed_step)
        return optimizer
    optimizers.create_optimizer = create_optimizer

    original_train = Trainer.train
    def observed_train(self, *positional, **keyword):
        result = original_train(self, *positional, **keyword)
        if rank == 0:
            atomic_json(output / 'status.json', {'phase': 'evaluating_after_one_epoch',
                        'optimizer_step': progress['step'], 'updated_unix': time.time(), 'pid': os.getpid()})
        return result
    Trainer.train = observed_train

    sys.argv = [str(root / 'main_stage1.py'),
        '-m', str(root / 'configs/lsun-church/stage1/church256-rqvae-8x8x4.yaml'),
        '-r', str(output / 'upstream-results'), '-l', str(args.checkpoint.resolve()),
        'dataset.root=/tmp/laser-sign-data', 'experiment.batch_size=64']
    try:
        runpy.run_path(str(root / 'main_stage1.py'), run_name='__main__')
        if rank == 0:
            run_dir = Path(json.loads((output / 'run-directory.json').read_text())['path'])
            checkpoint = run_dir / 'epoch1_model.pt'
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            assert saved['epoch'] == 1 and saved['optimizer']['state']
            assert progress['step'] == (2 if args.smoke else 987)
            assert 'discriminator' in saved
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            atomic_json(output / 'complete.json', {'phase': 'complete', 'epoch': 1,
                'optimizer_steps': progress['step'], 'checkpoint': str(checkpoint),
                'checkpoint_sha256': digest, 'config': str(run_dir / 'config.yaml'),
                'finished_unix': time.time(), 'smoke_only': args.smoke})
            atomic_json(output / 'status.json', {'phase': 'complete', 'epoch': 1,
                'optimizer_step': progress['step'], 'updated_unix': time.time()})
    except BaseException as exc:
        atomic_json(output / f'failure-rank{rank}.json', {
            'type': type(exc).__name__, 'error': str(exc), 'time': time.time(),
            'optimizer_step': progress['step']})
        raise
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
