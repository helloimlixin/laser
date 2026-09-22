"""Evaluate an explicitly selected compound VAR checkpoint on all available ranks."""
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
from omegaconf import OmegaConf
from src.training.cli import load_config
from src.training.compound_var import CompoundExperiment
from src.training.distributed_failure import fatal_worker_error
from src.original_rq_training import atomic_json, file_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--samples', type=int, default=50000)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    experiment = None
    try:
        cfg = load_config(args.config)
        cfg.compound.mode = 'evaluate-best'
        if cfg.execution.get('media_dir'):
            cfg.execution.media_dir = str(Path(cfg.execution.media_dir).parent/'evaluate-best')
        experiment = CompoundExperiment(cfg)
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
        if checkpoint['tokenizer_sha256'] != experiment.tokenizer_sha256:
            raise ValueError('Evaluation checkpoint and tokenizer do not match')
        model = experiment.build_prior().eval().requires_grad_(False)
        model.load_state_dict(checkpoint['model'], strict=True)
        epoch = int(checkpoint['progress']['epoch'])
        del checkpoint
        if experiment.rank == 0:
            import wandb
            experiment.run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project,
                id=cfg.wandb.id + '-eval', name=cfg.wandb.id + '-eval', resume='allow',
                group=cfg.wandb.group, mode=cfg.wandb.mode, dir=cfg.execution.wandb_dir,
                config=dict(checkpoint=args.checkpoint, checkpoint_sha256=file_sha256(args.checkpoint),
                            tokenizer_sha256=experiment.tokenizer_sha256, selected_epoch=epoch,
                            patch_nums=list(cfg.model.patch_nums), generated_samples=args.samples,
                            real_reference_images=(60000 if cfg.evaluation.get('fid_reference_manifest') else len(experiment.val)),
                            world_size=experiment.world))
        experiment.validate(model, epoch, count=len(experiment.val))
        results = []
        for count in ([] if cfg.evaluation.get('fid_reference_manifest') else sorted({2000, args.samples})):
            started = time.monotonic()
            experiment.generate(model, epoch, count)
            if experiment.rank == 0:
                path = experiment.out / f'generation-epoch{epoch:03d}-{count}.json'
                record = json.loads(path.read_text())
                reference_count = min(int(cfg.evaluation.get('fid_reference_images', count)), len(experiment.val))
                record.update(real_reference_images=reference_count,
                              elapsed_seconds=time.monotonic()-started,
                              selected_checkpoint=args.checkpoint,
                              comparison=f'Diagnostic against {reference_count:,} held-out {cfg.data.dataset} images; not an official ADM FID50k benchmark')
                atomic_json(path, record)
                results.append(record)
        rq_results = []
        reference = cfg.evaluation.get('rq_fid_reference')
        if reference:
            from src.training.rq_reference_evaluation import evaluate_rq_reference
            candidates = [('best', args.checkpoint)]
            baseline = cfg.evaluation.get('rq_baseline_checkpoint')
            if baseline:
                candidates.append(('baseline', baseline))
            for label, candidate in candidates:
                if label == 'baseline':
                    saved = torch.load(candidate, map_location='cpu', weights_only=False, mmap=True)
                    if saved['tokenizer_sha256'] != experiment.tokenizer_sha256:
                        raise ValueError('RQ baseline and evaluation tokenizer do not match')
                    model.load_state_dict(saved['model'], strict=True)
                    epoch = int(saved['progress']['epoch'])
                    del saved
                record = evaluate_rq_reference(experiment, model, epoch, args.samples, reference,
                    cfg.evaluation.rq_fid_reference_sha256, label)
                if experiment.rank == 0:
                    record.update(checkpoint=str(candidate), checkpoint_sha256=file_sha256(candidate))
                    atomic_json(experiment.out / f'rq-reference-{label}-epoch{epoch:03d}-{args.samples}.json', record)
                    prefix = 'matched_train_fid50k' if cfg.evaluation.get('fid_reference_manifest') else 'rq_reference'
                    experiment.run.summary[f'{prefix}/{label}'] = record
                    rq_results.append(record)
        if experiment.rank == 0:
            artifact = wandb.Artifact(cfg.wandb.id + '-final-evaluation', type='evaluation')
            for path in experiment.out.glob('*.json'):
                artifact.add_file(str(path), name=path.name)
            artifact.add_file(str(experiment.out / 'resolved-config.yaml'), name='resolved-config.yaml')
            if (experiment.base / 'source-manifest.json').exists():
                artifact.add_file(str(experiment.base / 'source-manifest.json'), name='source-manifest.json')
            for path in experiment.media_path('unused').parent.glob('*.png'):
                artifact.add_file(str(path), name=path.name)
            if reference:
                artifact.add_file(str(reference), name='reference/ffhq_256_train.npz', policy='immutable')
                manifest = cfg.evaluation.get('fid_reference_manifest')
                if manifest:
                    artifact.add_file(str(manifest), name='reference/manifest.json', policy='immutable')
                for path in (Path(__file__), Path(__file__).resolve().parents[2] / 'src/training/rq_reference_evaluation.py'):
                    artifact.add_file(str(path), name='source/' + path.name)
            experiment.run.log_artifact(artifact).wait()
            atomic_json(experiment.out / 'complete.json', dict(checkpoint=args.checkpoint, results=results,
                                                             rq_reference_results=rq_results))
            experiment.run.finish()
        dist.barrier()
    except BaseException as error:
        fatal_worker_error(error, experiment.out if experiment else Path(args.checkpoint).parent, dist.get_rank())
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
