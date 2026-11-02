#!/usr/bin/env python3
"""Audit the literal full pair-ID native prior without allocating its weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def parameter_budget(model, microbatch, world_size, execution=None):
    execution = execution or {}
    parameters = sum(p.numel() for p in model.parameters())
    head = sum(p.numel() for p in model.classifier.linear.parameters())
    vocabulary = max(model.vocab_size)
    events = int(torch_product(model.block_size))
    state_bytes = parameters * 16  # FP32 master, gradients, two FP32 Adam moments.
    logits_bytes = microbatch * events * vocabulary * 2  # FP16 AMP output only.
    chunked = bool(execution.get('chunked_training_loss',False))
    materialized_logits = (min(microbatch*events,int(execution['loss_row_chunk_size']))
        * min(vocabulary,int(execution['loss_class_chunk_size'])) * 2) if chunked else logits_bytes
    return dict(parameter_count=parameters, classifier_parameters=head,
        backbone_parameters=parameters-head, vocabulary=vocabulary,
        compound_events_per_image=events, microbatch=microbatch, world_size=world_size,
        replicated_parameter_gradient_adam_bytes_per_gpu=state_bytes,
        fp16_full_logits_bytes_per_gpu=logits_bytes,
        full_logits_materialized_during_training=not chunked,
        fp16_materialized_training_logits_bytes_per_gpu=materialized_logits,
        replicated_minimum_bytes_per_gpu=state_bytes+materialized_logits,
        ideal_fully_sharded_persistent_bytes_per_gpu=state_bytes/world_size,
        note='Lower bounds exclude activations, logit gradients, AMP casts, frozen tokenizer, '
             'optimizer temporaries, communication buffers and sampling top-p sorting. '
             'Full-state sharding alone does not remove the full classifier all-gather or logits.')


def torch_product(shape):
    import math
    return math.prod(shape)


def main():
    import torch
    from omegaconf import OmegaConf
    from src import rq_upstream
    from rqvae.utils.config import augment_arch_defaults
    from src.training.laser_packed_integer_recipe import PackedIntegerLaserRecipe

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, default=ROOT/'configs/stage2/imagenet-laser-packed-pair512-native-20261017.yaml')
    parser.add_argument('--training-config', type=Path,
        default=Path('/tmp/laser-compound512-origlr-imagenet-20261016-r3/inputs/stage2-training-config.yaml'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--world-size', type=int, default=4)
    args = parser.parse_args()
    if args.world_size < 1:
        parser.error('--world-size must be positive')
    recipe = OmegaConf.load(args.recipe)
    training = OmegaConf.load(args.training_config)
    arch = augment_arch_defaults(training.arch)
    with tempfile.TemporaryDirectory(prefix='laser-packed-id-audit-') as directory:
        base = Path(directory)
        (base/'inputs').mkdir()
        OmegaConf.save(recipe, base/'inputs/recipe.yaml')
        profile = PackedIntegerLaserRecipe(base)
        with torch.device('meta'):
            model = profile.model(arch)
        budget = parameter_budget(model, int(recipe.microbatch), args.world_size,recipe.get('execution',{}))
        available = [torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())]
        result = dict(budget, integer_mapping='atom_id * 512 + coefficient_bin',
            architecture_type=f'{type(model).__module__}.{type(model).__name__}',
            native_transformer=type(model) is rq_upstream.RQTransformer,
            classifier_type=f'{type(model.classifier.linear).__module__}.{type(model.classifier.linear).__name__}',
            full_cartesian_vocabulary=True, separate_atom_or_coefficient_heads=False,
            allocated_model_weight_bytes=0, optimizer_created=False, training_launched=False,
            gpu_total_memory_bytes=available,
            replicated_fits_even_before_activations=bool(available) and min(available) >= budget['replicated_minimum_bytes_per_gpu'],
            fresh_stage2_required=True, shared_upstream=rq_upstream.provenance(),
            recipe_sha256=hashlib.sha256(args.recipe.read_bytes()).hexdigest(),
            recipe=OmegaConf.to_container(recipe, resolve=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('parameter_count','classifier_parameters','vocabulary',
        'replicated_minimum_bytes_per_gpu','replicated_fits_even_before_activations','training_launched')}))


if __name__ == '__main__':
    main()
