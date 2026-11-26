#!/usr/bin/env python3
"""Prepare the approved 320-token shared-pair stage-2 run."""
import json
from pathlib import Path
import shutil
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.training.rq_imagenet_runtime import sha256, write_json

RUN = 'imagenet-sharedpair24577-depth5-scratch100-4gpu-20261112'
BASE = Path('/tmp/laser-sharedpair24577-d5-4gpu-20261112')
OUT = ROOT / 'outputs' / RUN
QUALITY = ROOT / 'outputs/imagenet-shared-pair24577-improve-20261112'


def main():
    for name in ['inputs', 'data', 'native/inputs', 'wandb', 'runtime', 'staging', 'checkpoint-cache']:
        (BASE / name).mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    verified = json.loads((QUALITY / 'best-320.json').read_text())
    assert verified['depth'] == 5 and verified['tokens_per_image'] == 320
    assert sha256(QUALITY / 'best-320.pt') == verified['codebook_sha256']
    shutil.copyfile(QUALITY / 'best-320.pt', BASE / 'inputs/joint-codebook.pt')
    for source, target in [
        (Path('/tmp/laser-shared-pair-rfid/stage1.pt'), BASE / 'inputs/stage1-model.pt'),
        (Path('/tmp/laser-shared-pair-rfid/data/ILSVRC2012_img_val.tar'), BASE / 'data/ILSVRC2012_img_val.tar'),
        (ROOT / 'third_party/rq-vae-transformer', BASE / 'upstream'),
        (Path('/tmp/laser-shared-pair-rfid/torch-cache'), BASE / 'torch-cache'),
    ]:
        if not target.exists(): target.symlink_to(source)
    shutil.copyfile('/tmp/laser-shared-pair-rfid/data/val-index.npz', BASE / 'data/val-index.npz')
    # Permit disposable training-split smoke runs while the full local archive stages.
    if not (BASE / 'data/train-index.npz').exists():
        shutil.copyfile(ROOT / 'outputs/imagenet-beam451-aligned-joint16k-scratch100-8h100-20261108/data-verification/train-index.npz', BASE / 'data/train-index.npz')
    original = ROOT / 'outputs/imagenet-original-rq1400m-rfid473-scratch20-4h100-20261014/reproducibility'
    for name in ['stage1-model.pt', 'stage1-config.yaml']:
        shutil.copyfile(original / name, BASE / 'native/inputs' / name)
    reference = ROOT / 'outputs/imagenet-rq24576-native-scratch100-8h100-20261108/reproducibility'
    shutil.copyfile(reference / 'imagenet_256_train.npz', BASE / 'inputs/imagenet_256_train.npz')
    recipe = dict(version='shared-physical-pair24577-depth5-v1', run_id=RUN,
        name='ImageNet | shared pair24577 | depth5 320 tokens | fresh100 | 4 RTX6000',
        seed=261001, epochs=100, lr_schedule_epochs=100, global_batch=2048, microbatch=64,
        backbone='upstream_rqtransformer', decoder='native_joint_integer', target_mode='joint_residual_codes',
        tokenizer=dict(sha256='dd28db9306d526bdc9fbf8016403e106c4f317fd8f5c04792f0953e53310fdab',
            codebook_sha256=verified['codebook_sha256'], vocab_size=24577, atoms=8192, depth=5,
            tokens_per_image=320, measured_rfid=verified['rfid'], learned_compact_dictionary=True,
            encoder_decoder_frozen=True, shared_codebook_across_depths=True,
            coefficient_values_shared=True, depth_multipliers=False, separate_coefficient_tokens=False),
        targets=dict(policy='soft_stochastic', temperature=[.05]*5, chunk_size=512,
            stochastic_prefixes=True, encoding='native_greedy_shared_physical_pair_codes'),
        objective='native_soft_target_cross_entropy', auxiliary_loss_terms=0,
        sampling=dict(temperature=1., top_k=2048, top_p=.95),
        evaluation=dict(generated_images=50000, real_images=1281167, reference_split='train',
            every_epochs=2, include_epoch_one=True, batch_size=64,
            reference_sha256=sha256(BASE / 'inputs/imagenet_256_train.npz')),
        initialization=dict(reference_vocab_size=24577, paired_fresh_backbone=False,
            reset_stage2_seed_after_tokenizer=True, stage2_checkpoint_loaded=False),
        execution=dict(training_autocast_dtype='bfloat16', fused_adamw=True),
        logging=dict(wandb=True, create_workspace=False),
        training_rng=dict(policy='independent_rank_v1', base_seed=261001),
        preview=dict(every_updates=500, first_update=10, fixed_seed=261001),
        checkpoint=dict(every_updates=500, at_first_update=True, preserve_best_fid=True),
        lineage=dict(tokenizer_quality_report=str(QUALITY / 'full-rfid.json'),
            tokenizer_artifact=json.loads((QUALITY / 'wandb.json').read_text())['artifact'],
            stage2_initialization='fresh random weights and optimizer', previous_buggy_prior_loaded=False))
    (BASE / 'inputs/recipe.yaml').write_text(yaml.safe_dump(recipe, sort_keys=False))
    for name in ['stage2-training-config.yaml', 'stage2-archive-config.yaml']:
        config = yaml.safe_load((reference / name).read_text())
        config['dataset']['vocab_size'] = config['arch']['vocab_size'] = 24577
        config['arch']['block_size'] = [8, 8, 5]
        if name == 'stage2-training-config.yaml':
            config['experiment'].update(batch_size=64, total_batch_size=2048, epochs=100,
                save_ckpt_freq=1, test_freq=2, sample=dict(temp=1., top_k=2048, top_p=.95))
        else: config['sampling'] = dict(temp=1., top_k=2048, top_p=.95)
        (BASE / 'inputs' / name).write_text(yaml.safe_dump(config, sort_keys=False))
    execution = dict(world_size=4, microbatch=64, accumulation_steps=8, global_batch=2048,
        encoder_batch_size=8, generation_batch_size=64, training_autocast_dtype='bfloat16',
        training_sdpa=True, short_depth_sequence_math=True, fused_adamw=True)
    write_json(BASE / 'inputs/execution.json', execution)
    write_json(OUT / 'prepared.json', dict(base=str(BASE), recipe=recipe, execution=execution,
        teacher_calibration_required=True, stage2_training_started=False))
    print('Prepared', BASE, flush=True)


if __name__ == '__main__': main()
