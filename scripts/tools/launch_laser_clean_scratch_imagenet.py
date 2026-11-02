#!/usr/bin/env python3
"""Six-epoch stage-2 test with the scalar orthogonal/soft recipe from step zero."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path('/workspace/Projects/laser')
sys.path[:0] = [str(ROOT), str(ROOT/'scripts/tools')]
import launch_laser_hard_ar_imagenet as launcher

RUN = 'imagenet-laser-rq-scalar-orthogonal-soft-scratch6-4h100-20261015'
launcher.BASE = Path('/tmp/laser-rq-clean-scratch-imagenet-20261015')
launcher.SOURCE_BASE = Path('/tmp/laser-rq-validated-ar-imagenet-20261015')
launcher.RUN = RUN
launcher.OUT = ROOT/'outputs'/RUN
launcher.RECIPE_PATH = ROOT/'configs/stage2/imagenet-laser-rq-scalar-orthogonal-soft-scratch6-20261015.yaml'


def validate_recipe():
    from omegaconf import OmegaConf
    recipe = OmegaConf.load(launcher.RECIPE_PATH)
    assert recipe.run_id == RUN and recipe.epochs == 6
    assert recipe.decoder == 'scalar' and recipe.coefficient_policy == 'soft'
    assert recipe.ar_factors.coordinates == 'orthogonal'
    assert recipe.ar_factors.coefficient_bins == 2048
    assert recipe.global_batch == 2048 and recipe.microbatch == 32
    assert recipe.lr_schedule_epochs == 100 and recipe.auxiliary_loss_terms == 0
    assert recipe.target_mode == 'hard_sparse_tokens' and 'teacher' not in recipe
    assert recipe.evaluation.generated_images == 50000
    assert recipe.evaluation.reference_split == 'train' and recipe.evaluation.real_images == 1281167
    return recipe


def record_comparison():
    reference = ROOT/'outputs/imagenet-original-rq1400m-rfid473-scratch20-4h100-20261014/evaluations'
    curves = {}
    for epoch in [1,2,4,6]:
        name = f'epoch{epoch:03d}.json'
        own = json.loads((launcher.OUT/'evaluations'/name).read_text())
        rq = json.loads((reference/name).read_text())
        assert own['generated_images'] == rq['generated_images'] == 50000
        assert own['real_images'] == rq['real_images'] == 1281167
        assert own['real_split'] == rq['real_split'] == 'train'
        curves[str(epoch)] = dict(laser_fid=own['eval/fid'], rq_fid=rq['eval/fid'],
                                  laser_minus_rq=own['eval/fid']-rq['eval/fid'])
    result = dict(passed=True, fresh_stage2=True, fresh_optimizer=True,
        recipe_fixed_from_step_zero=True, completed_epochs=6, global_step=3756,
        generated_images_per_evaluation=50000, reference_images=1281167,
        curves=curves, continued_beyond_six_epochs=False)
    launcher.write(launcher.OUT/'comparison.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run'])
    args = parser.parse_args()
    recipe = validate_recipe()
    if args.action == 'prepare':
        launcher.prepare()
        prepared = json.loads((launcher.OUT/'prepared.json').read_text())
        prepared.update(initialization='random_stage2_from_step_zero',
            source_stage2_checkpoint=None, initial_optimizer_state_count=0,
            recipe_fixed_from_step_zero=True, context_residual=False,
            evaluation_epochs=[1,2,4,6], decision_epoch=6,
            rq_reference_fid_by_epoch={'1':94.22586333,'2':65.87001562,
                '4':48.64910363,'6':40.93746077547098},
            stop_updates=6*626, cosine_total_updates=100*626)
        launcher.write(launcher.OUT/'prepared.json', prepared)
    else:
        assert recipe.epochs == 6
        os.environ['WANDB_API_KEY'] = Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
        result = launcher.run(resume=False)
        if result == 0:
            record_comparison()
        raise SystemExit(result)


if __name__ == '__main__':
    main()
