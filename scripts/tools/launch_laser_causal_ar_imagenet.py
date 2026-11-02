#!/usr/bin/env python3
"""Launch the causal-coordinate continuation after its controlled FID gate."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import launch_laser_hard_ar_imagenet as launcher

launcher.BASE=Path('/tmp/laser-rq-causal-ar-imagenet-20261015')
launcher.SOURCE_BASE=Path('/tmp/laser-rq-hard-ar-imagenet-20261015')
launcher.RUN='imagenet-laser-rq-causal-ar-rfid421-continue20-20261015'
launcher.OUT=launcher.ROOT/'outputs'/launcher.RUN
launcher.RECIPE_PATH=launcher.ROOT/'configs/stage2/imagenet-laser-rq-causal-ar-20ep-20261015.yaml'
launcher.NEW_WANDB_RUN=True

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'))
    arguments=parser.parse_args()
    if arguments.action=='prepare':
        launcher.prepare()
        prepared=json.loads((launcher.OUT/'prepared.json').read_text())
        prepared.update(fresh_stage2=False,fresh_optimizer=False,
            initialization='continuation_from_matched_coordinate_trial',parent_global_step=2568,
            candidate_only=True,adoption_requires_matched_fid50k_and_reconstruction_fidelity=True)
        launcher.write(launcher.OUT/'prepared.json',prepared)
    else:
        adoption=json.loads((launcher.OUT/'candidate-adoption.json').read_text())
        assert adoption['passed'] and adoption['fid_generated_images']==50000
        assert adoption['reconstruction_fidelity_passed']
        sys.exit(launcher.run(resume=True))
