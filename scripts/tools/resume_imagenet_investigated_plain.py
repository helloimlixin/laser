"""Resume the evaluated plain endpoint using its verified local serialization."""
import importlib.util
import json
import os
from pathlib import Path
import shutil

import yaml

BASE = Path('/tmp/laser-imagenet-pair-memory-investigation-20261007')
OUT = Path('/workspace/Projects/laser/outputs/imagenet-pair-memory-investigation-20261007')


def main():
    proof = json.loads((OUT / 'local-production-resume-verification.json').read_text())
    assert proof['passed'] and proof['adam_age'] == 3798
    checkpoint = Path(proof['local_resume_path'])
    assert checkpoint.is_file() and str(checkpoint).startswith('/tmp/')
    config_path = BASE / 'production-train.yaml'
    config = yaml.safe_load(config_path.read_text())
    assert config['options']['wandb_id'] == 'imagenet-rfid421-classcond-8h100-20261007'
    config['options'].update(resume_checkpoint=str(checkpoint), max_optimizer_steps=0)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    shutil.copyfile(config_path, OUT / config_path.name)
    spec = importlib.util.spec_from_file_location('investigation_launcher', BASE / 'launch.py')
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    # Inherit the original supervisor's complete training environment.
    env = dict(os.environ)
    assert env['LASER_PAIR_MEMORY_BASE'] == str(BASE)
    assert env['LASER_PAIR_MEMORY_OUTPUT'] == str(OUT)
    assert env['LASER_WANDB_RESUME'] == 'must'
    assert env['CUDA_VISIBLE_DEVICES'] == '0,1,2,3,4,5,6,7'
    launcher.record(phase='resuming_production', resume_step=3798,
                    verified_local_checkpoint=str(checkpoint),
                    recovery_reason='Avoid workspace mmap checkpoint page faults')
    launcher.run(env, 'baseline', 'train', resume_step=3798, production=True)


if __name__ == '__main__':
    main()
