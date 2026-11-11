"""Publish the original dictionary's measured deterministic 128-bin codec."""
import json
import os
from pathlib import Path

ROOT = Path('/tmp/laser-compound128-20261107')
BASE = ROOT / 'frozen'


def main():
    report = json.loads((ROOT / 'tokenizer-verification.json').read_text())
    assert report['passed'] and report['measured_validation_rfid'] < 4.73
    os.environ.update(WANDB_API_KEY=Path('/root/.config/laser/wandb.key').read_text().strip(),
        WANDB_BASE_URL='https://api.forge.coreweave.com', WANDB_CONSOLE='off', WANDB_DISABLE_GIT='true',
        WANDB_CACHE_DIR=str(ROOT / 'wandb-cache'))
    import wandb
    run = wandb.init(entity='helloimlixin-rutgers', project='laser',
        id='imagenet-laser-compound128-tokenizer-quality-20261107',
        name='Original LASER compound128 | full50k reconstruction', job_type='tokenizer-validation',
        dir=str(ROOT / 'quality'), resume='allow', config=dict(atoms=16384, coefficient_bins=128,
            coefficient_bits=7, coefficient_noise=False, training_and_evaluation_codec_identical=True))
    run.log({**{'rfid/' + k: v for k, v in report['results'].items()},
        'rfid/quantization_delta':report['delta_vs_continuous']})
    artifact = wandb.Artifact('imagenet-original-laser-compound128-tokenizer-20261107',
        type='tokenizer', metadata=report)
    artifact.add_file(str(BASE / 'inputs/recipe.yaml'), name='recipe.yaml')
    artifact.add_file(str(ROOT / 'tokenizer-verification.json'), name='tokenizer-verification.json')
    artifact.add_file(str(ROOT / 'cpu-verification.json'), name='cpu-verification.json')
    for p in Path(__file__).parent.glob('*.py'):
        artifact.add_file(str(p), name='runtime/' + p.name)
    artifact = run.log_artifact(artifact)
    artifact.wait(timeout=180)
    assert str(artifact.state) == 'COMMITTED'
    run.use_artifact('helloimlixin-rutgers/laser/imga16384k4altbn64-b128-b300-20260830000755-stage1-checkpoints:v9')
    receipt = dict(artifact=artifact.qualified_name, state=str(artifact.state), rfid=report['measured_validation_rfid'], url=run.url)
    (BASE / 'inputs/paired-tokenizer-artifact.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (ROOT / 'quality/cloud-verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
    run.finish()
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
