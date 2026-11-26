#!/usr/bin/env python3
"""Track an existing shared-pair fitting process and archive its verified book.

The API key is read by W&B from the environment, never from the recipe.
"""
import argparse
import json
import os
from pathlib import Path
import time

import wandb


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--entity', default='helloimlixin-rutgers')
    parser.add_argument('--project', default='laser')
    args = parser.parse_args()
    out = args.output
    recipe = json.loads((out / 'recipe.json').read_text())
    pid = int((out / 'fit.pid').read_text())
    run = wandb.init(entity=args.entity, project=args.project, id=out.name,
        name='ImageNet | shared physical pair24577 | dictionary fit',
        job_type='shared-pair-tokenizer-fitting', resume='allow', config=recipe,
        dir=os.environ.get('WANDB_DIR', '/tmp'))
    (out / 'wandb.json').write_text(json.dumps(dict(url=run.url, id=run.id), indent=2) + '\n')
    run.summary.update(dict(shared_codebook=True, unique_physical_banks=1,
        tokens_per_image=64 * recipe['depth'], separate_coefficient_tokens=False,
        stage2_training_started=False))
    previous, last_iteration = None, -1
    while True:
        status_path = out / 'status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if status != previous:
            run.summary['preparation/status'] = status
            if 'images' in status:
                run.log({f"cache/{status['split']}_images": status['images']})
            previous = status
        fit_path = out / 'fit.json'
        if fit_path.exists():
            report = json.loads(fit_path.read_text())
            for row in report['trace']:
                if row['iteration'] > last_iteration:
                    run.log({f'fit/{key}': value for key, value in row.items()})
                    last_iteration = row['iteration']
        if status.get('phase') == 'fitted':
            verification = json.loads((out / 'verification.json').read_text())
            assert verification['passed']
            artifact = wandb.Artifact(out.name + '-tokenizer', type='tokenizer', metadata=verification)
            for filename in ('joint-codebook.pt', 'fit.json', 'verification.json', 'recipe.json',
                             'fit-image-ids.npy', 'held-image-ids.npy'):
                artifact.add_file(str(out / filename), name=filename)
            for filename in ('src/models/laser_joint_tokenizer.py',
                             'src/training/laser_shared_joint_vocabulary.py',
                             'scripts/tools/fit_shared_pair_vocabulary.py',
                             'tests/test_shared_pair_integer_ids.py'):
                artifact.add_file(filename, name='source/' + filename)
            run.log_artifact(artifact, aliases=['latest']).wait()
            run.summary.update(verification)
            run.summary['preparation/phase'] = 'fitted_pending_quality_evaluation'
            break
        if not alive(pid):
            run.summary['preparation/phase'] = 'fitting_failed'
            run.finish(exit_code=1)
            raise RuntimeError('Fitter exited before verification; inspect fit.log')
        time.sleep(10)
    run.finish()


if __name__ == '__main__':
    main()
