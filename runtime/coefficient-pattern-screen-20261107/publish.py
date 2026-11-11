"""Publish fitted coefficient vocabularies and full validation evidence."""
from common import BASE, ROOT, sha, write
import json
import os
from pathlib import Path
import zipfile


def main():
    report = json.loads((ROOT/'results.json').read_text())
    fit = json.loads((ROOT/'fit-results.json').read_text())
    cpu = json.loads((ROOT/'cpu-verification.json').read_text())
    assert report['passed'] and cpu['passed'] and report['frozen_weights_unchanged']
    assert not report['validation_used_for_fitting_or_seed_selection']
    os.environ.update(WANDB_API_KEY=Path('/root/.config/laser/wandb.key').read_text().strip(),
        WANDB_BASE_URL='https://api.forge.coreweave.com', WANDB_CONSOLE='off', WANDB_DISABLE_GIT='true',
        WANDB_CACHE_DIR=str(ROOT/'wandb-cache'))
    import wandb
    run = wandb.init(entity='helloimlixin-rutgers', project='laser',
        id='imagenet-coefficient-pattern-rfid-screen-20261107',
        name='Original LASER | one coefficient-pattern token | full50k screen',
        job_type='tokenizer-reconstruction-screen', dir=str(ROOT), resume='allow',
        config=dict(atoms=16384, coefficient_pattern_sizes=[1024,4096], decisions_per_site=5,
            fitting_images=16000, heldout_training_images=2000, validation_images=50000,
            coefficient_space='physical final least-squares values', assignment_metric='support Gram latent error',
            training_only_codebook_fit_and_seed_selection=True, stage2_training_launched=False))
    table = wandb.Table(columns=['codec','rfid','delta_vs_compound128','predictions_per_site','nominal_bits_per_site'])
    logs = {}
    for name, row in report['results'].items():
        table.add_data(name, row['rfid'], row.get('delta_rfid_vs_compound128'),
            row.get('decisions_per_site', 8 if name=='compound128' else None), row.get('nominal_bits_per_site',84 if name=='compound128' else None))
        logs['reconstruction/rfid_'+name] = row['rfid']
    run.log(dict(logs, **{'reconstruction/comparison':table}))
    artifact = wandb.Artifact('imagenet-original-laser-coefficient-patterns-screen-20261107',
        type='tokenizer', metadata=report)
    for size, row in fit['selected'].items():
        path = Path(row['path'])
        assert sha(path)==row['sha256']
        artifact.add_file(str(path), name='patterns-'+size+'.pt')
    for name in ['results.json','fit-results.json','experiment-plan.json','training-indices.json',
                 'cpu-verification.json','extraction-verification.json']:
        artifact.add_file(str(ROOT/name), name='verification/'+name)
    for rank in range(4):
        for prefix in ['fit-history','fit-result','training-samples']:
            artifact.add_file(str(ROOT/f'{prefix}-rank{rank}.json'), name=f'verification/{prefix}-rank{rank}.json')
    archive = ROOT/'source.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in (ROOT/'frozen-code').glob('*.py'):
            z.write(p, 'runtime/'+p.name)
        for p in [BASE/'inputs/recipe.yaml', BASE/'inputs/stage2-training-config.yaml']:
            z.write(p, 'baseline-inputs/'+p.name)
    artifact.add_file(str(archive), name='source.zip')
    artifact = run.log_artifact(artifact)
    artifact.wait(timeout=180)
    assert str(artifact.state)=='COMMITTED'
    for source in [
        'imga16384k4altbn64-b128-b300-20260830000755-stage1-checkpoints:v9',
        'imagenet-original-laser-compound128-tokenizer-20261107:v0',
        'imagenet-laser-compound128-native-scratch100-4a100-20261107-source:v0']:
        run.use_artifact('helloimlixin-rutgers/laser/'+source)
    receipt = dict(state='COMMITTED', artifact=artifact.qualified_name, url=run.url,
        rfid={name:row['rfid'] for name,row in report['results'].items()}, generation_quality_measured=False)
    write(ROOT/'cloud-verification.json', receipt)
    run.summary['screen/complete']=True
    run.finish()
    print(json.dumps(receipt), flush=True)


if __name__=='__main__':
    main()
