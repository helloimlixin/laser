"""Publish the actual aligned-teacher quality gate and its unchanged source book."""
import json
import os
from pathlib import Path

BASE=Path('/tmp/laser-aligned-beam451-20261105/frozen')
QUALITY=BASE.parent/'quality'
PUBLIC=Path('/workspace/Projects/laser/outputs/imnet-j16384-a4096-beam16-aligned-soft-scratch2-4a100-20261105')


def main():
    report=json.loads((QUALITY/'full-results.json').read_text());assert report['passed']
    proof=json.loads((QUALITY/'alignment-verification.json').read_text());assert proof['passed']
    os.environ.update(WANDB_API_KEY=Path('/root/.config/laser/wandb.key').read_text().strip(),
        WANDB_BASE_URL='https://api.forge.coreweave.com',WANDB_CONSOLE='off',WANDB_DISABLE_GIT='true',
        WANDB_CACHE_DIR=str(BASE.parent/'quality-wandb-cache'))
    import wandb
    run=wandb.init(entity='helloimlixin-rutgers',project='laser',
        id='imnet-j16384-a4096-beam16-aligned-teacher-quality-20261105',
        name='rFID4.512 book | aligned beam conditional soft teacher | full50k gate',
        job_type='tokenizer-teacher-validation',dir=str(QUALITY),resume='allow',
        config=dict(atoms=4096,vocabulary=16384,depth=4,beam_width=16,
            teacher='beam sequence posterior with exact sampled-prefix conditionals',
            calibration=report['calibration'],validation_images=50000,
            physical_dictionary_coefficients_and_CNN_unchanged=True,
            reconstruction_metric_is_generation_metric=False))
    run.log({'canonical/reconstruction_rfid':report['results']['canonical']['rfid'],
        'teacher/reconstruction_rfid':report['results']['teacher']['rfid'],
        'teacher/rfid_delta':report['teacher_minus_canonical_rfid'],
        'teacher/sequence_entropy':report['actual_mean_sequence_entropy'],
        'teacher/expected_entropy_per_depth':report['actual_mean_sequence_entropy']/4})
    run.summary.update({'verification/aligned_teacher_passed':True,
        'verification/actual_teacher_reconstruction_below473':report['results']['teacher']['rfid']<4.73,
        'generation/quality_measured':False})
    artifact=wandb.Artifact('imagenet-atoms4096-joint16384-beam451-aligned-soft-teacher-20261105',
        type='tokenizer',metadata=dict(atoms=4096,vocabulary=16384,depth=4,beam_width=16,
            canonical_rfid=report['results']['canonical']['rfid'],
            teacher_sampled_rfid=report['results']['teacher']['rfid'],
            teacher_mode='exact conditional beam-sequence posterior',
            sequence_temperature=report['calibration']['sequence_temperature'],
            physical_values_and_CNN_unchanged=True,validation_images=50000))
    for name in ['joint-codebook.pt','reconstruction-codebook.pt','recipe.yaml','measured-beam-stage1-rfid.json']:
        artifact.add_file(str(BASE/'inputs'/name),name=name)
    for path in QUALITY.glob('*.json'):artifact.add_file(str(path),name='verification/'+path.name)
    for path in Path(__file__).parent.glob('*.py'):artifact.add_file(str(path),name='runtime/'+path.name)
    artifact=run.log_artifact(artifact);artifact.wait(timeout=120);assert str(artifact.state)=='COMMITTED'
    run.use_artifact('helloimlixin-rutgers/laser/imagenet-atoms4096-joint16384-native-teacher-beam-reconstruction-tokenizer:v0')
    run.use_artifact('helloimlixin-rutgers/laser/imga16384k4altbn64-b128-b300-20260830000755-stage1-checkpoints:v9')
    receipt=dict(artifact=artifact.qualified_name,state=str(artifact.state),url=run.url,
        canonical_rfid=report['results']['canonical']['rfid'],teacher_sampled_rfid=report['results']['teacher']['rfid'])
    (QUALITY/'cloud-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (PUBLIC/'teacher-cloud-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (BASE/'inputs/paired-tokenizer-artifact.json').write_text(json.dumps(receipt,indent=2)+'\n')
    run.finish();print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
