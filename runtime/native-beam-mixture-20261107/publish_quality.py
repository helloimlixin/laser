"""Publish the gated mixture, including all predeclared candidate measurements."""
import json
import os
from pathlib import Path

ROOT=Path('/tmp/laser-native-beam-mixture-20261107');BASE=ROOT/'frozen-mixture';QUALITY=ROOT/'quality'


def main():
    report=json.loads((QUALITY/'full-results.json').read_text());assert report['passed']
    selected=report['selected_native_weight'];row=report['results'][str(selected)]
    replicate_path=QUALITY/'replicate-results.json'
    replicate=json.loads(replicate_path.read_text()) if replicate_path.exists() else None
    if replicate:
        assert replicate['passed'] and replicate['results'][str(selected)]['rfid']<4.73
    os.environ.update(WANDB_API_KEY=Path('/root/.config/laser/wandb.key').read_text().strip(),
        WANDB_BASE_URL='https://api.forge.coreweave.com',WANDB_CONSOLE='off',WANDB_DISABLE_GIT='true',
        WANDB_CACHE_DIR=str(ROOT/'wandb-cache'))
    import wandb
    run=wandb.init(entity='helloimlixin-rutgers',project='laser',
        id='imnet-native-beam-mixture-quality-20261107',name='Native-soft beam mixture | entropy and full50k reconstruction',
        job_type='tokenizer-teacher-validation',dir=str(QUALITY),resume='allow',
        config=dict(native_temperature=.5,selected_native_weight=selected,
            native_component_full_vocabulary=True,prefix_conditionals='exact Bayes mixture',
            validation_used_for_predeclared_weight_selection=True,validation_used_for_gradient_training=False))
    table=wandb.Table(columns=['native_weight','rfid','teacher_entropy','native_fraction'])
    for key,r in report['results'].items():table.add_data(float(key),r['rfid'],r['teacher_entropy'],r['native_fraction_observed'])
    run.log({'quality/candidates':table,'quality/teacher_rfid':row['rfid'],
        'quality/teacher_entropy':row['teacher_entropy'],'quality/native_weight':selected})
    if replicate:
        run.log({'quality/teacher_rfid_independent_seed':replicate['results'][str(selected)]['rfid']})
    artifact=wandb.Artifact('imagenet-j16384-a4096-native-beam-mixture-teacher-20261107',type='tokenizer',
        metadata=dict(native_weight=selected,native_temperature=.5,teacher_rfid=row['rfid'],
            teacher_entropy=row['teacher_entropy'],vocabulary=16384,atoms=4096,
            full_vocabulary_native_component=True,exact_prefix_conditionals=True,
            dictionary_coefficients_and_CNN_unchanged=True))
    for name in ['joint-codebook.pt','reconstruction-codebook.pt','recipe.yaml']:
        artifact.add_file(str(BASE/'inputs'/name),name=name)
    for p in QUALITY.glob('*.json'):artifact.add_file(str(p),name='verification/'+p.name)
    artifact.add_file(str(ROOT/'mixture-math-verification.json'),name='verification/mixture-math.json')
    for p in Path(__file__).parent.glob('*.py'):artifact.add_file(str(p),name='runtime/'+p.name)
    artifact=run.log_artifact(artifact);artifact.wait(timeout=180);assert str(artifact.state)=='COMMITTED'
    run.use_artifact('helloimlixin-rutgers/laser/imagenet-atoms4096-joint16384-native-teacher-beam-reconstruction-tokenizer:v0')
    receipt=dict(artifact=artifact.qualified_name,state=str(artifact.state),url=run.url,
        native_weight=selected,teacher_rfid=row['rfid'],teacher_entropy=row['teacher_entropy'])
    if replicate:
        receipt['teacher_rfid_independent_seed']=replicate['results'][str(selected)]['rfid']
        receipt['independent_seed']=replicate['measurement_seed']
    for path in [QUALITY/'cloud-verification.json',BASE/'inputs/paired-tokenizer-artifact.json']:
        path.write_text(json.dumps(receipt,indent=2)+'\n')
    for name in ['full-results.json','mixture-verification.json','screen-results.json','replicate-results.json']:
        if not (QUALITY/name).exists():continue
        (BASE/'inputs'/name).write_bytes((QUALITY/name).read_bytes())
    run.finish();print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
