"""Publish the completed reconstruction comparison and selected compact book."""
from pathlib import Path
import json
import os

BASE=Path('/tmp/laser-recon-4096-20261105')
PUBLIC=Path('/workspace/Projects/laser/outputs/imagenet-native-joint-atom-reduction-20261105')
os.environ.update(WANDB_API_KEY=Path('/root/.config/laser/wandb.key').read_text().strip(),
    WANDB_BASE_URL='https://api.forge.coreweave.com',WANDB_CONSOLE='off',WANDB_DISABLE_GIT='true',
    WANDB_CACHE_DIR=str(BASE/'wandb-cache'))
import wandb

report=json.loads((PUBLIC/'full-results.json').read_text())
run=wandb.init(entity='helloimlixin-rutgers',project='laser',
    id='imnet-native4096-reconstruction-v8192-frozen-20261105',
    name='ImageNet | 4096 vs8192 atoms | frozen CNN | full50k recon quality',
    job_type='tokenizer-reconstruction-validation',dir=str(BASE),resume='allow',
    config=dict(control_atoms=8192,compact_atoms=4096,control_vocabulary=32769,compact_vocabulary=16385,
        coefficient_levels_per_atom=4,residual_depth=4,encoder_decoder_frozen=True,
        reconstruction_validation_images=50000,selection_split='heldout training2048',
        validation_used_for_fitting_or_selection=False,refinement_training_images=4096,
        selected_refinement_updates=50,transformer_training_started=False,
        source_checkpoint_sha256=report['source_checkpoint_sha256'],
        source_run='imnet-j32769-a8192-upstream-native-scratch-2b300-20261031'))
table=wandb.Table(columns=['candidate','atoms','vocabulary','reconstruction_FID','LPIPS','PSNR_dB'])
for name,row in report['results'].items():
    table.add_data(name,row['atoms'],row['vocabulary'],row['rfid'],row['lpips'],row['psnr_db'])
selected=report['results'][report['selected']]
control=report['results']['control8192']
run.log({'quality/comparison':table,'quality/control_rfid':control['rfid'],
    'quality/compact_rfid':selected['rfid'],'quality/rfid_delta':report['rfid_delta'],
    'quality/relative_rfid_change':report['relative_rfid_change'],
    'quality/lpips_delta':report['lpips_delta'],'quality/psnr_delta_db':report['psnr_delta_db']})
run.summary.update({'quality/rfid_preserved_within_two_percent':report['rfid_preserved_within_two_percent'],
    'quality/encoder_decoder_bitwise_unchanged':True,'quality/transformer_training_started':False})
artifact=wandb.Artifact('imagenet-native-joint4096-reconstruction-selected-20261105',type='tokenizer',
    metadata=dict(atoms=4096,vocabulary=16385,depth=4,levels_per_atom=4,
        rfid=selected['rfid'],lpips=selected['lpips'],psnr_db=selected['psnr_db'],
        control_rfid=control['rfid'],encoder_decoder_frozen=True,
        source_checkpoint_sha256=report['source_checkpoint_sha256']))
artifact.add_file(str(PUBLIC/'compact4096-reconstruction-selected.pt'),name='joint-codebook.pt')
for name in ['full-results.json','screen-results.json','refinement-verification.json',
             'validation-ready.json','representation-verification.json','preview-policy.json','reconstruction-comparison.png']:
    artifact.add_file(str(PUBLIC/name),name=name)
for path in Path(__file__).parent.glob('*.py'):artifact.add_file(str(path),name='runtime/'+path.name)
artifact=run.log_artifact(artifact);artifact.wait();assert str(artifact.state)=='COMMITTED'
receipt=dict(run_url=run.url,artifact=artifact.qualified_name,state=str(artifact.state),
    transformer_training_started=False)
(PUBLIC/'cloud-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
run.finish();print(json.dumps(receipt),flush=True)
