"""Publish the fully measured compact tokenizer and explicit paired encoding policy."""
import json
import os
from pathlib import Path

ROOT=Path('/workspace/Projects/laser')
BASE=Path('/tmp/laser-recon-4096-beam-20261105')
PUBLIC=ROOT/'outputs/imagenet-native4096-below473-20261105'
report=json.loads((PUBLIC/'full-results.json').read_text())
proof=json.loads((PUBLIC/'paired-teacher-verification.json').read_text());assert proof['passed']
selected=report['results'][report['selected']];assert selected['rfid']<4.73
policy=json.loads((PUBLIC/'tokenizer-policy.json').read_text())
os.environ.update(WANDB_API_KEY=Path('/root/.config/laser/wandb.key').read_text().strip(),
    WANDB_BASE_URL='https://api.forge.coreweave.com',WANDB_CONSOLE='off',WANDB_DISABLE_GIT='true',
    WANDB_CACHE_DIR=str(BASE/'wandb-cache'))
import wandb
run=wandb.init(entity='helloimlixin-rutgers',project='laser',
    id='imnet-native4096-recon-below473-softteacher-20261105',
    name='ImageNet4096 atoms | native soft teacher | reconstruction-only beam | full50k',
    job_type='tokenizer-reconstruction-validation',dir=str(BASE),resume='allow',
    config=dict(atoms=4096,vocabulary=16385,depth=4,levels_per_atom=4,
        ar_teacher=policy['ar_teacher'],reconstruction=policy['reconstruction'],
        encoder_decoder_weights_unchanged=True,physical_codewords_fixed=True,
        reconstruction_search_changes_training_targets=False,
        validation_images=50000,selection_split='heldout training2048',
        transformer_training_started=False,reference_original_rq_rfid=4.73))
table=wandb.Table(columns=['candidate','reconstruction_beam','rfid','LPIPS','PSNR_dB'])
for name,row in report['results'].items():
    table.add_data(name,row['reconstruction_beam_width'],row['rfid'],row['lpips'],row['psnr_db'])
run.log({'reconstruction/comparison':table,'reconstruction/rfid':selected['rfid'],
    'reconstruction/lpips':selected['lpips'],'reconstruction/psnr_db':selected['psnr_db'],
    'reconstruction/rfid_margin_below473':4.73-selected['rfid']})
run.summary.update({'quality/beats_original_rq473':True,'quality/validation_images':50000,
    'verification/native_soft_targets_bitwise_unchanged':True,
    'verification/encoder_decoder_bitwise_unchanged':True,'training/ar_started':False})
artifact=wandb.Artifact('imagenet-native4096-paired-softteacher-below473-20261105',type='tokenizer',
    metadata=dict(atoms=4096,vocabulary=16385,depth=4,rfid=selected['rfid'],
        reconstruction_beam_width=selected['reconstruction_beam_width'],
        teacher_encoding_beam_width=1,teacher_mode='native_stochastic_soft',
        source_checkpoint_sha256=report['source_checkpoint_sha256'],
        encoder_decoder_unchanged=True,validation_images=50000))
for name in ['joint-codebook.pt','tokenizer-policy.json','paired_tokenizer.py','beam_search.py',
             'full-results.json','screen-results.json','paired-teacher-verification.json','candidates.json']:
    artifact.add_file(str(PUBLIC/name),name=name)
for path in Path(__file__).parent.glob('*.py'):artifact.add_file(str(path),name='runtime/'+path.name)
artifact.add_file('/tmp/laser-j32769-a8192-20261105/source-artifact/source.zip',name='original-native-runtime-source.zip')
artifact=run.log_artifact(artifact);artifact.wait();assert str(artifact.state)=='COMMITTED'
receipt=dict(run_url=run.url,artifact=artifact.qualified_name,state=str(artifact.state),
    rfid=selected['rfid'],beats_original_rq=True,ar_training_started=False)
(PUBLIC/'cloud-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
run.finish();print(json.dumps(receipt),flush=True)
